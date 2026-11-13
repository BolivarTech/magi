# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""Unit tests for S53-S54 (REQ-RV22: a secret in a file's content never leaves the process)."""

import copy
import pathlib
import unittest

from smoke import runs
from smoke.outcome import Outcome
from smoke.scenarios import review_masking as rm
from smoke.tests.rust_source import secret_openings
from smoke.tests.support import scenario_outcomes as _outcomes, scenario_result as _result

_KEY = runs.REVIEW_MASKED_KEY.decode("ascii")
_BODY = runs.REVIEW_KEYBLOCK_BODY_LINES[0]
#: [MAGI S6 MS3 loop2, R7b Melchior/Caspar] with the run-log field the stderr layer prints after the
#: message, as the measured refusals of S48/S49 (``_ROOT_STDERR``, ``_SKIPPED_STDERR``) carry it.
_REFUSAL = (b"error: nothing to review: no reviewable file in the input; left out: "
            b"%s (secret) review.error=nothing_to_review\n" % runs.REVIEW_KEYBLOCK_FILE.encode())


def _doc(masked=None, masked_unlisted=0, skipped=None, skipped_unlisted=0) -> dict:
    """The review JSON S53 reads, defaulting to what the product must publish."""
    return {"code_review": {
        # MAGI FIX (MS3 §6 loop13, R7b): a copy, so a test that edits its document cannot edit the
        # scenario's expected value.
        "masked": copy.deepcopy(rm.S53_EXPECTED_MASKED) if masked is None else masked,
        "masked_unlisted": masked_unlisted,
        "skipped": [] if skipped is None else skipped,
        "skipped_unlisted": skipped_unlisted}}


class S53Tests(unittest.TestCase):
    """S53 (R18): the keyed file is masked, declared, reviewed and never leaked."""

    def _judge(self, index, result, log=b"") -> Outcome:
        return _outcomes("S53", result, log)[rm.S53_ASSERTIONS[index]]

    def test_a_masked_and_reviewed_file_passes_every_assertion(self) -> None:
        self.assertEqual({Outcome.PASS}, set(_outcomes("S53", _result("R18", 0, _doc())).values()))

    def test_a_backslash_path_is_read_as_a_slash(self) -> None:
        doc = _doc(masked=[dict(rm.S53_EXPECTED_MASKED[0], path=runs.REVIEW_MASKED_FILE.replace("/", "\\"))])
        self.assertEqual(Outcome.PASS, self._judge(1, _result("R18", 0, doc)))

    def test_a_nonzero_exit_fails_the_first_assertion(self) -> None:
        self.assertEqual(Outcome.FAIL, self._judge(0, _result("R18", 1, _doc())))

    def test_a_wrong_pattern_fails_the_masked_assertion(self) -> None:
        bad = [dict(rm.S53_EXPECTED_MASKED[0], pattern="url-credential")]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=bad))))

    def test_an_extra_masked_entry_fails_exactly(self) -> None:
        two = rm.S53_EXPECTED_MASKED * 2
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=two))))

    def test_an_unlisted_masked_count_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked_unlisted=1))))

    def test_a_missing_masked_key_is_a_failure_not_an_empty_list(self) -> None:
        doc = _doc()
        del doc["code_review"]["masked"]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, doc)))

    def test_the_expected_entry_is_the_documented_one_not_the_scenarios_own_value(self) -> None:
        # [MAGI S6 MS3 loop2, R7b Caspar] the pass test defaults to rm.S53_EXPECTED_MASKED, so a
        # wrong expectation would pass itself; pinned here against the README's pattern name.
        self.assertEqual([{"path": runs.REVIEW_MASKED_FILE, "cause": "secret", "pattern": "anthropic-key"}],
                         rm.S53_EXPECTED_MASKED)

    def test_a_wrong_masked_path_fails_the_masked_assertion(self) -> None:
        # [MAGI S6 MS3 loop2, R7b Balthasar] the path is part of the exact entry.
        bad = [dict(rm.S53_EXPECTED_MASKED[0], path="elsewhere.rs")]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=bad))))

    def test_an_unlisted_skipped_count_fails_the_reviewed_assertion(self) -> None:
        # [MAGI S6 MS3 loop2, R7b Melchior/Balthasar] the README says skipped_unlisted must be 0.
        self.assertEqual(Outcome.FAIL, self._judge(2, _result("R18", 0, _doc(skipped_unlisted=1))))

    def test_a_timed_out_review_still_judges_the_leak(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7b): a hang is the backend's, so the body assertions are
        # CANNOT_TEST, but bytes already captured that hold the key are a leak all the same.
        from smoke.registry import DEFAULT_REGISTRY
        entry = DEFAULT_REGISTRY.get("S53")
        self.assertTrue(entry.inspects_timeouts)
        self.assertTrue(entry.needs_backend)
        clean = _result("R18", 0, None, timed_out=True)
        self.assertEqual([Outcome.CANNOT_TEST] * 4, [self._judge(i, clean) for i in range(4)])
        leaky = _result("R18", 0, None, stderr=b"oops " + runs.REVIEW_MASKED_KEY, timed_out=True)
        self.assertEqual([Outcome.CANNOT_TEST] * 3 + [Outcome.FAIL], [self._judge(i, leaky) for i in range(4)])

    def test_a_hung_runs_leak_cannot_test_keeps_its_own_detail(self) -> None:
        # MAGI FIX (MS3 §6 loop14, R7a2 Melchior): the leak judge's own CANNOT_TEST (here the relay's
        # record is missing) was replaced by "its capture holds no key", which claimed a check the
        # judge had refused to make.
        hung = _result("R18", 0, None, timed_out=True)
        findings = list(rm.a_key_in_a_files_content_is_masked_and_the_file_reviewed(hung))
        self.assertEqual(Outcome.CANNOT_TEST, findings[3].outcome)
        self.assertIn("relay's record", findings[3].detail)
        self.assertNotIn("holds no key", findings[3].detail)

    def test_the_masked_prompt_claim_is_s53s_own(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7a Balthasar): the generic leak judge no longer carries it.
        run = _result("R18", 0, _doc(), requests=(("/api/chat", b"{}"),))
        generic = rm._leak_finding(rm.S53_ASSERTIONS, 3, "R18", run, (runs.REVIEW_MASKED_KEY,), "k", relayed=True)
        self.assertEqual(Outcome.PASS, generic.outcome)
        self.assertEqual(Outcome.FAIL, self._judge(3, run))

    def test_a_skipped_keyed_file_fails_the_reviewed_assertion(self) -> None:
        doc = _doc(skipped=[{"path": runs.REVIEW_MASKED_FILE, "cause": "secret"}])
        self.assertEqual(Outcome.FAIL, self._judge(2, _result("R18", 0, doc)))

    def test_stdout_that_is_not_json_fails_both_json_assertions(self) -> None:
        result = _result("R18", 0, None)
        self.assertEqual(Outcome.FAIL, self._judge(1, result))
        self.assertEqual(Outcome.FAIL, self._judge(2, result))

    def test_valid_json_that_is_not_an_object_fails_both_json_assertions(self) -> None:
        # [MAGI S6 MS3 loop2, R7a Melchior/Caspar] `.get` on a list escaped the scenario.
        try:
            result = _result("R18", 0, ["not", "an", "object"])
            outcomes = (self._judge(1, result), self._judge(2, result))
        except AttributeError as exc:
            self.fail("a non-object document crashed S53: %s" % exc)
        self.assertEqual((Outcome.FAIL, Outcome.FAIL), outcomes)

    def test_the_raw_key_in_stdout_fails_the_leak_assertion(self) -> None:
        doc = _doc()
        doc["code_review"]["note"] = _KEY
        self.assertEqual(Outcome.FAIL, self._judge(3, _result("R18", 0, doc)))

    def test_the_raw_key_in_stderr_fails_the_leak_assertion(self) -> None:
        self.assertEqual(Outcome.FAIL, self._judge(3, _result("R18", 0, _doc(), stderr=_KEY.encode())))

    def test_the_raw_key_in_the_run_log_fails_the_leak_assertion(self) -> None:
        self.assertEqual(Outcome.FAIL, self._judge(3, _result("R18", 0, _doc()), log=_KEY.encode()))

    def test_a_partial_echo_of_the_key_fails_the_leak_assertion(self) -> None:
        # [MAGI S6 MS3 loop7, R7a Melchior] Only the whole key was looked for: a truncated echo
        # left bytes of the secret in an output and passed.
        part = _KEY.encode()[10:22]
        self.assertEqual(Outcome.FAIL, self._judge(3, _result("R18", 0, _doc(), stderr=b"x " + part)))

    def test_any_window_of_the_key_fails_the_leak_assertion(self) -> None:
        # [MAGI S6 MS3 loop8, R7a Melchior] One inner slice was probed: an echo of another part
        # of the key (its tail, say) passed.
        # MAGI FIX (MS3 §6 loop11): every fragment of the random body down to MIN_LEAK_FRAGMENT.
        body = rm.code_review.key_body(_KEY.encode())
        for i in range(len(body) - rm.MIN_LEAK_FRAGMENT + 1):
            part = body[i:i + rm.MIN_LEAK_FRAGMENT]
            with self.subTest(at=i):
                self.assertEqual(Outcome.FAIL, self._judge(3, _result("R18", 0, _doc(), stderr=b"x " + part)))

    def test_a_masked_entry_without_a_path_fails_without_crashing(self) -> None:
        # [MAGI S6 MS3 loop8, R7a Caspar] A missing `path` must be a FAIL finding, never a KeyError.
        bad = [{"cause": "secret", "pattern": runs.REVIEW_MASKED_PATTERN}]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=bad))))

    def test_a_masked_entry_with_another_cause_fails(self) -> None:
        # [MAGI S6 MS3 loop8, R7b Melchior] The README promises exact equality; each drift fails.
        bad = [dict(rm.S53_EXPECTED_MASKED[0], cause="binary")]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=bad))))

    def test_a_masked_entry_with_an_extra_field_fails(self) -> None:
        bad = [dict(rm.S53_EXPECTED_MASKED[0], line=3)]
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, _doc(masked=bad))))

    def test_a_null_masked_list_fails(self) -> None:
        doc = _doc()
        doc["code_review"]["masked"] = None
        self.assertEqual(Outcome.FAIL, self._judge(1, _result("R18", 0, doc)))

    def test_the_key_is_also_in_a_doc_comment(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7b): a key in a doc comment reaches the documentation index
        # and its embedder; the leak check reads those requests too (relayed), so S53 exercises
        # the doc-chunk path of RV22 as well as the batch's.
        (_, body), = runs.review_masked_files()
        doc_lines = [line for line in body.splitlines() if line.startswith(b"///")]
        self.assertTrue(any(runs.REVIEW_MASKED_KEY in line for line in doc_lines))
        # the plain doc comment S55 reads stays as it was
        self.assertIn(b"/// " + runs.REVIEW_MASKED_DOC, doc_lines)

    def test_the_fixture_file_holds_the_key_but_its_name_does_not(self) -> None:
        (path, body), = runs.review_masked_files()
        self.assertIn(runs.REVIEW_MASKED_KEY, body)
        self.assertNotIn(_KEY, path)
        # [MAGI S6 MS3 loop6, R7b Melchior] S53 needs the path to match no secret pattern at all
        # (a path match leaves the file out instead of masking it), not only to lack the full
        # key: no opening of a pattern may appear in it, in any case.
        # [MAGI S6 MS3 loop7, R7b] The openings are read from the product's table now, not
        # copied here: `test_rust_sync.ReviewMaskingSyncTests`
        # (`test_the_s53_fixture_path_holds_no_opening_of_any_product_pattern`). MAGI FIX (MS3 §6
        # loop13, R7b): the guard lives there once, not here too.


class SecretOpeningsTests(unittest.TestCase):
    """[MAGI S6 MS3 loop10, R7b Caspar] Identifiers in NON_TOKEN_PREFIXES resolve to their values."""

    def test_every_fixed_opening_is_read(self) -> None:
        got = secret_openings(pathlib.Path(__file__).resolve().parents[2])
        for fixed in ("-----begin ", "---- begin ssh2 ", "aws_secret", "secretaccesskey", "://"):
            self.assertIn(fixed, got)

    def test_an_identifier_is_resolved_or_refused(self) -> None:
        import tempfile
        table = ('const prefixes_a: () = (); // prefixes: &["sk-", "ghp_"]\n'
                 'const A: &[&str] = &["x"];\nconst URL_AUTHORITY_START: &str = "://";\n'
                 'const NEW_START: &str = "zz-";\n'
                 'const NON_TOKEN_PREFIXES: &[&str] = &[\n    "aws_secret",\n    NEW_START,\n'
                 '    URL_AUTHORITY_START,\n];\nfn f() { let p = Pattern { prefixes: &["sk-", "ghp_"] }; }\n')
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp, "src", "review")
            src.mkdir(parents=True)
            (src / "secrets.rs").write_text(table, encoding="utf-8")
            self.assertIn("zz-", secret_openings(pathlib.Path(tmp)))
            (src / "secrets.rs").write_text(table.replace('const NEW_START: &str = "zz-";\n', ""),
                                            encoding="utf-8")
            with self.assertRaises(AssertionError):
                secret_openings(pathlib.Path(tmp))


    def test_a_matcher_it_cannot_read_is_refused(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7b): a pattern whose opening is not a literal prefix
        # contributed zero openings and no assertion fired.
        import tempfile
        table = ('const NON_TOKEN_PREFIXES: &[&str] = &["aws_secret"];\n'
                 'fn f() { let p = Pattern { prefixes: &["sk-", "ghp_"] }; }\n'
                 'const P: &[SecretPattern] = &[SecretPattern { name: "x", matcher: Matcher::%s }];\n')
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp, "src", "review")
            src.mkdir(parents=True)
            (src / "secrets.rs").write_text(table % "UrlCredential", encoding="utf-8")
            self.assertIn("sk-", secret_openings(pathlib.Path(tmp)))
            (src / "secrets.rs").write_text(table % "Novel(1)", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Novel"):
                secret_openings(pathlib.Path(tmp))


class S54Tests(unittest.TestCase):
    """S54 (R19): a private-key block file is left out as secret and its body never printed."""

    def _judge(self, index, result, log=b"") -> Outcome:
        return _outcomes("S54", result, log)[rm.S54_ASSERTIONS[index]]

    def test_a_refusal_naming_only_the_key_file_as_secret_passes(self) -> None:
        r = _result("R19", 2, None, _REFUSAL)
        self.assertEqual({Outcome.PASS}, set(_outcomes("S54", r).values()))

    def test_another_cause_fails_the_listing(self) -> None:
        r = _result("R19", 2, None, _REFUSAL.replace(b"(secret)", b"(binary)"))
        self.assertEqual(Outcome.FAIL, self._judge(1, r))

    def test_an_extra_omitted_file_fails_the_listing(self) -> None:
        r = _result("R19", 2, None, _REFUSAL.replace(b"(secret)", b"(secret), other.txt (binary)"))
        self.assertEqual(Outcome.FAIL, self._judge(1, r))

    def test_an_exit_other_than_two_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._judge(0, _result("R19", 0, None, _REFUSAL)))

    def test_a_body_line_in_stderr_fails_the_leak_assertion(self) -> None:
        r = _result("R19", 2, None, _REFUSAL + _BODY)
        self.assertEqual(Outcome.FAIL, self._judge(2, r))

    def test_a_partial_body_line_fails_the_leak_assertion(self) -> None:
        # [MAGI S6 MS3 loop7, R7a Melchior] A slice of a body line is a byte of the block too.
        r = _result("R19", 2, None, _REFUSAL)
        self.assertEqual(Outcome.FAIL, self._judge(2, r, log=b"x " + _BODY[13:25]))

    def test_the_tail_of_a_body_line_fails_the_leak_assertion(self) -> None:
        # [MAGI S6 MS3 loop8, R7a Melchior] Every window is probed, the last one included.
        r = _result("R19", 2, None, _REFUSAL)
        self.assertEqual(Outcome.FAIL, self._judge(2, r, log=b"x " + _BODY[-rm.MIN_LEAK_FRAGMENT:]))

    def test_body_fragments_below_eight_bytes_fail(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a/R7b Melchior): the floor was 8 bytes; 6 and 7 fail now.
        r = _result("R19", 2, None, _REFUSAL)
        for n in (6, 7):
            self.assertEqual(Outcome.FAIL, self._judge(2, r, log=b"x " + _BODY[3:3 + n]), n)

    def test_the_second_body_line_in_the_run_log_fails_too(self) -> None:
        r = _result("R19", 2, None, _REFUSAL)
        self.assertEqual(Outcome.FAIL, self._judge(2, r, log=runs.REVIEW_KEYBLOCK_BODY_LINES[1]))

    def test_the_block_header_or_footer_in_any_output_fails_the_leak_assertion(self) -> None:
        # [MAGI S6 MS3 loop4, R7a Caspar] RV22 leaves the whole block out, its marker lines too.
        for line in (runs.REVIEW_KEYBLOCK_HEADER, runs.REVIEW_KEYBLOCK_FOOTER):
            with self.subTest(line=line):
                r = _result("R19", 2, None, _REFUSAL + line)
                self.assertEqual(Outcome.FAIL, self._judge(2, r))

    def test_the_fixture_is_a_whole_block_with_unique_body_lines(self) -> None:
        (_, body), = runs.review_keyblock_files()
        lines = body.split(b"\n")
        self.assertTrue(lines[0].startswith(b"-----" + b"BEGIN "))
        self.assertTrue(lines[-2].startswith(b"-----" + b"END "))
        for line in runs.REVIEW_KEYBLOCK_BODY_LINES:
            self.assertEqual(1, lines.count(line))

    def test_a_hung_refusal_fails_every_assertion(self) -> None:
        r = _result("R19", 0, None, b"", timed_out=True)
        self.assertEqual({Outcome.FAIL}, set(_outcomes("S54", r).values()))

    def test_every_one_of_the_three_assertions_is_judged_on_a_hang_or_no_capture(self) -> None:
        # [MAGI S6 MS3 loop3, R7a Melchior; pinned] `_refusal`, written for S48/S49's two texts,
        # loops over all of the texts it is given on these paths, so S54's third is never lost.
        from smoke.scenarios.code_review import refusal as _refusal
        hung = _result("R19", 0, None, b"", timed_out=True)
        for run, outcome in ((None, Outcome.CANNOT_TEST), (hung, Outcome.FAIL)):
            with self.subTest(outcome=outcome):
                findings = list(_refusal(rm.S54_ASSERTIONS, "R19", run, lambda _t: True, ""))
                self.assertEqual(list(rm.S54_ASSERTIONS), [f.assertion for f in findings])
                self.assertEqual({outcome}, {f.outcome for f in findings})


if __name__ == "__main__":
    unittest.main()
