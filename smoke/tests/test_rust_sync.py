# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-29
"""The code-review scenarios' copies of the product's numbers and wording.

The harness reads no Rust at run time, so S47-S51 carry their own copies of
the batch constants, the per-batch floor and the texts they parse out of
``stderr``. A copy stops being true silently when the product changes, and the
first sign would be an opaque ``FAIL`` at SMOKE time. These tests read the Rust
source and fail here, at the harness's own gate, the moment the two sides
drift. Each pattern must match: a pattern that matched nothing would let the
comparison below it pass by never running.
"""

import pathlib
import re
import unittest

from smoke import logs, runs
from smoke.scenarios import code_review

#: The repository root, three levels above this file.
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _source(relative: str) -> str:
    """Read a Rust source file with its line endings normalised.

    rustfmt writes CRLF on Windows and LF elsewhere, so ``\\r`` is stripped
    before any pattern runs. Bytes that are not UTF-8 (a test literal) are
    replaced; nothing read here lives in them.

    Args:
        relative: The path under the repository root, with forward slashes.

    Returns:
        str: The file's text.
    """
    return (_ROOT / relative).read_bytes().decode("utf-8", errors="replace").replace("\r", "")


def _usize_const(text: str, name: str) -> int:
    """The value of ``pub const <name>: usize = <n>;`` in *text*.

    Args:
        text: A Rust source file's text.
        name: The constant's name.

    Returns:
        int: Its value, with ``_`` separators removed.

    Raises:
        AssertionError: When the declaration is not found.
    """
    match = re.search(r"pub const %s: usize = ([\d_]+);" % re.escape(name), text)
    if match is None:
        raise AssertionError("pub const %s: usize not found" % name)
    return int(match.group(1).replace("_", ""))


class BatchConstantTests(unittest.TestCase):
    """The batch arithmetic R12's input is sized with (``smoke/runs.py``)."""

    # MAGI FIX (S6 loop 5): the copies had no automated sync (F1-b2).
    def setUp(self) -> None:
        self.window = _source("src/review/window.rs")

    def test_bytes_per_token_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "BYTES_PER_TOKEN"), runs.BYTES_PER_TOKEN)

    def test_the_unmeasured_window_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "DEFAULT_UNMEASURED_WINDOW_TOKENS"),
                         runs.UNMEASURED_WINDOW_TOKENS)

    def test_the_harness_sizes_for_a_batch_at_least_as_large_as_the_product_fills(self) -> None:
        # The harness sizes at the larger of the product's two values (50 provisional, 60 measured);
        # a product fraction above the harness's would plan fewer batches than R12's input assumes.
        percent = _usize_const(self.window, "BATCH_WINDOW_FRACTION_PERCENT")
        self.assertGreaterEqual(round(runs.BATCH_WINDOW_FRACTION * 100), percent)

    def test_the_floor_is_the_products_two_minimums(self) -> None:
        text = _source("src/magi/mod.rs")
        found = {name: int(secs) for name, secs in re.findall(
            r"const (MIN_OPERATION_BUDGET|MIN_CLIENT_TIMEOUT): Duration = Duration::from_secs\((\d+)\);", text)}
        self.assertEqual({"MIN_OPERATION_BUDGET", "MIN_CLIENT_TIMEOUT"}, set(found))
        self.assertRegex(text, r"AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS: u64 =\s*"
                               r"MIN_OPERATION_BUDGET\.as_secs\(\) \+ MIN_CLIENT_TIMEOUT\.as_secs\(\);")
        self.assertEqual(sum(found.values()), code_review.FLOOR_SECS)

    def test_the_least_need_per_batch_is_the_products_lenient_arithmetic(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c3] S50's bounds lean on the classifier reserve and the slack.
        text = _source("src/magi/mod.rs")
        self.assertIn("pub const CLASSIFY_TIMEOUT_SECS: u64 = %d;" % code_review.CLASSIFY_SECS, text)
        self.assertIn("pub const HEADLESS_TIMEOUT_SLACK_PCT: u64 = %d;" % code_review.SLACK_PCT, text)
        self.assertEqual(24, code_review.MIN_NEED_PER_BATCH_SECS)

    def test_the_budget_split_and_ceiling_are_the_ones_s50_checks(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-m2] _budget_fields_ok asserts wall == total // batches and a ceiling
        # below the floor: pre_budget's integer split and its RAW (never floor-lifted) refusal ceiling.
        text = _source("src/review/clock.rs")
        self.assertRegex(text, r"Ok\(n\) => remaining_secs / n,")
        self.assertRegex(text, r"let per_batch = batch_wall_clock_secs\(total_secs, batches\);")
        self.assertRegex(text, r"per_batch_ceiling_secs: raw_ceiling_from_factor\(\s*per_batch,")


class StderrWordingTests(unittest.TestCase):
    """The texts S48-S51 parse out of the product's ``stderr``."""

    # MAGI FIX (S6 loop 5): the regexes were tied to the product's wording with
    # nothing to say when that wording moved (F1-b1).
    def setUp(self) -> None:
        self.error = _source("src/review/error.rs")
        self.filter = _source("src/review/filter.rs")

    def test_the_omission_causes_are_the_products(self) -> None:
        body = re.search(r"fn as_str\(self\) -> &'static str \{(.*?)\n    \}\n", self.filter, re.S)
        self.assertIsNotNone(body, "SkipCause::as_str not found in src/review/filter.rs")
        causes = re.findall(r'Self::\w+ => "([^"]+)"', body.group(1))
        # [MAGI §6 MS2 loop 12] an arm may name a constant instead of a literal (HARD_LINK_CAUSE, which
        # src/review/source.rs defines because the explicit-file error shares it); resolve it there.
        source = _source("src/review/source.rs")
        for name in re.findall(r"Self::\w+ => ([A-Z_]+),", body.group(1)):
            value = re.search(r'pub const %s: &str = "([^"]+)";' % name, source)
            self.assertIsNotNone(value, "%s not found in src/review/source.rs" % name)
            causes.append(value.group(1))
        self.assertTrue(causes)
        self.assertEqual(sorted(causes), sorted(code_review.OMISSION_CAUSES))

    def test_missing_is_an_error_cause_never_an_omission_cause(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-c2] "missing" names an explicit file in NotReviewable; it is not a
        # SkipCause, so it can never reach the omission list and OMISSION_CAUSES must not carry it.
        self.assertIn('pub(crate) const MISSING: &str = "missing";', self.filter)
        self.assertIn('#[error("{} is not reviewable: {cause}", display_path(.path))]', self.error)
        self.assertNotIn('=> "missing"', self.filter)
        self.assertNotIn("missing", code_review.OMISSION_CAUSES)

    def test_the_omission_list_is_rendered_as_the_patterns_expect(self) -> None:
        self.assertIn('write!(f, "{} ({})", self.path, self.cause.as_str())', self.filter)
        self.assertIn('format!("; left out: {}{more}", shown.join(", "))', self.error)
        self.assertIn('format!(", and {rest} more")', self.error)
        rendered = "error: x; left out: a b (binary), .magi/c (ignored), and 2 more"
        entries, failure = code_review._omissions(rendered)
        self.assertEqual("", failure)
        self.assertEqual([("a b", "binary"), (".magi/c", "ignored")], entries)

    def test_the_budget_refusal_template_matches_the_pattern(self) -> None:
        template = re.search(r'#\[error\(\s*"(\{batches\} batches leave.*?)",\s*budget_advice', self.error, re.S)
        self.assertIsNotNone(template, "the Budget message was not found in src/review/error.rs")
        self.assertIn('BudgetLimit::Timeout => format!("pass --timeout {min_secs} or more")', self.error)
        template_text = re.sub(r"\\\n\s*", "", template.group(1))
        # [MAGI §6 MS2 loop 10, F1a-m1] the count is the bare "{batches} batches" at every N: no plural
        # branch exists in the template, so one batch renders "1 batches" and the pattern still matches.
        self.assertTrue(template_text.startswith("{batches} batches leave "), template_text)
        # R15's 20 s under the default config: one batch gets 20 s (raw ceiling 1 s), two get 10 s (0 s).
        for batches, wall, ceiling in (("1", "20", "1"), ("2", "10", "0")):
            values = {"batches": batches, "per_batch_wall_clock_secs": wall, "min_per_batch_secs": "114",
                      "floor_secs": "15", "per_batch_ceiling_secs": ceiling}
            text = template_text
            for name, value in values.items():
                text = text.replace("{%s}" % name, value)
            minimum = str(int(batches) * 114)
            text = text.replace("{}", "pass --timeout %s or more" % minimum)
            match = code_review.BUDGET_PATTERN.fullmatch(text)
            self.assertIsNotNone(match, text)
            self.assertEqual((batches, wall, "114", "15", ceiling, minimum),
                             match.group("batches", "wall", "need", "floor", "ceiling", "minimum"))
            self.assertTrue(code_review._budget_fields_ok(match), text)

    def test_the_fixed_phrases_are_the_products(self) -> None:
        self.assertIn('"path is outside the workspace: {}"', self.error)
        self.assertIn(code_review.OUTSIDE_WORKSPACE_TEXT, "path is outside the workspace")
        self.assertIn('"nothing to review: no reviewable file in the input{}"', self.error)
        self.assertIn(code_review.NOTHING_TO_REVIEW_TEXT, "nothing to review: no reviewable file")


class ReportKeyTests(unittest.TestCase):
    """The JSON keys S47 reads exist in the product's report and envelope."""

    # MAGI FIX (S6 loop 6): S47's keys sat outside the drift guard, so a rename
    # would surface as an unexplained FAIL at SMOKE time (F1a-m2).
    def test_every_summary_key_s47_reads_is_one_the_report_pins(self) -> None:
        scenario = _source("smoke/scenarios/code_review.py")
        read = set(re.findall(r'section\.get\("(\w+)"\)', scenario))
        read.update(re.findall(r'_list_under\(section, "(\w+)"\)', scenario))
        self.assertTrue(read, "no code_review key read by S47 was found")
        report = _source("src/review/report.rs")
        pinned = re.search(r"every documented key is present.*?for k in \[(.*?)\]", report, re.S)
        self.assertIsNotNone(pinned, "the summary key list was not found in src/review/report.rs")
        self.assertEqual(set(), read - set(re.findall(r'"(\w+)"', pinned.group(1))))

    def test_the_entry_fields_s47_reads_are_serialized_under_their_own_names(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-m1] S47 reads file and line off each main finding and cause off each
        # discarded one; a serde rename on either struct would surface as an opaque FAIL at SMOKE time.
        for path, struct, fields in (("src/review/consolidate.rs", "ConsolidatedFinding",
                                      ("pub file: Option<String>,", "pub line: Option<u32>,")),
                                     ("src/review/validate.rs", "Discarded", ("pub cause: DiscardCause,",))):
            body = re.search(r"#\[derive\([^)]*serde::Serialize[^)]*\)\]\npub struct %s \{(.*?)\n\}" % struct,
                             _source(path), re.S)
            self.assertIsNotNone(body, "%s was not found in %s" % (struct, path))
            self.assertNotIn("rename", body.group(1), struct)
            for field in fields:
                self.assertIn(field, body.group(1), struct)

    def test_the_run_field_the_harness_matches_is_the_one_the_log_writes(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-m1] S50 binds its run-log line to this run through logs.run_needle;
        # the product spells the field once, in src/logging/render.rs.
        self.assertIn('pub(crate) const RUN_FIELD: &str = "%s";' % logs.RUN_FIELD,
                      _source("src/logging/render.rs"))
        self.assertEqual(b"run=4242-deadbeef", logs.run_needle("4242-deadbeef"))

    def test_the_stderr_run_id_is_printed_whatever_the_log_level(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-m2/b4] S50 runs at --log-level error and reads `run: <id>` off stderr.
        # The product prints it with eprint_audited, outside the tracing layer, so no level filters it; a
        # move into a tracing macro would make S50's log half unreachable and must fail here first.
        main = _source("src/main.rs")
        needle = 'magi_rs::notices::eprint_audited(&format!("%s{}", magi_rs::logging::run_id()));'
        self.assertIn(needle % logs.STDERR_RUN_PREFIX, main)
        self.assertNotRegex(main, r'(?:info|warn|error|debug|trace|event)!\([^;]*"%s\{\}"'
                            % re.escape(logs.STDERR_RUN_PREFIX))

    def test_the_consensus_s47_reads_is_the_one_the_envelope_writes(self) -> None:
        self.assertIn('"consensus".into(),\n        crate::tools::consult::consensus_json(&c.consensus)',
                      _source("src/tools/code_review.rs"))
        consult = _source("src/tools/consult.rs")
        for key in code_review.CONSENSUS_TEXT_KEYS:
            self.assertIn('"%s": c.%s,' % (key, key), consult)


class GitConfigTests(unittest.TestCase):
    """The harness's git settings are a subset of the product's."""

    # MAGI FIX (S6 loop 8): HARNESS_GIT_CONFIG claimed to be a subset of
    # NEUTRAL_GIT_CONFIG with nothing checking it (F1a-m3).
    def test_every_harness_git_setting_is_one_the_product_passes(self) -> None:
        listing = re.search(r"pub\(crate\) const NEUTRAL_GIT_CONFIG: &\[&str\] = &\[(.*?)\];",
                            _source("src/system/review_git.rs"), re.S)
        self.assertIsNotNone(listing, "NEUTRAL_GIT_CONFIG was not found in src/system/review_git.rs")
        product = set(re.findall(r'"([^"]*)"', listing.group(1)))
        self.assertTrue(runs.HARNESS_GIT_CONFIG)
        self.assertEqual(set(), set(runs.HARNESS_GIT_CONFIG) - product)

    # MAGI FIX (S6 loop 11, F1a-m2): GIT_VISIBLE_QUERY copies the product's
    # enumeration query with nothing checking it; a product query that dropped
    # one of its flags would surface as an opaque SMOKE FAIL.
    def test_the_visibility_query_uses_only_flags_the_products_listing_passes(self) -> None:
        listing = re.search(r"const LS_FILES_ALL: &\[&str\] = &\[(.*?)\];",
                            _source("src/review/input.rs"), re.S)
        self.assertIsNotNone(listing, "LS_FILES_ALL was not found in src/review/input.rs")
        product = re.findall(r'"([^"]*)"', listing.group(1))
        self.assertEqual("ls-files", runs.GIT_VISIBLE_QUERY[0])
        self.assertEqual(product[0], runs.GIT_VISIBLE_QUERY[0])
        self.assertEqual(set(), set(runs.GIT_VISIBLE_QUERY) - set(product))


class SecretPrefixDocTests(unittest.TestCase):
    """The README never names a secret prefix the detector does not check."""

    # MAGI FIX (S6 loop 5): the README's pattern list could drift past
    # src/review/secrets.rs and promise a protection that is not there (F2-b1).
    def test_every_prefix_the_readme_names_is_a_detector_prefix(self) -> None:
        readme = _source("README.md")
        sentence = re.search(r"The patterns are named prefixes and shapes[.:](.*?)Not covered, by design", readme, re.S)
        self.assertIsNotNone(sentence, "the README's secret pattern list was not found")
        named = {token for token in re.findall(r"`([^`\s]+)`", sentence.group(1))
                 if re.fullmatch(r"[A-Za-z]+(?:[-_][a-z]+)*[-_]|AKIA|ASIA|AIza", token)}
        self.assertTrue(named)
        prefixes = set()
        for array in re.findall(r"prefixes: &\[([^\]]*)\]", _source("src/review/secrets.rs")):
            prefixes.update(re.findall(r'"([^"]+)"', array))
        self.assertEqual(set(), named - prefixes)


if __name__ == "__main__":
    unittest.main()
