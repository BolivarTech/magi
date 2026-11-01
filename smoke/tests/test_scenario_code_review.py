# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for S47-S51 (code-review against the release binary)."""

import dataclasses
import json
import pathlib
import tempfile
import types
import unittest

from smoke import runner as runner_module
from smoke import runs
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY, Registry
from smoke.runner import Ambient, Runner
from smoke.runs import RunResult
from smoke.scenarios import code_review  # noqa: F401 - import registers it

#: The ``.magi/`` files git lists in R16's workspace when the environment's ``.magi`` is copied in (the three
#: :data:`_ROOT_STDERR` names). [MAGI §6 MS2 loop 9, F1a-c1] S51 reads them off the result.
_R16_PROTECTED = (".magi/.magi-rs-memory.db", ".magi/logs/magi-2026-10-29.log", ".magi/magi.toml")


def _result(run_id: str, exit_code: int = 0, document=None, stderr: bytes = b"", timed_out: bool = False,
            review_log=None, protected_visible=None) -> RunResult:
    """A real ``RunResult`` for *run_id* whose stdout is *document* serialised (``None`` → empty stdout).

    *protected_visible* defaults to :data:`_R16_PROTECTED` for R16 and to nothing for every other run.
    """
    body = b"" if document is None else json.dumps(document).encode()
    if protected_visible is None:
        protected_visible = _R16_PROTECTED if run_id == "R16" else ()
    return RunResult(run_id=run_id,
                     output=ProductOutput(stdout=body, stderr=stderr, exit_code=exit_code, command=["magi-rs"]),
                     duration_s=1.0, timed_out=timed_out, planted=(), review_log=review_log,
                     protected_visible=protected_visible)


RUN_ID = "4242-0123456789abcdef"
#: [MAGI §6 MS2 loop 10, F1a-c3] the product's arithmetic (``pre_budget``) for R15's 20 s total over three
#: batches under the default ``max_rotations = 2`` with retries on: 20 // 3 = 6 s per batch; a batch needs
#: 6 + 15 x 7.2 = 114 s; the per-batch ceiling is the RAW one, (6 - 6) x 100 // 720 = 0 s, never lifted to
#: the floor; the minimum is 3 x 114 = 342 s.
_BUDGET_MSG = (b"3 batches leave 6 s per batch, less than the 114 s a batch needs at the 15 s floor "
               b"(per-batch ceiling 0 s); pass --timeout 342 or more")
_NL = bytes([10])
_BUDGET_STDERR = b"error: " + _BUDGET_MSG + _NL
#: One run-log line of the refusal; ``%s`` is the run id.
_BUDGET_LOG = b'ERROR run=%s target=magi_rs::code_review ' + _BUDGET_MSG + _NL


def _product_refusal(batches: int, rotations: int, retry_disabled: bool) -> bytes:
    """The refusal ``pre_budget`` renders for R15's total over *batches* under one ``[magi]`` config.

    A mirror of ``src/review/clock.rs`` and ``src/magi/mod.rs`` for the tests only: classifier 6 s, attempt
    factor ``calls x models x 120`` in hundredths, raw ceiling ``(wall - 6) x 100 // factor``.
    """
    factor = (1 if retry_disabled else 2) * (rotations + 1) * 120
    wall = runs.REVIEW_TOO_SHORT_TIMEOUT_S // batches
    need = 6 + 15 * factor // 100
    ceiling = max(wall - 6, 0) * 100 // factor
    return (b"error: %d batches leave %d s per batch, less than the %d s a batch needs at the 15 s floor "
            b"(per-batch ceiling %d s); pass --timeout %d or more" % (batches, wall, need, ceiling, batches * need)
            + _NL)


def _with_id(tail: bytes) -> bytes:
    """Stderr as the product prints it: the ``run: <id>`` line, then *tail*."""
    return b"run: " + RUN_ID.encode() + _NL + tail


def _outcomes(scenario_id: str, result, log=b"") -> dict[str, Outcome]:
    """Outcomes of *scenario_id*; *log* is the run log captured on the result (``None``: no directory).

    [MAGI §6 MS2 loop 3, F4] the log rides on the ``RunResult`` the executor captured right after the
    run; the scratch area handed to ``runs`` stays empty, so a scenario that read the live directory
    would find nothing.
    """
    result = dataclasses.replace(result, review_log=None if log is None else (log,))
    with tempfile.TemporaryDirectory() as tmp:
        scratch = pathlib.Path(tmp)
        runs.configure(None, types.SimpleNamespace(scratch_dir=scratch, prepare_scratch=lambda: None), None)
        try:
            return {f.assertion: f.outcome for f in DEFAULT_REGISTRY.get(scenario_id).func(result)}
        finally:
            runs.reset_for_test()


def _review(batches: int = 3, findings=None, discarded=None) -> dict:
    # [CP2 MS2 S2 loop 1] the reviewed set is code_review.S47_FILES, the scenario's own generated input.
    # [CP2 MS2 S2 loop 2] the main findings live in code_review.findings (summary_json), not in consult.
    # [CP2 MS2 S2b loop 6] code_review.discarded, RV-S5's other half, each entry with its cause.
    # [CP2 MS2 S2b loop 8] unmeasured_models: RV04's window; with one unmeasured model it is the 32 768 default.
    # [MAGI §6 MS2 loop 6, F1a-c3] a consensus object with magi-core's label and verdict.
    return {"consult": {"consensus": {"consensus": "STRONG GO", "consensus_verdict": "approve",
                                      "agent_count": 3}},
            # [MAGI §6 MS2 loop 12, F1a-c3] the product always writes failed_batches, empty when all ran.
            "code_review": {"batches_planned": batches, "batches_completed": batches, "failed_batches": [],
                            "unmeasured_models": ["kimi-k2.6:cloud"],
                            "findings": [{"file": "gen/a.rs", "line": 1}] if findings is None else findings,
                            "discarded": [{"cause": "file not reviewed"}] if discarded is None else discarded}}


#: R16's stderr as the release binary printed it over a workspace root holding the environment's
#: ``.magi`` and R14's two skipped files (measured 2026-10-29).
_ROOT_STDERR = (
    b"error: nothing to review: no reviewable file in the input; left out: "
    b".magi/.magi-rs-memory.db (ignored), .magi/logs/magi-2026-10-29.log (ignored), "
    b".magi/magi.toml (ignored), skipped/leak.txt (secret), skipped/pixel.png (binary) "
    b"review.error=nothing_to_review" + _NL)


#: R14's stderr: the ``skipped`` directory's binary and secret, each left out with its cause.
_SKIPPED_STDERR = (
    b"error: nothing to review: no reviewable file in the input; left out: "
    b"skipped/leak.txt (secret), skipped/pixel.png (binary) review.error=nothing_to_review" + _NL)


class WorkspaceRootScenarioTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 3, F2] S51: reviewing the workspace root never takes .magi or .git."""

    def test_s51_is_registered_on_r16_without_a_backend(self) -> None:
        entry = DEFAULT_REGISTRY.get("S51")
        self.assertEqual("R16", entry.run)
        self.assertFalse(entry.needs_backend)

    def test_s51_passes_when_the_root_has_nothing_to_review_and_magi_is_left_out_as_ignored(self) -> None:
        r = _result("R16", 2, None, _with_id(_ROOT_STDERR))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S51", r).values()))

    def test_s51_fails_when_git_lists_a_magi_file_the_refusal_never_names(self) -> None:
        # [MAGI §6 MS2 loop 9, F1a-c1/F1a-m2] this used to PASS: a refusal naming no protected path left the
        # third assertion vacuous. Git listed .magi/ files, so the product must name them as ignored.
        r = _result("R16", 2, None, _with_id(
            b"error: nothing to review: no reviewable file in the input; left out: "
            b"skipped/leak.txt (secret), skipped/pixel.png (binary)" + _NL))
        outcomes = _outcomes("S51", r)
        self.assertEqual(Outcome.PASS, outcomes[code_review.S51_ASSERTIONS[1]])
        self.assertEqual(Outcome.FAIL, outcomes[code_review.S51_ASSERTIONS[2]])

    def test_s51_cannot_test_the_exclusion_when_git_lists_no_magi_file(self) -> None:
        # [MAGI §6 MS2 loop 9, F1a-c1/F1a-m2] a global ignore rule hiding .magi/ hides it from the product too:
        # no protected path was an enumeration candidate, so the exclusion half did not run.
        r = _result("R16", 2, None, _with_id(_ROOT_STDERR.replace(
            b".magi/.magi-rs-memory.db (ignored), .magi/logs/magi-2026-10-29.log (ignored), "
            b".magi/magi.toml (ignored), ", b"")), protected_visible=())
        outcomes = _outcomes("S51", r)
        self.assertEqual(Outcome.PASS, outcomes[code_review.S51_ASSERTIONS[1]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[code_review.S51_ASSERTIONS[2]])

    def test_s51_matches_a_backslash_separated_magi_name_against_what_git_lists(self) -> None:
        stderr = _ROOT_STDERR.replace(b".magi/magi.toml (ignored)", b".magi\\magi.toml (ignored)")
        r = _result("R16", 2, None, _with_id(stderr), protected_visible=(".magi/magi.toml",))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S51", r).values()))

    def test_s51_reads_a_backslash_in_what_git_lists_as_a_slash_too(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a2-c2] the omission list was read with "\\" as "/" but the listed .magi/
        # paths were compared raw, so a backslash-separated listing never intersected and S51 false-FAILed.
        r = _result("R16", 2, None, _with_id(_ROOT_STDERR), protected_visible=(".magi\\magi.toml",))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S51", r).values()))

    def test_s51_never_passes_over_a_protected_name_the_omission_list_cannot_split(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-c3] a .magi name holding " (ignored), " reads as two entries, so the
        # product classifying it as a secret looked like a protected path left out unread plus another file.
        planted = ".magi/z (ignored), q"
        stderr = _ROOT_STDERR.replace(b"skipped/pixel.png (binary)",
                                      planted.encode() + b" (secret), skipped/pixel.png (binary)")
        r = _result("R16", 2, None, _with_id(stderr), protected_visible=_R16_PROTECTED + (planted,))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S51", r)[code_review.S51_ASSERTIONS[2]])

    def test_s51_fails_when_a_protected_file_was_taken_for_review(self) -> None:
        # A .magi/magi.toml taken for review makes the input reviewable: the run reaches the budget
        # refusal (--timeout 20) instead of "nothing to review".
        r = _result("R16", 2, None, _with_id(_BUDGET_STDERR))
        outcomes = _outcomes("S51", r)
        self.assertEqual(Outcome.FAIL, outcomes[code_review.S51_ASSERTIONS[1]])

    def test_s51_fails_when_a_protected_path_was_read_for_its_content(self) -> None:
        for path in (b".magi/.magi-rs-memory.db (binary)", b".git/config (secret)",
                     b"sub/.magi/magi.toml (not utf-8)"):
            stderr = _ROOT_STDERR.replace(b"skipped/pixel.png (binary)", path)
            r = _result("R16", 2, None, _with_id(stderr))
            self.assertEqual(Outcome.FAIL, _outcomes("S51", r)[code_review.S51_ASSERTIONS[2]], path)

    def test_s51_fails_when_a_read_protected_path_holds_a_comma_semicolon_or_space(self) -> None:
        # [MAGI §6 MS2 loop 4, F1-1] the old entry pattern stopped a path at ',', ';' or a space, so the
        # protected component fell outside the match and a read .magi file passed unseen.
        for path in (b".magi/a,b.db (binary)", b".magi/x;y.toml (secret)", b".git/with space (not utf-8)",
                     b"sub dir/.magi/z.db (binary)"):
            stderr = _ROOT_STDERR.replace(b"skipped/pixel.png (binary)", path)
            r = _result("R16", 2, None, _with_id(stderr))
            self.assertEqual(Outcome.FAIL, _outcomes("S51", r)[code_review.S51_ASSERTIONS[2]], path)

    def test_s51_fails_when_the_omission_list_cannot_be_parsed(self) -> None:
        # [MAGI §6 MS2 loop 4, F1-5] a drifted rendering used to match nothing and pass vacuously. R16's
        # root always holds R14's two skipped files, so a missing list is drift as well.
        drifted = (
            _ROOT_STDERR.replace(b" (ignored)", b" [ignored]").replace(b" (secret)", b" [secret]")
            .replace(b" (binary)", b" [binary]"),
            _ROOT_STDERR.replace(b".magi/magi.toml (ignored)", b".magi/magi.toml (read)"),
            _ROOT_STDERR.replace(b"; left out: ", b"; skipped: "),
        )
        for stderr in drifted:
            r = _result("R16", 2, None, _with_id(stderr))
            self.assertEqual(Outcome.FAIL, _outcomes("S51", r)[code_review.S51_ASSERTIONS[2]], stderr)

    def test_s51_fails_when_the_omission_list_is_empty(self) -> None:
        # [MAGI S6 MS2 loop 5, F1-m3/F1-c1] the product writes "; left out: " only before at least one
        # entry (``omissions`` in src/review/error.rs), so an empty list is drift and must not pass.
        for tail in (b"", b" review.error=nothing_to_review"):
            stderr = b"error: nothing to review: no reviewable file in the input; left out: " + tail + _NL
            r = _result("R16", 2, None, _with_id(stderr))
            self.assertEqual(Outcome.FAIL, _outcomes("S51", r)[code_review.S51_ASSERTIONS[2]], tail)

    def test_s51_reads_an_omission_list_cut_at_the_display_cap(self) -> None:
        stderr = _ROOT_STDERR.replace(b"skipped/pixel.png (binary)", b"skipped/pixel.png (binary), and 3 more")
        r = _result("R16", 2, None, _with_id(stderr))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S51", r).values()))

    def test_s51_cannot_test_the_exclusion_when_the_cap_may_have_cut_every_visible_magi_file(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a1-m2] the product names at most MAX_OMISSIONS_SHOWN entries; when it cut
        # the list and none of the .magi/ files git listed is among those shown, they may be in the cut tail,
        # so "names none of them" is the environment's size, never a product FAIL.
        stderr = (b"error: nothing to review: no reviewable file in the input; left out: "
                  b"skipped/leak.txt (secret), skipped/pixel.png (binary), and 3 more" + _NL)
        r = _result("R16", 2, None, _with_id(stderr))
        outcome = [f for f in DEFAULT_REGISTRY.get("S51").func(r)][2]
        self.assertEqual(Outcome.CANNOT_TEST, outcome.outcome)
        self.assertIn("3 more", outcome.detail)

    def test_s51_fails_on_a_wrong_exit(self) -> None:
        r = _result("R16", 0, None, _with_id(_ROOT_STDERR))
        self.assertEqual(Outcome.FAIL, _outcomes("S51", r)[code_review.S51_ASSERTIONS[0]])

    def test_s51_without_a_capture_cannot_test(self) -> None:
        findings = list(DEFAULT_REGISTRY.get("S51").func(None))
        self.assertEqual({Outcome.CANNOT_TEST}, {f.outcome for f in findings})
        self.assertEqual(len(code_review.S51_ASSERTIONS), len(findings))


class CodeReviewScenarioTests(unittest.TestCase):

    def test_registration(self) -> None:
        self.assertTrue(DEFAULT_REGISTRY.get("S47").needs_backend)
        for sid in ("S48", "S49", "S50"):
            self.assertFalse(DEFAULT_REGISTRY.get(sid).needs_backend, sid)

    def test_s47_passes_on_one_report_over_three_batches_with_findings_inside(self) -> None:
        r = _result("R12", 0, _review(3, [{"file": "gen/b.rs", "line": 4}]))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S47", r).values()))

    def test_s47_fails_below_three_batches(self) -> None:
        r = _result("R12", 0, _review(2))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[2]])

    def test_s47_fails_when_a_planned_batch_did_not_complete(self) -> None:
        # [MAGI §6 MS2 F loop 1] one report over five planned batches of which one ran is not a
        # report over three batches: every planned batch must have completed.
        doc = _review(5)
        doc["code_review"]["batches_completed"] = 1
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])
        del doc["code_review"]["batches_completed"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_cannot_test_a_shortfall_whose_failures_are_all_provider_class(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-m1/b1] a batch lost to quota, transport, a timeout or auth is the
        # environment refusing (PM-S-8), never a product FAIL.
        for why in ("http error 429: usage credits exhausted", "network error: connection reset",
                    "timeout: the seat did not answer", "auth error: 401", "endpoint down: no lineage reachable (a)",
                    "retry abandoned after 2 attempt(s): OperationBudgetExhausted",
                    "insufficient agents: 0 succeeded, 2 required", "clock expired"):
            doc = _review(5)
            # [MAGI §6 MS2 loop 13, F1a-b1] batch 5: a clock expiry names every batch from the cut one to the last.
            doc["code_review"].update(batches_completed=4, failed_batches=[[5, why]])
            self.assertEqual(Outcome.CANNOT_TEST,
                             _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]], why)

    def test_s47_fails_a_shortfall_that_is_not_all_provider_class(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-m1/b1] any other shortfall stays FAIL: a cause that is not the backend's,
        # one provider failure beside it, or a shortfall the failed list does not account for.
        cases = ([[3, "response contract violated: no verdict"]],
                 [[3, "timeout: slow"], [4, "the trio build did not finish: panicked"]],
                 [], [[3, "timeout: slow"], [4, "timeout: slow"]], [["3", 7]], "timeout")
        for failed in cases:
            doc = _review(5)
            doc["code_review"].update(batches_completed=4, failed_batches=failed)
            self.assertEqual(Outcome.FAIL,
                             _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]], failed)

    def test_s47_fails_a_provider_shortfall_whose_batch_ids_are_not_the_missing_batches(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-b1] the ids were never read: a repeated, zero-based, out-of-range or
        # non-integer id, a completed batch declared failed, or a clock expiry that is not the tail all read as
        # CANNOT_TEST beside a provider-class cause.
        completed_batch = {"index": 2, "input_tokens": 900, "output_tokens": 80, "ms": 10}
        cases = (
            (3, [[2, "timeout: a"], [2, "timeout: b"]], None),
            (4, [[0, "timeout: a"]], None),
            (4, [[6, "timeout: a"]], None),
            (4, [[True, "timeout: a"]], None),
            (4, [[2.0, "timeout: a"]], None),
            (4, [[2, "timeout: a"]], [completed_batch]),
            (3, [[3, "clock expired"], [4, "timeout: a"]], None),
            (3, [[3, "clock expired"], [4, "clock expired"]], None),
        )
        for completed, failed, metrics in cases:
            doc = _review(5)
            doc["code_review"].update(batches_completed=completed, failed_batches=failed)
            if metrics is not None:
                doc["code_review"]["batches"] = metrics
            self.assertEqual(Outcome.FAIL,
                             _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]], failed)

    def test_s47_cannot_test_a_clock_expiry_over_the_last_batches(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-b1] the product declares every batch from the one the clock cut to the
        # last as clock expired, after any earlier provider failure.
        doc = _review(5)
        doc["code_review"].update(
            batches_completed=2, failed_batches=[[2, "timeout: a"], [4, "clock expired"], [5, "clock expired"]],
            batches=[{"index": 2, "input_tokens": 0, "output_tokens": 0, "ms": 10}])
        self.assertEqual(Outcome.CANNOT_TEST,
                         _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_fails_when_the_counts_agree_but_a_batch_is_declared_failed(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-c3] "every one completed" is checked against the failed list too.
        doc = _review(3)
        doc["code_review"]["failed_batches"] = [[2, "timeout: slow"]]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])
        del doc["code_review"]["failed_batches"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])
        self.assertIn("failed_batches", code_review.S47_ASSERTIONS[2])

    def test_s47_below_three_batches_with_every_model_measured_cannot_test(self) -> None:
        # [CP2 MS2 S2b loop 8] the generator is sized for the 32 768-token unmeasured window (RV04); a pool that
        # measured a larger window plans fewer batches for an environmental reason: CANNOT_TEST, never FAIL.
        doc = _review(2)
        doc["code_review"]["unmeasured_models"] = []
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_fails_on_a_finding_outside_the_reviewed_files(self) -> None:
        r = _result("R12", 0, _review(3, [{"file": "elsewhere.rs", "line": 1}]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_skips_findings_without_a_file_in_the_location_check(self) -> None:
        # [CP2 MS2 S2b loop 9] an unlocated finding is not an invented location (RV09).
        r = _result("R12", 0, _review(3, [{"file": None, "line": None}, {"file": "gen/a.rs", "line": 1}]))
        self.assertEqual(Outcome.PASS, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])
        r = _result("R12", 0, _review(3, [{"file": None, "line": None}]))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_fails_on_a_finding_with_a_line_but_no_file(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c1] a line without a file is malformed, never an unlocated finding: a
        # report holding only such findings must FAIL, not fall to CANNOT_TEST.
        for findings in ([{"file": None, "line": 5}],
                         [{"file": None, "line": 5}, {"file": "gen/a.rs", "line": 1}],
                         [{"line": 99999}]):
            r = _result("R12", 0, _review(3, findings))
            self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]], findings)

    def test_s47_reads_a_backslash_separated_file_as_the_generated_one(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-b1] S49 and S51 read a backslash as ``/``; S47 must too, or a Windows
        # rendering of a generated file false-FAILs. The line is still checked against that file.
        count = runs.review_gen_line_counts()["gen/a.rs"]
        r = _result("R12", 0, _review(3, [{"file": "gen\\a.rs", "line": count}]))
        self.assertEqual(Outcome.PASS, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])
        r = _result("R12", 0, _review(3, [{"file": "gen\\a.rs", "line": count + 1}]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s48_passes_on_a_misuse_exit_naming_the_path(self) -> None:
        r = _result("R13", 2, None, b"error: path is outside the workspace: C:\\ws\\..\\smoke-hostile-target\n")
        self.assertEqual({Outcome.PASS}, set(_outcomes("S48", r).values()))

    def test_s48_fails_when_stderr_has_the_phrase_but_not_the_path(self) -> None:
        # The fixed phrase alone must not satisfy "names the outside path".
        r = _result("R13", 2, None, b"error: path is outside the workspace: ?\n")
        self.assertEqual(Outcome.FAIL, _outcomes("S48", r)[code_review.S48_ASSERTIONS[1]])

    def test_s49_passes_on_nothing_to_review(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c4] the refusal must leave out R14's two files with their causes.
        r = _result("R14", 2, None, _SKIPPED_STDERR)
        self.assertEqual({Outcome.PASS}, set(_outcomes("S49", r).values()))

    def test_s49_fails_when_the_omission_causes_are_missing_or_wrong(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c4] "nothing to review" alone, swapped causes, a missing file or an
        # extra one is not the classification R14 sets up.
        bare = b"error: nothing to review: no reviewable file in the input" + _NL
        swapped = _SKIPPED_STDERR.replace(b"(secret)", b"(tmp)").replace(b"(binary)", b"(secret)").replace(
            b"(tmp)", b"(binary)")
        missing = _SKIPPED_STDERR.replace(b", skipped/pixel.png (binary)", b"")
        extra = _SKIPPED_STDERR.replace(b" review.error", b", skipped/x.rs (ignored) review.error")
        for stderr in (bare, swapped, missing, extra):
            r = _result("R14", 2, None, stderr)
            self.assertEqual(Outcome.FAIL, _outcomes("S49", r)[code_review.S49_ASSERTIONS[1]], stderr)

    def test_s50_passes_when_stderr_and_the_run_log_carry_the_refusal(self) -> None:
        # [MAGI §6 MS2 F loop 2] RV-S7: stderr AND the run log, naming batches, ceiling, floor and minimum.
        log = _BUDGET_LOG % RUN_ID.encode()
        r = _result("R15", 2, None, _with_id(_BUDGET_STDERR))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S50", r, log=log).values()))

    def test_s50_fails_without_a_minimum_or_a_ceiling_on_stderr(self) -> None:
        for text in (b"error: below the 15 s floor" + _NL,
                     b"error: 3 batches leave 10 s per batch, less than the 783 s a batch needs at the 15 s floor; "
                     b"pass --timeout 2349 or more" + _NL):
            r = _result("R15", 2, None, _with_id(text))
            self.assertEqual(Outcome.FAIL, _outcomes("S50", r, log=_BUDGET_LOG % RUN_ID.encode())[
                code_review.S50_ASSERTIONS[1]], text)

    def test_s50_fails_when_the_minimum_does_not_exceed_the_total_asked_for(self) -> None:
        text = _BUDGET_STDERR.replace(b"--timeout 342", b"--timeout %d" % runs.REVIEW_TOO_SHORT_TIMEOUT_S)
        r = _result("R15", 2, None, _with_id(text))
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]])

    def test_s50_passes_when_a_measured_window_plans_fewer_than_three_batches(self) -> None:
        # [MAGI S6 MS2 loop 5, F1-m1] the refusal fires at any batch count, and a pool that measured a
        # window larger than the unmeasured default plans fewer batches over the same tree: an
        # environmental reason, not a regression, so the range check asks only what the refusal
        # guarantees (one batch or more).
        text = _product_refusal(1, 2, False)
        log = b"ERROR run=" + RUN_ID.encode() + b" target=magi_rs::code_review " + text[len(b"error: "):]
        r = _result("R15", 2, None, _with_id(text))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S50", r, log=log).values()))

    def test_s50_fails_when_the_refusal_names_no_batch(self) -> None:
        text = _BUDGET_STDERR.replace(b"3 batches", b"0 batches")
        r = _result("R15", 2, None, _with_id(text))
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]])

    def test_s50_fails_when_the_per_batch_ceiling_reaches_the_floor_or_exceeds_the_wall_clock(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c3] the refusal shows the RAW ceiling of the per-batch wall clock, which a
        # refused batch never lifts to the floor and never exceeds: 15 s, 999 s or 7 s over 6 s is not the
        # product's arithmetic. (Loop 7 required the opposite, a ceiling of at least 15 s, which every real
        # refusal of a 20 s total fails.)
        for ceiling in (b"15", b"999", b"7"):
            text = _BUDGET_STDERR.replace(b"(per-batch ceiling 0 s)", b"(per-batch ceiling " + ceiling + b" s)")
            r = _result("R15", 2, None, _with_id(text))
            self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]], ceiling)

    def test_s50_passes_the_products_refusal_at_every_batch_count_and_config(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c3, F1a-m1] the bounds hold for whatever the product renders: one batch
        # or many (the count is printed as "N batches" at every N), rotation and retry on or off.
        for batches in range(1, 7):
            for rotations, retry_disabled in ((0, True), (0, False), (2, False), (4, False)):
                text = _product_refusal(batches, rotations, retry_disabled)
                match = code_review.BUDGET_PATTERN.search(text.decode())
                self.assertTrue(code_review._budget_fields_ok(match), text)

    def test_s50_fails_on_budget_numbers_the_product_cannot_print_together(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c3] the fixture knows the 20 s asked and the 24 s a batch needs at the
        # floor under the most lenient config: each number must agree with the others.
        incoherent = (
            (b"3 batches leave 6 s", b"3 batches leave 7 s"),   # the wall clock is not 20 // 3
            (b"the 114 s a batch", b"the 23 s a batch"),        # below the 24 s a batch needs at the least
            (b"--timeout 342", b"--timeout 343"),               # the minimum is not batches x need
            (b"--timeout 342", b"--timeout 21"),
        )
        for old, new in incoherent:
            text = _BUDGET_STDERR.replace(old, new)
            r = _result("R15", 2, None, _with_id(text))
            self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]], new)
        # Each of these breaks ONE bound and keeps every other number coherent, so each bound is needed.
        one_batch = _product_refusal(1, 2, False)
        alone = (
            _BUDGET_STDERR.replace(b"the 114 s a batch", b"the 23 s a batch").replace(
                b"--timeout 342", b"--timeout 69"),                      # need below 24 s, all else agreeing
            one_batch.replace(b"(per-batch ceiling 1 s)", b"(per-batch ceiling 15 s)"),  # ceiling at the floor
        )
        for text in alone:
            r = _result("R15", 2, None, _with_id(text))
            self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]], text)

    def test_s50_fails_when_the_floor_is_not_the_15_s_one(self) -> None:
        text = _BUDGET_STDERR.replace(b"the 15 s floor", b"the 20 s floor")
        r = _result("R15", 2, None, _with_id(text))
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]])

    def test_s50_fails_when_the_run_log_has_no_refusal_for_this_run(self) -> None:
        r = _result("R15", 2, None, _with_id(_BUDGET_STDERR))
        other = b"run=" + RUN_ID.encode() + b" ERROR something else" + _NL
        for log in (b"", _BUDGET_LOG % b"1-0000000000000000", other):
            self.assertEqual(Outcome.FAIL, _outcomes("S50", r, log=log)[code_review.S50_ASSERTIONS[2]], log)

    def test_s50_cannot_test_the_log_when_a_log_file_could_not_be_read(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-c2] a log file the executor could not open (held by the binary or an
        # antivirus on Windows) is the environment refusing, never a product FAIL; what was read still counts.
        r = dataclasses.replace(_result("R15", 2, None, _with_id(_BUDGET_STDERR)),
                                review_log_unreadable=("magi-2026-10-31.log (locked)",))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S50", r, log=b"")[code_review.S50_ASSERTIONS[2]])
        self.assertEqual(Outcome.PASS, _outcomes("S50", r, log=_BUDGET_LOG % RUN_ID.encode())[
            code_review.S50_ASSERTIONS[2]])

    def test_s50_fails_when_the_log_line_differs_from_stderr(self) -> None:
        r = _result("R15", 2, None, _with_id(_BUDGET_STDERR))
        log = (_BUDGET_LOG % RUN_ID.encode()).replace(b"342", b"9999")
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r, log=log)[code_review.S50_ASSERTIONS[2]])

    def test_s50_fails_the_log_without_a_run_id_or_a_log_directory(self) -> None:
        # [MAGI §6 MS2 loop 3, F1] R15 is deterministic and RV-S7 requires the run log: a missing run id
        # or log directory is a product failure, never an inconclusive CANNOT_TEST.
        r = _result("R15", 2, None, _BUDGET_STDERR)
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r, log=_BUDGET_LOG % RUN_ID.encode())[
            code_review.S50_ASSERTIONS[2]])
        r = _result("R15", 2, None, _with_id(_BUDGET_STDERR))
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r, log=None)[code_review.S50_ASSERTIONS[2]])

    def test_s47_fails_on_a_line_outside_the_generated_file(self) -> None:
        # [MAGI §6 MS2 F loop 2] RV09: an out-of-range line in the main list is a defect, not just a bad file.
        count = runs.review_gen_line_counts()["gen/a.rs"]
        ok = _result("R12", 0, _review(3, [{"file": "gen/a.rs", "line": count}]))
        self.assertEqual(Outcome.PASS, _outcomes("S47", ok)[code_review.S47_ASSERTIONS[4]])
        for bad in (count + 1, 0, -3, "7", True):
            r = _result("R12", 0, _review(3, [{"file": "gen/a.rs", "line": bad}]))
            self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]], bad)

    def test_s47_fails_on_a_findings_entry_that_is_not_an_object(self) -> None:
        r = _result("R12", 0, _review(3, [{"file": "gen/a.rs", "line": 1}, "gen/a.rs:1"]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_leaves_a_timed_out_run_to_the_runner(self) -> None:
        # The runner turns a timed-out run into CANNOT_TEST (runner.py) for every scenario that does not
        # declare inspects_timeouts; S47 must not declare it, or a slow trio would read as a FAIL.
        # [MAGI §6 MS2 loop 9, F1a-c2] only S47: S48-S51 need no backend, so a hang there is a product
        # defect they report themselves (test_a_hung_refusal_fails_every_assertion).
        self.assertFalse(DEFAULT_REGISTRY.get("S47").inspects_timeouts)

    def test_a_hung_refusal_fails_every_assertion(self) -> None:
        # [MAGI §6 MS2 loop 9, F1a-c2] R13-R16 are refused before any model is reached, so no slow provider
        # can explain a timeout: the runner must hand the hung run to the scenario, and every assertion is a
        # FAIL naming the hang. The executor builds a hung run with exit_code 0, so this also pins that no
        # assertion is judged on that synthetic exit.
        for sid, run_id, texts in (("S48", "R13", code_review.S48_ASSERTIONS),
                                   ("S49", "R14", code_review.S49_ASSERTIONS),
                                   ("S50", "R15", code_review.S50_ASSERTIONS),
                                   ("S51", "R16", code_review.S51_ASSERTIONS)):
            with self.subTest(scenario=sid):
                registry = Registry()
                registry.add(DEFAULT_REGISTRY.get(sid))
                hung = _result(run_id, 0, None, timed_out=True)
                findings = Runner(registry, {run_id: hung}, backend_reachable=False,
                                  ambient=Ambient(tree_snapshot=None, ceiling_fraction=None,
                                                  memory_settings={})).run()
                self.assertEqual([(text, Outcome.FAIL) for text in texts],
                                 [(f.assertion, f.outcome) for f in findings])
                self.assertTrue(all("timed out" in f.detail for f in findings), findings)

    def test_s47_never_judges_the_exit_of_a_timed_out_r12(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-c1] the executor builds a timed-out run with exit_code 0, so S47 asked
        # directly would PASS "exit 0" on a hang. The runner stops it first: S47 registered as it is, driven
        # through Runner with a timed-out R12, yields one CANNOT_TEST and S47's body never runs.
        entry = DEFAULT_REGISTRY.get("S47")
        registry = Registry()
        registry.add(entry)
        hung = _result("R12", 0, None, timed_out=True)
        findings = Runner(registry, {"R12": hung}, backend_reachable=True,
                          ambient=Ambient(tree_snapshot=None, ceiling_fraction=None, memory_settings={})).run()
        self.assertEqual([(runner_module._RUN_TIMED_OUT, Outcome.CANNOT_TEST)],
                         [(f.assertion, f.outcome) for f in findings])

    def test_s47_fails_on_missing_or_non_integer_batch_counts_even_with_every_model_measured(self) -> None:
        # [MAGI §6 MS2 loop 8, F1a-c1] a missing or non-integer batch count is a contract defect: FAIL, never the
        # CANNOT_TEST a fully measured pool earns for planning fewer batches.
        cases = {"both missing": None, "float planned": (2.0, 2), "bool counts": (True, True),
                 "string counts": ("2", "2")}
        for label, counts in cases.items():
            doc = _review(2)
            doc["code_review"]["unmeasured_models"] = []
            if counts is None:
                del doc["code_review"]["batches_planned"]
                del doc["code_review"]["batches_completed"]
            else:
                doc["code_review"]["batches_planned"], doc["code_review"]["batches_completed"] = counts
            with self.subTest(label):
                self.assertEqual(Outcome.FAIL,
                                 _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_fails_on_zero_or_negative_batch_counts_even_with_every_model_measured(self) -> None:
        # [MAGI §6 MS2 loop 9, F1a-c3] R12 reviews non-empty files, so zero batches is impossible however large
        # a measured window is, and a negative count is malformed: FAIL, never the fully measured CANNOT_TEST.
        for count in (0, -1, -3):
            doc = _review(count)
            doc["code_review"]["unmeasured_models"] = []
            with self.subTest(count=count):
                self.assertEqual(Outcome.FAIL,
                                 _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_with_every_model_measured_cannot_test_only_one_or_two_batches(self) -> None:
        # [MAGI §6 MS2 loop 9, F1a-c3] the environmental CANNOT_TEST covers exactly the counts a larger
        # measured window can produce: one or two batches.
        for count in (1, 2):
            doc = _review(count)
            doc["code_review"]["unmeasured_models"] = []
            with self.subTest(count=count):
                self.assertEqual(Outcome.CANNOT_TEST,
                                 _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_without_the_findings_key_fails_never_an_empty_pass(self) -> None:
        doc = _review(3)
        del doc["code_review"]["findings"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[4]])

    def test_s47_without_the_discarded_key_fails(self) -> None:
        # [CP2 MS2 S2b loop 10] symmetric to the findings key: an absent key is a shape drift, never "nothing discarded".
        doc = _review(3)
        del doc["code_review"]["discarded"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[5]])

    def test_s47_without_the_unmeasured_key_treats_the_pool_as_unmeasured(self) -> None:
        # [CP2 MS2 S2b loop 10] an absent key never softens the batch assertion to CANNOT_TEST.
        doc = _review(2)
        del doc["code_review"]["unmeasured_models"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_with_no_findings_cannot_test_the_location_property(self) -> None:
        r = _result("R12", 0, _review(3, []))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_fails_when_a_discarded_finding_has_no_cause(self) -> None:
        r = _result("R12", 0, _review(3, None, [{"cause": "line out of range"}, {"cause": ""}]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[5]])

    def test_s47_with_nothing_discarded_cannot_test_the_cause_property(self) -> None:
        r = _result("R12", 0, _review(3, None, []))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[5]])

    def test_s47_fails_on_a_nonzero_exit_unparsed_stdout_or_no_consensus(self) -> None:
        # [CP2 MS2 S2b loop 7] the negative half of each guarded S47 property.
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 1, _review(3)))[code_review.S47_ASSERTIONS[0]])
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, None))[code_review.S47_ASSERTIONS[1]])
        doc = _review(3)
        del doc["consult"]["consensus"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[3]])

    def test_s47_fails_on_a_json_document_that_is_not_an_object(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a2-m1] read as an AttributeError inside the scenario (a harness defect,
        # exit 3); ProductOutput.json() refuses a non-object with ProductOutputError, which S47 turns into a
        # FAIL of every assertion after the exit, so the scenario never reaches document.get.
        for document in ([1, 2], "text", 7):
            outcomes = _outcomes("S47", _result("R12", 0, document))
            for index in range(1, len(code_review.S47_ASSERTIONS)):
                self.assertEqual(Outcome.FAIL, outcomes[code_review.S47_ASSERTIONS[index]], document)

    def test_s47_fails_on_a_consensus_that_is_not_one_report_object(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c3] a truthy value is not a consensus: a list of two reports, a bare
        # string, or an object without magi-core's label and verdict all FAIL.
        report = _review(3)["consult"]["consensus"]
        for bad in ([report, report], "STRONG GO", {"verdict": "approve"},
                    {"consensus": "", "consensus_verdict": "approve"}):
            doc = _review(3)
            doc["consult"]["consensus"] = bad
            self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[3]],
                             bad)

    def test_s47_fails_when_consult_carries_the_consensus_key_twice(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c3] json.loads keeps the last duplicate silently; two reports under
        # one key are still two.
        body = json.dumps(_review(3)).encode()
        twice = body.replace(b'"consensus": {', b'"consensus": {"consensus": "GO"}, "consensus": {', 1)
        r = dataclasses.replace(_result("R12", 0, None),
                                output=ProductOutput(stdout=twice, stderr=b"", exit_code=0, command=["magi-rs"]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[3]])

    def test_s48_and_s49_fail_on_a_wrong_exit_or_a_silent_stderr(self) -> None:
        for sid, run, asserts in (("S48", "R13", code_review.S48_ASSERTIONS), ("S49", "R14", code_review.S49_ASSERTIONS)):
            self.assertEqual(Outcome.FAIL, _outcomes(sid, _result(run, 0, None, b""))[asserts[0]], sid)
            self.assertEqual(Outcome.FAIL, _outcomes(sid, _result(run, 2, None, b"error: something else\n"))[asserts[1]], sid)
