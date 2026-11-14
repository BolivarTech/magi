# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-13
"""S58 and R22: a review with seeded history and an unreachable embedder (RV07, RV-S8)."""

import dataclasses
import pathlib
import tempfile
import types
import unittest
import unittest.mock

from smoke import runs
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DECLARED_ASSERTION_COUNT, DECLARED_SCENARIO_COUNT, DEFAULT_REGISTRY
from smoke.scenarios import review_context as rc
from smoke.tests.support import scenario_outcomes as _outcomes, scenario_result as _result

P, F, C = Outcome.PASS, Outcome.FAIL, Outcome.CANNOT_TEST
_with = dataclasses.replace

#: Located finding of the first setup review: what the history is seeded from.
_FINDING = {"file": runs.REVIEW_HISTORY_FILE, "line": 3, "title": "overflow"}


def _out(code: int = 0, document=None) -> ProductOutput:
    """A product output whose stdout is *document* serialised."""
    import json
    body = b"" if document is None else json.dumps(document).encode()
    return ProductOutput(stdout=body, stderr=b"", exit_code=code, command=["magi-rs"])


def _degraded(**extra) -> dict:
    """A measured review's JSON that declares the embedder down for history and docs."""
    summary = {"without_embedder": True, "findings": [],
               "trimmed": [{"kind": "documentation", "label": "the embedder did not answer"},
                           {"kind": "history", "label": "the embedder did not answer"}]}
    summary.update(extra)
    return {"code_review": summary}


def _run(measured, setup_codes=(0, 0)):
    """An R22 result: *measured* is the document the third review printed."""
    base = _result("R22", 0, measured)
    return _with(base, setup_outputs=tuple(_out(c, {"code_review": {"findings": [_FINDING]}})
                                           for c in setup_codes))


def _judged(result) -> list:
    outcomes = _outcomes("S58", result)
    return [outcomes[a] for a in rc.S58_ASSERTIONS]


class DegradedHistoryScenarioTests(unittest.TestCase):

    def test_the_scenario_is_registered_and_counted(self) -> None:
        self.assertEqual(3, len(rc.S58_ASSERTIONS))
        self.assertIn("S58", DEFAULT_REGISTRY.registered_ids())
        self.assertEqual(58, DECLARED_SCENARIO_COUNT)
        self.assertEqual(174, DECLARED_ASSERTION_COUNT)

    def test_a_degraded_review_that_reports_passes(self) -> None:
        self.assertEqual([P, P, P], _judged(_run(_degraded())))

    def test_the_embedder_must_be_declared_down(self) -> None:
        for doc in (_degraded(without_embedder=False), _degraded(without_embedder="yes"),
                    {"code_review": {"findings": [], "trimmed": _degraded()["code_review"]["trimmed"]}}):
            with self.subTest(doc=doc):
                self.assertEqual([P, F, P], _judged(_run(doc)))

    def test_the_history_layer_must_name_the_degraded_cause(self) -> None:
        for trimmed in ([], [{"kind": "documentation", "label": "the embedder did not answer"}],
                        [{"kind": "history", "label": "no room left in the batch"}], "x"):
            with self.subTest(trimmed=trimmed):
                self.assertEqual([P, F, P], _judged(_run(_degraded(trimmed=trimmed))))

    def test_a_review_that_delivers_no_findings_list_fails(self) -> None:
        for findings in ("none", None):
            with self.subTest(findings=findings):
                self.assertEqual([P, P, F], _judged(_run(_degraded(findings=findings))))

    def test_a_setup_or_measured_review_that_exits_nonzero_fails_the_first_assertion(self) -> None:
        self.assertEqual(F, _judged(_run(_degraded(), setup_codes=(0, 1)))[0])
        bad = _with(_run(_degraded()), output=_out(2, _degraded()))
        self.assertEqual(F, _judged(bad)[0])

    def test_a_missing_capture_or_setup_is_cannot_test(self) -> None:
        self.assertEqual([C, C, C], _judged(_with(_run(_degraded()), setup_outputs=())))

    def test_no_seeded_finding_leaves_the_history_declaration_cannot_test(self) -> None:
        # MAGI FIX (MS3 §6 loop17, R7a1 Caspar): with no located finding in the first review
        # there is no history to be unretrievable; that is the trio's answer, not a product FAIL.
        for findings in ([], [{"file": "other.rs", "line": 3, "title": "x"}],
                         [{"file": runs.REVIEW_HISTORY_FILE, "line": 99, "title": "x"}]):
            with self.subTest(findings=findings):
                setup = tuple(_out(0, {"code_review": {"findings": findings}}) for _ in range(2))
                for measured in (_degraded(), _degraded(trimmed=[])):
                    self.assertEqual([P, C, P], _judged(_with(_run(measured), setup_outputs=setup)))

    def test_a_stdout_that_is_not_json_fails_the_declarations(self) -> None:
        bad = _with(_run(_degraded()), output=_out(0, None))
        self.assertEqual([P, F, F], _judged(bad))


class DegradedHistoryRunTests(unittest.TestCase):

    def test_r22_seeds_history_then_runs_with_the_embedder_away(self) -> None:
        r22, r21 = runs.DEFINITIONS["R22"], runs.DEFINITIONS["R21"]
        self.assertTrue(r22.unreachable_embedder)
        self.assertFalse(r22.records_requests)
        self.assertFalse(r22.forbids_model_calls)
        self.assertEqual(r21.setup, r22.setup, "the same seeding as R21's")
        self.assertEqual(r21.argv, r22.argv)
        self.assertTrue(r22.needs_trio)

    def test_a_setup_may_run_before_an_embedder_that_is_pointed_away(self) -> None:
        # The setup is how the history is seeded, against the live embedder; only the measured
        # run is pointed away. The other rewrites stay refused beside a setup.
        base = dict(run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS,
                    stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL,
                    timeout_s=5 + runs.MIN_MEASURED_ALLOWANCE_S,
                    setup=(runs.Invocation(argv=("code-review",), timeout_s=5),))
        runs.RunDefinition(**base, unreachable_embedder=True)

    def test_the_setup_sees_the_live_embedder_and_the_measured_run_the_away_one(self) -> None:
        from smoke.tests.test_review_runs import _executor, _git_available
        if not _git_available():
            self.skipTest("git is required")
        definition = runs.RunDefinition(
            run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS,
            stdin=b"", needs_trio=False, payload_size=runs.PAYLOAD_SMALL,
            timeout_s=5 + runs.MIN_MEASURED_ALLOWANCE_S, unreachable_embedder=True,
            setup=(runs.Invocation(argv=("code-review",), timeout_s=5),))
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            (root / ".magi" / "magi.toml").write_text(
                'base_url = "http://real/v1"\n[embedding]\nbase_url = "http://emb/v1"\nmodel = "m"\n',
                encoding="utf-8")
            seen = []

            def spy(definition, stdin):
                workspace = root / "scratch" / runs.REVIEW_WORKSPACE_DIR
                text = workspace.joinpath(*runs.CONFIG_PATH_PARTS).read_text(encoding="utf-8")
                seen.append("http://emb/v1" in text)
                return ProductOutput(stdout=b"{}", stderr=b"", exit_code=0, command=())
            with unittest.mock.patch.object(executor, "_invoke", spy):
                executor.execute(definition)
        self.assertEqual([True, False], seen, "setup against the live embedder, then the away one")


if __name__ == "__main__":
    unittest.main()
