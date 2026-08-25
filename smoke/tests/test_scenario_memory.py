# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for S9: the derivation, and what a missing field must do."""

import json
import pathlib
import unittest

from smoke import runs
from smoke.errors import HarnessError
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runner import Ambient
from smoke.runs import RunResult
from smoke.scenarios import memory  # noqa: F401 - import registers it
from smoke.tests import support

#: A minimal environment config the scenario can copy and patch.
_ENV_CONFIG = (
    "provider = \"ollama\"\n"
    "base_url = \"http://localhost:11434/v1\"\n"
    "\n"
    "[embedding]\n"
    "# base_url = \"http://localhost:11434/v1\"\n"
    "model = \"nomic-embed-text\"\n"
)


def _seed_env_config() -> None:
    """Give the fake environment the ``magi.toml`` the scenario copies.

    The embedder probe reuses the environment's own configuration so it runs
    against whatever models the run is already using; without a file to copy
    the scenario has nothing to point at a closed port.
    """
    config_dir = runs.workspace_root() / memory.MAGI_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / memory.MAGI_TOML_NAME).write_text(_ENV_CONFIG,
                                                    encoding="utf-8")

#: A ``[memory]`` block declaring all three fields the ceiling is derived from.
_COMPLETE_SETTINGS = {
    "context_budget_tokens": 8000,
    "response_headroom_tokens": 1024,
    "safety_margin_ratio": 0.1,
}

#: The startup line a healthy run prints, with memories present and nothing
#: waiting to be embedded.
_HEALTHY_LINE = (
    b"note: memory: 12 active, 0 archived, 0 pending re-embed (~34 KB index)\n"
)


def _capture(document, stderr=b"", exit_code=0) -> ProductOutput:
    """Build a capture from an object and a stderr stream.

    Args:
        document: What the product printed to stdout, serialised here.
        stderr: What it wrote to the error stream.
        exit_code: What it exited with.

    Returns:
        ProductOutput: The capture.
    """
    return ProductOutput(stdout=json.dumps(document).encode(), stderr=stderr,
                         exit_code=exit_code, command=["magi-rs", "query"])


def _run(run_id, input_tokens, stderr=b"") -> RunResult:
    """Build one of S9's three results.

    Args:
        run_id: Which definition it stands for.
        input_tokens: What ``usage.input_tokens`` should report.
        stderr: The startup lines the product printed.

    Returns:
        RunResult: The real type, not a double.
    """
    document = {"usage": {"input_tokens": input_tokens, "output_tokens": 5}}
    return RunResult(run_id=run_id, output=_capture(document, stderr=stderr),
                     duration_s=1.0, timed_out=False, planted=())


def _runs(r3_tokens=9000, r2_tokens=1000, r3_stderr=_HEALTHY_LINE):
    """The mapping a scenario declaring three runs receives.

    Args:
        r3_tokens: ``usage.input_tokens`` for R3.
        r2_tokens: The same for R2, the ``--no-memory`` control.
        r3_stderr: R3's startup lines.

    Returns:
        dict[str, RunResult]: Keyed by run id, as the runner hands it over.
    """
    return {"R1": _run("R1", 800), "R2": _run("R2", r2_tokens),
            "R3": _run("R3", r3_tokens, stderr=r3_stderr)}


def _ambient(settings=None, margin=100, fraction=0.8) -> Ambient:
    """Build the ambient state S9 reads.

    Args:
        settings: The environment's ``[memory]`` block.
        margin: The configured margin in tokens.
        fraction: The configured saturation fraction.

    Returns:
        Ambient: The state, with no tree snapshot -- S9 never reads one.
    """
    return Ambient(
        tree_snapshot=None,
        margin_tokens=margin,
        ceiling_fraction=fraction,
        memory_settings=dict(_COMPLETE_SETTINGS if settings is None
                             else settings),
    )


def _outcomes(runs_map, ambient) -> dict[str, Outcome]:
    """Run S9 and index what it concluded by assertion text.

    Args:
        runs_map: The three results.
        ambient: The ambient state.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S9").func(runs_map, ambient))
    return {finding.assertion: finding.outcome for finding in findings}


class MemoryScenarioShapeTests(unittest.TestCase):
    """S9 hangs off three runs and reads state no run produced."""

    def test_s9_is_registered_with_its_declared_runs(self) -> None:
        entry = DEFAULT_REGISTRY.get("S9")
        self.assertEqual(("R1", "R2", "R3"), entry.run)
        self.assertTrue(entry.needs_backend)
        self.assertTrue(entry.needs_ambient)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "the startup line reports N active with N > 0",
                "pending re-embed is 0 — the embedder answered",
                "usage.input_tokens of R3 exceeds R2's by more than the "
                "declared margin",
                "the environment is below the saturation ceiling — otherwise "
                "3 degrades to CANNOT_TEST",
                "with the embedder down, the run completes with a degradation "
                "notice",
            ],
            list(memory.ASSERTIONS),
        )


class SaturationCeilingTests(unittest.TestCase):
    """The ceiling is DERIVED, over a fixed table of inputs."""

    def test_the_derivation_follows_the_declared_formula(self) -> None:
        """(budget - headroom) * (1 - ratio), to the digit."""
        for settings, expected in (
            ({"context_budget_tokens": 8000, "response_headroom_tokens": 1024,
              "safety_margin_ratio": 0.1}, (8000 - 1024) * 0.9),
            ({"context_budget_tokens": 4000, "response_headroom_tokens": 0,
              "safety_margin_ratio": 0.0}, 4000.0),
            ({"context_budget_tokens": 1000, "response_headroom_tokens": 500,
              "safety_margin_ratio": 0.5}, 250.0),
        ):
            with self.subTest(settings=settings):
                self.assertAlmostEqual(expected,
                                       memory.usable_budget(settings))

    def test_a_missing_field_refuses_to_derive(self) -> None:
        """Absent is not zero.

        Filling in the product's own default would be a second source of truth,
        and the copy is always the one that forgets to be updated.
        """
        for absent in memory.CEILING_FIELDS:
            with self.subTest(absent=absent):
                partial = {key: value
                           for key, value in _COMPLETE_SETTINGS.items()
                           if key != absent}
                self.assertIsNone(memory.usable_budget(partial))

    def test_an_empty_block_refuses_to_derive(self) -> None:
        self.assertIsNone(memory.usable_budget({}))

    def test_a_non_numeric_field_refuses_to_derive(self) -> None:
        settings = dict(_COMPLETE_SETTINGS, context_budget_tokens="lots")
        self.assertIsNone(memory.usable_budget(settings))


class MemoryScenarioBodyTests(unittest.TestCase):
    """Injection is measured by DIFFERENCE, and only below the ceiling."""

    def setUp(self) -> None:
        support.install_fake_runs(self)

    def test_a_healthy_environment_passes_the_first_four(self) -> None:
        outcomes = _outcomes(_runs(), _ambient())
        for text in memory.ASSERTIONS[:4]:
            self.assertEqual(Outcome.PASS, outcomes[text], text)

    def test_no_active_memories_fails_the_first(self) -> None:
        line = b"note: memory: 0 active, 0 archived, 0 pending re-embed (~0 KB index)\n"
        outcomes = _outcomes(_runs(r3_stderr=line), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[0]])

    def test_pending_re_embeds_fail_the_second(self) -> None:
        line = b"note: memory: 12 active, 0 archived, 3 pending re-embed (~34 KB index)\n"
        outcomes = _outcomes(_runs(r3_stderr=line), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[1]])

    def test_a_missing_startup_line_cannot_test_the_first_two(self) -> None:
        outcomes = _outcomes(_runs(r3_stderr=b"note: something else\n"),
                             _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[1]])

    def test_a_difference_below_the_margin_fails_the_third(self) -> None:
        outcomes = _outcomes(_runs(r3_tokens=1050, r2_tokens=1000),
                             _ambient(margin=100))
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[2]])

    def test_an_uncalibrated_margin_cannot_test_the_third(self) -> None:
        """A margin of zero cannot tell "it injected" from "it did not".

        Zero is what ``smoke.toml`` carries until phase 5 measures the real
        number, and R2 and R3 carry different prompts, so almost any pair of
        token counts clears it. Reporting PASS there is a green that means
        nothing.
        """
        outcomes = _outcomes(_runs(), _ambient(margin=0))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[2]])

    def test_a_saturated_environment_cannot_test_the_third(self) -> None:
        """The false green assertion 3b exists to stop.

        Once the assembler saturates its budget, the R3 minus R2 difference
        stops measuring "it injected what we planted" and starts measuring "the
        budget is full" -- and it still clears the margin, so assertion 3 would
        pass in exactly the same way while measuring something else.
        """
        saturated = (8000 - 1024) * 0.9 * 0.8
        outcomes = _outcomes(
            _runs(r3_tokens=1000 + int(saturated) + 1, r2_tokens=1000),
            _ambient(margin=100, fraction=0.8))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[3]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[2]])

    def test_an_undeclared_ceiling_cannot_test_the_fourth(self) -> None:
        """An EMPTY block is not a block full of zeros."""
        outcomes = _outcomes(_runs(), _ambient(settings={}))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[3]])

    def test_an_uncalibrated_fraction_cannot_test_the_fourth(self) -> None:
        outcomes = _outcomes(_runs(), _ambient(fraction=0.0))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[3]])

    def test_a_missing_run_reports_all_five(self) -> None:
        outcomes = _outcomes({}, _ambient())
        self.assertEqual(list(memory.ASSERTIONS), list(outcomes))
        self.assertNotIn(Outcome.PASS, set(list(outcomes.values())[:4]))


class EmbedderDownTests(unittest.TestCase):
    """Assertion 4 is a one-off invocation, not a ninth shared run."""

    def test_the_endpoint_it_points_at_is_a_closed_local_port(self) -> None:
        self.assertTrue(memory.DEAD_EMBEDDING_ENDPOINT.startswith(
            "http://127.0.0.1:"))

    def test_the_override_lands_inside_the_embedding_table(self) -> None:
        """A line-level edit, checked rather than trusted.

        The standard library has no TOML writer, so the override is inserted
        after the section header. Landing it in the wrong table would point the
        MAIN provider at the closed port, and the run would fail outright
        instead of degrading -- a red that looks like the assertion working.
        """
        patched = memory.point_embedding_at(
            "provider = \"ollama\"\n"
            "base_url = \"http://localhost:11434/v1\"\n"
            "\n"
            "[embedding]\n"
            "# base_url = \"http://localhost:11434/v1\"\n"
            "model = \"nomic-embed-text\"\n"
        )
        lines = patched.splitlines()
        header = lines.index("[embedding]")
        self.assertEqual("base_url = \"%s\"" % memory.DEAD_EMBEDDING_ENDPOINT,
                         lines[header + 1])
        self.assertEqual("base_url = \"http://localhost:11434/v1\"", lines[1])

    def test_a_config_without_the_table_is_refused(self) -> None:
        with self.assertRaises(HarnessError):
            memory.point_embedding_at("provider = \"ollama\"\n")

    def test_a_run_that_completes_with_the_notice_passes(self) -> None:
        support.install_fake_runs(self, responder=_FakeDegraded(notice=True))
        _seed_env_config()
        outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.PASS, outcomes[memory.ASSERTIONS[4]])

    def test_a_run_that_completes_without_a_notice_fails(self) -> None:
        """REQ-29 has two halves and this is the one nothing else covers.

        An operator whose embedder is down and who is told nothing watches
        memories pile up unembedded with no signal that anything degraded.
        """
        support.install_fake_runs(self, responder=_FakeDegraded(notice=False))
        _seed_env_config()
        outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[4]])

    def test_a_run_the_embedder_took_down_fails(self) -> None:
        support.install_fake_runs(
            self, responder=_FakeDegraded(notice=True, exit_code=1))
        _seed_env_config()
        outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[4]])

    def test_the_probe_leaves_no_workspace_behind(self) -> None:
        """The throwaway workspace is removed in ``finally``.

        It is built outside the repository so it cannot reach ``git status``,
        but a harness that leaks a directory per run leaks a database per run
        with it, and nothing else in the suite would ever say so.
        """
        binary = support.install_fake_runs(self,
                                           responder=_FakeDegraded(notice=True))
        _seed_env_config()
        _outcomes(_runs(), _ambient())
        scaffolded = [call.cwd for call in binary.calls
                      if list(call.args)[:1] == [memory.INIT_SUBCOMMAND]]
        self.assertEqual(1, len(scaffolded))
        self.assertFalse(pathlib.Path(scaffolded[0]).exists(), scaffolded[0])

    def test_a_scaffold_that_failed_cannot_test(self) -> None:
        support.install_fake_runs(self)
        _seed_env_config()
        outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[4]])

    def test_a_missing_environment_config_cannot_test(self) -> None:
        support.install_fake_runs(self, responder=_FakeDegraded(notice=True))
        outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[4]])


class _FakeDegraded:
    """A product double that scaffolds, then answers the degraded query.

    Attributes:
        notice: Whether the query reports the degradation on stderr.
        exit_code: What the query exits with.
    """

    def __init__(self, notice: bool, exit_code: int = 0) -> None:
        """Create the double.

        Args:
            notice: Whether to print the degradation notice.
            exit_code: The query's exit code.
        """
        self.notice = notice
        self.exit_code = exit_code

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation, scaffolding a real tree for ``init``.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if args[:1] == [memory.INIT_SUBCOMMAND]:
            root = pathlib.Path(call.cwd)
            (root / memory.MAGI_DIR_NAME).mkdir(parents=True, exist_ok=True)
            (root / memory.MAGI_DIR_NAME / memory.MAGI_TOML_NAME).write_text(
                "[embedding]\nmodel = \"x\"\n", encoding="utf-8")
            return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                                 command=["magi-rs", memory.INIT_SUBCOMMAND])
        if args[:1] == ["query"]:
            stderr = (b"note: " + memory.DEGRADATION_MARKER.encode() + b"\n"
                      if self.notice else b"note: nothing to report\n")
            return _capture({"response": "ok"}, stderr=stderr,
                            exit_code=self.exit_code)
        return None


if __name__ == "__main__":
    unittest.main()
