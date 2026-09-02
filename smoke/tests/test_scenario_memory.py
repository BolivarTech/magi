# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-09-02
"""Unit tests for S9: the derivation, and what a missing field must do."""

import json
import pathlib
import tomllib
import unittest
from unittest import mock
import urllib.error
import urllib.request

from smoke import logs, runs
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


#: A stand-in endpoint for the tests that only check WHERE the override lands.
#: The ones that check it is reachable open a real one instead.
_PROBE_ENDPOINT = "http://127.0.0.1:65000/v1"

#: A ``[memory]`` block declaring all three fields the ceiling is derived from.
_COMPLETE_SETTINGS = {
    "context_budget_tokens": 8000,
    "response_headroom_tokens": 1024,
    "safety_margin_ratio": 0.1,
}

#: The run id R3's fixture publishes on stderr, so the log correlation
#: `_startup_counts` performs has something to match against. Since the
#: screen policy the counts no longer live on R3's own capture at all -- they
#: are read out of the shared, persistent log directory, keyed by this id.
_R3_RUN_ID = "424242-deadbeefcafef00d"

#: The startup line body a healthy run's log entry carries, with memories
#: present and nothing waiting to be embedded. Just the message half of a
#: rendered event -- :func:`_write_r3_log` supplies the header.
_HEALTHY_BODY = "memory: 12 active, 0 archived, 0 pending re-embed (~34 KB index)"


def _write_r3_log(body, run_id=_R3_RUN_ID):
    """Write one log line into the fake environment's log directory.

    Shaped like ``render.rs::header_of`` renders a real event: a timestamp,
    level, ``run=<id>`` and a target ahead of the message, which is exactly
    what :func:`smoke.scenarios.memory._startup_line_matcher` requires to
    correlate a line to one run.

    Args:
        body: The message half of the line, e.g. the ``memory: ...`` text.
        run_id: The id to stamp the line with.
    """
    directory = runs.workspace_root() / ".magi" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    line = "2026-08-31T00:00:00Z INFO run=%s magi_rs::main: %s\n" % (
        run_id, body)
    (directory / "2026-08-31.log").write_bytes(line.encode("utf-8"))


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


def _run(run_id, input_tokens, stderr=b"", turns=1) -> RunResult:
    """Build one of S9's three results.

    Args:
        run_id: Which definition it stands for.
        input_tokens: What ``usage.input_tokens`` should report.
        stderr: The startup lines the product printed.
        turns: How many messages the transcript carries. This is what
            assertion 3 reads: a run with memory carries the turns the
            assembler loaded on top of its own.

    Returns:
        RunResult: The real type, not a double.
    """
    document = {"usage": {"input_tokens": input_tokens, "output_tokens": 5},
                "transcript": [{"role": "user", "content": "x"}] * turns}
    return RunResult(run_id=run_id, output=_capture(document, stderr=stderr),
                     duration_s=1.0, timed_out=False, planted=())


def _runs(r3_tokens=4000, r2_tokens=1000, r3_body=_HEALTHY_BODY,
          r3_run_id=_R3_RUN_ID, write_r3_log=True, r3_turns=40, r2_turns=4):
    """The mapping a scenario declaring three runs receives.

    Args:
        r3_tokens: ``usage.input_tokens`` for R3.
        r2_tokens: The same for R2, the ``--no-memory`` control.
        r3_body: The message half of the log line R3's run leaves behind, or
            None to publish a run id with no matching line at all -- the
            "searched clean" case.
        r3_run_id: The id R3 publishes on stderr and the log line is stamped
            with. None publishes no id, so correlation itself fails.
        write_r3_log: Whether the fake ``.magi/logs`` directory is created at
            all -- False is the "nothing was written to disk" case, which is
            a different finding from an existing directory searched clean.
        r3_turns: Transcript length for R3, the run with memory.
        r2_turns: Transcript length for R2, the control.

    Returns:
        dict[str, RunResult]: Keyed by run id, as the runner hands it over.
    """
    if write_r3_log:
        if r3_body is not None:
            _write_r3_log(r3_body, run_id=r3_run_id)
        else:
            # A directory that exists but carries nothing matching is a
            # different finding from one that was never written at all.
            (runs.workspace_root() / ".magi" / "logs").mkdir(
                parents=True, exist_ok=True)
    r3_stderr = (("run: %s\n" % r3_run_id).encode("utf-8")
                if r3_run_id else b"")
    return {"R1": _run("R1", 800),
            "R2": _run("R2", r2_tokens, turns=r2_turns),
            "R3": _run("R3", r3_tokens, stderr=r3_stderr, turns=r3_turns)}


def _ambient(settings=None, fraction=0.8) -> Ambient:
    """Build the ambient state S9 reads.

    Args:
        settings: The environment's ``[memory]`` block.
        fraction: The configured saturation fraction.

    Returns:
        Ambient: The state, with no tree snapshot -- S9 never reads one.
    """
    return Ambient(
        tree_snapshot=None,
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


def _details(runs_map, ambient) -> dict[str, str]:
    """Run S9 and index the CAUSE it gave, by assertion text.

    The outcome alone cannot tell two ``CANNOT_TEST`` branches apart, and S9
    has two that a reader has to act on differently: nothing was written to
    disk at all, versus a directory that exists and holds no matching line.
    A test that reads only the outcome passes for either, so the pair it
    claims to distinguish is not actually distinguished anywhere.

    Args:
        runs_map: The three results.
        ambient: The ambient state.

    Returns:
        dict[str, str]: What each assertion gave as its cause.
    """
    findings = list(DEFAULT_REGISTRY.get("S9").func(runs_map, ambient))
    return {finding.assertion: finding.detail for finding in findings}


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
                "R3's transcript carries turns R2's does not — the "
                "assembler loaded them",
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
        body = "memory: 0 active, 0 archived, 0 pending re-embed (~0 KB index)"
        outcomes = _outcomes(_runs(r3_body=body), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[0]])

    def test_pending_re_embeds_fail_the_second(self) -> None:
        body = "memory: 12 active, 0 archived, 3 pending re-embed (~34 KB index)"
        outcomes = _outcomes(_runs(r3_body=body), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[1]])

    def test_a_missing_startup_line_cannot_test_the_first_two(self) -> None:
        """The log directory exists, but nothing under it matches R3's id."""
        runs_map = _runs(r3_body="note: something else")
        outcomes = _outcomes(runs_map, _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[1]])
        detail = _details(_runs(r3_body="note: something else"),
                          _ambient())[memory.ASSERTIONS[0]]
        self.assertIn("no line under", detail)
        self.assertNotIn("does not exist", detail)

    def test_a_missing_log_directory_cannot_test_the_first_two(self) -> None:
        """Nothing written to disk at all is a DIFFERENT finding from an
        existing directory searched clean -- the cause string has to say
        which one this run hit.

        The outcome is the same ``CANNOT_TEST`` in both cases, so asserting
        on it alone made this test and its sibling above indistinguishable:
        each passed for the other's condition and neither checked the
        distinction the name promises. The cause is where that distinction
        lives, so the cause is what is read.
        """
        outcomes = _outcomes(_runs(write_r3_log=False), _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[1]])
        detail = _details(_runs(write_r3_log=False),
                          _ambient())[memory.ASSERTIONS[0]]
        self.assertIn("does not exist", detail)
        self.assertIn("wrote no log at all", detail)

    def test_an_unreadable_sibling_never_discards_a_line_that_was_found(
            self) -> None:
        """A found line is the answer, whatever else in the directory would
        not open.

        ``.magi/logs`` is shared and persistent, so an old rotated file that
        cannot be read is an ordinary condition -- and it has nothing to do
        with the line this run wrote and this scenario already located. S23
        and S24 both honour a hit before they look at the unreadable list;
        S9 checked the list first and threw the evidence away, reporting
        "cannot test" while holding the answer.
        """
        real_scan = logs.scan

        def scan_reporting_an_unreadable_sibling(matcher):
            result, dir_existed, _ = real_scan(matcher)
            return result, dir_existed, ["2026-01-01.log (Permission denied)"]

        with mock.patch.object(logs, "scan",
                               scan_reporting_an_unreadable_sibling):
            outcomes = _outcomes(_runs(), _ambient())
        self.assertEqual(Outcome.PASS, outcomes[memory.ASSERTIONS[0]])
        self.assertEqual(Outcome.PASS, outcomes[memory.ASSERTIONS[1]])

    def test_no_published_run_id_cannot_test_the_first_two(self) -> None:
        """A run that names no id of its own cannot be told apart from any
        other run's line in the shared, persistent log directory.
        """
        outcomes = _outcomes(_runs(r3_run_id=None), _ambient())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[1]])

    def test_a_transcript_no_longer_than_the_control_fails_the_third(self) -> None:
        """Memory loaded nothing, and that is the regression to catch."""
        outcomes = _outcomes(_runs(r3_turns=4, r2_turns=4), _ambient())
        self.assertEqual(Outcome.FAIL, outcomes[memory.ASSERTIONS[2]])

    def test_a_longer_transcript_passes_the_third(self) -> None:
        outcomes = _outcomes(_runs(r3_turns=40, r2_turns=4), _ambient())
        self.assertEqual(Outcome.PASS, outcomes[memory.ASSERTIONS[2]])

    def test_a_missing_transcript_cannot_test_the_third(self) -> None:
        """Absent is not empty. A capture with no transcript says nothing
        about what the assembler did.
        """
        results = _runs()
        broken = json.loads(results["R3"].output.stdout)
        broken.pop("transcript")
        results["R3"] = RunResult(
            run_id="R3", output=_capture(broken, stderr=b""),
            duration_s=1.0, timed_out=False, planted=())
        self.assertEqual(Outcome.CANNOT_TEST,
                         _outcomes(results, _ambient())[memory.ASSERTIONS[2]])

    def test_a_saturated_environment_cannot_test_the_third(self) -> None:
        """The false green assertion 3b exists to stop.

        Once the assembler saturates its budget it loads bulk history rather
        than what the query asked for, so a longer transcript stops meaning
        "it recalled what we planted" -- and it is longer in exactly the same
        way, so assertion 3 would pass while measuring something else.
        """
        saturated = (8000 - 1024) * 0.9 * 0.8
        outcomes = _outcomes(
            _runs(r3_tokens=1000 + int(saturated) + 1, r2_tokens=1000),
            _ambient(fraction=0.8))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[3]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[memory.ASSERTIONS[2]])

    def test_a_whole_number_fraction_is_still_a_calibration(self) -> None:
        """TOML ``ceiling_fraction = 1`` parses as an int.

        The check asked for a float, so a calibrated 1 reported "no ceiling
        fraction is calibrated" -- a message that is false about the file in
        front of it, and 3b and 3 both degrade over a setting the operator did
        write.
        """
        outcomes = _outcomes(_runs(), _ambient(fraction=1))
        self.assertEqual(Outcome.PASS, outcomes[memory.ASSERTIONS[3]])

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

    def test_the_override_lands_inside_the_embedding_table(self) -> None:
        """A line-level edit, checked rather than trusted.

        The standard library has no TOML writer, so the override is inserted
        after the section header. Landing it in the wrong table would point the
        MAIN provider at the failing endpoint, and the run would fail outright
        instead of degrading -- a red that looks like the assertion working.
        """
        patched = memory.point_embedding_at(
            _PROBE_ENDPOINT,
            "provider = \"ollama\"\n"
            "base_url = \"http://localhost:11434/v1\"\n"
            "\n"
            "[embedding]\n"
            "# base_url = \"http://localhost:11434/v1\"\n"
            "model = \"nomic-embed-text\"\n"
        )
        lines = patched.splitlines()
        header = lines.index("[embedding]")
        self.assertEqual("base_url = \"%s\"" % _PROBE_ENDPOINT,
                         lines[header + 1])
        self.assertEqual("base_url = \"http://localhost:11434/v1\"", lines[1])

    def test_a_config_without_the_table_is_refused(self) -> None:
        with self.assertRaises(HarnessError):
            memory.point_embedding_at(_PROBE_ENDPOINT,
                                      "provider = \"ollama\"\n")

    def test_the_probe_declares_no_fixed_port(self) -> None:
        """The same defect as R6's, in a second copy of the same comment.

        The module declared ``127.0.0.1:9`` as "the discard service, reserved
        and unused". Where that service is RUNNING it accepts the connection
        and never answers, so the probe waited out its entire ceiling and
        assertion 4 reported CANNOT_TEST instead of exercising REQ-29 at all.
        A literal port reads perfectly well, which is why this is checked
        against the source rather than left to a reviewer's eye.
        """
        text = pathlib.Path(memory.__file__).read_text(encoding="utf-8")
        self.assertNotIn("127.0.0.1:9", text)

    def test_the_probe_points_the_embedder_at_something_that_answers(self) -> None:
        """End to end through the double: the URL the scenario wrote into the
        probe's own configuration is read back and connected to. Answering is
        the whole difference between exercising REQ-29 and waiting.
        """
        responder = _AnsweringEmbedder()
        support.install_fake_runs(self, responder=responder)
        _seed_env_config()
        _outcomes(_runs(), _ambient())
        self.assertEqual(runs.ERROR_BACKEND_STATUS, responder.status)

    def test_the_marker_is_a_phrase_the_product_actually_emits(self) -> None:
        """The guardian that could never pass.

        ``text-only persistence`` is emitted on exactly one path: the embedder
        CLIENT failing to construct, which happens for a malformed URL or an
        unresolvable vault entry. It never happens for an endpoint that is
        unreachable or that answers with an error, because the client is built
        lazily and constructs fine either way. The probe creates the second
        kind, so assertion 4 matched a string the run it performs cannot
        produce, and hid behind CANNOT_TEST for as long as the probe also hung.

        The path the probe does reach is the assembler's, which announces
        itself differently. Checked against the product's source, because the
        failure is a string that reads perfectly well on its own.
        """
        root = pathlib.Path(__file__).resolve().parent.parent.parent
        source = (root / "src" / "agent" / "mod.rs").read_text(
            encoding="utf-8", errors="replace")
        self.assertTrue(
            memory.DEGRADATION_MARKER in source,
            "%r appears nowhere in src/agent/mod.rs, so no run can ever "
            "produce it" % memory.DEGRADATION_MARKER)

    def test_the_probe_plants_a_memory_before_it_breaks_the_embedder(self) -> None:
        """The fixture could not trip the condition either.

        The throwaway workspace starts with no memories at all, so the recall
        path never asks the embedder for anything and the degradation never
        happens. The probe has to put something in the store first, with the
        embedder still working, and only then break it.
        """
        binary = support.install_fake_runs(self, responder=_FakeDegraded(notice=True))
        _seed_env_config()
        _outcomes(_runs(), _ambient())
        queries = [call for call in binary.calls
                   if list(call.args[:1]) == ["query"]]
        self.assertGreaterEqual(len(queries), 2)

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
            # The planting half must succeed whatever this double stands in
            # for: exit_code describes the run under test, which is the
            # RECALL, and failing the plant instead only ever yields
            # CANNOT_TEST over a store that was never filled.
            if call.stdin == memory.PLANT_PROMPT:
                return _capture({"response": "ok"}, stderr=b"", exit_code=0)
            stderr = (b"note: " + memory.DEGRADATION_MARKER.encode() + b"\n"
                      if self.notice else b"note: nothing to report\n")
            return _capture({"response": "ok"}, stderr=stderr,
                            exit_code=self.exit_code)
        return None


class _AnsweringEmbedder(_FakeDegraded):
    """Connects to whatever endpoint the scenario pointed the embedder at.

    Attributes:
        status: What that endpoint answered, or None if nothing was reached.
    """

    def __init__(self) -> None:
        """Create the double with nothing measured yet."""
        super().__init__(notice=True)
        self.status = None

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation, probing the configured embedder first.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: Whatever :class:`_FakeDegraded` would answer.
        """
        args = list(call.args)
        if args[:1] == ["query"] and "-w" in args:
            root = pathlib.Path(args[args.index("-w") + 1])
            config = root / memory.MAGI_DIR_NAME / memory.MAGI_TOML_NAME
            if config.is_file():
                declared = tomllib.loads(config.read_text(encoding="utf-8"))
                url = declared.get("embedding", {}).get("base_url", "")
                if url:
                    self.status = _status_of(url)
        return super().__call__(call)


def _status_of(url: str) -> int | None:
    """Post to *url* and report the status it answered with.

    Args:
        url: The endpoint to reach.

    Returns:
        int | None: The status code, or None when nothing answered.
    """
    request = urllib.request.Request(
        url.rstrip("/") + "/embeddings", data=b"{}",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as answer:
            return answer.status
    except urllib.error.HTTPError as refused:
        return refused.code
    except OSError:
        return None


class RunIdAnchorTests(unittest.TestCase):
    """The correlation binds to the id, not to what the renderer puts after it.

    The needle used to end in a space, which tied the scenario to a field
    separator REQ-L63 never promised. Two properties have to hold at once and
    neither is free: a line whose separator is not a space must still be found,
    and another run's line must still be refused.
    """

    #: One rendered event, with a placeholder where the separator goes.
    _LINE = ("2026-08-31T00:00:00Z INFO run=%s%smagi_rs::main: memory: "
             "5 active, 1 archived, 0 pending re-embed (~9 KB index)")

    def test_the_line_is_found_whatever_separates_the_id_from_the_target(
            self) -> None:
        matcher = memory._startup_line_matcher(_R3_RUN_ID)
        for separator in (" ", "|", "\t", " target="):
            with self.subTest(separator=separator):
                line = (self._LINE % (_R3_RUN_ID, separator)).encode("utf-8")
                match = matcher(line)
                self.assertIsNotNone(
                    match,
                    "the run's own line stopped being found because the "
                    "separator changed, which misattributes rather than fails")
                self.assertEqual((b"5", b"1", b"0"), match.groups())

    def test_another_run_s_line_is_still_refused(self) -> None:
        matcher = memory._startup_line_matcher(_R3_RUN_ID)
        other = (self._LINE % ("999999-0123456789abcdef", " ")).encode("utf-8")
        self.assertIsNone(
            matcher(other),
            "dropping the trailing space must not widen the match to a "
            "neighbouring run's line")


if __name__ == "__main__":
    unittest.main()
