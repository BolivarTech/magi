# Author: Julian Bolivar
# Version: 0.22.0
# Date: 2026-10-02
"""Unit tests for the decide scenarios D1-D10 and the helpers they share.

The product is doubled by :class:`_FakeDecide`, which scaffolds a workspace the
way ``magi init`` does and answers ``magi decide`` from the ``magi.toml`` it
finds there. That is the point of the double: a scenario that rewrites the
configuration wrongly gets the answer the wrong configuration deserves, so the
rewrite is under test and not only the judging.

Every scenario is run twice over: against a healthy product, where it must pass,
and against one with the defect it exists to catch, where it must not.
"""

import json
import os
import pathlib
import socket
import tomllib
import unittest
from unittest import mock

from smoke import runs
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import decide
from smoke.scenarios import docs
from smoke.tests import support

_LOCAL_MODEL = "tev1:0.8b"
_LOCAL_URL = "http://localhost:11434"

#: What ``magi init`` writes, reduced to the parts the scenarios touch.
_SCAFFOLD = (
    'provider = "ollama"\nbase_url = "http://localhost:11434/v1"\n\n'
    '[openai]\nmodel = "kimi-k2.6:cloud"\n\n'
    '[decision]\nbase_url = "http://localhost:11434"\nmodel = "nimble"\n'
    '# min_confidence = 0.5\n\n'
    '[[decision.fallback]]\nmodel = "tev1"\n\n'
    '[magi]\nmelchior_model = "m"\n'
)

_KEY = "hosted-key-for-tests"


def _capture(stdout: bytes, stderr: bytes, exit_code: int) -> ProductOutput:
    """Build a canned capture.

    Args:
        stdout: What the product printed.
        stderr: What it wrote to the error stream.
        exit_code: The code it exited with.

    Returns:
        ProductOutput: The capture.
    """
    return ProductOutput(stdout=stdout, stderr=stderr, exit_code=exit_code,
                         command=["magi-rs"])


def _envelope(model, chain, answers=None, error=None, cost=None):
    """Build a decide envelope in the shape the product prints.

    Args:
        model: The answering model, or None.
        chain: The hops.
        answers: The answers, or None for an error envelope.
        error: The ``error`` object, or None.
        cost: ``usage.cost``.

    Returns:
        bytes: The JSON the product would print.
    """
    document = {"schema_version": 1, "run_id": "1-x", "model": model,
                "endpoint": "http://localhost:11434"}
    if answers is not None:
        document["answers"] = answers
    document["chain"] = chain
    if error is not None:
        document["error"] = error
    document["usage"] = {"input_tokens": 10, "output_tokens": 2, "cost": cost}
    document["timings"] = {"total_ms": 5}
    return json.dumps(document).encode("utf-8")


def _answers(request):
    """Answer every question of *request* the way a model would.

    Args:
        request: The decide request.

    Returns:
        dict: One answer per question, with the product's confidence.
    """
    out = {}
    for name, question in request["questions"].items():
        if question["type"] == "choice":
            options = list(question["criteria"])
            out[name] = {"type": "choice", "choice": options[0],
                         "probabilities": {o: 1 / len(options) for o in options},
                         "confidence": 0.0 if len(options) == 2 else 0.5}
        elif question["type"] == "score":
            levels = question["criteria"]
            out[name] = {"type": "score", "score": 1.0,
                         "legend": {str(i): v for i, v in enumerate(levels)},
                         "probabilities": {str(i): 1 / len(levels)
                                           for i in range(len(levels))},
                         "confidence": 0.4}
        else:
            out[name] = {"type": "noul", "noul": 0.9, "confidence": 0.8}
    return out


class _FakeDecide:
    """A product double for ``init``, ``decide`` and ``vault ls``.

    Attributes:
        key_leak: Whether the 401 path copies the key into stderr.
        ignores_chain: Whether an absent primary is answered as if present.
        ignores_timeout_flag: Whether ``--timeout`` is ignored.
        ignores_explicit_model: Whether ``model`` in the input is ignored.
        reachable_hosted: Whether OpenRouter answers at all.
    """

    def __init__(self, key_leak=False, ignores_chain=False,
                 ignores_timeout_flag=False, ignores_explicit_model=False,
                 reachable_hosted=True, writes_logs=True):
        """Create the double.

        Args:
            key_leak: Copy the key into stderr on a refusal.
            ignores_chain: Never rotate past the primary.
            ignores_timeout_flag: Never honour ``--timeout``.
            ignores_explicit_model: Ignore ``model`` in the input.
            reachable_hosted: Whether OpenRouter is reachable.
            writes_logs: Whether a run log is written.
        """
        self.key_leak = key_leak
        self.ignores_chain = ignores_chain
        self.ignores_timeout_flag = ignores_timeout_flag
        self.ignores_explicit_model = ignores_explicit_model
        self.reachable_hosted = reachable_hosted
        self.writes_logs = writes_logs

    def __call__(self, call):
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput | None: The canned answer, or None for the default.
        """
        args = list(call.args)
        if args[:1] == ["init"]:
            root = pathlib.Path(call.cwd)
            (root / ".magi").mkdir(parents=True, exist_ok=True)
            (root / ".magi" / "magi.toml").write_text(_SCAFFOLD,
                                                      encoding="utf-8")
            return _capture(b"", b"", 0)
        if args[:2] == ["vault", "ls"]:
            return _capture(b"", b"", 0)
        if args[:1] == ["decide"]:
            return self._decide(call, args)
        return None

    def _decide(self, call, args):
        """Answer ``decide`` from the workspace's configuration.

        Args:
            call: The invocation.
            args: Its argv.

        Returns:
            ProductOutput: The answer.
        """
        root = pathlib.Path(args[args.index("-w") + 1])
        text = (root / ".magi" / "magi.toml").read_text(encoding="utf-8")
        if not text.strip():
            return _capture(b"", b"magi.toml declares nothing. Delete it to "
                            b"run with defaults, or run `magi-rs init`.\n", 2)
        config = tomllib.loads(text)
        if "decision" not in config:
            return _capture(b"", b"magi.toml has no [decision] section: run "
                            b"`magi-rs init`.\n", 2)
        if self.writes_logs:
            logs = root / ".magi" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "run.jsonl").write_text('{"msg":"decide"}' + chr(10),
                                            encoding="utf-8")
        request = json.loads(call.stdin.decode("utf-8"))
        if "-q" in args and "questions" in request:
            return _capture(_envelope(None, [], error={
                "kind": "invalid_input",
                "message": "questions in both places"}), b"", 2)
        if "-q" in args:
            request["questions"] = json.loads(
                pathlib.Path(args[args.index("-q") + 1]).read_text())
        section = config["decision"]
        models = [section["model"]] + [
            f["model"] for f in section.get("fallback", [])]
        if request.get("model") and not self.ignores_explicit_model:
            models = [request["model"]]
        env = call.env or {}
        key = env.get("DECISION_API_KEY", "")
        return self._walk(call, args, section, models, request, key,
                          "-o" in args and args[args.index("-o") + 1])

    def _walk(self, call, args, section, models, request, key, out_path):
        """Walk the chain and render the envelope.

        Args:
            call: The invocation.
            args: Its argv.
            section: The ``[decision]`` table.
            models: The models to try, in order.
            request: The parsed input.
            key: The ``DECISION_API_KEY`` the product was given.
            out_path: The ``-o`` target, or a falsy value.

        Returns:
            ProductOutput: The answer.
        """
        base_url = section["base_url"]
        hops = []
        for model in models:
            hop = self._attempt(args, section, base_url, model, key)
            hops.append(hop)
            if hop["outcome"] == "answered" or hop.get("stop"):
                break
        last = hops[-1]
        stderr = b""
        if last.get("stop") == "key":
            stderr = (b"credential " + key.encode() if self.key_leak else b"")
            hops = [{k: v for k, v in last.items() if k != "stop"}]
            body = _envelope(None, hops, error={
                "kind": "missing_key",
                "message": "set DECISION_API_KEY in the environment"})
            return _capture(body, stderr, 2)
        for hop in hops:
            hop.pop("stop", None)
        if last["outcome"] == "answered":
            body = _envelope(last["model"], hops, _answers(request),
                             cost=0.001 if "openrouter" in base_url else None)
            if out_path:
                pathlib.Path(out_path).write_text(body.decode("utf-8"),
                                                  encoding="utf-8")
                return _capture(b"", b"", 0)
            return _capture(body, b"", 0)
        body = _envelope(None, hops, error={
            "kind": "decision_unavailable", "message": "chain exhausted"})
        return _capture(body, b"", 1)

    def _attempt(self, args, section, base_url, model, key):
        """One hop of the chain.

        Args:
            args: The argv.
            section: The ``[decision]`` table.
            base_url: The endpoint.
            model: The model.
            key: The key handed in.

        Returns:
            dict: The hop; ``stop`` marks a refusal that ends the chain.
        """
        hosted = "openrouter.ai" in base_url
        if hosted and not self.reachable_hosted:
            return {"model": model, "outcome": "failed", "cause": "transport"}
        if base_url == "https://openrouter.ai":
            # What a real OpenRouter does: a non-JSON 200 for the prefix-less path.
            return {"model": model, "outcome": "failed", "cause": "invalid_response",
                    "detail": "invalid response: the response body is not valid JSON; no "
                    "System One endpoint answered at https://openrouter.ai/v1/systemone; "
                    "check that base_url includes the provider's API prefix"}
        if "127.0.0.1" in base_url and model == "smoke-echo-model":
            return {"model": model, "outcome": "failed",
                    "cause": "unauthorized", "stop": "key"}
        if hosted and not key:
            return {"model": model, "outcome": "failed",
                    "cause": "unauthorized", "stop": "key"}
        if "127.0.0.1" in base_url:
            honoured = "--timeout" in args and not self.ignores_timeout_flag
            return {"model": model, "outcome": "failed",
                    "cause": "timeout" if honoured else "transport"}
        if model == decide.ABSENT_DECISION_MODEL and not self.ignores_chain:
            return {"model": model, "outcome": "failed",
                    "cause": "model_not_found"}
        if model == decide.OPENROUTER_POLICY_MODEL:
            return {"model": model, "outcome": "failed",
                    "cause": "account_policy"}
        threshold = section.get("min_confidence")
        if threshold is not None and threshold >= 0.9:
            return {"model": model, "outcome": "low_confidence",
                    "confidence": 0.0, "min_confidence": threshold}
        return {"model": model, "outcome": "answered"}


def _run(case, product, key=None):
    """Install the double and a local environment toml; return a runner.

    Args:
        case: The test case.
        product: The product double.
        key: A hosted key to export for the test, or None for none.

    Returns:
        Callable: Takes a scenario id and returns ``{assertion: Finding}``.
    """
    root = pathlib.Path(support.scratch_dir(case)) / "env"
    (root / ".magi").mkdir(parents=True)
    (root / ".magi" / "magi.toml").write_text(
        _SCAFFOLD.replace('model = "nimble"', 'model = "%s"' % _LOCAL_MODEL),
        encoding="utf-8")
    support.install_fake_runs(case, responder=product, env_root=root)
    patched = {"DECISION_API_KEY": key} if key is not None else {}
    cleared = mock.patch.dict(os.environ, patched, clear=False)
    cleared.start()
    case.addCleanup(cleared.stop)
    if key is None:
        os.environ.pop("DECISION_API_KEY", None)

    def run_scenario(scenario_id):
        findings = list(DEFAULT_REGISTRY.get(scenario_id).func(None))
        return {f.assertion: f for f in findings}

    return run_scenario


class RegistrationTests(unittest.TestCase):
    """The ten scenarios are registered, standalone, with the promised counts."""

    _COUNTS = {"D1": 3, "D2": 3, "D3": 2, "D4": 2, "D5": 3, "D6": 2,
               "D7": 2, "D8": 2, "D9": 3, "D10": 4}

    def test_each_scenario_declares_its_promised_assertions(self) -> None:
        for scenario_id, count in self._COUNTS.items():
            with self.subTest(scenario=scenario_id):
                entry = DEFAULT_REGISTRY.get(scenario_id)
                self.assertEqual(count, len(entry.assertions))
                self.assertIsNone(entry.run)

    def test_the_decide_assertions_total_twenty_six(self) -> None:
        self.assertEqual(26, sum(self._COUNTS.values()))

    def test_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            ("the planted credential does not reach stdout",
             "the planted credential does not reach stderr",
             "the planted credential does not reach the run log"),
            decide.D9_ASSERTIONS)
        self.assertEqual(
            ("-q with questions also in the input exits 2 before any request",
             "-o writes the envelope and leaves stdout empty",
             "--timeout overrides timeout_secs per attempt",
             "an explicit model is tried alone"), decide.D10_ASSERTIONS)

    def test_only_the_local_backend_scenarios_need_a_backend(self) -> None:
        needing = sorted(i for i in self._COUNTS
                         if DEFAULT_REGISTRY.get(i).needs_backend)
        self.assertEqual(["D1", "D10", "D3", "D8"], needing)


class HealthyProductTests(unittest.TestCase):
    """Against a product that behaves, every scenario passes."""

    def _all_pass(self, scenario_id, key=None, product=None):
        run = _run(self, product or _FakeDecide(), key=key)
        findings = run(scenario_id)
        declared = DEFAULT_REGISTRY.get(scenario_id).assertions
        self.assertEqual(set(declared), set(findings))
        for text, finding in findings.items():
            self.assertEqual(Outcome.PASS, finding.outcome,
                             "%s: %s" % (text, finding.detail))

    def test_d1_passes(self) -> None:
        self._all_pass("D1")

    def test_d2_passes_with_a_key(self) -> None:
        self._all_pass("D2", key=_KEY)

    def test_d3_passes(self) -> None:
        self._all_pass("D3")

    def test_d4_passes_with_a_key(self) -> None:
        self._all_pass("D4", key=_KEY)

    def test_d5_passes(self) -> None:
        self._all_pass("D5")

    def test_d6_passes(self) -> None:
        self._all_pass("D6")

    def test_d7_passes(self) -> None:
        self._all_pass("D7")

    def test_d8_passes(self) -> None:
        self._all_pass("D8")

    def test_d9_passes(self) -> None:
        self._all_pass("D9")

    def test_d10_passes(self) -> None:
        self._all_pass("D10")


class DefectTests(unittest.TestCase):
    """Each scenario fails, and names the right assertion, on its defect."""

    def _finding_of(self, scenario_id, index, product, key=None):
        run = _run(self, product, key=key)
        declared = DEFAULT_REGISTRY.get(scenario_id).assertions
        return run(scenario_id)[declared[index]]

    def test_a_product_that_never_rotates_fails_d3(self) -> None:
        finding = self._finding_of("D3", 0, _FakeDecide(ignores_chain=True))
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_a_key_copied_into_stderr_fails_d9(self) -> None:
        finding = self._finding_of("D9", 1, _FakeDecide(key_leak=True))
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_a_clean_stdout_still_passes_d9_when_stderr_leaks(self) -> None:
        finding = self._finding_of("D9", 0, _FakeDecide(key_leak=True))
        self.assertEqual(Outcome.PASS, finding.outcome)

    def test_a_product_that_writes_no_log_cannot_test_d9(self) -> None:
        finding = self._finding_of("D9", 2, _FakeDecide(writes_logs=False))
        self.assertEqual(Outcome.CANNOT_TEST, finding.outcome)

    def test_a_product_ignoring_timeout_fails_d10(self) -> None:
        finding = self._finding_of("D10", 2,
                                _FakeDecide(ignores_timeout_flag=True))
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_a_product_ignoring_the_explicit_model_fails_d10(self) -> None:
        finding = self._finding_of("D10", 3,
                                _FakeDecide(ignores_explicit_model=True))
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_an_unreachable_host_cannot_test_the_prefix_hint(self) -> None:
        finding = self._finding_of("D7", 1, _FakeDecide(reachable_hosted=False))
        self.assertEqual(Outcome.CANNOT_TEST, finding.outcome)

    def test_an_unreachable_host_cannot_test_the_missing_key(self) -> None:
        finding = self._finding_of("D5", 0, _FakeDecide(reachable_hosted=False))
        self.assertEqual(Outcome.CANNOT_TEST, finding.outcome)

    def test_a_product_that_does_nothing_reports_every_assertion(self) -> None:
        for scenario_id in ("D1", "D3", "D6", "D7", "D8", "D10"):
            with self.subTest(scenario=scenario_id):
                run = _run(self, lambda call: None)
                findings = run(scenario_id)
                declared = DEFAULT_REGISTRY.get(scenario_id).assertions
                self.assertEqual(set(declared), set(findings))
                self.assertNotIn(Outcome.PASS,
                                 {f.outcome for f in findings.values()})


class HostedKeyTests(unittest.TestCase):
    """The hosted scenarios need a key and say so, never passing without it."""

    def test_d2_and_d4_cannot_test_without_a_key(self) -> None:
        run = _run(self, _FakeDecide(), key=None)
        for scenario_id in ("D2", "D4"):
            with self.subTest(scenario=scenario_id):
                outcomes = {f.outcome for f in run(scenario_id).values()}
                self.assertEqual({Outcome.CANNOT_TEST}, outcomes)

    def test_a_blank_key_counts_as_absent(self) -> None:
        run = _run(self, _FakeDecide(), key="   ")
        outcomes = {f.outcome for f in run("D2").values()}
        self.assertEqual({Outcome.CANNOT_TEST}, outcomes)

    def test_the_key_is_never_written_into_a_toml_or_an_archive(self) -> None:
        product = _FakeDecide()
        _run(self, product, key=_KEY)("D2")
        archive = runs.archive_root()
        for path in archive.rglob("*"):
            if path.is_file():
                self.assertNotIn(_KEY.encode(), path.read_bytes())


class ConfigurationRewriteTests(unittest.TestCase):
    """The ``[decision]`` text the scenarios write and strip."""

    def test_the_section_renders_a_chain_that_parses(self) -> None:
        text = decide._decision_section(
            "http://x", "a", fallbacks=("b", "c"), min_confidence=0.5,
            timeout_secs=7)
        section = tomllib.loads(text)["decision"]
        self.assertEqual("a", section["model"])
        self.assertEqual(0.5, section["min_confidence"])
        self.assertEqual(7, section["timeout_secs"])
        self.assertEqual(["b", "c"], [f["model"] for f in section["fallback"]])

    def test_without_decision_removes_the_section_and_its_fallbacks(
            self) -> None:
        stripped = tomllib.loads(decide._without_decision(_SCAFFOLD))
        self.assertNotIn("decision", stripped)
        self.assertEqual("m", stripped["magi"]["melchior_model"])

    def test_without_decision_keeps_what_follows_a_dropped_block(self) -> None:
        stripped = decide._without_decision(_SCAFFOLD)
        self.assertIn("[openai]", stripped)
        self.assertIn("[magi]", stripped)

    def test_installing_a_section_leaves_exactly_one_decision_table(
            self) -> None:
        root = pathlib.Path(support.scratch_dir(self))
        (root / ".magi").mkdir()
        (root / ".magi" / "magi.toml").write_text(_SCAFFOLD, encoding="utf-8")
        self.assertTrue(decide._install_decision(
            root, decide._decision_section("http://y", "z")))
        doc = tomllib.loads((root / ".magi" / "magi.toml").read_text())
        self.assertEqual("z", doc["decision"]["model"])
        self.assertNotIn("fallback", doc["decision"])
        self.assertEqual("m", doc["magi"]["melchior_model"])

    def test_installing_without_a_toml_reports_it(self) -> None:
        root = pathlib.Path(support.scratch_dir(self))
        self.assertFalse(decide._install_decision(
            root, decide._decision_section("http://y", "z")))

    def test_the_local_target_is_read_off_the_environment_toml(self) -> None:
        _run(self, _FakeDecide())
        self.assertEqual((_LOCAL_URL, _LOCAL_MODEL), decide._local_target())


class SilentEndpointTests(unittest.TestCase):
    """The D10 listener accepts a connection and never answers."""

    def test_it_accepts_and_stays_quiet(self) -> None:
        with decide._silent_endpoint() as base_url:
            port = int(base_url.rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", port), 2) as conn:
                conn.sendall(b"POST / HTTP/1.1\r\n\r\n")
                conn.settimeout(0.5)
                with self.assertRaises(socket.timeout):
                    conn.recv(1)

    def test_it_stops_listening_afterwards(self) -> None:
        with decide._silent_endpoint() as base_url:
            port = int(base_url.rsplit(":", 1)[1])
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), 1).close()


def _help_text(flags):
    """Render a clap-shaped options section naming *flags*.

    Args:
        flags: The long flags to list.

    Returns:
        str: The help text.
    """
    return "\n".join(["Usage: magi-rs decide [OPTIONS]", "", "Options:"]
                     + ["      %s <V>" % flag for flag in flags]
                     + ["  -h, --help"])


class _DocsProduct:
    """Answers ``--help``, ``init`` and the configuration probe for S13.

    Attributes:
        decide_flags: The flags ``decide --help`` lists.
        probe_exit: What a configuration probe exits with.
    """

    def __init__(self, decide_flags=docs.DECIDE_FLAGS, probe_exit=0):
        """Create the double.

        Args:
            decide_flags: The flags the help lists.
            probe_exit: The exit code of a configuration probe.
        """
        self.decide_flags = decide_flags
        self.probe_exit = probe_exit

    def __call__(self, call):
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput | None: The canned answer.
        """
        args = list(call.args)
        if args[:1] == ["init"]:
            root = pathlib.Path(call.cwd)
            (root / ".magi").mkdir(parents=True, exist_ok=True)
            (root / ".magi" / "magi.toml").write_text(
                '[magi]\nmelchior_model = "a"\n', encoding="utf-8")
            return _capture(b"", b"", 0)
        if args[:1] == ["query"]:
            return _capture(b"", b"", self.probe_exit)
        if args == ["--help"]:
            return _capture(b"Usage: magi-rs\n\nCommands:\n"
                            b"  decide  Ask\n  init  Scaffold\n", b"", 0)
        if args == ["decide", "--help"]:
            return _capture(_help_text(self.decide_flags).encode(), b"", 0)
        return None


class PublishedDocsDecideTests(unittest.TestCase):
    """S13's two decide assertions: the flags, and the annotated example."""

    _EXAMPLE = '[decision]\nbase_url = "http://localhost:11434"\nmodel = "n"\n'
    _README = "```bash\nmagi-rs decide -i request.json\n```\n"

    def _findings(self, files, product):
        """Run S13 over a scratch repository holding *files*.

        Args:
            files: Relative path to content.
            product: The product double.

        Returns:
            dict: Assertion text to finding.
        """
        repo = pathlib.Path(support.scratch_dir(self))
        for name, body in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(body, encoding="utf-8")
        support.install_fake_runs(self, product, repo_root=repo)
        with mock.patch.object(
                docs, "published_docs",
                return_value=[pathlib.Path(n) for n in files
                              if n.endswith(".md")]):
            findings = list(docs.the_published_documentation_is_still_true(
                None))
        return {f.assertion: f for f in findings}

    def _flags(self, files, product):
        """The fifth assertion's finding."""
        return self._findings(files, product)[docs.S13_ASSERTIONS[4]]

    def _example(self, files, product):
        """The sixth assertion's finding."""
        return self._findings(files, product)[docs.S13_ASSERTIONS[5]]

    def test_the_new_texts_are_the_declared_ones(self) -> None:
        self.assertEqual("the documented decide flags exist",
                         docs.S13_ASSERTIONS[4])
        self.assertEqual("the [decision] example in docs/magi.toml.example "
                         "parses", docs.S13_ASSERTIONS[5])

    def test_a_documented_decide_with_every_flag_passes(self) -> None:
        finding = self._flags({"README.md": self._README}, _DocsProduct())
        self.assertEqual(Outcome.PASS, finding.outcome, finding.detail)

    def test_a_decide_that_lost_a_flag_fails_naming_it(self) -> None:
        product = _DocsProduct(decide_flags=("--input", "--output"))
        finding = self._flags({"README.md": self._README}, product)
        self.assertEqual(Outcome.FAIL, finding.outcome)
        self.assertIn("--no-memory", finding.detail)

    def test_a_decide_the_docs_never_show_fails(self) -> None:
        finding = self._flags({"README.md": "Nothing here."}, _DocsProduct())
        self.assertEqual(Outcome.FAIL, finding.outcome)
        self.assertIn("no published document", finding.detail)

    def test_an_example_with_a_decision_section_passes(self) -> None:
        finding = self._example(
            {"README.md": "x", "docs/magi.toml.example": self._EXAMPLE},
            _DocsProduct())
        self.assertEqual(Outcome.PASS, finding.outcome, finding.detail)

    def test_an_example_without_a_decision_section_fails(self) -> None:
        finding = self._example(
            {"README.md": "x", "docs/magi.toml.example": "[magi]\n"},
            _DocsProduct())
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_an_example_that_is_not_toml_fails(self) -> None:
        finding = self._example(
            {"README.md": "x", "docs/magi.toml.example": "[decision\n"},
            _DocsProduct())
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_an_example_the_product_refuses_fails(self) -> None:
        finding = self._example(
            {"README.md": "x", "docs/magi.toml.example": self._EXAMPLE},
            _DocsProduct(probe_exit=2))
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_a_missing_example_fails(self) -> None:
        finding = self._example({"README.md": "x"}, _DocsProduct())
        self.assertEqual(Outcome.FAIL, finding.outcome)


if __name__ == "__main__":
    unittest.main()
