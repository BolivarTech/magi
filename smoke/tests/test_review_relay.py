# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""The recording relay, the setup steps and S53/S55-S57 over the relay's record (MAGI S6 MS3 loop9, R7a)."""

import dataclasses
import http.server
import json
import pathlib
import shutil
import subprocess
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request

from smoke import runs
from smoke.errors import HarnessError
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.scenarios import review_context as rc, review_masking as rm
from smoke.tests.support import R18_HEALTHY_REQUESTS, scenario_outcomes as _outcomes, scenario_result as _result


class _Echo(http.server.BaseHTTPRequestHandler):
    """An upstream that answers 201 with what it was sent."""

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        reply = b"echo:" + self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(201)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args) -> None:
        """Keep the test output quiet."""


def _chunk(text: str) -> tuple:
    """An ``/embeddings`` request for *text*, as the relay records it."""
    return ("/v1/embeddings", json.dumps({"input": [text]}).encode())


class _Redirect(http.server.BaseHTTPRequestHandler):
    """An upstream that answers every request with a redirect off the relay."""

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:
        """Keep the test output quiet."""


class RelayTests(unittest.TestCase):

    def test_relay_never_forwards_a_3xx(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Melchior): a redirect the product followed would leave
        # the relay unrecorded; it is answered as an error and the record marked incomplete.
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with runs.recording_relay("http://127.0.0.1:%d/v1" % upstream.server_address[1]) as (url, rec):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    opener.open(urllib.request.Request(url + "/embeddings", data=b"x"), timeout=5)
                self.assertGreaterEqual(caught.exception.code, 500)
                self.assertIsNone(caught.exception.headers.get("Location"))
        finally:
            upstream.shutdown()
            upstream.server_close()
        self.assertFalse(rec.complete)

    def test_the_relay_forwards_and_records_each_request(self) -> None:
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with runs.recording_relay("http://127.0.0.1:%d/v1" % upstream.server_address[1]) as (url, rec):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(urllib.request.Request(url + "/embeddings", data=b"x"), timeout=5) as r:
                    self.assertEqual((201, b"echo:x"), (r.status, r.read()))
        finally:
            upstream.shutdown()
            upstream.server_close()
        self.assertEqual(((("/v1/embeddings", b"x"),), True), (rec.recorded, rec.complete))

    def test_the_relay_keeps_the_query_and_flags_a_bad_body(self) -> None:
        # [MAGI S6 MS3 loop10, R7a Melchior, Caspar] The query was dropped from the record, and
        # a body shorter or longer than its Content-Length left the record "complete".
        import socket
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            for path, declared, body, complete in (("/v1/embeddings?k=v", 1, b"x", True),
                                                   ("/v1/embeddings", 10, b"abc", False),
                                                   ("/v1/embeddings", 1, b"xyz", False)):
                with self.subTest(declared=declared, body=body), \
                        runs.recording_relay("http://127.0.0.1:%d/v1" % upstream.server_address[1]) as (url, rec):
                    port = int(url.split(":")[2].split("/")[0])
                    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
                        s.sendall(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s"
                                  % (path.encode(), declared, body))
                        s.shutdown(socket.SHUT_WR)
                        while s.recv(4096):
                            pass
                    self.assertEqual(complete, rec.complete)
                    if complete:
                        self.assertEqual(path, rec.recorded[0][0])
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_a_failed_restore_is_retried_then_named(self) -> None:
        # [MAGI S6 MS3 loop10, R7a Caspar] A restore that raised left the workspace pointed at a
        # closed backend, unsaid.
        real = runs._replace_bytes
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "magi.toml"
            original = b'base_url = "http://a/v1"\n[embedding]\nbase_url = "http://a/v1"\n'
            path.write_bytes(original)
            for failures, raises in ((1, False), (2, True)):
                left = [failures]

                def flaky(p, data, what, left=left):
                    if what.startswith("restore") and left[0]:
                        left[0] -= 1
                        raise HarnessError("disk full")
                    real(p, data, what)
                with self.subTest(failures=failures), unittest.mock.patch.object(runs, "_replace_bytes", flaky):
                    if raises:
                        with self.assertRaisesRegex(HarnessError, "reset-env"):
                            with runs.embedder_pointed_away(path):
                                pass
                    else:
                        with runs.embedder_pointed_away(path):
                            pass
                        self.assertEqual(original, path.read_bytes())

    def test_rewrite_must_parse(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Balthasar): the line-based rewrite was never parsed back;
        # one that is not TOML is a harness error, and the file is left as it was.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "magi.toml"
            broken = b'[embedding]\nbase_url = """http://a\n/v1"""\nmodel = "m"\n'
            path.write_bytes(broken)
            with self.assertRaisesRegex(HarnessError, "TOML"):
                runs.point_embedding_at(path, "http://127.0.0.1:9/v1")
            self.assertEqual(broken, path.read_bytes())

    def test_a_multiline_string_line_naming_the_embedder_is_refused_loudly(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a Caspar): declared line-based limit; a refusal, never a rewrite.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "magi.toml"
            text = 'note = """\nembedding = 1\n"""\n[embedding]\nmodel = "m"\n'
            path.write_text(text)
            with self.assertRaises(HarnessError):
                runs.point_embedding_at(path, "http://127.0.0.1:9/v1")
            self.assertEqual(text, path.read_text(), "the file is left as it was")

    def test_setup_cannot_be_combined_with_a_run_the_harness_rewrites(self) -> None:
        # MAGI FIX (MS3 §6 loop14, R7a1): this replaces `test_setup_clears_the_embedder_rewrite`
        # and `test_setup_clears_the_model_ban` (loops 11 and 13), which built definitions that
        # inherited the run's flags into a setup invocation. The combination is refused when the
        # definition is built, so a setup review can never run uncounted or against a live embedder.
        base = dict(run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS,
                    stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
                    setup=(runs.Invocation(argv=("code-review",), timeout_s=5),))
        # An unreachable embedder is allowed beside a setup (R22 seeds its history against the live
        # one); see `test_a_setup_may_run_before_an_embedder_that_is_pointed_away`.
        for extra in ({"forbids_model_calls": True}, {"rotates": True}):
            with self.subTest(extra=extra), self.assertRaises(HarnessError):
                runs.RunDefinition(**base, **extra)

    def test_a_temporary_file_name_is_not_shared_between_writers(self) -> None:
        # MAGI FIX (MS3 §6 loop14, R7a1): the sibling was always `<name>.tmp`, so two writers of one
        # file would collide on it.
        seen = []
        real = runs.os.replace

        def spy(source, target):
            seen.append(pathlib.Path(source).name)
            real(source, target)
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(runs.os, "replace", spy):
            path = pathlib.Path(tmp) / "magi.toml"
            path.write_bytes(b"a")
            runs._replace_bytes(path, b"b", "first write")
            runs._replace_bytes(path, b"c", "second write")
            self.assertEqual(b"c", path.read_bytes())
            self.assertEqual([], [p.name for p in pathlib.Path(tmp).iterdir() if p.name != "magi.toml"])
        self.assertEqual(2, len(set(seen)), seen)
        self.assertNotIn("magi.toml.tmp", seen)

    def test_setup_shares_the_run_clock(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7a Balthasar): the setup reviews drew a clock each, unbounded
        # by the run's own; now every step is capped by what is left of the one clock.
        clock, seen = [0.0], []

        def invoke(definition, stdin):
            seen.append(definition.timeout_s)
            clock[0] += 70

        executor = runs.RunExecutor(binary=None, env={}, config=None)
        executor._clock_end = 100.0
        step = runs.Invocation(argv=("code-review",), timeout_s=60)
        definition = runs.RunDefinition(
            run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS, stdin=b"",
            needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=3 * 60 + runs.MIN_MEASURED_ALLOWANCE_S,
            setup=(step, step, step))
        with unittest.mock.patch.object(executor, "_invoke", invoke), \
                unittest.mock.patch.object(runs.time, "monotonic", lambda: clock[0]):
            executor._run_setup(definition, pathlib.Path("."))
        self.assertEqual([60, 30], seen)
        self.assertIsNotNone(executor._setup_expired)

    def test_r21_declares_its_three_reviews(self) -> None:
        # MAGI FIX (MS3 §6 loop13): the declared cost is the sum of the three reviews, one clock.
        self.assertEqual(3 * runs.TRIO_RUNS_MEASURED_S["R18"], runs.TRIO_RUNS_MEASURED_S["R21"])
        self.assertEqual(3 * runs.DEFINITIONS["R20"].timeout_s, runs.DEFINITIONS["R21"].timeout_s)

    def test_restore_backs_off(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Balthasar): the retry ran at once, where a lock held by an
        # indexer clears after a moment.
        real, left = runs._replace_bytes, [1]

        def flaky(p, data, what):
            if left[0]:
                left[0] -= 1
                raise HarnessError("locked")
            real(p, data, what)
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(runs, "_replace_bytes", flaky), \
                unittest.mock.patch.object(runs.time, "sleep") as slept:
            path = pathlib.Path(tmp) / "magi.toml"
            path.write_bytes(b"rewritten")
            runs._restore(path, b"x")
            self.assertEqual(b"x", path.read_bytes())
        slept.assert_called_once_with(runs.RESTORE_RETRY_DELAY_S)

    def test_the_upstream_is_the_one_endpoint_the_config_declares(self) -> None:
        with self.assertRaises(HarnessError), runs.recording_relay("http://u:p@127.0.0.1:1/v1"):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "magi.toml"
            path.write_text('base_url = "http://a/v1"\n[magi]\nbase_url = "http://a/v1"\n', encoding="utf-8")
            self.assertEqual("http://a/v1", runs.relay_upstream(path, {}))
            with self.assertRaises(HarnessError):
                runs.relay_upstream(path, {"OPENAI_BASE_URL": "http://b/v1"})
            path.write_text('base_url = "http://a/v1"\n[embedding]\nbase_url = "http://b/v1"\n', encoding="utf-8")
            with self.assertRaises(HarnessError):
                runs.relay_upstream(path, {})

    def test_a_recording_run_needs_a_workspace_and_no_other_endpoint_rewrite(self) -> None:
        base = dict(run_id="RX", stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
                    records_requests=True)
        for extra in ({"argv": ("code-review",)},
                      {"argv": ("code-review",) + runs.REVIEW_FLAGS, "forbids_model_calls": True},
                      {"argv": ("code-review",) + runs.REVIEW_FLAGS, "unreachable_embedder": True}):
            with self.subTest(extra=extra), self.assertRaises(HarnessError):
                runs.RunDefinition(**base, **extra)

    @unittest.skipUnless(shutil.which("git") is not None, "git is required")
    def test_a_setup_commit_records_exactly_its_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            subprocess.run(runs.git_command(ws, "init", "-q"), check=True, env=runs.git_env())
            (ws / "stray.txt").write_bytes(b"not committed")
            runs.commit_files(ws, runs.Commit(files=(("a/b.rs", b"fn b() {}\n"),), message="one"))
            listed = subprocess.run(runs.git_command(ws, "ls-files"), check=True, env=runs.git_env(),
                                    capture_output=True, text=True).stdout.split()
        self.assertEqual(["a/b.rs"], listed)

    def test_two_endpoints_are_unavailable(self) -> None:
        # MAGI FIX (MS3 §6 loop13): a second endpoint is the environment's limit (CANNOT_TEST),
        # not a harness defect (exit 3).
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "magi.toml"
            path.write_text('base_url = "http://a/v1"\n[embedding]\nbase_url = "http://b/v1"\n', encoding="utf-8")
            with self.assertRaises(runs.RelayUnavailable):
                runs.relay_upstream(path, {})
            path.write_text('base_url = "http://a/v1"\n', encoding="utf-8")
            with self.assertRaises(runs.RelayUnavailable):
                runs.relay_upstream(path, {"OPENAI_BASE_URL": "http://b/v1"})

    def test_unrunnable_relay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            (ws / ".magi").mkdir()
            (ws / ".magi" / "magi.toml").write_text(
                'base_url = "http://a/v1"\n[embedding]\nbase_url = "http://b/v1"\n', encoding="utf-8")
            definition = runs.RunDefinition(
                run_id="RX", stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
                records_requests=True, argv=("code-review",) + runs.REVIEW_FLAGS)
            executor = runs.RunExecutor(None, None, None)
            with executor._recorded_requests(definition, ws, []) as runnable:
                self.assertIsNone(runnable)
            self.assertIn("more than one endpoint", executor._unrunnable_cause)

    def test_a_non_utf8_config_is_a_harness_error(self) -> None:
        # MAGI FIX (MS3 §6 loop13): a config that is not UTF-8 is a typed harness error (exit 3)
        # on every path that reads it, never a UnicodeDecodeError.
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            (ws / ".magi").mkdir()
            (ws / ".magi" / "magi.toml").write_bytes(b'base_url = "http://a/v1"\n# \xff\xfe\n')
            definition = runs.RunDefinition(
                run_id="RX", stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
                records_requests=True, argv=("code-review",) + runs.REVIEW_FLAGS)
            with self.assertRaises(HarnessError):
                with runs.RunExecutor(None, None, None)._recorded_requests(definition, ws, []):
                    pass

    def test_relay_keeps_no_credential(self) -> None:
        # MAGI FIX (MS3 §6 loop13): the record scenarios read must not hold the operator's key.
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with runs.recording_relay("http://127.0.0.1:%d/v1" % upstream.server_address[1]) as (url, rec):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request = urllib.request.Request(url + "/embeddings", data=b"x", headers={
                    "Authorization": "Bearer REALCREDENTIAL", "X-Api-Key": "REALCREDENTIAL",
                    "X-Trace": "visible"})
                opener.open(request, timeout=5).read()
        finally:
            upstream.shutdown()
            upstream.server_close()
        kept = rec.headers[0]
        self.assertNotIn(b"REALCREDENTIAL", kept)
        self.assertIn(b"visible", kept)


class LoopSixteenRelayTests(unittest.TestCase):
    """MAGI FIX (MS3 §6 loop16, R7a1): the relay's staging, snapshot and fixtures."""

    def _definition(self) -> "runs.RunDefinition":
        return runs.RunDefinition(
            run_id="RX", stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
            records_requests=True, argv=("code-review",) + runs.REVIEW_FLAGS)

    def test_a_dry_run_cause_ends_the_run_unrunnable_and_never_stages_the_relay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            (ws / ".magi").mkdir()
            (ws / ".magi" / "magi.toml").write_text('base_url = "http://a/v1"\n', encoding="utf-8")
            executor = runs.RunExecutor(None, None, None)
            executor._unrunnable_cause = "the relay cannot be staged: off the relay"
            with unittest.mock.patch.object(runs, "recording_relay",
                                            side_effect=AssertionError("staged a refused relay")):
                with executor._recorded_requests(self._definition(), ws, []) as runnable:
                    self.assertIsNone(runnable)
            self.assertIn("off the relay", executor._unrunnable_cause)

    def test_a_credentialed_or_non_http_upstream_is_unavailable_not_a_harness_error(self) -> None:
        for root in ("http://u:p@127.0.0.1:1/v1", "ftp://a/v1", "http:///v1", "file:///x"):
            with self.subTest(root=root), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "magi.toml"
                path.write_text('base_url = "%s"\n' % root, encoding="utf-8")
                with self.assertRaises(runs.RelayUnavailable):
                    runs.relay_upstream(path, {})

    def test_one_locked_snapshot_returns_bodies_headers_and_completeness(self) -> None:
        rec = runs.RequestBodies(cap=1)
        rec.record("/a", b"x", b"H: 1\n")
        self.assertEqual(((("/a", b"x"),), (b"H: 1\n",), True), rec.snapshot())
        rec.record("/b", b"y", b"H: 2\n")
        self.assertEqual(((("/a", b"x"),), (b"H: 1\n",), False), rec.snapshot())

    def test_the_result_is_built_from_the_snapshot_alone(self) -> None:
        source = pathlib.Path(runs.__file__).read_text(encoding="utf-8").replace("\r", "")
        execute = source.split("    def execute(", 1)[1].split("    def _recorded_requests(", 1)[0]
        self.assertIn(".snapshot()", execute)
        for forbidden in (".recorded", ".complete", ".headers"):
            self.assertNotIn("recorded[0]" + forbidden, execute)

    def test_an_incomplete_record_is_never_read_as_nothing_sent(self) -> None:
        for reader in (runs.trio_bodies, runs.gate_bodies):
            with self.subTest(reader=reader.__name__), self.assertRaises(HarnessError):
                reader(None)

    @unittest.skipUnless(shutil.which("git") is not None, "git is required")
    def test_a_setup_commit_names_its_files_literally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            subprocess.run(runs.git_command(ws, "init", "-q"), check=True, env=runs.git_env())
            (ws / "w1.rs").write_bytes(b"stray")
            runs.commit_files(ws, runs.Commit(files=(("w[1].rs", b"fn b() {}\n"),), message="one"))
            listed = subprocess.run(runs.git_command(ws, "ls-files"), check=True, env=runs.git_env(),
                                    capture_output=True, text=True).stdout.split()
        self.assertEqual(["w[1].rs"], listed)

    def test_the_measured_review_has_one_reviews_allowance_whatever_setup_spent(self) -> None:
        for run_id in ("R21", "R22"):
            definition = dataclasses.replace(runs.DEFINITIONS[run_id], records_requests=False,
                                             unreachable_embedder=False)
            self.assertEqual(runs._HISTORY_SETUP_TIMEOUT_S, runs.measured_allowance(definition), run_id)
            executor = runs.RunExecutor(None, None, None)
            seen = []

            def burn(_definition, _executor=executor):
                _executor._clock_end = 0.0  # the whole shared clock is gone before the measured review
                return None

            with unittest.mock.patch.object(executor, "_prepare_review_workspace", burn), \
                    unittest.mock.patch.object(executor, "_stdin_for", return_value=b""), \
                    unittest.mock.patch.object(executor, "_capture_review_log", return_value=None), \
                    unittest.mock.patch.object(
                        executor, "_invoke",
                        lambda d, s, _seen=seen: _seen.append(d.timeout_s) or ProductOutput(
                            stdout=b"", stderr=b"", exit_code=0, command=())):
                executor.execute(definition)
            self.assertEqual([runs._HISTORY_SETUP_TIMEOUT_S], seen, run_id)

    def test_s57_a_finding_outside_the_seeded_file_is_cannot_test(self) -> None:
        for line in (0, 99):
            found = {"code_review": {"findings": [{"file": runs.REVIEW_HISTORY_FILE, "line": line,
                                                   "title": "off by one"}]}}
            first = ProductOutput(stdout=json.dumps(found).encode(), stderr=b"", exit_code=0, command=["m"])
            r = _with(_result("R21", 0, {}), request_bodies=(), setup_outputs=(first, first))
            self.assertEqual([P, C], _judged("S57", r, rc.S57_ASSERTIONS), line)

    def test_s56_a_missing_header_record_is_cannot_test_for_its_header_assertion(self) -> None:
        # MAGI FIX (MS3 §6 loop16, R7a2): `request_headers or ()` read a missing record as no header.
        chunk = ("/v1/embeddings", json.dumps({"input": [runs.REVIEW_COMMITTED_DOC.decode()]}).encode())
        r = _with(_result("R20", 0, {}), request_bodies=(chunk, _S56_TRIO), request_headers=None)
        self.assertEqual([P, P, C], _judged("S56", r, rc.S56_ASSERTIONS))

    def test_s56_names_what_its_third_assertion_checks(self) -> None:
        self.assertIn("embedded chunk", rc.S56_ASSERTIONS[2])
        self.assertNotIn("embedding request carries the parent", rc.S56_ASSERTIONS[2])

    def test_the_fixture_needles_are_disjoint(self) -> None:
        needles = (runs.REVIEW_COMMITTED_DOC, runs.REVIEW_PARENT_DOC, runs.REVIEW_DIRTY_DOC,
                   runs.REVIEW_MASKED_DOC, rc._ANSWER_CODE.encode())
        for first in needles:
            for second in needles:
                if first is not second:
                    self.assertNotIn(first, second)

    def test_the_readme_and_docstring_declare_what_the_scenarios_leave_open(self) -> None:
        root = pathlib.Path(runs.__file__).parent
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("RV-S37", readme)
        self.assertIn("RV-S39", readme)
        self.assertIn("**S58**", (root / "scenarios" / "review_context.py").read_text(encoding="utf-8"))

    def test_the_forbidden_docs_are_forbidden_as_chunks_not_as_the_reviewed_diff(self) -> None:
        # RV21 sends the old side of a replaced hunk; the fixture says so instead of forbidding it.
        doc = runs.REVIEW_PARENT_DOC.decode()
        source = pathlib.Path(runs.__file__).read_text(encoding="utf-8").replace("\r", "")
        constants = source.split("#: The doc comment of R18's file", 1)[1].split("REVIEW_DOCREV_FILE", 1)[0]
        constants = " ".join(constants.replace("#:", " ").split())
        self.assertIn("old side", constants)
        self.assertNotIn("no request may carry either", constants)
        self.assertTrue(doc)


P, F, C = Outcome.PASS, Outcome.FAIL, Outcome.CANNOT_TEST
# [MAGI S6 MS3 loop11, R7b Caspar] False positive: `scenario_result` has no `setup_outputs`
# parameter on purpose; S57's results add it with `dataclasses.replace` (`_with`), so the relay
# history subcases run and judge (10 tests, OK).
_with = dataclasses.replace


def _judged(sid: str, result, texts) -> list:
    """The outcomes of *sid* over *result*, in the order of *texts*."""
    outcomes = _outcomes(sid, result)
    return [outcomes[a] for a in texts]


#: A trio request carrying R20's reviewed commit's code (S56's second assertion reads it).
_S56_TRIO = ("/api/chat", b"+pub fn answer() -> u32 " + runs.REVIEW_COMMITTED_DOC)


class RelayScenarioTests(unittest.TestCase):

    def test_s53_reads_the_forwarded_requests(self) -> None:
        doc = {"code_review": {"masked": rm.S53_EXPECTED_MASKED, "masked_unlisted": 0,
                               "skipped": [], "skipped_unlisted": 0}}
        body = rm.code_review.key_body(runs.REVIEW_MASKED_KEY)
        part = body[3:3 + rm.MIN_LEAK_FRAGMENT]
        r = _with(_result("R18", 0, doc), request_bodies=(("/api/chat", b'"' + part + b'"'),))
        self.assertEqual([F], _judged("S53", r, rm.S53_ASSERTIONS[3:]))
        self.assertEqual([C], _judged("S53", _with(r, request_bodies=None), rm.S53_ASSERTIONS[3:]))
        # MAGI FIX (MS3 §6 loop11): a key in a forwarded request's headers fails too.
        clean = _with(r, request_bodies=R18_HEALTHY_REQUESTS)
        leaky = _with(clean, request_headers=(b"X-Key: " + part + b"\n",))
        self.assertEqual([F], _judged("S53", leaky, rm.S53_ASSERTIONS[3:]))
        # [MAGI S6 MS3 loop10, R7a Melchior] A file declared masked but never sent, or a key in a
        # request's query, passed.
        embeds = R18_HEALTHY_REQUESTS[:-1]
        unsent = embeds + (("/api/chat", b'{"messages": []}'),)
        queried = R18_HEALTHY_REQUESTS + (("/v1/embeddings?k=" + runs.REVIEW_MASKED_KEY.decode(), b""),)
        for bodies, want in ((R18_HEALTHY_REQUESTS, P), (unsent, F), (queried, F)):
            r = _with(_result("R18", 0, doc), request_bodies=bodies)
            self.assertEqual([want], _judged("S53", r, rm.S53_ASSERTIONS[3:]), bodies)

    def test_s53_reads_the_gate_requests_too(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7b Caspar): RV22 covers the gate's `state` (a finding with its
        # code, sent to `/systemone`). S53's oracle reads every forwarded request, so a gate body
        # with a window of the key fails it, and a masked state passes. R18's healthy record has
        # no gate request (whether the trio reports a finding is the model's answer), so this
        # exercises the oracle, not the product: the product side is
        # `a_state_carrying_a_one_line_secret_is_sent_masked` in `src/review/gate.rs`.
        doc = {"code_review": {"masked": rm.S53_EXPECTED_MASKED, "masked_unlisted": 0,
                               "skipped": [], "skipped_unlisted": 0}}
        part = rm.code_review.key_body(runs.REVIEW_MASKED_KEY)[3:3 + rm.MIN_LEAK_FRAGMENT]
        masked = b'{"state": {"code": "let k = <redacted: anthropic-key>;"}}'
        leaky = b'{"state": {"code": "let k = ' + part + b';"}}'
        for gate, want in ((masked, P), (leaky, F)):
            r = _with(_result("R18", 0, doc), request_bodies=R18_HEALTHY_REQUESTS + (("/v1/systemone", gate),))
            self.assertEqual([want], _judged("S53", r, rm.S53_ASSERTIONS[3:]), gate)

    def test_s55_needs_the_chunk_and_the_query(self) -> None:
        doc = {"code_review": {"without_embedder": False, "trimmed": []}}
        doc_text = runs.REVIEW_MASKED_DOC.decode()
        for bodies, want in ((R18_HEALTHY_REQUESTS, P), ((_chunk("pub fn token() " + doc_text),), F),
                             ((_chunk(doc_text),), F)):
            r = _with(_result("R18", 0, doc), request_bodies=bodies)
            self.assertEqual([want], _judged("S55", r, rc.S55_ASSERTIONS[2:]), bodies)

    def test_s56_never_reads_the_dirty_doc(self) -> None:
        good, dirty = _chunk(runs.REVIEW_COMMITTED_DOC.decode()), _chunk(runs.REVIEW_DIRTY_DOC.decode())
        for bodies, want in (((good, _S56_TRIO), [P, P, P]), ((good, _S56_TRIO, dirty), [P, P, F]),
                             ((), [P, F, P]), (None, [P, C, C]),
                             # [MAGI S6 MS3 loop15, R7a2] right docs, no trio request with the commit's code.
                             ((good,), [P, F, P])):
            r = _with(_result("R20", 0, {}), request_bodies=bodies)
            self.assertEqual(want, _judged("S56", r, rc.S56_ASSERTIONS), bodies)
        # ... and a dirty copy in a request header or target fails the third assertion too.
        clean = _with(_result("R20", 0, {}), request_bodies=(good, _S56_TRIO))
        for leaky in (_with(clean, request_headers=(b"X: " + runs.REVIEW_DIRTY_DOC,)),
                      _with(clean, request_bodies=(good, _S56_TRIO,
                                                   ("/q?d=" + runs.REVIEW_DIRTY_DOC.decode(), b"")))):
            self.assertEqual([P, P, F], _judged("S56", leaky, rc.S56_ASSERTIONS))

    def test_s56_tells_head_from_parent(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Melchior): both setup commits wrote one doc comment, so
        # documentation read from HEAD^ passed; the parent's differs now and must stay out.
        self.assertNotEqual(runs.REVIEW_PARENT_DOC, runs.REVIEW_COMMITTED_DOC)
        good, parent = _chunk(runs.REVIEW_COMMITTED_DOC.decode()), _chunk(runs.REVIEW_PARENT_DOC.decode())
        r = _with(_result("R20", 0, {}), request_bodies=(good, parent, _S56_TRIO))
        self.assertEqual([P, P, F], _judged("S56", r, rc.S56_ASSERTIONS))
        setup = runs.DEFINITIONS["R20"].setup
        self.assertIn(runs.REVIEW_PARENT_DOC, setup[0].files[0][1])
        self.assertIn(runs.REVIEW_COMMITTED_DOC, setup[-1].files[0][1])

    def test_s56_allows_the_parents_lines_in_the_reviewed_diff(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a Melchior): the parent's doc comment is the REMOVED line of
        # the diff the trio reviews (RV21 reads the diff from the reviewed commit), so it may sit
        # in a trio prompt; only an embedded chunk or the dirty copy is another revision's doc.
        good = _chunk(runs.REVIEW_COMMITTED_DOC.decode())
        diff = ("/api/chat", b"-    /// " + runs.REVIEW_PARENT_DOC)
        self.assertEqual([P, P, P], _judged("S56", _with(_result("R20", 0, {}),
                                                         request_bodies=(good, diff, _S56_TRIO)),
                                            rc.S56_ASSERTIONS))
        gate = ("/v1/systemone", runs.REVIEW_DIRTY_DOC)
        self.assertEqual([P, P, F], _judged("S56", _with(_result("R20", 0, {}),
                                                         request_bodies=(good, gate, _S56_TRIO)),
                                            rc.S56_ASSERTIONS), "the dirty copy is in no request at all")

    def test_s56_allows_the_parents_lines_in_the_retrieval_query(self) -> None:
        # MAGI FIX (MS3 §6 loop14, R7a2 Melchior): the retrieval query is an EMBEDDING request
        # built from the batch's raw hunk (`batch_text` in src/memory/review_rag.rs), removed
        # lines included, so it carries the parent's doc comment beside the batch's code. Only a
        # chunk (the doc comment without that code) is another revision's documentation.
        good = _chunk(runs.REVIEW_COMMITTED_DOC.decode())
        query = _chunk("search_query: -    /// " + runs.REVIEW_PARENT_DOC.decode() + "\n+pub fn answer() {}")
        self.assertEqual([P, P, P], _judged("S56", _with(_result("R20", 0, {}),
                                                         request_bodies=(good, query, _S56_TRIO)),
                                            rc.S56_ASSERTIONS))

    def test_trio_bodies_never_hold_gate_requests(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a Caspar/Melchior): the oracle of S53 and the history check
        # of S57 read the trio's prompts; the gate's `state` (a finding with its code) goes to
        # `/systemone` and must satisfy and trip neither.
        record = (("/api/chat", b"t1"), ("/v1/chat/completions?x=1", b"t2"),
                  ("/v1/systemone", b"g1"), ("/v1/systemone?m=1", b"g2"), ("/v1/embeddings", b"e"))
        self.assertEqual((b"t1", b"t2"), runs.trio_bodies(record))
        self.assertEqual((b"g1", b"g2"), runs.gate_bodies(record))
        self.assertEqual((), runs.trio_bodies(()))
        self.assertEqual((), runs.gate_bodies(()))

    def test_embedded_texts_read_the_path_alone_and_a_single_string(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a): a recorded target carries its query, and the
        # OpenAI-compatible `input` may be one string.
        query = ("/v1/embeddings?encoding_format=float", json.dumps({"input": ["a"]}).encode())
        single = ("/v1/embeddings", json.dumps({"input": "b"}).encode())
        other = ("/v1/systemone", json.dumps({"input": ["c"]}).encode())
        self.assertEqual(["a", "b"], rc._embedded_texts((query, single, other)))

    def test_a_result_without_output_is_a_missing_capture_not_a_crash(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a Melchior): S52, S53, S55 and S56 read `run.output.*`
        # unguarded; S57 alone had the guard.
        for sid, run_id, texts in (("S52", "R17", rc.S52_ASSERTIONS), ("S53", "R18", rm.S53_ASSERTIONS),
                                   ("S55", "R18", rc.S55_ASSERTIONS), ("S56", "R20", rc.S56_ASSERTIONS)):
            with self.subTest(sid=sid):
                r = _with(_result(run_id, 0, {}), output=None)
                self.assertTrue(all(o is C for o in _judged(sid, r, texts)), sid)

    def test_s53_needs_the_header_record(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7b): bodies without their headers is an incomplete record,
        # pinned like missing bodies (CANNOT_TEST), never "the headers held no key".
        doc = {"code_review": {"masked": rm.S53_EXPECTED_MASKED, "masked_unlisted": 0,
                               "skipped": [], "skipped_unlisted": 0}}
        run = _result("R18", 0, doc)
        self.assertEqual([P], _judged("S53", run, rm.S53_ASSERTIONS[3:]))
        self.assertEqual([C], _judged("S53", _with(run, request_headers=None), rm.S53_ASSERTIONS[3:]))

    def test_the_setup_never_records(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7b): the record a scenario reads is the measured run's alone.
        # S57's oracle looks for a history frame the setup reviews would also produce.
        executor = runs.RunExecutor(binary=None, env={}, config=None)
        step = runs.Invocation(argv=("code-review",), timeout_s=5)
        definition = runs.RunDefinition(
            run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS, stdin=b"",
            needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=5 + runs.MIN_MEASURED_ALLOWANCE_S,
            records_requests=True, setup=(step,))
        seen = []
        with unittest.mock.patch.object(executor, "_invoke", lambda d, s: seen.append(d)):
            executor._run_setup(definition, pathlib.Path("."))
        self.assertEqual([False], [d.records_requests for d in seen])

    def test_s54_without_output_is_three_findings(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7a Melchior): the leak check read `run.output.*` on a result
        # that left no capture; S54 yields its three findings, as its five siblings do.
        r = _with(_result("R19", 0, {}), output=None)
        outcomes = _outcomes("S54", r)
        self.assertEqual(3, len(outcomes))

    def test_s57_serves_validated_history(self) -> None:
        found = {"code_review": {"findings": [{"file": runs.REVIEW_HISTORY_FILE, "line": 3,
                                               "title": "off by one"}]}}
        first = ProductOutput(stdout=json.dumps(found).encode(), stderr=b"", exit_code=0, command=["m"])
        empty = _with(first, stdout=b'{"code_review": {"findings": []}}')
        frame = b"<<<context kind=history file=%s>>>" % runs.REVIEW_HISTORY_FILE.encode()
        served = (("/api/chat", frame + b"\\noff by one (attended)\\n"),)
        # [MAGI S6 MS3 loop10, R7a Melchior] A frame to another endpoint, an empty or
        # unvalidated item, or another finding's text passed.
        elsewhere = (("/v1/embeddings", served[0][1]),)
        bare = (("/api/chat", frame),)
        pending = (("/api/chat", frame + b"\\noff by one (pending)\\n"),)
        other = (("/api/chat", frame + b"\\nanother (attended)\\n"),)
        for setup, bodies, want in (((first, first), served, [P, P]), ((first, first), (), [P, F]),
                                    ((empty, first), served, [P, C]),
                                    ((first, _with(first, exit_code=1)), served, [F, P]),
                                    ((first, first), elsewhere, [P, F]), ((first, first), bare, [P, F]),
                                    ((first, first), pending, [P, F]), ((first, first), other, [P, F])):
            r = _with(_result("R21", 0, {}), request_bodies=bodies, setup_outputs=setup)
            self.assertEqual(want, _judged("S57", r, rc.S57_ASSERTIONS), bodies)
        r = _with(_result("R21", 0, {}), request_bodies=served, setup_outputs=(first, first), output=None)
        self.assertEqual([C, C], _judged("S57", r, rc.S57_ASSERTIONS))

    def test_s57_a_malformed_findings_value_is_cannot_test_not_a_crash(self) -> None:
        # MAGI FIX (MS3 §6 loop14, R7a2 Caspar): `findings or ()` iterated a non-list value.
        for bad in (5, True, "x", {"file": "f"}):
            first = ProductOutput(stdout=json.dumps({"code_review": {"findings": bad}}).encode(),
                                  stderr=b"", exit_code=0, command=["m"])
            r = _with(_result("R21", 0, {}), request_bodies=(), setup_outputs=(first, first))
            self.assertEqual([P, C], _judged("S57", r, rc.S57_ASSERTIONS), bad)

    def test_s57_matches_utf8_and_separators(self) -> None:
        # MAGI FIX (MS3 §6 loop13, R7a): the product's JSON body keeps non-ASCII characters
        # (serde_json does not escape them) and may write the path with either separator.
        title = "décalage à l'index"
        found = {"code_review": {"findings": [{"file": runs.REVIEW_HISTORY_FILE, "line": 3, "title": title}]}}
        first = ProductOutput(stdout=json.dumps(found).encode(), stderr=b"", exit_code=0, command=["m"])
        item = ("\\n%s (attended)\\n" % title).encode("utf-8")
        path = runs.REVIEW_HISTORY_FILE
        for shown in (path, path.replace("/", "\\"), path.replace("/", "\\\\")):
            frame = b"<<<context kind=history file=%s>>>" % shown.encode()
            r = _with(_result("R21", 0, {}), request_bodies=(("/api/chat", frame + item),),
                      setup_outputs=(first, first))
            self.assertEqual([P, P], _judged("S57", r, rc.S57_ASSERTIONS), shown)


class LoopSeventeenClockTests(unittest.TestCase):
    """MAGI FIX (MS3 §6 loop17, R7a1/R7b1): a run's clock and what it declares."""

    @staticmethod
    def _with_setup(timeout_s: int) -> "runs.RunDefinition":
        step = runs.Invocation(argv=("code-review",), timeout_s=900)
        return runs.RunDefinition(
            run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS, stdin=b"",
            needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=timeout_s, setup=(step, step))

    def test_a_setup_that_leaves_the_measured_review_too_little_is_a_harness_error(self) -> None:
        for timeout_s in (2 * 900 - 1, 2 * 900, 2 * 900 + runs.MIN_MEASURED_ALLOWANCE_S - 1):
            with self.subTest(timeout_s=timeout_s), self.assertRaises(HarnessError):
                self._with_setup(timeout_s)
        enough = self._with_setup(2 * 900 + runs.MIN_MEASURED_ALLOWANCE_S)
        self.assertEqual(runs.MIN_MEASURED_ALLOWANCE_S, runs.measured_allowance(enough))

    def test_r21_and_r22_cannot_outrun_their_declared_clock(self) -> None:
        # R7b1: the worst case is every invocation, setup and measured, running to the timeout it
        # is granted. The declared `timeout_s` (three reviews) must bound it, not be its floor.
        for run_id in ("R21", "R22"):
            definition = dataclasses.replace(runs.DEFINITIONS[run_id], records_requests=False,
                                             unreachable_embedder=False)
            clock, granted = [0.0], []
            executor = runs.RunExecutor(None, None, None)

            def invoke(d, _stdin, _clock=clock, _granted=granted):
                _granted.append(d.timeout_s)
                _clock[0] += d.timeout_s
                return ProductOutput(stdout=b"", stderr=b"", exit_code=0, command=())

            def prepare(d, _executor=executor):
                _executor._run_setup(d, pathlib.Path("."))

            with unittest.mock.patch.object(runs.time, "monotonic", lambda _clock=clock: _clock[0]), \
                    unittest.mock.patch.object(runs, "commit_files"), \
                    unittest.mock.patch.object(executor, "_prepare_review_workspace", prepare), \
                    unittest.mock.patch.object(executor, "_stdin_for", return_value=b""), \
                    unittest.mock.patch.object(executor, "_capture_review_log", return_value=None), \
                    unittest.mock.patch.object(executor, "_command", return_value=()), \
                    unittest.mock.patch.object(executor, "_invoke", invoke):
                result = executor.execute(definition)
            self.assertFalse(result.timed_out, (run_id, granted))
            self.assertEqual(3, len(granted), run_id)
            self.assertLessEqual(sum(granted), definition.timeout_s, (run_id, granted))
            self.assertEqual(runs.measured_allowance(definition), granted[-1], run_id)
        # The measured review's allowance is one review like R20's, so the declared 3 x R20 holds.
        self.assertEqual(runs.DEFINITIONS["R20"].timeout_s, runs.measured_allowance(runs.DEFINITIONS["R21"]))


if __name__ == "__main__":
    unittest.main()
