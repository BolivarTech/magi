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

    def test_setup_clears_the_embedder_rewrite(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Balthasar): a setup invocation inherited the run's flags.
        seen = []
        executor = runs.RunExecutor(binary=None, env={}, config=None)
        definition = runs.RunDefinition(
            run_id="RX", argv=("code-review", runs.REVIEW_WORKSPACE_TOKEN) + runs.REVIEW_FLAGS,
            stdin=b"", needs_trio=True, payload_size=runs.PAYLOAD_SMALL, timeout_s=60,
            unreachable_embedder=True, setup=(runs.Invocation(argv=("code-review",), timeout_s=5),))
        with unittest.mock.patch.object(executor, "_invoke", lambda d, s: seen.append(d)):
            executor._run_setup(definition, pathlib.Path("."))
        self.assertEqual([False], [d.unreachable_embedder for d in seen])

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


P, F, C = Outcome.PASS, Outcome.FAIL, Outcome.CANNOT_TEST
# [MAGI S6 MS3 loop11, R7b Caspar] False positive: `scenario_result` has no `setup_outputs`
# parameter on purpose; S57's results add it with `dataclasses.replace` (`_with`), so the relay
# history subcases run and judge (10 tests, OK).
_with = dataclasses.replace


def _judged(sid: str, result, texts) -> list:
    """The outcomes of *sid* over *result*, in the order of *texts*."""
    outcomes = _outcomes(sid, result)
    return [outcomes[a] for a in texts]


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
        for bodies, want in (((good,), [P, P, P]), ((good, dirty), [P, P, F]), ((), [P, F, P]),
                             (None, [P, C, C])):
            r = _with(_result("R20", 0, {}), request_bodies=bodies)
            self.assertEqual(want, _judged("S56", r, rc.S56_ASSERTIONS), bodies)

    def test_s56_tells_head_from_parent(self) -> None:
        # MAGI FIX (MS3 §6 loop11, R7a Melchior): both setup commits wrote one doc comment, so
        # documentation read from HEAD^ passed; the parent's differs now and must stay out.
        self.assertNotEqual(runs.REVIEW_PARENT_DOC, runs.REVIEW_COMMITTED_DOC)
        good, parent = _chunk(runs.REVIEW_COMMITTED_DOC.decode()), _chunk(runs.REVIEW_PARENT_DOC.decode())
        r = _with(_result("R20", 0, {}), request_bodies=(good, parent))
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
        self.assertEqual([P, P, P], _judged("S56", _with(_result("R20", 0, {}), request_bodies=(good, diff)),
                                            rc.S56_ASSERTIONS))
        gate = ("/v1/systemone", runs.REVIEW_DIRTY_DOC)
        self.assertEqual([P, P, F], _judged("S56", _with(_result("R20", 0, {}), request_bodies=(good, gate)),
                                            rc.S56_ASSERTIONS), "the dirty copy is in no request at all")

    def test_trio_bodies_never_hold_gate_requests(self) -> None:
        # MAGI FIX (MS3 §6 loop12, R7a Caspar/Melchior): the oracle of S53 and the history check
        # of S57 read the trio's prompts; the gate's `state` (a finding with its code) goes to
        # `/systemone` and must satisfy and trip neither.
        record = (("/api/chat", b"t1"), ("/v1/chat/completions?x=1", b"t2"),
                  ("/v1/systemone", b"g1"), ("/v1/systemone?m=1", b"g2"), ("/v1/embeddings", b"e"))
        self.assertEqual((b"t1", b"t2"), runs.trio_bodies(record))
        self.assertEqual((b"g1", b"g2"), runs.gate_bodies(record))
        self.assertEqual((), runs.trio_bodies(None))
        self.assertEqual((), runs.gate_bodies(None))

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


if __name__ == "__main__":
    unittest.main()
