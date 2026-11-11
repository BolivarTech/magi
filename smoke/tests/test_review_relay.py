# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""The recording relay, the setup steps and S53/S55-S57 over the relay's record (MAGI S6 MS3 loop9, R7a)."""

import dataclasses
import http.server
import json
import pathlib
import subprocess
import tempfile
import threading
import unittest
import unittest.mock
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


class RelayTests(unittest.TestCase):

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

    def test_a_setup_commit_records_exactly_its_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            subprocess.run(runs.git_command(ws, "init", "-q"), check=True, env=runs.git_env())
            (ws / "stray.txt").write_bytes(b"not committed")
            runs.commit_files(ws, runs.Commit(files=(("a/b.rs", b"fn b() {}\n"),), message="one"))
            listed = subprocess.run(runs.git_command(ws, "ls-files"), check=True, env=runs.git_env(),
                                    capture_output=True, text=True).stdout.split()
        self.assertEqual(["a/b.rs"], listed)


P, F, C = Outcome.PASS, Outcome.FAIL, Outcome.CANNOT_TEST
_with = dataclasses.replace


def _judged(sid: str, result, texts) -> list:
    """The outcomes of *sid* over *result*, in the order of *texts*."""
    outcomes = _outcomes(sid, result)
    return [outcomes[a] for a in texts]


class RelayScenarioTests(unittest.TestCase):

    def test_s53_reads_the_forwarded_requests(self) -> None:
        doc = {"code_review": {"masked": rm.S53_EXPECTED_MASKED, "masked_unlisted": 0,
                               "skipped": [], "skipped_unlisted": 0}}
        part = runs.REVIEW_MASKED_KEY[3:3 + rm.LEAK_SLICE_BYTES]
        r = _with(_result("R18", 0, doc), request_bodies=(("/api/chat", b'"' + part + b'"'),))
        self.assertEqual([F], _judged("S53", r, rm.S53_ASSERTIONS[3:]))
        self.assertEqual([C], _judged("S53", _with(r, request_bodies=None), rm.S53_ASSERTIONS[3:]))
        # [MAGI S6 MS3 loop10, R7a Melchior] A file declared masked but never sent, or a key in a
        # request's query, passed.
        embeds = R18_HEALTHY_REQUESTS[:-1]
        unsent = embeds + (("/api/chat", b'{"messages": []}'),)
        queried = R18_HEALTHY_REQUESTS + (("/v1/embeddings?k=" + runs.REVIEW_MASKED_KEY.decode(), b""),)
        for bodies, want in ((R18_HEALTHY_REQUESTS, P), (unsent, F), (queried, F)):
            r = _with(_result("R18", 0, doc), request_bodies=bodies)
            self.assertEqual([want], _judged("S53", r, rm.S53_ASSERTIONS[3:]), bodies)

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
