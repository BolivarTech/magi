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

    def test_an_upstream_with_credentials_is_refused(self) -> None:
        with self.assertRaises(HarnessError):
            with runs.recording_relay("http://u:p@127.0.0.1:1/v1"):
                pass

    def test_the_upstream_is_the_one_endpoint_the_config_declares(self) -> None:
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

    def test_the_commit_and_history_runs_record_their_requests(self) -> None:
        for run_id in ("R18", "R20", "R21"):
            self.assertTrue(runs.DEFINITIONS[run_id].records_requests, run_id)
        self.assertEqual(2, sum(isinstance(s, runs.Invocation) for s in runs.DEFINITIONS["R21"].setup))


class RelayScenarioTests(unittest.TestCase):

    def test_s53_fails_when_a_forwarded_request_carries_a_window_of_the_key(self) -> None:
        doc = {"code_review": {"masked": rm.S53_EXPECTED_MASKED, "masked_unlisted": 0,
                               "skipped": [], "skipped_unlisted": 0}}
        part = runs.REVIEW_MASKED_KEY[3:3 + rm.LEAK_SLICE_BYTES]
        r = dataclasses.replace(_result("R18", 0, doc), request_bodies=(("/api/chat", b'{"c":"' + part + b'"}'),))
        self.assertEqual(Outcome.FAIL, _outcomes("S53", r)[rm.S53_ASSERTIONS[3]])
        unrecorded = dataclasses.replace(r, request_bodies=None)
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S53", unrecorded)[rm.S53_ASSERTIONS[3]])

    def test_s55_needs_the_doc_comment_indexed_and_a_query_embedded(self) -> None:
        doc = {"code_review": {"without_embedder": False, "trimmed": []}}
        query = _chunk("pub fn token() " + runs.REVIEW_MASKED_DOC.decode())
        for bodies, want in ((R18_HEALTHY_REQUESTS, Outcome.PASS), ((query,), Outcome.FAIL),
                             ((_chunk(runs.REVIEW_MASKED_DOC.decode()),), Outcome.FAIL)):
            r = dataclasses.replace(_result("R18", 0, doc), request_bodies=bodies)
            self.assertEqual(want, _outcomes("S55", r)[rc.S55_ASSERTIONS[2]], bodies)

    def test_s56_reads_the_committed_doc_and_never_the_dirty_one(self) -> None:
        committed = _chunk(runs.REVIEW_COMMITTED_DOC.decode())
        dirty = _chunk(runs.REVIEW_DIRTY_DOC.decode())
        cases = (((committed,), [Outcome.PASS, Outcome.PASS, Outcome.PASS]),
                 ((committed, dirty), [Outcome.PASS, Outcome.PASS, Outcome.FAIL]),
                 ((), [Outcome.PASS, Outcome.FAIL, Outcome.PASS]),
                 (None, [Outcome.PASS, Outcome.CANNOT_TEST, Outcome.CANNOT_TEST]))
        for bodies, want in cases:
            r = dataclasses.replace(_result("R20", 0, {}), request_bodies=bodies)
            self.assertEqual(want, [_outcomes("S56", r)[a] for a in rc.S56_ASSERTIONS], bodies)

    def test_s57_serves_the_validated_history_or_says_it_could_not(self) -> None:
        located = {"code_review": {"findings": [{"file": runs.REVIEW_HISTORY_FILE, "line": 3}]}}
        first = ProductOutput(stdout=json.dumps(located).encode(), stderr=b"", exit_code=0, command=["m"])
        empty = dataclasses.replace(first, stdout=json.dumps({"code_review": {"findings": []}}).encode())
        served = (("/api/chat", b'"<<<context kind=history file=' + runs.REVIEW_HISTORY_FILE.encode() + b'>>>"'),)
        cases = (((first, first), served, [Outcome.PASS, Outcome.PASS]),
                 ((first, first), (), [Outcome.PASS, Outcome.FAIL]),
                 ((empty, first), served, [Outcome.PASS, Outcome.CANNOT_TEST]),
                 ((first, dataclasses.replace(first, exit_code=1)), served, [Outcome.FAIL, Outcome.PASS]))
        for setup, bodies, want in cases:
            r = dataclasses.replace(_result("R21", 0, {}), request_bodies=bodies, setup_outputs=setup)
            self.assertEqual(want, [_outcomes("S57", r)[a] for a in rc.S57_ASSERTIONS])


if __name__ == "__main__":
    unittest.main()
