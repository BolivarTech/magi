# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-18
"""Unit tests for the loopback stand-ins the product is pointed at.

[MAGI §6 MS2 loop 29, F1a1b-melchior] R6 points the product at a loopback
error backend, but only the refusal runs took the proxy out of its way, so on
a proxied host the backend could see nothing and S10 search output that never
walked the path it guards; nothing recorded whether the backend was reached.
And a Content-Length longer than Python's integer-conversion limit raised a
``ValueError`` neither handler caught.
"""

import os
import socket
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock

from smoke import binary, runs
from smoke.binary import ReleaseBinary
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.scenarios import redaction
from smoke.secrets import PlantedSecret
from smoke.tests import support


def _port(url: str) -> int:
    """The port of a loopback stand-in's base URL."""
    return urllib.parse.urlsplit(url).port


def _first_bytes(port: int, data: bytes) -> bytes:
    """Send *data* on one connection to *port*, half-close, and return what came back."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
        client.sendall(data)
        client.shutdown(socket.SHUT_WR)
        chunks = []
        try:
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except OSError:
            pass
        return b"".join(chunks)


#: A Content-Length past Python's default 4300-digit integer-conversion limit.
_HUGE_LENGTH = b"9" * 5000


class OversizedContentLengthTests(unittest.TestCase):
    """A decimal Content-Length too long to convert is answered, never a handler traceback."""

    def test_the_drain_does_not_raise_on_an_over_long_decimal_length(self) -> None:
        import io
        runs._drain_body(io.BytesIO(b"{}"), {"Content-Length": _HUGE_LENGTH.decode()})

    def test_the_error_backend_answers_an_over_long_length(self) -> None:
        with runs.error_backend() as url:
            answer = _first_bytes(_port(url), b"POST /v1/chat HTTP/1.1\r\nHost: x\r\nContent-Length: "
                                  + _HUGE_LENGTH + b"\r\n\r\n{}")
        self.assertTrue(answer.startswith(b"HTTP/1.0 %d" % runs.ERROR_BACKEND_STATUS), answer[:80])

    def test_the_counter_answers_and_records_an_over_long_length(self) -> None:
        with runs.model_counter() as (url, log):
            answer = _first_bytes(_port(url), b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: "
                                  + _HUGE_LENGTH + b"\r\n\r\n{}")
        self.assertTrue(answer.startswith(b"HTTP/1.0 %d" % runs.MODEL_COUNTER_STATUS), answer[:80])
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))

    def test_both_stand_ins_close_after_every_answer(self) -> None:
        # Why an unframed or chunked body cannot desynchronise a later request: there is none.
        self.assertEqual("HTTP/1.0", runs._EchoingHandler.protocol_version)
        self.assertEqual("HTTP/1.0", runs._CountingHandler.protocol_version)


class LoopbackExemptTests(unittest.TestCase):
    """Every product invocation reaches a loopback stand-in directly, whatever proxy the host sets."""

    def test_loopback_is_added_to_an_inherited_no_proxy(self) -> None:
        environment = binary.exempt_loopback({"no_proxy": "corp.example", "HTTP_PROXY": "http://p:3128"})
        self.assertEqual("http://p:3128", environment["HTTP_PROXY"])
        self.assertNotIn("no_proxy", environment)
        entries = environment["NO_PROXY"].split(",")
        self.assertIn("corp.example", entries)
        for host in binary.LOOPBACK_HOSTS:
            self.assertIn(host, entries)

    def test_no_inherited_no_proxy_still_exempts_loopback(self) -> None:
        environment = binary.exempt_loopback({"PATH": "p"})
        self.assertEqual(",".join(binary.LOOPBACK_HOSTS), environment["NO_PROXY"])
        self.assertEqual("p", environment["PATH"])

    def test_a_loopback_entry_is_not_repeated(self) -> None:
        environment = binary.exempt_loopback({"NO_PROXY": "127.0.0.1, x"})
        self.assertEqual(1, environment["NO_PROXY"].split(",").count("127.0.0.1"))

    def test_the_release_binary_hands_the_child_a_loopback_exempt_environment(self) -> None:
        seen = {}

        def run(command, **kwargs):
            seen.update(kwargs["env"])
            return mock.Mock(stdout=b"", stderr=b"", returncode=0)

        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://p:3128", "NO_PROXY": "corp.example"}), \
                mock.patch.object(binary.subprocess, "run", side_effect=run):
            ReleaseBinary(".").invoke(["query"], timeout=5)
        entries = seen["NO_PROXY"].split(",")
        self.assertIn("corp.example", entries)
        self.assertIn("127.0.0.1", entries)


class ErrorBackendReachedTests(unittest.TestCase):
    """R6 records how many requests its error backend answered; S10 needs at least one."""

    def test_the_error_backend_counts_what_it_answers(self) -> None:
        log = runs.RequestLog()
        with runs.error_backend(log) as url:
            request = urllib.request.Request(url + "/chat/completions", data=b"{}", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                support.loopback_open(request)
            caught.exception.close()
        self.assertEqual(1, log.total)

    def _execute_r6(self, contact: bool):
        def responder(call):
            if list(call.args[:1]) == ["vault"]:
                return support.vault_succeeds(call)
            if contact:
                # The URL carries "[user]:[password]@" placeholders, which urlsplit refuses as a host.
                url = "http://" + call.env["OPENAI_BASE_URL"].rsplit("@", 1)[1] + "/chat/completions"
                try:
                    support.loopback_open(urllib.request.Request(url, data=b"{}", method="POST")).close()
                except urllib.error.HTTPError as exc:
                    exc.close()
            return ProductOutput(stdout=b"", stderr=b"", exit_code=1, command=["magi-rs"])

        support.install_fake_runs(self, responder=responder)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config, credential="the-real-credential")
        return executor.execute(runs.DEFINITIONS["R6"])

    def test_r6_records_a_reached_backend(self) -> None:
        self.assertEqual(1, self._execute_r6(contact=True).backend_requests)

    def test_r6_records_a_backend_nobody_reached(self) -> None:
        self.assertEqual(0, self._execute_r6(contact=False).backend_requests)

    def test_a_run_without_the_backend_records_nothing(self) -> None:
        support.install_fake_runs(self, responder=support.vault_succeeds)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config, credential="the-real-credential")
        self.assertIsNone(executor.execute(runs.DEFINITIONS["R1"]).backend_requests)

    def test_s10_cannot_test_when_the_backend_saw_no_request(self) -> None:
        run = mock.Mock(output=ProductOutput(stdout=b"", stderr=b"", exit_code=1, command=["magi-rs"]),
                        planted=(), timed_out=False, run_id="R6", backend_requests=0)
        findings = list(redaction.no_credential_leaks(run))
        self.assertEqual(len(redaction.ASSERTIONS), len(findings))
        self.assertTrue(all(f.outcome is Outcome.CANNOT_TEST for f in findings), findings)
        self.assertIn("error backend", findings[0].detail)

    def test_s10_fails_a_leak_even_when_the_backend_saw_no_request(self) -> None:
        """[F1a2] an unanswered backend downgrades a clean search, never a leak.

        A product that leaks the planted credential in a transport error -- the backend never answered, so
        nothing reached the 401 path -- still printed the secret. S10 used to return CANNOT_TEST for every
        assertion before searching, so that leak read as untested instead of FAIL.
        """
        secret = PlantedSecret("sup3r@s3cr3t/value#1", "backend credential")
        stderr = b"error: connect to https://u:sup3r@s3cr3t/value#1@host/v1 refused"
        run = mock.Mock(output=ProductOutput(stdout=b"", stderr=stderr, exit_code=1, command=["magi-rs"]),
                        planted=(secret,), timed_out=False, run_id="R6", backend_requests=0)
        outcomes = [f.outcome for f in redaction.no_credential_leaks(run)]
        self.assertEqual(len(redaction.ASSERTIONS), len(outcomes))
        self.assertIs(Outcome.FAIL, outcomes[0], outcomes)
        self.assertIs(Outcome.FAIL, outcomes[3], outcomes)
        self.assertIs(Outcome.CANNOT_TEST, outcomes[1], outcomes)


class _BrokenWriter:
    """A response stream whose every write fails, as when the client hangs up before the answer."""

    def write(self, data: bytes) -> int:
        raise ConnectionAbortedError("the client hung up")

    def flush(self) -> None:
        raise ConnectionAbortedError("the client hung up")


class _FlushFailingWriter:
    """A response stream that takes every write and fails only on flush, as when the client hangs up last."""

    def __init__(self) -> None:
        self.written = b""

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def flush(self) -> None:
        raise ConnectionAbortedError("the client hung up")


class EchoingHandlerHardeningTests(unittest.TestCase):
    """[MAGI F1a1b] the error backend records and answers whatever arrives, as the counter does.

    The echoing handler recorded from ``do_POST``: a request line that did not parse never reached it and
    went unrecorded, a target ``urlsplit`` refuses (``http://[x``) raised before the record, and a write to a
    client that had hung up raised out of the handler.
    """

    def _handler(self, request: bytes, log, wfile=None):
        import io
        import types
        handler = runs._EchoingHandler.__new__(runs._EchoingHandler)
        handler.rfile = io.BufferedReader(io.BytesIO(request))
        handler.wfile = io.BytesIO() if wfile is None else wfile
        handler.server = types.SimpleNamespace(log=log)
        handler.client_address = ("127.0.0.1", 0)
        # [F1b2c] Preset to the value the handler must CHANGE, so an assertion that it closed the connection
        # reads what the handler did and not what this helper started it with.
        handler.close_connection = False
        return handler

    def test_a_request_line_that_does_not_parse_is_recorded(self) -> None:
        log = runs.RequestLog()
        with runs.error_backend(log) as url:
            _first_bytes(_port(url), b"NOT A REQUEST LINE AT ALL\r\n\r\n")
        self.assertEqual((runs.UNPARSED_REQUEST,), log.recorded)

    def test_a_target_that_does_not_split_is_recorded_unparsed_and_answered(self) -> None:
        log = runs.RequestLog()
        with runs.error_backend(log) as url:
            # An absolute-form target with an unclosed IPv6 bracket: urlsplit raises ValueError on it.
            answer = _first_bytes(_port(url), b"POST http://[x/v1 HTTP/1.0\r\nContent-Length: 0\r\n\r\n")
        self.assertEqual((("POST", runs.UNPARSED_REQUEST[1]),), log.recorded)
        self.assertTrue(answer.startswith(b"HTTP/1.0 %d" % runs.ERROR_BACKEND_STATUS), answer[:80])

    def test_a_client_gone_before_the_answer_raises_nothing(self) -> None:
        log = runs.RequestLog()
        handler = self._handler(b"POST /v1/chat HTTP/1.0\r\nContent-Length: 0\r\n\r\n", log, _BrokenWriter())
        try:
            handler.handle_one_request()
        except OSError as exc:
            self.fail("the answer's write failure escaped the handler: %r" % exc)
        self.assertEqual((("POST", "/v1/chat"),), log.recorded)
        self.assertTrue(handler.close_connection)

    def test_a_flush_that_fails_after_the_answer_closes_the_connection(self) -> None:
        """[F1b2c] the flush's own guard, isolated from the two that also close.

        An HTTP/1.0 parse and a failed write in ``_answer`` both set ``close_connection`` already, so the
        test above stays green with the flush guard removed. Here the request is HTTP/1.1 on an HTTP/1.1
        handler (the parse keeps the connection open) and only the flush fails.
        """
        log = runs.RequestLog()
        handler = self._handler(b"POST /v1/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n", log,
                                _FlushFailingWriter())
        handler.protocol_version = "HTTP/1.1"
        try:
            handler.handle_one_request()
        except OSError as exc:
            self.fail("the flush failure escaped the handler: %r" % exc)
        self.assertEqual((("POST", "/v1/chat"),), log.recorded)
        self.assertTrue(handler.close_connection)


class HandlerThreadBoundTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1a1b-caspar] a flood of connections cannot spawn handler threads without bound.

    ThreadingHTTPServer starts one thread per connection; each lives until its deadline, so a product that
    opened connections faster than they closed grew the harness's thread count with nothing to stop it.
    Past the cap a connection is served on the accepting thread: accepting slows down, every request is
    still answered and counted.
    """

    _REQUEST = b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"

    def test_both_stand_ins_bound_their_handler_threads(self) -> None:
        self.assertTrue(issubclass(runs._BoundedThreadingServer, runs.http.server.ThreadingHTTPServer))
        self.assertGreater(runs.MAX_LOOPBACK_HANDLER_THREADS, 0)
        with mock.patch.object(runs.http.server, "ThreadingHTTPServer",
                               side_effect=AssertionError("an unbounded server")):
            with runs.error_backend():
                pass
            with runs.model_counter():
                pass

    def test_the_decide_silent_endpoint_bounds_its_handler_threads_too(self) -> None:
        # Swept: D10's listener parks one thread per connection until it shuts down.
        from smoke.scenarios import decide
        self.assertTrue(issubclass(decide._SilentServer, runs.BoundedThreadsMixIn))
        self.assertTrue(issubclass(runs._BoundedThreadingServer, runs.BoundedThreadsMixIn))
        with decide._silent_endpoint() as url:
            with socket.create_connection(("127.0.0.1", _port(url)), timeout=5):
                pass

    def test_past_the_cap_a_request_is_still_answered_and_counted_without_a_thread(self) -> None:
        import socketserver
        with mock.patch.object(runs, "MAX_LOOPBACK_HANDLER_THREADS", 0), \
                mock.patch.object(socketserver.ThreadingMixIn, "process_request",
                                  side_effect=AssertionError("a thread was spawned past the cap")):
            with runs.model_counter() as (url, log):
                answer = _first_bytes(_port(url), self._REQUEST)
        self.assertTrue(answer.startswith(b"HTTP/1.0 %d" % runs.MODEL_COUNTER_STATUS), answer[:80])
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))

    def test_a_finished_thread_gives_its_slot_back(self) -> None:
        import socketserver
        with mock.patch.object(runs, "MAX_LOOPBACK_HANDLER_THREADS", 1):
            server = runs._BoundedThreadingServer(("127.0.0.1", 0), runs._CountingHandler)
        try:
            with mock.patch.object(socketserver.ThreadingMixIn, "process_request"):
                server.process_request(None, None)
            self.assertFalse(server._slots.acquire(blocking=False), "the only slot was not taken")
            with mock.patch.object(socketserver.ThreadingMixIn, "process_request_thread"):
                server.process_request_thread(None, None)
            self.assertTrue(server._slots.acquire(blocking=False), "the ended thread kept its slot")
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
