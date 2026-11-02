# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-02
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


if __name__ == "__main__":
    unittest.main()
