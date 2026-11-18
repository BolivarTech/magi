# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-18
"""Unit tests for the probes of S56 and S57 (the reviewed commit's docs and the served history)."""

import types
import unittest

from smoke import runs
from smoke.scenarios import review_context

_PATH = runs.REVIEW_HISTORY_FILE
_TRIO = "/api/chat"
_FRAME = ("<<<context kind=history file=%s>>>" % _PATH).encode()


def _located(*titles: str) -> list:
    """A located finding in the product's shape: ``titles`` is a list (``with_gate``)."""
    return [{"file": _PATH, "line": 3, "titles": list(titles)}]


def _body(*items: str) -> bytes:
    return b"".join(_FRAME + item.encode() for item in items) + b"<<<end>>>"


class HistoryServedTest(unittest.TestCase):
    """``_history_served`` reads every history frame and cuts only at the next frame."""

    def test_a_second_frame_of_the_same_file_is_read(self):
        bodies = ((_TRIO, _body("other (open)\n", "Guard missing (attended)\n")),)
        self.assertTrue(review_context._history_served(bodies, _located("Guard missing")))

    def test_a_title_holding_the_frame_marker_prefix_is_not_cut(self):
        title = "Shift <<< overflows"
        bodies = ((_TRIO, _body(title + " (attended)\n")),)
        self.assertTrue(review_context._history_served(bodies, _located(title)))

    def test_a_merged_finding_is_read_by_its_joined_titles(self):
        # The history stores the merged titles joined with a newline (JSON-escaped in the body).
        bodies = ((_TRIO, _body("First\\nSecond (attended)\\n")),)
        self.assertTrue(review_context._history_served(bodies, _located("First", "Second")))

    def test_a_merged_finding_is_not_read_by_a_title_it_does_not_hold(self):
        bodies = ((_TRIO, _body("First\\nSecond (attended)\\n")),)
        self.assertFalse(review_context._history_served(bodies, _located("Third")))

    def test_a_finding_without_titles_is_not_served(self):
        bodies = ((_TRIO, _body("First (attended)\\n")),)
        self.assertFalse(review_context._history_served(bodies, [{"file": _PATH, "line": 3}]))

    def test_a_title_in_another_status_is_not_served(self):
        bodies = ((_TRIO, _body("Guard missing (open)\n")),)
        self.assertFalse(review_context._history_served(bodies, _located("Guard missing")))


class DirtyDocTest(unittest.TestCase):
    """``_dirty_doc_sent`` looks in bodies, targets and headers."""

    def _run(self, target=_TRIO, body=b"", headers=()):
        return types.SimpleNamespace(request_bodies=((target, body),), request_headers=headers)

    def test_a_clean_request_reports_nothing(self):
        self.assertEqual([], review_context._dirty_doc_sent(self._run(body=b"clean")))

    def test_the_body_is_read(self):
        run = self._run(body=b"x " + runs.REVIEW_DIRTY_DOC)
        self.assertEqual([_TRIO], review_context._dirty_doc_sent(run))

    def test_the_target_is_read(self):
        target = _TRIO + "?q=" + runs.REVIEW_DIRTY_DOC.decode()
        self.assertEqual([target], review_context._dirty_doc_sent(self._run(target=target)))

    def test_a_header_is_read(self):
        run = self._run(headers=(b"X-Note: " + runs.REVIEW_DIRTY_DOC,))
        self.assertEqual(["a header"], review_context._dirty_doc_sent(run))


if __name__ == "__main__":
    unittest.main()
