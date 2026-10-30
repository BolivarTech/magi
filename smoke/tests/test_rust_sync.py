# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-29
"""The code-review scenarios' copies of the product's numbers and wording.

The harness reads no Rust at run time, so S47-S51 carry their own copies of
the batch constants, the per-batch floor and the texts they parse out of
``stderr``. A copy stops being true silently when the product changes, and the
first sign would be an opaque ``FAIL`` at SMOKE time. These tests read the Rust
source and fail here, at the harness's own gate, the moment the two sides
drift. Each pattern must match: a pattern that matched nothing would let the
comparison below it pass by never running.
"""

import pathlib
import re
import unittest

from smoke import runs
from smoke.scenarios import code_review

#: The repository root, three levels above this file.
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _source(relative: str) -> str:
    """Read a Rust source file with its line endings normalised.

    rustfmt writes CRLF on Windows and LF elsewhere, so ``\\r`` is stripped
    before any pattern runs. Bytes that are not UTF-8 (a test literal) are
    replaced; nothing read here lives in them.

    Args:
        relative: The path under the repository root, with forward slashes.

    Returns:
        str: The file's text.
    """
    return (_ROOT / relative).read_bytes().decode("utf-8", errors="replace").replace("\r", "")


def _usize_const(text: str, name: str) -> int:
    """The value of ``pub const <name>: usize = <n>;`` in *text*.

    Args:
        text: A Rust source file's text.
        name: The constant's name.

    Returns:
        int: Its value, with ``_`` separators removed.

    Raises:
        AssertionError: When the declaration is not found.
    """
    match = re.search(r"pub const %s: usize = ([\d_]+);" % re.escape(name), text)
    if match is None:
        raise AssertionError("pub const %s: usize not found" % name)
    return int(match.group(1).replace("_", ""))


class BatchConstantTests(unittest.TestCase):
    """The batch arithmetic R12's input is sized with (``smoke/runs.py``)."""

    # MAGI FIX (S6 loop 5): the copies had no automated sync (F1-b2).
    def setUp(self) -> None:
        self.window = _source("src/review/window.rs")

    def test_bytes_per_token_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "BYTES_PER_TOKEN"), runs.BYTES_PER_TOKEN)

    def test_the_unmeasured_window_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "DEFAULT_UNMEASURED_WINDOW_TOKENS"),
                         runs.UNMEASURED_WINDOW_TOKENS)

    def test_the_harness_sizes_for_a_batch_at_least_as_large_as_the_product_fills(self) -> None:
        # The harness sizes at the larger of the product's two values (50 provisional, 60 measured);
        # a product fraction above the harness's would plan fewer batches than R12's input assumes.
        percent = _usize_const(self.window, "BATCH_WINDOW_FRACTION_PERCENT")
        self.assertGreaterEqual(round(runs.BATCH_WINDOW_FRACTION * 100), percent)

    def test_the_floor_is_the_products_two_minimums(self) -> None:
        text = _source("src/magi/mod.rs")
        found = {name: int(secs) for name, secs in re.findall(
            r"const (MIN_OPERATION_BUDGET|MIN_CLIENT_TIMEOUT): Duration = Duration::from_secs\((\d+)\);", text)}
        self.assertEqual({"MIN_OPERATION_BUDGET", "MIN_CLIENT_TIMEOUT"}, set(found))
        self.assertRegex(text, r"AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS: u64 =\s*"
                               r"MIN_OPERATION_BUDGET\.as_secs\(\) \+ MIN_CLIENT_TIMEOUT\.as_secs\(\);")
        self.assertEqual(sum(found.values()), code_review.FLOOR_SECS)


if __name__ == "__main__":
    unittest.main()
