# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The only path a scenario has to the product's output.

Every method that INTERPRETS the output translates whatever breaks into
``ProductOutputError``, which the runner maps to ``FAIL``: a malformed output
is the product's defect even though it reaches the harness as an exception.
``raw()`` is the named exemption for the byte-level searches of S3 and S10.
"""

import dataclasses
import json
import re

from smoke.errors import ProductOutputError

_UTF8 = "utf-8"


@dataclasses.dataclass(frozen=True)
class ProductOutput:
    """One completed invocation of the release binary.

    Args:
        stdout: Everything the process wrote to stdout.
        stderr: Everything it wrote to stderr.
        exit_code: The process exit code.
        command: The argv **exactly as invoked**, unscrubbed. R6's authenticated
            ``base_url`` puts a real credential in this list, and the harness
            needs it raw here because S10 asserts on what the product emitted,
            not on a copy someone already cleaned. Scrubbing happens in
            ``RunExecutor.archive`` at the moment of writing to disk, and
            nowhere else. An earlier version of this docstring said "already
            scrubbed for archiving", which is the most dangerous kind of wrong:
            an implementer who believes it skips the scrub and writes the
            credential to ``smoke/env/runs/``
            (section 2.5). The passphrase never appears here because it
            travels in ``MAGI_PASSPHRASE``, never in ``-p``.
    """

    stdout: bytes
    stderr: bytes
    exit_code: int
    command: list[str]

    def json(self) -> dict:
        """Parse stdout as JSON.

        Returns:
            The decoded object.

        Raises:
            ProductOutputError: If stdout is not valid UTF-8 JSON, or decodes
                to something that is not an object.
        """
        try:
            decoded = json.loads(self.stdout.decode(_UTF8))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductOutputError(f"stdout is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ProductOutputError(
                f"stdout decoded to {type(decoded).__name__}, expected an object"
            )
        return decoded

    def key(self, path: str) -> object:
        """Read one dotted key out of the JSON output.

        Args:
            path: A dotted path such as ``"applied_caps.ceiling_floored"``.

        Returns:
            The value at that path.

        Raises:
            ProductOutputError: If any segment is missing, or a segment of the
                path is not an object.
        """
        current: object = self.json()
        walked: list[str] = []
        for segment in path.split("."):
            walked.append(segment)
            if not isinstance(current, dict) or segment not in current:
                raise ProductOutputError(
                    f"missing key {'.'.join(walked)} in the product's JSON output"
                )
            current = current[segment]
        return current

    def require_exit(self, code: int) -> None:
        """Assert the process exited with the expected code.

        Args:
            code: The exit code the scenario expects.

        Raises:
            ProductOutputError: If the code differs.
        """
        if self.exit_code != code:
            raise ProductOutputError(
                f"expected exit {code}, got {self.exit_code}"
            )

    def raw(self) -> bytes:
        """Return both captured streams untouched.

        This is the named exemption to the boundary: S3 and S10 walk the bytes
        hunting secrets and authority patterns, and parsing or scrubbing first
        would destroy exactly what they assert. It parses nothing, so it has
        no failure mode to translate.

        Returns:
            stdout and stderr as emitted, joined by exactly one newline --
            no decoding, no parsing, no redaction, and no other change.
        """
        return self.stdout + b"\n" + self.stderr

#: Where ``vault diagnose`` starts listing per-table row counts.
COUNTS_HEADING = "counts:"

#: One line of that block. A table the product has not created yet renders as
#: ``missing`` rather than a number, and ``vault`` -- created lazily -- is the
#: FIRST label printed, so a parser that stopped at the first non-numeric line
#: dropped every history table behind it and reported an empty database.
_COUNT_LINE = re.compile(r"^\s+(\w+):\s+(\d+)\s*$")
_BLOCK_LINE = re.compile(r"^\s+\w+:\s+\S+\s*$")


def diagnose_counts(report: bytes) -> dict:
    """Per-table row counts out of a ``vault diagnose`` report.

    One parser, in one place. There were two -- the baseline's and the
    scenario's -- sharing a regex and disagreeing about everything else: only
    one filtered to the history tables, and they answered differently for an
    unreadable report. Filtering is the CALLER's business; this returns what
    the product said.

    A table reported ``missing`` is absent from the mapping rather than
    recorded as zero: unknown is not empty, and a caller that flattens the two
    reports data loss on a table nobody measured.

    Complexity: ``O(lines)``.

    Args:
        report: What ``vault diagnose`` printed.

    Returns:
        dict: Table name to row count, for every table that gave a number.
        Empty when the report carries no counts block.
    """
    counts = {}
    inside = False
    for line in report.decode("utf-8", errors="replace").splitlines():
        if line.strip() == COUNTS_HEADING:
            inside = True
            continue
        if not inside:
            continue
        match = _COUNT_LINE.match(line)
        if match is not None:
            counts[match.group(1)] = int(match.group(2))
            continue
        if _BLOCK_LINE.match(line):
            # A label with a non-numeric value -- ``missing``. Still inside the
            # block, so keep going; stopping here is what lost the tables that
            # follow ``vault``.
            continue
        break
    return counts

