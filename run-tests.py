#!/usr/bin/env python3
"""Run cargo nextest and pipe output through the tdd-guard-rust reporter.

The reporter only parses the **structured** libtest-JSON stream, not nextest's
human-readable output. Running plain `cargo nextest run` leaves test.json stuck
at `{"reason":"passed"}` even when tests fail, which silently defeats TDD-Guard's
green-phase gate. We therefore emit `--message-format libtest-json-plus` (gated by
the NEXTEST_EXPERIMENTAL_LIBTEST_JSON env var) on stdout and feed that to the
reporter; nextest's pretty progress still goes to stderr for the human.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent

    env = dict(os.environ)
    env["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"

    # JSON results on stdout (captured); pretty progress inherits stderr.
    # --no-fail-fast so a failure does not truncate the result stream the
    # reporter needs to classify every test.
    nextest = subprocess.run(
        [
            "cargo",
            "nextest",
            "run",
            "--no-fail-fast",
            "--message-format",
            "libtest-json-plus",
        ],
        stdout=subprocess.PIPE,
        cwd=project_root,
        env=env,
    )

    guard = subprocess.run(
        [
            "tdd-guard-rust",
            "--project-root",
            str(project_root),
            "--passthrough",
            "--runner",
            "nextest",
        ],
        input=nextest.stdout,
        cwd=project_root,
    )

    return guard.returncode


if __name__ == "__main__":
    sys.exit(main())
