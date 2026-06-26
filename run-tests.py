#!/usr/bin/env python3
"""Run cargo nextest and pipe output through tdd-guard-rust reporter."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent

    nextest = subprocess.run(
        ["cargo", "nextest", "run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_root,
    )

    guard = subprocess.run(
        [
            "tdd-guard-rust",
            "--project-root",
            str(project_root),
            "--passthrough",
        ],
        input=nextest.stdout,
        cwd=project_root,
    )

    return guard.returncode


if __name__ == "__main__":
    sys.exit(main())
