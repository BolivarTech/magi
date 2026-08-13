#!/usr/bin/env python3
"""Run the full Rust test suite: cargo nextest (via tdd-guard-rust) + doctests.

Drop this at the root of any Rust project using cargo-nextest + TDD-Guard.
No configuration and no dependencies beyond the standard library.

Two independent steps, deliberately not merged:

1. **Unit/integration tests** -- `cargo nextest`, whose structured stream feeds
   the TDD-Guard reporter. The reporter only parses the **structured**
   libtest-JSON stream, not nextest's human-readable output. Running plain
   `cargo nextest run` leaves test.json stuck at `{"reason":"passed"}` even when
   tests fail, which silently defeats TDD-Guard's green-phase gate. We therefore
   emit `--message-format libtest-json-plus` (gated by the
   NEXTEST_EXPERIMENTAL_LIBTEST_JSON env var) on stdout and feed that to the
   reporter; nextest's pretty progress still goes to stderr for the human.

2. **Doctests** -- `cargo test --doc`. **cargo nextest does not run doctests**
   (upstream limitation), so without this step the rustdoc examples sit outside
   every quality gate and rot unnoticed. Since the rustdoc *is* the published
   documentation, a broken example ships to docs.rs.

Why the two steps stay separate -- do NOT pipe doctest output into the reporter:
`tdd-guard-rust --runner nextest` expects nextest's libtest-JSON. Feeding it
`cargo test --doc` output would corrupt `.claude/tdd-guard/data/test.json` and
break the Red/Green inference -- the exact failure mode step 1 exists to prevent.

Consequence, by design: TDD-Guard's phase inference is driven by the **unit
suite only**. A failing doctest with a green unit suite leaves test.json at
"passed" (so the TDD cycle is not blocked), but this script still exits non-zero
so the verification gate and the human both see the failure.

Step 2 is skipped automatically when the project exposes no doctest-able target
(e.g. a binary-only crate): there, `cargo test --doc` fails with "no library
targets found" and exit 101, which would pin the gate red forever. Detection
uses `cargo metadata`'s per-target `doctest` flag -- not the presence of
`src/lib.rs` (that path is configurable in Cargo.toml) and not stderr string
matching (fragile and locale-dependent).

Feature parity: both steps run with **default features**, and neither passes
`--workspace`. Doctests behind a feature gate are not exercised here; run
`cargo test --doc --all-features` separately when that matters.

Any other argument is forwarded verbatim to `cargo nextest run`, so the inner loop
of a red-green cycle can be scoped instead of paying for the whole suite. Two
exceptions, and both are about the contract above:

* `--no-doc` is CONSUMED here and withheld from nextest, which would reject it.
* `--message-format`, `--no-capture` and `--nocapture` are REFUSED (exit 2). The
  first is last-wins in clap, so forwarding one overrides the pinned
  `libtest-json-plus` and feeds human-readable text to the reporter; the others let
  test stdout inherit the same pipe and interleave with the JSON. Either leaves
  test.json stale or garbled -- the failure step 1 exists to prevent, reachable by a
  shorter path than the one this docstring warns about. Refused loudly rather than
  stripped, because silently dropping a flag someone typed ends a debugging session
  in confusion about why it did nothing.

Usage:
    python run-tests.py                          # nextest + reporter + doctests
    python run-tests.py --no-doc                 # skip step 2 (faster inner loop)
    python run-tests.py -E 'test(/my_case/)'     # scope the run; still feeds the guard
    python run-tests.py --no-doc -E 'binary(x)'  # both
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def has_doctest_target(project_root: Path) -> bool:
    """Report whether any local package exposes a target with doctests enabled.

    Reads `cargo metadata --no-deps`, whose target entries carry an explicit
    `doctest` boolean -- the same flag cargo itself consults when deciding what
    `--doc` runs. Binary-only crates yield False, letting the caller skip
    `cargo test --doc` instead of failing on "no library targets found".

    Args:
        project_root: Directory containing the Cargo manifest.

    Returns:
        True if at least one target has doctests enabled; False when the project
        has none, or when metadata cannot be run or parsed. The caller announces
        the skip on stderr, so a False from a broken metadata call degrades
        visibly rather than silently.
    """
    try:
        meta = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=project_root,
            text=True,
        )
    except OSError:
        return False
    if meta.returncode != 0:
        return False
    try:
        packages = json.loads(meta.stdout).get("packages", [])
    except json.JSONDecodeError:
        return False
    return any(
        target.get("doctest", False)
        for package in packages
        for target in package.get("targets", [])
    )


def main() -> int:
    project_root = Path(__file__).resolve().parent
    run_doctests = "--no-doc" not in sys.argv
    # Everything except this script's own flag is FORWARDED to nextest, so
    # `python run-tests.py -E '<filter>'` scopes the run exactly as
    # `cargo nextest run -E ...` would. `--no-doc` is consumed here and must NOT
    # travel on: nextest rejects an unknown flag and the whole run dies, which
    # would read as "the test runner is broken" rather than "that flag is mine".
    nextest_args = [arg for arg in sys.argv[1:] if arg != "--no-doc"]

    # Two forwarded flags would silently break the very contract step 1 exists to keep, so they
    # are refused rather than passed on. `--message-format` is last-wins in clap, so a
    # forwarded one overrides the pinned `libtest-json-plus` and feeds human-readable text to
    # the reporter; `--no-capture` lets test stdout inherit the same pipe and interleave with
    # the JSON. Either leaves `test.json` stale or garbled -- the failure this file's docstring
    # spends a paragraph explaining, reachable by a shorter path than the one it warns about.
    #
    # Refused loudly instead of stripped: silently dropping a flag someone typed is how a
    # debugging session ends in confusion about why `--no-capture` printed nothing.
    for banned in ("--message-format", "--no-capture", "--nocapture"):
        if any(a == banned or a.startswith(banned + "=") for a in nextest_args):
            print(
                f"run-tests.py: refusing to forward {banned} -- it breaks the "
                "libtest-JSON stream the TDD-Guard reporter parses, leaving "
                ".claude/tdd-guard/data/test.json stale. Run cargo nextest directly if you "
                "need it, and re-run this script afterwards to resync the guard.",
                file=sys.stderr,
            )
            return 2

    env = dict(os.environ)
    env["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"

    # --- Step 1: unit/integration tests -> TDD-Guard reporter ---
    # JSON results on stdout (captured); pretty progress inherits stderr.
    # --no-fail-fast so a failure does not truncate the result stream the
    # reporter needs to classify every test.
    #
    # Why forwarding matters for the GUARD specifically: without it this script
    # only ever ran the whole suite, which on a mid-size crate is minutes. That is
    # fine as a phase-close gate and unusable as the inner loop of a red-green
    # cycle -- and the temptation it creates is to reach for plain
    # `cargo nextest run` for quick feedback, which is precisely what leaves
    # test.json stale and defeats the green-phase gate this script exists to feed.
    #
    # A filtered run still writes an HONEST test.json for what it ran; what it
    # cannot do is speak for the tests it skipped. So scope the inner loop, and
    # close every phase on an unfiltered run.
    nextest = subprocess.run(
        [
            "cargo",
            "nextest",
            "run",
            "--no-fail-fast",
            "--message-format",
            "libtest-json-plus",
            *nextest_args,
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

    # --- Step 2: doctests (not piped to the reporter -- see module docstring) ---
    # Runs even when step 1 failed: a verification gate must report the complete
    # picture, not "tests failed, doctests unknown".
    doc_returncode = 0
    if run_doctests:
        if has_doctest_target(project_root):
            print("\n--- doctests (cargo test --doc) ---", flush=True)
            doc_returncode = subprocess.run(
                ["cargo", "test", "--doc"],
                cwd=project_root,
            ).returncode
        else:
            # Announced, never silent: an unexplained skip is indistinguishable
            # from the very gap this step was added to close.
            print(
                "\n--- doctests skipped: no doctest-able target "
                "(binary-only crate, or cargo metadata unavailable) ---",
                file=sys.stderr,
                flush=True,
            )

    # The reporter (`--passthrough`) exits 0 even when tests fail -- it only writes
    # test.json. The authoritative pass/fail is each runner's own exit code, so a
    # real failure must propagate (else the gate silently passes on red).
    # Precedence: unit suite first (it drives the TDD phase), then doctests.
    if nextest.returncode != 0:
        return nextest.returncode
    if doc_returncode != 0:
        return doc_returncode
    return guard.returncode


if __name__ == "__main__":
    sys.exit(main())
