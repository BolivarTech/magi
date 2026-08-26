# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The harness's command line, orchestration and exit codes.

Exit code 3 exists so a bug in the harness is never read as a defect in
magi-rs. It is emitted for the two failed reconciliations (section 2.4), an
invalid ``smoke.toml`` at runtime, and any uncaught exception ORIGINATING IN
THE HARNESS. An exception born while interpreting the product's output is a
different thing and becomes ``FAIL`` (section 3.1).
"""

import argparse
import datetime
import pathlib
import subprocess
import sys
import traceback

from smoke import CERTIFICATE_PATH, runs
from smoke.binary import ReleaseBinary
from smoke.certificate import (Certificate, ROUNDS_FILENAME, RoundCounter,
                               may_certify)
from smoke.config import ModelProfile, SmokeConfig
from smoke.env import Environment
from smoke.errors import HarnessError, PreflightError
from smoke.lock import RunLock
from smoke.preflight import Preflight
from smoke.registry import DEFAULT_REGISTRY
from smoke.report import Report
from smoke.runner import Ambient, Runner, StampedFinding, capture_tree
from smoke.runs import DEFINITIONS, RunExecutor, RunResult, needed_runs
# Imported for its side effect: the decorator registers at import time, so a
# scenario nobody imported is a scenario nobody runs. Without this line the
# harness reconciled an EMPTY registry against itself -- every set difference
# empty, nothing invoked, nothing reported -- and exited 0 having evaluated
# nothing. It sits last so the ordering says what it is: not a symbol this
# module uses, but the moment the scenarios come into existence.
from smoke import scenarios  # noqa: F401

EXIT_OK = 0
EXIT_NOT_PASSED = 1
EXIT_PREFLIGHT = 2
EXIT_HARNESS = 3

#: Module-level, not literals inside main: the tests patch around them, and a
#: literal buried in a function is not patchable.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "smoke" / "smoke.toml"
ENV_ROOT = REPO_ROOT / "smoke" / "env"
#: Beside env/, never inside it -- which is what lets the lock survive the
#: --reset-env that deletes the directory.
LOCK_PATH = REPO_ROOT / "smoke" / ".lock"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the harness's command line.

    Args:
        argv: The arguments, without the program name.

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: On a parse error or a mutually exclusive combination.
    """
    parser = argparse.ArgumentParser(
        prog="python -m smoke",
        description="Exercise the release magi-rs binary against a real backend.",
    )
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--smoke-1", action="store_true",
                      help="run everything; never write a certificate")
    gate.add_argument("--smoke-2", action="store_true",
                      help="run everything; certify only with no profile and no blocking outcome")

    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--init-env", action="store_true",
                           help="create smoke/env/ and initialise it")
    lifecycle.add_argument("--reset-env", action="store_true",
                           help="destroy smoke/env/ and rebuild it")

    parser.add_argument(
        "--profile",
        default=None,
        help=("model profile to run with. Its ABSENCE is the certificate's "
              "condition (REQ-S35), not an implicit default."),
    )
    return parser.parse_args(argv)


def exit_code_for(findings: list[StampedFinding]) -> int:
    """Map the findings to the process exit code.

    Args:
        findings: Every stamped finding of the run.

    Returns:
        ``EXIT_OK`` when nothing blocks the gate, ``EXIT_NOT_PASSED``
        otherwise. ``OUT_OF_SCOPE`` never counts against a green run;
        ``CANNOT_TEST`` always does (D-17).
    """
    if any(finding.outcome.blocks_gate for finding in findings):
        return EXIT_NOT_PASSED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the harness.

    Args:
        argv: The arguments, without the program name; defaults to
            ``sys.argv[1:]``.

    Returns:
        One of the four exit codes.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = SmokeConfig.load(CONFIG_PATH)
        # REQ-S35: the profile exists only when it was asked for. There is no
        # config.profile to fall back on -- see Task 6.
        profile = (ModelProfile.load(pathlib.Path(args.profile))
                   if args.profile else None)
        env = Environment(ENV_ROOT)
        if args.init_env or args.reset_env:
            # Section 4.1: the lifecycle mutates the environment harder than any
            # run does, so it takes the same lock.
            with RunLock(LOCK_PATH):
                env.reset() if args.reset_env else env.init()
            return EXIT_OK
        certifying = args.smoke_2 and profile is None
        binary = ReleaseBinary(REPO_ROOT)
        runs.configure(binary, env, config)
        # The snapshot is taken BEFORE the preflight, not after. The scenario
        # that asserts the harness left no trace compares against it, and the
        # preflight already writes -- it takes the lock and normalises the
        # environment's magi.toml. A snapshot taken afterwards would declare
        # the harness clean of exactly the changes it had just made.
        snapshot = capture_tree(REPO_ROOT)
        # The preflight normalises the environment itself, as its step 7b --
        # inside the lock it takes at step 2. main() must NOT do it here.
        backend = Preflight(config, env, binary,
                            RunLock(LOCK_PATH)).run(certifying, profile)
        # And the [memory] table is read AFTER that, for the opposite reason:
        # step 7b is what writes the file it comes from. Read before, S9's
        # saturation ceiling derives from the previous run's configuration, or
        # from nothing at all on a first run where there is no file yet -- and
        # a fresh environment could then never certify.
        ambient = Ambient(
            tree_snapshot=snapshot,
            ceiling_fraction=config.ceiling_fraction,
            memory_settings=env.memory_settings(),
        )
        executor = RunExecutor(binary, env, config)
        run_results: dict[str, RunResult] = {}
        for run_id in needed_runs(DEFAULT_REGISTRY, backend.reachable):
            result = executor.execute(DEFINITIONS[run_id])
            # Archived per result and immediately, never in a batch at the end:
            # a harness that dies mid-run should leave the outputs it already
            # had, scrubbed, rather than nothing.
            executor.archive(result)
            run_results[run_id] = result
        findings = Runner(DEFAULT_REGISTRY, run_results, backend.reachable,
                          ambient).run()
    except PreflightError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except HarnessError as exc:
        print(f"harness failure: {exc}", file=sys.stderr)
        return EXIT_HARNESS
    growth = env.growth()
    print(Report(findings, growth).render(), end="")
    try:
        certify(args, profile, findings, env, binary, run_results, growth)
    except HarnessError as exc:
        print(f"harness failure: {exc}", file=sys.stderr)
        return EXIT_HARNESS
    return exit_code_for(findings)


def certify(args, profile, findings, env, binary, run_results, growth) -> None:
    """Count this round and emit the certificate if this run may.

    The round is counted for EVERY ``--smoke-2``, not only for the one that
    certifies: what the document reports is how many attempts the release
    needed, and the attempts that did not certify are the ones that number
    exists to record.

    Args:
        args: The parsed command line.
        profile: The model profile in force, or None.
        findings: Every stamped finding.
        env: The persistent environment.
        binary: The release binary under test.
        run_results: What each shared run produced.
        growth: The environment's measured size.

    Raises:
        HarnessError: If the commit or the binary's identity cannot be read.
            Both belong to the harness, and a certificate that names neither
            is one nobody can check.
    """
    if not args.smoke_2:
        return
    counter = RoundCounter(env.root / ROUNDS_FILENAME)
    rounds = counter.bump()
    if not may_certify(True, profile, findings):
        return
    Certificate(
        version=product_version(binary),
        commit=short_commit(),
        date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        binary_sha256=binary.sha256(),
        run_count=len(run_results),
        duration_s=sum(result.duration_s for result in run_results.values()),
        rounds=rounds,
        evaluated=len({finding.scenario for finding in findings}),
        growth=growth,
        findings=findings,
    ).write(REPO_ROOT / CERTIFICATE_PATH)
    counter.reset()


def product_version(binary) -> str:
    """The version the binary reports about itself.

    Read from the product rather than from ``Cargo.toml``: the certificate is
    about the binary that ran, and a manifest edited after the build would
    make the document name a version nothing was measured against.

    Args:
        binary: The release binary.

    Returns:
        str: The version token of what ``--version`` printed.

    Raises:
        HarnessError: If the answer carries no version to take.
    """
    answer = binary.version().split()
    if not answer:
        raise HarnessError("the binary answered --version with nothing")
    return answer[-1]


def short_commit() -> str:
    """The short sha the certificate points a reader at.

    Returns:
        str: The abbreviated commit of ``HEAD``.

    Raises:
        HarnessError: If git cannot answer. It is the harness's own
            dependency, never a verdict on the product -- and a certificate
            without a commit cannot be recovered with ``git show``, which is
            the whole reason the path is fixed.
    """
    try:
        answer = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, check=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(
            "the commit could not be read, so the certificate would name no "
            "artifact: %s" % exc
        ) from exc
    return answer.stdout.strip()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(EXIT_HARNESS)
