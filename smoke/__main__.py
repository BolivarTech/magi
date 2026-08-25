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
import pathlib
import sys
import traceback

from smoke import runs
from smoke.binary import ReleaseBinary
from smoke.config import ModelProfile, SmokeConfig
from smoke.env import Environment
from smoke.errors import HarnessError, PreflightError
from smoke.lock import RunLock
from smoke.preflight import Preflight
from smoke.registry import DEFAULT_REGISTRY
from smoke.runner import Ambient, Runner, StampedFinding

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
        # The preflight normalises the environment itself, as its step 7b --
        # inside the lock it takes at step 2. main() must NOT do it here.
        backend = Preflight(config, env, binary,
                            RunLock(LOCK_PATH)).run(certifying, profile)
        runs.configure(binary, env, config)
        findings = Runner(DEFAULT_REGISTRY, {}, backend.reachable,
                          Ambient(tree_snapshot=None, margin_tokens=config.margin_tokens,
                                  ceiling_fraction=config.ceiling_fraction,
                                  memory_settings=env.memory_settings())).run()
    except PreflightError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except HarnessError as exc:
        print(f"harness failure: {exc}", file=sys.stderr)
        return EXIT_HARNESS
    for finding in findings:
        print(f"[{finding.outcome.value}] {finding.scenario} - {finding.assertion}")
    return exit_code_for(findings)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(EXIT_HARNESS)
