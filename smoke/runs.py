# Author: Julian Bolivar
# Version: 0.21.0
# Date: 2026-09-28
"""The single door between a scenario and the product.

``run=None`` never meant *leaves the product alone*. Several standalone
scenarios invoke it -- one runs the product's ``init``, another exercises a
workdir flag, another passes a blank environment variable -- and the dependency
rule of section 2.1 says a scenario reaches the product **through this module**,
never by importing the binary itself. Offering only the named shared runs left
those scenarios with no legal path at all, which was a hole in the design rather
than an edge case for an implementer to improvise around.

What separates a shared run from an ``invoke()`` is **cost, not rank**. The
named runs exist to amortise the trio, the large payload and a session with
accumulated history; a product ``init`` on an empty directory has nothing to
amortise, and promoting it to a shared definition would be a table row exactly
one scenario reads.

Module-level state lives here, and this is the **second** such exception in the
harness after the scenario registry. Two is a number worth writing down, because
a third gets added without a discussion by someone reading these as precedent.
Both are write-once at startup and read-only afterwards, which is what keeps
them out of the concurrency argument, and :func:`invoke` failing loudly when
unconfigured is what makes the write-once half checkable rather than assumed.
"""

import contextlib
import dataclasses
import datetime
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

from smoke.config import PLACEHOLDER_ENTRIES, ROTATION_MARKER
from smoke.errors import (HarnessError, PayloadTooSmall,
                          ProductOutputError, TimedOut)
from smoke.payload import PayloadBuilder
from smoke.product import ProductOutput, diagnose_counts
from smoke.secrets import PlantedSecret, mint_credential, scrub

#: How the passphrase reaches the product. Never ``-p``: that is a global flag,
#: so it would ride in the archived command line of every run, and any process
#: on the machine can read a live one.
PASSPHRASE_VARIABLE = "MAGI_PASSPHRASE"

#: What a definition declares about the size of its prompt. The large one is
#: generated from the tree at a declared byte count; every other run carries a
#: prompt short enough to read.
PAYLOAD_SMALL = "small"
PAYLOAD_LARGE = "large"

#: The value R7 rotates the backend credential TO. It only has to DIFFER from
#: the real one: the scenario asserts that the local database still opens with
#: the same passphrase and that prior history survives, and for that the stored
#: value merely has to change. A second credential that actually authenticates
#: would add nothing to any assertion and one more secret to protect.
ROTATION_SENTINEL = b"smoke-harness-rotated-sentinel-value"

#: What is stored under :data:`~smoke.config.ROTATION_MARKER` while a rotation
#: is in flight. Nothing reads it -- the preflight recognises the rotation by
#: the entry's NAME, because the product never prints a stored value -- so this
#: exists only because ``vault set`` needs something to store.
ROTATION_MARKER_VALUE = b"a rotation was in flight when this was written"

_VAULT_SUBCOMMAND = "vault"
_SET_SUBCOMMAND = "set"
_REMOVE_SUBCOMMAND = "rm"
_FORCE_FLAG = "--force"

#: How long one vault call of the rotation is given, in seconds. It writes a
#: single row and touches no network.
VAULT_TIMEOUT_S = 60

_binary = None
_env = None
_config = None

_UNCONFIGURED = (
    "smoke.runs was asked for %s before runs.configure ran; the module needs a "
    "binary, an environment and a configuration before a scenario can use it"
)


@dataclasses.dataclass(frozen=True)
class Attempt:
    """One invocation that either produced a capture or explained why it did not.

    The runner turns a raised ``ProductOutputError`` into a SINGLE finding for
    the whole scenario, which is right for a scenario built around one
    invocation and wrong for the eight autonomous ones: four assertions would
    collapse into one line and three of them would vanish from the report
    without anybody noticing they were never evaluated. A scenario that wants
    to keep reporting after an invocation dies asks for an Attempt instead.

    Attributes:
        output: The capture, or None when the invocation never completed.
        failure: Why it did not complete; empty when it did.
    """

    output: ProductOutput | None
    failure: str

    @property
    def ok(self) -> bool:
        """Whether there is a capture to assert on.

        Returns:
            True when the invocation completed, whatever its exit code. A
            non-zero exit is an answer, not a failure to answer.
        """
        return self.output is not None


def configure(binary, env, config) -> None:
    """Give the module the collaborators every invocation needs.

    Called once, by the wiring in ``__main__``. Anything that calls
    :func:`invoke` before this gets a named failure rather than an attribute
    error from somewhere deeper.

    Args:
        binary: The release binary under test.
        env: The persistent test environment, for the archive directory.
        config: The harness configuration.
    """
    global _binary, _env, _config
    _binary = binary
    _env = env
    _config = config


def reset_for_test() -> None:
    """Return the module to its unconfigured state.

    Exists for the harness's own tests: module state that cannot be cleared
    makes every test after the first depend on the order they ran in.
    """
    global _binary, _env, _config
    _binary = None
    _env = None
    _config = None


def repo_root():
    """The checkout the binary under test came from.

    Args:
        None.

    Returns:
        pathlib.Path: The repository root.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _binary is None:
        raise HarnessError(_UNCONFIGURED % "the repository root")
    return pathlib.Path(_binary.repo_root)


def workspace_root():
    """The directory the product's ``-w`` should point at.

    It is the environment root and NOT ``env/.magi``: ``-w`` names the
    directory the walk-up starts from, and handing it the workspace itself is
    the mistake that makes a scenario resolve one level too deep.

    Args:
        None.

    Returns:
        pathlib.Path: The persistent environment's root.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _env is None:
        raise HarnessError(_UNCONFIGURED % "the workspace root")
    return pathlib.Path(_env.root)


def scratch_root():
    """The fixture area for the scenarios that need a tree with no ``.magi/``.

    Created on demand, because a scenario that finds it missing after somebody
    cleaned up by hand should still run rather than report the harness's own
    tidiness as a product defect. The environment does the creating, so the
    directory comes back with its self-protecting ignore rule rather than as a
    bare ``mkdir`` whose contents the next ``git status`` would report.

    Args:
        None.

    Returns:
        pathlib.Path: The scratch area beside the environment, existing.

    Raises:
        HarnessError: If :func:`configure` has not run, or the directory cannot
            be created.
    """
    if _env is None:
        raise HarnessError(_UNCONFIGURED % "the scratch area")
    _env.prepare_scratch()
    return pathlib.Path(_env.scratch_dir)


def passphrase():
    """The environment's vault passphrase.

    A scenario needs it to open the product's database, and it travels to the
    child in ``MAGI_PASSPHRASE`` -- never in ``-p``, which is a global flag and
    would ride in a command line any process on the machine can read.

    Args:
        None.

    Returns:
        str: The configured passphrase.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _config is None:
        raise HarnessError(_UNCONFIGURED % "the vault passphrase")
    return _config.passphrase


def payload_floor():
    """The token floor S8 asserts the large run's input against.

    Read off the CONFIGURATION rather than imported from
    :mod:`smoke.config`'s defaults: ``[payload].token_floor`` is overridable in
    ``smoke.toml``, and a scenario comparing against the built-in default would
    silently ignore an operator who moved it -- the second source of truth that
    disagrees the first time somebody changes a setting.

    Args:
        None.

    Returns:
        int: The configured floor, in tokens.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _config is None:
        raise HarnessError(_UNCONFIGURED % "the payload token floor")
    return _config.payload_token_floor


def payload_target():
    """How many bytes the large payload is declared to be.

    Args:
        None.

    Returns:
        int: The configured size, in bytes.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _config is None:
        raise HarnessError(_UNCONFIGURED % "the payload size")
    return _config.payload_target_bytes


def archive_root():
    """Where this run's invocations were archived, or ``None`` if unconfigured.

    The archive is the third channel S10 searches, and it is the one a reader
    keeps. What is written there is SCRUBBED, so a secret found in it is one
    the scrubber was never told about -- exactly the case worth reporting
    rather than the case worth hiding.

    Returns:
        pathlib.Path | None: The directory, or None before :func:`configure`.
    """
    if _env is None:
        return None
    return pathlib.Path(_env.runs_dir)


def attempt(argv, stdin=None, *, timeout_s, label, planted=(), cwd=None,
            env=None):
    """Invoke the product, returning the capture or the reason there is none.

    Args:
        argv: Arguments for the product, without the program name.
        stdin: Bytes to write to the child's standard input, or ``None``.
        timeout_s: Wall clock the invocation may take.
        label: What to call this invocation in the archive.
        planted: The secrets this invocation put in front of the product.
        cwd: The directory to run the product in, or ``None``.
        env: Variables to overlay on the harness's own environment.

    Returns:
        Attempt: The capture, or the failure that replaced it.

    Raises:
        HarnessError: If :func:`configure` has not run. A harness that was
            never wired is not a verdict on the product, so it is NOT folded
            into the returned failure.
    """
    try:
        output = invoke(argv, stdin, timeout_s=timeout_s, label=label,
                        planted=planted, cwd=cwd, env=env)
    except ProductOutputError as exc:
        return Attempt(output=None, failure=str(exc))
    return Attempt(output=output, failure="")


def invoke(argv, stdin=None, *, timeout_s, label, planted=(), cwd=None,
           env=None):
    """Run the product once, return the capture raw, archive it scrubbed.

    The order of those last two is the point, and half the fix: assertions run
    over the **in-memory capture at full fidelity**, and only what reaches disk
    is scrubbed. Scrubbing first would weaken the security scenarios into empty
    greens -- they would be searching text the harness had already cleaned.

    Args:
        argv: Arguments for the product, without the program name.
        stdin: Bytes to write to the child's standard input, or ``None``.
        timeout_s: Wall clock the invocation may take.
        label: What to call this invocation in the archive.
        planted: The :class:`~smoke.secrets.PlantedSecret` values this
            invocation put in front of the product. The scrubber removes what
            it is told about and nothing else, so an undeclared secret is
            written to the archive in clear.
        cwd: The directory to run the product in, or ``None`` for the
            harness's own. Never a ``chdir``: one process runs every scenario
            in turn, so a directory changed for one of them stays changed for
            the next.
        env: Variables to overlay on the harness's own environment. This is how
            ``MAGI_PASSPHRASE`` reaches the product, and how S15 hands it a
            blank one.

    Returns:
        ProductOutput: The capture, unscrubbed.

    Raises:
        HarnessError: If :func:`configure` has not run.
        ProductOutputError: If the product could not be started, or did not
            finish inside *timeout_s*. A scenario with several assertions
            should call :func:`attempt` instead, which returns that failure
            rather than raising it.
    """
    if _binary is None or _env is None:
        raise HarnessError(
            "runs.invoke was called before runs.configure; the module needs a "
            "binary and an environment before it can reach the product"
        )
    output = _binary.invoke(argv, stdin=stdin, env=env, timeout=timeout_s,
                            cwd=cwd)
    _archive(output, label, planted)
    return output


def _archive(output, label, planted) -> None:
    """Write one invocation's command and output to the run directory.

    A harness that dies mid-run should leave the outputs it already had rather
    than nothing, so each invocation is archived as it lands instead of in a
    batch at the end.

    The environment's PASSPHRASE is added to whatever the caller declared.
    ``RunExecutor.archive`` has always done this; this path had not, and every
    autonomous scenario goes through here with the same passphrase in the
    child's environment. One guarded path and one unguarded, for the secret
    that opens the whole vault.

    Args:
        output: The capture to persist.
        label: What to call it on disk.
        planted: Secrets to remove before writing.
    """
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    directory = pathlib.Path(_env.runs_dir) / ("%s-%s" % (label, stamp))
    directory.mkdir(parents=True, exist_ok=True)
    body = b"$ %s\n%s" % (
        " ".join(output.command).encode("utf-8", errors="replace"),
        output.raw(),
    )
    secrets = tuple(planted) + (PlantedSecret(passphrase(), "vault passphrase"),)
    (directory / "invocation.log").write_bytes(scrub(body, secrets))


@dataclasses.dataclass(frozen=True)
class RunDefinition:
    """One shared run: what to invoke, what to feed it, and what it costs.

    Attributes:
        run_id: The id scenarios declare, e.g. ``"R4"``.
        argv: The product's arguments, with :data:`WORKSPACE_TOKEN` standing in
            for the environment's path. The token is resolved at execution
            because the path is not known when the table is written, and a
            table that could only be built after the environment existed would
            have to be assembled somewhere, by somebody, at some moment -- one
            more thing to forget.
        stdin: The prompt. It travels on standard input so a payload of a
            quarter of a megabyte never meets a command-line length limit.
        needs_trio: Whether the run reaches the MAGI trio. Declared here so the
            table says what each run costs; nothing in this module branches on
            it, and nothing should -- a run's cost is a fact about the run, and
            reading it as an instruction is how a declaration turns into
            behaviour nobody asked for.
        payload_size: :data:`PAYLOAD_SMALL` or :data:`PAYLOAD_LARGE`.
        timeout_s: The wall clock this run may take. Every definition carries
            its own, and none may omit it: a run against a real backend can
            hang forever, and the values differ by an order of magnitude
            between a consult over a large payload and an invocation designed
            to fail fast.
        planted: The secrets this run puts in front of the product, copied onto
            the result so the scenario searches for exactly what went in.
            Anything the scenario reconstructed for itself would be a second
            derivation that can disagree with the first.
        env: Variables overlaid on the child's environment, as ordered pairs so
            the definition stays immutable. The passphrase is NOT here: every
            run gets it, so the executor adds it rather than eight rows
            repeating it.
        rotates: Whether the run rotates the backend credential and puts it
            back. Declared rather than inferred from the run id, for the same
            reason ``planted`` is: an id comparison is a second source of truth
            that disagrees the first time somebody renames a row.
        files: Workspace-relative files written into the review workspace
            (``smoke/scratch/review-ws``) before the run, as ordered
            ``(path, content)`` pairs. A review run reads its input from disk,
            so the input has to exist first. The workspace is rebuilt from
            nothing on every run, so the pairs are always written fresh.
    """

    run_id: str
    argv: tuple[str, ...]
    stdin: bytes
    needs_trio: bool
    payload_size: str
    timeout_s: int
    planted: tuple[PlantedSecret, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    rotates: bool = False
    files: tuple[tuple[str, bytes], ...] = ()


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one shared run produced.

    Attributes:
        run_id: Which definition produced it.
        output: The capture, UNSCRUBBED. The assertions run over this; only
            what reaches disk is scrubbed.
        duration_s: Wall clock the invocation took.
        timed_out: Whether it exceeded its own ceiling. The runner reads this
            and turns it into ``CANNOT_TEST`` for every scenario attached to
            the run, so no scenario author has to remember to check -- which is
            the difference between a rule and a guarantee.
        planted: What the run put in front of the product.
        stdin_bytes: How many bytes went to the child's standard input. What
            the run CARRIED, not what its definition declared: a large run's
            prompt is 54 bytes and its payload is a quarter of a megabyte, and
            reading the declaration told S8 the payload had never been sent
            while the product was busy answering it.
    """

    run_id: str
    output: ProductOutput
    duration_s: float
    timed_out: bool
    planted: tuple[PlantedSecret, ...]
    #: Defaulted so the doubles in the tests stay short. In production there is
    #: exactly one constructor -- the executor -- and it always sets it.
    stdin_bytes: int = 0
    #: Per-table row counts taken BEFORE the run changed anything, for the one
    #: run that rotates. ``None`` means no baseline was recorded, which is a
    #: different state from "the tables were empty" and is reported as such:
    #: without it, "the previous history is still there" can only ask whether
    #: SOME history is there, and a rotation that destroyed most of it passes.
    baseline: dict | None = None
    #: The bytes of every file under the review workspace's ``.magi/logs``,
    #: in sorted path order, read right after the run. The workspace starts
    #: without that directory, so this is what the run wrote. ``None`` means the
    #: run used no review workspace or the directory did not exist. Captured at run
    #: time because the next review run rebuilds the workspace from nothing, so
    #: a scenario reading the live directory would depend on run order.
    review_log: tuple[bytes, ...] | None = None
    #: The ``.magi/`` paths git lists in the review workspace right before the
    #: run, with the product's own query (:data:`GIT_VISIBLE_QUERY`), sorted.
    #: Empty when the run uses no review workspace, or when an ignore rule
    #: (a global ``core.excludesFile``, say) hides ``.magi/`` from git and so
    #: from the product: then no protected path was an enumeration candidate.
    protected_visible: tuple[str, ...] = ()


#: Where the product writes its logs, relative to a workspace root. The one
#: definition: :mod:`smoke.logs` re-exports it, because it imports this module
#: and this module cannot import it back.
LOG_DIR_PARTS = (".magi", "logs")

#: The token that stands for the environment's path inside a definition's argv.
WORKSPACE_TOKEN = "<workspace>"

#: The flags every invocation carries. ``--output-format json`` is opt-in and
#: without it the output is text, so no structural assertion is possible at
#: all; ``-w`` names the directory the product's walk-up starts from, so a run
#: never depends on where the harness was launched.
COMMON_FLAGS = ("--output-format", "json", "-w", WORKSPACE_TOKEN)

#: R6 hands the product a credential and an endpoint that will not answer. The
#: credential is minted per process: its marker is long, random and purely
#: alphanumeric, so it survives any level of encoding and cannot occur by
#: chance in a model's own prose.
_R6_CREDENTIAL = mint_credential()


#: What R4 gives the product as its wall clock, in seconds. MEASURED against
#: this repository's own backend, never chosen: at 300 the product derives 40s
#: per mage and 24s per attempt, and the 250 kB consult abandoned after 81.7s
#: with a typed provider timeout -- eight red assertions across S6, S7, S8 and
#: S18, every one of them from this single number. The same payload against the
#: same backend at 1800 derives 249s per mage and 149s per attempt, and
#: completed in 103s with three real verdicts.
#:
#: It looks enormous beside a healthy run of under two minutes, and that is the
#: product's arithmetic rather than padding: the budget is spread over two
#: attempts of each of three models per mage, so what a healthy run needs is a
#: small fraction of what a fully rotating one is allowed.
#:
#: v0.21.0 (PM-S-5) moved it again, for the same reason: the shipped output
#: cap grew to 65 536 tokens and the default trio seats reasoning models that
#: need up to 700s per request to use it, so 1800 (74s per request) would
#: certify the pool's fallbacks recovering the seats, not the trio that
#: actually ships. 16820 derives 2335s per mage and 700s per attempt -- the
#: same clock ``docs/`` recommends for a MAGI gate -- and was measured in
#: ``planning/milestones/MS2-smoke-measure.md`` (2026-09-27T08:48:48Z,
#: http://localhost:11434/v1, release binary of ``03f81f4``): R4 finished in
#: 283s with three real verdicts, no HTTP 400 and no context-window rejection.
LARGE_CONSULT_TIMEOUT_S = 16820

#: The harness's own ceiling for that run, 20 % above the product's. The
#: comment in the table says why it can never be the smaller of the two.
LARGE_CONSULT_CEILING_S = 20184

#: Seconds each trio-touching run took at the 0.21.0 defaults, MEASURED --
#: never estimated -- against ``http://localhost:11434/v1`` (local daemon,
#: ``:cloud`` models) with the release binary of ``03f81f4``, 2026-09-27
#: 08:48:48-09:09:47 UTC. Copied verbatim (rounded up) from
#: ``planning/milestones/MS2-smoke-measure.md``: R4 282.296s, R5 169.610s,
#: R8 66.062s. The key set is exact: it names every run whose
#: ``needs_trio`` is true, no more and no fewer.
#: R10 carries R8's figure PROVISIONALLY: it is the same kind of run (a small
#: consult) and has not been measured yet. Replace it with R10's own wall clock
#: at the next run against a real backend.
#: R12 carries PROVISIONAL figures too: the generated input is about 72 000
#: tokens and the product fills a batch to 50 % of the 32 768-token default
#: window (16 384 tokens), so a review plans at least five batches. Five times
#: R4's 283 s plus the gate is about 1 500 s. Replace it with R12's own wall
#: clock at the first run against a real backend.
TRIO_RUNS_MEASURED_S: dict[str, int] = {"R4": 283, "R5": 170, "R8": 67,
                                        "R10": 67, "R12": 1500}

#: The factor a trio run's ``timeout_s`` must clear over its measurement in
#: :data:`TRIO_RUNS_MEASURED_S`. The cloud endpoint's pace varies by time
#: slot (PM-S-8) and the measurement above is a single run in one slot -- a
#: US-morning slot measured slower elsewhere in this project's history --
#: so the harness timeout carries double that single sample rather than
#: trusting it at face value.
MEASURED_WALL_CLOCK_MARGIN = 2

#: The default of ``[headless] max_system_bytes``. R11 sends one byte more, so
#: it fails if the product's default cap moves without the harness noticing.
SYSTEM_CAP_DEFAULT_BYTES = 16384

#: Flags that make the product read the prompt as a JSON envelope.
ENVELOPE_FLAGS = ("--input-format", "json")

#: The system text R9 and R10 carry. R9 is told it will be ignored and R10
#: that it will be applied; neither assertion depends on what a model answers.
_ENVELOPE_SYSTEM = "Answer in as few words as you can."


#: The batch constants below DUPLICATE the product's on purpose (the harness
#: reads no Rust): ``BYTES_PER_TOKEN``, ``DEFAULT_UNMEASURED_WINDOW_TOKENS`` and
#: ``BATCH_WINDOW_FRACTION_PERCENT`` live in ``src/review/window.rs`` (the
#: unmeasured window is also referenced from ``src/defaults.rs``). Change one
#: side and the other must follow, or the generated input stops filling the
#: batches S47 requires; ``smoke/tests/test_rust_sync.py`` reads the Rust and
#: fails on drift.
#:
#: The same heuristic the product uses to turn bytes into tokens (one token per
#: four bytes). The batch arithmetic below is only as good as this number.
BYTES_PER_TOKEN = 4

#: The window the product assumes for a model it could not measure. R12's
#: input is sized against it: with a measured, larger window the product plans
#: fewer batches for an environmental reason, and S47 says so.
UNMEASURED_WINDOW_TOKENS = 32768

#: The fraction of that window one batch may fill, taken at the larger of the
#: product's two values (0.50 provisional, 0.60 measured): sizing for the
#: larger batch guarantees at least three batches under either.
BATCH_WINDOW_FRACTION = 0.60

#: Batches the generated review input must be able to fill.
REVIEW_MIN_BATCHES = 3

#: Bytes of each generated source file, and the files themselves.
REVIEW_FILE_BYTES = 96000
REVIEW_GEN_PATHS = ("gen/a.rs", "gen/b.rs", "gen/c.rs")

#: Directory (workspace-relative) the review runs read, and the one that holds
#: only files a review must skip.
REVIEW_GEN_DIR = "gen"
REVIEW_SKIPPED_DIR = "skipped"

#: The token the review runs use for ``-w``, and the directory (under the
#: scratch area) it resolves to. The review input cannot live in the
#: environment: its ``.gitignore`` is ``*``, and the product leaves out what
#: ``.gitignore`` excludes. The workspace is therefore a nested git repository
#: of its own, so git's top level is the workspace and no outer rule applies.
REVIEW_WORKSPACE_TOKEN = "<review-workspace>"
REVIEW_WORKSPACE_DIR = "review-ws"

#: The flags of the review runs: :data:`COMMON_FLAGS` with the review workspace.
REVIEW_FLAGS = ("--output-format", "json", "-w", REVIEW_WORKSPACE_TOKEN)

#: The git query the product's own enumeration is built on; the harness runs
#: it before a review so an input git cannot see is the harness's failure.
GIT_VISIBLE_QUERY = ("ls-files", "--others", "--exclude-standard", "-z")

#: The ``-c`` settings of every git call the harness makes, the subset of the
#: product's ``NEUTRAL_GIT_CONFIG`` (``src/system/review_git.rs``) that applies
#: to ``init`` and ``ls-files``: no filesystem monitor and no hooks. The rest of
#: that list shapes diffs and fetches, which the harness never runs.
HARNESS_GIT_CONFIG = ("core.fsmonitor=false", "core.hooksPath=/dev/null")

#: The prefix of the environment variables that select or configure a git
#: repository (``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``,
#: ``GIT_CONFIG_*``...). The product starts git from an allowlisted environment
#: that carries none of them, so the harness drops them all.
GIT_ENV_PREFIX = "GIT_"


def git_command(workspace, *args) -> list[str]:
    """The command line of a harness git call run inside *workspace*.

    Args:
        workspace: The directory git runs in (``-C``).
        *args: The git subcommand and its arguments.

    Returns:
        list: ``git -C <workspace>``, the :data:`HARNESS_GIT_CONFIG` settings,
        then *args*.
    """
    command = ["git", "-C", str(workspace)]
    for setting in HARNESS_GIT_CONFIG:
        command += ["-c", setting]
    return command + list(args)


def git_env() -> dict[str, str]:
    """The environment of a harness git call: this process's, without git's own.

    An inherited ``GIT_DIR`` or ``GIT_WORK_TREE`` would point git at another
    repository, so the visibility check could approve an input other than the
    one the product sees, and ``git init`` could initialise somewhere else.

    Returns:
        dict: ``os.environ`` without any variable starting with
        :data:`GIT_ENV_PREFIX`.
    """
    return {name: value for name, value in os.environ.items()
            if not name.upper().startswith(GIT_ENV_PREFIX)}

#: Seconds the harness gives each git command it runs.
GIT_TIMEOUT_S = 60

#: A path outside the workspace, for the hostile-path run. Its last component
#: appears in no fixed product text, so S48 can require it in stderr without
#: the product's own phrase satisfying the check.
REVIEW_OUTSIDE_COMPONENT = "smoke-hostile-target"
REVIEW_OUTSIDE_PATH = "../" + REVIEW_OUTSIDE_COMPONENT

#: The total R15 asks for, below what even ONE batch needs at the product's
#: floor under the most lenient config: no rotation and no retry give an attempt
#: factor of 1.2, so a batch needs 6 s (classifier) + 15 s x 1.2 = 24 s. The
#: refusal therefore fires at any batch count and any ``[magi]`` setting. R16
#: asks for it too, so a protected file wrongly taken for review ends at the
#: budget refusal instead of at a model.
# MAGI FIX (S6 loop 6): 30 s refused one batch only with rotation or retry on;
# under ``max_rotations = 0`` and retry disabled one batch got 30 s >= 24 s and
# reached the trio (F1a-c1).
REVIEW_TOO_SHORT_TIMEOUT_S = 20

#: The input R16 reviews: the review workspace's root, which holds its
#: ``.magi`` and ``.git`` beside the files the run writes.
REVIEW_ROOT_PATH = "."

#: Toy contents of the files R14 reviews. The PNG signature plus NUL bytes make
#: the first a binary; the second carries a key-shaped toy secret. Both are
#: worth nothing.
_TOY_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
_TOY_SECRET_FILE = b"token = sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"  # allow-secret-scan


def review_input_files() -> tuple[tuple[str, bytes], ...]:
    """The deterministic source files R12 and R15 review.

    Each file is a run of small, documented, uniquely named functions, so the
    product's unit splitter can cut batches anywhere and no function repeats.
    The total is checked against the batch arithmetic: an input too small to
    fill :data:`REVIEW_MIN_BATCHES` batches would make S47 report a property
    of the harness as a product result, so it raises instead.

    Complexity: ``O(total bytes)``.

    Returns:
        tuple: ``(path, content)`` pairs, one per entry of
        :data:`REVIEW_GEN_PATHS`.

    Raises:
        HarnessError: If the generated bytes cannot fill the batches.
    """
    files = []
    for path in REVIEW_GEN_PATHS:
        stem = path.rsplit("/", 1)[-1].split(".")[0]
        blocks: list[str] = []
        size = 0
        index = 0
        while size < REVIEW_FILE_BYTES:
            block = (
                "/// Mixes the input with the constant %d.\n"
                "pub fn %s_%05d(x: u32) -> u32 {\n"
                "    x.wrapping_mul(%d).wrapping_add(%d)\n"
                "}\n\n" % (index, stem, index, 31 + index % 7, index))
            blocks.append(block)
            size += len(block)
            index += 1
        files.append((path, "".join(blocks).encode("utf-8")))
    needed = REVIEW_MIN_BATCHES * int(UNMEASURED_WINDOW_TOKENS
                                      * BATCH_WINDOW_FRACTION)
    tokens = sum(len(body) for _, body in files) // BYTES_PER_TOKEN
    if tokens < needed:
        raise HarnessError(
            "the generated review input is %d tokens, under the %d needed for "
            "%d batches" % (tokens, needed, REVIEW_MIN_BATCHES))
    return tuple(files)


def review_gen_line_counts() -> dict[str, int]:
    """Line count of each generated review file, by workspace-relative path.

    Complexity: ``O(total bytes)``.

    Returns:
        dict: ``{path: lines}``; a line number in ``1..lines`` points inside
        the file, anything else points outside it.

    Raises:
        HarnessError: If the generated input cannot fill the batches.
    """
    return {path: body.count(b"\n") for path, body in review_input_files()}


def visible_paths(workspace) -> frozenset[str]:
    """The paths git, run from *workspace*, lists as the product would.

    The product enumerates a directory with ``git ls-files --others
    --exclude-standard`` (:data:`GIT_VISIBLE_QUERY`), so a file this query does
    not list is invisible to a review.

    Complexity: ``O(output bytes)``.

    Args:
        workspace: The review workspace (a git repository of its own).

    Returns:
        frozenset[str]: The listed paths, ``/``-separated and relative to
        *workspace*. The empty string the ``-z`` terminator leaves is dropped.

    Raises:
        HarnessError: If git cannot be run or fails.
    """
    try:
        # MAGI FIX (S6 loop 6): git ran with the harness's own environment, so
        # an inherited GIT_DIR inspected another repository (F1a-c2).
        done = subprocess.run(
            git_command(workspace, *GIT_VISIBLE_QUERY), env=git_env(),
            capture_output=True, timeout=GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError("could not run git in %s: %s" % (workspace, exc)) from exc
    if done.returncode != 0:
        raise HarnessError("git failed in %s: %s" % (
            workspace, done.stderr.decode("utf-8", errors="replace").strip()))
    return frozenset(path for path in
                     done.stdout.decode("utf-8", errors="replace").split("\0")
                     if path)


def verify_visible(workspace, paths) -> None:
    """Check that git, run from *workspace*, lists every path of *paths*.

    A file :func:`visible_paths` does not list is invisible to a review. That
    is a fact about the harness's workspace and not a product verdict, so it
    raises the harness error (exit 3) instead of letting a scenario report
    ``FAIL`` over an input the product never saw.

    Complexity: ``O(output bytes)``.

    Args:
        workspace: The review workspace (a git repository of its own).
        paths: Workspace-relative paths that must be visible, ``/``-separated.

    Raises:
        HarnessError: If git fails, or lists fewer than all of *paths*.
    """
    listed = visible_paths(workspace)
    hidden = [path for path in paths if path not in listed]
    if hidden:
        raise HarnessError(
            "git cannot see the review input %s in %s; an ignore rule hides "
            "it from the product too" % (", ".join(hidden), workspace))


def protected_candidates(workspace) -> tuple[str, ...]:
    """The ``.magi/`` paths git lists in *workspace*, as the product would see them.

    Args:
        workspace: The review workspace (a git repository of its own).

    Returns:
        tuple[str, ...]: The listed paths under ``.magi/``, sorted.

    Raises:
        HarnessError: If git fails.
    """
    # MAGI FIX (S6 loop 9): S51 passed vacuously when an ignore rule hid
    # .magi/ from git and so from the product (F1a-c1, F1a-m2).
    prefix = LOG_DIR_PARTS[0] + "/"
    return tuple(sorted(path for path in visible_paths(workspace)
                        if path.startswith(prefix)))


def _is_link(path) -> bool:
    """Whether *path* is a symlink or, on Windows, a junction.

    ``os.path.isjunction`` exists from Python 3.12; the harness's 3.11 floor
    has no junction test, and there a junction reads as not a link.

    Args:
        path: The path to test, which need not exist.

    Returns:
        bool: ``True`` for a symlink or a junction.
    """
    is_junction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(is_junction and is_junction(path))


def _reject_links(top) -> None:
    """Refuse a tree that holds a symlink or a junction, *top* included.

    The review workspace is copied from the environment's ``.magi`` with
    :func:`shutil.copytree`, which follows links: an entry pointing outside
    would copy what it points at into the workspace the product reads. The
    environment never needs one, so any link is refused rather than copied
    as a link or followed.

    Args:
        top: The directory to check.

    Raises:
        HarnessError: Naming the first link found.
    """
    if _is_link(top):
        raise HarnessError("refusing to copy %s: it is a link" % top)
    for directory, dirs, files in os.walk(top):
        for name in dirs + files:
            path = os.path.join(directory, name)
            if _is_link(path):
                raise HarnessError(
                    "refusing to copy %s: %s is a link, and the review "
                    "workspace never follows one" % (top, path))


def _skip_top_level_logs(magi_dir):
    """A ``copytree`` filter that leaves out the ``logs`` directory of *magi_dir*.

    The review workspace is built from the environment's ``.magi``, whose
    ``logs/`` holds what earlier runs wrote. Leaving it out makes the log a
    review run captures exactly what that run wrote.

    Args:
        magi_dir: The ``.magi`` directory being copied.

    Returns:
        callable: The ``ignore`` argument of :func:`shutil.copytree`.
    """
    top = os.path.normcase(os.path.abspath(magi_dir))

    def ignore(directory, names):
        if os.path.normcase(os.path.abspath(directory)) != top:
            return set()
        return {name for name in names if name == LOG_DIR_PARTS[1]}

    return ignore


def review_skipped_files() -> tuple[tuple[str, bytes], ...]:
    """The files R14 reviews, every one of which the product must skip.

    Returns:
        tuple: ``(path, content)`` pairs: a binary and a file with a secret.
    """
    return ((REVIEW_SKIPPED_DIR + "/pixel.png", _TOY_PNG),
            (REVIEW_SKIPPED_DIR + "/leak.txt", _TOY_SECRET_FILE))


def _envelope(prompt: str, system: str) -> bytes:
    """Serialise a ``{prompt, system}`` request envelope.

    Args:
        prompt: The request text.
        system: The envelope's ``system`` field.

    Returns:
        bytes: The envelope as UTF-8 JSON, ready for standard input.
    """
    return json.dumps({"prompt": prompt, "system": system}).encode("utf-8")


#: The question R3 asks, and the one R2 asks without memory. ONE constant
#: for both, so the control cannot drift away from the run it controls.
RECALL_PROMPT = b"Which language do I prefer for systems programming?\n"

#: The token standing for the endpoint R6 fails against. It is resolved when
#: the run executes and never written into the table: a fixed URL is a promise
#: about whichever machine the harness happens to run on, and two versions of
#: that promise have already been measured false here.
#:
#: It replaced ``http://127.0.0.1:9/v1``, chosen as "the discard service,
#: reserved and unused". Where that service is actually RUNNING -- Windows
#: ships it and this repository measured it live -- the connection is accepted
#: and never answered, so R6 hung past its ceiling, was killed with no output,
#: and S10 reported CANNOT_TEST over three assertions it never got to search.
#: An ephemeral port bound and released fails the same way here: the machine
#: drops the packet rather than refusing it. And bounding the run with the
#: product's own ``--timeout`` finishes but carries neither the credential nor
#: the endpoint into the output, so all three assertions would pass over
#: nothing.
ERROR_BACKEND_TOKEN = "<error-backend>"

#: What the product substitutes from the vault into an authenticated URL.
#: Literals, because they are the product's own spelling and it refuses a
#: base_url that carries anything else.
USER_PLACEHOLDER = "[user]"
PASSWORD_PLACEHOLDER = "[password]"

#: The account name the placeholders resolve to. Only the PASSWORD is the
#: planted secret; the user half is ordinary and carries no marker.
BACKEND_USER = "smoke-probe"

#: What the local endpoint answers. 401 is what a real provider returns for a
#: credential it rejects, which is the path the redaction defect lived on.
ERROR_BACKEND_STATUS = 401

#: How long the handler waits on a request before giving up on reading it.
ERROR_BACKEND_TIMEOUT_S = 5.0


class _EchoingHandler(http.server.BaseHTTPRequestHandler):
    """Answers every request with an error that repeats what it was sent.

    Echoing the Authorization header is the whole point rather than a
    convenience: it PLANTS the credential on the product's error path, which
    is where the redaction defect this repository already fixed once actually
    lived. A backend that answered 401 with an empty body would only show that
    the product invents no leak; this one shows it does not repeat one it was
    handed.
    """

    #: Bounds a request that stops mid-headers, so a stuck client cannot hold
    #: the run open past its ceiling.
    timeout = ERROR_BACKEND_TIMEOUT_S

    def do_POST(self) -> None:  # noqa: N802 - the base class names it
        """Answer one POST with :data:`ERROR_BACKEND_STATUS`."""
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps({
            "error": {
                "message": "unauthorized",
                "authorization": self.headers.get("Authorization", ""),
                "url": self.path,
            }
        }).encode("utf-8")
        self.send_response(ERROR_BACKEND_STATUS)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        """Answer one GET the same way, so a probe gets an answer too."""
        self.do_POST()

    def log_message(self, fmt: str, *args) -> None:
        """Swallow the access log.

        The default writes every request to stderr, and the request line
        carries whatever the product put in the URL. That is the harness
        printing a credential the scenario is about to search for.

        Args:
            fmt: The format string the base class passes.
            *args: Its arguments.
        """


@contextlib.contextmanager
def error_backend():
    """Serve an error-answering endpoint on loopback for the block's duration.

    Yields:
        str: The base URL to point the product at, already ending in ``/v1``.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoingHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/v1" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=ERROR_BACKEND_TIMEOUT_S)

_BACKEND_URL_VARIABLE = "OPENAI_BASE_URL"
_BACKEND_KEY_VARIABLE = "OPENAI_API_KEY"

#: The eight shared runs, complete. They land in ONE place rather than a row at
#: a time as later tasks need them, because a run whose only owner is a task
#: nobody wrote never gets written at all -- which is exactly how R8 came to be
#: named in the design and claimed by no one. ``needed_runs`` decides which of
#: them actually execute, so declaring all eight costs nothing.
DEFINITIONS: dict[str, RunDefinition] = {
    "R1": RunDefinition(
        run_id="R1",
        argv=("query",) + COMMON_FLAGS + ("--auto",),
        stdin=(b"Use the ls tool to list the files in the current directory. "
               b"Then remember these two facts: I prefer Rust over Python for "
               b"systems programming, and this project is called Magi.\n"),
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    # R2 and R3 are one measurement, so they carry ONE prompt. R2
    # duplicated R1's planting prompt instead, four times longer, and S9
    # then subtracted two runs that differed in prompt length as well as
    # in memory: the 8-token "injection" it reported was measuring
    # neither. With one prompt on both sides, memory on against memory
    # off came out at 1668 against 392 input tokens against this
    # repository's own backend.
    "R2": RunDefinition(
        run_id="R2",
        argv=("query",) + COMMON_FLAGS + ("--no-memory",),
        stdin=RECALL_PROMPT,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    "R3": RunDefinition(
        run_id="R3",
        argv=("query",) + COMMON_FLAGS,
        stdin=RECALL_PROMPT,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    "R4": RunDefinition(
        run_id="R4",
        argv=("consult",) + COMMON_FLAGS + ("--timeout",
                                            str(LARGE_CONSULT_TIMEOUT_S),
                                            "--structured-verdicts"),
        stdin=b"Review the following source for correctness and risk.\n",
        # LARGE_CONSULT_CEILING_S against the product's own timeout is not a
        # second knob: the harness
        # ceiling has to sit ABOVE the product's, or the harness would kill the
        # very abandonment this run exists to observe. The margin covers
        # process start and the JSON write.
        needs_trio=True, payload_size=PAYLOAD_LARGE,
        timeout_s=LARGE_CONSULT_CEILING_S),
    "R5": RunDefinition(
        run_id="R5",
        # --auto is not a convenience: under the default tier the consult
        # tool is DENIED, and the only thing in tool_calls[] is that denial.
        # S19's subject is the SHAPE of the consult tool result, so a run
        # that cannot produce one leaves it reporting CANNOT_TEST forever.
        argv=("query",) + COMMON_FLAGS + ("--consult", "--auto"),
        stdin=b"Should this project adopt a logging framework? Consult first.\n",
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=420),
    "R6": RunDefinition(
        run_id="R6",
        argv=("query",) + COMMON_FLAGS,
        stdin=b"say ok\n",
        # Short on purpose: the invocation is designed to fail authentication,
        # and a run expected to fail fast must not be allowed to hang for three
        # minutes before saying so.
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        planted=(_R6_CREDENTIAL,),
        env=((_BACKEND_URL_VARIABLE, ERROR_BACKEND_TOKEN),
             (_BACKEND_KEY_VARIABLE, _R6_CREDENTIAL.value))),
    "R7": RunDefinition(
        run_id="R7",
        argv=("query",) + COMMON_FLAGS,
        stdin=b"say ok\n",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180,
        rotates=True),
    "R8": RunDefinition(
        run_id="R8",
        argv=("consult",) + COMMON_FLAGS,
        stdin=b"Is a single-line change worth a full review? Answer briefly.\n",
        # It touches the trio -- a consult that never consults emits no
        # envelope -- but carries the MINIMAL payload, because the shape of the
        # envelope does not depend on what was consulted.
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=300),
    # R9-R11 feed the envelope ``system`` field to the product (S44-S46).
    # R9 is a query with no override flag: the field must be ignored loudly,
    # and --log-level error proves the stderr warning does not depend on it.
    "R9": RunDefinition(
        run_id="R9",
        argv=("query",) + COMMON_FLAGS + ENVELOPE_FLAGS
        + ("--no-memory", "--log-level", "error"),
        stdin=_envelope("say ok", _ENVELOPE_SYSTEM),
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    # R10 is a full trio consult with the override flag; its timeout is R8's.
    "R10": RunDefinition(
        run_id="R10",
        argv=("consult",) + COMMON_FLAGS + ENVELOPE_FLAGS
        + ("--allow-system-override",),
        stdin=_envelope("Is a single-line change worth a full review? "
                        "Answer briefly.", _ENVELOPE_SYSTEM),
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=300),
    # R11 is refused before any model is reached, so it can be short.
    "R11": RunDefinition(
        run_id="R11",
        argv=("query",) + COMMON_FLAGS + ENVELOPE_FLAGS + ("--no-memory",),
        stdin=_envelope("say ok", "x" * (SYSTEM_CAP_DEFAULT_BYTES + 1)),
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60),
    # R12-R16 feed ``code-review`` (S47-S51). R12 reviews the generated tree in
    # at least three batches against the real trio; R13-R16 are refused before
    # any model is reached.
    "R12": RunDefinition(
        run_id="R12",
        argv=("code-review",) + REVIEW_FLAGS + ("--structured-verdicts",
                                                REVIEW_GEN_DIR),
        stdin=b"",
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=3000,
        files=review_input_files()),
    "R13": RunDefinition(
        run_id="R13",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_OUTSIDE_PATH,),
        stdin=b"",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60),
    "R14": RunDefinition(
        run_id="R14",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_SKIPPED_DIR,),
        stdin=b"",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_skipped_files()),
    "R15": RunDefinition(
        run_id="R15",
        argv=("code-review",) + REVIEW_FLAGS
        + ("--timeout", str(REVIEW_TOO_SHORT_TIMEOUT_S), "--log-level",
           "error", REVIEW_GEN_DIR),
        stdin=b"",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_input_files()),
    # MAGI FIX (S6 loop 3): the workspace ROOT, so ``.magi`` and ``.git`` are
    # enumeration candidates; R12-R15 read subdirectories and never were (F2).
    "R16": RunDefinition(
        run_id="R16",
        argv=("code-review",) + REVIEW_FLAGS
        + ("--timeout", str(REVIEW_TOO_SHORT_TIMEOUT_S), REVIEW_ROOT_PATH),
        stdin=b"",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_skipped_files()),
}


def needed_runs(registry, backend_reachable: bool) -> list[str]:
    """Which definitions actually have to execute.

    A run whose only consumers are scenarios that cannot execute is never paid
    for. With the backend down that is most of them, and executing them anyway
    would spend the expensive half of the harness to produce ``CANNOT_TEST``.

    Complexity: ``O(registered scenarios)``.

    Args:
        registry: The registered scenarios.
        backend_reachable: Whether the preflight reached the backend.

    Returns:
        list[str]: The ids to execute, in :data:`DEFINITIONS` order, so the
        expensive runs never reorder between invocations.

    Raises:
        HarnessError: If a scenario declares a run id the table does not carry.
            Silently skipping it would leave that scenario reading ``None``
            and reporting a product defect it never observed.
    """
    wanted: set[str] = set()
    for entry in registry.entries():
        if entry.run is None:
            continue
        if entry.needs_backend and not backend_reachable:
            continue
        ids = (entry.run,) if isinstance(entry.run, str) else entry.run
        for run_id in ids:
            if run_id not in DEFINITIONS:
                raise HarnessError(
                    "scenario %s declares run %r, which is not in the table"
                    % (entry.scenario_id, run_id)
                )
            wanted.add(run_id)
    return [run_id for run_id in DEFINITIONS if run_id in wanted]


class RunExecutor:
    """Runs the shared definitions and archives what they produced.

    It executes and archives; it never asserts. The separation is what lets the
    capture reach the scenarios at full fidelity while only the copy on disk is
    scrubbed -- and that order is the whole of the secret-hygiene design.
    """

    def __init__(self, binary, env, config, credential: str | None = None) -> None:
        """Bind the collaborators one run needs.

        Args:
            binary: The release binary under test.
            env: The persistent test environment.
            config: The harness configuration.
            credential: The real backend credential, for the run that rotates
                it and puts it back. ``None`` means "read it from the variable
                the configuration names, when it is needed"; the empty string
                means "there is none", which is refused rather than guessed at.
        """
        self._binary = binary
        self._env = env
        self._config = config
        self._credential = credential

    def execute(self, definition: RunDefinition) -> RunResult:
        """Run one definition and return everything it produced.

        A timeout comes back as a RESULT rather than an exception. Raising
        would abort the whole harness over one slow provider and lose every
        scenario that had nothing to do with it; the runner turns the flag into
        ``CANNOT_TEST`` for the scenarios that hang off this run, because a
        hang says nothing about their assertions.

        Args:
            definition: What to run.

        Returns:
            RunResult: The capture, the wall clock it took, whether it expired,
            and what it planted.

        Raises:
            HarnessError: If the run rotates a credential and there is none to
                put back afterwards, or if its review workspace cannot be
                built or git cannot see the files it declares.
            ProductOutputError: If the product could not be started. That is
                not a hang and not a verdict, so it is not folded into the
                result.
        """
        started = time.monotonic()
        workspace = self._prepare_review_workspace(definition)
        # A second listing, after the run's files are written and before the
        # product starts: what the product's own enumeration will see.
        protected = () if workspace is None else protected_candidates(workspace)
        stdin = self._stdin_for(definition)
        baseline = None
        try:
            if definition.rotates:
                # The baseline is taken INSIDE, after the guard that refuses to
                # rotate a credential there is nothing to put back. Reading the
                # environment before that guard runs touches it on the one path
                # that is supposed to have changed nothing.
                with self._rotated_credential() as counts:
                    baseline = counts
                    output = self._invoke(definition, stdin)
            else:
                output = self._invoke(definition, stdin)
            timed_out = False
        except TimedOut as expired:
            output = expired.output or ProductOutput(
                stdout=b"", stderr=b"", exit_code=0,
                command=self._command(definition))
            timed_out = True
        review_log = self._capture_review_log(definition)
        return RunResult(
            run_id=definition.run_id,
            output=output,
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            planted=definition.planted,
            stdin_bytes=len(stdin),
            baseline=baseline,
            review_log=review_log,
            protected_visible=protected,
        )

    def _review_workspace(self) -> pathlib.Path:
        """The directory :data:`REVIEW_WORKSPACE_TOKEN` stands for.

        Returns:
            pathlib.Path: ``<scratch>/review-ws``.
        """
        return pathlib.Path(self._env.scratch_dir) / REVIEW_WORKSPACE_DIR

    def _capture_review_log(self, definition: RunDefinition):
        """Read the review workspace's run log right after a run.

        MAGI FIX (S6 loop 3): the next review run rebuilds the workspace from
        nothing, so a scenario that read the live directory after the runs
        depended on being evaluated before a later review run wiped it (F4).

        Args:
            definition: The run that just finished.

        Returns:
            tuple[bytes, ...] | None: Each file's bytes under ``.magi/logs``,
            in sorted path order; ``None`` for a run without a review
            workspace or when the directory does not exist. A file that cannot
            be read is left out: the caller judges what is there.
        """
        if REVIEW_WORKSPACE_TOKEN not in definition.argv:
            return None
        directory = self._review_workspace().joinpath(*LOG_DIR_PARTS)
        if not directory.is_dir():
            return None
        captured = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            try:
                captured.append(path.read_bytes())
            except OSError:
                continue
        return tuple(captured)

    def _prepare_review_workspace(self, definition: RunDefinition):
        """Build the nested-repository workspace a review run executes in.

        It is rebuilt from nothing on every run: the environment's ``.magi``
        (configuration and the encrypted store with its vault) is copied in,
        ``git init`` makes the workspace its own repository, the run's files are
        written, and git is asked whether it can see them. Nothing is committed.

        Args:
            definition: The run about to execute.

        Returns:
            pathlib.Path | None: The workspace, or ``None`` for a run that does
            not use one.

        Raises:
            HarnessError: If the workspace cannot be built, or git cannot see
                the run's files.
        """
        if REVIEW_WORKSPACE_TOKEN not in definition.argv:
            return None
        workspace = self._review_workspace()
        scratch = pathlib.Path(self._env.scratch_dir).resolve()
        # MAGI FIX (S6 loop 3): containment is checked before anything is
        # deleted OR created. The tree is deleted wholesale and then rebuilt,
        # so only a path that resolves strictly under the scratch area may be
        # touched at all (F3).
        if scratch not in workspace.resolve().parents:
            raise HarnessError(
                "refusing to build the review workspace at %s: it resolves "
                "outside the scratch area %s" % (workspace, scratch))
        # MAGI FIX (S6 loop 10): copytree followed a link in .magi, and a
        # workspace path that is itself a link inside the scratch area passed
        # the resolve() check above (F1a-c4). Both are refused before anything
        # is deleted or copied. What is left is a parent swapped for a link
        # between the check and rmtree: the scratch area lives inside the
        # repository under the environment's lock, rmtree removes links inside
        # the tree without following them, and whoever can swap a directory
        # there already controls the harness's own code.
        if _is_link(workspace):
            raise HarnessError(
                "refusing to delete the review workspace at %s: it is a "
                "link" % workspace)
        _reject_links(self._env.magi_dir)
        try:
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True)
            # MAGI FIX (S6 loop 6): the environment's logs/ came along, so the
            # captured review log carried earlier runs' lines too (F1a-m4).
            shutil.copytree(self._env.magi_dir, workspace / ".magi",
                            ignore=_skip_top_level_logs(self._env.magi_dir))
            subprocess.run(git_command(workspace, "init", "-q"), check=True,
                           env=git_env(), capture_output=True,
                           timeout=GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError, shutil.Error) as exc:
            raise HarnessError("could not build the review workspace at %s: %s"
                               % (workspace, exc)) from exc
        self._write_files(definition, workspace)
        verify_visible(workspace, [path for path, _ in definition.files])
        return workspace

    def _write_files(self, definition: RunDefinition,
                     root: pathlib.Path) -> None:
        """Write the files a run reads under *root*.

        Args:
            definition: The run about to execute.
            root: The directory the paths are relative to.

        Raises:
            HarnessError: If a file cannot be written, or its path leaves
                *root*.
        """
        if not definition.files:
            return
        root = pathlib.Path(root).resolve()
        for relative, body in definition.files:
            # Containment is checked on the RESOLVED target: resolve() follows
            # every symlink and collapses every "..", so a link under *root*
            # that points outside lands outside and is refused like "../x".
            target = (root / relative).resolve()
            if root not in target.parents:
                raise HarnessError(
                    "run %s declares a file outside its workspace: %s"
                    % (definition.run_id, relative))
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            except OSError as exc:
                raise HarnessError(
                    "could not write %s for run %s: %s"
                    % (target, definition.run_id, exc)) from exc

    def archive(self, result: RunResult) -> None:
        """Write one run's command and streams to the environment, scrubbed.

        Called per result and immediately, never in a batch at the end: a
        harness that dies mid-run should leave the outputs it already had.

        The passphrase is scrubbed alongside whatever the definition declared.
        It is not in ``planted`` because it is not something one run puts in
        front of the product -- every run gets it -- but it reaches the child
        all the same, so the archive has to be clean of it too.

        Args:
            result: What to persist.

        Raises:
            HarnessError: If the directory or its files cannot be written.
        """
        directory = pathlib.Path(self._env.runs_dir) / result.run_id
        parts = {
            "command": " ".join(result.output.command).encode(
                "utf-8", errors="replace"),
            "stdout": result.output.stdout,
            "stderr": result.output.stderr,
            "exit_code": str(result.output.exit_code).encode("ascii"),
        }
        secrets = tuple(result.planted) + (self._passphrase_secret(),)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for name, body in parts.items():
                (directory / name).write_bytes(scrub(body, secrets))
        except OSError as exc:
            raise HarnessError(
                "could not archive run %s to %s: %s"
                % (result.run_id, directory, exc)
            ) from exc

    def _command(self, definition: RunDefinition) -> list[str]:
        """The argv for one definition, with the workspace resolved.

        Args:
            definition: The run.

        Returns:
            list[str]: The arguments, without the program name.
        """
        root = str(self._env.root)
        resolved = {WORKSPACE_TOKEN: root}
        if REVIEW_WORKSPACE_TOKEN in definition.argv:
            # MAGI FIX (S6 loop 11, F1a-m1): resolved, because the product starts with
            # cwd = this workspace and a relative -w would be joined to it a second time.
            resolved[REVIEW_WORKSPACE_TOKEN] = str(self._review_workspace().resolve())
        return [resolved.get(word, word) for word in definition.argv]

    def _invoke(self, definition: RunDefinition,
                stdin: bytes) -> ProductOutput:
        """Run the product once for *definition*.

        Args:
            definition: The run.
            stdin: What to hand the child, already built.

        Returns:
            ProductOutput: The capture, unscrubbed.

        Raises:
            ProductOutputError: If the product could not be started, or
                :class:`~smoke.errors.TimedOut` if it did not finish in time.
        """
        declared = dict(definition.env)
        if ERROR_BACKEND_TOKEN not in declared.values():
            return self._invoke_with(definition, declared, stdin)
        with self._authenticated_backend(definition) as url:
            resolved = {name: (url if value == ERROR_BACKEND_TOKEN else value)
                        for name, value in declared.items()}
            return self._invoke_with(definition, resolved, stdin)

    @contextlib.contextmanager
    def _authenticated_backend(self, definition: RunDefinition):
        """Serve an error backend the product must AUTHENTICATE against.

        The credential goes into the URL's authority, not only into a header,
        and that is the whole point of the run. The defect it guards lives in
        the vault percent-encoding a credential into the authority: a reserved
        character there moves the last ``@`` that ``redact_url`` anchors on,
        and the function used to return its input unchanged. A credential
        carried in ``Authorization`` never travels that path.

        The product refuses a literal credential in ``base_url`` -- measured,
        with a message that names the placeholder mechanism and does not echo
        the value -- so the credential travels through the vault and the URL
        carries placeholders. The entries are removed afterwards: one left
        behind makes the next run's endpoint authenticated by accident.

        Args:
            definition: The run, for the secret it declares planting.

        Yields:
            str: The authenticated URL to hand the product.
        """
        password = definition.planted[0].value if definition.planted else ""
        with error_backend() as url:
            # The plant sits INSIDE the try, so a `vault set` that fails
            # part-way still reaches the removal for the entries that did land.
            # Outside it, a failure on the third of four left the first two in
            # the persistent vault with no cleanup path -- and unlike the
            # rotation marker, nothing recovers these.
            planted: list[str] = []
            try:
                for name in PLACEHOLDER_ENTRIES:
                    self._vault_set(
                        name,
                        (password if name.endswith("PASSWORD")
                         else BACKEND_USER).encode("utf-8"))
                    planted.append(name)
                yield url.replace(
                    "http://",
                    "http://%s:%s@" % (USER_PLACEHOLDER, PASSWORD_PLACEHOLDER))
            finally:
                self._remove_placeholders(planted)

    def _remove_placeholders(self, planted: list[str]) -> None:
        """Take every placeholder back out, whatever happened to the others.

        Each removal gets its own attempt: one raising used to skip the rest,
        so a single slow `vault rm` left three entries behind. A failure here
        is reported once, after all four have been tried, because an entry left
        in the vault makes the NEXT run's endpoint authenticated by accident
        and that is worth surfacing rather than swallowing.

        Only what was actually planted is removed. Sweeping all four blindly
        would call ``vault rm`` on names this run never wrote, and a product
        that refuses to remove what is not there would then be reported as a
        leak that does not exist.

        It WARNS rather than raises, and the distinction is the whole point.
        This runs from a ``finally``, so raising replaces whatever exception
        is already in flight -- R6's own ``TimedOut`` among them, which the
        runner needs in order to substitute CANNOT_TEST for S10 -- and then
        escapes ``execute``. Main types only ``PreflightError`` and
        ``HarnessError``, so it would reach the last-resort catch and exit 3,
        which says the HARNESS failed. The event is one ``vault rm`` the
        product refused, and a leftover entry is inert besides: the preflight
        regenerates the environment's ``magi.toml`` on every run, so no
        ``base_url`` declares the placeholders that would read them back.

        Args:
            planted: The entries this run really wrote, in order.
        """
        failed = []
        for name in planted:
            try:
                self._vault_remove(name)
            except (OSError, ProductOutputError) as exc:
                failed.append("%s (%s)" % (name, exc))
        if failed:
            print("[runs] the run left placeholder entries in the vault: %s. "
                  "The next preflight sweeps them." % "; ".join(failed),
                  file=sys.stderr)

    def _invoke_with(self, definition: RunDefinition,
                     declared: dict, stdin: bytes) -> ProductOutput:
        """Run the product once with *declared* overlaid on its environment.

        Args:
            definition: The run.
            declared: The environment the definition asks for, already
                resolved.
            stdin: What to hand the child.

        Returns:
            ProductOutput: The capture, unscrubbed.

        Raises:
            ProductOutputError: If the product could not be started, or
                :class:`~smoke.errors.TimedOut` if it did not finish in time.
        """
        overlay = {PASSPHRASE_VARIABLE: self._config.passphrase}
        overlay.update(declared)
        # MAGI FIX (S6 loop 4): a review run starts IN its workspace, so its
        # relative input ('gen', '.', '../smoke-hostile-target') names the same
        # path whether the product joins it to -w or to its working directory;
        # R16's '.' was the harness's own directory otherwise (F1-2).
        cwd = (self._review_workspace()
               if REVIEW_WORKSPACE_TOKEN in definition.argv else None)
        return self._binary.invoke(
            self._command(definition), stdin=stdin,
            env=overlay, timeout=definition.timeout_s, cwd=cwd)

    def _stdin_for(self, definition: RunDefinition) -> bytes:
        """What actually goes on the run's standard input.

        ``payload_size`` was a declaration nothing read, so the one run that
        exists to carry a quarter of a megabyte went out with its 53-byte
        prompt -- and S8 would have asserted a token floor against a size the
        product never received, which S8's own trap text forbids. The
        declaration is load-bearing now: a run marked :data:`PAYLOAD_LARGE` is
        handed its prompt followed by the generated body.

        The size comes from the CONFIG and not from the module constant, so an
        operator who lowers ``[payload].target_bytes`` for a backend that
        cannot take the default gets the size they asked for rather than a copy
        that forgot to update.

        Args:
            definition: The run about to execute.

        Returns:
            bytes: The prompt, plus the generated payload for a large run.
        """
        if definition.payload_size != PAYLOAD_LARGE:
            return definition.stdin
        # A tree too small to build the payload is a fact about the CHECKOUT,
        # not a bug in the harness, and exit 3 is reserved for the latter. The
        # run goes out carrying its prompt alone, and S8 -- which compares what
        # the run CARRIED against what was declared -- reports CANNOT_TEST
        # naming the shortfall. That branch already existed and nothing could
        # reach it.
        # The executor's OWN binary, not the module-level accessor: this
        # class is handed everything it needs, and reaching for module
        # state here would make it depend on configure() having run.
        try:
            body = PayloadBuilder(self._binary.repo_root).build(
                self._config.payload_target_bytes)
        except PayloadTooSmall:
            return definition.stdin
        return definition.stdin + body

    def _passphrase_secret(self) -> PlantedSecret:
        """The environment's passphrase, as something the scrubber can remove.

        Returns:
            PlantedSecret: Labelled so a reader of an archived run can tell a
            scrubbed artifact from one that never carried the value.
        """
        return PlantedSecret(self._config.passphrase, "vault passphrase")

    @contextlib.contextmanager
    def _rotated_credential(self):
        """Rotate the backend credential for the body, and put it back after.

        Three pieces, each covering a different death. The marker is written
        BEFORE the credential moves, so a process killed in between leaves a
        vault the next preflight can recognise -- written afterwards it would
        be absent in exactly the case it exists for. The restore runs on
        BOTH exits -- the caller has an ``except`` and an ``else`` -- covering
        a failure and an orderly completion alike. What neither covers is a
        power cut, and that is what the marker is for.

        The marker is a separate entry recognised by NAME rather than a
        sentinel recognised by its value, and that is forced by the product:
        it never prints a stored value, so a preflight cannot read one back.

        Yields:
            dict | None: The per-table row counts taken before anything moved,
            so the scenario can tell "the history survived" from "some history
            is there". None when they could not be read.

        Raises:
            HarnessError: If there is no real credential to restore. Rotating a
                value the harness cannot put back would destroy it, so this is
                checked BEFORE anything is written.
        """
        credential = self._real_credential()
        if not credential:
            raise HarnessError(
                "the run that rotates the backend credential has nothing to "
                "put back: export %s, which smoke.toml names as the variable "
                "holding it" % self._config.backend_key_env
            )
        name = self._config.backend_key_env
        baseline = self._history_counts()
        try:
            # Typed for the same reason the restore is: reading the exit code
            # widened what these can raise from "could not start the product"
            # to "the product ran and refused", and an untyped one here
            # escapes execute into main's last-resort catch -- a traceback and
            # exit 3, which says the HARNESS failed.
            self._vault_set(ROTATION_MARKER, ROTATION_MARKER_VALUE)
        except (OSError, ProductOutputError) as exc:
            raise HarnessError(
                "the rotation could not write its marker, so a death part "
                "way through would leave no way to recover: %s" % exc
            ) from exc
        try:
            self._vault_set(name, ROTATION_SENTINEL)
            yield baseline
        except BaseException:
            # The LANGUAGE answers "was something already failing?", rather
            # than sys.exc_info(), which reports the innermost exception being
            # handled anywhere on the stack -- so a caller that happened to be
            # inside an unrelated except block would have made the restore
            # suppress a genuine failure and blame an exception that had
            # already finished. No production caller does that today, which is
            # exactly why it would have gone unnoticed.
            self._restore(name, credential, in_flight=True)
            raise
        else:
            self._restore(name, credential, in_flight=False)

    def _restore(self, name: str, credential: str,
                 in_flight: bool) -> None:
        """Put the real credential back and drop the marker.

        Runs on both exits of the rotation, and the three rules that follow
        from that were learned the expensive way on the placeholder path.

        **It raises a TYPED error.** A ``ProductOutputError`` escaping here
        reaches main's last-resort catch, which is a traceback and exit 3 --
        the code reserved for a defect in the HARNESS -- with no report and
        every finding from every completed run discarded. The event is a
        ``vault set`` the product refused.

        **It stops the run, unlike a leftover placeholder.** Not because the
        sentinel would be used -- it would not. The product resolves the key
        env-first (``resolve_openai_key``), step 3 of the preflight requires
        the real value in the environment, and every child inherits it, so a
        vault entry holding the sentinel is shadowed. What makes this
        different is the MARKER: the rotation writes one before it moves the
        credential, precisely so a death in the middle is recoverable, and
        the restore is what clears it. A restore that quietly failed while
        the removal succeeded leaves the environment rotated with nothing
        left to say so -- the one state the marker's ordering exists to
        prevent.

        **It never replaces an exception already in flight.** R7's own
        ``TimedOut`` is what the runner needs in order to substitute
        CANNOT_TEST for the scenarios that read the run.

        Args:
            name: The variable the backend credential lives under.
            credential: The real value to put back.
            in_flight: Whether the run was already failing when this ran. The
                caller knows, because it has an ``except`` and an ``else``;
                asking ``sys.exc_info()`` here would answer a broader
                question -- the innermost exception being handled anywhere on
                the stack, unrelated ones included.

        Raises:
            HarnessError: If a vault write failed and nothing else was
                already failing.
        """
        try:
            self._vault_set(name, credential.encode("utf-8"))
            self._vault_remove(ROTATION_MARKER)
        except (OSError, ProductOutputError) as exc:
            if in_flight:
                print("[runs] the rotated backend credential could not be "
                      "restored (%s), and the run was already failing. The "
                      "next preflight restores it from %s."
                      % (exc, name), file=sys.stderr)
                return
            raise HarnessError(
                "the rotated backend credential could not be restored: %s. "
                "The environment still holds the sentinel, so every backend "
                "run would fail to authenticate. The next preflight restores "
                "it from %s; if that also fails, run --reset-env."
                % (exc, name)
            ) from exc

    def _real_credential(self) -> str:
        """The credential the rotation has to be able to restore.

        Resolved lazily rather than in the constructor: only one run rotates,
        so every other executor would otherwise read a variable it never uses.

        Returns:
            str: What was handed in, or the value of the variable the
            configuration names. An explicitly empty string is an answer --
            "there is none" -- and never falls back to the environment.
        """
        if self._credential is not None:
            return self._credential
        return os.environ.get(self._config.backend_key_env, "")

    def _history_counts(self) -> dict | None:
        """The environment's per-table row counts, before the run touches it.

        Read through ``vault diagnose``, which needs NO passphrase (REQ-H32) --
        the only reason a baseline can be taken at all without unlocking the
        very envelope the run is about to rotate around.

        Returns:
            dict | None: Table name to row count, or None when the report
            could not be read. None means NOT MEASURED, and the scenario that
            reads it refuses rather than assuming zero.
        """
        try:
            output = self._binary.invoke(
                [_VAULT_SUBCOMMAND, "-w", str(self._env.root), "diagnose"],
                stdin=b"", env={}, timeout=VAULT_TIMEOUT_S)
        except (OSError, ProductOutputError, TimedOut):
            return None
        if output.exit_code != 0:
            return None
        return diagnose_counts(output.stdout) or None

    def _vault_set(self, name: str, value: bytes) -> None:
        """Store one secret, overwriting without asking.

        Args:
            name: The entry's name.
            value: The value, fed on standard input -- never as an argument,
                which any process on the machine could read.

        Raises:
            ProductOutputError: If the product could not be run, or ran and
                refused. The second half is not pedantry: ``invoke`` treats a
                non-zero exit as an answer rather than an error, so without
                this check a ``vault set`` the product REFUSED returned
                normally and the rotation reported a success that never
                happened.
        """
        self._require_zero(self._binary.invoke(
            [_VAULT_SUBCOMMAND, _SET_SUBCOMMAND, name, _FORCE_FLAG,
             "-w", str(self._env.root)],
            stdin=value,
            env={PASSPHRASE_VARIABLE: self._config.passphrase},
            timeout=VAULT_TIMEOUT_S), _SET_SUBCOMMAND, name)

    def _vault_remove(self, name: str) -> None:
        """Delete one secret, skipping the confirmation.

        Args:
            name: The entry's name.

        Raises:
            ProductOutputError: If the product could not be run, or ran and
                refused. Same reason as :meth:`_vault_set`.
        """
        self._require_zero(self._binary.invoke(
            [_VAULT_SUBCOMMAND, _REMOVE_SUBCOMMAND, name, _FORCE_FLAG,
             "-w", str(self._env.root)],
            stdin=b"",
            env={PASSPHRASE_VARIABLE: self._config.passphrase},
            timeout=VAULT_TIMEOUT_S), _REMOVE_SUBCOMMAND, name)

    @staticmethod
    def _require_zero(output, verb: str, name: str):
        """Turn a refusal into the error the callers already handle.

        Args:
            output: What the invocation captured.
            verb: The vault subcommand, for the message.
            name: The entry it acted on.

        Raises:
            ProductOutputError: When it exited non-zero, carrying the
                product's own stderr rather than a summary of it.
        """
        if output.exit_code != 0:
            raise ProductOutputError(
                "vault %s %s exited %d: %s"
                % (verb, name, output.exit_code,
                   output.stderr.decode("utf-8", errors="replace").strip()))
