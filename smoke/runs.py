# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
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
import io
import itertools
import json
import os
import pathlib
import selectors
import shutil
import socket
import stat
import string
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse

from smoke.binary import LOOPBACK_HOSTS, NO_PROXY_VARIABLE
from smoke.config import PLACEHOLDER_ENTRIES, ROTATION_MARKER
from smoke.env import MAGI_DIR_NAME, MAGI_TOML_NAME
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
            the definition stays immutable. A ``None`` value removes the
            variable instead (:func:`smoke.binary.child_environment`). The passphrase is NOT here: every
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
        forbids_model_calls: Whether the run must end before any model is
            contacted. Its exit code and stderr cannot show that -- a product
            that paid for a model and then refused would read the same -- so
            the executor points every endpoint the review workspace declares
            at a local counter (:func:`model_counter`) and records each
            request past the capability probe on the result, and the runner
            fails every scenario that reads the run when there is one. It
            needs the review workspace, whose configuration is the one
            rewritten.
        unreachable_embedder: Whether the review workspace's ``[embedding]``
            is pointed at :data:`UNREACHABLE_EMBEDDER_URL` before the run
            (:func:`point_embedding_at`), so documentation and history cannot
            be retrieved while the trio still answers. It needs the review
            workspace too.
    """

    run_id: str
    argv: tuple[str, ...]
    stdin: bytes
    needs_trio: bool
    payload_size: str
    timeout_s: int
    planted: tuple[PlantedSecret, ...] = ()
    env: tuple[tuple[str, str | None], ...] = ()
    rotates: bool = False
    files: tuple[tuple[str, bytes], ...] = ()
    forbids_model_calls: bool = False
    unreachable_embedder: bool = False

    def __post_init__(self) -> None:
        """Refuse files, or a model-call ban, on a run without a review workspace.

        Raises:
            HarnessError: When ``files`` is not empty, or
                ``forbids_model_calls`` is set, and ``argv`` does not carry
                :data:`REVIEW_WORKSPACE_TOKEN`: only the review workspace is
                written, so the files would silently never exist and the ban
                would guard nothing.
        """
        # MAGI FIX (S6 loop 14): such files were ignored, and the run executed
        # without its declared input (F1a1-m1).
        if self.files and REVIEW_WORKSPACE_TOKEN not in self.argv:
            raise HarnessError(
                "run %s declares files but no %s in its argv, so nothing would "
                "write them" % (self.run_id, REVIEW_WORKSPACE_TOKEN))
        # MAGI FIX (S6 loop 19): see forbids_model_calls (F1a1-balthasar).
        if self.forbids_model_calls and REVIEW_WORKSPACE_TOKEN not in self.argv:
            raise HarnessError(
                "run %s forbids model calls but has no %s whose endpoints "
                "could be counted" % (self.run_id, REVIEW_WORKSPACE_TOKEN))
        if self.unreachable_embedder and REVIEW_WORKSPACE_TOKEN not in self.argv:
            raise HarnessError(
                "run %s points its embedder away but has no %s whose "
                "configuration could be rewritten" % (self.run_id, REVIEW_WORKSPACE_TOKEN))


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
    #: The entries under ``.magi/logs`` left out of :attr:`review_log`, each as
    #: ``"<name> (<why>)"`` like :func:`smoke.logs.scan`'s list: a file that
    #: could not be read or was over a capture cap, a link, or an entry that is
    #: not a regular file. A scenario
    #: that finds nothing in :attr:`review_log` while this is not empty cannot
    #: tell a product that never logged from a file it was refused.
    review_log_unreadable: tuple[str, ...] = ()
    #: The ``.magi/`` paths git lists in the review workspace right before the
    #: run, with the product's own query (:data:`GIT_VISIBLE_QUERY`), sorted.
    #: Empty when the run uses no review workspace, or when an ignore rule
    #: (a global ``core.excludesFile``, say) hides ``.magi/`` from git and so
    #: from the product: then no protected path was an enumeration candidate.
    protected_visible: tuple[str, ...] = ()
    #: ``"<METHOD> <path>"`` of every request a run that forbids model calls
    #: sent past the capability probe (:data:`PROBE_PATHS`), in arrival order.
    #: Empty for every other run. Any entry is a product defect, which the
    #: runner reports as ``FAIL`` for every scenario reading the run.
    model_requests: tuple[str, ...] = ()
    #: How many requests the run's loopback error backend
    #: (:data:`ERROR_BACKEND_TOKEN`, R6) answered; ``None`` for a run without
    #: one. Zero means the product never reached it, so nothing it printed
    #: walked the error path the backend plants a credential on.
    backend_requests: int | None = None


#: Where the product writes its logs, relative to a workspace root. The one
#: definition: :mod:`smoke.logs` re-exports it, because it imports this module
#: and this module cannot import it back.
LOG_DIR_PARTS = (MAGI_DIR_NAME, "logs")

#: Where a workspace's configuration lives, relative to its root.
CONFIG_PATH_PARTS = (MAGI_DIR_NAME, MAGI_TOML_NAME)

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
#: R17's 300 s is provisional too, not a measurement: one batch over one small
#: file (R10's 67 s consult), plus the trio per batch and the gate, rounded up.
#: Replace it with R17's own wall clock at the first run against a real backend.
#: R18's 300 s is the same provisional figure as R17's: one batch over one
#: four-line file and the gate.
TRIO_RUNS_MEASURED_S: dict[str, int] = {"R4": 283, "R5": 170, "R8": 67,
                                        "R10": 67, "R12": 1500, "R17": 300,
                                        "R18": 300}

#: The factor a trio run's ``timeout_s`` must reach over its measurement in
#: :data:`TRIO_RUNS_MEASURED_S`: at least this many times it, equality
#: included (R12's provisional 1500 s against 3000 s sits exactly on it; when
#: its real figure replaces it, keep it at or under half the timeout). Only
#: ``smoke/tests/test_runs.py`` reads it; the executor compares nothing.
#: The cloud endpoint's pace varies by time slot (PM-S-8) and the
#: measurement above is a single run in one slot -- a US-morning slot
#: measured slower elsewhere in this project's history -- so the harness
#: timeout carries double that single sample rather than trusting it at
#: face value.
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
#:
#: It is ONE fixed path, rebuilt from nothing before each review run and read
#: right after it, so the review runs must execute one at a time: two at once
#: would delete each other's workspace mid-run. They do, because the executor
#: runs every definition in turn (``smoke/__main__.py``) under the
#: environment's lock; anything that runs definitions in parallel has to give
#: each review run a workspace of its own first.
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
#: that carries none of those, so the harness drops every one.
GIT_ENV_PREFIX = "GIT_"

#: Every ``GIT_*`` variable the product's git runs DO carry: the two its tool
#: allowlist inherits (``TOOL_ENV_ALLOWLIST``, ``src/tools/proc_group.rs``)
#: and the three its scrubbed spawn sets (``src/system/review_git.rs``). None
#: changes what :data:`GIT_VISIBLE_QUERY` lists: ``GIT_EXEC_PATH`` only says
#: where git's helpers live, ``GIT_TEMPLATE_DIR`` only feeds ``git init``
#: (the harness's, which drops it; the product never runs ``init``),
#: ``GIT_NO_LAZY_FETCH`` only stops a partial clone fetching, and
#: ``GIT_LITERAL_PATHSPECS`` only reads a pathspec, which the query has none
#: of, and ``GIT_GRAFT_FILE`` only changes commit ancestry, which a listing of
#: the index and the working tree never reads. So :func:`git_env` drops them
#: with the rest. ``test_rust_sync.py`` fails when the product's set moves, so
#: a new variable is assessed here.
PRODUCT_GIT_VARIABLES = frozenset({"GIT_EXEC_PATH", "GIT_TEMPLATE_DIR",
                                   "GIT_NO_LAZY_FETCH", "GIT_LITERAL_PATHSPECS",
                                   "GIT_GRAFT_FILE"})


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


def _error_detail(exc: BaseException) -> str:
    """The text of *exc*, with the stderr a subprocess error captured.

    ``CalledProcessError`` and ``TimeoutExpired`` keep the child's stderr on
    the exception but leave it out of ``str()``, so a harness error built from
    ``str(exc)`` alone loses the child's own reason.

    Args:
        exc: The error to describe.

    Returns:
        str: ``str(exc)``, then ``": <stderr>"`` when the error carries a
        non-empty captured stderr, decoded as UTF-8 with replacement.

    Example:
        >>> _error_detail(subprocess.CalledProcessError(1, "git", stderr=b"bad"))
        "Command 'git' returned non-zero exit status 1.: bad"
    """
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    stderr = (stderr or "").strip()
    return "%s: %s" % (exc, stderr) if stderr else str(exc)

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

#: The name of R14's secret file: the toy key itself. Since T3.7 (RV22 amended)
#: a secret in a file's CONTENT is masked and the file reviewed, which would
#: reach a model; a secret in its PATH still leaves the file out unread, since
#: the path is the file's identity and cannot be masked.
_TOY_SECRET_NAME = _TOY_SECRET_FILE.split(b"=", 1)[1].strip().decode("ascii") + ".txt"

#: How the product names that file in an omission list (``display_path`` in
#: ``src/review/secrets.rs``): the whole path redacted, by its pattern.
REVIEW_SECRET_PATH_SHOWN = "<path redacted: anthropic-key>"


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


#: Upper-case ASCII letters to their lower-case forms, and nothing else.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

#: What the product trims off the end of a path component before comparing.
TRIMMED_TRAILING = ". "


def same_product_name(component: str, name: str) -> bool:
    """Whether *component* names *name* under the product's spelling rule.

    The product (``is_never_reviewed`` in ``src/review/source.rs``) trims
    trailing ``.`` and `` `` and compares ASCII-case-insensitively, as Windows
    resolves ``.MAGI`` and ``.magi.`` to ``.magi``. Only ASCII letters fold:
    ``str.lower`` would also fold letters such as the Kelvin sign.

    Args:
        component: One path component, as listed.
        name: The lower-case name it is compared with.

    Returns:
        bool: ``True`` when the two name the same entry for the product.

    Example:
        >>> same_product_name(".MAGI.", ".magi"), same_product_name(".magic", ".magi")
        (True, False)
    """
    return component.rstrip(TRIMMED_TRAILING).translate(_ASCII_LOWER) == name


def protected_candidates(workspace) -> tuple[str, ...]:
    """The ``.magi/`` paths git lists in *workspace*, as the product would see them.

    Args:
        workspace: The review workspace (a git repository of its own).

    Returns:
        tuple[str, ...]: The listed paths with any component that is
        ``.magi`` as the product spells it (:func:`same_product_name`), with
        ``/`` and ``\\`` both read as separators, as the product splits them;
        sorted.

    Raises:
        HarnessError: If git fails.
    """
    # MAGI FIX (S6 loop 9): S51 passed vacuously when an ignore rule hid
    # .magi/ from git and so from the product (F1a-c1, F1a-m2).
    # MAGI FIX (S6 loop 16): only an exact lowercase ".magi/" counted, while the
    # product folds ASCII case and trims trailing dots and spaces (F1a1-caspar).
    # MAGI FIX (S6 loop 18): only the first component was tested, while the
    # product's is_never_reviewed tests every one (F1a1-caspar).
    return tuple(sorted(
        path for path in visible_paths(workspace)
        if any(same_product_name(component, LOG_DIR_PARTS[0])
               for component in path.replace("\\", "/").split("/"))))


#: The repository the harness lives in, two levels above this file, resolved
#: the way ``smoke/__main__.py`` resolves it, so the scratch area it derives
#: lies lexically under this path. No component from here down to the review
#: workspace may be a link.
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent


#: The reparse tag of a Windows junction (``IO_REPARSE_TAG_MOUNT_POINT``,
#: ``0xA0000003``). The ``stat`` module exports it on Windows since Python 3.8,
#: so it is set on the 3.11 floor (checked against a real junction by
#: ``test_a_junction_is_a_link_without_os_path_isjunction``); off Windows it is
#: ``None`` and ``os.lstat`` reports no ``st_reparse_tag``, so nothing is a
#: junction there. The constant comes from the C module ``_stat``, which
#: defines the ``IO_REPARSE_TAG_*`` names only on Windows; ``Lib/stat.py``
#: defines just the ``FILE_ATTRIBUTE_*`` ones on every platform (checked on
#: Linux with 3.13 and on Windows with 3.11 and 3.15, MAGI S6 loop 19).
JUNCTION_REPARSE_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)


def _is_link(path) -> bool:
    """Whether *path* is a symlink or, on Windows, a junction.

    ``os.path.isjunction`` exists from Python 3.12. On the harness's 3.11
    floor the junction is read from the reparse tag ``os.lstat`` reports on
    Windows (``st_reparse_tag``, there since Python 3.8), so a junction is a
    link on every supported version.

    Args:
        path: The path to test, which need not exist.

    Returns:
        bool: ``True`` for a symlink or a junction.

    Raises:
        OSError: When the path exists and cannot be inspected (``os.lstat``);
            a missing path, or one under a file, is not a link.
    """
    if os.path.islink(path):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None:
        return bool(is_junction(path))
    # MAGI FIX (S6 loop 13): on 3.11 a junction read as an ordinary directory,
    # so copytree and rmtree went through it (F1a-m3, F1a-b2, F1a-c2).
    if JUNCTION_REPARSE_TAG is None:
        return False
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return tag == JUNCTION_REPARSE_TAG


def _linked_component(base, relative):
    """The first link on the way from *base* down to ``base / relative``.

    *base* itself is not tested; callers pick it with
    :func:`_chain_from_repository`, so every component below the repository
    root is.

    Args:
        base: The directory the walk starts under.
        relative: The path below *base*, one or more components.

    Returns:
        pathlib.Path | None: The first component that is a symlink or a
        junction (see :func:`_is_link`), or ``None`` when there is none.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _linked_component(pathlib.Path(tmp), "a/b") is None
        True
    """
    path = pathlib.Path(base)
    for part in pathlib.PurePath(relative).parts:
        path = path / part
        if _is_link(path):
            return path
    return None


def _chain_from_repository(path) -> tuple[pathlib.Path, pathlib.PurePath]:
    """Where a link walk down to *path* starts, and the components it tests.

    A path under :data:`REPOSITORY_ROOT` (the scratch area always is) is
    walked from the repository root, so every component from there down,
    the scratch area included, is tested. The harness's scratch area is
    always under it, even in a checkout reached through a link: both this
    root and ``smoke/__main__.py``'s environment root are derived from the
    RESOLVED location of the package. Any other path (a test double's
    temporary scratch area) is walked from its own parent, so its last
    component is still tested; callers append what lies below it. The path
    is made absolute without resolving links: resolving would walk through
    the very link being looked for.

    Args:
        path: The path the walk ends at.

    Returns:
        tuple[pathlib.Path, pathlib.PurePath]: The untested base, and the path
        below it whose every component :func:`_linked_component` tests.

    Example:
        >>> base, below = _chain_from_repository(REPOSITORY_ROOT / "smoke" / "scratch")
        >>> base == REPOSITORY_ROOT, below.as_posix()
        (True, 'smoke/scratch')
    """
    absolute = pathlib.Path(os.path.abspath(path))
    try:
        return REPOSITORY_ROOT, absolute.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return absolute.parent, pathlib.PurePath(absolute.name)


def _resolved(path, what: str) -> pathlib.Path:
    """*path* with every link and ``..`` resolved, or a harness error naming *what*.

    MAGI FIX (S6 loop 28): ``Path.resolve`` raises :class:`RuntimeError` on a
    symlink loop (an :class:`OSError` from Python 3.13) and an
    :class:`OSError` on a component it cannot inspect, and the executor
    called it outside every guard, so the run ended in a traceback instead of
    a named harness failure (F1a1-caspar).

    Args:
        path: The path to resolve; it need not exist.
        what: What the path is, for the message.

    Returns:
        pathlib.Path: The resolved path.

    Raises:
        HarnessError: When the path cannot be resolved.
    """
    try:
        return pathlib.Path(path).resolve()
    except (OSError, RuntimeError) as exc:
        raise HarnessError("could not resolve %s %s: %s" % (what, path, exc)) from exc


def _linked_ancestor(path):
    """The first link from the repository root down to *path*, *path* itself excluded.

    Args:
        path: A path whose parents must hold no link (:func:`_chain_from_repository`).

    Returns:
        pathlib.Path | None: The first linked component above *path*, or ``None``.

    Raises:
        OSError: When a component cannot be inspected.
    """
    base, below = _chain_from_repository(path)
    if len(below.parts) < 2:
        return None
    return _linked_component(base, below.parent)


def _walk_failed(error: OSError) -> None:
    """The ``onerror`` of :func:`_reject_links`'s walk: refuse what it cannot list.

    Args:
        error: The error ``os.walk`` met listing a directory.

    Raises:
        HarnessError: Always, naming the error.
    """
    raise HarnessError("refusing to copy the environment's .magi: could not "
                       "list a directory to check it for links: %s" % error)


def _reject_links(top) -> None:
    """Refuse a tree that holds a symlink, a junction or a hard-linked file, *top* included.

    The review workspace is copied from the environment's ``.magi``, and an
    entry pointing outside would put what it points at into the workspace
    the product reads. The environment never needs one, so any link is
    refused rather than copied as a link or followed. This is the check made
    before the old workspace is deleted; the copy itself
    (:func:`_copy_tree_without_links`) refuses the same entries again, so a
    link swapped in after this walk is refused there.

    Args:
        top: The directory to check.

    Raises:
        HarnessError: Naming the first link or hard-linked file found, or a directory the walk
            could not list (a link inside it could not be ruled out).
        OSError: When a link test itself fails; the executor turns it into a
            :class:`HarnessError`.
    """
    if _is_link(top):
        raise HarnessError("refusing to copy %s: it is a link" % top)
    # MAGI FIX (S6 loop 13): os.walk skipped a directory it could not list, so a
    # link inside it went unchecked; the walk now fails instead (F1a-c1).
    for directory, dirs, files in os.walk(top, onerror=_walk_failed):
        for name in dirs + files:
            path = os.path.join(directory, name)
            if _is_link(path):
                raise HarnessError(
                    "refusing to copy %s: %s is a link, and the review "
                    "workspace never follows one" % (top, path))
        # MAGI FIX (S6 loop 29): a hard link is a regular file to every link
        # test, so one to a file outside the environment was copied as that
        # file's bytes (F1a1b-caspar). The log capture already refused them.
        for name in files:
            path = os.path.join(directory, name)
            if os.lstat(path).st_nlink != 1:
                raise HarnessError("refusing to copy %s: %s is %s"
                                   % (top, path, HARD_LINKED))


#: Whether this platform's ``chmod`` can change a link itself rather than what
#: it points at (``follow_symlinks=False``): Windows from Python 3.13, macOS,
#: the BSDs. Read once, at import: the platform does not change mid-run.
_CHMOD_SKIPS_LINKS = os.chmod in os.supports_follow_symlinks

#: Whether :func:`_set_mode` can set a mode on a descriptor opened with
#: ``O_NOFOLLOW`` instead (Linux, where ``chmod`` follows every link).
_CHMOD_BY_DESCRIPTOR = hasattr(os, "O_NOFOLLOW") and os.chmod in os.supports_fd


def _set_mode(path, mode: int) -> None:
    """Set the mode of *path* without following a link that stands there now.

    MAGI FIX (S6 loop 27): :func:`_retry_writable` tested for a link and then
    changed the mode by path, so a link swapped in between had the mode of
    whatever it points at changed (F1a1-caspar). Where ``chmod`` can leave a
    link unfollowed (Windows from Python 3.13, macOS and the BSDs) it is told
    to; where it cannot but ``O_NOFOLLOW`` exists (Linux), the mode is set on
    a descriptor opened without following, so a link fails the open instead.
    Anywhere else (Windows before 3.13) ``chmod`` changes the link's own
    attribute and never what it points at.

    Args:
        path: The entry.
        mode: The new mode.

    Raises:
        OSError: When the mode cannot be set, a link at *path* included on
            Linux (``ELOOP``), or an entry that cannot be opened for reading.
    """
    if _CHMOD_SKIPS_LINKS:
        os.chmod(path, mode, follow_symlinks=False)
    elif _CHMOD_BY_DESCRIPTOR:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW
                             | getattr(os, "O_NONBLOCK", 0))
        try:
            os.chmod(descriptor, mode)
        finally:
            os.close(descriptor)
    else:
        os.chmod(path, mode)


def _retry_writable(func, path, error: BaseException) -> None:
    """The deletion error handler of :func:`_remove_tree`: retry once, writable.

    On Windows a read-only file (or directory) cannot be deleted, and
    :func:`shutil.copytree` keeps the read-only bit it copies, so one such
    entry would stop every later rebuild. The bit is cleared and the deletion
    retried once. A link is never touched: changing its mode would change
    what it points at, so the original error stands; and the mode is set by
    :func:`_set_mode`, which does not follow a link that replaced the entry
    after that test.

    The recovery is Windows-only in effect. On POSIX a deletion fails with
    :class:`PermissionError` because the PARENT directory is not writable,
    which a mode change of the entry does not fix, so the retry fails again
    and that error propagates; the executor reports it as a harness failure
    (exit 3). A directory is made ``0o700`` and a file ``0o600``: ``0o600``
    on a directory would drop its search bit.
    The harness writes every file of the workspace itself, so this is not
    expected to happen.

    Args:
        func: The function that failed (``os.unlink``, ``os.rmdir``...).
        path: The path it failed on.
        error: The error it raised.

    Raises:
        BaseException: *error* itself, for a link or anything other than a
            :class:`PermissionError`; whatever the retry raises otherwise.
    """
    if not isinstance(error, PermissionError) or _is_link(path):
        raise error
    is_directory = stat.S_ISDIR(os.lstat(path).st_mode)
    _set_mode(path, stat.S_IRWXU if is_directory else stat.S_IREAD | stat.S_IWRITE)
    func(path)


def _remove_tree(path) -> None:
    """Delete the tree at *path*, clearing a read-only bit that blocks it.

    ``shutil.rmtree`` takes the handler as ``onexc`` from Python 3.12 and as
    the deprecated ``onerror`` (with an ``exc_info`` triple) before it; both
    reach :func:`_retry_writable`. Links inside the tree are unlinked, never
    followed, as ``shutil.rmtree`` does on its own.

    Args:
        path: The tree to delete; it must not itself be a link (the caller
            refuses that first).

    Raises:
        OSError: When an entry cannot be deleted even writable.
    """
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry_writable)
    else:
        shutil.rmtree(path, onerror=lambda func, failed, info:
                      _retry_writable(func, failed, info[1]))


#: The largest run-log file :meth:`RunExecutor._capture_review_log` reads.
#: The review runs that scenarios inspect stop before any model call, so their
#: log is a few lines; a file this large means a faulty or hostile product,
#: and reading it whole could exhaust the harness before the scenario reports.
MAX_REVIEW_LOG_FILE_BYTES = 16 * 1024 * 1024

#: The most run-log bytes one capture keeps, over all files. Holds at least
#: one file of :data:`MAX_REVIEW_LOG_FILE_BYTES`.
MAX_REVIEW_LOG_TOTAL_BYTES = 64 * 1024 * 1024

#: Why a capture leaves out a file over :data:`MAX_REVIEW_LOG_FILE_BYTES`.
OVER_FILE_CAP = "larger than %d bytes, not read"

#: Why a capture leaves out a file once :data:`MAX_REVIEW_LOG_TOTAL_BYTES` is spent.
OVER_TOTAL_CAP = "past the %d byte capture total, not read"


#: Why :func:`_regular_files_without_links` leaves a link out.
LINK_NOT_FOLLOWED = "a link, not followed"

#: Why :func:`_regular_files_without_links` leaves out a fifo, socket or device.
NOT_A_REGULAR_FILE = "not a regular file, not read"

#: Why a capture leaves out a file that, once opened, is not the one the walk
#: listed (:func:`_open_listed`).
CHANGED_AFTER_LISTING = "changed after it was listed, not read"

#: Why a capture leaves out a file with more than one link. A hard link is a
#: regular file to every test and its identity is the one the walk lists, so
#: nothing else tells a link to a file outside the workspace from the run's
#: own log; the product never writes one.
HARD_LINKED = "a file with more than one link, not read"

#: The most entries one capture lists under ``.magi/logs``, files,
#: directories and links alike. The byte caps bound what is read, not what is
#: listed: a faulty product could write a vast number of empty files, each one
#: materialised as an entry at no captured byte. The runs scenarios inspect
#: write a few files, so this leaves them whole.
MAX_REVIEW_LOG_ENTRIES = 4096

#: Why a capture stops listing: the directory named holds entries past
#: :data:`MAX_REVIEW_LOG_ENTRIES`, which are neither listed nor read.
OVER_ENTRY_CAP = "more than %d entries under the log directory, the rest not listed"

#: The flags :func:`_open_listed` opens a log file with. ``O_NOFOLLOW`` and
#: ``O_NONBLOCK`` exist on POSIX only: there a link is refused by the open
#: itself and a fifo cannot block it. Windows has neither, so there the
#: identity check after the open is what refuses a swapped entry.
_LOG_OPEN_FLAGS = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                   | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))


def _identity(status) -> tuple[int, int]:
    """The device and file number of a stat result: what names one file.

    Args:
        status: An ``os.stat_result``.

    Returns:
        tuple[int, int]: ``(st_dev, st_ino)``. On Windows they are the volume
        serial number and the file index, which ``os.lstat`` and ``os.fstat``
        report alike.
    """
    return status.st_dev, status.st_ino


def _open_listed(path, expected):
    """Open the log file *path*, only if it is still the file *expected* names.

    MAGI FIX (S6 loop 19): the file was opened again by path, following
    whatever stood there by then, so a link (or a hard link) swapped in after
    the walk listed it had the harness read bytes outside the workspace
    (F1a1-balthasar).

    Args:
        path: The file to open.
        expected: Its ``(st_dev, st_ino)`` when it was checked
            (:func:`_identity`).

    Returns:
        io.BufferedReader: The open file, binary; the caller closes it.

    MAGI FIX (S6 loop 28): ``O_NOFOLLOW`` guards only the last component, so
    a directory above the file swapped for a link had the open follow it
    (F1a1-melchior); where descriptor-relative opens exist the file is
    opened through :func:`_open_anchored`, which refuses a link anywhere
    from the repository root down. A file with more than one link is refused
    too (:data:`HARD_LINKED`).

    Raises:
        OSError: When it cannot be opened (a link, on POSIX), or when the
            object opened is not a regular file or not *expected*, with
            :data:`CHANGED_AFTER_LISTING` as its text, or has more than one
            link, with :data:`HARD_LINKED`.
    """
    descriptor = (_open_anchored(path, _LOG_OPEN_FLAGS) if _ANCHORED_OPENS
                  else os.open(path, _LOG_OPEN_FLAGS))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != expected:
            raise OSError(CHANGED_AFTER_LISTING)
        if opened.st_nlink != 1:
            raise OSError(HARD_LINKED)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


#: The flags :func:`_listed_directory` opens a nested log directory with on
#: POSIX: the open itself refuses a link, and the listing reads the directory
#: that was opened, never a path that may have changed since.
_DIRECTORY_OPEN_FLAGS = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))

#: Whether :func:`_listed_directory` can list a directory it holds open:
#: ``os.scandir`` takes a descriptor and ``O_NOFOLLOW`` exists (POSIX only).
_LISTS_BY_DESCRIPTOR = os.scandir in os.supports_fd and hasattr(os, "O_NOFOLLOW")

#: Whether the capture opens every component relative to its parent's
#: descriptor (:func:`_open_anchored`): POSIX, where ``os.open`` takes
#: ``dir_fd``. Windows has no such open; there the capture tests the chain by
#: path before and after each listing, and a link swapped onto a component
#: above ``.magi/logs`` between those tests can have the walk list, or a read
#: open, what it points at. The read then fails the identity the file was
#: listed with, and :meth:`RunExecutor._capture_review_log` tests the whole
#: chain once more after the read and drops the capture, so no byte from
#: behind such a link reaches a result; it may reach the harness's memory.
_ANCHORED_OPENS = _LISTS_BY_DESCRIPTOR and os.open in os.supports_dir_fd


def _open_anchored(path, flags: int) -> int:
    """Open *path* one component at a time, each relative to its parent's descriptor.

    MAGI FIX (S6 loop 28): ``O_NOFOLLOW`` guards only the last component of a
    path, so a directory above it swapped for a link had the open follow that
    link (F1a1-melchior). Here every component below the walk's untested base
    (:func:`_chain_from_repository`) is opened with ``O_NOFOLLOW``, relative to
    the directory opened before it, so a link anywhere on the way fails the
    open (``ELOOP``) and no later swap of a component already opened can
    redirect the rest. POSIX only (:data:`_ANCHORED_OPENS`).

    Args:
        path: The entry to open.
        flags: The flags for the last component; ``O_NOFOLLOW`` belongs in
            them when that component must not be a link.

    Returns:
        int: The open descriptor; the caller closes it.

    Raises:
        OSError: When a component cannot be opened, a link included.
    """
    base, below = _chain_from_repository(path)
    parts = below.parts
    if not parts:
        return os.open(base, flags)
    descriptor = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            # MAGI FIX (F1a1b): the child is the one held before the parent
            # is closed, so a failed close leaves the finally to close the
            # child instead of leaking it (and closing the parent twice).
            parent, descriptor = descriptor, child
            os.close(parent)
        return os.open(parts[-1], flags, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def _unchanged_directory(path, expected) -> None:
    """Refuse *path* unless it is still the directory *expected* names, not a link.

    The chain above it is tested too (:func:`_linked_ancestor`): Windows lists
    by path, so a linked component above *path* would redirect the listing.

    Args:
        path: The directory to test.
        expected: Its :func:`_identity` when the walk listed it.

    Raises:
        OSError: When it, or a component above it, is a link
            (:data:`LINK_NOT_FOLLOWED`), when it is not a directory or another
            directory now (:data:`CHANGED_AFTER_LISTING`), or when it cannot
            be inspected.
    """
    if _linked_ancestor(path) is not None:
        raise OSError(LINK_NOT_FOLLOWED)
    status = os.lstat(path)
    if (_is_link(path) or not stat.S_ISDIR(status.st_mode)
            or _identity(status) != expected):
        raise OSError(CHANGED_AFTER_LISTING)


def _first_entries(listing, limit: int) -> list:
    """At most ``limit + 1`` ``(name, entry)`` pairs of the listing *listing*, which is closed.

    Args:
        listing: An ``os.scandir`` iterator.
        limit: The most entries the caller keeps; one more tells it there are more.

    Returns:
        list[tuple[str, os.DirEntry]]: The entries taken, in listing order.
    """
    try:
        return [(entry.name, entry) for entry in itertools.islice(listing, limit + 1)]
    finally:
        close = getattr(listing, "close", None)
        if close is not None:
            close()


def _listed_directory(path, expected, limit: int | None = None) -> list:
    """List the log directory *path*, only if it is still the one listed.

    MAGI FIX (S6 loop 25): an approved directory was listed later by path, and
    ``os.scandir`` follows a link, so one swapped in after its link test had
    the walk list, and the read open, what it points at (F1a1-caspar).

    On POSIX the directory is opened through :func:`_open_anchored`, so a link
    anywhere from the repository root down refuses the open; its identity is
    read from the open descriptor and the listing taken from it, so no later
    swap can redirect it. Each entry's identity is read relative to that
    descriptor while it is open. Windows can list only by path, so the
    directory and the chain above it are tested before and after the listing
    (:func:`_unchanged_directory`); see :data:`_ANCHORED_OPENS` for what is
    left.

    MAGI FIX (S6 loop 28): the listing was taken whole, so a directory with a
    vast number of entries was materialised at no captured byte
    (F1a1-melchior). At most ``limit + 1`` entries are taken.

    Args:
        path: The directory to list.
        expected: Its :func:`_identity` when its parent was listed, or
            ``None`` for the walk's root on POSIX, which the anchored open
            itself vouches for.
        limit: The most entries the caller keeps; ``None`` reads
            :data:`MAX_REVIEW_LOG_ENTRIES`.

    Returns:
        list[tuple[str, os.DirEntry, tuple | None]]: Each entry's name, the
        entry (whose ``is_dir``/``is_file`` with ``follow_symlinks=False``
        describe it as listed), and on POSIX its :func:`_identity`, read
        without following a link (``None`` when that failed, and always on
        Windows, where the caller reads it). One entry past *limit* means the
        directory holds more.

    Raises:
        OSError: When the directory cannot be opened or listed, or is not the
            one *expected* names (:data:`CHANGED_AFTER_LISTING`).
    """
    limit = MAX_REVIEW_LOG_ENTRIES if limit is None else limit
    if not _ANCHORED_OPENS:
        if expected is None:
            expected = _identity(os.lstat(path))
        _unchanged_directory(path, expected)
        entries = _first_entries(os.scandir(path), limit)
        _unchanged_directory(path, expected)
        return [(name, entry, None) for name, entry in entries]
    descriptor = _open_anchored(path, _DIRECTORY_OPEN_FLAGS)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (
                expected is not None and _identity(opened) != expected):
            raise OSError(CHANGED_AFTER_LISTING)
        listed = []
        for name, entry in _first_entries(os.scandir(descriptor), limit):
            try:
                identity = _identity(entry.stat(follow_symlinks=False))
            except OSError:
                identity = None
            listed.append((name, entry, identity))
        return listed
    finally:
        os.close(descriptor)


def _regular_files_without_links(directory, refused, identities=None):
    """The files under *directory*, sorted, without descending into or reading a link.

    ``Path.rglob`` descends into a junction on Windows, and a file test follows
    a symlink, so neither bounds a walk to what is really under *directory*.
    This one lists each directory with :func:`_listed_directory` and tests
    every entry with :func:`_is_link` first.

    Complexity: ``O(entries)``, at most :data:`MAX_REVIEW_LOG_ENTRIES` listed
    in all; past that the walk stops and names the directory it stopped in.

    Args:
        directory: The directory to walk. It is listed like every nested
            directory (:func:`_listed_directory`), so a link there is refused
            and named, never listed; the chain above it is the caller's to
            test, as :meth:`RunExecutor._capture_review_log` does.
        refused: A list that receives ``"<path> (<why>)"``, *path* relative
            to *directory* (:func:`_name_under`), for each entry
            left out: a link (neither listed nor read), an entry that is
            neither a directory nor a regular file (a fifo, socket or device,
            never opened: reading a fifo can block), a directory that could
            not be listed, an entry whose test failed, or the directory where
            the entry cap (:data:`OVER_ENTRY_CAP`) stopped the walk.
        identities: A dict that receives, for each file returned, its
            :func:`_identity` as listed, so the read can refuse a file
            swapped in afterwards (:func:`_open_listed`). ``None`` records
            nothing.

    Returns:
        list[pathlib.Path]: The regular files found, in sorted path order.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _ = (pathlib.Path(tmp) / "a.log").write_bytes(b"x")
        ...     [p.name for p in _regular_files_without_links(tmp, [])]
        ['a.log']
    """
    root = pathlib.Path(directory)
    # MAGI FIX (S6 loop 15): entries were named by ``path.name``, so a nested
    # link lost the directory that says which one was left out (F1a1-b2).
    # A nested directory carries its identity from the listing of its parent,
    # so it is listed only while it is still that directory
    # (:func:`_listed_directory`).
    # MAGI FIX (S6 loop 27): the root was listed by path, so a link swapped
    # onto it after the caller's test was listed, and its files read, once
    # before the caller's second test dropped the capture (F1a1-melchior). The
    # root is listed the same way, which refuses a link.
    # MAGI FIX (S6 loop 28): the walk listed every entry whatever their number
    # (F1a1-melchior); it now stops at MAX_REVIEW_LOG_ENTRIES and names the
    # directory it stopped in, once.
    files, pending, budget = [], [(root, None)], MAX_REVIEW_LOG_ENTRIES
    stopped = False
    while pending:
        current, expected = pending.pop()
        try:
            entries = _listed_directory(current, expected, budget)
        except OSError as exc:
            refused.append("%s (%s)" % (_name_under(root, current), exc))
            continue
        if len(entries) > budget:
            refused.append("%s (%s)" % (_name_under(root, current),
                                        OVER_ENTRY_CAP % MAX_REVIEW_LOG_ENTRIES))
            entries, pending, stopped = entries[:budget], [], True
        budget -= len(entries)
        for name, entry, listed_identity in entries:
            path = current / name
            try:
                if _is_link(path):
                    refused.append("%s (%s)" % (_name_under(root, path),
                                                LINK_NOT_FOLLOWED))
                elif entry.is_dir(follow_symlinks=False):
                    if not stopped:
                        pending.append((path, listed_identity
                                        or _identity(os.lstat(path))))
                elif entry.is_file(follow_symlinks=False):
                    if identities is not None:
                        identities[path] = (listed_identity
                                            or _identity(os.lstat(path)))
                    files.append(path)
                else:
                    # MAGI FIX (S6 loop 16): a fifo, socket or device file was
                    # dropped unnamed, so a scenario could not tell it from a
                    # product that wrote nothing (F1a1-melchior).
                    refused.append("%s (%s)" % (_name_under(root, path),
                                                NOT_A_REGULAR_FILE))
            except OSError as exc:
                refused.append("%s (%s)" % (_name_under(root, path), exc))
    return sorted(files)


#: How many bytes :func:`_read_at_most` asks for at a time.
READ_CHUNK_BYTES = 64 * 1024


def _read_at_most(handle, limit: int) -> bytes:
    """Read the open file *handle*, stopping once it holds more than *limit* bytes.

    Args:
        handle: A binary file open for reading (:func:`_open_listed`).
        limit: The most bytes the caller can keep, at least 0.

    Returns:
        bytes: The whole file when it holds at most *limit* bytes; otherwise
        its first ``limit + 1`` bytes, which tell the caller it is over.

    Raises:
        OSError: When the file cannot be read.

    Example:
        >>> _read_at_most(io.BytesIO(b"12345"), 2), _read_at_most(io.BytesIO(b"12345"), 9)
        (b'123', b'12345')
    """
    chunks, held = [], 0
    while held <= limit:
        chunk = handle.read(min(READ_CHUNK_BYTES, limit + 1 - held))
        if not chunk:
            break
        chunks.append(chunk)
        held += len(chunk)
    return b"".join(chunks)


def _read_bounded(root, paths, refused, identities=None) -> list:
    """Read *paths* in order, within the per-file and total capture caps.

    MAGI FIX (S6 loop 16): every log file was read whole with no bound, so a
    huge or numerous log from a faulty product could exhaust the harness before
    the scenario reported ``CANNOT_TEST`` (F1a1-caspar).

    A file's size is read before the file is; one over
    :data:`MAX_REVIEW_LOG_FILE_BYTES`, or one that would take the capture past
    :data:`MAX_REVIEW_LOG_TOTAL_BYTES`, is named and not read. The read itself
    stops one byte past what is still allowed (:func:`_read_at_most`), because
    the file may have grown since its size was read, or still be growing: a
    file that turns out larger is named and dropped.

    MAGI FIX (S6 loop 17): the file was read whole after the size check, so a
    file that grew in between was held in memory unbounded (F1a1-caspar,
    F1a1-melchior, F1a1-balthasar).

    The size is read with ``os.lstat``, which never follows a link, and the
    file is opened only if it is still the file that was checked
    (:func:`_open_listed`): the one the walk listed when *identities* names
    it, otherwise the one ``os.lstat`` just saw.

    Complexity: ``O(bytes kept)``, at most :data:`MAX_REVIEW_LOG_TOTAL_BYTES`
    plus one chunk held at once.

    Args:
        root: The directory the files are named under (:func:`_name_under`).
        paths: The files to read, in the order they are kept.
        refused: A list that receives ``"<path> (<why>)"`` for each file that
            could not be read, was over a cap, or was not the file checked
            (:data:`CHANGED_AFTER_LISTING`).
        identities: Each path's :func:`_identity` as the walk listed it
            (:func:`_regular_files_without_links`), or ``None``.

    Returns:
        list[bytes]: The bytes of every file kept, in *paths* order.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "a.log"
        ...     _ = path.write_bytes(b"x")
        ...     _read_bounded(tmp, [path], [])
        [b'x']
    """
    kept, spent = [], 0
    for path in paths:
        name = _name_under(root, path)
        data = None
        try:
            checked = os.lstat(path)
            size = checked.st_size
            if (size <= MAX_REVIEW_LOG_FILE_BYTES
                    and spent + size <= MAX_REVIEW_LOG_TOTAL_BYTES):
                expected = (identities or {}).get(path, _identity(checked))
                with _open_listed(path, expected) as handle:
                    data = _read_at_most(handle, min(
                        MAX_REVIEW_LOG_FILE_BYTES,
                        MAX_REVIEW_LOG_TOTAL_BYTES - spent))
                size = len(data)
        except OSError as exc:
            refused.append("%s (%s)" % (name, exc))
            continue
        if size > MAX_REVIEW_LOG_FILE_BYTES:
            why = OVER_FILE_CAP % MAX_REVIEW_LOG_FILE_BYTES
        elif spent + size > MAX_REVIEW_LOG_TOTAL_BYTES:
            why = OVER_TOTAL_CAP % MAX_REVIEW_LOG_TOTAL_BYTES
        else:
            kept.append(data)
            spent += size
            continue
        refused.append("%s (%s)" % (name, why))
    return kept


#: The largest file :func:`read_log_files` reads from a log directory that is
#: read whole. The product writes one file per day, a few hundred KB in the
#: harness's environment; a file this large means a faulty or hostile product.
MAX_ENVIRONMENT_LOG_FILE_BYTES = 64 * 1024 * 1024

#: The most bytes one :func:`read_log_files` walk yields, over all files.
#: Holds the environment's whole history many times over, and at least one
#: file of :data:`MAX_ENVIRONMENT_LOG_FILE_BYTES`.
MAX_ENVIRONMENT_LOG_TOTAL_BYTES = 256 * 1024 * 1024


def _directory_chain_refusal(directory):
    """Why a read under *directory* must be dropped: it, or a component above it, is a link.

    Args:
        directory: The directory a walk read under.

    Returns:
        str | None: ``"<component> (<why>)"`` for the first linked component
        from the repository root down to *directory* itself, or for a test
        that failed; ``None`` when none is a link.
    """
    try:
        linked = _linked_ancestor(directory)
        if linked is None and _is_link(directory):
            linked = pathlib.Path(directory)
    except OSError as exc:
        return "%s (%s)" % (_name_under(directory, directory), exc)
    if linked is None:
        return None
    return "%s (%s)" % (linked, LINK_NOT_FOLLOWED)


def read_log_files(directory, unreadable, offsets=None):
    """Yield the bytes of each regular file under *directory*, in sorted path order, never through a link.

    For a log directory that is read whole, such as the persistent
    environment's ``.magi/logs`` (:func:`smoke.logs.scan`): the walk is
    :func:`_regular_files_without_links` and each file is opened by
    :func:`_open_listed`, so a link, a hard link or a file swapped after the
    listing is named, never read. The walk's entry cap
    (:data:`MAX_REVIEW_LOG_ENTRIES`) applies, and a directory past it is
    named in *unreadable*.

    MAGI FIX (S6 loop 28): ``smoke.logs.scan`` walked with ``rglob`` and read
    with ``read_bytes``, both of which follow a link (F1a1-melchior, the
    defect the review-log capture was fixed for).

    MAGI FIX (S6 loop 29): each file was read whole with no bound, so one huge
    file a faulty product wrote under the persistent directory exhausted the
    harness before a scenario could report it (F1a1b-melchior). A file is
    read only up to :data:`MAX_ENVIRONMENT_LOG_FILE_BYTES`, and the walk
    yields at most :data:`MAX_ENVIRONMENT_LOG_TOTAL_BYTES`; a file past
    either is named (:data:`OVER_FILE_CAP`, :data:`OVER_TOTAL_CAP`) and not
    yielded. Both caps sit far above what the environment writes, so a
    healthy one is read whole.

    Complexity: ``O(bytes yielded)``, one file of at most
    :data:`MAX_ENVIRONMENT_LOG_FILE_BYTES` held at a time.

    Args:
        directory: The directory to read.
        unreadable: A list that receives ``"<path> (<why>)"``, *path*
            relative to *directory*, for each entry left out or file that
            could not be read or was over a cap.
        offsets: For a file it names (by its path under *directory*, as
            :func:`_regular_files_without_links` builds it), where the read
            starts: the bytes before it are skipped and not counted. ``None``
            reads every file from its start.

    MAGI FIX (F1a1): Windows lists and opens by path, so a link swapped onto
    a component above *directory* during the read had a scenario see the
    bytes behind it; unlike :meth:`RunExecutor._capture_review_log`, nothing
    tested the chain again after reading. Each file's bytes are yielded only
    after the chain (:func:`_directory_chain_refusal`) is tested once more;
    when it fails, the refusal is named in *unreadable* and the walk stops,
    so nothing read since the last passing test is yielded.

    Yields:
        bytes: Each file's contents from its offset.
    """
    identities, spent = {}, 0
    files = _regular_files_without_links(directory, unreadable, identities)
    for path in files:
        limit = min(MAX_ENVIRONMENT_LOG_FILE_BYTES,
                    MAX_ENVIRONMENT_LOG_TOTAL_BYTES - spent)
        try:
            with _open_listed(path, identities[path]) as handle:
                handle.seek((offsets or {}).get(path, 0))
                data = _read_at_most(handle, limit)
        except OSError as exc:
            unreadable.append("%s (%s)" % (_name_under(directory, path), exc))
            continue
        refusal = _directory_chain_refusal(directory)
        if refusal is not None:
            unreadable.append(refusal)
            return
        if len(data) > limit:
            why = (OVER_FILE_CAP % MAX_ENVIRONMENT_LOG_FILE_BYTES
                   if limit == MAX_ENVIRONMENT_LOG_FILE_BYTES
                   else OVER_TOTAL_CAP % MAX_ENVIRONMENT_LOG_TOTAL_BYTES)
            unreadable.append("%s (%s)" % (_name_under(directory, path), why))
            continue
        spent += len(data)
        yield data


def _name_under(root, path) -> str:
    """How a log entry is named: its ``/``-separated path below *root*.

    Args:
        root: The directory the walk started at.
        path: A path at or below *root*.

    Returns:
        str: *path* relative to *root*, or *root*'s own name for *root*.

    Example:
        >>> _name_under(pathlib.Path("logs"), pathlib.Path("logs/sub/a.log"))
        'sub/a.log'
    """
    relative = pathlib.Path(path).relative_to(root).as_posix()
    return pathlib.Path(root).name if relative == "." else relative


#: The most entries :func:`_copy_tree_without_links` copies, files and
#: directories alike. The environment's ``.magi`` holds a handful (the
#: configuration, the encrypted store and its journal files), so a tree past
#: this is not the environment the harness built and is refused rather than
#: copied without bound.
MAX_COPIED_MAGI_ENTRIES = 4096

#: The flags :func:`_copy_listed_file` creates a workspace file with: a new
#: file only, never through a link (``O_NOFOLLOW`` on POSIX).
_COPY_CREATE_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                      | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))

#: The mode a copied file is created with before its own is applied: readable
#: and writable by the harness only, so its bytes are never exposed wider in
#: between.
_COPY_CREATE_MODE = 0o600

#: The mode a copied directory is created with before its own is applied,
#: once its contents are in.
_COPY_DIRECTORY_MODE = 0o700


def _copy_listed_file(source, target, expected) -> None:
    """Copy the regular file *source* to the new file *target*, only if it is the one listed.

    *source* is opened by :func:`_open_listed`, so a link, a hard link
    (:data:`HARD_LINKED`) or a file swapped in after the listing
    (:data:`CHANGED_AFTER_LISTING`) is refused, and the bytes copied are those
    of the file that was opened. *target* must not exist. Its mode is the
    source's, as :func:`shutil.copy2` kept it.

    Complexity: ``O(file size)``, :data:`READ_CHUNK_BYTES` held at once.

    Args:
        source: The file to copy.
        target: Where to write it; created, never overwritten.
        expected: The source's :func:`_identity` as listed.

    Raises:
        OSError: When either file cannot be opened, read or written, or the
            source is not the file listed.
    """
    with _open_listed(source, expected) as reader:
        mode = stat.S_IMODE(os.fstat(reader.fileno()).st_mode)
        with os.fdopen(os.open(target, _COPY_CREATE_FLAGS, _COPY_CREATE_MODE),
                       "wb") as writer:
            shutil.copyfileobj(reader, writer, READ_CHUNK_BYTES)
    _set_mode(target, mode)


def _copy_tree_without_links(source, destination, ignore) -> None:
    """Copy the directory *source* to the new directory *destination*, refusing every link.

    MAGI FIX (S6 loop 29): the workspace was copied with
    :func:`shutil.copytree` after a separate walk had checked the tree for
    links. They were two looks at the tree, and copytree follows whatever
    stands at a path when it copies it, so a link swapped in between (or a
    hard link, which no link test sees) put bytes from outside the
    environment in front of the product (F1a1b-melchior, F1a1b-caspar).

    Here the tree is walked and copied in one pass, with the machinery the
    log capture reads with: each directory is listed only while it is still
    the one its parent listed, never through a link
    (:func:`_listed_directory`), and each file is copied from the descriptor
    :func:`_copy_listed_file` opened, which refuses a link, a hard link and a
    file that is not the one listed. A link, a fifo, socket or device, an
    unlistable directory or a tree past :data:`MAX_COPIED_MAGI_ENTRIES` stops
    the copy. Modes are kept; a directory's own is applied after its
    contents, so a read-only one can still be filled.

    Complexity: ``O(entries + bytes)``.

    Args:
        source: The directory to copy; refused when it is itself a link.
        destination: Where to create the copy; it must not exist, its parent
            must.
        ignore: Called with a directory and the names listed in it, like the
            ``ignore`` argument of :func:`shutil.copytree`; the names it
            returns are not copied (:func:`_skip_top_level_logs`).

    Raises:
        HarnessError: Naming the link, hard link, changed or irregular entry,
            or the directory that could not be listed or was past the cap.
        OSError: When a directory or file cannot be created or written, which
            the executor turns into a :class:`HarnessError`.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     source = pathlib.Path(tmp) / "src"
        ...     source.mkdir()
        ...     _ = (source / "a.toml").write_bytes(b"x")
        ...     _copy_tree_without_links(source, pathlib.Path(tmp) / "dst",
        ...                              lambda directory, names: set())
        ...     (pathlib.Path(tmp) / "dst" / "a.toml").read_bytes()
        b'x'
    """
    source = pathlib.Path(source)
    pending = [(source, pathlib.Path(destination), None)]
    modes, budget = [], MAX_COPIED_MAGI_ENTRIES
    while pending:
        current, target, expected = pending.pop()
        try:
            entries = _listed_directory(current, expected, budget)
        except OSError as exc:
            raise HarnessError("refusing to copy %s: %s could not be listed "
                               "as the directory checked: %s"
                               % (source, current, exc)) from exc
        if len(entries) > budget:
            raise HarnessError("refusing to copy %s: it holds more than %d "
                               "entries" % (source, MAX_COPIED_MAGI_ENTRIES))
        budget -= len(entries)
        target.mkdir(mode=_COPY_DIRECTORY_MODE)
        modes.append((target, stat.S_IMODE(os.lstat(current).st_mode)))
        skipped = ignore(str(current), [name for name, _, _ in entries])
        for name, entry, identity in entries:
            if name in skipped:
                continue
            path = current / name
            if _is_link(path):
                raise HarnessError("refusing to copy %s: %s is %s"
                                   % (source, path, LINK_NOT_FOLLOWED))
            listed = identity or _identity(os.lstat(path))
            if entry.is_dir(follow_symlinks=False):
                pending.append((path, target / name, listed))
            elif entry.is_file(follow_symlinks=False):
                try:
                    _copy_listed_file(path, target / name, listed)
                except OSError as exc:
                    raise HarnessError("refusing to copy %s: %s: %s"
                                       % (source, path, exc)) from exc
            else:
                raise HarnessError("refusing to copy %s: %s is %s"
                                   % (source, path, NOT_A_REGULAR_FILE))
    for directory, mode in reversed(modes):
        _set_mode(directory, mode)


def _skip_top_level_logs(magi_dir):
    """A copy filter that leaves out the ``logs`` directory of *magi_dir*.

    The name is matched as the product spells it (:func:`same_product_name`),
    so ``Logs`` or ``logs.`` is left out too.

    The review workspace is built from the environment's ``.magi``, whose
    ``logs/`` holds what earlier runs wrote. Leaving it out makes the log a
    review run captures exactly what that run wrote.

    Args:
        magi_dir: The ``.magi`` directory being copied.

    Returns:
        callable: The ``ignore`` argument of :func:`_copy_tree_without_links`.
    """
    top = os.path.normcase(os.path.abspath(magi_dir))

    def ignore(directory, names):
        if os.path.normcase(os.path.abspath(directory)) != top:
            return set()
        # MAGI FIX (S6 loop 16): an existing "Logs" is the product's log
        # directory on a case-insensitive file system, yet was copied as stale
        # history (F1a1-caspar).
        return {name for name in names
                if same_product_name(name, LOG_DIR_PARTS[1])}

    return ignore


def review_skipped_files() -> tuple[tuple[str, bytes], ...]:
    """The files R14 reviews, every one of which the product must skip.

    Returns:
        tuple: ``(path, content)`` pairs: a binary and a file whose name is
        a key (its content holds the same key).
    """
    return ((REVIEW_SKIPPED_DIR + "/pixel.png", _TOY_PNG),
            (REVIEW_SKIPPED_DIR + "/" + _TOY_SECRET_NAME, _TOY_SECRET_FILE))


#: Directory (workspace-relative) R18 reviews (S53, REQ-RV22): one small source
#: file whose CONTENT holds a single-line toy key and whose name does not.
REVIEW_MASKED_DIR = "rv22keyed"
REVIEW_MASKED_FILE = REVIEW_MASKED_DIR + "/keyed.rs"

#: The toy key inside that file. Built from two pieces so the source holds no
#: key-shaped literal; it matches the product's ``anthropic-key`` pattern.
_TOY_MASKED_PREFIX = "sk-ant-api03-"  # allow-secret-scan
REVIEW_MASKED_KEY = (_TOY_MASKED_PREFIX + "M" * 28).encode("ascii")

#: The pattern the product reports for it (``name`` in ``src/review/secrets.rs``).
REVIEW_MASKED_PATTERN = "anthropic-key"

#: Directory R19 reviews (S54): only a file holding a multi-line private-key
#: block, which the product leaves out whole as ``secret``.
REVIEW_KEYBLOCK_DIR = "rv22pem"
REVIEW_KEYBLOCK_FILE = REVIEW_KEYBLOCK_DIR + "/server.pem"

#: The body lines of the toy block, each unique and none carrying a header.
#: S54 looks for each in every output: a body has no pattern of its own, so the
#: only protection is leaving the whole file out.
REVIEW_KEYBLOCK_BODY_LINES = (b"TOYBODYLINEONE" + b"Q" * 40, b"TOYBODYLINETWO" + b"R" * 40)


def review_masked_files() -> tuple[tuple[str, bytes], ...]:
    """The file R18 reviews: valid Rust holding one single-line toy key.

    Returns:
        tuple: One ``(path, content)`` pair, :data:`REVIEW_MASKED_FILE`.
    """
    body = (b"/// Returns the token of the toy service.\n"
            b"pub fn token() -> &'static str {\n"
            b'    "' + REVIEW_MASKED_KEY + b'"\n'
            b"}\n")
    return ((REVIEW_MASKED_FILE, body),)


def review_keyblock_files() -> tuple[tuple[str, bytes], ...]:
    """The file R19 reviews: a toy PEM private-key block, header to footer.

    Returns:
        tuple: One ``(path, content)`` pair, :data:`REVIEW_KEYBLOCK_FILE`.
    """
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"  # allow-secret-scan
    footer = "-----END " + "RSA PRIVATE KEY-----"  # allow-secret-scan
    lines = [header.encode("ascii"), *REVIEW_KEYBLOCK_BODY_LINES, footer.encode("ascii")]
    return ((REVIEW_KEYBLOCK_FILE, b"\n".join(lines) + b"\n"),)


#: Directory (workspace-relative) R17 reviews: one small source file, so the run
#: costs one batch.
REVIEW_CONTEXT_DIR = "ctx"

#: The one file R17 reviews: a short, plain function, enough for one batch.
_REVIEW_CONTEXT_FILE = (b"/// Adds two numbers.\n"
                        b"fn add(a: i32, b: i32) -> i32 {\n"
                        b"    a + b\n"
                        b"}\n")

#: Where R17's ``[embedding]`` points: TCP port 9 (discard) on loopback, which
#: nothing listens on in the harness, so every embedding request is refused at
#: once and the review must go on without documentation and history.
UNREACHABLE_EMBEDDER_URL = "http://127.0.0.1:9/v1"

#: The table whose endpoint :func:`point_embedding_at` rewrites.
_EMBEDDING_TABLE = "[embedding]"

#: Its name, as :func:`_table_name` reads a header.
_EMBEDDING_TABLE_NAME = "embedding"


def _table_name(header: str) -> str:
    """The name a TOML table header line declares, or ``""`` for anything else.

    A trailing ``#`` comment is dropped and the spaces TOML allows inside the
    brackets are trimmed; an array-of-tables header (``[[x]]``) declares no
    plain table and gives ``""``.

    Args:
        header: A stripped line that starts with ``[``.

    Returns:
        str: The table's name.

    Example:
        >>> _table_name("[ embedding ] # the embedder"), _table_name("[[magi.fallback]]")
        ('embedding', '')
    """
    code = header.split("#", 1)[0].strip()
    if code.startswith("[[") or not code.endswith("]"):
        return ""
    return code[1:-1].strip()


def review_context_files() -> tuple[tuple[str, bytes], ...]:
    """The file R17 reviews.

    Returns:
        tuple: One ``(path, content)`` pair: a small Rust function.
    """
    return ((REVIEW_CONTEXT_DIR + "/add.rs", _REVIEW_CONTEXT_FILE),)


def point_embedding_at(config_path, url: str) -> None:
    """Rewrite the ``magi.toml`` at *config_path* so ``[embedding]`` reaches *url*.

    Every other line is kept: the trio and the gate still reach the real
    backend. An active ``base_url`` inside ``[embedding]`` is replaced; without
    one the line is added right after the table header; without the table, the
    table is appended. Line-based like :func:`point_config_at`, and parsed back
    by the product, which refuses a duplicate key.

    Complexity: ``O(file bytes)``.

    Args:
        config_path: The configuration to rewrite in place.
        url: The endpoint ``[embedding]`` must reach.

    Raises:
        HarnessError: When the file cannot be read or written.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "magi.toml"
        ...     _ = path.write_text('base_url = "r"\\n[embedding]\\nmodel = "m"\\n')
        ...     point_embedding_at(path, "http://127.0.0.1:9/v1")
        ...     path.read_text().splitlines()
        ['base_url = "r"', '[embedding]', 'base_url = "http://127.0.0.1:9/v1"', 'model = "m"']
    """
    line = 'base_url = "%s"' % url
    try:
        text = pathlib.Path(config_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HarnessError("could not read %s to point its embedder away: %s"
                           % (config_path, exc)) from exc
    lines, in_embedding, placed = [], False, False
    for original in text.splitlines():
        stripped = original.strip()
        if stripped.startswith("["):
            # MAGI FIX (S6 MS3 loop2): TOML allows spaces inside the brackets and a comment
            # after the header; an exact match appended a second [embedding] table.
            in_embedding = _table_name(stripped) == _EMBEDDING_TABLE_NAME
            lines.append(original)
            if in_embedding:
                lines.append(line)
                placed = True
            continue
        is_endpoint = (stripped.startswith("base_url")
                       and stripped[len("base_url"):].lstrip().startswith("="))
        if in_embedding and is_endpoint:
            continue
        lines.append(original)
    if not placed:
        lines.extend((_EMBEDDING_TABLE, line))
    try:
        pathlib.Path(config_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise HarnessError("could not point %s's embedder away: %s"
                           % (config_path, exc)) from exc


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


#: The most digits of a Content-Length :func:`_drain_body` converts: every
#: length a 64-bit size can hold, and far below the integer-conversion limit.
MAX_CONTENT_LENGTH_DIGITS = 19


def _drain_body(rfile, headers) -> None:
    """Read and discard a request body, never holding more than one chunk.

    MAGI FIX (S6 loop 28): the error backend read ``int(Content-Length)``
    bytes in one call, so a non-numeric value raised in the handler (the
    product got no answer), a negative one read to the end of the stream,
    and a huge one asked the reader for that many bytes at once. The length
    is read as a non-negative integer, anything else counting as no body,
    and the body is read :data:`MODEL_COUNTER_CHUNK_BYTES` at a time until it
    is spent or the client stops sending. Both loopback handlers drain
    through here.

    Args:
        rfile: The request's input stream.
        headers: Its headers.

    Raises:
        OSError: When a read fails or times out.
    """
    declared = (headers.get("Content-Length") or "").strip()
    # MAGI FIX (S6 loop 29): a decimal length past Python's integer-conversion
    # limit (4300 digits) raised ValueError, which neither handler catches, so
    # the request got a traceback instead of an answer (F1a1b-melchior,
    # F1a1b-caspar). A length longer than MAX_CONTENT_LENGTH_DIGITS counts as
    # no body like any other unusable value; both stand-ins answer HTTP/1.0
    # and close, so an undrained body never reaches a later request.
    usable = (declared.isascii() and declared.isdigit()
              and len(declared) <= MAX_CONTENT_LENGTH_DIGITS)
    left = int(declared) if usable else 0
    while left > 0:
        chunk = rfile.read(min(MODEL_COUNTER_CHUNK_BYTES, left))
        if not chunk:
            return
        left -= len(chunk)


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    """Reads one request, records it whatever its shape, and hands it to :meth:`_answer`.

    The loopback stand-ins share this: every connection that carries a byte
    leaves exactly one record in the server's ``log`` (a :class:`RequestLog`,
    or ``None`` to record nothing), by method and path when the request line
    parsed and as :data:`UNPARSED_REQUEST` when it did not, and every method is
    answered alike. A subclass implements :meth:`_answer`, which must catch
    the ``OSError`` of a client that hung up.
    """

    def handle_one_request(self) -> None:
        """Read one request, record it whatever its shape, and answer it.

        MAGI FIX (S6 loop 28): the request was recorded from the ``do_*``
        methods, so a method the handler does not implement (``OPTIONS``,
        ``TRACE``, an unknown verb) got the base class's 501, and a malformed
        or over-long request line its 400 or 414, all unrecorded: a refusal
        run whose product sent one passed (F1a1-melchior). MAGI FIX (F1a1b):
        the error backend kept that old shape; both stand-ins read requests
        here now.
        """
        try:
            carried = self.rfile.peek(1)[:1]
        except OSError:
            carried = b""
        if not carried:
            self.close_connection = True
            return
        self.command = self.path = None
        try:
            self.raw_requestline = self.rfile.readline(MAX_REQUEST_LINE_BYTES + 1)
        except OSError:
            self._record(UNPARSED_REQUEST)
            self.close_connection = True
            return
        parsed = False
        try:
            if len(self.raw_requestline) <= MAX_REQUEST_LINE_BYTES:
                parsed = self.parse_request()
        except OSError:
            parsed = False
        finally:
            self._record(self._parsed_key())
        if not parsed:
            self.close_connection = True
            return
        self._answer()
        try:
            self.wfile.flush()
        except OSError:
            self.close_connection = True

    def _record(self, key: tuple[str, str]) -> None:
        """Record *key* in the server's log, when it has one.

        Args:
            key: The ``(method, path)`` pair :meth:`RequestLog.record` takes.
        """
        log = getattr(self.server, "log", None)
        if log is not None:
            log.record(*key)

    def _parsed_key(self) -> tuple[str, str]:
        """The ``(method, path)`` of the request just parsed, or :data:`UNPARSED_REQUEST`.

        ``parse_request`` sets the method and the path together, so a method
        means the path is this request's. The path is kept without its query,
        and a path that does not split as a URL is kept as unparsed rather
        than raw: the authority of an absolute-form target may hold a
        credential.

        Returns:
            tuple[str, str]: What :meth:`RequestLog.record` takes.
        """
        if not self.command or self.path is None:
            return UNPARSED_REQUEST
        try:
            return self.command, urllib.parse.urlsplit(self.path).path
        except ValueError:
            return self.command, UNPARSED_REQUEST[1]

    def _answer(self) -> None:
        """Answer the request just recorded.

        Raises:
            NotImplementedError: Always; each stand-in answers its own way.
        """
        raise NotImplementedError


class _EchoingHandler(_RecordingHandler):
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

    def _answer(self) -> None:
        """Drain the request's body and answer it :data:`ERROR_BACKEND_STATUS`, echoing it."""
        try:
            _drain_body(self.rfile, self.headers)
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
        except OSError:
            # The client stopped mid-body or hung up: nobody is left to answer.
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        """Swallow the access log.

        The default writes every request to stderr, and the request line
        carries whatever the product put in the URL. That is the harness
        printing a credential the scenario is about to search for.

        Args:
            fmt: The format string the base class passes.
            *args: Its arguments.
        """


#: The most connections a loopback stand-in serves on threads of their own at
#: once. A run talks to it over a handful; past this, a connection is served
#: on the accepting thread (:class:`BoundedThreadsMixIn`).
MAX_LOOPBACK_HANDLER_THREADS = 32


class BoundedThreadsMixIn:
    """Mixed in before a ``socketserver.ThreadingMixIn`` server: never more than a fixed number of handler threads.

    MAGI FIX (S6 loop 29): a threading server starts a thread per
    connection, and each lives until its deadline, so a product that opened
    connections faster than they closed grew the harness's thread count
    without bound (F1a1b-caspar). Up to :data:`MAX_LOOPBACK_HANDLER_THREADS`
    connections run on threads; past that, a connection is served on the
    accepting thread itself, so accepting slows down while every request is
    still answered and, by the counter, recorded. A connection's own deadline
    (the handlers' ``timeout``, the counter's total deadline, a silent
    listener's release) bounds how long one served inline holds the
    accepting thread.

    Example:
        >>> server = _BoundedThreadingServer(("127.0.0.1", 0), _EchoingHandler)
        >>> server._slots.acquire(blocking=False)
        True
        >>> server.server_close()
    """

    def __init__(self, *args, **kwargs) -> None:
        """Build the server, with :data:`MAX_LOOPBACK_HANDLER_THREADS` read now.

        Args:
            *args: The server's own positional arguments.
            **kwargs: Its keyword arguments.

        Raises:
            OSError: When the server's address cannot be bound.
        """
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(MAX_LOOPBACK_HANDLER_THREADS)

    def process_request(self, request, client_address) -> None:
        """Serve one accepted connection on a thread of its own, or here when every slot is taken.

        Args:
            request: The accepted socket.
            client_address: The peer's address.
        """
        if self._slots.acquire(blocking=False):
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._slots.release()
                raise
            return
        try:
            self.finish_request(request, client_address)
        except Exception:  # noqa: BLE001 - reported the way a handler thread reports it
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def process_request_thread(self, request, client_address) -> None:
        """Serve one connection on its thread, then give the thread's slot back.

        Args:
            request: The accepted socket.
            client_address: The peer's address.
        """
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _BoundedThreadingServer(BoundedThreadsMixIn, http.server.ThreadingHTTPServer):
    """The loopback stand-ins' HTTP server, its handler threads bounded (:class:`BoundedThreadsMixIn`)."""


@contextlib.contextmanager
def error_backend(log=None):
    """Serve an error-answering endpoint on loopback for the block's duration.

    Args:
        log: A :class:`RequestLog` that records every request the backend
            answers, so a caller can tell a backend the product reached from
            one it never did; ``None`` records nothing.

    Yields:
        str: The base URL to point the product at, already ending in ``/v1``.
    """
    server = _BoundedThreadingServer(("127.0.0.1", 0), _EchoingHandler)
    server.daemon_threads = True
    server.log = log
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

#: The variable that overrides the root ``provider``. A refusal run blanks it,
#: which the product reads as absent, so the harness's own shell cannot steer
#: the run to a provider the counter does not stand in for.
_PROVIDER_VARIABLE = "MAGI_PROVIDER"

#: The variables through which the product's HTTP client (reqwest, system proxy
#: on by default) sends a request to a proxy, matched in either case. A refusal
#: run removes them: reqwest does not exempt ``127.0.0.1``, so an inherited
#: ``HTTP_PROXY`` sends the counter's traffic to the proxy, the counter sees
#: nothing, and the "no model call" proof passes vacuously.
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")

#: The proxy exemption list a refusal run sets, replacing any inherited one, so
#: loopback is exempt even if a proxy variable reached the child some other way.
_NO_PROXY_VARIABLE = NO_PROXY_VARIABLE

#: Every spelling of the loopback host the counter may be reached by.
_LOOPBACK_HOSTS = ",".join(LOOPBACK_HOSTS)

#: The only provider kind a refusal run may name. The harness writes the
#: product's own scaffold (``provider = "ollama"``, no ``[magi].kind``), and the
#: counter is built for it: :data:`PROBE_PATHS` are Ollama's, and an
#: ``anthropic`` seat posts to a fixed URL that no ``base_url`` steers
#: (magi-core's ``ClaudeProvider``). Any other kind is a setup the counter was
#: not measured against, so it is refused rather than assumed countable.
_COUNTED_KIND = "ollama"

#: Where the product reads a provider kind: the root ``provider`` and
#: ``[magi].kind`` (``src/config.rs``, ``ProviderKind::parse``, trimmed and
#: case-sensitive, blank meaning absent). ``[embedding].provider`` is not one:
#: it is a label with its own vocabulary (``"openai"`` by default) that routes
#: nothing. No other section has a kind (``deny_unknown_fields`` everywhere).
_KIND_PATHS = (("provider",), ("magi", "kind"))

#: The only key :func:`point_config_at` may change, in any table.
_ENDPOINT_KEY = "base_url"

#: The paths a run that forbids model calls may still request: the capability
#: probe (``src/magi/probe.rs``), which sizes the batches from the root of an
#: Ollama daemon before the budget refusal and costs no completion. Anything
#: else -- a trio completion, a gate decision, an embedding, a path nobody
#: listed -- is counted. They are magi-core's ``OllamaProvider`` ``window()``
#: and ``digest()`` requests, under the daemon root (``/v1`` is normalised
#: away), and ``smoke/tests/test_rust_sync.py`` pins them to the version
#: ``Cargo.lock`` resolves.
PROBE_PATHS = frozenset({"/api/show", "/api/tags"})

#: What the counter answers: unavailable, so a probe fails open (the window
#: counts as unmeasured) and a completion fails at once instead of waiting.
MODEL_COUNTER_STATUS = 503

#: Bytes of a request body the counter drains at a time.
MODEL_COUNTER_CHUNK_BYTES = 64 * 1024

#: Requests the counter keeps by method and path. A product that floods the
#: loopback counter during a run would otherwise grow the record without
#: limit; past this many only the totals grow. 64 is far above the three
#: probes a refusal run sends, so a run inside the contract is kept whole.
MODEL_COUNTER_MAX_RECORDED = 64

#: The longest one connection to the counter may stay open, from accept to
#: close. A per-read timeout alone does not bound a client that keeps sending
#: a byte at a time, and the counter's shutdown joins every handler thread, so
#: without a total a trickling product would hold the smoke run open. Twice
#: the per-read timeout leaves a well-behaved request far more than it needs.
MODEL_COUNTER_CONNECTION_DEADLINE_S = 2 * ERROR_BACKEND_TIMEOUT_S

#: The longest request line the counter parses, the limit
#: ``BaseHTTPRequestHandler`` applies; a longer one is recorded unparsed.
MAX_REQUEST_LINE_BYTES = 65536

#: How the counter records a connection whose request line did not parse (or
#: whose path did not split as a URL). Its path is in no probe list, so it
#: always counts as a request past the probe.
UNPARSED_REQUEST = ("UNPARSED", "<request line that did not parse>")


class RequestLog:
    """A bounded, thread-safe record of the requests :func:`model_counter` saw.

    Every request is counted, and classified against :data:`PROBE_PATHS` as it
    arrives, but only the first *cap* are kept by method and path. A request
    past the probe therefore always counts, even one past the cap whose path
    was not kept, so a dropped record can never let a run pass.

    Example:
        >>> log = RequestLog(cap=1)
        >>> log.record("POST", "/api/show")
        >>> log.record("POST", "/api/chat")
        >>> log.recorded, log.total, log.past_the_probe_total, log.overflowed
        ((('POST', '/api/show'),), 2, 1, True)
    """

    def __init__(self, cap: int | None = None) -> None:
        """Start an empty log.

        Args:
            cap: Requests kept by method and path; ``None`` reads
                :data:`MODEL_COUNTER_MAX_RECORDED` at construction.
        """
        self._cap = MODEL_COUNTER_MAX_RECORDED if cap is None else cap
        self._lock = threading.Lock()
        self._recorded: list[tuple[str, str]] = []
        self._total = 0
        self._past_the_probe_total = 0

    def record(self, method: str, path: str) -> None:
        """Count one request and keep it while the log is under its cap.

        Args:
            method: The request's method.
            path: Its path, without the query.
        """
        with self._lock:
            self._total += 1
            if path not in PROBE_PATHS:
                self._past_the_probe_total += 1
            if len(self._recorded) < self._cap:
                self._recorded.append((method, path))

    @property
    def recorded(self) -> tuple[tuple[str, str], ...]:
        """tuple: The kept ``(method, path)`` pairs, in arrival order."""
        with self._lock:
            return tuple(self._recorded)

    @property
    def total(self) -> int:
        """int: Every request seen, kept or not."""
        with self._lock:
            return self._total

    @property
    def past_the_probe_total(self) -> int:
        """int: Every request seen whose path is not in :data:`PROBE_PATHS`."""
        with self._lock:
            return self._past_the_probe_total

    @property
    def overflowed(self) -> bool:
        """bool: Whether any request was counted but not kept."""
        with self._lock:
            return self._total > len(self._recorded)


class _CountingHandler(_RecordingHandler):
    """Records every request, then answers it :data:`MODEL_COUNTER_STATUS`.

    The request is recorded as soon as its line and headers are read, before
    its body, so one the product abandons half-sent still counts. Only the
    method and the path are kept: no header, query or body, which is where a
    credential would travel.
    """

    #: Bounds a request that stops mid-body, so a stuck client cannot hold the
    #: run (or the server's shutdown, which joins its threads) open.
    timeout = ERROR_BACKEND_TIMEOUT_S

    def setup(self) -> None:
        """Open the connection's streams and arm its total deadline.

        MAGI FIX (S6 loop 27): the per-read :attr:`timeout` does not bound a
        client that sends a byte at a time, and the counter's shutdown joins
        this thread, so a trickling product held the smoke run open
        (F1a1-caspar). After :data:`MODEL_COUNTER_CONNECTION_DEADLINE_S` the
        socket is shut down, which ends any read in progress; the request,
        recorded once its headers arrived, still counts.
        """
        super().setup()
        self._deadline = threading.Timer(MODEL_COUNTER_CONNECTION_DEADLINE_S,
                                         self._cut)
        self._deadline.daemon = True
        self._deadline.start()

    def _cut(self) -> None:
        """Shut the connection down once its deadline passes."""
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed by the client or by finish()

    def finish(self) -> None:
        """Disarm the deadline, then close the connection's streams."""
        self._deadline.cancel()
        super().finish()

    def _answer(self) -> None:
        """Drain this request's body and answer it unavailable."""
        try:
            _drain_body(self.rfile, self.headers)
            self.send_response(MODEL_COUNTER_STATUS)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
        except OSError:
            # The deadline cut the connection, or the client hung up: the
            # request is recorded, and there is nobody left to answer.
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        """Swallow the access log, as :class:`_EchoingHandler` does.

        Args:
            fmt: The format string the base class passes.
            *args: Its arguments.
        """


@contextlib.contextmanager
def model_counter():
    """Serve a loopback endpoint that records every request, for the block.

    Yields:
        tuple: The base URL to point the product at, ending in ``/v1``, and
        the :class:`RequestLog` that counts every request and keeps the first
        :data:`MODEL_COUNTER_MAX_RECORDED` by ``(method, path)``. Read it
        after the block: leaving it joins every request's thread.
    """
    server = _BoundedThreadingServer(("127.0.0.1", 0), _CountingHandler)
    # Joined on close, unlike ThreadingHTTPServer's default, so the log is
    # complete when the block ends; each connection's total deadline
    # (MODEL_COUNTER_CONNECTION_DEADLINE_S) bounds the join.
    server.daemon_threads = False
    server.log = RequestLog()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/v1" % server.server_address[1], server.log
    finally:
        server.shutdown()
        _drain_backlog(server)
        server.server_close()
        thread.join(timeout=ERROR_BACKEND_TIMEOUT_S)


def _drain_backlog(server: http.server.ThreadingHTTPServer) -> None:
    """Handle every connection still waiting to be accepted by *server*.

    MAGI FIX (S6 loop 25): ``serve_forever`` stops as soon as shutdown is
    requested, even with a connection ready to accept, and ``server_close``
    then drops it, so a request the product sent just before it exited was
    never counted (F1a1-caspar). Each one is accepted and handled here, on a
    thread ``server_close`` joins, before the listening socket is closed.

    The drain stops once nothing is waiting, or after
    :data:`ERROR_BACKEND_TIMEOUT_S`, so a client that keeps connecting cannot
    hold the run open; the product has exited by then, so whatever arrives
    later is not this run's.

    Args:
        server: The counter, already shut down (its loop has returned).
    """
    server.timeout = 0
    deadline = time.monotonic() + ERROR_BACKEND_TIMEOUT_S
    with selectors.DefaultSelector() as selector:
        selector.register(server.socket, selectors.EVENT_READ)
        while selector.select(0) and time.monotonic() < deadline:
            server.handle_request()


def requests_past_the_probe(log: RequestLog) -> tuple[str, ...]:
    """The requests *log* saw that are not the capability probe.

    Empty exactly when no request past the probe arrived, whether or not it
    was kept: a request the log counted but did not keep is still reported,
    by a trailing entry with the total and the fact that the log overflowed.

    Args:
        log: The :class:`RequestLog` :func:`model_counter` yields.

    Returns:
        tuple[str, ...]: ``"<METHOD> <path>"`` of each kept request whose path
        is not in :data:`PROBE_PATHS`, in order, then, when some were not
        kept, one entry naming how many there were in all.

    Example:
        >>> log = RequestLog(cap=2)
        >>> for path in ("/api/show", "/api/chat", "/api/chat"):
        ...     log.record("POST", path)
        >>> for entry in requests_past_the_probe(log):
        ...     print(entry)
        POST /api/chat
        2 requests past the probe in all; the counter overflowed after recording the first 2 requests
    """
    recorded = log.recorded
    named = tuple("%s %s" % (method, path) for method, path in recorded
                  if path not in PROBE_PATHS)
    seen = log.past_the_probe_total
    if seen == len(named):
        return named
    return named + ("%d %s past the probe in all; the counter overflowed after "
                    "recording the first %d requests"
                    % (seen, "request" if seen == 1 else "requests", len(recorded)),)


def point_config_at(config_path, url: str) -> None:
    """Rewrite the ``magi.toml`` at *config_path* so every endpoint is *url*.

    Every active ``base_url`` line, in any table, becomes ``base_url = "<url>"``
    (a commented one is left alone, and its table inherits the root). With no
    root ``base_url`` one is added first: the product's built-in default is a
    real local daemon.

    The rewrite reads lines, not TOML: any line starting with ``[`` ends the
    root table, so a root multi-line array whose element line starts with
    ``[`` (``  [1, 2],``) hides a root ``base_url`` after it, and the inserted
    one duplicates the key. The product's root holds no array, and the result
    is parsed back before the run (:func:`endpoints_off_the_counter`), so the
    duplicate is a :class:`HarnessError` (exit 3), never a miscount.

    Complexity: ``O(file bytes)``.

    Args:
        config_path: The configuration to rewrite in place.
        url: The endpoint every section must reach.

    Raises:
        HarnessError: When the file is missing or cannot be read or written.
            The workspace is the harness's own copy of the environment, so a
            missing file is a harness setup failure: :mod:`smoke.__main__`
            maps it to exit 3, never to a scenario ``FAIL``.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "magi.toml"
        ...     _ = path.write_text('[magi]\\n  base_url = "http://x/v1"\\n')
        ...     point_config_at(path, "http://127.0.0.1:9/v1")
        ...     path.read_text().splitlines()
        ['base_url = "http://127.0.0.1:9/v1"', '[magi]', 'base_url = "http://127.0.0.1:9/v1"']
    """
    line = 'base_url = "%s"' % url
    try:
        text = pathlib.Path(config_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HarnessError("could not read %s to count its endpoints: %s"
                           % (config_path, exc)) from exc
    lines, in_root, root_declared = [], True, False
    for original in text.splitlines():
        stripped = original.strip()
        if stripped.startswith("["):
            in_root = False
        if stripped.startswith("base_url") and stripped[len("base_url"):].lstrip().startswith("="):
            root_declared = root_declared or in_root
            original = line
        lines.append(original)
    if not root_declared:
        lines.insert(0, line)
    try:
        pathlib.Path(config_path).write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    except OSError as exc:
        raise HarnessError("could not point %s at the request counter: %s"
                           % (config_path, exc)) from exc


def endpoints_off_the_counter(config_path, url: str) -> tuple[str, ...]:
    """What in the ``magi.toml`` at *config_path* could reach past *url*.

    MAGI FIX (smoke review, Melchior): the counter proves only that it saw no
    request past the probe, so an endpoint :func:`point_config_at` missed (a
    quoted key, an inline table) would let a model call go elsewhere and the
    refusal run pass vacuously. Counting at least one request instead is not
    a proof: R13 is refused before the probe, and only ``kind = "ollama"``
    probes at all. So the file is parsed as the product parses it and every
    endpoint it declares is checked.

    Complexity: ``O(file bytes)``.

    Args:
        config_path: The rewritten configuration.
        url: The counter's URL, which every endpoint must be.

    The product's endpoint keys are exactly the ``base_url`` of the root,
    ``[magi]``, ``[embedding]``, ``[decision]`` and ``[code_review]``; every
    section is ``deny_unknown_fields``, so no ``endpoint``/``api_base``/``url``
    key can exist, and a ``[[magi.fallback]]`` or gate fallback names only a
    model, which is sent to its section's endpoint. The environment variables
    that steer one, ``OPENAI_BASE_URL``, ``MAGI_PROVIDER`` and the proxy
    variables (:data:`_PROXY_VARIABLES`, ``NO_PROXY``), are set or removed by
    :meth:`RunExecutor._counted_endpoints`.

    Returns:
        tuple[str, ...]: One entry per problem, table by table: the root
        ``base_url`` missing or not *url*, any other ``base_url`` that is not
        *url* (by dotted key), and a provider kind (:data:`_KIND_PATHS`) that
        is neither blank nor :data:`_COUNTED_KIND`. Empty when every endpoint
        is the counter.

    Raises:
        HarnessError: When the file cannot be read or is not valid TOML.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "magi.toml"
        ...     _ = path.write_text('base_url = "u"\\n[magi]\\nkind = "anthropic"\\n')
        ...     endpoints_off_the_counter(path, "u")
        ('magi.kind = "anthropic", not "ollama", the only kind the counter stands in for',)
    """
    document = _parsed_config(config_path)
    problems = []
    if document.get(_ENDPOINT_KEY) != url:
        problems.append("the root base_url is %r, not the counter"
                        % document.get(_ENDPOINT_KEY))
    for path, value in _config_leaves(document):
        if path[-1] == _ENDPOINT_KEY and len(path) > 1 and value != url:
            problems.append("%s is %r, not the counter" % (".".join(path), value))
    for path in _KIND_PATHS:
        value = _config_value(document, path)
        # Absent or blank resolves to the product's default kind, which
        # smoke/tests/test_rust_sync.py (CountedKindTests) pins to _COUNTED_KIND.
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str) or value.strip() != _COUNTED_KIND:
            problems.append('%s = "%s", not "%s", the only kind the counter '
                            "stands in for" % (".".join(path), value, _COUNTED_KIND))
    return tuple(problems)


#: Marks a key missing from one side of :func:`changes_besides_endpoints`.
_ABSENT = object()


def changes_besides_endpoints(original: str, config_path) -> tuple[str, ...]:
    """What :func:`point_config_at` changed in the config besides its endpoints.

    MAGI FIX (smoke review, Caspar): the rewrite is line-based, so a line that
    only looks like a ``base_url`` -- inside a multi-line string, say -- is
    rewritten too, and the run would then exercise a configuration nobody
    wrote. Both texts are parsed and compared with every ``base_url`` masked.

    Complexity: ``O(file bytes)``.

    Args:
        original: The configuration text before the rewrite.
        config_path: The rewritten configuration.

    Returns:
        tuple[str, ...]: One entry per dotted key whose value differs, was
        added or was removed, in sorted order. Empty when only endpoints moved.

    Raises:
        HarnessError: When either text cannot be read or is not valid TOML.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "magi.toml"
        ...     _ = path.write_text('base_url = "u"\\nprovider = "b"\\n')
        ...     changes_besides_endpoints('provider = "a"\\n', path)
        ("provider is 'b', was 'a'",)
    """
    try:
        before = tomllib.loads(original)
    except tomllib.TOMLDecodeError as exc:
        raise HarnessError("could not parse the configuration before pointing "
                           "it at the counter: %s" % exc) from exc
    masked = [{path: value for path, value in _config_leaves(document)
               if path[-1] != _ENDPOINT_KEY}
              for document in (before, _parsed_config(config_path))]
    was, now = masked
    return tuple("%s is %r, was %r" % (".".join(path), now.get(path, "<absent>"),
                                      was.get(path, "<absent>"))
                 for path in sorted(was.keys() | now.keys())
                 if was.get(path, _ABSENT) != now.get(path, _ABSENT))



def _parsed_config(config_path) -> dict:
    """Read and parse the ``magi.toml`` at *config_path*.

    Args:
        config_path: The file.

    Returns:
        dict: The parsed document.

    Raises:
        HarnessError: When the file cannot be read or is not valid TOML.
    """
    try:
        return tomllib.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise HarnessError("could not read %s back to check its endpoints: %s"
                           % (config_path, exc)) from exc


def _config_leaves(document: dict):
    """Every scalar of a parsed TOML document, with its key path.

    Arrays are indexed by position (``"0"``, ``"1"``, ...); an empty table or
    array is a leaf of its own, so removing one is visible.

    Complexity: ``O(nodes)``.

    Args:
        document: A parsed document.

    Yields:
        tuple[tuple[str, ...], object]: The path and the value, breadth first.
    """
    pending = [((), document)]
    while pending:
        prefix, node = pending.pop(0)
        items = (node.items() if isinstance(node, dict)
                 else ((str(i), v) for i, v in enumerate(node)))
        for key, value in items:
            path = prefix + (key,)
            if isinstance(value, (dict, list)) and value:
                pending.append((path, value))
            else:
                yield path, value


def _config_value(document: dict, path: tuple):
    """The value at *path* in *document*, or ``None`` where any table is missing.

    Args:
        document: A parsed document.
        path: The key path.

    Returns:
        object: The value, or ``None``.
    """
    node = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node

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
    # any model is reached. Their input is the files written into the review
    # workspace, not stdin, which is empty: ``payload_size`` only selects the
    # stdin payload (:meth:`RunExecutor._stdin_for`), so ``PAYLOAD_SMALL``
    # here means "no generated stdin payload", not the size of what they read.
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
        forbids_model_calls=True,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60),
    "R14": RunDefinition(
        run_id="R14",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_SKIPPED_DIR,),
        stdin=b"",
        forbids_model_calls=True,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_skipped_files()),
    "R15": RunDefinition(
        run_id="R15",
        argv=("code-review",) + REVIEW_FLAGS
        + ("--timeout", str(REVIEW_TOO_SHORT_TIMEOUT_S), "--log-level",
           "error", REVIEW_GEN_DIR),
        stdin=b"",
        forbids_model_calls=True,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_input_files()),
    # MAGI FIX (S6 loop 3): the workspace ROOT, so ``.magi`` and ``.git`` are
    # enumeration candidates; R12-R15 read subdirectories and never were (F2).
    "R16": RunDefinition(
        run_id="R16",
        argv=("code-review",) + REVIEW_FLAGS
        + ("--timeout", str(REVIEW_TOO_SHORT_TIMEOUT_S), REVIEW_ROOT_PATH),
        stdin=b"",
        forbids_model_calls=True,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_skipped_files()),
    # R17 feeds S52: a one-batch review against the real trio whose embedder is
    # unreachable, so it must report without documentation and history.
    "R17": RunDefinition(
        run_id="R17",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_CONTEXT_DIR,),
        stdin=b"",
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=900,
        files=review_context_files(), unreachable_embedder=True),
    # R18 feeds S53: a one-batch review by the real trio of a file whose content
    # holds a toy key, which the product must mask and review. R19 feeds S54:
    # a directory of only a private-key block, refused before any model.
    "R18": RunDefinition(
        run_id="R18",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_MASKED_DIR,),
        stdin=b"",
        needs_trio=True, payload_size=PAYLOAD_SMALL, timeout_s=900,
        files=review_masked_files()),
    "R19": RunDefinition(
        run_id="R19",
        argv=("code-review",) + REVIEW_FLAGS + (REVIEW_KEYBLOCK_DIR,),
        stdin=b"",
        forbids_model_calls=True,
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=60,
        files=review_keyblock_files()),
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
        #: How many requests the error backend of the run in progress answered,
        #: one entry per backend served (:meth:`_invoke`); reset by
        #: :meth:`execute`, which runs one definition at a time.
        self._reached: list[int] = []

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
        if definition.unreachable_embedder:
            if workspace is None:
                raise HarnessError("run %s points its embedder away but has no review "
                                   "workspace" % definition.run_id)
            point_embedding_at(workspace.joinpath(*CONFIG_PATH_PARTS),
                               UNREACHABLE_EMBEDDER_URL)
        # A second listing, after the run's files are written and before the
        # product starts: what the product's own enumeration will see.
        protected = () if workspace is None else protected_candidates(workspace)
        stdin = self._stdin_for(definition)
        baseline = None
        contacted: list[str] = []
        self._reached = []
        with self._counted_endpoints(definition, workspace, contacted) as runnable:
            try:
                if runnable.rotates:
                    # The baseline is taken INSIDE, after the guard that refuses
                    # to rotate a credential there is nothing to put back.
                    # Reading the environment before that guard runs touches it
                    # on the one path that is supposed to have changed nothing.
                    with self._rotated_credential() as counts:
                        baseline = counts
                        output = self._invoke(runnable, stdin)
                else:
                    output = self._invoke(runnable, stdin)
                timed_out = False
            except TimedOut as expired:
                output = expired.output or ProductOutput(
                    stdout=b"", stderr=b"", exit_code=0,
                    command=self._command(runnable))
                timed_out = True
        unreadable = []
        review_log = self._capture_review_log(definition, unreadable)
        return RunResult(
            run_id=definition.run_id,
            output=output,
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            planted=definition.planted,
            stdin_bytes=len(stdin),
            baseline=baseline,
            review_log=review_log,
            review_log_unreadable=tuple(unreadable),
            protected_visible=protected,
            model_requests=tuple(contacted),
            backend_requests=sum(self._reached) if self._reached else None,
        )

    @contextlib.contextmanager
    def _counted_endpoints(self, definition: RunDefinition, workspace, contacted):
        """Point a run that forbids model calls at a request counter.

        MAGI FIX (S6 loop 19): R13-R16 never showed that no model was
        contacted, so a product that paid for one and then refused passed
        S48-S51 (F1a1-balthasar).

        Every endpoint of the review workspace's ``magi.toml``
        (:func:`point_config_at`) and ``OPENAI_BASE_URL``, which overrides the
        root one, reach :func:`model_counter`; nothing reaches a real backend.
        The rewritten file is checked before the run
        (:func:`endpoints_off_the_counter`), and compared with the original
        everywhere but its endpoints (:func:`changes_besides_endpoints`),
        ``MAGI_PROVIDER`` is blanked, the proxy variables
        (:data:`_PROXY_VARIABLES`, in every case spelling) are removed and
        ``NO_PROXY`` is set to loopback, so the harness's own shell cannot
        steer the run elsewhere: reqwest honours the system proxy and does not
        exempt ``127.0.0.1``, so an inherited ``HTTP_PROXY`` would carry the
        counter's requests to the proxy and leave the counter empty.

        Only these runs lose the proxy. Every other run keeps the harness's
        environment as it is, because some of them reach a real backend off
        loopback (the decide scenarios name ``openrouter.ai``) and an operator
        behind a proxy needs it there; a refusal run talks to nothing but the
        local counter, so it can never need one. An endpoint
        off the counter is the harness's setup, not the product's verdict, so
        it is a :class:`HarnessError` (exit 3), never a scenario ``FAIL``. The
        requests the counter saw are recorded even when the run times out:
        the timeout is caught inside this block. Any other exception that
        leaves the block carries them as a note (``add_note``), which
        :mod:`smoke.__main__` prints beside the harness failure.

        The proof covers exactly what the configuration and these variables
        steer. A product defect that reached a model some other way
        (a hardcoded URL, a subprocess with its own endpoint) would leave the
        counter empty and the run would pass; that is outside what a stand-in
        endpoint can observe, and the scenarios do not claim it.

        Args:
            definition: The run about to execute.
            workspace: Its review workspace, or ``None``.
            contacted: A list that receives, once the run is over, each request
                past the probe (:func:`requests_past_the_probe`).

        Yields:
            RunDefinition: *definition* itself for any other run; otherwise a
            copy whose ``env`` also points ``OPENAI_BASE_URL`` at the counter,
            blanks ``MAGI_PROVIDER``, removes the proxy variables (a ``None``
            value, :func:`smoke.binary.child_environment`) and sets
            ``NO_PROXY`` to :data:`_LOOPBACK_HOSTS`.

        Raises:
            HarnessError: When there is no workspace, its configuration is
                missing or cannot be rewritten, or the rewritten configuration
                still declares an endpoint off the counter, names a provider
                kind other than :data:`_COUNTED_KIND`, or differs from the
                original in anything but an endpoint.
        """
        if not definition.forbids_model_calls:
            yield definition
            return
        if workspace is None:
            raise HarnessError("run %s forbids model calls but has no review "
                               "workspace to point at a counter" % definition.run_id)
        config = workspace.joinpath(*CONFIG_PATH_PARTS)
        try:
            original = config.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise HarnessError("could not read %s to count its endpoints: %s"
                               % (config, exc)) from exc
        log = None
        try:
            with model_counter() as (url, log):
                point_config_at(config, url)
                off = (endpoints_off_the_counter(config, url)
                       + changes_besides_endpoints(original, config))
                if off:
                    raise HarnessError(
                        "run %s forbids model calls, but its configuration could "
                        "still reach past the request counter: %s"
                        % (definition.run_id, "; ".join(off)))
                yield dataclasses.replace(
                    definition, env=definition.env + ((_BACKEND_URL_VARIABLE, url),
                                                      (_PROVIDER_VARIABLE, ""))
                    + tuple((name, None) for name in _PROXY_VARIABLES)
                    + ((_NO_PROXY_VARIABLE, _LOOPBACK_HOSTS),))
        except BaseException as escaping:
            # MAGI FIX (S6 loop 27): an exception other than the timeout
            # skipped the record, so the exit-3 report lost the requests that
            # would explain it (F1a1-melchior). The counter has been joined
            # here, so the log is complete; it rides on the exception.
            seen = () if log is None else requests_past_the_probe(log)
            if seen:
                escaping.add_note("requests the counter saw before this "
                                  "failure: %s" % "; ".join(seen))
            raise
        contacted.extend(requests_past_the_probe(log))

    def _review_workspace(self) -> pathlib.Path:
        """The directory :data:`REVIEW_WORKSPACE_TOKEN` stands for.

        Returns:
            pathlib.Path: ``<scratch>/review-ws``.
        """
        return pathlib.Path(self._env.scratch_dir) / REVIEW_WORKSPACE_DIR

    def _first_link_down_to(self, *below: str):
        """The first link from the repository root down to the workspace path *below*.

        Args:
            *below: Components under the review workspace; none means the
                workspace itself.

        Returns:
            pathlib.Path | None: The first component that is a symlink or a
            junction, the scratch area and the workspace included, or ``None``.

        Raises:
            OSError: When a component cannot be inspected.
        """
        base, scratch = _chain_from_repository(self._env.scratch_dir)
        return _linked_component(
            base, scratch.joinpath(REVIEW_WORKSPACE_DIR, *below))

    def _capture_review_log(self, definition: RunDefinition, unreadable=None):
        """Read the review workspace's run log right after a run.

        MAGI FIX (S6 loop 3): the next review run rebuilds the workspace from
        nothing, so a scenario that read the live directory after the runs
        depended on being evaluated before a later review run wiped it (F4).

        MAGI FIX (S6 loop 12): a file that could not be read was dropped
        without a word, so a locked log read as a product that never logged
        (F1a-c2). It is now named in *unreadable*.

        Args:
            definition: The run that just finished.
            unreadable: A list that receives ``"<path> (<error>)"``, *path*
                relative to ``.magi/logs``, for each
                file that could not be read or was over a cap
                (:func:`_read_bounded`), and each link or unlistable
                directory left out (:func:`_regular_files_without_links`); a
                linked ``.magi`` or ``.magi/logs`` is named by its path under
                the workspace instead, and a linked workspace or component
                above it by its full path. ``None`` discards them.

        Returns:
            tuple[bytes, ...] | None: Each readable file's bytes under
            ``.magi/logs`` within the caps, in sorted path order; ``None`` for a run without a
            review workspace or when the directory does not exist; ``()`` when
            any component from the repository root down to ``.magi/logs`` is
            a link (or cannot be tested), before the walk or after the read,
            which is named in *unreadable* and never followed.
        """
        if REVIEW_WORKSPACE_TOKEN not in definition.argv:
            return None
        workspace = self._review_workspace()
        directory = workspace.joinpath(*LOG_DIR_PARTS)
        # MAGI FIX (S6 loop 15): the walk tests every entry but not its top, and
        # os.scandir follows a linked .magi or .magi/logs, so the harness read
        # what the link points at (F1a1-m1). Each component is tested first.
        # MAGI FIX (S6 loop 17): only .magi and .magi/logs were tested, while
        # the product runs with the workspace as its cwd and may swap the
        # workspace (or a component above it) for a link before it exits
        # (F1a1-caspar, F1a1-melchior). The whole chain from the repository
        # root is tested again here, after the run.
        refusal = self._log_chain_refusal(workspace)
        if refusal is not None:
            if unreadable is not None:
                unreadable.append(refusal)
            return ()
        if not directory.is_dir():
            return None
        # MAGI FIX (S6 loop 14): rglob descended into a junction and read_bytes
        # followed a link, so a link under .magi/logs had the harness read what
        # it points at (F1a1-m3). Links are named in *unreadable*, never read.
        refused, identities = [], {}
        files = _regular_files_without_links(directory, refused, identities)
        captured = _read_bounded(directory, files, refused, identities)
        # MAGI FIX (S6 loop 19): a link swapped onto the chain between the test
        # above and the walk had the walk list, and the read open, what it
        # points at (F1a1-balthasar). The chain is tested once more after the
        # read, and a capture taken through a link is dropped whole.
        refusal = self._log_chain_refusal(workspace)
        if refusal is not None:
            if unreadable is not None:
                unreadable.append(refusal)
            return ()
        if unreadable is not None:
            unreadable.extend(refused)
        return tuple(captured)

    def _log_chain_refusal(self, workspace: pathlib.Path):
        """Why the log capture must not go through the chain down to ``.magi/logs``.

        Args:
            workspace: The review workspace.

        Returns:
            str | None: ``"<component> (<why>)"`` for the first link from the
            repository root down to ``.magi/logs`` (named under *workspace*
            when it lies there, by its full path otherwise), or for a test that
            failed; ``None`` when no component is a link.
        """
        try:
            linked = self._first_link_down_to(*LOG_DIR_PARTS)
        except OSError as exc:
            return "%s (%s)" % ("/".join(LOG_DIR_PARTS), exc)
        if linked is None:
            return None
        named = (linked.relative_to(workspace).as_posix()
                 if workspace in linked.parents else str(linked))
        return "%s (%s)" % (named, LINK_NOT_FOLLOWED)

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
        scratch = _resolved(self._env.scratch_dir, "the scratch area")
        # MAGI FIX (S6 loop 3): containment is checked before anything is
        # deleted OR created. The tree is deleted wholesale and then rebuilt,
        # so only a path that resolves strictly under the scratch area may be
        # touched at all (F3).
        if scratch not in _resolved(workspace, "the review workspace").parents:
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
        # MAGI FIX (S6 loop 12): only the leaf was checked, so a parent inside
        # the scratch area that is a link resolved under it too and rmtree
        # deleted the tree it names (F1a-c1). Every component below the scratch
        # area is checked, the leaf included.
        # MAGI FIX (S6 loop 13): an OSError raised by the link checks escaped the
        # executor as a traceback instead of exit 3 (F1a-c1).
        # MAGI FIX (S6 loop 14): F1a1-c1 read rmtree as following a junction
        # inside the old workspace on 3.11. It does not: since Python 3.8
        # shutil's own Windows walk (``_rmtree_isdir``) treats a directory with
        # the mount-point reparse tag as a link and unlinks it without listing
        # it, whatever ``os.path.isjunction`` says; only the TOP path being a
        # link is unsafe, and that is refused above. Pinned on 3.11 and later by
        # test_rebuilding_the_workspace_never_deletes_through_a_junction_inside_it.
        # MAGI FIX (S6 loop 17): the walk started below the scratch area, so a
        # scratch area that is itself a link (or a linked component above it)
        # passed both checks, and rmtree and copytree went through it
        # (F1a1-caspar). The walk starts at the repository root.
        # MAGI FIX (S6 loop 28): the source .magi was tested with everything
        # under it but not the directories above it, so an environment
        # directory swapped for a link had copytree copy the tree it points at
        # (F1b1-balthasar). Its chain from the repository root is tested too.
        try:
            linked = (self._first_link_down_to()
                      or _linked_ancestor(self._env.magi_dir))
            if linked is None:
                _reject_links(self._env.magi_dir)
        except OSError as exc:
            raise HarnessError("could not check the review workspace at %s "
                               "for links: %s" % (workspace, exc)) from exc
        if linked is not None:
            raise HarnessError(
                "refusing to delete the review workspace at %s: %s is a "
                "link" % (workspace, linked))
        try:
            # MAGI FIX (S6 loop 15): a read-only file in the old workspace made
            # rmtree raise on Windows and every later review run end at exit 3
            # (F1a1-m3); _remove_tree clears the bit and retries once.
            if workspace.exists():
                _remove_tree(workspace)
            workspace.mkdir(parents=True)
            # MAGI FIX (S6 loop 6): the environment's logs/ came along, so the
            # captured review log carried earlier runs' lines too (F1a-m4).
            # MAGI FIX (S6 loop 29): copytree followed whatever stood at each
            # path when it copied, so a link (or a hard link) swapped in after
            # the check above reached the workspace (F1a1b-melchior,
            # F1a1b-caspar). The copy itself refuses them now.
            _copy_tree_without_links(
                self._env.magi_dir, workspace / MAGI_DIR_NAME,
                _skip_top_level_logs(self._env.magi_dir))
            subprocess.run(git_command(workspace, "init", "-q"), check=True,
                           env=git_env(), capture_output=True,
                           timeout=GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError, shutil.Error) as exc:
            # MAGI FIX (S6 loop 15): CalledProcessError's str() carries no
            # captured stderr, so git's own reason was dropped (F1a1-b1).
            raise HarnessError("could not build the review workspace at %s: %s"
                               % (workspace, _error_detail(exc))) from exc
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
        root = _resolved(root, "the workspace of run %s" % definition.run_id)
        for relative, body in definition.files:
            # Containment is checked on the RESOLVED target: resolve() follows
            # every symlink and collapses every "..", so a link under *root*
            # that points outside lands outside and is refused like "../x".
            target = _resolved(root / relative, "a file of run %s" % definition.run_id)
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

        Raises:
            HarnessError: If the review workspace cannot be resolved.
        """
        root = str(self._env.root)
        resolved = {WORKSPACE_TOKEN: root}
        if REVIEW_WORKSPACE_TOKEN in definition.argv:
            # MAGI FIX (S6 loop 11, F1a-m1): resolved, because the product starts with
            # cwd = this workspace and a relative -w would be joined to it a second time.
            resolved[REVIEW_WORKSPACE_TOKEN] = str(
                _resolved(self._review_workspace(), "the review workspace"))
        return [resolved.get(word, word) for word in definition.argv]

    def _invoke(self, definition: RunDefinition,
                stdin: bytes) -> ProductOutput:
        """Run the product once for *definition*.

        For a run with an error backend (:data:`ERROR_BACKEND_TOKEN`), how
        many requests the backend answered is appended to the run's
        ``_reached`` record, even when the run times out or fails.

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
        # MAGI FIX (S6 loop 29): nothing recorded whether the product reached
        # the backend, so a run whose requests went elsewhere (a proxy) let S10
        # search output that never walked the error path (F1a1b-melchior).
        log = RequestLog()
        try:
            with self._authenticated_backend(definition, log) as url:
                resolved = {name: (url if value == ERROR_BACKEND_TOKEN else value)
                            for name, value in declared.items()}
                return self._invoke_with(definition, resolved, stdin)
        finally:
            self._reached.append(log.total)

    @contextlib.contextmanager
    def _authenticated_backend(self, definition: RunDefinition, log=None):
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
            log: A :class:`RequestLog` the backend records every request in
                (:func:`error_backend`), or ``None``.

        Yields:
            str: The authenticated URL to hand the product.
        """
        password = definition.planted[0].value if definition.planted else ""
        with error_backend(log) as url:
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
