# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
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
import threading
import time

from smoke.config import ROTATION_MARKER
from smoke.errors import HarnessError, ProductOutputError, TimedOut
from smoke.payload import PayloadBuilder
from smoke.product import ProductOutput
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
LARGE_CONSULT_TIMEOUT_S = 1800

#: The harness's own ceiling for that run, 20 % above the product's. The
#: comment in the table says why it can never be the smaller of the two.
LARGE_CONSULT_CEILING_S = 2160

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

#: The vault entries the substitution reads. BOTH pairs, and that is measured
#: rather than assumed: with only the root pair the product asked for
#: ``MAGI_BASE_URL_USER``, and with only the prefixed pair it asked for
#: ``BASE_URL_USER``. The trio section inherits the root endpoint, so both
#: resolutions happen and both need their entry.
PLACEHOLDER_ENTRIES = ("BASE_URL_USER", "BASE_URL_PASSWORD",
                       "MAGI_BASE_URL_USER", "MAGI_BASE_URL_PASSWORD")

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
                put back afterwards.
            ProductOutputError: If the product could not be started. That is
                not a hang and not a verdict, so it is not folded into the
                result.
        """
        started = time.monotonic()
        stdin = self._stdin_for(definition)
        try:
            if definition.rotates:
                with self._rotated_credential():
                    output = self._invoke(definition, stdin)
            else:
                output = self._invoke(definition, stdin)
            timed_out = False
        except TimedOut as expired:
            output = expired.output or ProductOutput(
                stdout=b"", stderr=b"", exit_code=0,
                command=self._command(definition))
            timed_out = True
        return RunResult(
            run_id=definition.run_id,
            output=output,
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            planted=definition.planted,
            stdin_bytes=len(stdin),
        )

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
        return [root if word == WORKSPACE_TOKEN else word
                for word in definition.argv]

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
            for name in PLACEHOLDER_ENTRIES:
                self._vault_set(
                    name,
                    (password if name.endswith("PASSWORD") else BACKEND_USER)
                    .encode("utf-8"))
            try:
                yield url.replace(
                    "http://",
                    "http://%s:%s@" % (USER_PLACEHOLDER, PASSWORD_PLACEHOLDER))
            finally:
                for name in PLACEHOLDER_ENTRIES:
                    self._vault_remove(name)

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
        return self._binary.invoke(
            self._command(definition), stdin=stdin,
            env=overlay, timeout=definition.timeout_s)

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
        # The executor's OWN binary, not the module-level accessor: this
        # class is handed everything it needs, and reaching for module
        # state here would make it depend on configure() having run.
        body = PayloadBuilder(self._binary.repo_root).build(
            self._config.payload_target_bytes)
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
        be absent in exactly the case it exists for. The restore lives in a
        ``finally``, covering an exception and an orderly interruption. What
        neither covers is a power cut, and that is what the marker is for.

        The marker is a separate entry recognised by NAME rather than a
        sentinel recognised by its value, and that is forced by the product:
        it never prints a stored value, so a preflight cannot read one back.

        Yields:
            None: While the credential holds the sentinel.

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
        self._vault_set(ROTATION_MARKER, ROTATION_MARKER_VALUE)
        try:
            self._vault_set(name, ROTATION_SENTINEL)
            yield
        finally:
            self._vault_set(name, credential.encode("utf-8"))
            self._vault_remove(ROTATION_MARKER)

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

    def _vault_set(self, name: str, value: bytes) -> None:
        """Store one secret, overwriting without asking.

        Args:
            name: The entry's name.
            value: The value, fed on standard input -- never as an argument,
                which any process on the machine could read.

        Raises:
            ProductOutputError: If the product could not be run.
        """
        self._binary.invoke(
            [_VAULT_SUBCOMMAND, _SET_SUBCOMMAND, name, _FORCE_FLAG,
             "-w", str(self._env.root)],
            stdin=value,
            env={PASSPHRASE_VARIABLE: self._config.passphrase},
            timeout=VAULT_TIMEOUT_S)

    def _vault_remove(self, name: str) -> None:
        """Delete one secret, skipping the confirmation.

        Args:
            name: The entry's name.

        Raises:
            ProductOutputError: If the product could not be run.
        """
        self._binary.invoke(
            [_VAULT_SUBCOMMAND, _REMOVE_SUBCOMMAND, name, _FORCE_FLAG,
             "-w", str(self._env.root)],
            stdin=b"",
            env={PASSPHRASE_VARIABLE: self._config.passphrase},
            timeout=VAULT_TIMEOUT_S)
