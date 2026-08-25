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
import os
import pathlib
import time

from smoke.config import ROTATION_MARKER
from smoke.errors import HarnessError, ProductOutputError, TimedOut
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
    (directory / "invocation.log").write_bytes(scrub(body, planted))


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
    """

    run_id: str
    output: ProductOutput
    duration_s: float
    timed_out: bool
    planted: tuple[PlantedSecret, ...]


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

#: An endpoint nothing listens on. Port 9 is the discard service, reserved and
#: unused, so the run fails quickly and predictably rather than waiting out a
#: routing black hole.
_DEAD_ENDPOINT = "http://127.0.0.1:9/v1"

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
    "R2": RunDefinition(
        run_id="R2",
        argv=("query",) + COMMON_FLAGS + ("--auto", "--no-memory"),
        stdin=(b"Use the ls tool to list the files in the current directory. "
               b"Then remember these two facts: I prefer Rust over Python for "
               b"systems programming, and this project is called Magi.\n"),
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    "R3": RunDefinition(
        run_id="R3",
        argv=("query",) + COMMON_FLAGS,
        stdin=b"Which language do I prefer for systems programming?\n",
        needs_trio=False, payload_size=PAYLOAD_SMALL, timeout_s=180),
    "R4": RunDefinition(
        run_id="R4",
        argv=("consult",) + COMMON_FLAGS + ("--timeout", "300",
                                            "--structured-verdicts"),
        stdin=b"Review the following source for correctness and risk.\n",
        # 420 against the product's own 300 is not a second knob: the harness
        # ceiling has to sit ABOVE the product's, or the harness would kill the
        # very abandonment this run exists to observe. The margin covers
        # process start and the JSON write.
        needs_trio=True, payload_size=PAYLOAD_LARGE, timeout_s=420),
    "R5": RunDefinition(
        run_id="R5",
        argv=("query",) + COMMON_FLAGS + ("--consult",),
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
        env=((_BACKEND_URL_VARIABLE, _DEAD_ENDPOINT),
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
        try:
            if definition.rotates:
                with self._rotated_credential():
                    output = self._invoke(definition)
            else:
                output = self._invoke(definition)
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

    def _invoke(self, definition: RunDefinition) -> ProductOutput:
        """Run the product once for *definition*.

        Args:
            definition: The run.

        Returns:
            ProductOutput: The capture, unscrubbed.

        Raises:
            ProductOutputError: If the product could not be started, or
                :class:`~smoke.errors.TimedOut` if it did not finish in time.
        """
        overlay = {PASSPHRASE_VARIABLE: self._config.passphrase}
        overlay.update(dict(definition.env))
        return self._binary.invoke(
            self._command(definition), stdin=definition.stdin, env=overlay,
            timeout=definition.timeout_s)

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
