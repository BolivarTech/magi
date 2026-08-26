# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Locating, rebuilding, hashing and invoking the release binary.

The harness runs ``target/release/magi-rs`` and never ``cargo run`` (REQ-S07):
``cargo run`` recompiles, so it would certify whatever the source tree happens
to hold rather than the artifact a user receives.

Nothing otherwise ties the certificate's commit to the binary that answered.
A ``target/`` three commits old answers ``--version`` just as well, because the
version only moves when somebody bumps it. Two separate pieces close that gap
and neither is enough alone: the certifying run always REBUILDS, which is what
says the binary comes from that commit, and the certificate records the
binary's SHA-256, which is what says it was this binary and not another. The
hash is a record of identity, not a proof of derivation -- Rust builds are not
bit-for-bit reproducible by default, so the same commit compiled elsewhere
yields a different digest.

Iterating runs reuse whatever binary is already there. They do not certify, so
they do not need the guarantee, and a rebuild would only cost time in the loop
where time is the whole point.
"""

import hashlib
import os
import pathlib
import subprocess
import sys

from smoke.errors import PreflightError, ProductOutputError, TimedOut
from smoke.product import ProductOutput

#: What a killed process is recorded as exiting with. It never exited, so no
#: real code applies; the negative value keeps it from colliding with anything
#: the product publishes, and nothing asserts on it.
KILLED_EXIT_CODE = -1


def _partial_capture(expired: subprocess.TimeoutExpired,
                     command: list[str]) -> ProductOutput:
    """Wrap whatever a timed-out child had already written.

    ``subprocess.run`` attaches the buffered streams to the exception on
    Windows and leaves them ``None`` on POSIX, so both shapes are handled here
    and an absent stream becomes empty bytes -- "nothing was captured", which
    is not the same claim as "the product emitted nothing".

    Args:
        expired: The expiry raised by ``subprocess.run``.
        command: The argv as invoked.

    Returns:
        ProductOutput: The partial capture, with :data:`KILLED_EXIT_CODE`.
    """
    return ProductOutput(
        stdout=expired.stdout or b"",
        stderr=expired.stderr or b"",
        exit_code=KILLED_EXIT_CODE,
        command=command,
    )

#: The binary's name, without the platform suffix.
BINARY_STEM = "magi-rs"

#: Cargo's release output directory, relative to the repository root.
RELEASE_PARTS = ("target", "release")

#: Appended on Windows only.
WINDOWS_SUFFIX = ".exe"
WINDOWS_PLATFORM = "win32"

#: The binary is hashed in fixed-size chunks and never read whole. A release
#: binary is megabytes; loading it into one buffer to digest it is an
#: allocation nobody needs, and the streaming form costs nothing extra.
HASH_CHUNK_BYTES = 65536

#: How long a release build is given, in seconds. A cold ``cargo build
#: --release`` over this tree is minutes, not seconds, so the bound is generous
#: on purpose: it is here to end a hung toolchain, not to police a slow one.
BUILD_TIMEOUT_SECONDS = 1800

#: How long the binary is given to answer ``--version``, in seconds. It parses
#: no configuration and opens no vault, so anything past this is hung.
VERSION_TIMEOUT_SECONDS = 60

BUILD_COMMAND = ("cargo", "build", "--release")
VERSION_FLAG = "--version"


class ReleaseBinary:
    """The release artifact of one magi-rs checkout.

    Example:
        >>> root = pathlib.Path(__file__).resolve().parent.parent
        >>> ReleaseBinary(root).path.name.startswith("magi-rs")
        True
    """

    def __init__(self, repo_root: pathlib.Path | str) -> None:
        """Bind to one checkout without touching the filesystem.

        Args:
            repo_root: The magi-rs tree whose ``target/release`` holds the
                binary.
        """
        self._repo_root = pathlib.Path(repo_root)

    @property
    def repo_root(self) -> pathlib.Path:
        """The checkout this binary belongs to.

        Returns:
            The repository root.
        """
        return self._repo_root

    @property
    def path(self) -> pathlib.Path:
        """Where the release binary lives, existing or not.

        The property is total on purpose: the preflight needs to NAME the file
        it could not find, and a property that raised when the binary is
        missing could not be used to write that message.

        Returns:
            ``<repo_root>/target/release/magi-rs`` with ``.exe`` appended on
            Windows.
        """
        name = BINARY_STEM
        if sys.platform == WINDOWS_PLATFORM:
            name += WINDOWS_SUFFIX
        return self._repo_root.joinpath(*RELEASE_PARTS, name)

    def rebuild(self) -> None:
        """Build the release binary from the current tree.

        The certifying run calls this without first asking whether it is
        needed. A ``cargo build --release`` over an already-built tree is
        cheap; certifying a stale artifact under a commit it did not come from
        is not.

        Raises:
            PreflightError: If cargo cannot be run, exits non-zero, or does not
                finish within :data:`BUILD_TIMEOUT_SECONDS`.
        """
        try:
            completed = subprocess.run(
                list(BUILD_COMMAND),
                cwd=str(self._repo_root),
                capture_output=True,
                timeout=BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreflightError(
                "could not run %s in %s: %s"
                % (" ".join(BUILD_COMMAND), self._repo_root, exc)
            ) from exc
        if completed.returncode != 0:
            raise PreflightError(
                "%s exited %d; the certifying run needs a binary built from "
                "this commit"
                % (" ".join(BUILD_COMMAND), completed.returncode)
            )

    def sha256(self) -> str:
        """Digest the binary on disk.

        Read in BINARY mode and in chunks: binary because a text-mode read
        would translate line endings and digest something the file does not
        contain, and chunked because the size is unbounded.

        Complexity: ``O(size of the binary)`` in time, ``O(1)`` in memory.

        Returns:
            The digest as 64 lowercase hexadecimal characters.

        Raises:
            PreflightError: If the binary is missing or cannot be read. A
                certificate cannot name an artifact that is not there, so this
                cuts rather than degrading.
        """
        digest = hashlib.sha256()
        try:
            with self.path.open("rb") as handle:
                while True:
                    chunk = handle.read(HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError as exc:
            raise PreflightError(
                "cannot read the release binary at %s; build it with %s"
                % (self.path, " ".join(BUILD_COMMAND))
            ) from exc
        return digest.hexdigest()

    def version(self) -> str:
        """Ask the binary what version it is.

        Returns:
            The trimmed text the binary printed.

        Raises:
            PreflightError: If the binary cannot be run, exits non-zero, or
                does not answer within :data:`VERSION_TIMEOUT_SECONDS`.
        """
        try:
            completed = subprocess.run(
                [str(self.path), VERSION_FLAG],
                capture_output=True,
                timeout=VERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreflightError(
                "the release binary at %s did not answer %s: %s"
                % (self.path, VERSION_FLAG, exc)
            ) from exc
        if completed.returncode != 0:
            raise PreflightError(
                "the release binary at %s exited %d for %s"
                % (self.path, completed.returncode, VERSION_FLAG)
            )
        return completed.stdout.decode("utf-8", errors="replace").strip()

    def invoke(self, args: list[str], stdin: bytes | None = None,
               env: dict[str, str] | None = None,
               timeout: float | None = None,
               cwd: pathlib.Path | str | None = None) -> ProductOutput:
        """Run the binary once and capture everything it emitted.

        Three rules hold here and each one was a way to get it wrong.

        The prompt travels on **stdin**, never as an argument, so a payload of
        a quarter of a megabyte never meets a command-line length limit.

        The passphrase travels in the **environment**, as ``MAGI_PASSPHRASE``,
        and this method never constructs ``-p``. ``-p`` is a global flag: it
        would ride in the archived command line of every single run, and while
        the child lives any process on the machine can read that command line.

        A non-zero exit is **not** an error here. R6 fails on purpose, and what
        an exit code means belongs to the scenario, through
        :meth:`ProductOutput.require_exit`.

        Args:
            args: The argv after the program name.
            stdin: Bytes to feed the process, or ``None`` for no input.
            env: Variables to overlay on the harness's own environment. The
                overlay is on a COPY of ``os.environ`` rather than a bare
                mapping, because a child started with an empty environment
                loses ``PATH`` -- and, on Windows, ``SystemRoot``, without
                which sockets do not initialise.
            timeout: Seconds to wait, or ``None`` to wait indefinitely.
            cwd: The directory to run the child in, or ``None`` for the
                harness's own. It is a parameter and never a ``chdir`` because
                the harness is one process running every scenario in turn: a
                directory changed for one of them stays changed for the next,
                and the failure that produces surfaces somewhere else entirely.
                S2 and S14 need it because the only way to seed a workspace
                without using the very flag S14 tests is to run ``init`` with
                the target as the process's working directory.

        Returns:
            ProductOutput: The captured streams, exit code, and the argv
            exactly as invoked. The command is stored RAW: R6's authenticated
            ``base_url`` puts a real credential in it, and S10 has to assert on
            what the product emitted rather than on a copy somebody already
            cleaned. Scrubbing happens when the archive is written, and there
            only.

        Raises:
            ProductOutputError: If the process does not finish within
                *timeout*, or cannot be started at all. A run that never
                completed produced no output to interpret, so it is not a
                verdict the scenario can read.
        """
        command = [str(self.path)] + list(args)
        child_env = dict(os.environ)
        if env:
            child_env.update(env)
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                env=child_env,
                cwd=None if cwd is None else str(cwd),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Whatever the child had already written is carried out with the
            # failure rather than dropped. A hang is not silence: a consult
            # that expires may already have emitted the block a scenario reads
            # to tell a degraded ceiling from a slow provider, and re-running
            # to recover it would be a second run of the thing that hung.
            raise TimedOut(
                "the product did not finish within %s seconds" % timeout,
                output=_partial_capture(exc, command),
            ) from exc
        except OSError as exc:
            raise ProductOutputError(
                "could not start the release binary at %s: %s" % (self.path, exc)
            ) from exc
        return ProductOutput(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            command=command,
        )
