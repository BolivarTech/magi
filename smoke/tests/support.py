# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Doubles the scenario tests share, so no two of them build their own.

A scenario reaches the product only through :mod:`smoke.runs`, which is exactly
what makes it testable without a backend: configure that module with a fake
binary and the scenario runs end to end in milliseconds. The doubles live here
rather than in one test module because five test modules need them, and a copy
per module is five places for the fake's behaviour to drift from the real
``ReleaseBinary``.

The default response is a FAILED invocation on purpose. A double that answers
success by default would let a scenario reach ``PASS`` while asserting nothing,
which is the shape of guardian this whole harness is written against; a
scenario tested against the default should report FAIL or CANNOT_TEST for every
assertion, and it must still report ALL of them.
"""

import dataclasses
import pathlib
import shutil
import tempfile
import unittest
from typing import Callable, Optional

from smoke import runs
from smoke.product import ProductOutput

#: What the default responder returns: an invocation that ran and failed. Not
#: an exception, because "the binary refused" and "the harness could not start
#: it" are different states and a scenario handles them differently.
DEFAULT_EXIT_CODE = 1


@dataclasses.dataclass(frozen=True)
class Call:
    """One invocation the fake binary received.

    Attributes:
        args: The argv after the program name.
        stdin: The bytes handed to the child, or None.
        env: The environment overlay, or None.
        cwd: The working directory the child was to run in, or None.
        timeout: The wall clock the invocation was given.
    """

    args: tuple[str, ...]
    stdin: bytes | None
    env: dict[str, str] | None
    cwd: object
    timeout: float | None


Responder = Callable[[Call], Optional[ProductOutput]]


class FakeBinary:
    """A stand-in for :class:`~smoke.binary.ReleaseBinary` that starts nothing.

    Example:
        >>> binary = FakeBinary(pathlib.Path("."))
        >>> binary.invoke(["--version"]).exit_code
        1
    """

    def __init__(self, repo_root: pathlib.Path,
                 responder: Responder | None = None) -> None:
        """Create the double.

        Args:
            repo_root: What :meth:`repo_root` should answer. Scenarios that
                shell out to git need a real repository here.
            responder: Called with each :class:`Call`; returns the capture to
                answer with, or None to fall back to the failed default.
        """
        self._repo_root = pathlib.Path(repo_root)
        self._responder = responder
        self.calls: list[Call] = []

    @property
    def repo_root(self) -> pathlib.Path:
        """The checkout this fake claims to belong to.

        Returns:
            The root handed to the constructor.
        """
        return self._repo_root

    @property
    def path(self) -> pathlib.Path:
        """Where the fake pretends the binary lives.

        Returns:
            A path under the fake's repository root. Nothing reads the file.
        """
        return self._repo_root / "target" / "release" / "magi-rs"

    def invoke(self, args: list[str], stdin: bytes | None = None,
               env: dict[str, str] | None = None,
               timeout: float | None = None,
               cwd: object = None) -> ProductOutput:
        """Record the call and answer it.

        The signature mirrors ``ReleaseBinary.invoke`` exactly, including the
        parameter names, so a test that passes an argument the real binary does
        not accept fails here rather than passing against a laxer double.

        Args:
            args: The argv after the program name.
            stdin: Bytes for the child, or None.
            env: Overlay on the harness's environment, or None.
            timeout: Seconds the invocation may take.
            cwd: The directory to run in, or None.

        Returns:
            ProductOutput: Whatever the responder chose, or a failed capture.
        """
        call = Call(args=tuple(args), stdin=stdin, env=env, cwd=cwd,
                    timeout=timeout)
        self.calls.append(call)
        answer = self._responder(call) if self._responder else None
        if answer is not None:
            return answer
        return ProductOutput(stdout=b"", stderr=b"", exit_code=DEFAULT_EXIT_CODE,
                             command=[str(self.path)] + list(args))


class FakeEnvironment:
    """The three directories :mod:`smoke.runs` reads off an ``Environment``."""

    def __init__(self, root: pathlib.Path) -> None:
        """Create the directories under *root*.

        Args:
            root: The environment root, created if absent.
        """
        self.root = pathlib.Path(root)
        self.runs_dir = self.root / "runs"
        self.scratch_dir = self.root / "scratch"
        for directory in (self.root, self.runs_dir, self.scratch_dir):
            directory.mkdir(parents=True, exist_ok=True)


class FakeConfig:
    """The one field :mod:`smoke.runs` reads off a ``SmokeConfig``."""

    def __init__(self, passphrase: str) -> None:
        """Store the passphrase.

        Args:
            passphrase: What ``runs.passphrase()`` should answer.
        """
        self.passphrase = passphrase


#: The passphrase the doubles hand out. Long enough to clear the product's own
#: floor, so a scenario that forwards it is not rejected before it starts.
FAKE_PASSPHRASE = "correct-horse-battery-staple"


def install_fake_runs(case: unittest.TestCase,
                      responder: Responder | None = None,
                      repo_root: pathlib.Path | None = None) -> FakeBinary:
    """Point :mod:`smoke.runs` at doubles for the duration of one test.

    Cleanup is registered rather than left to ``tearDown``: a test that fails
    half way through still has to return the module to its unconfigured state,
    or every test after it inherits this one's fakes.

    Args:
        case: The test to register the cleanup on.
        responder: How the fake binary should answer; see :class:`FakeBinary`.
        repo_root: The repository the fake claims; defaults to the real one, so
            a scenario that shells out to git finds a repository.

    Returns:
        FakeBinary: The double, so the test can read ``calls``.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="smoke-fake-env-"))
    case.addCleanup(shutil.rmtree, root, ignore_errors=True)
    case.addCleanup(runs.reset_for_test)
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    binary = FakeBinary(repo_root, responder)
    runs.reset_for_test()
    runs.configure(binary, FakeEnvironment(root), FakeConfig(FAKE_PASSPHRASE))
    return binary
