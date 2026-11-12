# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
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
import json
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
import urllib.request
from typing import Callable, Optional

from smoke import runs
from smoke.product import ProductOutput

#: What the default responder returns: an invocation that ran and failed. Not
#: an exception, because "the binary refused" and "the harness could not start
#: it" are different states and a scenario handles them differently.
DEFAULT_EXIT_CODE = 1

#: How long a stand-in waits for its loopback server to answer.
LOOPBACK_TIMEOUT_S = 5

#: An opener with no proxy handler: a stand-in's request to a loopback server
#: goes straight to it, whatever ``HTTP_PROXY`` or ``ALL_PROXY`` the host sets.
_LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def loopback_open(request, timeout: float = LOOPBACK_TIMEOUT_S):
    """Send *request* to a loopback server the test runs, never through a proxy.

    MAGI FIX (S6 loop 28): the stand-in doubles used ``urlopen``, which honours
    the host's proxy variables when ``no_proxy`` does not cover 127.0.0.1, so on
    a proxied host their requests never reached the counter and a test that
    asserts nothing was counted passed vacuously (F1b1-melchior).

    Args:
        request: A ``urllib.request.Request`` or a URL.
        timeout: Seconds to wait for the answer.

    Returns:
        http.client.HTTPResponse: The open response; the caller closes it.

    Raises:
        urllib.error.HTTPError: For a non-2xx answer (the counter's 503).
        OSError: When nothing answers.
    """
    return _LOOPBACK_OPENER.open(request, timeout=timeout)


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

    def __init__(self, root: pathlib.Path, create: bool = True) -> None:
        """Bind to *root*, optionally creating what a real environment holds.

        Args:
            root: The environment root.
            create: Whether to create the directories. A test that hands in a
                root of its own may be checking what happens when it does NOT
                exist, and creating it here would answer a different question.
        """
        self.root = pathlib.Path(root)
        self.runs_dir = self.root / "runs"
        # Beside the root, as the real Environment places it: the root carries
        # the product's workspace and ``magi init`` refuses to nest inside one.
        self.scratch_dir = self.root.parent / "scratch"
        if create:
            for directory in (self.root, self.runs_dir, self.scratch_dir):
                directory.mkdir(parents=True, exist_ok=True)

    def prepare_scratch(self) -> None:
        """Create the scratch area, as ``Environment`` does.

        The double carries it because :func:`smoke.runs.scratch_root` calls it,
        and a double missing a method the code under test uses would fail here
        for a reason no scenario has.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)


#: What the double declares as the large payload's size. Small on purpose: the
#: builder reads the repository's own sources, and a test that measures whether
#: the payload was attached does not need a quarter of a megabyte to see it.
FAKE_PAYLOAD_TARGET_BYTES = 2000


class FakeConfig:
    """The fields :mod:`smoke.runs` reads off a ``SmokeConfig``."""

    def __init__(self, passphrase: str,
                 payload_target_bytes: int = FAKE_PAYLOAD_TARGET_BYTES,
                 backend_key_env: str = "SMOKE_FAKE_KEY") -> None:
        """Store what the module asks for.

        Args:
            passphrase: What ``runs.passphrase()`` should answer.
            payload_target_bytes: How large a run marked large should be. The
                double carried only the passphrase until the executor started
                measuring what it sends, at which point every large run through
                a double raised ``AttributeError`` -- an under-specified double,
                not a defect in what it stands for.
            backend_key_env: The variable the rotating run puts back. Same
                story: the double grew it when a test first drove R7 through
                the executor.
        """
        self.passphrase = passphrase
        self.payload_target_bytes = payload_target_bytes
        self.backend_key_env = backend_key_env


#: The passphrase the doubles hand out. Long enough to clear the product's own
#: floor, so a scenario that forwards it is not rejected before it starts.
FAKE_PASSPHRASE = "correct-horse-battery-staple"


def vault_succeeds(call: "Call") -> ProductOutput | None:
    """A responder that accepts the executor's vault bookkeeping and nothing else.

    The default response is a FAILED invocation, deliberately, so a scenario
    tested against a bare double cannot reach PASS while asserting nothing.
    That default is right for the scenario and wrong for the executor's own
    plumbing: R6 plants placeholder entries and R7 rotates a credential, both
    through ``vault set``, and a refusal there now stops the run before it
    starts -- which is the point of that check, but it is not what a test
    about resolved URLs is asking.

    Args:
        call: What the fake binary was asked to run.

    Returns:
        ProductOutput: Exit 0 for a vault subcommand; None for anything else,
        which leaves the failed default in place for the run itself.
    """
    if list(call.args[:1]) != ["vault"]:
        return None
    return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                         command=["magi-rs"] + list(call.args))


def scratch_dir(case: unittest.TestCase) -> pathlib.Path:
    """A temporary directory that goes away when the test does.

    ``tempfile.mkdtemp()`` on its own does not. The suite had forty-three of
    them and six cleanups, so every full run left thirty-seven directories
    behind in the system temp area. One helper rather than thirty-seven
    ``addCleanup`` lines, because the version that is easy to call is the one
    that gets called.

    Args:
        case: The test to register the removal on.

    Returns:
        pathlib.Path: The directory, already created.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="smoke-test-"))
    case.addCleanup(_remove_quietly, root)
    return root


def _remove_quietly(root: pathlib.Path) -> None:
    """Remove *root*, saying so on stderr if it could not be removed.

    ``ignore_errors=True`` would leave the directory behind in silence, which
    is the leak this helper exists to stop, made invisible. On Windows an open
    handle is enough to hit it. The harness's own rule is that what could not
    be done gets reported, so this reports rather than raising: a test that
    passed did not fail because of a directory.

    Args:
        root: The directory to remove.
    """
    try:
        shutil.rmtree(root)
    except OSError as exc:
        print("[tests] could not remove %s: %s" % (root, exc),
              file=sys.stderr)


def install_fake_runs(case: unittest.TestCase,
                      responder: Responder | None = None,
                      repo_root: pathlib.Path | None = None,
                      env_root: pathlib.Path | None = None) -> FakeBinary:
    """Point :mod:`smoke.runs` at doubles for the duration of one test.

    Cleanup is registered rather than left to ``tearDown``: a test that fails
    half way through still has to return the module to its unconfigured state,
    or every test after it inherits this one's fakes.

    Args:
        case: The test to register the cleanup on.
        responder: How the fake binary should answer; see :class:`FakeBinary`.
        repo_root: The repository the fake claims; defaults to the real one, so
            a scenario that shells out to git finds a repository.
        env_root: Where the fake environment lives; defaults to a temporary
            directory that is removed afterwards. S12 asks git what it can see
            of the environment, and git can only answer about a path inside a
            repository, so that test passes one -- and keeps ownership of it,
            which is why a caller-supplied root is never removed here.

    Returns:
        FakeBinary: The double, so the test can read ``calls``.
    """
    if env_root is None:
        root = pathlib.Path(tempfile.mkdtemp(prefix="smoke-fake-env-"))
        case.addCleanup(shutil.rmtree, root, ignore_errors=True)
    else:
        root = pathlib.Path(env_root)
    case.addCleanup(runs.reset_for_test)
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    binary = FakeBinary(repo_root, responder)
    runs.reset_for_test()
    runs.configure(binary, FakeEnvironment(root, create=env_root is None),
                   FakeConfig(FAKE_PASSPHRASE))
    return binary

#: The ``.magi/`` files git lists in R16's workspace when the environment's
#: ``.magi`` is copied in. [MAGI §6 MS2 loop 9, F1a-c1] S51 reads them off the result.
R16_PROTECTED = (".magi/.magi-rs-memory.db", ".magi/logs/magi-2026-10-29.log", ".magi/magi.toml")

#: A healthy R18's relay record (MAGI S6 MS3 loop9): chunk, masked query, trio request.
R18_HEALTHY_REQUESTS = tuple(
    ("/v1/embeddings", json.dumps({"input": [text]}).encode())
    for text in ("search_document: " + runs.REVIEW_MASKED_DOC.decode(),
                 "search_query: " + runs.REVIEW_MASKED_DOC.decode() + "\npub fn token() <redacted: anthropic-key>")
) + (("/api/chat", json.dumps({"messages": [{"content": "pub fn token() -> &'static str {\n    \"<redacted: anthropic-key>\"\n}"}]}).encode()),)


def scenario_result(run_id: str, exit_code: int = 0, document=None, stderr: bytes = b"",
                    timed_out: bool = False, protected_visible=None, requests=None):
    """A real ``RunResult`` for *run_id* whose stdout is *document* serialised.

    [MAGI S6 MS3 loop7, R7b Balthasar] Moved here from the code-review scenario
    tests, so the masking tests depend on the harness's shared doubles, not on
    another test module.

    Args:
        run_id: The run the result stands for.
        exit_code: The product's exit code.
        document: What stdout carries, JSON-serialised; ``None`` for empty stdout.
        stderr: The product's stderr.
        timed_out: Whether the run was cut by its wall clock.
        protected_visible: The protected paths git listed; defaults to
            :data:`R16_PROTECTED` for R16 and to nothing for every other run.
        requests: What the recording relay saw, ``(path, body)`` pairs; defaults to
            :data:`R18_HEALTHY_REQUESTS` for R18 and to nothing recorded (``()``) otherwise.

    Returns:
        RunResult: The result.
    """
    from smoke.runs import RunResult
    body = b"" if document is None else json.dumps(document).encode()
    if protected_visible is None:
        protected_visible = R16_PROTECTED if run_id == "R16" else ()
    if requests is None:
        requests = R18_HEALTHY_REQUESTS if run_id == "R18" else ()
    return RunResult(run_id=run_id,
                     output=ProductOutput(stdout=body, stderr=stderr, exit_code=exit_code,
                                          command=["magi-rs"]),
                     duration_s=1.0, timed_out=timed_out, planted=(),
                     protected_visible=protected_visible, request_bodies=requests,
                     # MAGI FIX (MS3 §6 loop13): one header record per request, as the relay keeps.
                     request_headers=tuple(b"content-type: application/json\n" for _ in requests))


def scenario_findings(scenario_id: str, result, log=b"") -> list:
    """The findings *scenario_id* yields over *result*, with *log* as its run log.

    [MAGI §6 MS2 loop 3, F4] the log rides on the ``RunResult`` the executor
    captured right after the run; the scratch area handed to ``runs`` stays
    empty, so a scenario that read the live directory would find nothing.

    Args:
        scenario_id: The registered scenario.
        result: The ``RunResult`` it reads.
        log: The run log captured on the result (``None``: no directory).

    Returns:
        list: The findings, in order.
    """
    from smoke.registry import DEFAULT_REGISTRY
    result = dataclasses.replace(result, review_log=None if log is None else (log,))
    with tempfile.TemporaryDirectory() as tmp:
        scratch = pathlib.Path(tmp)
        runs.configure(None, types.SimpleNamespace(scratch_dir=scratch,
                                                   prepare_scratch=lambda: None), None)
        try:
            return list(DEFAULT_REGISTRY.get(scenario_id).func(result))
        finally:
            runs.reset_for_test()


def scenario_outcomes(scenario_id: str, result, log=b"") -> dict:
    """Outcomes of *scenario_id*, keyed by assertion text (see :func:`scenario_findings`).

    Args:
        scenario_id: The registered scenario.
        result: The ``RunResult`` it reads.
        log: The run log captured on the result.

    Returns:
        dict: Assertion text to outcome.
    """
    return {f.assertion: f.outcome for f in scenario_findings(scenario_id, result, log)}
