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

import datetime
import pathlib

from smoke.errors import HarnessError
from smoke.secrets import scrub

_binary = None
_env = None
_config = None


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


def invoke(argv, stdin=None, *, timeout_s, label, planted=()):
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

    Returns:
        ProductOutput: The capture, unscrubbed.

    Raises:
        HarnessError: If :func:`configure` has not run.
    """
    if _binary is None or _env is None:
        raise HarnessError(
            "runs.invoke was called before runs.configure; the module needs a "
            "binary and an environment before it can reach the product"
        )
    output = _binary.invoke(argv, stdin=stdin, timeout=timeout_s)
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
