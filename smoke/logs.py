# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-08-31
"""The one place that locates and searches the product's on-disk log.

``.magi/logs`` is the workspace's day-file directory the logging layer owns
(REQ-L63, REQ-L19) -- not a harness artifact. Three scenarios each need to
answer some form of "does a line matching X live under there", and before
this module existed each carried its own copy of the directory path and its
own file walk: S23 and S24 in ``smoke/scenarios/``, and S3 in
``smoke/scenarios/vault.py``. A path computed three times is a path that
drifts the day one copy is renamed and the other two are not, so this is the
only module that computes it and the only one that walks it for a needle.

S3 keeps its own before/after byte-diff (:func:`~smoke.scenarios.vault._log_snapshot`
and friends) rather than routing through :func:`scan` here -- it is measuring
NEW bytes since a snapshot, a different question from "does a marker appear
anywhere", and forcing the two into one function would blur what each is
answering. It still calls :func:`log_directory` for the path, which is the
part that must not drift.

Every function here degrades to "not found" rather than raising, on a
directory that does not exist or a file that cannot be read. Telling a
missing directory apart from an existing one searched clean is the caller's
job: only the caller knows whether an empty result is expected or is itself
the finding.
"""

import pathlib

from smoke import runs
from smoke.product import ProductOutputError

#: Where the product writes its logs, relative to the workspace root. The
#: default did not move when the JSONL format did.
LOG_DIR_PARTS = (".magi", "logs")

#: The prefix REQ-L63 puts on stderr, ahead of the run id.
STDERR_RUN_PREFIX = "run: "


def log_directory() -> pathlib.Path:
    """The workspace's log directory.

    Args:
        None.

    Returns:
        pathlib.Path: ``.magi/logs`` under the persistent environment's
        root. May not exist; callers check with ``is_dir()``.
    """
    return runs.workspace_root().joinpath(*LOG_DIR_PARTS)


def scan(matcher):
    """Walk every file under the log directory, in sorted order.

    Complexity: ``O(bytes in the log directory)``.

    Args:
        matcher: Called with each file's raw bytes; returns a non-``None``
            result on a match, or ``None`` to keep looking.

    Returns:
        tuple[object | None, bool, list[str]]: ``(result, directory
        existed, unreadable file names)``. ``result`` is *matcher*'s first
        non-``None`` return, or ``None`` when nothing matched anywhere. The
        directory flag is kept apart from the result because a missing
        directory (nothing written to disk at all) and an existing one
        searched clean (whatever was sought was never produced) are
        different findings.
    """
    directory = log_directory()
    if not directory.is_dir():
        return None, False, []
    unreadable = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            unreadable.append("%s (%s)" % (path.name, exc))
            continue
        result = matcher(data)
        if result is not None:
            return result, True, unreadable
    return None, True, unreadable


def contains(marker):
    """Whether *marker* appears in some file under the log directory.

    Args:
        marker: Text to search for, encoded as UTF-8 before matching.

    Returns:
        tuple[bool, bool, list[str]]: ``(found, directory existed,
        unreadable file names)`` -- the same shape :func:`scan` returns,
        with the result coerced to a plain bool.
    """
    needle = marker.encode("utf-8")
    found, dir_existed, unreadable = scan(
        lambda data: True if needle in data else None)
    return bool(found), dir_existed, unreadable


def id_on_stderr(stderr):
    """Extract the run id from the ``run: <id>`` line REQ-L63 puts on stderr.

    Args:
        stderr: The run's stderr, unscrubbed.

    Returns:
        str | None: The id, or None when no such line was emitted.

    Example:
        >>> id_on_stderr(b"warming up\\nrun: 4242-deadbeef\\n")
        '4242-deadbeef'
        >>> id_on_stderr(b"nothing here\\n") is None
        True
    """
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith(STDERR_RUN_PREFIX):
            candidate = stripped[len(STDERR_RUN_PREFIX):].strip()
            if candidate:
                return candidate
    return None


def resolve_id(output):
    """The run's id, from whichever surface published one.

    The precedence is stderr first, then the envelope, and it lives here so
    it lives in ONE place. Three scenarios need the same answer -- S9 and S24
    to bind a log search to their own run, S23 to search for the id it agreed
    on -- and each carried its own copy of the same two-line ``or``. Two
    copies of a precedence rule is two chances for one to be edited and the
    others not, and nothing about the resulting disagreement raises an error:
    the scenarios simply start binding to different runs.

    Telling "no id anywhere" from "an id on one surface only" is the
    CALLER's job, which is why this returns a single value rather than the
    pair: S23's first assertion is precisely that the two surfaces agree, so
    it reads both itself and uses this only for the search that follows.

    Args:
        output: The run's ``ProductOutput``.

    Returns:
        str | None: The id, or None when neither surface published one.

    Example:
        >>> from smoke.product import ProductOutput
        >>> both = ProductOutput(stdout=b'{"run_id": "from-envelope"}',
        ...                      stderr=b"run: from-stderr\\n", exit_code=0,
        ...                      command=["magi-rs"])
        >>> resolve_id(both)
        'from-stderr'
        >>> quiet = ProductOutput(stdout=b'{"run_id": "from-envelope"}',
        ...                       stderr=b"", exit_code=0,
        ...                       command=["magi-rs"])
        >>> resolve_id(quiet)
        'from-envelope'
    """
    return id_on_stderr(output.stderr) or id_in_envelope(output)


def id_in_envelope(output):
    """Read ``run_id`` out of the JSON envelope.

    Args:
        output: The run's ``ProductOutput``.

    Returns:
        str | None: The id, or None when the key is absent or not a string.
    """
    try:
        value = output.key("run_id")
    except ProductOutputError:
        return None
    return value if isinstance(value, str) and value else None
