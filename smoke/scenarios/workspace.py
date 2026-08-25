# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S2 -- the workspace is created, not clobbered, and discovered.

Protects ``system/workspace.rs``: ``magi init`` scaffolds a restricted
``.magi/``, refuses to touch one that already exists, and the walk-up finds the
nearest ancestor from a nested directory.

Everything here runs under ``env/scratch/``, the declared exception to REQ-S30.
``init`` can only be exercised against a directory that does NOT yet carry
``.magi/``, and the persistent environment does.

The seed never uses ``-w``. That flag is what S14 tests, and a precondition
built on the mechanism under test cannot fail: with the flag ignored the seed
would land in the current directory, the assertion would resolve that same
directory, find a workspace there and pass.
"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8. They reach the
#: certificate unchanged, so they are written once and only here.
ASSERTIONS = (
    "magi init creates .magi/ in an empty directory",
    "the permissions are restrictive — POSIX bits on Unix, ACL on Windows",
    "a second init refuses and leaves the directory unchanged",
    "query from a nested subdirectory finds the ancestor .magi/",
)

#: The product's subcommands this scenario drives.
INIT_SUBCOMMAND = "init"
QUERY_SUBCOMMAND = "query"

#: What the product scaffolds, and the one file inside it whose permissions say
#: whether the restriction reached the contents as well as the directory.
MAGI_DIR_NAME = ".magi"
MAGI_TOML_NAME = "magi.toml"

#: How long the harness waits for one invocation, in seconds. ``init`` writes a
#: handful of small files; ``query`` opens the encrypted database, which costs
#: an Argon2 derivation, and then tries to reach a backend that is not required
#: to be there.
INIT_TIMEOUT_S = 120
QUERY_TIMEOUT_S = 180

#: The wall clock handed to the PRODUCT for the discovery probe. The scenario
#: does not care whether the model answers -- only where the product decided
#: its workspace was -- so it is bounded well below the harness's own timeout,
#: which is what keeps a missing backend cheap instead of slow.
PRODUCT_TIMEOUT_S = 20

#: What a POSIX ``.magi/`` and its configuration must be, and nothing wider.
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
MODE_MASK = 0o777

#: The tool that reads a Windows DACL, and the accounts whose presence means
#: inheritance survived. Every one of them is an account REQ-H38 excludes.
ICACLS = "icacls"
ICACLS_TIMEOUT_S = 60
BROAD_PRINCIPALS = (
    "everyone",
    "builtin\\users",
    "builtin\\administrators",
    "nt authority\\system",
    "nt authority\\authenticated users",
    "authenticated users",
    "users",
)

#: How the product refuses when the walk-up found nothing. Its presence is the
#: difference between "the product looked and did not find" -- which is the
#: defect this assertion hunts -- and "the product never got that far".
NO_WORKSPACE_MARKER = b"no .magi/ state directory found"

#: The prompt the discovery probe feeds the product. Its content is irrelevant
#: on purpose: no assertion in this harness may depend on what a model answers.
DISCOVERY_PROMPT = b"say ok\n"

WINDOWS_PLATFORM = "win32"


@scenario("S2")
def the_workspace_is_created_not_clobbered_and_discovered(run):
    """Exercise ``magi init`` and the walk-up on a virgin tree.

    Args:
        run: Always ``None``; S2 declares no shared run and reaches the product
            through :func:`smoke.runs.attempt`.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order, whatever
        the product did. A scenario that stops early reads as silence, and the
        runner calls silence a harness failure.
    """
    root = pathlib.Path(
        tempfile.mkdtemp(prefix="s2-", dir=runs.scratch_root())
    )
    magi_dir = root / MAGI_DIR_NAME
    seed = runs.attempt([INIT_SUBCOMMAND], stdin=b"", timeout_s=INIT_TIMEOUT_S,
                        label="s2-init", cwd=root,
                        env={"MAGI_PASSPHRASE": runs.passphrase()})
    yield _creation_finding(seed, magi_dir)
    yield _permission_finding(magi_dir)
    yield _refusal_finding(root, magi_dir)
    yield _discovery_finding(root, magi_dir)


def _creation_finding(seed, magi_dir):
    """Judge assertion 1 from the seeding invocation.

    Args:
        seed: The :class:`~smoke.runs.Attempt` that ran ``init``.
        magi_dir: Where the workspace should now be.

    Returns:
        Finding: PASS when the directory exists, FAIL when the product exited
        non-zero or wrote nothing, CANNOT_TEST when the invocation never
        completed -- which is not a verdict, because nothing was observed.
    """
    if not seed.ok:
        return _finding(0, Outcome.CANNOT_TEST, seed.failure)
    if magi_dir.is_dir():
        return _finding(0, Outcome.PASS, "")
    return _finding(
        0, Outcome.FAIL,
        "init exited %d and left no %s at %s: %s"
        % (seed.output.exit_code, MAGI_DIR_NAME, magi_dir,
           _excerpt(seed.output)),
    )


def _permission_finding(magi_dir):
    """Judge assertion 2, written differently on each platform.

    Checking POSIX mode bits on Windows would be an empty green -- the bits are
    synthesised there and say nothing about who can read the file -- so the two
    platforms get two different checks behind one assertion text.

    Args:
        magi_dir: The scaffolded workspace.

    Returns:
        Finding: The platform's verdict, or CANNOT_TEST when the workspace was
        never created and there is nothing whose permissions could be read.
    """
    if not magi_dir.is_dir():
        return _finding(1, Outcome.CANNOT_TEST,
                        "no %s was created to inspect" % MAGI_DIR_NAME)
    if sys.platform == WINDOWS_PLATFORM:
        return _windows_permission_finding(magi_dir)
    return _posix_permission_finding(magi_dir)


def _posix_permission_finding(magi_dir):
    """Check the POSIX mode bits of the workspace and its configuration.

    Args:
        magi_dir: The scaffolded workspace.

    Returns:
        Finding: PASS when the directory is ``0700`` and the configuration
        ``0600``; FAIL naming every path that is wider; CANNOT_TEST when a path
        could not be stat'ed.
    """
    expected = {magi_dir: DIRECTORY_MODE, magi_dir / MAGI_TOML_NAME: FILE_MODE}
    wider = []
    for path, mode in expected.items():
        try:
            found = path.stat().st_mode & MODE_MASK
        except OSError as exc:
            return _finding(1, Outcome.CANNOT_TEST,
                            "cannot read the mode of %s: %s" % (path, exc))
        if found != mode:
            wider.append("%s is %04o, expected %04o" % (path, found, mode))
    if wider:
        return _finding(1, Outcome.FAIL, "; ".join(wider))
    return _finding(1, Outcome.PASS, "")


def _windows_permission_finding(magi_dir):
    """Check that the workspace's DACL names one account and no broad one.

    The product grants the current user full control and then removes every
    other entry, so an inherited ``BUILTIN\\Users`` or ``NT AUTHORITY\\SYSTEM``
    surviving is exactly the regression REQ-H38 forbids. The check counts the
    entries as well as naming them: a second allow entry for some other
    individual account would widen access just as much and carries no
    well-known name to match against.

    Args:
        magi_dir: The scaffolded workspace.

    Returns:
        Finding: PASS for a single non-broad entry, FAIL otherwise,
        CANNOT_TEST when ``icacls`` could not be run or reported nothing.
    """
    try:
        completed = subprocess.run([ICACLS, str(magi_dir)], capture_output=True,
                                   timeout=ICACLS_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _finding(1, Outcome.CANNOT_TEST,
                        "could not run %s: %s" % (ICACLS, exc))
    if completed.returncode != 0:
        return _finding(1, Outcome.CANNOT_TEST,
                        "%s exited %d" % (ICACLS, completed.returncode))
    principals = _icacls_principals(
        completed.stdout.decode("utf-8", errors="replace"), magi_dir
    )
    if not principals:
        return _finding(1, Outcome.CANNOT_TEST,
                        "%s listed no access entry for %s" % (ICACLS, magi_dir))
    broad = [name for name in principals if name.lower() in BROAD_PRINCIPALS]
    if broad:
        return _finding(1, Outcome.FAIL,
                        "the DACL still grants %s" % ", ".join(broad))
    if len(principals) != 1:
        return _finding(
            1, Outcome.FAIL,
            "the DACL names %d accounts (%s); the restriction leaves exactly one"
            % (len(principals), ", ".join(principals)),
        )
    return _finding(1, Outcome.PASS, "")


def _icacls_principals(text, magi_dir):
    """Extract the account of each access entry ``icacls`` printed.

    ``icacls`` writes the path once, then one ``PRINCIPAL:(flags)`` entry per
    line with the first entry sharing the path's line. Splitting on the LAST
    colon before the parenthesised flags is what survives an account name that
    is itself a path-like ``DOMAIN\\user`` and a drive letter in the path.

    Complexity: ``O(len(text))``.

    Args:
        text: What ``icacls`` printed.
        magi_dir: The path it was asked about, stripped from the first line.

    Returns:
        list[str]: One account name per access entry, in the order printed.
    """
    principals = []
    for line in text.splitlines():
        entry = line.strip()
        if entry.startswith(str(magi_dir)):
            entry = entry[len(str(magi_dir)):].strip()
        marker = entry.find(":(")
        if not entry or marker <= 0:
            continue
        principals.append(entry[:marker].strip())
    return principals


def _refusal_finding(root, magi_dir):
    """Judge assertion 3: a second ``init`` refuses AND changes nothing.

    "Refuses" alone is half the assertion. The half that matters is that the
    tree is byte-for-byte what it was, because the failure this guards against
    is an ``init`` that takes over a workspace holding the user's data.

    Args:
        root: The directory holding the workspace.
        magi_dir: The workspace itself.

    Returns:
        Finding: PASS on a non-zero exit with an unchanged tree; FAIL when it
        succeeded or when anything under *root* moved; CANNOT_TEST when there
        was no workspace to defend or the invocation never completed.
    """
    if not magi_dir.is_dir():
        return _finding(2, Outcome.CANNOT_TEST,
                        "no %s was created for a second init to refuse"
                        % MAGI_DIR_NAME)
    before = _snapshot(root)
    second = runs.attempt([INIT_SUBCOMMAND], stdin=b"",
                          timeout_s=INIT_TIMEOUT_S, label="s2-init-again",
                          cwd=root, env={"MAGI_PASSPHRASE": runs.passphrase()})
    if not second.ok:
        return _finding(2, Outcome.CANNOT_TEST, second.failure)
    after = _snapshot(root)
    if second.output.exit_code == 0:
        return _finding(2, Outcome.FAIL,
                        "the second init exited 0 over an existing %s"
                        % MAGI_DIR_NAME)
    changed = _changed_paths(before, after)
    if changed:
        return _finding(
            2, Outcome.FAIL,
            "the second init refused but changed %s" % ", ".join(changed),
        )
    return _finding(2, Outcome.PASS, "")


def _discovery_finding(root, magi_dir):
    """Judge assertion 4 by where the product WROTE, not by what it answered.

    A ``query`` from a nested directory needs a backend to answer, and this
    scenario runs with none. What it does not need a backend for is resolving
    its workspace and opening the database there, and that leaves a trace: the
    contents of the resolved ``.magi/`` move. So the assertion reads the
    filesystem, which the product controls, instead of the reply, which it does
    not.

    Args:
        root: The directory holding the workspace.
        magi_dir: The workspace the walk-up should find.

    Returns:
        Finding: PASS when the ancestor workspace changed; FAIL when the
        product said it found no workspace; CANNOT_TEST when it neither wrote
        nor said so, since that leaves the question unanswered.
    """
    if not magi_dir.is_dir():
        return _finding(3, Outcome.CANNOT_TEST,
                        "no ancestor %s exists to be discovered"
                        % MAGI_DIR_NAME)
    nested = root / "nested" / "deeper"
    try:
        nested.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _finding(3, Outcome.CANNOT_TEST,
                        "cannot create a nested directory under %s: %s"
                        % (root, exc))
    before = _snapshot(magi_dir)
    probe = runs.attempt(
        [QUERY_SUBCOMMAND, "--output-format", "json",
         "--timeout", str(PRODUCT_TIMEOUT_S)],
        stdin=DISCOVERY_PROMPT, timeout_s=QUERY_TIMEOUT_S,
        label="s2-discover", cwd=nested,
        env={"MAGI_PASSPHRASE": runs.passphrase()},
    )
    if not probe.ok:
        return _finding(3, Outcome.CANNOT_TEST, probe.failure)
    if _changed_paths(before, _snapshot(magi_dir)):
        return _finding(3, Outcome.PASS, "")
    if NO_WORKSPACE_MARKER in probe.output.raw():
        return _finding(
            3, Outcome.FAIL,
            "the walk-up from %s reported no workspace although %s exists"
            % (nested, magi_dir),
        )
    return _finding(
        3, Outcome.CANNOT_TEST,
        "the query exited %d without touching %s and without reporting a "
        "missing workspace, so where it resolved is unknown: %s"
        % (probe.output.exit_code, magi_dir, _excerpt(probe.output)),
    )


def _snapshot(root):
    """Hash every file under *root*, keyed by its path relative to it.

    A mapping rather than one digest of the whole tree, so a change can be
    NAMED. A diff nobody can read is a red nobody acts on.

    Complexity: ``O(total bytes under root)``.

    Args:
        root: The directory to walk.

    Returns:
        dict[str, str]: Relative POSIX path -> sha256 of the file's bytes. A
        file that vanishes between the walk and the read is recorded as
        unreadable rather than dropped, because dropping it would make a
        deletion look like nothing at all.
    """
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(root).as_posix()
        try:
            entries[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            entries[key] = "unreadable: %s" % exc
    return entries


def _changed_paths(before, after):
    """Name every path that appeared, vanished or moved between two snapshots.

    Complexity: ``O(len(before) + len(after))``.

    Args:
        before: The earlier snapshot.
        after: The later one.

    Returns:
        list[str]: The changed paths, sorted, each tagged with what happened.
    """
    changed = []
    for key in sorted(set(before) | set(after)):
        if key not in after:
            changed.append("%s (removed)" % key)
        elif key not in before:
            changed.append("%s (added)" % key)
        elif before[key] != after[key]:
            changed.append("%s (modified)" % key)
    return changed


def _excerpt(output, limit=400):
    """Render the beginning of a capture for a finding's detail.

    Args:
        output: The capture to quote.
        limit: How many bytes to keep. A detail line that carries a whole
            failed run is a detail line nobody reads.

    Returns:
        str: The first *limit* bytes of both streams, decoded leniently.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _finding(index, outcome, detail):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Indexing the tuple rather than repeating the text is what keeps the
    certificate's wording and the scenario's wording the same string: a
    hand-copied assertion is a report that credits the wrong claim.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S2 is standalone.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=None)
