# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S3 -- the vault stores and never reveals.

Protects REQ-V09: there is no ``get``/``cat``/``show``/``reveal``/``export``
subcommand, and the stored value never reaches any output.

Two things here are done the hard way on purpose.

The leak check **searches the capture** for the planted value instead of
reading the product's source, and it searches every derived form
(:func:`smoke.secrets.find_secret`) rather than the raw bytes alone -- a
raw-only search is the hole section 2.5 exists to close, because a value that
travels percent-encoded shares no substring with the value that was planted.

Assertion 4 is verified against ``--help``, the parser's own output. "I am not
aware of a subcommand that prints a value" is not evidence; a Commands list is.
"""

import re

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.secrets import find_secret, mint_credential

#: The verbatim assertion texts of the spec's section 8, for S3.
S3_ASSERTIONS = (
    "vault set accepts the value from stdin and vault ls lists its name",
    "vault ls prints name and timestamps, never the value",
    "the planted value appears nowhere in stdout, stderr or the run log",
    "no subcommand exists that prints a stored value",
    "vault rm removes it and ls no longer lists it",
)

#: The entry this scenario plants. A FIXED name rather than a random one: a run
#: killed between ``set`` and ``rm`` leaves it behind, and a fixed name is one
#: the next run overwrites and then removes instead of accumulating a new
#: orphan per crash.
PROBE_NAME = "SMOKE_S3_PROBE"

VAULT_SUBCOMMAND = "vault"
WORKDIR_FLAG = "-w"
FORCE_FLAG = "--force"
HELP_FLAG = "--help"

#: How long one vault invocation may take, in seconds. Each opens the database,
#: which costs an Argon2id derivation at OWASP 2025 cost, so the bound is well
#: above what the work takes: it is here to end a hang, not to police a machine
#: under load.
VAULT_TIMEOUT_S = 180

#: What ``ls`` puts between a name and its two timestamps.
LS_SEPARATOR = " · "
LS_FIELDS = 3

#: Where the product writes its run log, relative to the workspace. A leak into
#: a file nobody reads is still a leak, and the archived harness copy is
#: scrubbed, so the product's own log is the only unscrubbed artifact left.
LOG_DIR_PARTS = (".magi", "logs")

#: Subcommand names that would print a stored value. Matched against the
#: ``Commands:`` block of ``--help`` only, so a FLAG spelled the same way --
#: ``set --show`` echoes what is being typed now, never what is stored -- is
#: not mistaken for one.
REVEALING_SUBCOMMANDS = ("get", "cat", "show", "reveal", "export", "print",
                         "read", "dump")

_COMMANDS_HEADING = re.compile(r"^\s*(commands|subcommands):\s*$",
                               re.IGNORECASE)
_DIGIT = re.compile(r"\d")


@scenario("S3")
def the_vault_stores_and_never_reveals(run):
    """Plant a credential in the persistent vault, then take it back out.

    Args:
        run: Always ``None``; S3 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S3_ASSERTIONS`, in that order.
    """
    secret = mint_credential()
    planted = (secret,)
    logs_before = _log_snapshot()

    stored = _vault(["set", PROBE_NAME, FORCE_FLAG],
                    stdin=(secret.value + "\n").encode("utf-8"),
                    label="s3-set", planted=planted)
    listed = _vault(["ls"], label="s3-ls", planted=planted)
    helped = _vault([HELP_FLAG], label="s3-help")
    removed = _vault(["rm", PROBE_NAME, FORCE_FLAG], label="s3-rm")
    relisted = _vault(["ls"], label="s3-ls-after")

    yield _stored_finding(stored, listed)
    yield _listing_finding(listed, secret)
    yield _leak_finding([stored, listed, helped, removed, relisted], secret,
                        logs_before,
                        planted_ok=stored.ok and stored.output.exit_code == 0)
    yield _no_reveal_finding(helped)
    yield _removed_finding(removed, relisted)


def _vault(args, stdin=None, label="s3", planted=()):
    """Run one ``vault`` subcommand against the persistent environment.

    Args:
        args: The vault subcommand and its arguments.
        stdin: Bytes for the child, or None.
        label: What to call the invocation in the archive.
        planted: The secrets this invocation puts in front of the product, so
            the archived copy is scrubbed. An invocation that omits a secret it
            planted writes it to ``env/runs/`` in clear.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return runs.attempt(
        [VAULT_SUBCOMMAND, WORKDIR_FLAG, str(runs.workspace_root())] + list(args),
        stdin=stdin, timeout_s=VAULT_TIMEOUT_S, label=label, planted=planted,
        env={"MAGI_PASSPHRASE": runs.passphrase()},
    )


def _log_snapshot():
    """Record the product's existing run-log files and their sizes.

    Only new bytes matter: the environment is persistent, so logs from earlier
    runs are already there and re-reading them would search text this scenario
    never produced.

    Complexity: ``O(number of log files)``.

    Returns:
        dict[pathlib.Path, int]: Each log file and its size in bytes.
    """
    directory = runs.workspace_root().joinpath(*LOG_DIR_PARTS)
    sizes = {}
    if not directory.is_dir():
        return sizes
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                sizes[path] = path.stat().st_size
        except OSError:
            # A log the product is still writing can vanish between the walk
            # and the stat. Treating it as absent means its whole content is
            # searched below, which errs towards looking harder.
            continue
    return sizes


def _new_log_bytes(before):
    """Read everything the product appended to its run log since *before*.

    Complexity: ``O(new bytes)``.

    Args:
        before: The snapshot taken before the scenario ran.

    Returns:
        bytes: The appended text of every log file, concatenated.
    """
    directory = runs.workspace_root().joinpath(*LOG_DIR_PARTS)
    if not directory.is_dir():
        return b""
    chunks = []
    for path in sorted(directory.rglob("*")):
        try:
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                handle.seek(before.get(path, 0))
                chunks.append(handle.read())
        except OSError:
            # Unreadable is not empty, and saying so is better than a silent
            # gap: the finding names the file the search could not cover.
            chunks.append(b"<unreadable log: %s>"
                          % str(path).encode("utf-8", errors="replace"))
    return b"".join(chunks)


def _stored_finding(stored, listed):
    """Judge assertion 1: the value went in from stdin and the name came back.

    Args:
        stored: The ``vault set`` attempt.
        listed: The ``vault ls`` attempt that followed it.

    Returns:
        Finding: PASS when both succeeded and the listing names the entry.
    """
    if not stored.ok:
        return _s3(0, Outcome.CANNOT_TEST, stored.failure)
    if stored.output.exit_code != 0:
        return _s3(0, Outcome.FAIL,
                   "vault set exited %d: %s"
                   % (stored.output.exit_code, _excerpt(stored.output)))
    if not listed.ok:
        return _s3(0, Outcome.CANNOT_TEST, listed.failure)
    if PROBE_NAME.encode("utf-8") not in listed.output.stdout:
        return _s3(0, Outcome.FAIL,
                   "vault ls does not name %s after set stored it" % PROBE_NAME)
    return _s3(0, Outcome.PASS, "")


def _listing_finding(listed, secret):
    """Judge assertion 2: the listing carries a name and two timestamps.

    Args:
        listed: The ``vault ls`` attempt.
        secret: The planted credential.

    Returns:
        Finding: PASS when the entry's line has all three fields and none of
        them is the value.
    """
    if not listed.ok:
        return _s3(1, Outcome.CANNOT_TEST, listed.failure)
    line = _entry_line(listed.output.stdout)
    if line is None:
        return _s3(1, Outcome.FAIL,
                   "no line of vault ls names %s" % PROBE_NAME)
    fields = [field.strip() for field in line.split(LS_SEPARATOR)]
    if len(fields) != LS_FIELDS or not all(fields):
        return _s3(1, Outcome.FAIL,
                   "the entry line carries %d fields, expected name and two "
                   "timestamps: %r" % (len(fields), line))
    if not all(_DIGIT.search(field) for field in fields[1:]):
        return _s3(1, Outcome.FAIL,
                   "the two fields after the name are not timestamps: %r" % line)
    if find_secret(listed.output.raw(), (secret,)):
        return _s3(1, Outcome.FAIL, "vault ls printed the stored value")
    return _s3(1, Outcome.PASS, "")


def _leak_finding(attempts, secret, logs_before, planted_ok):
    """Judge assertion 3 by searching, never by reading the product's code.

    A leak found is a FAIL whatever else happened -- the value reached the
    product on stdin, so it could surface in a failure message just as easily
    as in a listing. A leak NOT found is only worth a PASS if the value was
    actually stored: over a vault that refused it, "the value appears nowhere"
    is true of a value that was never there, which is a green that asserts
    nothing.

    Complexity: ``O(total captured bytes * derived forms)``.

    Args:
        attempts: Every invocation the scenario made.
        secret: The planted credential.
        logs_before: The run-log snapshot taken before it was planted.
        planted_ok: Whether ``vault set`` actually stored the value.

    Returns:
        Finding: PASS when no derived form appears and the value was stored.
    """
    captured = b"".join(attempt.output.raw() for attempt in attempts
                        if attempt.ok)
    if not captured:
        return _s3(2, Outcome.CANNOT_TEST,
                   "no invocation completed, so there was nothing to search")
    if find_secret(captured, (secret,)):
        return _s3(2, Outcome.FAIL,
                   "the planted value appears in the product's output")
    if find_secret(_new_log_bytes(logs_before), (secret,)):
        return _s3(2, Outcome.FAIL,
                   "the planted value appears in the product's run log")
    if not planted_ok:
        return _s3(2, Outcome.CANNOT_TEST,
                   "the value was never stored, so finding it nowhere says "
                   "nothing about whether the vault reveals what it holds")
    return _s3(2, Outcome.PASS, "")


def _no_reveal_finding(helped):
    """Judge assertion 4 against the parser's own list of subcommands.

    Args:
        helped: The ``vault --help`` attempt.

    Returns:
        Finding: PASS when no listed subcommand would print a stored value.
    """
    if not helped.ok:
        return _s3(3, Outcome.CANNOT_TEST, helped.failure)
    listed = _help_subcommands(helped.output.raw())
    if not listed:
        return _s3(3, Outcome.CANNOT_TEST,
                   "vault --help listed no subcommands to check")
    revealing = sorted(set(listed) & set(REVEALING_SUBCOMMANDS))
    if revealing:
        return _s3(3, Outcome.FAIL,
                   "vault offers %s" % ", ".join(revealing))
    return _s3(3, Outcome.PASS, "")


def _removed_finding(removed, relisted):
    """Judge assertion 5: ``rm`` takes the entry out and ``ls`` agrees.

    Args:
        removed: The ``vault rm`` attempt.
        relisted: The ``vault ls`` attempt that followed it.

    Returns:
        Finding: PASS when the removal succeeded and the name is gone.
    """
    if not removed.ok:
        return _s3(4, Outcome.CANNOT_TEST, removed.failure)
    if removed.output.exit_code != 0:
        return _s3(4, Outcome.FAIL,
                   "vault rm exited %d: %s"
                   % (removed.output.exit_code, _excerpt(removed.output)))
    if not relisted.ok:
        return _s3(4, Outcome.CANNOT_TEST, relisted.failure)
    if PROBE_NAME.encode("utf-8") in relisted.output.stdout:
        return _s3(4, Outcome.FAIL,
                   "vault ls still names %s after rm removed it" % PROBE_NAME)
    return _s3(4, Outcome.PASS, "")


def _entry_line(stdout):
    """Find the listing line that names the planted entry.

    Args:
        stdout: What ``vault ls`` printed.

    Returns:
        str | None: The line, or None when no line names it.
    """
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if PROBE_NAME in line:
            return line.strip()
    return None


def _help_subcommands(capture):
    """Extract the subcommand names out of a clap help listing.

    clap prints a ``Commands:`` heading followed by one indented
    ``name  description`` line per subcommand, and ends the block at the next
    unindented line. Reading only that block is what keeps a FLAG named like a
    forbidden subcommand out of the answer.

    Complexity: ``O(len(capture))``.

    Args:
        capture: What the product printed.

    Returns:
        list[str]: The subcommand names, in the order listed.
    """
    names = []
    inside = False
    for line in capture.decode("utf-8", errors="replace").splitlines():
        if _COMMANDS_HEADING.match(line):
            inside = True
            continue
        if not inside:
            continue
        if not line.strip():
            continue
        if not line[:1].isspace():
            break
        names.append(line.split()[0].strip(","))
    return names


def _excerpt(output, limit=400):
    """Render the beginning of a capture for a finding's detail.

    Args:
        output: The capture to quote.
        limit: How many bytes to keep.

    Returns:
        str: The first *limit* bytes of both streams, decoded leniently.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _s3(index, outcome, detail):
    """Build the finding for one entry of :data:`S3_ASSERTIONS`.

    Args:
        index: Position in :data:`S3_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S3 is standalone.
    """
    return Finding(assertion=S3_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)
