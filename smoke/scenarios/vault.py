# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The three vault scenarios: S3 -- stores and never reveals; S4 -- never
deletes; S16 -- rotating a third-party credential costs no local data.

S3 protects REQ-V09: there is no ``get``/``cat``/``show``/``reveal``/``export``
subcommand, and the stored value never reaches any output.

S4 protects REQ-V35, the never-delete policy, and it is the scenario whose
failure is total loss of the user's data. Under the envelope a wrong passphrase
and a corrupt wrapped key fail the same GCM-SIV tag check, so a product that
wiped on failure would turn a typo into an unrecoverable loss.

Two things here are done the hard way on purpose.

The leak check **searches the capture** for the planted value instead of
reading the product's source, and it searches every derived form
(:func:`smoke.secrets.find_secret`) rather than the raw bytes alone -- a
raw-only search is the hole section 2.5 exists to close, because a value that
travels percent-encoded shares no substring with the value that was planted.

Assertion 4 is verified against ``--help``, the parser's own output. "I am not
aware of a subcommand that prints a value" is not evidence; a Commands list is.

S16 protects the durable invariant of the project's coding standards: rotating
a third-party credential never invalidates the local encrypted database. The
internal test of that invariant injects doubles; here the rotation is real, over
a database holding real data, performed by the binary an operator runs.

Its blind spot is recorded rather than closed, because the product's own design
makes closing it impossible: if somebody edited ``smoke.toml`` between a crash
and the restart, the restore writes the new value with no warning that it
differs from what was there. Comparing the vault's value against the file would
settle it, and REQ-V09 says a stored value is never printed and offers no
subcommand that would print one. What is missing is the warning, not the
correction: the file is the configuration's source of truth.
"""

import re

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.product import diagnose_counts
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

#: The verbatim assertion texts of the spec's section 8, for S4.
S4_ASSERTIONS = (
    "opening with a wrong passphrase fails with a typed WrongPassphrase, exit 1",
    "reopening with the correct passphrase still finds the accumulated history",
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

#: What S4 opens the database with. It must differ from the configured
#: passphrase -- a "wrong" one that happens to be right makes assertion 1 red
#: for the wrong reason and assertion 2 vacuous -- and it is long enough that
#: the product's own strength floor is never what rejects it.
WRONG_PASSPHRASE = "this-is-not-the-passphrase-0000"

#: The subcommand S4 measures the environment with. It is read-only and needs
#: NO passphrase (REQ-H32), which is the only reason the precondition can be
#: checked at all: every other way of counting the rows would first have to
#: open the very envelope the scenario is about to attack.
DIAGNOSE_SUBCOMMAND = "diagnose"

#: How ``diagnose`` announces that there is a wrapped key to fail against, and
#: the block its per-table counts follow.
ENVELOPE_PRESENT_LINE = "envelope: present"
COUNTS_HEADING = "counts:"

#: The tables whose rows are the user's accumulated history. ``vault`` is
#: excluded: S3 plants and removes an entry there in the same run, so counting
#: it would make S4's precondition depend on when S3 happened to run.
HISTORY_TABLES = ("sessions", "messages", "knowledge", "memories")

#: What the product prints when the unwrap fails. The assertion is on the TYPED
#: failure, not merely on a non-zero exit: a database that could not be opened
#: for some other reason is a different event with the same exit code.
WRONG_PASSPHRASE_MARKER = b"incorrect passphrase"

_COMMANDS_HEADING = re.compile(r"^\s*(commands|subcommands):\s*$",
                               re.IGNORECASE)
_DIGIT = re.compile(r"\d")


@scenario("S3", assertions=S3_ASSERTIONS)
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


def _vault(args, stdin=None, label="s3", planted=(), passphrase=None):
    """Run one ``vault`` subcommand against the persistent environment.

    Args:
        args: The vault subcommand and its arguments.
        stdin: Bytes for the child, or None.
        label: What to call the invocation in the archive.
        planted: The secrets this invocation puts in front of the product, so
            the archived copy is scrubbed. An invocation that omits a secret it
            planted writes it to ``env/runs/`` in clear.
        passphrase: What to unlock with; the configured one when omitted. It
            travels in ``MAGI_PASSPHRASE`` and never in ``-p``, which is a
            global flag and would ride in a command line every process on the
            machine can read while the child lives.

    Returns:
        Attempt: The capture, or why there is none.
    """
    if passphrase is None:
        passphrase = runs.passphrase()
    return runs.attempt(
        [VAULT_SUBCOMMAND, WORKDIR_FLAG, str(runs.workspace_root())] + list(args),
        stdin=stdin, timeout_s=VAULT_TIMEOUT_S, label=label, planted=planted,
        env={"MAGI_PASSPHRASE": passphrase},
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


@scenario("S4", assertions=S4_ASSERTIONS)
def a_wrong_passphrase_destroys_nothing(run):
    """Fail an unwrap on purpose, then check nothing was lost doing it.

    The precondition is verified, not assumed. Over a database with no envelope
    there is nothing for a wrong passphrase to fail against -- the product
    would bootstrap one and exit 0 -- and over a database with no rows,
    "the accumulated history is still there" is true of a history that never
    existed. Either way the scenario reports CANNOT_TEST naming what was
    missing, and never PASS.

    Args:
        run: Always ``None``; S4 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S4_ASSERTIONS`, in that order.
    """
    before = _diagnose("s4-diagnose-before")
    refused = _vault(["ls"], label="s4-wrong",
                     passphrase=WRONG_PASSPHRASE)
    after = _diagnose("s4-diagnose-after")
    reopened = _vault(["ls"], label="s4-reopen")
    yield _refusal_is_typed_finding(before, refused)
    yield _history_survived_finding(before, after, reopened)


def _diagnose(label):
    """Read the database's structure without unlocking it.

    Args:
        label: What to call the invocation in the archive.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return runs.attempt(
        [VAULT_SUBCOMMAND, WORKDIR_FLAG, str(runs.workspace_root()),
         DIAGNOSE_SUBCOMMAND],
        timeout_s=VAULT_TIMEOUT_S, label=label,
    )


def _envelope_present(attempt):
    """Whether ``diagnose`` reported a wrapped key.

    Args:
        attempt: The ``vault diagnose`` attempt.

    Returns:
        bool: True when the report names a present envelope.
    """
    if not attempt.ok:
        return False
    text = attempt.output.stdout.decode("utf-8", errors="replace")
    return any(line.strip() == ENVELOPE_PRESENT_LINE
               for line in text.splitlines())


def _history_counts(attempt):
    """Extract the per-table row counts out of a ``diagnose`` report.

    Only the lines inside the ``counts:`` block are read: the report also
    carries an envelope line and a verdict, and a looser match would take
    ``fec: ok`` for a table.

    Complexity: ``O(lines)``.

    Args:
        attempt: The ``vault diagnose`` attempt.

    Returns:
        dict[str, int] | None: The counts of :data:`HISTORY_TABLES` that the
        report gave a number for, or None when the report could not be read. A
        table reported as ``missing`` is absent from the mapping rather than
        recorded as zero -- unknown is not empty.
    """
    if not attempt.ok or attempt.output.exit_code != 0:
        return None
    return {table: count
            for table, count in diagnose_counts(attempt.output.stdout).items()
            if table in HISTORY_TABLES}


def _refusal_is_typed_finding(before, refused):
    """Judge assertion 1: the refusal is typed, and it is exit 1.

    Args:
        before: The ``diagnose`` taken before the attempt.
        refused: The open attempted with the wrong passphrase.

    Returns:
        Finding: PASS on exit 1 carrying the typed refusal.
    """
    if not _envelope_present(before):
        return _s4(0, Outcome.CANNOT_TEST,
                   "the environment carries no envelope, so a wrong "
                   "passphrase has no wrapped key to fail against")
    if not refused.ok:
        return _s4(0, Outcome.CANNOT_TEST, refused.failure)
    if refused.output.exit_code == 0:
        return _s4(0, Outcome.FAIL,
                   "the vault opened with a passphrase that is not its own")
    if WRONG_PASSPHRASE_MARKER not in refused.output.raw():
        return _s4(0, Outcome.FAIL,
                   "the refusal is untyped: exit %d without naming an "
                   "incorrect passphrase: %s"
                   % (refused.output.exit_code, _excerpt(refused.output)))
    if refused.output.exit_code != 1:
        return _s4(0, Outcome.FAIL,
                   "the typed refusal exited %d, expected 1"
                   % refused.output.exit_code)
    return _s4(0, Outcome.PASS, "")


def _history_survived_finding(before, after, reopened):
    """Judge assertion 2: nothing was deleted, and the vault still opens.

    Args:
        before: The ``diagnose`` taken before the wrong passphrase.
        after: The one taken after it.
        reopened: The open attempted with the correct passphrase.

    Returns:
        Finding: PASS when every table holds what it held and the correct
        passphrase still works.
    """
    counts_before = _history_counts(before)
    counts_after = _history_counts(after)
    if counts_before is None or counts_after is None:
        return _s4(1, Outcome.CANNOT_TEST,
                   "the database's structure could not be read, so nothing "
                   "can be said about what survived")
    if not any(counts_before.values()):
        return _s4(1, Outcome.CANNOT_TEST,
                   "the environment holds no accumulated history yet (%s), so "
                   "this assertion would pass over nothing"
                   % _render_counts(counts_before))
    lost = ["%s went from %d to %d"
            % (table, counts_before[table], counts_after.get(table, 0))
            for table in sorted(counts_before)
            if counts_after.get(table, 0) < counts_before[table]]
    if lost:
        return _s4(1, Outcome.FAIL,
                   "a refused open destroyed data: %s" % "; ".join(lost))
    if not reopened.ok:
        return _s4(1, Outcome.CANNOT_TEST, reopened.failure)
    if reopened.output.exit_code != 0:
        return _s4(1, Outcome.FAIL,
                   "the correct passphrase no longer opens the vault: exit "
                   "%d: %s" % (reopened.output.exit_code,
                               _excerpt(reopened.output)))
    return _s4(1, Outcome.PASS, "")


def _render_counts(counts):
    """Render a counts mapping for a finding's detail.

    Args:
        counts: Table name to row count.

    Returns:
        str: The pairs, sorted by table, or a note that none were reported.
    """
    if not counts:
        return "no table reported a count"
    return ", ".join("%s=%d" % (table, counts[table])
                     for table in sorted(counts))


def _s4(index, outcome, detail):
    """Build the finding for one entry of :data:`S4_ASSERTIONS`.

    Args:
        index: Position in :data:`S4_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S4 is standalone.
    """
    return Finding(assertion=S4_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)


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


#: The verbatim assertion texts of the spec's section 8, for S16.
S16_ASSERTIONS = (
    "after rotating the stored API key, the DB still opens with the same "
    "passphrase",
    "the previous history is still there",
)

#: The run S16 reads, named once so the decorator and every finding's run id
#: cannot drift apart.
S16_RUN_ID = "R7"


@scenario("S16", assertions=S16_ASSERTIONS, run=S16_RUN_ID,
           needs_backend=True)
def rotating_a_credential_keeps_the_database(run):
    """Check what R7's rotation left behind: the same passphrase, the same rows.

    The rotation is over by the time this runs -- R7 puts the real credential
    back in a ``finally`` -- so the scenario reads the aftermath rather than
    driving the rotation itself. That is deliberate: one place performs the
    dangerous half, and it is the place that also knows how to undo it.

    Both assertions are refused when R7 did not complete. A database that opens
    is not evidence about a rotation, and over a truncated run it is not even
    evidence that one was attempted.

    Args:
        run: R7's ``RunResult``, or ``None`` when the run never produced one.

    Yields:
        Finding: One per entry of :data:`S16_ASSERTIONS`, in that order.
    """
    # A timed-out run never reaches here either: S16 does not declare
    # ``inspects_timeouts``, so the runner substitutes for the whole scenario.
    if run is None:
        for index in range(len(S16_ASSERTIONS)):
            yield _s16(index, Outcome.CANNOT_TEST,
                       "run %s never executed, so no credential was rotated; "
                       "a database that still opens says nothing about "
                       "surviving a rotation that did not happen" % S16_RUN_ID)
        return

    diagnosed = _diagnose("s16-diagnose")
    reopened = _vault(["ls"], label="s16-reopen")
    yield _still_opens_finding(diagnosed, reopened)
    yield _history_intact_finding(diagnosed, reopened, run.baseline)


def _still_opens_finding(diagnosed, reopened):
    """Judge assertion 1: the passphrase that worked before still works.

    Args:
        diagnosed: The ``vault diagnose`` taken after the rotation.
        reopened: The open attempted with the configured passphrase.

    Returns:
        Finding: PASS when the database opens and there was an envelope for the
        rotation to have damaged.
    """
    if not _envelope_present(diagnosed):
        return _s16(0, Outcome.CANNOT_TEST,
                    "the environment carries no envelope, so opening it says "
                    "nothing about a wrapped key surviving the rotation")
    if not reopened.ok:
        return _s16(0, Outcome.CANNOT_TEST, reopened.failure)
    if reopened.output.exit_code != 0:
        return _s16(0, Outcome.FAIL,
                    "the configured passphrase no longer opens the database "
                    "after the backend credential was rotated: exit %d: %s"
                    % (reopened.output.exit_code, _excerpt(reopened.output)))
    return _s16(0, Outcome.PASS, "")


def _history_intact_finding(diagnosed, reopened, baseline):
    """Judge assertion 2: the rows that were there are still there.

    Against a BEFORE, because without one the assertion can only ask whether
    SOME history is there -- and a rotation that destroyed most of it leaves
    every table non-zero and passes. S4 has always compared before against
    after; this did not, and the two sit in the same module.

    The same precondition trap as S4 applies on top, and is refused the same
    way. Over a database with no rows the claim is true of a history that never
    existed, which is a green that checked nothing.

    Args:
        diagnosed: The ``vault diagnose`` taken after the rotation.
        reopened: The open attempted with the configured passphrase.
        baseline: The counts the rotating run took before it started, or None
            when it recorded none.

    Returns:
        Finding: PASS when every table holds at least what it held.
    """
    counts = _history_counts(diagnosed)
    if baseline is None:
        return _s16(1, Outcome.CANNOT_TEST,
                    "the rotation recorded no baseline, so there is nothing "
                    "to compare what survived against")
    if counts is None:
        return _s16(1, Outcome.CANNOT_TEST,
                    "the database's structure could not be read, so nothing "
                    "can be said about what survived")
    # HISTORY_TABLES, not everything the report listed. ``vault`` is the
    # rotation's own workspace: R7 plants a marker before it rotates and
    # removes it after, and R6's authenticated endpoint plants four
    # placeholder entries and removes those. Counting that table reads the
    # run's own bookkeeping as the user losing data, and a live run said so:
    # "vault went from 1 to 0". S4 excludes it for the same reason.
    baseline = {table: count for table, count in baseline.items()
                if table in HISTORY_TABLES}
    if not any(baseline.values()):
        return _s16(1, Outcome.CANNOT_TEST,
                    "the environment held no accumulated history before the "
                    "rotation (%s), so this assertion would pass over nothing"
                    % _render_counts(baseline))
    # Absent and zero are different answers, and flattening them reports data
    # loss on a table nobody measured. ``diagnose_counts`` leaves a table the
    # product rendered ``missing`` out of the mapping and says so; reading it
    # back with a zero default contradicted that in the next three lines.
    unmeasured = sorted(table for table in baseline if table not in counts)
    if unmeasured:
        return _s16(1, Outcome.CANNOT_TEST,
                    "the report gave no count for %s, so what survived there "
                    "is unknown rather than lost" % ", ".join(unmeasured))
    lost = ["%s went from %d to %d"
            % (table, baseline[table], counts[table])
            for table in sorted(baseline)
            if counts[table] < baseline[table]]
    if lost:
        return _s16(1, Outcome.FAIL,
                    "rotating the credential destroyed data: %s"
                    % "; ".join(lost))
    if not reopened.ok:
        return _s16(1, Outcome.CANNOT_TEST, reopened.failure)
    if reopened.output.exit_code != 0:
        return _s16(1, Outcome.CANNOT_TEST,
                    "the database no longer opens, so what it holds could not "
                    "be confirmed")
    return _s16(1, Outcome.PASS, "")


def _s16(index, outcome, detail):
    """Build the finding for one entry of :data:`S16_ASSERTIONS`.

    Args:
        index: Position in :data:`S16_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, attributed to the run whose rotation it reads.
    """
    return Finding(assertion=S16_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=S16_RUN_ID)
