# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S10 -- no credential leaks.

Protects ``src/redact.rs`` and the five sites of the v0.12.0 gate, which were
the most recurrent defect of that milestone: strings composed by another crate
reaching the output with a credential inside. A ``failed_agents`` cause, an
HTTP error body carrying the ``Authorization`` header, an ``EndpointError``'s
``Display`` in a notice. **None of those is a URL field of a JSON object** --
they are prose, on stderr and in the log, with a URL in the middle. An
assertion that only inspected well-typed JSON URLs would be looking at exactly
the channel where the product never failed, which is why this one walks four.

**What fails on purpose is the RUN, not the scenario.** S10 is a regression
detector and is expected green; a red here is a defect to fix today, not a
known debt. R6 is built so the product *fails* -- a credential that does not
authenticate -- because the redaction defect lives in the **error path**, and a
successful run never walks it. Provoking the failure is how the code under test
is reached; the verdict on that code is a separate thing.

**The first three assertions and the fourth fail for different reasons, and
that asymmetry is the design.** One to three search the planted value in every
derived form, so a single unforeseen encoding fools all three at once -- they
share one derivation and therefore one blind spot. The fourth looks at the
SHAPE of what was emitted: an authority whose userinfo survived is a leak no
matter how the credential is spelled inside it, and it fires on a credential
nobody planted, which is the hole the first three cannot close.
"""

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.secrets import find_secret, find_unredacted_authorities

#: The verbatim assertion texts of the spec's section 8, for S10.
ASSERTIONS = (
    "the credential appears in neither stdout nor stderr",
    "nor in the JSON output",
    "nor in the run log",
    "every authority emitted anywhere is redacted",
)


@scenario("S10", run="R6", needs_backend=True)
def no_credential_leaks(run):
    """Search four channels for the planted credential, then for shape.

    Args:
        run: The completed R6, or ``None`` when it did not execute.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    if run is None or getattr(run, "timed_out", False):
        cause = ("R6 did not execute" if run is None
                 else "R6 exceeded its ceiling, so its output is truncated")
        for index in range(len(ASSERTIONS)):
            yield _finding(index, Outcome.CANNOT_TEST,
                           "%s; finding no leak in output that was cut short "
                           "proves nothing" % cause)
        return

    planted = tuple(run.planted)
    streams = run.output.raw()
    log = _archived_log(run.run_id)

    yield _leak_finding(0, streams, planted,
                        "stdout and stderr")
    yield _leak_finding(1, run.output.stdout, planted, "the JSON output")
    if log is None:
        yield _finding(2, Outcome.CANNOT_TEST,
                       "no archived run log was found for %s, so it could not "
                       "be searched" % run.run_id)
    else:
        yield _leak_finding(2, log, planted, "the run log")
    yield _authority_finding(streams, log)


def _leak_finding(index, data, planted, channel):
    """Judge one channel against every derived form of every planted secret.

    Searching the derived set rather than the raw value is not thoroughness for
    its own sake: the vault percent-encodes a value before it enters a URL, so
    in R6's error path the credential arrives encoded and a raw-only search
    reports clean over a real leak.

    Complexity: ``O(len(data) * total derived forms)``.

    Args:
        index: Which assertion this is.
        data: The bytes to search, at full fidelity and unscrubbed.
        planted: What the executor put in front of the product.
        channel: What to call the channel in the detail.

    Returns:
        Finding: FAIL naming each secret found, PASS otherwise.
    """
    found = find_secret(data, planted)
    if found:
        return _finding(index, Outcome.FAIL,
                        "%s carries %s" % (channel, ", ".join(sorted(set(found)))))
    return _finding(index, Outcome.PASS, "")


def _authority_finding(streams, log):
    """Judge assertion 4 on SHAPE, never on the secret.

    Args:
        streams: Both product streams, unscrubbed.
        log: The archived run log, or None when it was not found.

    Returns:
        Finding: FAIL naming each unredacted authority, PASS otherwise.
    """
    exposed = find_unredacted_authorities(streams)
    if log is not None:
        exposed = exposed + find_unredacted_authorities(log)
    if exposed:
        return _finding(
            3, Outcome.FAIL,
            "an authority reached the output with its userinfo intact: %s"
            % "; ".join(sorted(set(exposed))))
    return _finding(3, Outcome.PASS, "")


def _archived_log(run_id):
    """Read back what the executor archived for THIS run.

    The archive is the third channel, and it is the one a reader keeps. The
    capture the assertions walk is NOT what is persisted: the archive is
    scrubbed (section 2.5), so a leak found here is a leak the scrubber did not
    know about, which is precisely the case worth reporting.

    It is scoped to the run's own directory, and that scoping is the fix to a
    real false positive. The first version globbed every ``invocation.log``
    under the archive root and concatenated them, so it read fixtures other
    scenarios had written. One of those is the placeholder the product
    documents and requires, ``https://[user]:[password]@host``, and assertion 4
    reported the harness's own fixture as an authority the product had leaked.

    Complexity: ``O(archived bytes for this run)``.

    Args:
        run_id: The run whose archive to read.

    Returns:
        bytes | None: The archived bytes, or None when nothing was archived.
    """
    try:
        directory = runs.archive_root()
    except Exception:  # noqa: BLE001 - an unconfigured module means "cannot tell"
        return None
    if directory is None:
        return None
    mine = directory / run_id
    if not mine.is_dir():
        return None
    chunks = []
    for path in sorted(mine.rglob("*")):
        try:
            if path.is_file():
                chunks.append(path.read_bytes())
        except OSError:
            continue
    if not chunks:
        return None
    return b"\n".join(chunks)


def _finding(index, outcome, detail):
    """Build one finding for this scenario.

    Args:
        index: Which assertion it answers.
        outcome: The verdict.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, attributed to R6.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id="R6")
