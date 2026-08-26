# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S17 -- the structured flag exists only where it should, and fails closed.

Protects REQ-EA01 and REQ-EA06. ``--structured-verdicts`` belongs to
``consult`` and to nothing else, and it requires JSON output; a flag that
parsed on ``query`` and quietly did nothing is a failure an operator only
discovers when the field they expected is missing from a batch run's output.

Same family as S14: this is clap surface verified in the RELEASE profile, where
a ``debug_assert!`` cannot save you.

**Assertion 1 separates three outcomes that all look like "it didn't work".**
A parse error is the contract. A silently accepted no-op is the defect. A
runtime exit 2 is a different defect -- the flag parsed on a subcommand that
should not have it and failed later -- and it shares an exit code with the
contract, so the exit code alone cannot tell them apart. The scenario reads
both the code and whether the message came from the argument parser.

**Assertion 2 carries the mirror image of that trap.** ``consult`` refusing
``--output-format text`` exits 2, and so does clap refusing a flag ``consult``
no longer has. Reading the exit code alone would report the feature as
protected on the very run that removed it, so this assertion also requires the
refusal NOT to be a parse error.
"""

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S17.
ASSERTIONS = (
    "query --structured-verdicts is a clap parse error — not an accepted "
    "no-op, not a runtime exit 2",
    "consult --structured-verdicts --output-format text exits 2",
)

QUERY_SUBCOMMAND = "query"
CONSULT_SUBCOMMAND = "consult"
STRUCTURED_VERDICTS_FLAG = "--structured-verdicts"
OUTPUT_FORMAT_FLAG = "--output-format"
TEXT_FORMAT = "text"
WORKDIR_FLAG = "-w"

#: How long the harness waits, in seconds. Both invocations are expected to die
#: during argument handling, before the database is opened or a socket is
#: touched, so this bound only has to outlast a process start.
INVOCATION_TIMEOUT_S = 120

#: The exit code clap uses for a parse error, which the product also uses for
#: CLI misuse -- which is exactly why the code is never read on its own here.
MISUSE_EXIT_CODE = 2

#: What the argument parser prints and the program does not. Its presence is
#: what makes a refusal a PARSE error rather than a run that started.
CLAP_USAGE_MARKER = b"usage:"

#: Both invocations are given a real workspace, so that a refusal cannot be
#: about a missing ``.magi/`` instead of about the flag.
PROBE_PROMPT = b"say ok\n"


@scenario("S17", assertions=ASSERTIONS)
def the_structured_flag_exists_only_where_it_should(run):
    """Offer the flag to the subcommand that lacks it, then misuse it properly.

    Args:
        run: Always ``None``; S17 declares no shared run.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    workspace = str(runs.workspace_root())
    on_query = runs.attempt(
        [QUERY_SUBCOMMAND, WORKDIR_FLAG, workspace, STRUCTURED_VERDICTS_FLAG],
        stdin=PROBE_PROMPT, timeout_s=INVOCATION_TIMEOUT_S,
        label="s17-query-flag", env={"MAGI_PASSPHRASE": runs.passphrase()},
    )
    on_consult = runs.attempt(
        [CONSULT_SUBCOMMAND, WORKDIR_FLAG, workspace,
         STRUCTURED_VERDICTS_FLAG, OUTPUT_FORMAT_FLAG, TEXT_FORMAT],
        stdin=PROBE_PROMPT, timeout_s=INVOCATION_TIMEOUT_S,
        label="s17-consult-text", env={"MAGI_PASSPHRASE": runs.passphrase()},
    )
    yield _query_rejects_at_parse_finding(on_query)
    yield _consult_fails_closed_finding(on_consult)


def _query_rejects_at_parse_finding(attempt):
    """Judge assertion 1 by classifying WHICH of the three outcomes happened.

    Args:
        attempt: The ``query`` capture.

    Returns:
        Finding: PASS only for the parse error, with the detail naming the
        other two by what they are rather than by their shared symptom.
    """
    if not attempt.ok:
        return _s17(0, Outcome.CANNOT_TEST, attempt.failure)
    if _is_parse_error(attempt.output):
        return _s17(0, Outcome.PASS, "")
    if attempt.output.exit_code == MISUSE_EXIT_CODE:
        return _s17(0, Outcome.FAIL,
                    "query exited %d without a parse error, so the flag was "
                    "accepted by the parser and rejected later: %s"
                    % (MISUSE_EXIT_CODE, _excerpt(attempt.output)))
    return _s17(0, Outcome.FAIL,
                "query accepted %s and exited %d, so the flag parses on a "
                "subcommand that cannot honour it: %s"
                % (STRUCTURED_VERDICTS_FLAG, attempt.output.exit_code,
                   _excerpt(attempt.output)))


def _consult_fails_closed_finding(attempt):
    """Judge assertion 2: the refusal is the program's, not the parser's.

    Args:
        attempt: The ``consult`` capture.

    Returns:
        Finding: PASS on a non-parse refusal with the misuse exit code.
    """
    if not attempt.ok:
        return _s17(1, Outcome.CANNOT_TEST, attempt.failure)
    if _is_parse_error(attempt.output):
        return _s17(1, Outcome.FAIL,
                    "consult no longer parses %s, so the exit code says the "
                    "combination was refused when the flag is simply gone: %s"
                    % (STRUCTURED_VERDICTS_FLAG, _excerpt(attempt.output)))
    if attempt.output.exit_code != MISUSE_EXIT_CODE:
        return _s17(1, Outcome.FAIL,
                    "consult exited %d with %s and %s %s, expected %d"
                    % (attempt.output.exit_code, STRUCTURED_VERDICTS_FLAG,
                       OUTPUT_FORMAT_FLAG, TEXT_FORMAT, MISUSE_EXIT_CODE))
    return _s17(1, Outcome.PASS, "")


def _is_parse_error(output):
    """Whether a capture is the argument parser refusing, not the program.

    Args:
        output: The capture to classify.

    Returns:
        bool: True when the exit code is the parser's AND the output carries
        both its usage block and the flag's name. All three are required: the
        code is shared with a runtime refusal, the usage block also appears in
        ``--help``, and a usage block about some other argument would say
        nothing about this flag.
    """
    lowered = output.raw().lower()
    return (output.exit_code == MISUSE_EXIT_CODE
            and CLAP_USAGE_MARKER in lowered
            and STRUCTURED_VERDICTS_FLAG.encode("utf-8") in lowered)


def _excerpt(output, limit=400):
    """Render the beginning of a capture for a finding's detail.

    Args:
        output: The capture to quote.
        limit: How many bytes to keep.

    Returns:
        str: The first *limit* bytes of both streams, decoded leniently.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _s17(index, outcome, detail):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S17 is standalone.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=None)
