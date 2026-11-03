# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-22
"""S44-S46 -- what the headless commands do with the envelope ``system`` field.

The suite proves each piece of the behaviour with doubles. What it cannot see is
the release binary doing it end to end: the warning reaching a real stderr at
``--log-level error``, the flag telling the truth in the JSON a CI consumer
parses, and the size cap refusing before a model is paid for.

* **S44** (``query``, envelope with ``system``, no override flag): the field is
  ignored loudly, on stderr and in ``applied_caps``.
* **S45** (``consult --allow-system-override``): the flag says the adjustment
  was installed. The two flag assertions read ``applied_caps`` from *any*
  envelope the run produced, error envelopes included, because the flag records
  what was installed before the mages were called and not what they answered.
  Only the first assertion depends on the trio being healthy, so a backend that
  is down cannot make the certificate claim the override went uncovered.
* **S46** (``query``, ``system`` one byte over the default cap): exit 2 naming
  the cap, with nothing on stdout.
"""

from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.scenarios.trio import (ENVIRONMENTAL_ERROR_KINDS, ERROR_KEY,
                                  _envelope_of)

#: The verbatim assertion texts of S44, S45 and S46.
S44_ASSERTIONS = (
    "exit 0",
    "stderr contains the ignored-system warning even at --log-level error",
    "applied_caps.system_ignored is true",
    "applied_caps.system_override_applied is false",
)
S45_ASSERTIONS = (
    "the run produced a consult envelope (a provider or timeout error is CANNOT_TEST)",
    "applied_caps.system_override_applied is true in any envelope the run produced",
    "applied_caps.system_ignored is false in any envelope the run produced",
)
S46_ASSERTIONS = (
    "exit 2",
    "stderr names max_system_bytes",
    "no consult or query output on stdout",
)

#: The warning the product writes when it ignores a ``system`` field.
IGNORED_SYSTEM_WARNING = (
    "warning: the request's \"system\" field was ignored; pass "
    "--allow-system-override or set [headless] allow_system_override = true "
    "to apply it"
)

#: The configuration key the size refusal must name.
SYSTEM_CAP_KEY = "max_system_bytes"

#: Exit code of a successful run and of CLI misuse.
EXIT_OK = 0
EXIT_MISUSE = 2

#: ``RunOutcome`` keys that carry product output on a run that got as far as
#: calling a model; none may be populated when the request is refused.
OUTPUT_KEYS = ("response", "consult")

#: The key under which the output document publishes the run's caps.
APPLIED_CAPS_KEY = "applied_caps"


def _finding(texts, index, outcome, run_id, detail=""):
    """Build the finding for assertion *index* of *texts*.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        outcome: What became of it.
        run_id: The shared run it came from.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=texts[index], outcome=outcome, detail=detail,
                   run_id=run_id)


def _verdict(texts, index, run_id, held, detail):
    """Turn a boolean check into a PASS or FAIL finding.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        run_id: The shared run it came from.
        held: Whether the property held.
        detail: The cause reported when it did not.

    Returns:
        Finding: ``PASS`` when *held*, otherwise ``FAIL`` carrying *detail*.
    """
    return _finding(texts, index, Outcome.PASS if held else Outcome.FAIL,
                    run_id, "" if held else detail)


def _all_cannot_test(texts, run_id, detail):
    """One ``CANNOT_TEST`` finding per assertion of a scenario.

    Args:
        texts: The scenario's assertion texts.
        run_id: The shared run that produced nothing.
        detail: The cause.

    Yields:
        Finding: One per entry of *texts*, in order.
    """
    for index in range(len(texts)):
        yield _finding(texts, index, Outcome.CANNOT_TEST, run_id, detail)


def _all_hung(texts, run_id, run):
    """One ``FAIL`` finding per assertion when a refusal before any model hung.

    MAGI FIX (F1a2): R11 is refused before any model is reached, so a timeout
    there has no slow provider to blame. The executor builds a hung run with
    ``exit_code`` 0 and whatever output it caught, so no assertion is judged on
    that synthetic result. The same rule as S48-S51.

    Args:
        texts: The scenario's assertion texts.
        run_id: The shared run that hung.
        run: Its ``RunResult``.

    Yields:
        Finding: ``FAIL`` for every entry of *texts*, in order.
    """
    detail = ("run %s timed out after %.0f s: a refusal that needs no model "
              "must not hang" % (run_id, run.duration_s))
    for index in range(len(texts)):
        yield _finding(texts, index, Outcome.FAIL, run_id, detail)


def _applied_caps(run):
    """Read ``applied_caps`` from the run's stdout document.

    Args:
        run: The run's ``RunResult`` (never ``None``).

    Returns:
        tuple: ``(caps, failure)``: the ``applied_caps`` object, or ``None``
        and the reason it could not be read.
    """
    try:
        document = run.output.json()
    except ProductOutputError as exc:
        return None, "stdout is not a JSON document: %s" % exc
    caps = document.get(APPLIED_CAPS_KEY)
    if not isinstance(caps, dict):
        return None, "the document carries no applied_caps object"
    return caps, ""


def _flag_finding(texts, index, run_id, caps, failure, key, expected):
    """Judge one boolean of ``applied_caps``.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        run_id: The shared run it came from.
        caps: The ``applied_caps`` object, or ``None``.
        failure: Why there is none.
        key: The boolean being checked.
        expected: The value it must have.

    Returns:
        Finding: ``FAIL`` when the object is missing or the value differs.
    """
    if caps is None:
        return _finding(texts, index, Outcome.FAIL, run_id, failure)
    value = caps.get(key)
    return _verdict(texts, index, run_id, value is expected,
                    "applied_caps.%s is %r, expected %r" % (key, value, expected))


def _stderr_text(run):
    """Decode a run's stderr leniently.

    Args:
        run: The run's ``RunResult``.

    Returns:
        str: The decoded text.
    """
    return run.output.stderr.decode("utf-8", errors="replace")


def _exit_finding(run):
    """Judge S44's exit, degrading a failure the environment caused.

    MAGI FIX (F1a2): a backend that passes preflight and then fails during the
    run (provider error, quota, missing model, timeout) makes ``query`` exit
    non-zero with a typed ``error`` whose kind is the environment's, not the
    product's. A bare exit-code check called that a product FAIL; S45 and the
    trio scenarios already read the same kinds as ``CANNOT_TEST``.

    Args:
        run: R9's ``RunResult`` (never ``None``).

    Returns:
        Finding: ``PASS`` on exit 0, ``CANNOT_TEST`` on an environmental error,
        ``FAIL`` otherwise.
    """
    code = run.output.exit_code
    if code != EXIT_OK:
        try:
            error = run.output.json().get(ERROR_KEY)
        except ProductOutputError:
            error = None
        if isinstance(error, dict) and error.get("kind") in ENVIRONMENTAL_ERROR_KINDS:
            return _finding(S44_ASSERTIONS, 0, Outcome.CANNOT_TEST, "R9",
                            "exited %d with a %s error: %s"
                            % (code, error.get("kind"), error.get("message", "")))
    return _verdict(S44_ASSERTIONS, 0, "R9", code == EXIT_OK,
                    "exited %d" % code)


@scenario("S44", assertions=S44_ASSERTIONS, run="R9", needs_backend=True)
def an_ignored_system_is_announced(run):
    """Check that an ignored ``system`` is reported on stderr and in the JSON.

    Args:
        run: R9's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S44_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(S44_ASSERTIONS, "R9",
                                    "run R9 produced no capture to inspect")
        return
    yield _exit_finding(run)
    yield _verdict(S44_ASSERTIONS, 1, "R9",
                   IGNORED_SYSTEM_WARNING in _stderr_text(run),
                   "stderr does not carry the warning")
    caps, failure = _applied_caps(run)
    yield _flag_finding(S44_ASSERTIONS, 2, "R9", caps, failure,
                        "system_ignored", True)
    yield _flag_finding(S44_ASSERTIONS, 3, "R9", caps, failure,
                        "system_override_applied", False)


@scenario("S45", assertions=S45_ASSERTIONS, run="R10", needs_backend=True)
def an_applied_system_reaches_the_mages(run):
    """Check that ``consult`` with the override flag reports it installed.

    Args:
        run: R10's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S45_ASSERTIONS`, in that order. The
        first inherits the trio-health classification; the other two read the
        flags from whatever document the run wrote.
    """
    envelope = _envelope_of(run, "R10")
    healthy = envelope.body is not None
    yield _finding(S45_ASSERTIONS, 0,
                   Outcome.PASS if healthy else envelope.outcome, "R10",
                   "" if healthy else envelope.detail)
    if run is None:
        for index in (1, 2):
            yield _finding(S45_ASSERTIONS, index, Outcome.CANNOT_TEST, "R10",
                           "run R10 produced no capture to inspect")
        return
    caps, failure = _applied_caps(run)
    yield _flag_finding(S45_ASSERTIONS, 1, "R10", caps, failure,
                        "system_override_applied", True)
    yield _flag_finding(S45_ASSERTIONS, 2, "R10", caps, failure,
                        "system_ignored", False)


def _stdout_has_no_output(run):
    """Whether stdout carries no model-produced output.

    Args:
        run: The run's ``RunResult``.

    Returns:
        bool: ``True`` when stdout is empty or a JSON document whose
        :data:`OUTPUT_KEYS` are all absent or null.
    """
    if not run.output.stdout.strip():
        return True
    try:
        document = run.output.json()
    except ProductOutputError:
        return False
    return all(document.get(key) is None for key in OUTPUT_KEYS)


@scenario("S46", assertions=S46_ASSERTIONS, run="R11", inspects_timeouts=True)
def an_oversized_system_is_refused(run):
    """Check that a ``system`` over the cap is refused before any model runs.

    The scenario declares ``inspects_timeouts``: the refusal needs no model,
    so a timed-out R11 is a hang and every assertion is ``FAIL``.

    Args:
        run: R11's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S46_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(S46_ASSERTIONS, "R11",
                                    "run R11 produced no capture to inspect")
        return
    if run.timed_out:
        yield from _all_hung(S46_ASSERTIONS, "R11", run)
        return
    code = run.output.exit_code
    yield _verdict(S46_ASSERTIONS, 0, "R11", code == EXIT_MISUSE,
                   "exited %d, expected %d" % (code, EXIT_MISUSE))
    yield _verdict(S46_ASSERTIONS, 1, "R11",
                   SYSTEM_CAP_KEY in _stderr_text(run),
                   "stderr does not name %s" % SYSTEM_CAP_KEY)
    yield _verdict(S46_ASSERTIONS, 2, "R11", _stdout_has_no_output(run),
                   "stdout carries product output")
