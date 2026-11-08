# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""S53-S54 -- ``code-review`` never lets a secret in a file's CONTENT leave the process (REQ-RV22).

S49 covers a secret in a file's NAME. These two cover its content, which the
product treats in two ways:

* **S53** (R18, the real trio): a file holding a single-line key is REVIEWED
  with the key replaced by ``<redacted: anthropic-key>``. The product declares
  it under ``code_review.masked`` (path, cause and pattern, never the value),
  keeps it out of ``code_review.skipped``, and the raw key appears in neither
  stdout, stderr nor the run log. The ``<redacted: ...>`` marker itself goes to
  the model only; the product publishes the pattern name, which is what is read.
* **S54** (R19): a directory holding only a multi-line private-key block has
  nothing to review. The block is left out whole as ``secret`` (a body line has
  no pattern of its own, so masking cannot catch it), the refusal names the file
  with that cause, and no body line appears in stdout, stderr or the run log.
  The refusal carries no JSON document, so the omission is read from stderr's
  list, like S49's.

Both measure what the product does with the bytes, never what a model answers.
S54 needs no backend and declares ``inspects_timeouts``; S53 reaches the trio,
so a timed-out R18 is reported as ``CANNOT_TEST`` by the runner.

What S53 cannot see is the request a model endpoint receives: R18 talks to a
real backend, which keeps no record the harness can read, so its leak check
covers the process's own outputs (stdout, stderr, the run log). The bytes sent
to the trio and to the embedder are pinned in the unit suite by recording
doubles (``no_byte_of_a_one_line_secret_reaches_the_embedder_or_the_trio``,
``units_and_definitions_holding_a_one_line_secret_reach_the_trio_masked``,
``a_batch_quoting_a_one_line_secret_sends_its_documentation_query_masked``).
"""

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.scenarios import code_review

#: The verbatim assertion texts of S53 and S54.
S53_ASSERTIONS = (
    "exit 0",
    "code_review.masked lists exactly the keyed file with cause secret and pattern "
    "anthropic-key, and masked_unlisted is 0",
    "code_review.skipped is empty and skipped_unlisted is 0: the keyed file was reviewed",
    "the toy key reaches neither stdout, stderr nor the run log",
)
S54_ASSERTIONS = (
    "exit 2",
    "stderr says nothing to review, leaving out the private-key file as secret and nothing else",
    "no line of the private-key body reaches stdout, stderr or the run log",
)

#: Exit code of a review that produced its report.
EXIT_OK = 0

#: The keys of the product's JSON S53 reads.
CODE_REVIEW_KEY = "code_review"
MASKED_KEY = "masked"
MASKED_UNLISTED_KEY = "masked_unlisted"
SKIPPED_KEY = "skipped"
SKIPPED_UNLISTED_KEY = "skipped_unlisted"

#: The one entry ``code_review.masked`` must hold: the product's ``Masked`` object
#: serialized (``src/review/report.rs``), the cause being ``SkipCause::Secret``.
S53_EXPECTED_MASKED = [{"path": runs.REVIEW_MASKED_FILE, "cause": "secret",
                        "pattern": runs.REVIEW_MASKED_PATTERN}]

#: What S54's omission list must read: the key-block file, as ``secret``.
S54_EXPECTED_OMISSIONS = [(runs.REVIEW_KEYBLOCK_FILE, "secret")]


def _no_capture(texts, run_id):
    """One ``CANNOT_TEST`` finding per assertion of a run that left no capture.

    Args:
        texts: The scenario's assertion texts.
        run_id: The run that produced nothing.

    Yields:
        Finding: One per entry of *texts*.
    """
    yield from code_review._all_cannot_test(
        texts, run_id, "run %s produced no capture to inspect" % run_id)


def _leak_finding(texts, index, run_id, run, needles, what):
    """Judge an assertion that none of *needles* left the process.

    Reads stdout, stderr and the captured run log. The needles are never
    printed, only the outputs that hold one.

    Args:
        texts: The scenario's assertion texts.
        index: Position of the assertion in *texts*.
        run_id: The run it came from.
        run: Its ``RunResult``, neither missing nor hung.
        needles: Byte strings that must appear in no output.
        what: Short name of what the needles are, for the failure detail.

    Returns:
        Finding: ``FAIL`` naming each output holding a needle; ``CANNOT_TEST``
        when none does but a log file could not be read whole; otherwise
        ``PASS`` (an absent log directory carries nothing).
    """
    outputs = (("stdout", (run.output.stdout,)), ("stderr", (run.output.stderr,)),
               ("the run log", run.review_log or ()))
    leaked = [name for name, bodies in outputs
              if any(needle in body for needle in needles for body in bodies)]
    if leaked:
        return code_review._finding(texts, index, Outcome.FAIL, run_id,
                                    "%s appears in %s" % (what, ", ".join(leaked)))
    if run.review_log_unreadable:
        return code_review._finding(
            texts, index, Outcome.CANNOT_TEST, run_id,
            "the run log could not be read whole: %s" % ", ".join(run.review_log_unreadable))
    return code_review._finding(texts, index, Outcome.PASS, run_id)


def _section_list(section, key, unlisted_key):
    """Read one omission list and its ``*_unlisted`` count from ``code_review``.

    Args:
        section: The ``code_review`` object.
        key: The list's key.
        unlisted_key: The key of the count of entries not listed.

    Returns:
        tuple: ``(items, unlisted, failure)``; ``failure`` is empty when both
        are present and well typed. A missing key is a failure, never ``[]``.
    """
    items, failure = code_review._list_under(section, key)
    if items is None:
        return None, None, failure
    unlisted = section.get(unlisted_key)
    if not code_review._is_count(unlisted):
        return None, None, "code_review.%s is missing or not an integer" % unlisted_key
    return items, unlisted, ""


def _normalized_masked(items):
    """Read a ``masked`` list with its paths' separators as ``/``.

    Args:
        items: The list read from the product's JSON.

    Returns:
        list: The same entries with each dict's ``path`` slashed; anything that
        is not a dict with a string path is returned unchanged, so it fails the
        comparison.
    """
    return [dict(item, path=code_review._slashed(item["path"]))
            if isinstance(item, dict) and isinstance(item.get("path"), str) else item
            for item in items]


@scenario("S53", assertions=S53_ASSERTIONS, run="R18", needs_backend=True)
def a_key_in_a_files_content_is_masked_and_the_file_reviewed(run):
    """Check that a single-line key in a file's content is masked, declared and never leaked.

    Args:
        run: R18's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S53_ASSERTIONS`, in that order.
    """
    texts = S53_ASSERTIONS
    if run is None:
        yield from _no_capture(texts, "R18")
        return
    code = run.output.exit_code
    yield code_review._verdict(texts, 0, "R18", code == EXIT_OK, "exited %d" % code)
    try:
        section = run.output.json().get(CODE_REVIEW_KEY)
    except ProductOutputError as exc:
        section = None
        detail = "stdout is not a JSON document: %s" % exc
    else:
        detail = "code_review is missing or not an object"
    if not isinstance(section, dict):
        yield code_review._finding(texts, 1, Outcome.FAIL, "R18", detail)
        yield code_review._finding(texts, 2, Outcome.FAIL, "R18", detail)
    else:
        yield _masked_finding(section)
        yield _skipped_finding(section)
    yield _leak_finding(texts, 3, "R18", run, (runs.REVIEW_MASKED_KEY,), "the toy key")


def _masked_finding(section):
    """Judge S53's second assertion: the keyed file is declared masked, exactly.

    Args:
        section: R18's ``code_review`` object.

    Returns:
        Finding: ``PASS`` when ``masked`` equals :data:`S53_EXPECTED_MASKED` and
        ``masked_unlisted`` is 0; ``FAIL`` otherwise.
    """
    items, unlisted, failure = _section_list(section, MASKED_KEY, MASKED_UNLISTED_KEY)
    if items is None:
        return code_review._finding(S53_ASSERTIONS, 1, Outcome.FAIL, "R18", failure)
    held = _normalized_masked(items) == S53_EXPECTED_MASKED and unlisted == 0
    return code_review._verdict(
        S53_ASSERTIONS, 1, "R18", held,
        "code_review.masked has %d entries and masked_unlisted is %r; expected exactly the "
        "keyed file and 0" % (len(items), unlisted))


def _skipped_finding(section):
    """Judge S53's third assertion: nothing was left out.

    Args:
        section: R18's ``code_review`` object.

    Returns:
        Finding: ``PASS`` when ``skipped`` is empty and ``skipped_unlisted`` is 0.
    """
    items, unlisted, failure = _section_list(section, SKIPPED_KEY, SKIPPED_UNLISTED_KEY)
    if items is None:
        return code_review._finding(S53_ASSERTIONS, 2, Outcome.FAIL, "R18", failure)
    return code_review._verdict(
        S53_ASSERTIONS, 2, "R18", not items and unlisted == 0,
        "code_review.skipped has %d entries and skipped_unlisted is %r; expected none"
        % (len(items), unlisted))


def _keyblock_listed_as_secret(stderr):
    """Whether the refusal leaves out exactly the key-block file as ``secret``.

    Args:
        stderr: The decoded stderr.

    Returns:
        bool: ``True`` when the omission list parses and, with ``\\`` read as
        ``/``, equals :data:`S54_EXPECTED_OMISSIONS`.
    """
    entries, _ = code_review._omissions(stderr)
    if entries is None:
        return False
    return [(code_review._slashed(path), cause) for path, cause in entries] == S54_EXPECTED_OMISSIONS


@scenario("S54", assertions=S54_ASSERTIONS, run="R19", inspects_timeouts=True)
def a_private_key_block_is_left_out_and_never_leaked(run):
    """Check that a private-key block file is left out as secret and its body never printed.

    Args:
        run: R19's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S54_ASSERTIONS`, in that order.
    """
    yield from code_review._refusal(
        S54_ASSERTIONS, "R19", run,
        lambda text: code_review.NOTHING_TO_REVIEW_TEXT in text and _keyblock_listed_as_secret(text),
        "stderr does not say there is nothing to review with %s (secret) left out"
        % runs.REVIEW_KEYBLOCK_FILE)
    if run is not None and not run.timed_out:
        yield _leak_finding(S54_ASSERTIONS, 2, "R19", run, runs.REVIEW_KEYBLOCK_BODY_LINES,
                            "a private-key body line")
