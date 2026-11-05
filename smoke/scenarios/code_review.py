# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""S47-S51 -- ``code-review`` against the release binary.

The suite proves each stage of a review with doubles. What it cannot see is the
release binary doing it end to end on an input that really splits into batches,
and refusing the inputs it must refuse before a model is paid for.

* **S47** (R12, the real trio): a generated tree that plans at least three
  batches yields ONE report. Its two property assertions read the JSON the
  product publishes (``code_review.findings`` and ``code_review.discarded``;
  the batch numbers inside them are 1-based, and S47 reads only the count); a missing key is a ``FAIL`` and never an
  empty list that would pass, while an empty list is ``CANNOT_TEST`` because a
  real trio cannot be forced to hallucinate a location or to have something
  discarded.
* **S48** (R13): a path outside the workspace is refused with exit 2.
* **S49** (R14): a directory holding only a binary and a file with a secret has
  nothing to review: exit 2, and the omission list names each file with its
  cause, ``binary`` and ``secret``, and nothing else.
* **S50** (R15): ``--timeout 20`` over the S47 tree is refused before any model
  call, naming the batches (at least :data:`MIN_REFUSED_BATCHES`), the floor
  and the ``--timeout`` that would do.
* **S51** (R16): reviewing the workspace ROOT, which holds ``.magi`` and
  ``.git`` beside a binary and a secret, is "nothing to review": no protected
  file was taken for review, and any protected path the refusal names is left
  out as ``ignored`` (never read and classified by its content). Under
  ``--timeout 20``, a protected file wrongly taken for review ends at the
  budget refusal, never at a model.

S48-S51 need no backend, so they declare ``inspects_timeouts`` and report a
timed-out run as ``FAIL`` themselves: a refusal reached before any model call
has no slow provider to blame for a hang. That no model was contacted is checked by the
executor, not here: R13-R16 run against a local request counter, and any request past the
capability probe fails the scenario before it is read (``runs.RunDefinition.forbids_model_calls``).
The Rust integration tests pin the same with a stand-in that records every body.
"""

import json
import re

from smoke import logs, runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of S47-S50.
S47_ASSERTIONS = (
    "exit 0",
    "stdout parses as JSON",
    "code_review.batches_planned is at least 3, batches_completed equals it "
    "and failed_batches is empty",
    "consult has one consensus",
    "every main finding points inside the reviewed files",
)
S48_ASSERTIONS = ("exit 2", "stderr names the outside path")
S49_ASSERTIONS = (
    "exit 2",
    "stderr says nothing to review, leaving out the binary as binary and the secret as secret",
)
S50_ASSERTIONS = (
    "exit 2",
    "stderr names the batches, the per-batch ceiling, the 15 s floor and a minimum --timeout",
    "the run log carries the same refusal",
)
S51_ASSERTIONS = (
    "exit 2",
    "stderr says nothing to review: no file under .magi/ or .git/ was taken for review",
    "a .magi/ file git lists is left out as ignored, and no .magi/ or .git/ path is read",
)

#: Exit codes of a successful run and of CLI misuse.
EXIT_OK = 0
EXIT_MISUSE = 2

#: Batches S47 requires, and the keys of the product's JSON it reads.
MIN_BATCHES = runs.REVIEW_MIN_BATCHES

#: Batches S50's refusal must name: the refusal fires at any batch count
#: (20 s is below what even one batch needs at the floor, see
#: ``runs.REVIEW_TOO_SHORT_TIMEOUT_S``), and a pool that
#: measured a window larger than the unmeasured default plans fewer than
#: :data:`MIN_BATCHES` over the same tree. Requiring three there would be a
#: FAIL for the environmental reason S47 reports as CANNOT_TEST; the tree's
#: size is already guarded when it is generated (``review_input_files``).
# MAGI FIX (S6 loop 5): S50 asked for three batches, false-failing under a
# measured window (F1-m1).
MIN_REFUSED_BATCHES = 1
CODE_REVIEW_KEY = "code_review"
CONSULT_KEY = "consult"
CONSENSUS_KEY = "consensus"

#: The members of magi-core's consensus report S47 requires as non-empty
#: strings: the classification label and the final verdict.
CONSENSUS_TEXT_KEYS = ("consensus", "consensus_verdict")

#: The files S47 generated, as the product names them (relative to the
#: workspace). The scenario compares findings against what it knows on its own
#: account, so the product's summary gains no field for the purpose.
S47_FILES = runs.REVIEW_GEN_PATHS

#: What R14's refusal must leave out: its binary as ``binary`` and its
#: key-shaped file as ``secret`` (``runs.review_skipped_files``, in that order).
S49_EXPECTED_OMISSIONS = tuple(
    (path, cause) for (path, _), cause
    in zip(runs.review_skipped_files(), ("binary", "secret")))

#: The product's wording that S48 and S49 look for in stderr. Every stderr
#: text S48-S51 parse (these two phrases, ``; left out: ``, ``, and N more``,
#: :data:`OMISSION_CAUSES` and :data:`BUDGET_PATTERN`) is pinned against
#: ``src/review/error.rs`` and ``src/review/filter.rs`` by ``StderrWordingTests``
#: in ``smoke/tests/test_rust_sync.py``.
# MAGI FIX (S6 loop 15): F1a2-m3 asked whether these pins exist; they do, and
# the comment now says where.
OUTSIDE_WORKSPACE_TEXT = "outside the workspace"
NOTHING_TO_REVIEW_TEXT = "nothing to review"

#: The directories a review never takes, whatever input names them (RV22).
PROTECTED_DIRS = (".magi", ".git")

#: The name prefix of the encrypted store and its ``-wal``/``-shm`` files,
#: which a review never takes either (``NEVER_REVIEWED`` in
#: ``src/review/source.rs``, pinned by ``smoke/tests/test_rust_sync.py``).
MEMORY_DB_PREFIX = ".magi-rs-memory.db"

#: What Windows drops from the end of a path component, so ``.git.`` and
#: ``.git `` name ``.git``; the product trims them before it compares.
TRAILING_COMPONENT_CHARS = ". "

#: The cause the product gives a protected path it leaves out without reading
#: it. Any other cause (``binary``, ``secret``, ``not utf-8``) means the file
#: was opened and classified by its content.
PROTECTED_CAUSE = "ignored"

#: The suffixes of SQLite's side files, which the store opens and removes as
#: it works: one git listed before R16 may be gone when the product lists the
#: tree, so S51 never requires one to be named.
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")

#: Every cause the product prints for a file it leaves out
#: (``SkipCause::as_str`` in ``src/review/filter.rs``). An entry is anchored on
#: these words, so a path may hold commas, semicolons or spaces.
#: ``missing`` (``filter::MISSING``) is deliberately absent: it is the cause of
#: the ``<path> is not reviewable: missing`` error for a file named explicitly,
#: never an entry of this list. A listed file that vanishes between the listing
#: and the read is an omission (``disappeared``), as is one that cannot be read
#: (``unreadable``) or has more than one hard link (``hard link``).
# MAGI FIX (S6 loop 12): the three causes R1 added were missing, and the comment
# still called a vanished listed file an I/O error.
OMISSION_CAUSES = ("binary", "not utf-8", "secret", "ignored",
                   "outside workspace", "too large", "not a regular file",
                   "hard link", "unreadable", "disappeared")

#: The omission list of a ``nothing to review`` refusal, to the end of its line.
OMISSION_LIST_PATTERN = re.compile(r"; left out: (?P<list>.*)")

#: The ``, and N more`` the product writes after the
#: ``MAX_OMISSIONS_SHOWN`` entries it names (``omissions`` in
#: ``src/review/error.rs``): the list was cut and the rest are not named.
OMISSION_CUT_PATTERN = re.compile(r", and (?P<more>\d+) more")

#: What may follow the last entry on the line: the ``, and N more`` the product
#: writes past its display cap, then the run log's ``key=value`` fields that the
#: stderr layer prints after the message (``review.error=nothing_to_review``).
OMISSION_TAIL_PATTERN = re.compile(
    r"(?:, and \d+ more)?(?:\s+[A-Za-z_][\w.]*=\S*)*\s*$")

#: One ``path (cause)`` entry of that list, followed by the separator or the end.
# MAGI FIX (S6 loop 4): the path was ``[^\s,;]+``, so a protected path holding a
# comma, semicolon or space lost its ``.magi`` component and passed (F1-1).
OMISSION_PATTERN = re.compile(
    r"(?P<path>.+?) \((?P<cause>%s)\)(?:, |$)"
    % "|".join(re.escape(cause) for cause in OMISSION_CAUSES))

#: A path holding `` (<cause>)`` followed by ``, `` or its end: in the
#: omission list it reads the same as an entry boundary, so the list cannot
#: say where such a path ends.
AMBIGUOUS_PATH_PATTERN = re.compile(
    r" \((?:%s)\)(?:, |$)"
    % "|".join(re.escape(cause) for cause in OMISSION_CAUSES))

#: The floor of a per-batch ceiling in seconds
#: (``AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`` = ``MIN_OPERATION_BUDGET`` 10 s +
#: ``MIN_CLIENT_TIMEOUT`` 5 s in ``src/magi/mod.rs``).
FLOOR_SECS = 15

#: The classifier call every consult budget reserves first, in seconds
#: (``CLASSIFY_TIMEOUT_SECS`` in ``src/magi/mod.rs``).
CLASSIFY_SECS = 6

#: The headless slack folded into the attempt factor, in percent
#: (``HEADLESS_TIMEOUT_SLACK_PCT`` in ``src/magi/mod.rs``).
SLACK_PCT = 20

#: What each whole attempt adds to a batch's need at the floor, in seconds:
#: the attempt factor is ``attempts x models x (100 + SLACK_PCT)`` hundredths
#: (``attempt_factor`` in ``src/magi/mod.rs``), so a need is always
#: ``CLASSIFY_SECS + k x`` this step for a whole ``k >= 1`` (15 x 1.2 = 18 s,
#: exact in integers because 15 x 120 is a multiple of 100).
NEED_STEP_SECS = FLOOR_SECS * (100 + SLACK_PCT) // 100

#: The least wall clock any batch needs at the floor: the most lenient
#: ``[magi]`` (no rotation, no retry) gives ``k = 1``, so 6 + 18 = 24 s.
MIN_NEED_PER_BATCH_SECS = CLASSIFY_SECS + NEED_STEP_SECS

#: The product's refusal, as ``ReviewError::Budget`` renders it
#: (``src/review/error.rs``): batches, wall clock per batch, the wall clock a
#: batch needs at the floor, the floor, the per-batch ceiling and the smallest
#: ``--timeout``. The count is rendered as ``N batches`` at every N, one
#: included (the template has no plural branch). The numbers come from the
#: product's formula and are checked for coherence by :func:`_budget_fields_ok`;
#: the fields are what RV-S7 requires.
BUDGET_PATTERN = re.compile(
    r"(?P<batches>\d+) batches leave (?P<wall>\d+) s per batch, less than the (?P<need>\d+) s a "
    r"batch needs at the (?P<floor>\d+) s floor \(per-batch ceiling "
    r"(?P<ceiling>\d+) s\); pass --timeout (?P<minimum>\d+) or more")


def _finding(texts, index, outcome, run_id, detail=""):
    """Build the finding for assertion *index* of *texts*.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        outcome: What became of it.
        run_id: The shared run it came from.
        detail: The cause when the outcome is not ``PASS``; on ``PASS``, an
            observation without a verdict, or empty.

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
    """One ``FAIL`` finding per assertion when a deterministic refusal hung.

    MAGI FIX (S6 loop 9): S48-S51 declare ``inspects_timeouts`` because their
    runs are refused before any model is reached, so a timeout there has no
    slow provider to blame: it is a hang in path validation, git handling or
    filtering, or a model call that should never have happened (F1a-c2). The
    executor builds a hung run with ``exit_code`` 0, so no assertion is judged
    on that synthetic exit.

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


def _stderr_text(run):
    """Decode a run's stderr leniently.

    Args:
        run: The run's ``RunResult``.

    Returns:
        str: The decoded text.
    """
    return run.output.stderr.decode("utf-8", errors="replace")


def _refusal(texts, run_id, run, stderr_ok, stderr_detail):
    """The two assertions shared by the refusal scenarios S48 and S49.

    Args:
        texts: The scenario's two assertion texts.
        run_id: The shared run.
        run: Its ``RunResult``, or ``None``.
        stderr_ok: Predicate over the decoded stderr.
        stderr_detail: The cause reported when it does not hold.

    Yields:
        Finding: ``exit 2`` first, then the stderr assertion.
    """
    if run is None:
        yield from _all_cannot_test(
            texts, run_id, "run %s produced no capture to inspect" % run_id)
        return
    if run.timed_out:
        yield from _all_hung(texts, run_id, run)
        return
    code = run.output.exit_code
    yield _verdict(texts, 0, run_id, code == EXIT_MISUSE,
                   "exited %d, expected %d" % (code, EXIT_MISUSE))
    yield _verdict(texts, 1, run_id, stderr_ok(_stderr_text(run)), stderr_detail)


def _list_under(section, key):
    """Read a list the product publishes under ``code_review``.

    Args:
        section: The ``code_review`` object.
        key: The list's key.

    Returns:
        tuple: ``(items, failure)``: the list, or ``None`` and why it is
        unusable. A missing key is a failure, never an empty list.
    """
    items = section.get(key)
    if not isinstance(items, list):
        return None, "code_review.%s is missing or not a list" % key
    return items, ""


def _is_count(value):
    """Tell whether *value* is a JSON integer.

    Args:
        value: A value read from the product's JSON.

    Returns:
        bool: ``True`` for an ``int`` that is not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


#: How a batch failure the backend caused begins, in magi-core's error text
#: (``MagiError``'s ``Display`` in magi-core 4.2.0 ``error.rs``) or the
#: product's own ``clock expired`` (``CLOCK_EXPIRED`` in
#: ``src/review/pipeline.rs``): transport, timeout, quota (an ``http error
#: 429``), auth, and a trio that lost its quorum to them. ``insufficient
#: agents`` names no seat cause, so it is counted here too: the product does not
#: carry the seats' causes in the batch's text, and a real trio cannot be forced
#: to answer. Matched as a prefix, so a wrapper such as ``the trio build did not
#: finish: timeout`` is not taken for one.
PROVIDER_FAILURE_PREFIXES = (
    "http error", "network error", "timeout", "auth error", "retry abandoned",
    "endpoint down", "external provider error", "insufficient agents",
    "clock expired",
)


#: The cause the product gives every batch from the one the clock cut to the
#: last (``CLOCK_EXPIRED`` in ``src/review/pipeline.rs``).
CLOCK_EXPIRED_CAUSE = "clock expired"


def _completed_by_metrics(metrics):
    """The 1-based batches whose metrics show a report arrived.

    A batch that failed keeps its time and no tokens (``spent`` in
    ``src/review/pipeline.rs``), so a batch with tokens completed. A batch
    with none may have completed too (a backend that does not count), so this
    is a subset of the completed batches, never the whole set.

    Args:
        metrics: ``code_review.batches``, or anything else when it is absent.

    Returns:
        set: The ``index`` of every entry with input or output tokens.

    Example:
        >>> sorted(_completed_by_metrics([{"index": 2, "input_tokens": 9}, {"index": 3}]))
        [2]
    """
    if not isinstance(metrics, list):
        return set()
    return {entry["index"] for entry in metrics
            if isinstance(entry, dict) and _is_count(entry.get("index"))
            and any(_is_count(entry.get(key)) and entry.get(key) > 0
                    for key in ("input_tokens", "output_tokens"))}


def _provider_failures_only(failed, planned, shortfall, metrics=None):
    """Whether *failed* names the *shortfall* missing batches, all lost to the backend.

    Args:
        failed: ``code_review.failed_batches``: ``[batch, why]`` pairs.
        planned: ``batches_planned``, at least one.
        shortfall: ``batches_planned - batches_completed``, at least one.
        metrics: ``code_review.batches``, read by :func:`_completed_by_metrics`.

    Returns:
        bool: ``True`` when *failed* is a list of exactly *shortfall* pairs,
        each with a distinct 1-based integer batch no larger than *planned*
        and a text reason that starts with one of
        :data:`PROVIDER_FAILURE_PREFIXES`; when the clock expired, its entries
        are the last ones and name every batch from the one it cut to
        *planned* (the batches run in series); and when no named batch shows
        a report in *metrics*. The completed batches are not published, so
        that is the whole correspondence the JSON can prove.

    Example:
        >>> _provider_failures_only([[3, "http error 429: quota"]], 5, 1)
        True
        >>> _provider_failures_only([[3, "response contract violated: x"]], 5, 1)
        False
        >>> _provider_failures_only([[3, "timeout: a"], [3, "timeout: b"]], 5, 2)
        False
    """
    # MAGI FIX (S6 loop 13): the batch ids were never read, so a repeated,
    # zero-based or out-of-range id, or a completed batch, passed (F1a-b1).
    if not isinstance(failed, list) or len(failed) != shortfall:
        return False
    batches, reasons = [], []
    for entry in failed:
        if not (isinstance(entry, list) and len(entry) == 2
                and _is_count(entry[0]) and 1 <= entry[0] <= planned
                and isinstance(entry[1], str)):
            return False
        reason = entry[1].strip().lower()
        if not reason.startswith(PROVIDER_FAILURE_PREFIXES):
            return False
        batches.append(entry[0])
        reasons.append(reason)
    if len(set(batches)) != len(batches):
        return False
    if CLOCK_EXPIRED_CAUSE in reasons:
        first = reasons.index(CLOCK_EXPIRED_CAUSE)
        if (any(reason != CLOCK_EXPIRED_CAUSE for reason in reasons[first:])
                or batches[first:] != list(range(batches[first], planned + 1))):
            return False
    return not set(batches) & _completed_by_metrics(metrics)


def _every_batch_lost_to_the_backend(section):
    """Whether the review lost every planned batch, each to a cause the backend gave.

    MAGI FIX (F1b2b): with no batch completed the product exits non-zero
    (``no batch completed`` in ``src/review/pipeline.rs``) and writes no
    consensus, so a backend that refused every batch read as two product
    ``FAIL``\\ s; D-17 makes a down backend ``CANNOT_TEST``.

    Args:
        section: The ``code_review`` object.

    Returns:
        bool: ``True`` when ``batches_planned`` is a positive integer,
        ``batches_completed`` is 0 and ``failed_batches`` names every planned
        batch with a backend cause (:func:`_provider_failures_only`).
    """
    planned = section.get("batches_planned")
    completed = section.get("batches_completed")
    return (_is_count(planned) and planned >= 1 and _is_count(completed)
            and completed == 0
            and _provider_failures_only(section.get("failed_batches"), planned,
                                        planned, section.get("batches")))


def _batches_finding(section, run_id):
    """Judge ``code_review.batches_planned`` and ``code_review.batches_completed``.

    Args:
        section: The ``code_review`` object.
        run_id: The shared run.

    Returns:
        Finding: ``FAIL`` when either count is missing or not an integer (a
        ``bool`` is not one), when ``batches_planned`` is below one (R12's
        files are not empty, so no window plans zero batches), or when
        ``batches_completed`` differs from
        ``batches_planned`` (one report over batches that never ran is not the
        multi-batch report RV-S2 asks for) unless every missing batch is in
        ``failed_batches`` with a cause the backend gave
        (:func:`_provider_failures_only`), which is ``CANNOT_TEST`` unless the
        planned count alone is a ``FAIL`` (:func:`_under_planned_finding`); ``FAIL``
        too when the counts agree but ``failed_batches`` is not an empty list.
        Otherwise ``PASS`` at three batches
        or more; at one or two, ``CANNOT_TEST`` when the whole pool measured a
        window (the input is sized for the unmeasured default, so fewer batches
        is the environment's doing), else ``FAIL``. An absent
        ``unmeasured_models`` counts as unmeasured.
    """
    planned = section.get("batches_planned")
    completed = section.get("batches_completed")
    # [MAGI §6 MS2 loop 8, F1a-c1] a missing or non-integer count is a contract defect, never the
    # CANNOT_TEST below: 2.0 == 2 and True == 1 would otherwise slip past the equality check.
    if not (_is_count(planned) and _is_count(completed)):
        return _finding(
            S47_ASSERTIONS, 2, Outcome.FAIL, run_id,
            "batches_planned is %r and batches_completed is %r, expected two integers"
            % (planned, completed))
    # MAGI FIX (S6 loop 9): 0 == 0 with every model measured read as
    # CANNOT_TEST, and negative equal counts too. R12 reviews non-empty files,
    # so fewer than one batch is a defect whatever the window (F1a-c3).
    if planned < 1:
        return _finding(
            S47_ASSERTIONS, 2, Outcome.FAIL, run_id,
            "batches_planned is %r, expected a positive count: R12 reviews "
            "non-empty files" % planned)
    failed = section.get("failed_batches")
    # MAGI FIX (S6 loop 12): a batch lost to the backend (quota, transport,
    # timeout, auth) read as a product FAIL, and equal counts beside a
    # declared failure passed "every one completed" (F1a-m1, F1a-b1, F1a-c3).
    if completed != planned:
        if _provider_failures_only(failed, planned, planned - completed,
                                   section.get("batches")):
            # MAGI FIX (S6 loop 14): the planned count is judgeable whether or
            # not every batch ran, and an under-planned run that also lost a
            # batch to the backend read CANNOT_TEST (F1a2-m3).
            under = _under_planned_finding(section, planned, run_id)
            if under is not None and under.outcome is Outcome.FAIL:
                return under
            return _finding(
                S47_ASSERTIONS, 2, Outcome.CANNOT_TEST, run_id,
                "batches_completed is %r of %r planned, and every missing "
                "batch failed for a reason the backend gave: %r"
                % (completed, planned, failed))
        return _finding(
            S47_ASSERTIONS, 2, Outcome.FAIL, run_id,
            "batches_completed is %r, expected every one of the %r planned; "
            "failed_batches is %r" % (completed, planned, failed))
    if failed != []:
        return _finding(
            S47_ASSERTIONS, 2, Outcome.FAIL, run_id,
            "failed_batches is %r, expected an empty list when every one of "
            "the %r planned batches completed" % (failed, planned))
    under = _under_planned_finding(section, planned, run_id)
    return under or _finding(S47_ASSERTIONS, 2, Outcome.PASS, run_id)


def _under_planned_finding(section, planned, run_id):
    """Judge a positive ``batches_planned`` against :data:`MIN_BATCHES`.

    Args:
        section: The ``code_review`` object, read for ``unmeasured_models``.
        planned: ``batches_planned``, a positive integer.
        run_id: The shared run.

    Returns:
        Finding | None: ``None`` at :data:`MIN_BATCHES` or more; below it,
        ``CANNOT_TEST`` when the whole pool measured a window (the input is
        sized for the unmeasured default, so fewer batches is the
        environment's doing), else ``FAIL``. An absent ``unmeasured_models``
        counts as unmeasured.
    """
    if planned >= MIN_BATCHES:
        return None
    detail = "batches_planned is %r, expected at least %d" % (planned, MIN_BATCHES)
    unmeasured = section.get("unmeasured_models")
    if isinstance(unmeasured, list) and not unmeasured:
        return _finding(
            S47_ASSERTIONS, 2, Outcome.CANNOT_TEST, run_id,
            detail + "; every model measured a window, larger than the one "
            "the input is sized for")
    return _finding(S47_ASSERTIONS, 2, Outcome.FAIL, run_id, detail)


def _locations_finding(section, run_id):
    """Judge that every located main finding points inside the generated files.

    Args:
        section: The ``code_review`` object.
        run_id: The shared run.

    Returns:
        Finding: ``FAIL`` on a missing list, an entry that is not an object, a
        ``line`` without a file, a file outside :data:`S47_FILES` or a
        ``line`` outside the generated file's lines; ``CANNOT_TEST`` when no
        finding carries a file; otherwise ``PASS``. A finding with neither a
        file nor a line is unlocated, not invented.
    """
    findings, failure = _list_under(section, "findings")
    if findings is None:
        return _finding(S47_ASSERTIONS, 4, Outcome.FAIL, run_id, failure)
    if not all(isinstance(f, dict) for f in findings):
        return _finding(S47_ASSERTIONS, 4, Outcome.FAIL, run_id,
                        "code_review.findings holds an entry that is not an object")
    # MAGI FIX (S6 loop 10): a line with no file was dropped from every check,
    # so a report of only such findings read as CANNOT_TEST (F1a-c1).
    fileless = [f.get("line") for f in findings
                if f.get("file") is None and f.get("line") is not None]
    if fileless:
        return _finding(S47_ASSERTIONS, 4, Outcome.FAIL, run_id,
                        "findings carry a line but no file: %s" % fileless)
    located = [f for f in findings if f.get("file") is not None]
    if not located:
        return _finding(S47_ASSERTIONS, 4, Outcome.CANNOT_TEST, run_id,
                        "the trio reported no located finding, so there is "
                        "no location to check")
    counts = runs.review_gen_line_counts()
    # MAGI FIX (S6 loop 10): S49 and S51 read a backslash as "/", S47 did
    # not, so a Windows rendering of a generated file false-FAILed (F1a-b1).
    outside = [f.get("file") for f in located
               if _slashed(f.get("file")) not in S47_FILES]
    if outside:
        return _finding(S47_ASSERTIONS, 4, Outcome.FAIL, run_id,
                        "findings point outside the reviewed files: %s" % outside)
    beyond = [(f["file"], f.get("line")) for f in located
              if not _line_inside(f.get("line"), counts[_slashed(f["file"])])]
    return _verdict(S47_ASSERTIONS, 4, run_id, not beyond,
                    "findings point outside the lines of their file: %s" % beyond)


def _slashed(path):
    """Read *path* with every backslash as ``/``, the separator S47-S51 compare.

    Args:
        path: A path as the product printed it; anything else is returned
            unchanged, so a non-string ``file`` stays outside every file.

    Returns:
        The path with ``/`` separators, or *path* itself when it is not a
        string.

    Example:
        >>> _slashed("gen\\\\a.rs"), _slashed(None)
        ('gen/a.rs', None)
    """
    return path.replace("\\", "/") if isinstance(path, str) else path


def _line_inside(line, count):
    """Whether a finding's ``line`` is absent or inside a file of *count* lines.

    Args:
        line: The ``line`` the product reported; ``None`` means unlocated.
        count: Lines in the file.

    Returns:
        bool: ``True`` for ``None`` or an integer in ``1..count``; a boolean
        or any non-integer is shape drift and is not inside.
    """
    if line is None:
        return True
    return isinstance(line, int) and not isinstance(line, bool) and 1 <= line <= count


def _discard_observation(section):
    """Describe the discarded findings, without a verdict.

    Whether anything is discarded depends on the trio inventing a location,
    which no input can force, so a verdict here was ``CANNOT_TEST`` on every
    honest run and kept the certificate from ever being issued (SMOKE #2,
    2026-11-05; user decision). That a discard names its cause is held by the
    unit suite (``src/review/validate.rs``,
    ``findings_outside_what_was_reviewed_are_discarded_with_their_cause``, and
    ``src/review/pipeline.rs``,
    ``a_discarded_finding_names_its_batch_one_based_in_the_json``); the real
    run's discards are still read and reported here, so a bare cause or a
    missing list reaches the reader.

    Args:
        section: The ``code_review`` object.

    Returns:
        str: Empty when the list is present and empty; otherwise one sentence
        naming the shape drift, or how many findings were discarded and how
        many of them carry no cause.

    Example:
        >>> _discard_observation({"discarded": []})
        ''
        >>> _discard_observation({"discarded": [{"cause": "context only"}]})
        '1 discarded finding(s), 0 without a cause'
    """
    discarded, failure = _list_under(section, "discarded")
    if discarded is None:
        return failure
    if not discarded:
        return ""
    bare = [d for d in discarded
            if not (isinstance(d, dict) and isinstance(d.get("cause"), str)
                    and d["cause"].strip())]
    return "%d discarded finding(s), %d without a cause" % (len(discarded), len(bare))


def _repeated_keys(stdout):
    """The keys that occur more than once in some object of *stdout*.

    ``json.loads`` keeps the last of a repeated key and drops the others
    without a word, so two reports under one key would read as one.

    Args:
        stdout: The raw standard output, already known to parse.

    Returns:
        set: Each key repeated inside one object, anywhere in the document.

    Example:
        >>> sorted(_repeated_keys(b'{"a": {"b": 1, "b": 2}, "c": 3}'))
        ['b']
    """
    repeated = set()

    def pairs(items):
        seen = set()
        for key, _ in items:
            if key in seen:
                repeated.add(key)
            seen.add(key)
        return dict(items)

    json.loads(stdout, object_pairs_hook=pairs)
    return repeated


def _consensus_ok(consult, stdout):
    """Whether *consult* carries exactly one consensus report of the right shape.

    Args:
        consult: The document's ``consult`` member.
        stdout: The raw standard output, to catch a repeated key.

    Returns:
        bool: ``True`` when ``consult.consensus`` is an object whose
        :data:`CONSENSUS_TEXT_KEYS` are non-empty strings, and no object of
        the document repeats ``consult`` or ``consensus``.
    """
    if not isinstance(consult, dict):
        return False
    report = consult.get(CONSENSUS_KEY)
    if not isinstance(report, dict):
        return False
    if not all(isinstance(report.get(key), str) and report[key].strip()
               for key in CONSENSUS_TEXT_KEYS):
        return False
    return not _repeated_keys(stdout) & {CONSULT_KEY, CONSENSUS_KEY}


@scenario("S47", assertions=S47_ASSERTIONS, run="R12", needs_backend=True)
def a_large_review_yields_one_report(run):
    """Check that a review over three or more batches yields one report.

    A timed-out R12 never reaches this body. The executor builds a hung run's
    output with ``exit_code`` 0, so asked directly the first assertion would
    pass on a hang; the runner answers first instead, because S47 does not
    declare ``inspects_timeouts``: the whole scenario is one ``CANNOT_TEST``
    (``runner.py``, SS5.1). That is pinned end to end by
    ``test_s47_never_judges_the_exit_of_a_timed_out_r12``.

    Every JSON key S47 reads is a copy of the product's, and each copy is
    pinned against the Rust source in ``smoke/tests/test_rust_sync.py``, so a
    rename fails the harness's own tests before SMOKE time (F1a-m1, S6 loop 9):

    * the ``code_review`` summary keys (``batches_planned``,
      ``batches_completed``, ``unmeasured_models``, ``findings``,
      ``discarded``): ``test_every_summary_key_s47_reads_is_one_the_report_pins``
      checks them against the key list ``src/review/report.rs`` pins;
    * ``file`` and ``line`` of a main finding and ``cause`` of a discarded one:
      ``test_the_entry_fields_s47_reads_are_serialized_under_their_own_names``
      (``ConsolidatedFinding`` and ``Discarded``, no serde rename);
    * ``consult.consensus`` and magi-core's ``consensus`` and
      ``consensus_verdict`` inside it (:data:`CONSENSUS_TEXT_KEYS`):
      ``test_the_consensus_s47_reads_is_the_one_the_envelope_writes`` checks
      that ``src/tools/code_review.rs`` writes the key through
      ``consensus_json`` and that ``src/tools/consult.rs`` serializes both
      fields of magi-core's ``ConsensusResult`` under those names, so a field
      renamed upstream stops the build there.

    Args:
        run: R12's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S47_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(S47_ASSERTIONS, "R12",
                                    "run R12 produced no capture to inspect")
        return
    code = run.output.exit_code
    try:
        document = run.output.json()
    except ProductOutputError as exc:
        yield _verdict(S47_ASSERTIONS, 0, "R12", code == EXIT_OK, "exited %d" % code)
        for index in range(1, len(S47_ASSERTIONS)):
            yield _finding(S47_ASSERTIONS, index, Outcome.FAIL, "R12",
                           "stdout is not a JSON document: %s" % exc)
        return
    # MAGI FIX (S6 loop 15): F1a2-m2/c1 read document.get as able to meet a
    # JSON array or scalar. It cannot: ProductOutput.json raises
    # ProductOutputError for any top level that is not an object, so that
    # stdout takes the per-assertion FAIL branch above (pinned by
    # test_s47_fails_on_a_json_document_that_is_not_an_object).
    section = document.get(CODE_REVIEW_KEY)
    if not isinstance(section, dict):
        section = {}
    lost = code != EXIT_OK and _every_batch_lost_to_the_backend(section)
    lost_detail = ("exited %d with every planned batch lost to the backend: %r"
                   % (code, section.get("failed_batches")))
    if lost:
        yield _finding(S47_ASSERTIONS, 0, Outcome.CANNOT_TEST, "R12", lost_detail)
    else:
        yield _verdict(S47_ASSERTIONS, 0, "R12", code == EXIT_OK, "exited %d" % code)
    # The discard observation rides on the parse assertion, the one PASS that
    # is certain here, so it never mixes into the cause of a red.
    yield _finding(S47_ASSERTIONS, 1, Outcome.PASS, "R12", _discard_observation(section))
    yield _batches_finding(section, "R12")
    consult = document.get(CONSULT_KEY)
    # MAGI FIX (S6 loop 6): any truthy value passed, a list of two reports
    # included, and a repeated key was dropped by the parser (F1a-c3).
    if lost:
        yield _finding(S47_ASSERTIONS, 3, Outcome.CANNOT_TEST, "R12",
                       "no consensus: " + lost_detail)
    else:
        yield _verdict(S47_ASSERTIONS, 3, "R12",
                       _consensus_ok(consult, run.output.stdout),
                       "consult.consensus is not one consensus report object")
    yield _locations_finding(section, "R12")


@scenario("S48", assertions=S48_ASSERTIONS, run="R13",
          inspects_timeouts=True)
def a_hostile_path_is_refused(run):
    """Check that a path outside the workspace is refused with exit 2.

    Both the product's fixed phrase and the hostile path's last component must
    appear: the component is the separator-independent part of the path the
    product prints, and it occurs in no fixed text.

    Args:
        run: R13's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S48_ASSERTIONS`, in that order.
    """
    yield from _refusal(S48_ASSERTIONS, "R13", run,
                        lambda text: OUTSIDE_WORKSPACE_TEXT in text
                        and runs.REVIEW_OUTSIDE_COMPONENT in text,
                        "stderr does not name %s as outside the workspace"
                        % runs.REVIEW_OUTSIDE_COMPONENT)


@scenario("S49", assertions=S49_ASSERTIONS, run="R14",
          inspects_timeouts=True)
def only_skipped_files_are_nothing_to_review(run):
    """Check that a directory of only skipped files is nothing to review.

    Args:
        run: R14's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S49_ASSERTIONS`, in that order.
    """
    # MAGI FIX (S6 loop 6): the phrase alone passed a refusal that left out
    # both files with the wrong causes, or none at all (F1a-c4).
    yield from _refusal(S49_ASSERTIONS, "R14", run,
                        lambda text: NOTHING_TO_REVIEW_TEXT in text
                        and _skipped_as_expected(text),
                        "stderr does not say there is nothing to review with "
                        "%s left out" % ", ".join(
                            "%s (%s)" % entry for entry in S49_EXPECTED_OMISSIONS))


@scenario("S50", assertions=S50_ASSERTIONS, run="R15",
          inspects_timeouts=True)
def an_insufficient_timeout_is_refused_before_spending(run):
    """Check that a too-short ``--timeout`` is refused in stderr and the run log.

    RV-S7: the refusal names the batches, the per-batch ceiling, the floor and
    the smallest ``--timeout``, on stderr and in the run log. The numbers come
    from the product's formula and must agree with each other and with the total
    the run asked for (:func:`_budget_fields_ok`). The run log is the review workspace's own, and its line
    must carry this run's id and the very text stderr showed. It is read from
    the copy the executor captured right after R15, never from the live
    directory, which a later review run rebuilds.

    Args:
        run: R15's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S50_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(
            S50_ASSERTIONS, "R15", "run R15 produced no capture to inspect")
        return
    if run.timed_out:
        yield from _all_hung(S50_ASSERTIONS, "R15", run)
        return
    code = run.output.exit_code
    yield _verdict(S50_ASSERTIONS, 0, "R15", code == EXIT_MISUSE,
                   "exited %d, expected %d" % (code, EXIT_MISUSE))
    match = BUDGET_PATTERN.search(_stderr_text(run))
    yield _verdict(S50_ASSERTIONS, 1, "R15", _budget_fields_ok(match),
                   "stderr does not carry a budget refusal whose batches, "
                   "wall clock, need, %d s floor, ceiling and minimum agree "
                   "with the %d s asked for"
                   % (FLOOR_SECS, runs.REVIEW_TOO_SHORT_TIMEOUT_S))
    yield _run_log_finding(run, match)


def _budget_fields_ok(match):
    """Whether a parsed refusal's numbers are ones the product prints together.

    ``pre_budget`` (``src/review/clock.rs``) splits the total over the batches
    with integer division, needs ``headless_consult_timeout_secs(floor)`` per
    batch, shows the RAW ceiling of the per-batch wall clock (never lifted to
    the floor, so a refused batch's ceiling is below it) and asks for batches x
    need. The checks hold under every ``[magi]`` config, so the harness need
    not know the rotation or retry settings.

    Args:
        match: The :data:`BUDGET_PATTERN` match, or ``None``.

    Returns:
        bool: ``True`` for at least :data:`MIN_REFUSED_BATCHES` batch; a wall
        clock of exactly the total asked for divided by the batches; a need of
        :data:`CLASSIFY_SECS` plus a whole, positive number of
        :data:`NEED_STEP_SECS` (so at least :data:`MIN_NEED_PER_BATCH_SECS`);
        the :data:`FLOOR_SECS`
        floor; a ceiling below the floor and not above the wall clock; and a
        minimum of exactly batches x need. That the need exceeds the wall clock
        and the minimum the total follows: the total (20 s) is below
        :data:`MIN_NEED_PER_BATCH_SECS`.

    Example:
        >>> ok = BUDGET_PATTERN.search(
        ...     "3 batches leave 6 s per batch, less than the 114 s a batch "
        ...     "needs at the 15 s floor (per-batch ceiling 0 s); "
        ...     "pass --timeout 342 or more")
        >>> _budget_fields_ok(ok)
        True
    """
    # MAGI FIX (S6 loop 7): the ceiling was captured but never checked (F1a-c3).
    # MAGI FIX (S6 loop 10): loop 7 required a ceiling of at least the floor,
    # which the product's raw ceiling never is in a refusal, and each number
    # was checked alone, so a 999 s ceiling beside a 21 s minimum passed
    # (F1a-c3).
    # MAGI FIX (S6 loop 15): F1a2-c2 noted that the need is checked only from
    # below. That is declared, not fixed: the exact need depends on the
    # environment's [magi] rotation and retry settings, which the harness does
    # not read; the formula is pinned in Rust (headless_consult_timeout_secs in
    # src/magi/mod.rs and pre_budget in src/review/clock.rs), and minimum ==
    # batches x need ties the advice to whatever need the product printed.
    # MAGI FIX (F1a2-g): "at least 24 s" still passed a need no config can
    # print (25 s). Every config's need is CLASSIFY_SECS + k x NEED_STEP_SECS
    # for a whole k >= 1, so the shape is checked without reading [magi].
    if match is None:
        return False
    batches, wall, need, floor, ceiling, minimum = (
        int(match.group(name)) for name in
        ("batches", "wall", "need", "floor", "ceiling", "minimum"))
    if batches < MIN_REFUSED_BATCHES:
        return False
    asked = runs.REVIEW_TOO_SHORT_TIMEOUT_S
    return (wall == asked // batches
            and need >= MIN_NEED_PER_BATCH_SECS
            and (need - CLASSIFY_SECS) % NEED_STEP_SECS == 0
            and floor == FLOOR_SECS
            and ceiling < FLOOR_SECS
            and ceiling <= wall
            and minimum == batches * need)


def _run_log_finding(run, match):
    """Judge that the run log holds the refusal stderr showed.

    Args:
        run: R15's ``RunResult``.
        match: The :data:`BUDGET_PATTERN` match on stderr, or ``None``.

    Returns:
        Finding: ``PASS`` when a line carries ``run=<id>`` and the refusal
        (the text stderr showed, when it matched); ``CANNOT_TEST`` when no
        readable line does and some log file could not be read; else
        ``FAIL``. A missing run
        id or log directory is a ``FAIL`` too: R15 is deterministic, needs no
        backend, and RV-S7 requires the refusal in the run log, so a product
        that refuses on stderr but never logs is the defect, not an
        inconclusive environment.
    """
    # MAGI FIX (S6 loop 3): a missing run id or log is FAIL, never CANNOT_TEST (F1).
    run_id = logs.resolve_id(run.output)
    if run_id is None:
        return _finding(S50_ASSERTIONS, 2, Outcome.FAIL, "R15",
                        "the run published no run id, so its log lines cannot be told apart")
    # MAGI FIX (S6 loop 3): the log captured at run time, not the live directory (F4).
    captured = run.review_log
    # MAGI FIX (S6 loop 7): the needle comes from logs.run_needle, which the
    # sync test ties to the product's RUN_FIELD (F1a-m1).
    needle = logs.run_needle(run_id)
    exact = match.group(0).encode("utf-8") if match else None

    def matcher(data):
        for line in data.split(b"\n"):
            if needle not in line:
                continue
            if exact is not None and exact in line:
                return True
            if exact is None and BUDGET_PATTERN.search(line.decode("utf-8", errors="replace")):
                return True
        return False

    if captured is None:
        return _finding(S50_ASSERTIONS, 2, Outcome.FAIL, "R15",
                        "the review workspace has no %s directory, so the refusal "
                        "was never logged" % "/".join(logs.LOG_DIR_PARTS))
    found = any(matcher(data) for data in captured)
    # MAGI FIX (S6 loop 12): a log file the executor could not open is the
    # environment refusing, as in logs.scan, never a product FAIL (F1a-c2).
    if not found and run.review_log_unreadable:
        return _finding(S50_ASSERTIONS, 2, Outcome.CANNOT_TEST, "R15",
                        "no readable line of the run log carries the refusal, "
                        "and these files could not be read: %s"
                        % ", ".join(run.review_log_unreadable))
    return _verdict(S50_ASSERTIONS, 2, "R15", found,
                    "no line of the run log carries run=%s and the budget refusal"
                    % run_id)


def _is_protected(path):
    """Whether *path* is one the product never reviews (``is_never_reviewed``).

    Components split on ``/`` and ``\\``, lose trailing dots and spaces, and
    compare ASCII-case-insensitively, as the product's do.

    Args:
        path: A path as the product printed it.

    Returns:
        bool: ``True`` when a component is one of :data:`PROTECTED_DIRS` or
        starts with :data:`MEMORY_DB_PREFIX`.

    Example:
        >>> _is_protected(".magi/magi.toml"), _is_protected("sub/.git/config")
        (True, True)
        >>> _is_protected(".magi-rs-memory.db-wal"), _is_protected(".GIT./x")
        (True, True)
        >>> _is_protected("my.magi/x")
        False
    """
    # MAGI FIX (S6 loop 14): the memory database prefix, and the case and
    # trailing dots the product folds, were missing, so such a path read for
    # its content passed (F1a2-m2).
    for part in re.split(r"[/\\]", path):
        name = part.rstrip(TRAILING_COMPONENT_CHARS).lower()
        if name in PROTECTED_DIRS or name.startswith(MEMORY_DB_PREFIX):
            return True
    return False


def _omissions(stderr):
    """Parse the omission list of a ``nothing to review`` refusal.

    The list is read entry by entry from its start, so every byte of it must
    belong to an entry: a rendering the harness no longer understands is a
    failure, never an empty list that would pass.

    A path may hold `` (<cause>)`` itself: an entry ends only at a cause
    followed by ``, `` or the end, so the shortest such split is taken and the
    rest stays in the path. The one rendering the product's text cannot tell
    apart is a path holding `` (<cause>), ``: it splits early. S51 does not
    judge a list over such a name (``CANNOT_TEST``, see
    :func:`_exclusion_finding`), unless the early split already names a
    protected path with a cause other than ``ignored``, which is a ``FAIL``
    and never a ``PASS``; S49's names are the harness's own.

    Args:
        stderr: The decoded stderr.

    Returns:
        tuple: ``(entries, failure)``: ``(path, cause)`` pairs in order, or
        ``None`` and why the list is unusable (absent, empty, or text left
        over).

    Example:
        >>> _omissions("x; left out: a, b (binary), c (secret)")
        ([('a, b', 'binary'), ('c', 'secret')], '')
        >>> _omissions("x; left out: a [binary]")[0] is None
        True
        >>> _omissions("x; left out: ")[0] is None
        True
        >>> _omissions("x; left out: foo (binary).rs (secret)")
        ([('foo (binary).rs', 'secret')], '')
    """
    listed = OMISSION_LIST_PATTERN.search(stderr)
    if listed is None:
        return None, "stderr carries no omission list to check"
    text = OMISSION_TAIL_PATTERN.sub("", listed.group("list"), count=1)
    entries, position = [], 0
    while position < len(text):
        entry = OMISSION_PATTERN.match(text, position)
        if entry is None:
            return None, "the omission list does not parse from %r" % text[position:]
        entries.append((entry.group("path"), entry.group("cause")))
        position = entry.end()
    # MAGI FIX (S6 loop 5): the product writes the marker only before at least
    # one entry, so an empty list is drift, never a vacuous pass (F1-m3, F1-c1).
    if not entries:
        return None, "the omission list after '; left out: ' is empty"
    return entries, ""


def _skipped_as_expected(stderr):
    """Whether the refusal leaves out exactly R14's files with their causes.

    Args:
        stderr: The decoded stderr.

    Returns:
        bool: ``True`` when the omission list parses and, with ``\\``
        separators read as ``/``, equals :data:`S49_EXPECTED_OMISSIONS` in any
        order.
    """
    entries, _ = _omissions(stderr)
    if entries is None:
        return False
    found = sorted((_slashed(path), cause) for path, cause in entries)
    return found == sorted(S49_EXPECTED_OMISSIONS)


def _protected_read(stderr):
    """The protected paths stderr names with a cause other than ``ignored``.

    Args:
        stderr: The decoded stderr.

    Returns:
        tuple: ``(read, failure)``: ``"path (cause)"`` for each protected path
        read for its content (empty when every one named, possibly none, was
        left out unread), or ``None`` and why the omission list is unusable.
    """
    # MAGI FIX (S6 loop 4): an unparseable list used to match nothing and pass
    # vacuously; R16's root always holds R14's skipped files, so it fails (F1-5).
    entries, failure = _omissions(stderr)
    if entries is None:
        return None, failure
    return ["%s (%s)" % (path, cause) for path, cause in entries
            if _is_protected(path) and cause != PROTECTED_CAUSE], ""


@scenario("S51", assertions=S51_ASSERTIONS, run="R16",
          inspects_timeouts=True)
def the_workspace_root_never_reviews_magi_or_git(run):
    """Check that a review of the workspace root never takes ``.magi`` or ``.git``.

    MAGI FIX (S6 loop 3): R12-R15 review subdirectories, so the workspace's
    ``.magi`` was never an enumeration candidate and RV22's exclusion went
    untested (F2). R16 reviews the root, which holds only the environment's
    ``.magi``, git's ``.git`` and R14's binary and secret: the product must have
    nothing to review. A protected file taken for review would make the input
    reviewable and end at the budget refusal of ``--timeout 20`` instead; one
    read and classified by its content would be named with that cause. The
    third assertion also needs one of the ``.magi/`` files git listed before
    the run (``RunResult.protected_visible``) named as ``ignored``, and is
    ``CANNOT_TEST`` when git listed none.

    What it observes is the product's REPORT, not its file accesses. A
    product that opened a protected file and then reported it as ``ignored``,
    or read ``.git/config`` and named it nowhere, would pass here: that no
    protected file is read is the product's contract, held in Rust:
    ``classify`` in ``src/review/filter.rs`` leaves a protected path out by
    its name before any read of its content, and the unit tests there
    (``ignored_files_and_never_reviewed_paths_are_left_out``) pin the cause.
    This scenario adds no instrumentation to the
    release binary. What S51 does establish is that the binary built for
    release takes nothing under ``.magi/`` or ``.git/`` for review (the input
    stays "nothing to review") and gives every protected path it names the
    cause of a path left out by name.

    Args:
        run: R16's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S51_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(
            S51_ASSERTIONS, "R16", "run R16 produced no capture to inspect")
        return
    if run.timed_out:
        yield from _all_hung(S51_ASSERTIONS, "R16", run)
        return
    code = run.output.exit_code
    yield _verdict(S51_ASSERTIONS, 0, "R16", code == EXIT_MISUSE,
                   "exited %d, expected %d" % (code, EXIT_MISUSE))
    stderr = _stderr_text(run)
    yield _verdict(S51_ASSERTIONS, 1, "R16", NOTHING_TO_REVIEW_TEXT in stderr,
                   "stderr does not say there is nothing to review, so a file "
                   "of the root was taken for review")
    yield _exclusion_finding(run, stderr)


def _exclusion_finding(run, stderr):
    """Judge S51's third assertion: a visible ``.magi/`` file left out unread.

    MAGI FIX (S6 loop 9): a refusal naming no protected path passed, so an
    ignore rule hiding ``.magi/`` from git (and from the product), or a
    product that dropped protected paths silently, passed without RV22's
    exclusion ever running (F1a-c1, F1a-m2).

    Args:
        run: R16's ``RunResult``; ``protected_visible`` holds the ``.magi/``
            paths git listed in the workspace before the run.
        stderr: Its decoded stderr.

    Returns:
        Finding: ``FAIL`` when the omission list is unusable or names a
        protected path with a cause other than ``ignored``; ``CANNOT_TEST``
        when git listed no ``.magi/`` file, so none was an enumeration
        candidate, or when one of their names holds `` (<cause>), `` or ends
        in `` (<cause>)`` (:data:`AMBIGUOUS_PATH_PATTERN`), so the list
        cannot be split around it (a split that already reads as a protected
        path with another cause is the ``FAIL`` above); ``CANNOT_TEST`` too when the list was
        cut at the product's display cap (:data:`OMISSION_CUT_PATTERN`)
        without naming every one of them as ``ignored``, so the rest may be in
        the unnamed tail, read and classified;
        ``FAIL`` when git listed some and the uncut list names none of
        them, or not every one apart from SQLite side files
        (:data:`SQLITE_SIDE_SUFFIXES`), as ``ignored``; else ``PASS``. Names are compared with ``\\``
        read as ``/``.
    """
    read, failure = _protected_read(stderr)
    if read is None:
        return _finding(S51_ASSERTIONS, 2, Outcome.FAIL, "R16", failure)
    if read:
        return _finding(S51_ASSERTIONS, 2, Outcome.FAIL, "R16",
                        "protected paths were read and classified: %s" % read)
    if not run.protected_visible:
        return _finding(
            S51_ASSERTIONS, 2, Outcome.CANNOT_TEST, "R16",
            "git lists no .magi/ file in the review workspace (an ignore rule "
            "hides it from the product too), so the exclusion was not exercised")
    # MAGI FIX (S6 loop 13): a .magi name holding " (<cause>), " split in two,
    # so the product reading it for its content passed as a protected path
    # left out plus another file (F1a-c3).
    # MAGI FIX (S6 loop 15): F1a2-m1 noted that such a name embedding
    # " (<cause>), " can reach the read FAIL above before this CANNOT_TEST,
    # even when the product left it out as ignored. That order is kept on
    # purpose: the outcome is a loud FAIL, never a PASS (pinned by
    # test_s51_fails_never_passes_over_an_ambiguous_name_the_product_left_out_as_ignored).
    ambiguous = [path for path in run.protected_visible
                 if AMBIGUOUS_PATH_PATTERN.search(path)]
    if ambiguous:
        return _finding(
            S51_ASSERTIONS, 2, Outcome.CANNOT_TEST, "R16",
            "git lists %s, whose name the omission list cannot tell from an "
            "entry boundary, so what the product did with it is unreadable"
            % ", ".join(ambiguous))
    entries, _ = _omissions(stderr)
    named = {_slashed(path) for path, cause in entries
             if cause == PROTECTED_CAUSE}
    # MAGI FIX (S6 loop 14): only the omission list was read with "\\" as "/",
    # so a backslash-separated listing never matched (F1a2-c2).
    visible = {_slashed(path) for path in run.protected_visible}
    found = bool(named & visible)
    # MAGI FIX (S6 loop 14): the product names at most MAX_OMISSIONS_SHOWN
    # entries, so with the list cut every listed .magi/ file could sit in the
    # unnamed rest and S51 read the environment's size as a FAIL (F1a1-m2).
    # MAGI FIX (F1a2): one named file is not enough either, since another
    # could sit in the unnamed rest read and classified; a cut list passes
    # only when it names every listed .magi/ file as ignored.
    cut = OMISSION_CUT_PATTERN.search(
        OMISSION_LIST_PATTERN.search(stderr).group("list"))
    if cut is not None and not visible <= named:
        return _finding(
            S51_ASSERTIONS, 2, Outcome.CANNOT_TEST, "R16",
            "git lists %s, the refusal names %d of them as ignored and cut "
            "its list at %s more, so the rest may be among the unnamed"
            % (", ".join(run.protected_visible), len(named & visible),
               cut.group("more")))
    # MAGI FIX (F1a2-h): an uncut list names every entry the product left
    # out, so one visible .magi/ file absent from it was taken elsewhere; the
    # intersection alone passed a list naming one of three.
    stable = {path for path in visible
              if not path.endswith(SQLITE_SIDE_SUFFIXES)}
    missing = sorted(stable - named)
    return _verdict(S51_ASSERTIONS, 2, "R16", found and not missing,
                    "git lists %s, and stderr does not name %s as ignored"
                    % (", ".join(run.protected_visible),
                       ", ".join(missing) if found else "any of them"))
