# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""S52, S55-S58 -- ``code-review`` retrieves its documentation and history, or says it could not.

* **S52** (R17, the real trio): the review workspace's ``[embedding]`` points at
  an endpoint that answers an HTTP error at once
  (``runs.embedder_pointed_away``), so documentation and history cannot be
  retrieved. The review must still run to its report, exit 0, and say
  so: ``code_review.without_embedder`` is ``true`` and ``code_review.trimmed``
  names the cause (an unavailable index, an embedder that did not answer),
  RV07's declaration of what was left out.

The unit suite proves each half with doubles (``src/memory/review_rag.rs``:
``a_failing_embedder_degrades_to_no_documentation_or_history``,
``a_failing_reindex_degrades_and_never_fails_the_review``); what it cannot see is
the release binary meeting a failing embedder endpoint and
still delivering a report. A timed-out R17 never reaches this body: S52 does not
declare ``inspects_timeouts``, so the runner reports it as ``CANNOT_TEST``.

* **S55** (R18, the real trio and the environment's own embedder, the one S9
  needs answering): the same machinery with nothing pointed away must not
  degrade. ``code_review.without_embedder`` is ``false`` and ``code_review.trimmed``
  names neither an unavailable documentation index nor an embedder that did not
  answer. Without it the suite could not tell a working RAG from one that
  degrades on every run: S52 even expects the degradation. Whether a chunk was
  relevant is the model side and is never asserted; a trim for room is the
  budget at work and passes. R18 does call the embedder: the file it reviews
  carries a ``///`` doc comment, a documentation source of its own, so the
  documentation index is built and each batch's query embedded; its third
  assertion reads the relay's record (``runs.recording_relay``) for both.

* **S56** (R20, RV21, RV-S36): a commit review of a two-commit history whose
  working-tree copy of the file is left dirty. The embedder is sent the
  commit's doc comment as a chunk, the trio its code, and no request (body,
  target or header) carries the dirty one.

* **S57** (R21, RV08): a buggy commit is reviewed, every line of it fixed in
  the next, reviewed again (the reclassification marks the first entries
  attended, after its report), and reviewed a third time, whose trio request
  must carry a ``kind=history`` item on the file, with a first-review finding's
  title and ``(attended)``. Only that third review is relayed: the claim is
  about what it sends. With no located finding in the first review (the trio's
  answer) the second assertion is ``CANNOT_TEST``.

* **S58** (R22, RV07, RV-S8): R21's seeding against the live embedder, then the
  measured review with ``[embedding]`` on a served error: it still delivers its
  findings and declares, for the history layer, that the embedder did not
  answer. Not relayed: the declaration is read, not the absence of egress.

MAGI FIX (MS3 §6 loop9): every review here was of a working tree and read its
own flags; R18, R20 and R21 send every request through the recording relay, so
these observe the bytes that left the process. S52 still observes priorities
(3) and (4) only through their degraded declaration.
"""

import json

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.scenarios import code_review

#: The verbatim assertion texts of S52.
S52_ASSERTIONS = (
    "exit 0",
    "code_review.without_embedder is true",
    "code_review.trimmed names an unavailable index or an embedder that did not answer",
)

#: Exit code of a review that produced its report.
EXIT_OK = 0

#: The keys of the product's JSON S52 reads.
CODE_REVIEW_KEY = "code_review"
WITHOUT_EMBEDDER_KEY = "without_embedder"

#: The run S52 reads.
RUN_ID = "R17"

#: The key of the context the review left out, and of each entry's cause.
TRIMMED_KEY = "trimmed"
TRIMMED_LABEL_KEY = "label"

#: The ``trimmed`` labels of a degraded RAG (``INDEX_UNAVAILABLE`` and
#: ``EMBEDDER_DOWN`` in ``src/memory/review_rag.rs``); a trim for room is not one.
DEGRADED_LABELS = ("index unavailable", "the embedder did not answer")


def _finding(index, outcome, detail=""):
    """Build the finding for assertion *index* of :data:`S52_ASSERTIONS` (the shared builder).

    Args:
        index: Position in :data:`S52_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return code_review.finding(S52_ASSERTIONS, index, outcome, RUN_ID, detail)


def _verdict(index, held, detail):
    """Turn a boolean check into a ``PASS`` or ``FAIL`` finding of S52 (the shared builder).

    Args:
        index: Position in :data:`S52_ASSERTIONS`.
        held: Whether the property held.
        detail: The cause reported when it did not.

    Returns:
        Finding: ``PASS`` when *held*, otherwise ``FAIL`` carrying *detail*.

    Example:
        >>> _verdict(0, True, "x").outcome.value
        'PASS'
    """
    return code_review.verdict(S52_ASSERTIONS, index, RUN_ID, held, detail)


@scenario("S52", assertions=S52_ASSERTIONS, run=RUN_ID, needs_backend=True)
def a_review_without_an_embedder_runs_and_says_so(run):
    """Check that a review whose embedder is unreachable still reports, and declares it.

    The declaration must be the boolean ``true``: a missing key, or any other
    value, is shape drift and a ``FAIL``, never a pass by absence.

    Args:
        run: R17's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S52_ASSERTIONS`, in that order.
    """
    # MAGI FIX (MS3 §6 loop12): a result with no output is a missing capture, never an
    # AttributeError (S57's guard, now S52's, S53's, S55's and S56's too).
    if run is None or run.output is None:
        for index in range(len(S52_ASSERTIONS)):
            yield _finding(index, Outcome.CANNOT_TEST,
                           "run %s produced no capture to inspect" % RUN_ID)
        return
    code = run.output.exit_code
    yield _verdict(0, code == EXIT_OK, "exited %d" % code)
    # A root that is not an object (`null`, a list) never reaches `.get`: `ProductOutput.json`
    # raises ProductOutputError for it, a FAIL below (`test_s52_fails_on_valid_json_that_is_not_an_object`,
    # `test_s55_fails_on_valid_json_that_is_not_an_object`).
    try:
        report = run.output.json()
    except ProductOutputError as exc:
        for index in range(1, len(S52_ASSERTIONS)):
            yield _finding(index, Outcome.FAIL, "stdout is not a JSON document: %s" % exc)
        return
    summary = report.get(CODE_REVIEW_KEY)
    summary = summary if isinstance(summary, dict) else {}
    value = summary.get(WITHOUT_EMBEDDER_KEY)
    yield _verdict(1, value is True,
                   "code_review.without_embedder is %r, not true" % (value,))
    # MAGI FIX (S6 MS3 loop3): the boolean alone was read; RV07 declares the cause in `trimmed`.
    degraded = _degraded_labels(summary.get(TRIMMED_KEY))
    if degraded is None:
        yield _finding(2, Outcome.FAIL, "code_review.trimmed is missing or not a list of entries")
    else:
        yield _verdict(2, bool(degraded), "code_review.trimmed names no degraded cause")


#: The verbatim assertion texts of S55.
S55_ASSERTIONS = (
    "code_review.without_embedder is false: the embedder answered the review",
    "code_review.trimmed names no unavailable index and no embedder that did not answer",
    "the embedder was sent the file's doc comment alone, as a chunk, and a query of the batch",
)

#: The path of an embedding request, and the key of its texts (OpenAI-compatible).
EMBEDDINGS_PATH = "/embeddings"
EMBEDDING_INPUT_KEY = "input"


def _embedded_texts(bodies):
    """Every text the relay saw sent to the embedder.

    Args:
        bodies: The run's ``(path, body)`` record.

    Returns:
        list: The ``input`` strings of each ``/embeddings`` request whose body
        parses; anything else is no embedding request.
    """
    texts = []
    for path, body in bodies:
        # MAGI FIX (MS3 §6 loop12): the relay records the target with its query (`…?x=1`); only
        # the path decides, as in `runs.trio_bodies`.
        if not path.split("?", 1)[0].endswith(EMBEDDINGS_PATH):
            continue
        try:
            items = json.loads(body).get(EMBEDDING_INPUT_KEY)
        except (ValueError, AttributeError):
            continue
        # The OpenAI-compatible `input` is a string or a list of strings.
        items = [items] if isinstance(items, str) else items
        texts += [t for t in items if isinstance(t, str)] if isinstance(items, list) else []
    return texts


def _chunk_sent(bodies, doc, code):
    """Whether *doc* was embedded alone (a documentation chunk), without *code*.

    Args:
        bodies: The run's ``(path, body)`` record.
        doc: The doc comment's text.
        code: Text only the code carries, never the chunk.

    Returns:
        bool: Whether some embedded text holds *doc* and not *code*.
    """
    return any(doc in t and code not in t for t in _embedded_texts(bodies))


#: The run S55 reads: a review with the environment's own embedder.
S55_RUN_ID = "R18"


def _s55_finding(index, outcome, detail=""):
    """Build the finding for assertion *index* of :data:`S55_ASSERTIONS` (the shared builder).

    Args:
        index: Position in :data:`S55_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return code_review.finding(S55_ASSERTIONS, index, outcome, S55_RUN_ID, detail)


def _degraded_labels(trimmed):
    """The degraded causes *trimmed* names, or ``None`` when it is not a list of entries.

    Args:
        trimmed: The ``code_review.trimmed`` value read from the product.

    Returns:
        list: Each :data:`DEGRADED_LABELS` label found, in order; ``None`` for a
        shape the harness does not understand, an entry without a string
        ``label`` included (a missing label is shape drift, never a pass).

    Example:
        >>> _degraded_labels([{"kind": "history", "label": "no room left in the batch"}])
        []
        >>> _degraded_labels("x") is None
        True
        >>> _degraded_labels([{"kind": "history"}]) is None
        True
    """
    # MAGI FIX (S6 MS3 loop3): an entry without a string label was skipped, not refused.
    if not isinstance(trimmed, list) or not all(
            isinstance(t, dict) and isinstance(t.get(TRIMMED_LABEL_KEY), str) for t in trimmed):
        return None
    return [t.get(TRIMMED_LABEL_KEY) for t in trimmed
            if t.get(TRIMMED_LABEL_KEY) in DEGRADED_LABELS]


#: Said beside a failed "doc comment indexed" check (MAGI FIX, MS3 §6 loop14): the check holds
#: only while the environment's database does not already carry R18's doc chunk.
S55_PRECONDITION = (" (precondition: the environment's database must not already hold this doc chunk, "
                    "which makes the product skip its embedding; run --reset-env and retry)")


@scenario("S55", assertions=S55_ASSERTIONS, run=S55_RUN_ID, needs_backend=True)
def a_review_with_its_embedder_up_is_not_degraded(run):
    """Check that a review whose embedder answers retrieves without degrading.

    Both declarations must be present and well typed: a missing key is shape
    drift and a ``FAIL``, never a pass by absence.

    Declared preconditions (MAGI FIX, MS3 §6 loop13): the third assertion
    expects R18's doc chunk to be embedded, which holds while the environment's
    documentation index does not already carry it. The review workspace is
    rebuilt on every run (``smoke/scratch/review-ws``) with a fresh copy of the
    environment's ``.magi``, and nothing in the environment indexes R18's doc
    comment, so the chunk is new each time; an environment whose database
    already held it would make the product skip the embedding, and the
    assertion would read that as a defect (a ``--reset-env`` clears it).

    Args:
        run: R18's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S55_ASSERTIONS`, in that order.
    """
    # MAGI FIX (S6 MS3 loop2): no scenario observed a working RAG; S52 only its degradation.
    if run is None or run.output is None:
        for index in range(len(S55_ASSERTIONS)):
            yield _s55_finding(index, Outcome.CANNOT_TEST,
                               "run %s produced no capture to inspect" % S55_RUN_ID)
        return
    try:
        # `.json()` raises for a root that is not an object, so `.get` is always on a dict.
        summary = run.output.json().get(CODE_REVIEW_KEY)
    except ProductOutputError as exc:
        for index in range(len(S55_ASSERTIONS)):
            yield _s55_finding(index, Outcome.FAIL, "stdout is not a JSON document: %s" % exc)
        return
    summary = summary if isinstance(summary, dict) else {}
    value = summary.get(WITHOUT_EMBEDDER_KEY)
    held = value is False
    yield _s55_finding(0, Outcome.PASS if held else Outcome.FAIL,
                       "" if held else "code_review.without_embedder is %r, not false" % (value,))
    degraded = _degraded_labels(summary.get(TRIMMED_KEY))
    if degraded is None:
        yield _s55_finding(1, Outcome.FAIL, "code_review.trimmed is missing or not a list of entries")
    elif degraded:
        yield _s55_finding(1, Outcome.FAIL, "code_review.trimmed declares %s" % ", ".join(degraded))
    else:
        yield _s55_finding(1, Outcome.PASS)
    # MAGI FIX (MS3 §6 loop9): the flags alone passed a RAG that indexed and queried nothing.
    if run.request_bodies is None:
        yield _s55_finding(2, Outcome.CANNOT_TEST, "the relay's record is missing or incomplete")
        return
    doc, code = runs.REVIEW_MASKED_DOC.decode(), _TOKEN_CODE
    indexed = _chunk_sent(run.request_bodies, doc, code)
    queried = any(code in t for t in _embedded_texts(run.request_bodies))
    # MAGI FIX (MS3 §6 loop14, R7a2): the precondition travels with the failure, so a red S55 on a
    # used environment names its own escape instead of reading as a product defect.
    yield _s55_finding(2, Outcome.PASS if indexed and queried else Outcome.FAIL,
                       "" if indexed and queried else
                       "doc comment indexed: %s; query embedded: %s%s" % (
                           indexed, queried, "" if indexed else S55_PRECONDITION))


#: Code only R18's batch carries, never its doc comment's chunk.
_TOKEN_CODE = "pub fn token"

#: S56 (R20, RV21, RV-S36): a commit review reads its documentation from that
#: commit, never from the dirty working tree.
S56_ASSERTIONS = (
    "exit 0",
    "the embedder was sent the reviewed commit's doc comment as a chunk, and the trio its code",
    "no request the relay forwarded carries the working tree's doc comment, and no embedded "
    "chunk is the parent commit's doc comment alone",
)

#: S57 (R21, RV08): history validated across revisions reaches the trio.
S57_ASSERTIONS = (
    "every review exits 0",
    "the last review sends the trio a validated history item on the reviewed file",
)

#: Code only R20's batch carries, never its doc comment's chunk.
_ANSWER_CODE = "pub fn answer"


@scenario("S56", assertions=S56_ASSERTIONS, run="R20", needs_backend=True)
def documentation_comes_from_the_reviewed_commit(run):
    """Check that a commit review indexes that commit's doc comment, never the working tree's.

    Args:
        run: R20's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S56_ASSERTIONS`, in that order.
    """
    texts = S56_ASSERTIONS
    if run is None or run.output is None:
        yield from code_review.all_cannot_test(texts, "R20", "run R20 produced no capture")
        return
    code = run.output.exit_code
    yield code_review.verdict(texts, 0, "R20", code == EXIT_OK, "exited %d" % code)
    if run.request_bodies is None:
        yield from code_review.all_cannot_test(texts[1:], "R20", "the relay's record is missing or incomplete")
        return
    # MAGI FIX (MS3 §6 loop15, R7a2 Melchior): the reviewed commit's CODE must reach the trio too,
    # not only its documentation: a review of the wrong revision's code passed with the right docs.
    code_to_trio = any(_ANSWER_CODE.encode() in body and runs.REVIEW_COMMITTED_DOC in body
                       for body in runs.trio_bodies(run.request_bodies))
    yield code_review.verdict(texts, 1, "R20",
                  _chunk_sent(run.request_bodies, runs.REVIEW_COMMITTED_DOC.decode(), _ANSWER_CODE)
                  and code_to_trio,
                  "no embedded text is the committed doc comment alone, or no trio request "
                  "carries the commit's code")
    # MAGI FIX (MS3 §6 loop12): RV21/RV22 forbid another revision's DOCUMENTATION, which reaches a
    # model as an embedded chunk. The parent's doc comment may appear in a trio prompt as the
    # REMOVED lines of the very diff under review (R20's fixture: parent commit, then HEAD), so
    # it is looked for in embedding requests only; the dirty working-tree copy is in no diff.
    if run.request_headers is None:
        # MAGI FIX (MS3 §6 loop16, R7a2): a missing header record is no header free of the doc.
        yield Finding(assertion=texts[2], outcome=Outcome.CANNOT_TEST, run_id="R20",
                      detail="the relay's header record is missing")
        return
    dirty = _dirty_doc_sent(run)
    # MAGI FIX (MS3 §6 loop14, R7a2): the retrieval query is an embedding request too, built from
    # the batch's raw hunk, so it carries the parent's removed doc line beside the code. Only a
    # chunk (the doc comment without the batch's code) is the parent's documentation.
    parent = _chunk_sent(run.request_bodies, runs.REVIEW_PARENT_DOC.decode(), _ANSWER_CODE)
    yield code_review.verdict(texts, 2, "R20", not dirty and not parent,
                  "the working tree's doc comment went to %s, or the parent's was embedded: %s"
                  % (dirty, parent))


@scenario("S57", assertions=S57_ASSERTIONS, run="R21", needs_backend=True)
def validated_history_reaches_the_trio(run):
    """Check that a finding attended in a later commit is served to the trio as history.

    Whether the first review reports a finding on the file is the trio's answer:
    with none there is no history to serve, which is ``CANNOT_TEST`` for the
    second assertion, never a ``PASS``.

    Args:
        run: R21's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S57_ASSERTIONS`, in that order.
    """
    texts = S57_ASSERTIONS
    # MAGI FIX (MS3 §6 loop10): a measured review that left no output crashed the scenario.
    if run is None or run.output is None or len(run.setup_outputs) != 2:
        yield from code_review.all_cannot_test(texts, "R21", "run R21 did not run its three reviews")
        return
    codes = [o.exit_code for o in run.setup_outputs + (run.output,)]
    yield code_review.verdict(texts, 0, "R21", codes == [EXIT_OK] * 3, "the reviews exited %s" % codes)
    try:
        findings = run.setup_outputs[0].json().get(CODE_REVIEW_KEY, {}).get("findings")
    except (ProductOutputError, AttributeError):
        findings = None
    # MAGI FIX (MS3 §6 loop14, R7a2): a `findings` that is not a list is no finding, not a crash.
    findings = findings if isinstance(findings, list) else None
    # MAGI FIX (MS3 §6 loop15, R7a2): a located finding has an integer line (a bool is no line).
    located = [f for f in findings or () if isinstance(f, dict)
               and isinstance(f.get("line"), int) and not isinstance(f.get("line"), bool)
               and isinstance(f.get("file"), str)
               # MAGI FIX (MS3 §6 loop16, R7a1): a line outside the seeded file is a model
               # answer the fixture cannot turn into history: no precondition, CANNOT_TEST.
               and 1 <= f["line"] <= runs.REVIEW_HISTORY_BUGGY_LINES
               and f["file"].replace("\\", "/") == runs.REVIEW_HISTORY_FILE]
    if not located:
        yield Finding(assertion=texts[1], outcome=Outcome.CANNOT_TEST, run_id="R21",
                      detail="the first review reported no located finding on %s, so no history "
                             "was recorded to serve" % runs.REVIEW_HISTORY_FILE)
    elif run.request_bodies is None:
        yield Finding(assertion=texts[1], outcome=Outcome.CANNOT_TEST, run_id="R21",
                      detail="the relay's record is missing or incomplete")
    else:
        yield code_review.verdict(texts, 1, "R21", _history_served(run.request_bodies, located),
                      "no trio prompt carries an attended history item of the first review's "
                      "findings on %s" % runs.REVIEW_HISTORY_FILE)


#: S58 (R22, RV07, RV-S8): a review with seeded history and an unreachable embedder.
S58_ASSERTIONS = (
    "the setup reviews and the measured review all exit 0",
    "the measured review declares without_embedder true and its history layer names a degraded cause",
    "the measured review still delivers its findings list",
)

#: The ``trimmed`` kind of the history layer (``ContextKind::History`` in ``src/review/``).
_HISTORY_KIND = "history"


@scenario("S58", assertions=S58_ASSERTIONS, run="R22", needs_backend=True)
def a_review_with_history_and_a_down_embedder_declares_it(run):
    """Check that history which exists but cannot be retrieved is declared, and the report still comes.

    R22 seeds history exactly as R21 does, against the live embedder, then runs the
    measured review with ``[embedding]`` on a served error. The product must still
    deliver its findings and say, for the history layer, that the embedder did not
    answer (RV07, RV-S8). Whether the trio finds anything is its own answer: the
    ``findings`` list is asserted to exist, never to be non-empty.

    Args:
        run: R22's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S58_ASSERTIONS`, in that order.
    """
    texts = S58_ASSERTIONS
    if run is None or run.output is None or len(run.setup_outputs) != 2:
        yield from code_review.all_cannot_test(texts, "R22", "run R22 did not run its three reviews")
        return
    codes = [o.exit_code for o in run.setup_outputs + (run.output,)]
    yield code_review.verdict(texts, 0, "R22", codes == [EXIT_OK] * 3, "the reviews exited %s" % codes)
    try:
        report = run.output.json()
    except ProductOutputError as exc:
        for index in (1, 2):
            yield code_review.finding(texts, index, Outcome.FAIL, "R22", "stdout is not a JSON document: %s" % exc)
        return
    summary = report.get(CODE_REVIEW_KEY)
    summary = summary if isinstance(summary, dict) else {}
    trimmed = summary.get(TRIMMED_KEY)
    history_degraded = (isinstance(trimmed, list)
                        and any(isinstance(t, dict) and t.get("kind") == _HISTORY_KIND
                                and t.get(TRIMMED_LABEL_KEY) in DEGRADED_LABELS for t in trimmed))
    yield code_review.verdict(texts, 1, "R22",
                              summary.get(WITHOUT_EMBEDDER_KEY) is True and history_degraded,
                              "without_embedder is %r and trimmed is %r"
                              % (summary.get(WITHOUT_EMBEDDER_KEY), trimmed))
    yield code_review.verdict(texts, 2, "R22", isinstance(summary.get("findings"), list),
                              "code_review.findings is %r, not a list" % (summary.get("findings"),))


def _history_served(request_bodies, located) -> bool:
    """Whether a trio prompt carries a validated history item of *located* (MAGI FIX, MS3 §6 loop10).

    Only the measured (third) review goes through the relay; the two setup
    reviews that record and reclassify the history run before it, so this reads
    their effect on that review, never their own requests (R7a Caspar). The item
    must sit in a trio request, after its frame, with the title of a finding
    the first review located there and the ``(attended)`` status the second
    review's reclassification gave it (``history_item`` in
    ``src/memory/review_rag.rs``); titles are compared JSON-escaped, as sent.

    Coupling, declared (MAGI FIX, MS3 §6 loop11): this reads the product's
    prompt framing byte for byte (the ``<<<context kind=history file=…>>>``
    frame of ``context_frame`` in ``src/review/pipeline.rs`` and the
    ``title (status)`` line of ``history_item``). A change to either format
    turns S57 red while the feature may still work: update this probe with it.

    Args:
        request_bodies: The relay's ``(target, body)`` record.
        located: The first review's findings on :data:`runs.REVIEW_HISTORY_FILE`.

    Returns:
        bool: Whether one such item reached the trio.
    """
    # MAGI FIX (MS3 §6 loop13, R7a): the product's JSON keeps non-ASCII characters (serde_json
    # escapes none), so the titles are escaped without ensure_ascii and sent as UTF-8; the path
    # may be written with either separator, bare or JSON-escaped.
    path = runs.REVIEW_HISTORY_FILE
    frames = [b"<<<context kind=history file=" + shown.encode("utf-8") + b">>>"
              for shown in (path, path.replace("/", "\\"), path.replace("/", "\\\\"))]
    titles = [json.dumps(f["title"], ensure_ascii=False)[1:-1].encode("utf-8")
              for f in located if isinstance(f.get("title"), str)]
    # MAGI FIX (MS3 §6 loop15, R7a2 Caspar): every frame of the file in a body is read (a body can
    # carry several history frames), and an item ends at the next frame, never at a bare `<<<`,
    # which a finding title may contain.
    for body in runs.trio_bodies(request_bodies):
        for frame in frames:
            for after in body.split(frame)[1:]:
                item = after.split(_FRAME_OPEN, 1)[0]
                if any(title and title + b" (attended)" in item for title in titles):
                    return True
    return False


#: How the product opens any context frame (``context_frame`` in ``src/review/pipeline.rs``).
_FRAME_OPEN = b"<<<context kind="


def _dirty_doc_sent(run) -> list:
    """Where the working tree's doc comment left the process, in a request the relay recorded.

    MAGI FIX (MS3 §6 loop15, R7a2 Caspar): only request bodies were read; the doc comment could
    also leave in a request target or a header.

    Args:
        run: A ``RunResult`` whose ``request_bodies`` is not ``None``.

    Returns:
        list: The places that carry it (a target, ``"a header"``); empty when none does.
    """
    needle = runs.REVIEW_DIRTY_DOC
    places = [target for target, body in run.request_bodies
              if needle in body or needle.decode("utf-8", "replace") in target]
    if any(needle in header for header in run.request_headers or ()):
        places.append("a header")
    return places