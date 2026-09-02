# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-08-31
"""S9 -- memory persists, embeds and injects.

Protects ``src/memory/``, failure #5 and REQ-29.

**It never asserts that the model answers the planted fact** (D-12). That is a
property of the LLM, and as a gate assertion it would be intermittent. What the
scenario measures is what the PRODUCT did: did it persist, did the embedder
answer, did the assembler inject.

**Injection is measured by DIFFERENCE, because no JSON field reports it.** R2
runs with ``--no-memory`` and is the control; R3 runs with memory. They carry
the SAME prompt, which took two attempts to get right, and R2 writes nothing,
so it can run in any order.

**The difference that carries the signal is the TRANSCRIPT, not the token
count**, and that too is measured rather than argued. The first version
subtracted ``usage.input_tokens`` and required a calibrated margin. Its control
carried R1's planting prompt, four times longer than R3's question, so the
subtraction mixed prompt length with injection; with that fixed, the honest
number came out at 33 tokens -- while the run with memory answered the planted
question correctly and the control replied that it had no stored memory.
Selective mode recalls what was asked for instead of replaying history, so the
injection is real and cheap, the delta sits inside ordinary variance, and no
threshold belongs there. What the two runs do not share is their transcript:
the control carries only the turns it produced.

**Assertion 3b is what stops assertion 3 passing green while measuring
something else.** Once the environment has accumulated enough that the assembler
saturates its budget, it loads bulk history rather than what the query asked
for, so a longer transcript stops meaning "it recalled what we planted" -- and
it is longer in exactly the same way, so assertion 3 would pass while measuring
something else. A green that no longer means what it says is worse than a red,
so the saturated case degrades BOTH to ``CANNOT_TEST``.

**The ceiling is DERIVED from the environment's own configuration, never
declared.** A fixed number in ``smoke.toml`` would age in the worse direction:
the environment grows and the number does not, so one day it stops firing and
the false green comes back with a guardian that looks like it is in place.

    usable_budget = (context_budget_tokens - response_headroom_tokens)
                    * (1 - safety_margin_ratio)
    saturated     = (input_tokens(R3) - input_tokens(R2))
                    >= usable_budget * ceiling_fraction

**If the environment declares any of those three fields missing, the ceiling is
not derived and 3b degrades.** The harness does NOT fill in the product's own
default: that is a second source of truth, and the copy is always the one that
forgets to be updated. Absent is not zero -- asserting against zero would test a
computation that never happened.

**Assertions 1 and 2 read R3's startup line, not R1's.** The line reports the
state as it was when the process opened, so R1's is the count from BEFORE it
planted anything. R3 runs after both and is the run whose injection assertion 3
measures, which makes its count the one that has to be non-zero for the rest of
the scenario to mean anything.

**The screen policy (SC-L14, S24) moved that line off R3's own capture.** The
memory-count notice is ``INFO``, so as of the screen policy it reaches only
the day's log file under the workspace's log directory -- never stdout or
stderr, and R3's own ``ProductOutput`` no longer carries it at all. Reading
``result.output.raw()`` the way this scenario used to is therefore reading a
surface the product deliberately stopped writing to; S24 is the scenario that
protects the surface having moved on purpose, and this one now has to follow
it there. Because the log directory is shared and persistent across every run
in the environment -- R1 attaches memory too, and prints its OWN startup line
first, with the count from before anything was planted -- a plain substring
search would risk reading R1's line instead of R3's. The correlation is by
run id, on the model S24 established: every rendered log line carries
``run=<id>`` (REQ-L63/SC-L79), and the matcher requires ONE LINE to carry both
the id and the marker, so what is read is bounded to what R3 itself wrote.
S23 searches the same directory for the same kind of id but does not bind it
to a line, which is enough for "the log survived process exit" and not for
this.

**Assertion 4 is a one-off invocation, not a ninth shared run.** Exactly one
assertion wants that output, it touches neither the trio nor the large payload,
and a table row read by a single scenario is ceremony. The workspace it builds
lives outside the repository -- so it can never reach ``git status`` -- and is
removed in ``finally``, because a harness that leaks a directory per run leaks a
database with it.

**What it points the embedder at ANSWERS, and that is the second attempt.** The
module used to name the loopback discard port and call it "reserved and
unused". Where that service is actually running the connection is accepted and
then never answered, so the probe waited out its entire ceiling and assertion 4
reported ``CANNOT_TEST`` rather than exercising REQ-29 at all. An endpoint that
returns an error is a failing embedder too, and it is one on every machine.
"""

import pathlib
import shutil
import tempfile

from smoke import logs, runs
from smoke.env import INIT_SUBCOMMAND, MAGI_DIR_NAME, MAGI_TOML_NAME, STARTUP_LINE
from smoke.errors import HarnessError, ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S9.
ASSERTIONS = (
    "the startup line reports N active with N > 0",
    "pending re-embed is 0 — the embedder answered",
    "R3's transcript carries turns R2's does not — the assembler loaded "
    "them",
    "the environment is below the saturation ceiling — otherwise 3 degrades "
    "to CANNOT_TEST",
    "with the embedder down, the run completes with a degradation notice",
)

#: The runs S9 reads. R1 plants, R2 is the ``--no-memory`` control, R3 recalls.
PLANTING_RUN = "R1"
CONTROL_RUN = "R2"
RECALL_RUN = "R3"

#: The three ``[memory]`` fields the saturation ceiling is derived from. Named
#: as a group because the rule is about the group: all three or nothing.
CEILING_FIELDS = ("context_budget_tokens", "response_headroom_tokens",
                  "safety_margin_ratio")

#: Where the token count lives in the output contract. Assertion 3b reads it;
#: assertion 3 does NOT, and the reason is measured rather than argued. With
#: the control finally carrying the same prompt, memory on against memory off
#: came out at 1436 against 1403 input tokens -- 33 apart -- while the run with
#: memory answered the planted question correctly and the control said it had
#: no stored memory. The injection is real and it is CHEAP: selective mode
#: recalls what the query asked for instead of replaying history, so the token
#: delta is inside ordinary variance and cannot carry a threshold. A margin
#: tuned to fit 33 would fire on noise; one above it would fail on a working
#: product. The observable that does discriminate is the transcript.
INPUT_TOKENS_PATH = "usage.input_tokens"

#: Where the run's own messages live. A run with memory carries the turns the
#: assembler loaded on top of the ones it produced; the ``--no-memory`` control
#: carries only its own. Against this repository's environment that is 79
#: against 4.
TRANSCRIPT_PATH = "transcript"

#: The table the override belongs in. Getting this wrong points the MAIN
#: provider at the failing endpoint, and the run then fails outright instead of
#: degrading -- a red that looks exactly like the assertion working.
EMBEDDING_SECTION_HEADER = "[embedding]"

#: The key overridden inside that table.
BASE_URL_KEY = "base_url"

#: The product's own phrase for this degradation, taken from the path the
#: probe actually reaches: the assembler failing and falling back to the full
#: history (``src/agent/mod.rs``, ``summarize_assembly_error``).
#:
#: It used to be "text-only persistence", which the product emits on exactly
#: one path -- the embedder CLIENT failing to CONSTRUCT, which happens for a
#: malformed URL or an unresolvable vault entry and never for an endpoint that
#: is merely unreachable or that answers with an error, because the client is
#: built lazily and constructs fine either way. The probe creates the second
#: kind, so the assertion matched a string its own run could not produce, and
#: it hid behind CANNOT_TEST for as long as the probe also hung.
DEGRADATION_MARKER = "context assembly failed"

#: What the probe stores before it breaks the embedder. A workspace with no
#: memories never asks the embedder to do anything, so the degradation never
#: happens and the assertion evaluates a run in which nothing went wrong.
PLANT_PROMPT = b"Remember that my favourite colour is green, then say ok.\n"

#: What the probe asks once the embedder is failing. It has to be a RECALL,
#: because that is the path that embeds a query and therefore the path that
#: degrades.
RECALL_PROMPT = b"What is my favourite colour?\n"

#: What the planting half is called in the archive.
PLANT_LABEL = "s9-embedder-down-plant"

#: How long each half of the probe is given, in seconds. The scaffold touches
#: no network; the query is one short turn against the real backend.
SCAFFOLD_TIMEOUT_S = 60
PROBE_TIMEOUT_S = 180

#: What the two probe invocations are called in the archive.
SCAFFOLD_LABEL = "s9-embedder-down-init"
PROBE_LABEL = "s9-embedder-down-query"

#: Prefix of the throwaway workspace, so a leaked one is identifiable.
PROBE_PREFIX = "smoke-s9-embedder-"

#: The exit code a run that degraded gracefully still reports.
SUCCESS_EXIT_CODE = 0


def usable_budget(settings):
    """The assembler's usable token budget, derived from the environment.

    Complexity: ``O(len(CEILING_FIELDS))``.

    Args:
        settings: The environment's ``[memory]`` table, as parsed.

    Returns:
        float | None: The budget, or ``None`` when any of the three fields is
        absent or is not a number. ``None`` means *not derived*, which is a
        different state from a budget of zero and is reported as such.
    """
    values = []
    for field in CEILING_FIELDS:
        value = settings.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    budget, headroom, margin_ratio = values
    return (budget - headroom) * (1.0 - margin_ratio)


def point_embedding_at(endpoint, text):
    """Return *text* with the embedding endpoint overridden.

    The edit is line-level because the standard library ships no TOML writer
    (REQ-S02 forbids adding one). The override is inserted directly after the
    section header, which is where TOML resolves it into that table and nowhere
    else -- and where a later duplicate key inside the same table would lose to
    it, so the commented default the product ships cannot resurrect itself.

    Complexity: ``O(number of lines)``.

    Args:
        endpoint: The URL the embedder should be pointed at.
        text: The configuration to rewrite.

    Returns:
        str: The rewritten configuration.

    Raises:
        HarnessError: If the text declares no ``[embedding]`` table. Appending
            one would silently change which table every following key belongs
            to, so the harness refuses rather than guesses.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == EMBEDDING_SECTION_HEADER:
            override = '%s = "%s"' % (BASE_URL_KEY, endpoint)
            return "\n".join(lines[:index + 1] + [override]
                             + lines[index + 1:]) + "\n"
    raise HarnessError(
        "the environment's %s declares no %s table, so there is nothing to "
        "point at the failing endpoint"
        % (MAGI_TOML_NAME, EMBEDDING_SECTION_HEADER)
    )


@scenario("S9", assertions=ASSERTIONS,
           run=(PLANTING_RUN, CONTROL_RUN, RECALL_RUN),
          needs_backend=True, needs_ambient=True)
def memory_persists_embeds_and_injects(run, ambient):
    """Compare the three runs, then take the embedder down once.

    Args:
        run: The three ``RunResult`` objects keyed by id, as the runner hands a
            scenario that declares several.
        ambient: State captured before any run started. S9 reads the margin,
            the fraction and the environment's ``[memory]`` block off it.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    results = run or {}
    counts, counts_failure = _startup_counts(results.get(RECALL_RUN))
    yield _active_finding(counts, counts_failure)
    yield _pending_finding(counts, counts_failure)

    saturation = _saturation_finding(results, ambient)
    yield _injection_finding(results, ambient, saturation.outcome)
    yield saturation
    yield _embedder_down_finding()


def _startup_counts(result):
    """The three counts of R3's own startup line, read from the day's log.

    Since the screen policy (SC-L14), the memory-count notice is ``INFO`` and
    reaches only the log directory -- R3's own capture no longer carries it.
    The directory is shared by every run in the environment, so the line is
    found by correlating R3's own run id (REQ-L63) against the ``run=<id>``
    field every rendered log line carries -- the same LINE-BOUND matcher S24
    uses. S23 is the looser sibling and is not the model here: it asks only
    whether its id appears anywhere under the directory, which is enough for
    what it asserts and not enough for this, where the id has to sit on the
    very line the counts are read from.

    Complexity: ``O(bytes in the log directory)``.

    Args:
        result: R3's result, or None.

    Returns:
        tuple: ``(counts, failure)`` where *counts* is the ``(active,
        archived, pending)`` triple and *failure* explains its absence,
        naming where this looked. Exactly one of the two is set.
    """
    if result is None:
        return None, "run %s produced no capture to inspect" % RECALL_RUN
    run_id = logs.resolve_id(result.output)
    if run_id is None:
        return None, (
            "run %s published no run id on stderr or in its JSON envelope, "
            "so its own line cannot be told apart from another run's in the "
            "shared log directory" % RECALL_RUN
        )
    match, dir_existed, unreadable = logs.scan(_startup_line_matcher(run_id))
    # A HIT IS THE ANSWER, and it is read before anything is said about what
    # else in the directory would not open. ``.magi/logs`` is shared and
    # persistent, so an old rotated file that cannot be read is an ordinary
    # condition of the environment and has nothing to do with the line this run
    # wrote -- which the scan has already located. Asking about the unreadable
    # list first threw that evidence away and reported "cannot test" while
    # holding the counts. S23 and S24 both order it this way; S9 did not.
    if match is not None:
        return tuple(int(group) for group in match.groups()), ""
    directory = logs.log_directory()
    if not dir_existed:
        return None, (
            "%s does not exist, so %s wrote no log at all"
            % (directory, RECALL_RUN)
        )
    if unreadable:
        return None, (
            "run %s's startup line was not found, but these files under %s "
            "could not be read: %s"
            % (RECALL_RUN, directory, "; ".join(unreadable))
        )
    return None, (
        "no line under %s carries both run=%s and the memory "
        "diagnostics marker, so neither count can be read"
        % (directory, run_id)
    )


def _startup_line_matcher(run_id):
    """Build a :func:`smoke.logs.scan` matcher bound to one run's own lines.

    Splitting on ``b"\\n"`` is safe because the logging layer escapes every
    control character out of a rendered event before it is written (REQ-L64
    stage 3), so one physical line is always exactly one event -- a foreign
    string can never fold two events together or split one apart.

    **The needle ends at the id, not at the separator after it.** It used to
    carry a trailing space, which made the guard depend on how the renderer
    joins fields -- a detail REQ-L63 never promised. Anchoring on the id alone
    cannot widen the match either, because an id is ``<pid>-<16 hex>`` with the
    hex half a fixed width: no id is a prefix of a different one, so a bare
    ``run=<id>`` still binds the line to exactly this run.

    Args:
        run_id: The id to bind the search to.

    Returns:
        Callable[[bytes], re.Match | None]: A matcher over one file's raw
        bytes; returns the first line's :data:`STARTUP_LINE` match that also
        carries this run's id, or None.
    """
    needle = ("run=%s" % run_id).encode("utf-8")

    def matcher(data):
        for line in data.split(b"\n"):
            if needle in line:
                match = STARTUP_LINE.search(line)
                if match is not None:
                    return match
        return None

    return matcher


def _active_finding(counts, failure):
    """Judge assertion 1: something was persisted and survived.

    Args:
        counts: The startup triple, or None.
        failure: Why there is none.

    Returns:
        Finding: PASS when the active count is above zero.
    """
    if counts is None:
        return _s9(0, Outcome.CANNOT_TEST, RECALL_RUN, failure)
    active = counts[0]
    if active <= 0:
        return _s9(0, Outcome.FAIL, RECALL_RUN,
                   "the startup line reports %d active memories, so nothing "
                   "%s planted survived to be recalled" % (active, PLANTING_RUN))
    return _s9(0, Outcome.PASS, RECALL_RUN, "")


def _pending_finding(counts, failure):
    """Judge assertion 2: nothing is still waiting to be embedded.

    Args:
        counts: The startup triple, or None.
        failure: Why there is none.

    Returns:
        Finding: PASS when the pending count is zero. A non-zero count is the
        embedder having declined to answer, which is the failure the assertion
        names.
    """
    if counts is None:
        return _s9(1, Outcome.CANNOT_TEST, RECALL_RUN, failure)
    pending = counts[2]
    if pending != 0:
        return _s9(1, Outcome.FAIL, RECALL_RUN,
                   "%d memor%s are still waiting to be embedded, so the "
                   "embedder did not answer for them"
                   % (pending, "y" if pending == 1 else "ies"))
    return _s9(1, Outcome.PASS, RECALL_RUN, "")


def _injection_finding(results, ambient, saturation):
    """Judge assertion 3: memory loaded turns the run did not produce.

    Measured on the transcript rather than on a token delta, and the change is
    evidence-led. See :data:`INPUT_TOKENS_PATH` for the numbers: with the
    control finally carrying the same prompt, the two runs came out 33 tokens
    apart while the one with memory answered the planted question and the
    control said it had no stored memory. Selective mode recalls what was
    asked for instead of replaying history, so what it costs in tokens is
    inside ordinary variance and no threshold can sit there honestly. What the
    two runs do NOT share is their transcript: the control carries only its own
    turns.

    Args:
        results: The three results, keyed by id.
        ambient: The ambient state. Unused here now, kept because assertion 3b
            hands it the same signature.
        saturation: What assertion 3b concluded. Anything but PASS takes this
            assertion down with it: a saturated assembler loads bulk history
            rather than what the query asked for, so a longer transcript stops
            meaning it recalled what was planted.

    Returns:
        Finding: PASS when the run with memory carries more turns than the
        control.
    """
    del ambient
    if saturation is not Outcome.PASS:
        return _s9(2, Outcome.CANNOT_TEST, RECALL_RUN,
                   "the saturation check did not pass, so the difference "
                   "between %s and %s cannot be read as injection"
                   % (RECALL_RUN, CONTROL_RUN))
    counts = {}
    for run_id in (CONTROL_RUN, RECALL_RUN):
        result = results.get(run_id)
        if result is None:
            return _s9(2, Outcome.CANNOT_TEST, RECALL_RUN,
                       "run %s produced no capture" % run_id)
        try:
            turns = result.output.key(TRANSCRIPT_PATH)
        except ProductOutputError as exc:
            return _s9(2, Outcome.CANNOT_TEST, RECALL_RUN, str(exc))
        if not isinstance(turns, list):
            return _s9(2, Outcome.CANNOT_TEST, RECALL_RUN,
                       "%s of run %s is %s, expected a list"
                       % (TRANSCRIPT_PATH, run_id, type(turns).__name__))
        counts[run_id] = len(turns)
    if counts[RECALL_RUN] <= counts[CONTROL_RUN]:
        return _s9(2, Outcome.FAIL, RECALL_RUN,
                   "%s carries %d turns against %s's %d, so the assembler "
                   "loaded nothing the run did not produce itself"
                   % (RECALL_RUN, counts[RECALL_RUN], CONTROL_RUN,
                      counts[CONTROL_RUN]))
    return _s9(2, Outcome.PASS, RECALL_RUN, "")


def _saturation_finding(results, ambient):
    """Judge assertion 3b: the environment has not filled the budget.

    Args:
        results: The three results, keyed by id.
        ambient: The ambient state carrying the fraction and the ``[memory]``
            block.

    Returns:
        Finding: PASS below the ceiling; CANNOT_TEST when the ceiling is not
        derived, when the fraction is uncalibrated, or when the environment has
        reached it -- the last of which blocks the gate and sends whoever reads
        the report to recalibrate or reset, which is the decision that has to be
        taken with the number in front of them.
    """
    budget = usable_budget(ambient.memory_settings)
    if budget is None:
        return _s9(3, Outcome.CANNOT_TEST, RECALL_RUN,
                   "the environment's [memory] table does not declare all of "
                   "%s, so the ceiling is not derived. The harness does not "
                   "fill in the product's defaults: absent is not zero."
                   % ", ".join(CEILING_FIELDS))
    fraction = ambient.ceiling_fraction
    # A TOML ``ceiling_fraction = 1`` parses as an int, and demanding a float
    # made a calibrated whole number report "no ceiling fraction is
    # calibrated" -- false about the file in front of it. ``bool`` is excluded
    # for the same reason ``usable_budget`` excludes it: True is not a
    # fraction.
    if (not isinstance(fraction, (int, float)) or isinstance(fraction, bool)
            or fraction <= 0.0):
        return _s9(3, Outcome.CANNOT_TEST, RECALL_RUN,
                   "no ceiling fraction is calibrated in smoke.toml, so there "
                   "is no ceiling to compare against")
    delta, failure = _token_delta(results)
    if delta is None:
        return _s9(3, Outcome.CANNOT_TEST, RECALL_RUN, failure)
    ceiling = budget * fraction
    if delta >= ceiling:
        return _s9(3, Outcome.CANNOT_TEST, RECALL_RUN,
                   "the assembler added %d tokens against a ceiling of %.0f, "
                   "so the difference now measures a full budget rather than "
                   "the injection; recalibrate or reset the environment"
                   % (delta, ceiling))
    return _s9(3, Outcome.PASS, RECALL_RUN, "")


def _token_delta(results):
    """How many more input tokens R3 sent than R2.

    Args:
        results: The three results, keyed by id.

    Returns:
        tuple: ``(delta, failure)``; exactly one of the two is set.
    """
    counts = {}
    for run_id in (CONTROL_RUN, RECALL_RUN):
        result = results.get(run_id)
        if result is None:
            return None, "run %s produced no capture to inspect" % run_id
        try:
            value = result.output.key(INPUT_TOKENS_PATH)
        except ProductOutputError as exc:
            return None, "run %s: %s" % (run_id, exc)
        if not isinstance(value, int) or isinstance(value, bool):
            return None, ("run %s reported %s as %s, expected a number"
                          % (run_id, INPUT_TOKENS_PATH, type(value).__name__))
        counts[run_id] = value
    return counts[RECALL_RUN] - counts[CONTROL_RUN], ""


def _embedder_down_finding():
    """Judge assertion 4 by running once against an unreachable embedder.

    The workspace is scaffolded by the product's own ``init`` -- so it carries
    whatever permissions the product expects -- and its configuration is then
    replaced by the ENVIRONMENT's, patched. Reusing the environment's config is
    what keeps the probe on the same models and the same endpoint the rest of
    the run is using, instead of silently falling back to the product's
    defaults under a cheap profile.

    Returns:
        Finding: PASS when the run completed and said it had degraded.
    """
    workspace = pathlib.Path(tempfile.mkdtemp(prefix=PROBE_PREFIX))
    try:
        scaffold = runs.attempt([INIT_SUBCOMMAND], timeout_s=SCAFFOLD_TIMEOUT_S,
                                label=SCAFFOLD_LABEL, cwd=str(workspace))
        if not scaffold.ok:
            return _s9(4, Outcome.CANNOT_TEST, None, scaffold.failure)
        if scaffold.output.exit_code != SUCCESS_EXIT_CODE:
            return _s9(4, Outcome.CANNOT_TEST, None,
                       "%s exited %d, so there is no workspace to run the "
                       "probe in" % (INIT_SUBCOMMAND,
                                     scaffold.output.exit_code))
        try:
            source = (runs.workspace_root() / MAGI_DIR_NAME
                      / MAGI_TOML_NAME).read_text(encoding="utf-8")
        except OSError as exc:
            return _s9(4, Outcome.CANNOT_TEST, None,
                       "the environment's %s could not be read, so the probe "
                       "has no configuration to patch: %s"
                       % (MAGI_TOML_NAME, exc))
        # Plant FIRST, with the environment's own embedder still working, so
        # the store has something for the recall to look for. Only then break
        # it: a workspace with no memories never asks the embedder anything.
        (workspace / MAGI_DIR_NAME / MAGI_TOML_NAME).write_text(
            source, encoding="utf-8")
        planted = _query(workspace, PLANT_PROMPT, PLANT_LABEL)
        if not planted.ok:
            return _s9(4, Outcome.CANNOT_TEST, None, planted.failure)
        if planted.output.exit_code != SUCCESS_EXIT_CODE:
            return _s9(4, Outcome.CANNOT_TEST, None,
                       "the planting run exited %d, so there is nothing in "
                       "the store for the recall to embed"
                       % planted.output.exit_code)
        with runs.error_backend() as endpoint:
            (workspace / MAGI_DIR_NAME / MAGI_TOML_NAME).write_text(
                point_embedding_at(endpoint, source), encoding="utf-8")
            probe = _query(workspace, RECALL_PROMPT, PROBE_LABEL)
        return _judge_degraded(probe)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _query(workspace, prompt, label):
    """Run one query inside the throwaway workspace.

    Args:
        workspace: Where the product should look for its ``.magi/``.
        prompt: What to send on standard input.
        label: What to call the invocation in the archive.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return runs.attempt(
        ["query", "--output-format", "json", "-w", str(workspace), "--auto"],
        stdin=prompt, timeout_s=PROBE_TIMEOUT_S, label=label,
        env={runs.PASSPHRASE_VARIABLE: runs.passphrase()},
    )


def _judge_degraded(probe):
    """Decide assertion 4 from the probe's capture.

    Args:
        probe: The probe's :class:`~smoke.runs.Attempt`.

    Returns:
        Finding: PASS only when the run completed AND announced the
        degradation. A silent completion is reported as the failure it is: an
        operator whose embedder is down and who is told nothing watches
        memories accumulate unembedded with no signal that anything changed.
    """
    if not probe.ok:
        return _s9(4, Outcome.CANNOT_TEST, None, probe.failure)
    output = probe.output
    if output.exit_code != SUCCESS_EXIT_CODE:
        return _s9(4, Outcome.FAIL, None,
                   "an unreachable embedder took the run down: exit %d. "
                   "REQ-29 requires degrading to text-only persistence, not "
                   "failing the turn." % output.exit_code)
    if DEGRADATION_MARKER.encode("utf-8") not in output.raw():
        return _s9(4, Outcome.FAIL, None,
                   "the run completed but never said %r, so an operator whose "
                   "embedder is unreachable is told nothing"
                   % DEGRADATION_MARKER)
    return _s9(4, Outcome.PASS, None, "")


def _s9(index, outcome, run_id, detail):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        run_id: The shared run it came from, or None for the one-off probe.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=run_id)
