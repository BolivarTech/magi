# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-13
"""K1-K3 -- per-section API keys (``MAGI_API_KEY``) against the release binary.

The unit suite proves which key each section picks; what it cannot see is the
part this module covers: that a withheld key is recorded where the operator
looks and stays off the screen, that a key shaped for the wrong provider is
flagged before the first request, and that a real hosted trio really receives
its own key and names the variable when it is missing.

Three rules shape every scenario here.

**Properties of the product, never of the model.** Nothing reads what a model
said. K1 and K2 read the run log and the standard streams; K3 reads an exit
code, a JSON object and the variable named in a failure.

**Toy keys only.** The keys K1 and K2 put in front of the product are made up
and never leave the machine: neither scenario reaches a provider that could
accept or reject them, because the shape warning and the withheld-key entry are
both written at startup.

**The screen is both standard streams.** The product is run headless, where the
terminal the user watches is ``stdout`` plus ``stderr``. The interactive
window is not driven: it needs a pseudo-terminal the harness does not have, and
the notice sits on the same code path that decides which mouth a notice uses,
so the two surfaces differ in nothing this scenario judges.

K3 reads the hosted provider's key from ``DECISION_API_KEY`` in the harness's
own environment, the same variable ``decide.py`` uses. No key means
``CANNOT_TEST``, never ``PASS``.
"""

import json
import re
import shutil

from smoke import runs
from smoke.env import table_header
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.scenarios import decide
from smoke.secrets import PlantedSecret

QUERY_SUBCOMMAND = decide.QUERY_SUBCOMMAND
CONSULT_SUBCOMMAND = decide.CONSULT_SUBCOMMAND
WORKDIR_FLAG = decide.WORKDIR_FLAG
NO_MEMORY_FLAG = decide.NO_MEMORY_FLAG
JSON_FORMAT_ARGS = ("--output-format", "json")

#: The variable these scenarios exercise, and the variables the product also
#: reads that must be blank so the harness's own environment cannot leak in.
MAGI_KEY_VARIABLE = "MAGI_API_KEY"
OPENAI_KEY_VARIABLE = "OPENAI_API_KEY"
OTHER_KEY_VARIABLES = ("EMBEDDING_API_KEY", "DECISION_API_KEY",
                       "ANTHROPIC_API_KEY", "OPENAI_BASE_URL")

#: Blank means absent to the product.
BLANK_KEYS = {name: "" for name in OTHER_KEY_VARIABLES}

#: Toy keys. They are shaped like the real thing and worth nothing.
TOY_ANTHROPIC_SHAPED_KEY = "sk-ant-toy-0001"  # allow-secret-scan
TOY_OPENROUTER_SHAPED_KEY = "sk-or-toy-0001"  # allow-secret-scan
TOY_PRINCIPAL_KEY = "sk-toy-principal-0001"  # allow-secret-scan

#: K1: where the principal lives and where the trio is declared. Same host,
#: different spelling: a different origin, which is the case worth withholding.
ROOT_URL = "http://localhost:11434/v1"
TRIO_URL = "http://127.0.0.1:11434/v1"
TRIO_ORIGIN = "http://127.0.0.1:11434"

#: The fingerprint of the withheld-key entry: ``"OPENAI_API_KEY was not sent
#: to <origin>: ..."`` (``SectionKeys::startup_notices``).
RETAINED_FINGERPRINT = "was not sent to"

#: The fingerprint of the shape warning: ``"MAGI_API_KEY does not look like a
#: key for the provider of [magi].kind = ..."``.
SHAPE_WARNING_FINGERPRINT = "MAGI_API_KEY does not look like a key"
KIND_REFERENCE = "[magi].kind"

KIND_OPENAI_COMPAT = "openai-compat"
KIND_ANTHROPIC = "anthropic"

#: A prompt that costs nothing to answer; the properties are all startup ones.
STARTUP_PROMPT = b"Reply with the single word ok.\n"
CONSULT_PROMPT = (b"Is a single-line change worth a full review? Answer "
                  b"briefly.\n")
QUERY_TIMEOUT_S = 420
CONSULT_TIMEOUT_S = 600

#: K3: the hosted trio. Three distinct lineages (the product refuses fewer) and
#: free-tier models, so a certifying run costs nothing. Unmeasured: the first
#: run against a valid key either confirms these or replaces them.
HOSTED_BASE_URL = decide.OPENROUTER_BASE_URL
HOSTED_SEATS = (
    ("melchior", "meta-llama/llama-3.3-70b-instruct:free", "meta"),
    ("balthasar", "google/gemma-3-27b-it:free", "google"),
    ("caspar", "mistralai/mistral-small-3.2-24b-instruct:free", "mistral"),
)
MAGI_TABLE = "[magi]"
FALLBACK_TABLE = "[[magi.fallback]]"


# ---------------------------------------------------------------------------
# Configuration rewrites
# ---------------------------------------------------------------------------

def _toml_literal(value):
    """Render a string or an int as a TOML value.

    Args:
        value: A ``str`` or an ``int``.

    Returns:
        str: ``"quoted"`` text for a string, bare digits for an int.
    """
    return value if isinstance(value, int) else json.dumps(value)


def _set_in_table(text, table, settings, drop_tables=()):
    """Set keys of one table, replacing existing lines and dropping other tables.

    Existing lines for the given keys are removed from the table wherever they
    stand and the new ones are written straight after the header, so a comment
    or a reordered file cannot leave two definitions of one key.

    Complexity: ``O(lines)``.

    Args:
        text: A ``magi.toml``.
        table: The header spelling to edit, e.g. ``"[magi]"``.
        settings: Mapping of key to value.
        drop_tables: Header spellings whose whole blocks are removed.

    Returns:
        str | None: The new text, or None when *table* is absent.
    """
    out = []
    current = None
    found = False
    for line in text.splitlines():
        header = table_header(line)
        if header is not None:
            current = header
        if current in drop_tables:
            continue
        if header == table:
            found = True
            out.append(line)
            out.extend("%s = %s" % (key, _toml_literal(value))
                       for key, value in settings.items())
            continue
        if current == table and "=" in line and (
                line.split("=", 1)[0].strip() in settings):
            continue
        out.append(line)
    return "\n".join(out) + "\n" if found else None


def with_magi_endpoint(text, kind, base_url=None):
    """Declare the trio's kind and, optionally, its endpoint.

    Args:
        text: A ``magi.toml``.
        kind: ``[magi].kind``.
        base_url: ``[magi].base_url``, or None to inherit the root's.

    Returns:
        str | None: The new text, or None when the file has no ``[magi]``.
    """
    settings = {"kind": kind}
    if base_url is not None:
        settings["base_url"] = base_url
    return _set_in_table(text, MAGI_TABLE, settings)


def with_root_endpoint(text, base_url):
    """Replace the top-level ``base_url`` (the principal's endpoint).

    Complexity: ``O(lines)``.

    Args:
        text: A ``magi.toml``.
        base_url: The new root endpoint.

    Returns:
        str: The new text. A file with no root ``base_url`` gains one first.
    """
    new_line = "base_url = %s" % _toml_literal(base_url)
    out = []
    in_root = True
    replaced = False
    for line in text.splitlines():
        if table_header(line) is not None:
            in_root = False
        if in_root and "=" in line and line.split("=", 1)[0].strip() == "base_url":
            out.append(new_line)
            replaced = True
            continue
        out.append(line)
    if not replaced:
        out.insert(0, new_line)
    return "\n".join(out) + "\n"


def with_hosted_trio(text, base_url):
    """Point the whole trio at a hosted OpenAI-compatible endpoint.

    The seats get hosted model names with three distinct lineages, the pool of
    fallbacks (written for the local cloud tags) is dropped and rotation is
    switched off, so a failure names the key and not a model that does not
    exist on the provider.

    Args:
        text: A ``magi.toml``.
        base_url: The hosted endpoint.

    Returns:
        str | None: The new text, or None when the file has no ``[magi]``.
    """
    settings = {"kind": KIND_OPENAI_COMPAT, "base_url": base_url,
                "max_rotations": 0}
    for seat, model, lineage in HOSTED_SEATS:
        settings["%s_model" % seat] = model
        settings["%s_lineage" % seat] = lineage
    return _set_in_table(text, MAGI_TABLE, settings,
                         drop_tables=(FALLBACK_TABLE,))


# ---------------------------------------------------------------------------
# Judging helpers
# ---------------------------------------------------------------------------

def _streams(output):
    """Both standard streams of a capture, decoded leniently.

    Args:
        output: A capture.

    Returns:
        tuple[str, str]: ``(stdout, stderr)``.
    """
    return (output.stdout.decode("utf-8", errors="replace"),
            output.stderr.decode("utf-8", errors="replace"))


def shape_warning_shown(output):
    """Whether the shape warning reached the error stream.

    Args:
        output: A capture.

    Returns:
        bool: True when ``stderr`` carries the warning and names ``[magi].kind``.
    """
    stderr = _streams(output)[1]
    return SHAPE_WARNING_FINGERPRINT in stderr and KIND_REFERENCE in stderr


def leaks_value(output, value):
    """Whether a key's value appears on either stream.

    Args:
        output: A capture.
        value: The key.

    Returns:
        bool: True when the value is on ``stdout`` or ``stderr``.
    """
    return any(value in stream for stream in _streams(output))


def retained_entries(log):
    """Count the log lines that record a withheld key for the trio.

    Args:
        log: The concatenated run log, or None.

    Returns:
        int: Lines naming both ``MAGI_API_KEY`` and the trio's origin.
    """
    if not log:
        return 0
    return sum(1 for line in log.decode("utf-8", errors="replace").splitlines()
               if MAGI_KEY_VARIABLE in line and TRIO_ORIGIN in line)


def retained_on_screen(output):
    """Whether the withheld-key notice reached either standard stream.

    Args:
        output: A capture.

    Returns:
        bool: True when the notice's fingerprint is on ``stdout`` or ``stderr``.
    """
    return any(RETAINED_FINGERPRINT in stream for stream in _streams(output))


# ---------------------------------------------------------------------------
# K1 -- a withheld key goes to the run log and not to the screen
# ---------------------------------------------------------------------------

K1_ASSERTIONS = (
    "the run log has exactly one entry naming MAGI_API_KEY and http://127.0.0.1:11434",
    "the screen shows nothing about the retained key",
)


@scenario("K1", assertions=K1_ASSERTIONS, needs_backend=True)
def a_withheld_key_is_logged_and_not_shown(run):
    """Declare a trio on another spelling of the principal's host and start up.

    The principal's key must not travel to the trio, the run log must say so
    once, and the screen must stay quiet: an operator with an unauthenticated
    second server would otherwise see a warning on every start.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`K1_ASSERTIONS`, in that order.
    """
    root = decide._seed_scratch("k1")
    if root is None:
        yield from decide._all(K1_ASSERTIONS, Outcome.CANNOT_TEST,
                               decide._NO_SEED_DETAIL)
        return
    try:
        current = decide._read_toml(root)
        rewritten = (with_magi_endpoint(with_root_endpoint(current, ROOT_URL),
                                        KIND_OPENAI_COMPAT, TRIO_URL)
                     if current else None)
        if rewritten is None or not decide._write_toml(root, rewritten):
            yield from decide._all(K1_ASSERTIONS, Outcome.CANNOT_TEST,
                                   decide._NO_INSTALL_DETAIL)
            return
        planted = PlantedSecret(TOY_PRINCIPAL_KEY, "toy principal key")
        attempt = runs.attempt(
            [QUERY_SUBCOMMAND, *JSON_FORMAT_ARGS, WORKDIR_FLAG, str(root),
             NO_MEMORY_FLAG],
            stdin=STARTUP_PROMPT, timeout_s=QUERY_TIMEOUT_S, label="k1-query",
            planted=(planted,),
            env={OPENAI_KEY_VARIABLE: TOY_PRINCIPAL_KEY,
                 MAGI_KEY_VARIABLE: "", **BLANK_KEYS})
        if not attempt.ok:
            yield from decide._all(K1_ASSERTIONS, Outcome.CANNOT_TEST,
                                   attempt.failure)
            return
        log = decide._read_logs(root)
        entries = retained_entries(log)
        if log is None:
            yield decide._finding(
                K1_ASSERTIONS, 0, None,
                "the product wrote no run log under the workspace, so there is "
                "nothing to count")
        else:
            yield decide._finding(K1_ASSERTIONS, 0, "" if entries == 1 else (
                "%d run log lines name %s and %s, expected exactly one"
                % (entries, MAGI_KEY_VARIABLE, TRIO_ORIGIN)))
        if entries == 0:
            yield decide._finding(
                K1_ASSERTIONS, 1, None,
                "the entry is not in the run log, so a quiet screen cannot be "
                "told from a product that never withheld the key")
        else:
            yield decide._finding(K1_ASSERTIONS, 1, (
                "" if not retained_on_screen(attempt.output) else
                "the withheld-key notice appears on stdout or stderr: %s"
                % decide._excerpt(attempt.output)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# K2 -- a key shaped for another provider is flagged, not blocked
# ---------------------------------------------------------------------------

K2_ASSERTIONS = (
    "a sk-ant- MAGI_API_KEY under openai-compat warns naming MAGI_API_KEY and [magi].kind, without the value",
    "a sk-or- MAGI_API_KEY under anthropic warns the same way",
    "a sk-or- MAGI_API_KEY under openai-compat does not warn",
)

#: (key, trio kind, whether it must warn), one per assertion.
K2_CASES = (
    (TOY_ANTHROPIC_SHAPED_KEY, KIND_OPENAI_COMPAT, True),
    (TOY_OPENROUTER_SHAPED_KEY, KIND_ANTHROPIC, True),
    (TOY_OPENROUTER_SHAPED_KEY, KIND_OPENAI_COMPAT, False),
)


@scenario("K2", assertions=K2_ASSERTIONS)
def a_key_of_the_wrong_shape_warns_without_blocking(run):
    """Start the product three times with a different key and trio kind.

    No backend is needed: the principal is pointed at a closed port so the run
    ends at once, and the warning is written before any request. The third
    case only means something if the warning can be seen at all, so it is
    ``CANNOT_TEST`` unless one of the first two showed it.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`K2_ASSERTIONS`, in that order.
    """
    root = decide._seed_scratch("k2")
    if root is None:
        yield from decide._all(K2_ASSERTIONS, Outcome.CANNOT_TEST,
                               decide._NO_SEED_DETAIL)
        return
    try:
        base = decide._read_toml(root)
        dead = "http://127.0.0.1:%d/v1" % decide._closed_port()
        if base is None or not decide._write_toml(
                root, with_root_endpoint(base, dead)):
            yield from decide._all(K2_ASSERTIONS, Outcome.CANNOT_TEST,
                                   decide._NO_INSTALL_DETAIL)
            return
        base = decide._read_toml(root)
        results = [_run_k2_case(root, index, case, base)
                   for index, case in enumerate(K2_CASES)]
        warned_anywhere = any(r.warned for r in results if r.ran)
        for index, (_, _, must_warn) in enumerate(K2_CASES):
            yield _judge_k2_case(index, must_warn, results[index],
                                 warned_anywhere)
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _CaseResult:
    """What one K2 invocation showed.

    Attributes:
        ran: Whether the product produced a capture.
        warned: Whether the shape warning was on ``stderr``.
        leaked: Whether the key's value appeared on a stream.
        failure: Why there is no capture, or the install problem.
        output: The capture, or None.
    """

    def __init__(self, ran, warned, leaked, failure, output):
        """Record one case's evidence."""
        self.ran = ran
        self.warned = warned
        self.leaked = leaked
        self.failure = failure
        self.output = output


def _run_k2_case(root, index, case, base):
    """Run the product once for one K2 case.

    Args:
        root: The scratch workspace.
        index: Position in :data:`K2_CASES`, for the archive label.
        case: ``(key, kind, must_warn)``.
        base: The workspace's ``magi.toml`` with the dead principal endpoint.

    Returns:
        _CaseResult: The evidence.
    """
    key, kind, _ = case
    text = with_magi_endpoint(base, kind)
    if text is None or not decide._write_toml(root, text):
        return _CaseResult(False, False, False, decide._NO_INSTALL_DETAIL, None)
    attempt = runs.attempt(
        [QUERY_SUBCOMMAND, *JSON_FORMAT_ARGS, WORKDIR_FLAG, str(root),
         NO_MEMORY_FLAG],
        stdin=STARTUP_PROMPT, timeout_s=QUERY_TIMEOUT_S,
        label="k2-case-%d" % (index + 1),
        planted=(PlantedSecret(key, "toy key"),),
        env={MAGI_KEY_VARIABLE: key, OPENAI_KEY_VARIABLE: "", **BLANK_KEYS})
    if not attempt.ok:
        return _CaseResult(False, False, False, attempt.failure, None)
    return _CaseResult(True, shape_warning_shown(attempt.output),
                       leaks_value(attempt.output, key), "", attempt.output)


def _judge_k2_case(index, must_warn, result, warned_anywhere):
    """Judge one K2 assertion.

    Args:
        index: Which assertion.
        must_warn: Whether the case has to produce the warning.
        result: The case's evidence.
        warned_anywhere: Whether any case showed the warning.

    Returns:
        Finding: The finding.
    """
    if not result.ran:
        return decide._finding(K2_ASSERTIONS, index, None, result.failure)
    if must_warn:
        if not result.warned:
            return decide._finding(
                K2_ASSERTIONS, index,
                "no shape warning on stderr: %s"
                % decide._excerpt(result.output))
        return decide._finding(K2_ASSERTIONS, index, (
            "the warning repeats the key's value" if result.leaked else ""))
    if not warned_anywhere:
        return decide._finding(
            K2_ASSERTIONS, index, None,
            "neither of the other cases showed the warning, so silence here "
            "cannot be told from a warning that cannot be seen")
    return decide._finding(K2_ASSERTIONS, index, (
        "a key of the right shape produced the warning" if result.warned
        else ""))


# ---------------------------------------------------------------------------
# K3 -- the hosted trio and its own key
# ---------------------------------------------------------------------------

K3_ASSERTIONS = (
    "with MAGI_API_KEY the OpenRouter trio answers a consult",
    "without it the consult failure names MAGI_API_KEY",
)


@scenario("K3", assertions=K3_ASSERTIONS)
def the_hosted_trio_needs_and_names_its_own_key(run):
    """Run a consult against a hosted trio with and without its own key.

    The principal is local and carries a toy key. The hosted provider rejects
    that key, so a product that sent it would fail the second consult with a
    message that does not name the variable, and one that accepted it would
    succeed. The second assertion is only judged once the first showed the
    endpoint reachable with a valid key; otherwise a dead network would be
    reported as a product defect.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`K3_ASSERTIONS`, in that order.
            ``CANNOT_TEST`` without a key, or when the trio did not answer.
    """
    key = decide._hosted_key()
    if key is None:
        yield from decide._all(K3_ASSERTIONS, Outcome.CANNOT_TEST,
                               decide._NO_KEY_DETAIL)
        return
    root = decide._seed_scratch("k3")
    if root is None:
        yield from decide._all(K3_ASSERTIONS, Outcome.CANNOT_TEST,
                               decide._NO_SEED_DETAIL)
        return
    try:
        current = decide._read_toml(root)
        rewritten = with_hosted_trio(current, HOSTED_BASE_URL) if current else None
        if rewritten is None or not decide._write_toml(root, rewritten):
            yield from decide._all(K3_ASSERTIONS, Outcome.CANNOT_TEST,
                                   decide._NO_INSTALL_DETAIL)
            return
        with_key = _consult(root, "k3-with-key", key.value, (key,))
        answered = (with_key.ok and with_key.output.exit_code == 0
                    and _is_json_object(with_key.output.stdout))
        if not answered:
            why = (with_key.failure if not with_key.ok else
                   "the hosted trio did not answer with the key (exit %d): the "
                   "key may not be valid today or the endpoint unreachable: %s"
                   % (with_key.output.exit_code,
                      decide._excerpt(with_key.output)))
            yield from decide._all(K3_ASSERTIONS, Outcome.CANNOT_TEST, why)
            return
        yield decide._finding(K3_ASSERTIONS, 0, "")
        yield _without_key_finding(_consult(root, "k3-without-key", "", (key,)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _consult(root, label, magi_key, planted):
    """Run ``magi consult`` against the hosted trio.

    Args:
        root: The scratch workspace.
        label: The archive label.
        magi_key: The value for ``MAGI_API_KEY`` (blank for none).
        planted: Secrets this invocation puts in front of the product.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return runs.attempt(
        [CONSULT_SUBCOMMAND, *JSON_FORMAT_ARGS, WORKDIR_FLAG, str(root),
         NO_MEMORY_FLAG],
        stdin=CONSULT_PROMPT, timeout_s=CONSULT_TIMEOUT_S, label=label,
        planted=planted,
        env={MAGI_KEY_VARIABLE: magi_key, OPENAI_KEY_VARIABLE: TOY_PRINCIPAL_KEY,
             **BLANK_KEYS})


def _is_json_object(data):
    """Whether bytes parse as a JSON object.

    Args:
        data: Captured ``stdout``.

    Returns:
        bool: True for a JSON object.
    """
    try:
        return isinstance(json.loads(data.decode("utf-8")), dict)
    except (ValueError, UnicodeDecodeError):
        return False


def _without_key_finding(attempt):
    """Judge K3's second assertion.

    Args:
        attempt: The consult run without ``MAGI_API_KEY``.

    Returns:
        Finding: PASS when it failed and named the variable.
    """
    if not attempt.ok:
        return decide._finding(K3_ASSERTIONS, 1, None, attempt.failure)
    output = attempt.output
    if output.exit_code == 0:
        return decide._finding(
            K3_ASSERTIONS, 1,
            "the consult succeeded without %s, so the principal's key reached "
            "the hosted provider" % MAGI_KEY_VARIABLE)
    named = re.search(MAGI_KEY_VARIABLE, output.raw().decode("utf-8", "replace"))
    return decide._finding(K3_ASSERTIONS, 1, "" if named else (
        "the failure does not name %s: %s"
        % (MAGI_KEY_VARIABLE, decide._excerpt(output))))
