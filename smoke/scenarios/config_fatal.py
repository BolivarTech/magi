# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The two configuration scenarios: S11 -- a broken file; S15 -- a blank value.

S11 protects ``config/migrate.rs`` and the rule that a present-but-unparseable
``magi.toml`` is fatal rather than a warning that degrades to defaults. With
``base_url`` load-bearing for every endpoint, discarding a broken file silently
would run the agent against endpoints the operator never named.

Three things here are deliberate.

The variants share ONE seeded workspace and replace its ``magi.toml`` each
time. Seeding is the expensive half -- it creates an encrypted database, which
costs an Argon2id derivation -- and the configuration is the whole subject, so
four workspaces would buy nothing but four derivations.

Each variant is built from the file the PRODUCT generated, with the single
defect injected into it. A configuration written from scratch here would carry
this module's idea of the schema, and would then fail for a reason the scenario
never planted.

Assertion 2 points ``base_url`` at a port nothing is listening on and asserts
the failure carries no connection error. It does NOT measure time: a timing
assertion is a gate that goes red on a loaded machine, and this one has to
answer a question about ordering, which timing only approximates.

S15 protects ``non_blank``: a text-valued environment variable exported empty
or whitespace-only is ABSENT, never invalid. An exported-but-unfilled variable
in a CI script is an everyday accident, and breaking startup over it punishes
the accident instead of falling through to the next precedence level.

Its third assertion is what stops the rule from overshooting. A product that
treated EVERY value as absent would satisfy the first two perfectly, so a
scenario without the third would pass over the opposite defect and report the
whole family as protected.

One limit of S15 is declared here rather than left for a reader to discover.
The second assertion's clause about an empty credential short-circuiting the
vault lookup is checked only as far as the outside of the process allows: the
harness sees the exit code and the output, never the lookup, so what it
verifies is that no configuration or vocabulary failure stopped the run before
it reached the backend. A short-circuit that silently produced a wrong
credential and still reached the network would not be visible here.
"""

import pathlib
import re
import socket
import tempfile

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S11.
S11_ASSERTIONS = (
    "an unknown field in magi.toml exits 2 naming the field",
    "it cuts before any backend request is issued",
    "a v0.11.0-era file names every incompatibility at once, not just the first",
    "a seat declaring a model without its lineage fails naming all three seats",
)

#: The verbatim assertion texts of the spec's section 8, for S15.
S15_ASSERTIONS = (
    "each text-valued variable, exported empty or blank, falls through to the "
    "next precedence level",
    "startup succeeds — no vocabulary error, no empty credential "
    "short-circuiting the vault lookup",
    "a value that is present and unrecognised is still an error",
)

#: The eight text-valued variables the spec names. ``MAGI_MODEL_*`` is a family
#: and one member stands for it: the resolver is shared, so a second seat would
#: exercise the same code path and cost another invocation.
BLANKABLE_VARIABLES = (
    "MAGI_PROVIDER",
    "MAGI_PASSPHRASE",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "MAGI_MODEL_MELCHIOR",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

#: Blank means BOTH of these. An exported-but-unfilled variable arrives as the
#: first; a value trimmed to nothing by a shell arrives as the second.
BLANK_SPELLINGS = ("", "   ")

#: The variable whose blank case is different, and why. Falling through leads
#: to a TTY prompt that does not exist in this context, so the expected result
#: is the typed refusal -- never a hang, which is why every invocation carries
#: a timeout the harness controls.
PASSPHRASE_VARIABLE = "MAGI_PASSPHRASE"
PASSPHRASE_UNAVAILABLE_MARKER = b"no passphrase"
PASSPHRASE_UNAVAILABLE_EXIT = 1

#: A value nobody could mean. Assertion 3 requires the product to reject it,
#: which is what distinguishes "blank falls through" from "anything goes".
UNRECOGNISED_PROVIDER = "not-a-real-backend"

#: What a vocabulary refusal looks like. Their ABSENCE is the evidence for
#: assertions 1 and 2; the first one's PRESENCE is the evidence for 3.
VOCABULARY_MARKERS = (
    b"unknown provider",
    b"unknown mode",
    b"unknown field",
    b"invalid value",
)

INIT_SUBCOMMAND = "init"
QUERY_SUBCOMMAND = "query"
MAGI_DIR_NAME = ".magi"
MAGI_TOML_NAME = "magi.toml"

#: How long the harness waits for one invocation, in seconds, and the wall
#: clock it hands the product. Every variant is expected to die during
#: configuration, so the product's own bound is small: it is there to stop a
#: variant that did NOT cut from waiting on a socket nobody will answer.
INIT_TIMEOUT_S = 120
QUERY_TIMEOUT_S = 180
PRODUCT_TIMEOUT_S = 20

#: What the product should exit with when it rejects a configuration.
CONFIG_EXIT_CODE = 2

#: The root key variant A plants. Prefixed so a reader of a failing run can see
#: at a glance that the harness put it there and the operator did not.
UNKNOWN_FIELD = "smoke_unknown_field"

#: The three v0.11.0-era incompatibilities, and the token that says each one was
#: named. The guided migration exists because ``deny_unknown_fields`` aborts on
#: the FIRST unknown key, so the assertion counts that ALL THREE are named --
#: not that the output is non-empty, which any one of them alone would satisfy.
V011_MARKERS = ("provider", "base_url", "tool_result_cap_bytes")

#: The three seats. A seat that declares a model without its lineage is a load
#: error since v0.13.0, because a lineage is a failure domain the operator
#: chooses and the product will not infer one -- guessing would look exactly
#: like a declaration.
SEATS = ("melchior", "balthasar", "caspar")

#: What a failure that REACHED the backend looks like. Their absence is the
#: evidence for assertion 2; matched case-insensitively because the wording
#: comes from the platform's socket layer, not from the product.
CONNECTION_MARKERS = (
    b"connection refused",
    b"error sending request",
    b"failed to connect",
    b"tcp connect",
    b"os error 10061",
    b"os error 111",
)

#: The prompt every variant feeds the product. No assertion reads the answer.
PROBE_PROMPT = b"say ok\n"

_TABLE_LINE = re.compile(r"^\s*\[([^]]+)\]\s*$")
_LINEAGE_KEY = re.compile(r"^\s*(%s)_lineage\s*=" % "|".join(SEATS))
_ROOT_BASE_URL = re.compile(r"^\s*base_url\s*=")


@scenario("S11")
def a_broken_config_cuts_before_running(run):
    """Feed the product three broken configurations and read how it refuses.

    Args:
        run: Always ``None``; S11 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S11_ASSERTIONS`, in that order.
    """
    root = _seed_workspace()
    if root is None:
        for index in range(len(S11_ASSERTIONS)):
            yield _s11(index, Outcome.CANNOT_TEST,
                       "the product's %s did not scaffold a workspace to "
                       "configure" % INIT_SUBCOMMAND)
        return

    generated = _generated_config(root)
    if generated is None:
        for index in range(len(S11_ASSERTIONS)):
            yield _s11(index, Outcome.CANNOT_TEST,
                       "the scaffolded %s could not be read, so no variant "
                       "could be built from it" % MAGI_TOML_NAME)
        return

    dead_port = _closed_port()
    unknown = _run_variant(
        root, _with_unknown_field(generated, dead_port), "s11-unknown-field")
    legacy = _run_variant(root, _v011_config(dead_port), "s11-v011")
    seatless = _run_variant(
        root, _without_lineages(generated), "s11-missing-lineage")

    yield _unknown_field_finding(unknown)
    yield _cuts_early_finding(unknown, dead_port)
    yield _all_at_once_finding(legacy)
    yield _seat_lineage_finding(seatless)


def _seed_workspace():
    """Scaffold one workspace under the scratch area, by cwd and never by flag.

    Returns:
        pathlib.Path | None: The directory holding the new ``.magi/``, or None
        when the product did not create one.
    """
    root = pathlib.Path(
        tempfile.mkdtemp(prefix="s11-", dir=str(runs.scratch_root()))
    )
    seeded = runs.attempt([INIT_SUBCOMMAND], stdin=b"",
                          timeout_s=INIT_TIMEOUT_S, label="s11-init", cwd=root,
                          env={"MAGI_PASSPHRASE": runs.passphrase()})
    if not seeded.ok or not (root / MAGI_DIR_NAME).is_dir():
        return None
    return root


def _generated_config(root):
    """Read the configuration the product wrote for itself.

    Args:
        root: The seeded workspace's parent directory.

    Returns:
        str | None: The generated text, or None when it cannot be read.
    """
    try:
        return (root / MAGI_DIR_NAME / MAGI_TOML_NAME).read_text(
            encoding="utf-8")
    except OSError:
        return None


def _run_variant(root, text, label):
    """Install one broken configuration and run the product against it.

    Args:
        root: The seeded workspace's parent directory.
        text: The configuration to install.
        label: What to call the invocation in the archive.

    Returns:
        Attempt: The capture, or why there is none. A configuration that cannot
        be written yields an Attempt carrying that as its failure, so the
        caller reports CANNOT_TEST rather than judging the product on it.
    """
    path = root / MAGI_DIR_NAME / MAGI_TOML_NAME
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return runs.Attempt(output=None,
                            failure="cannot write %s: %s" % (path, exc))
    return runs.attempt(
        [QUERY_SUBCOMMAND, "--output-format", "json",
         "--timeout", str(PRODUCT_TIMEOUT_S)],
        stdin=PROBE_PROMPT, timeout_s=QUERY_TIMEOUT_S, label=label, cwd=root,
        env={"MAGI_PASSPHRASE": runs.passphrase()},
    )


def _closed_port():
    """Find a port nothing is listening on.

    The socket is bound to ask the operating system for a free port and closed
    immediately, so what the product meets is a refusal rather than a hang.
    Nothing guarantees the port stays free -- another process could take it in
    the interval -- which is why assertion 2 reads the ABSENCE of a connection
    error rather than the presence of a particular one.

    Returns:
        int: The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _with_unknown_field(generated, dead_port):
    """Build variant A: a valid file plus one unknown ROOT key.

    The key goes at the very top because a TOML root key written after a table
    header belongs to that table, and the assertion is about a root key.

    Args:
        generated: The product's own configuration.
        dead_port: The port to aim every endpoint at.

    Returns:
        str: The configuration text.
    """
    body = _with_dead_endpoint(generated, dead_port)
    return '%s = "planted by the smoke harness"\n%s' % (UNKNOWN_FIELD, body)


def _with_dead_endpoint(generated, dead_port):
    """Point the root ``base_url`` at a port nothing answers on.

    Complexity: ``O(lines)``.

    Args:
        generated: The product's own configuration.
        dead_port: The port to aim at.

    Returns:
        str: The configuration text, with the root endpoint replaced.
    """
    lines = []
    table = ""
    for line in generated.splitlines():
        match = _TABLE_LINE.match(line)
        if match:
            table = match.group(1).strip()
        elif not table and _ROOT_BASE_URL.match(line):
            line = 'base_url = "http://127.0.0.1:%d/v1"' % dead_port
        lines.append(line)
    return "\n".join(lines) + "\n"


def _v011_config(dead_port):
    """Build variant B: a file carrying all three v0.11.0-era patterns.

    Written out rather than derived. The point of the variant is the three
    incompatibilities and nothing else, and a file built by editing the current
    generated one would also carry today's keys, so a message naming them would
    be about the wrong thing.

    Args:
        dead_port: The port the retired ``[openai].base_url`` should name.

    Returns:
        str: The configuration text.
    """
    return (
        'provider = "openai"\n'
        "\n"
        "[openai]\n"
        'base_url = "http://127.0.0.1:%d/v1"\n'
        "\n"
        "[headless]\n"
        "tool_result_cap_bytes = 4096\n"
    ) % dead_port


def _without_lineages(generated):
    """Build variant C: three seats declaring a model and no lineage.

    Complexity: ``O(lines)``.

    Args:
        generated: The product's own configuration, which declares both halves
            of every seat.

    Returns:
        str: The configuration text, with every ``<seat>_lineage`` line gone.
    """
    kept = [line for line in generated.splitlines()
            if not _LINEAGE_KEY.match(line)]
    return "\n".join(kept) + "\n"


def _unknown_field_finding(attempt):
    """Judge assertion 1: exit 2, and the message names the offending field.

    Args:
        attempt: The variant A capture.

    Returns:
        Finding: PASS only when both halves hold; the detail says which half
        failed, because "it refused" and "it refused usefully" are different
        guarantees and only one of them is what the operator needs.
    """
    if not attempt.ok:
        return _s11(0, Outcome.CANNOT_TEST, attempt.failure)
    named = UNKNOWN_FIELD.encode("utf-8") in attempt.output.raw()
    if not named and attempt.output.exit_code == 0:
        return _s11(0, Outcome.FAIL,
                    "the unknown root key %s was accepted" % UNKNOWN_FIELD)
    if not named:
        return _s11(0, Outcome.FAIL,
                    "exited %d without naming %s: %s"
                    % (attempt.output.exit_code, UNKNOWN_FIELD,
                       _excerpt(attempt.output)))
    if attempt.output.exit_code != CONFIG_EXIT_CODE:
        return _s11(0, Outcome.FAIL,
                    "named %s but exited %d, expected %d"
                    % (UNKNOWN_FIELD, attempt.output.exit_code,
                       CONFIG_EXIT_CODE))
    return _s11(0, Outcome.PASS, "")


def _cuts_early_finding(attempt, dead_port):
    """Judge assertion 2 by what the failure does NOT say.

    Every endpoint in variant A points at a port nothing answers on, so a run
    that got as far as a request would carry the platform's connection failure.
    A run that cut at the configuration cannot.

    Args:
        attempt: The variant A capture.
        dead_port: The port the configuration named.

    Returns:
        Finding: PASS when the product refused and named no connection error.
    """
    if not attempt.ok:
        return _s11(1, Outcome.CANNOT_TEST, attempt.failure)
    if attempt.output.exit_code == 0:
        return _s11(1, Outcome.CANNOT_TEST,
                    "the broken configuration was accepted, so there was no "
                    "refusal whose ordering could be read")
    lowered = attempt.output.raw().lower()
    reached = [marker.decode("utf-8") for marker in CONNECTION_MARKERS
               if marker in lowered]
    if reached:
        return _s11(1, Outcome.FAIL,
                    "the refusal carries a connection failure against port %d "
                    "(%s), so a request was issued before the configuration "
                    "was rejected" % (dead_port, ", ".join(reached)))
    return _s11(1, Outcome.PASS, "")


def _all_at_once_finding(attempt):
    """Judge assertion 3 by counting the incompatibilities that were NAMED.

    Counting is the whole assertion. A message that names one of the three is
    exactly the one-at-a-time behaviour the guided migration exists to replace,
    and it is also a non-empty message -- so any check that only asked whether
    the product complained would pass over the defect.

    Args:
        attempt: The variant B capture.

    Returns:
        Finding: PASS when all three are named in one message.
    """
    if not attempt.ok:
        return _s11(2, Outcome.CANNOT_TEST, attempt.failure)
    if attempt.output.exit_code == 0:
        return _s11(2, Outcome.FAIL,
                    "a v0.11.0-era configuration was accepted")
    capture = attempt.output.raw()
    missing = [marker for marker in V011_MARKERS
               if marker.encode("utf-8") not in capture]
    if missing:
        return _s11(2, Outcome.FAIL,
                    "%d of %d incompatibilities named; missing %s: %s"
                    % (len(V011_MARKERS) - len(missing), len(V011_MARKERS),
                       ", ".join(missing), _excerpt(attempt.output)))
    return _s11(2, Outcome.PASS, "")


def _seat_lineage_finding(attempt):
    """Judge assertion 4: every seat missing a lineage is named, not just one.

    Args:
        attempt: The variant C capture.

    Returns:
        Finding: PASS when the refusal names all three seats.
    """
    if not attempt.ok:
        return _s11(3, Outcome.CANNOT_TEST, attempt.failure)
    if attempt.output.exit_code == 0:
        return _s11(3, Outcome.FAIL,
                    "three seats declared a model with no lineage and the "
                    "configuration was accepted, so a failure domain was "
                    "inferred rather than declared")
    lowered = attempt.output.raw().lower()
    missing = [seat for seat in SEATS if seat.encode("utf-8") not in lowered]
    if missing:
        return _s11(3, Outcome.FAIL,
                    "%d of %d seats named; missing %s: %s"
                    % (len(SEATS) - len(missing), len(SEATS),
                       ", ".join(missing), _excerpt(attempt.output)))
    return _s11(3, Outcome.PASS, "")


@scenario("S15")
def a_blank_environment_variable_is_absent_never_invalid(run):
    """Export each variable empty and blank, then export one filled and wrong.

    Args:
        run: Always ``None``; S15 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S15_ASSERTIONS`, in that order.
    """
    root = _seed_workspace()
    if root is None:
        for index in range(len(S15_ASSERTIONS)):
            yield _s15(index, Outcome.CANNOT_TEST,
                       "the product's %s did not scaffold a workspace to run "
                       "in" % INIT_SUBCOMMAND)
        return

    blanked = {}
    for name in BLANKABLE_VARIABLES:
        for spelling in BLANK_SPELLINGS:
            blanked[(name, spelling)] = _probe(
                root, {name: spelling},
                "s15-blank-%s" % name.lower(),
                keep_passphrase=name != PASSPHRASE_VARIABLE,
            )
    unrecognised = _probe(root, {"MAGI_PROVIDER": UNRECOGNISED_PROVIDER},
                          "s15-unrecognised")

    yield _falls_through_finding(blanked)
    yield _startup_succeeds_finding(blanked)
    yield _unrecognised_is_an_error_finding(unrecognised)


def _probe(root, overlay, label, keep_passphrase=True):
    """Run the product once with *overlay* on top of the environment.

    Args:
        root: The seeded workspace's parent directory.
        overlay: The variables to export for this invocation.
        label: What to call the invocation in the archive.
        keep_passphrase: Whether to supply the real passphrase. It is False for
            exactly one probe -- the one blanking the passphrase itself -- and
            supplying it there would overwrite the value under test.

    Returns:
        Attempt: The capture, or why there is none.
    """
    env = dict(overlay)
    if keep_passphrase:
        env.setdefault("MAGI_PASSPHRASE", runs.passphrase())
    return runs.attempt(
        [QUERY_SUBCOMMAND, "--output-format", "json",
         "--timeout", str(PRODUCT_TIMEOUT_S)],
        stdin=PROBE_PROMPT, timeout_s=QUERY_TIMEOUT_S, label=label, cwd=root,
        env=env,
    )


def _vocabulary_complaints(output):
    """Report which vocabulary refusals a capture carries.

    Complexity: ``O(len(capture) * len(VOCABULARY_MARKERS))``.

    Args:
        output: The capture to search.

    Returns:
        list[str]: The markers found, decoded, in declaration order.
    """
    lowered = output.raw().lower()
    return [marker.decode("utf-8") for marker in VOCABULARY_MARKERS
            if marker in lowered]


def _falls_through_finding(blanked):
    """Judge assertion 1 over every variable and both spellings of blank.

    Args:
        blanked: Each ``(variable, spelling)`` and what the product did.

    Returns:
        Finding: PASS when no blank value was read as a value, and the
        passphrase's own blank produced its typed refusal rather than a hang.
    """
    unreachable = [key for key, attempt in blanked.items() if not attempt.ok]
    if unreachable:
        return _s15(0, Outcome.CANNOT_TEST,
                    "%d probe(s) never completed, starting with %s: %s"
                    % (len(unreachable), unreachable[0][0],
                       blanked[unreachable[0]].failure))
    rejected = []
    for (name, spelling), attempt in sorted(blanked.items()):
        if name == PASSPHRASE_VARIABLE:
            rejected += _passphrase_complaints(spelling, attempt.output)
            continue
        complaints = _vocabulary_complaints(attempt.output)
        if complaints:
            rejected.append("%s=%r was read as a value (%s)"
                            % (name, spelling, ", ".join(complaints)))
    if rejected:
        return _s15(0, Outcome.FAIL, "; ".join(rejected))
    return _s15(0, Outcome.PASS, "")


def _passphrase_complaints(spelling, output):
    """Check the one variable whose fall-through has a different destination.

    Args:
        spelling: Which blank spelling was exported.
        output: What the product did with it.

    Returns:
        list[str]: What was wrong, empty when the typed refusal arrived.
    """
    if PASSPHRASE_UNAVAILABLE_MARKER not in output.raw().lower():
        return ["%s=%r did not produce the typed refusal: exit %d: %s"
                % (PASSPHRASE_VARIABLE, spelling, output.exit_code,
                   _excerpt(output))]
    if output.exit_code != PASSPHRASE_UNAVAILABLE_EXIT:
        return ["%s=%r refused with exit %d, expected %d"
                % (PASSPHRASE_VARIABLE, spelling, output.exit_code,
                   PASSPHRASE_UNAVAILABLE_EXIT)]
    return []


def _startup_succeeds_finding(blanked):
    """Judge assertion 2: nothing blank stopped the run during configuration.

    The passphrase probes are excluded from this one and named as excluded: a
    blank passphrase is SUPPOSED to stop the run, and folding it in would make
    the assertion contradict the previous one.

    Args:
        blanked: Each ``(variable, spelling)`` and what the product did.

    Returns:
        Finding: PASS when no probe died of a configuration complaint.
    """
    stopped = []
    for (name, spelling), attempt in sorted(blanked.items()):
        if name == PASSPHRASE_VARIABLE or not attempt.ok:
            continue
        complaints = _vocabulary_complaints(attempt.output)
        if complaints:
            stopped.append("%s=%r: %s" % (name, spelling,
                                          ", ".join(complaints)))
        elif attempt.output.exit_code == CONFIG_EXIT_CODE:
            stopped.append("%s=%r exited %d, the configuration-error code"
                           % (name, spelling, CONFIG_EXIT_CODE))
    if not any(attempt.ok for attempt in blanked.values()):
        return _s15(1, Outcome.CANNOT_TEST,
                    "no probe completed, so startup was never observed")
    if stopped:
        return _s15(1, Outcome.FAIL, "; ".join(stopped))
    return _s15(1, Outcome.PASS, "")


def _unrecognised_is_an_error_finding(attempt):
    """Judge assertion 3: a filled-in value nobody could mean still fails.

    Args:
        attempt: The capture of the probe carrying the bad value.

    Returns:
        Finding: PASS when the product refused it by name.
    """
    if not attempt.ok:
        return _s15(2, Outcome.CANNOT_TEST, attempt.failure)
    if attempt.output.exit_code == 0:
        return _s15(2, Outcome.FAIL,
                    "MAGI_PROVIDER=%r was accepted, so the blank rule reaches "
                    "values that are present and wrong" % UNRECOGNISED_PROVIDER)
    if not _vocabulary_complaints(attempt.output):
        return _s15(2, Outcome.FAIL,
                    "MAGI_PROVIDER=%r failed the run without a vocabulary "
                    "error, so it was not the value that was rejected: %s"
                    % (UNRECOGNISED_PROVIDER, _excerpt(attempt.output)))
    return _s15(2, Outcome.PASS, "")


def _s15(index, outcome, detail):
    """Build the finding for one entry of :data:`S15_ASSERTIONS`.

    Args:
        index: Position in :data:`S15_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S15 is standalone.
    """
    return Finding(assertion=S15_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)


def _excerpt(output, limit=600):
    """Render the beginning of a capture for a finding's detail.

    Args:
        output: The capture to quote.
        limit: How many bytes to keep.

    Returns:
        str: The first *limit* bytes of both streams, decoded leniently.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _s11(index, outcome, detail):
    """Build the finding for one entry of :data:`S11_ASSERTIONS`.

    Args:
        index: Position in :data:`S11_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S11 is standalone.
    """
    return Finding(assertion=S11_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)
