# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S11 -- a broken configuration cuts before anything runs.

Protects ``config/migrate.rs`` and the rule that a present-but-unparseable
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
