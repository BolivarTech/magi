# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S13 -- the documentation that ships is still true about the binary that ships.

The documents in the published package are the ones a user reads before they
have anything else to go on, and nothing executes them. A guide can therefore
go on telling people to run a flag that was retired two releases ago, and the
only signal is a bug report from somebody who copied it.

Three things about the shape of this scenario are deliberate.

**The product answers every question about its own surface.** The scenario
never carries a list of subcommands or flags; it reads them out of ``--help``,
per subcommand, at the moment it asks. A verifier holding its own copy of the
surface would age exactly the way the documents did, and then agree with them.

**The list of documents is derived, never written down** -- see
:func:`~smoke.docs_check.published_docs`. A hardcoded list stops covering a
document the moment somebody adds one, and nothing says so.

**A configuration "parses" when the product accepts it**, which is decided by
the exit code the product publishes for invalid input and nothing else. That
distinction is load-bearing: the README documents a ``base_url`` carrying
``[user]``/``[password]`` placeholders, which is correct and which fails at
run time here because this environment's vault has no such entries. A check
reading "did the run succeed" would report that correct example as a defect;
reading the invalid-input code separates a configuration the product refused
from one it accepted and could not act on.

One limit is declared rather than left to be discovered: every ``toml`` fence
in a published guide is treated as a candidate ``magi.toml``. Today every one
of them is. A fence holding some other TOML file would be reported as a defect
it is not, and the answer then is to narrow the rule here -- never to soften
the assertion.
"""

import pathlib
import re
import shutil
import tempfile

from smoke import runs
from smoke.docs_check import extract_configs, extract_invocations, published_docs
from smoke.errors import HarnessError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8. They reach the
#: certificate unchanged, so they are written once, here.
S13_ASSERTIONS = (
    "every magi-rs invocation in the published docs names an existing "
    "subcommand",
    "every flag in those invocations exists in that subcommand's --help",
    "every magi.toml embedded in those docs parses",
)

HELP_FLAG = "--help"
INIT_SUBCOMMAND = "init"
QUERY_SUBCOMMAND = "query"
MAGI_DIR_NAME = ".magi"
MAGI_TOML_NAME = "magi.toml"

#: What the product exits with when it refuses its input, configuration
#: included (README, REQ-H23). It is the ONLY signal assertion 3 reads: a
#: configuration the product accepted and then failed to act on -- an
#: unreachable backend, a vault entry this environment does not carry -- exits
#: with the runtime code instead, and is not a documentation defect.
INVALID_INPUT_EXIT = 2

#: How long the harness waits for one invocation, in seconds. ``--help`` parses
#: no configuration and opens no vault, so its bound is small; a configuration
#: probe may reach the backend when the configuration is accepted, so its bound
#: is the one that has to survive a slow provider.
HELP_TIMEOUT_S = 60
INIT_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 180

#: The wall clock handed to the product for a configuration probe. No assertion
#: reads the answer -- only whether the configuration was refused -- so this is
#: sized to return quickly rather than to let a model finish.
PRODUCT_TIMEOUT_S = 5

#: The prompt every configuration probe feeds the product. Nothing reads the
#: reply.
PROBE_PROMPT = b"say ok\n"

#: Where clap lists nested subcommands, and where it lists options. Both
#: headings are matched exactly, because a description line mentioning either
#: word must not open a section.
COMMANDS_HEADING = "Commands:"
OPTIONS_HEADING = "Options:"

#: One entry of a clap section. The spec sits at a shallow indent and its
#: description, when clap puts it on its own line, sits at a deeper one -- so
#: anchoring on the indent is what stops prose inside a long description from
#: being read as another flag. That direction of error is the dangerous one:
#: an over-collected flag set ACCEPTS a flag the product does not have.
#:
#: The comma is excluded from the first group deliberately. ``-i, --input`` has
#: no whitespace before the comma, so a ``\S+`` first group swallows it and the
#: optional second group can never match -- every long spelling silently
#: disappears from the answer, and the check then rejects every long flag the
#: documents use.
_ENTRY = re.compile(r"^ {2,6}([^\s,]+)(?:,\s+(\S+))?")
_STRIPS_VALUE = re.compile(r"[=<\[].*$")


@scenario("S13", assertions=S13_ASSERTIONS)
def the_published_documentation_is_still_true(run):
    """Check every published guide against the binary's own surface.

    Args:
        run: Always ``None``; S13 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S13_ASSERTIONS`, in that order. Always
        three, whatever went wrong: a scenario that stops early erases the
        assertions it never reached from the report.
    """
    try:
        documents = published_docs(runs.repo_root())
    except HarnessError as exc:
        for index in range(len(S13_ASSERTIONS)):
            yield _finding(index, Outcome.CANNOT_TEST,
                           "the published file list could not be read: %s" % exc)
        return

    invocations = _read_invocations(documents)
    surface = _Surface()
    root = surface.help_for(())
    if root is None:
        for index in (0, 1):
            yield _finding(index, Outcome.CANNOT_TEST,
                           "the product did not answer %s, so its surface "
                           "could not be read" % HELP_FLAG)
    else:
        yield _subcommand_finding(invocations, _subcommands_in(root))
        yield _flag_finding(invocations, surface)
    yield _config_finding(documents)


def _read_invocations(documents):
    """Collect every invocation of every published document.

    Complexity: ``O(total bytes of the documents)``.

    Args:
        documents: Relative paths, as :func:`published_docs` returns them.

    Returns:
        list[tuple[pathlib.Path, Invocation]]: Each invocation with the
        document it came from. A document that cannot be read is skipped
        rather than raised on: the file list came from cargo and the file is
        supposed to be there, but the harness accusing the product of a
        documentation defect because of an unreadable file would be a lie.
    """
    collected = []
    root = runs.repo_root()
    for relative in documents:
        try:
            text = (root / relative).read_text(encoding="utf-8")
        except OSError:
            continue
        for invocation in extract_invocations(text):
            collected.append((relative, invocation))
    return collected


class _Surface:
    """The product's own account of its subcommands and flags.

    Every answer comes from ``--help`` and is cached for the run, so a document
    naming ``vault set`` twenty times costs one invocation.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._help: dict[tuple[str, ...], str | None] = {}

    def help_for(self, path: tuple[str, ...]) -> str | None:
        """The help text for one subcommand path.

        Args:
            path: The subcommand words, empty for the root.

        Returns:
            str | None: What the product printed, or None when it could not be
            asked -- which includes a path that is not a subcommand at all, so
            the caller reads None as "stop walking here".
        """
        if path not in self._help:
            attempt = runs.attempt(
                list(path) + [HELP_FLAG], stdin=b"", timeout_s=HELP_TIMEOUT_S,
                label="s13-help-%s" % ("root" if not path else "-".join(path)))
            if not attempt.ok or attempt.output.exit_code != 0:
                self._help[path] = None
            else:
                self._help[path] = attempt.output.stdout.decode(
                    "utf-8", errors="replace")
        return self._help[path]

    def resolve(self, words) -> tuple[str, ...]:
        """The deepest subcommand path *words* actually names.

        The product's surface is two levels deep and the second level owns
        flags the first does not: ``-f`` belongs to ``vault set`` and not to
        ``vault``. Walking as far as the product admits, and stopping at the
        first word it does not list, is what separates a nested subcommand from
        a positional argument -- ``vault set ANTHROPIC_API_KEY`` names two
        subcommands and one secret.

        Complexity: ``O(depth)`` invocations, each cached.

        Args:
            words: The invocation's non-flag words.

        Returns:
            tuple[str, ...]: The resolved path, possibly empty.
        """
        path: tuple[str, ...] = ()
        for word in words:
            text = self.help_for(path)
            if text is None or word not in _subcommands_in(text):
                break
            path = path + (word,)
        return path


def _section(text: str, heading: str) -> list[str]:
    """The lines of one clap help section.

    Complexity: ``O(lines)``.

    Args:
        text: The help output.
        heading: ``Commands:`` or ``Options:``.

    Returns:
        list[str]: The section's lines, up to the next blank-line-separated
        heading.
    """
    collected: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.strip() == heading:
            inside = True
            continue
        if not inside:
            continue
        if line and not line.startswith(" "):
            break
        collected.append(line)
    return collected


def _subcommands_in(text: str) -> set[str]:
    """Every subcommand the product lists under ``Commands:``.

    Complexity: ``O(lines)``.

    Args:
        text: One ``--help`` output.

    Returns:
        set[str]: The subcommand names.
    """
    found = set()
    for line in _section(text, COMMANDS_HEADING):
        match = _ENTRY.match(line)
        if match is not None and not match.group(1).startswith("-"):
            found.add(match.group(1))
    return found


def _flags_in(text: str) -> set[str]:
    """Every flag the product lists under ``Options:``.

    Complexity: ``O(lines)``.

    Args:
        text: One ``--help`` output.

    Returns:
        set[str]: The short and long spellings, without their value
        placeholders.
    """
    found = set()
    for line in _section(text, OPTIONS_HEADING):
        match = _ENTRY.match(line)
        if match is None:
            continue
        for group in match.groups():
            if group and group.startswith("-"):
                found.add(_STRIPS_VALUE.sub("", group).rstrip(","))
    return found


def _subcommand_finding(invocations, root_subcommands):
    """Assertion 1: every invocation names a subcommand the product has.

    An invocation naming NO subcommand -- ``magi-rs --version``, or the bare
    launch -- is legitimate and is not judged here. Whether the flags it
    carries exist is assertion 2's question.

    Args:
        invocations: Pairs of document and invocation.
        root_subcommands: What the root ``--help`` lists.

    Returns:
        Finding: PASS, or FAIL naming every offender with its location.
    """
    offenders = [
        "%s:%d names subcommand %r, which the binary does not have (%s)"
        % (relative.as_posix(), item.line, item.subcommand, item.source)
        for relative, item in invocations
        if item.subcommand and item.subcommand not in root_subcommands
    ]
    if offenders:
        return _finding(0, Outcome.FAIL, "; ".join(offenders))
    return _finding(0, Outcome.PASS, "")


def _flag_finding(invocations, surface):
    """Assertion 2: every flag exists in the help of what it is passed to.

    Args:
        invocations: Pairs of document and invocation.
        surface: The product's cached account of itself.

    Returns:
        Finding: PASS, or FAIL naming every offender with its location.
    """
    offenders = []
    for relative, item in invocations:
        path = surface.resolve(item.words)
        text = surface.help_for(path)
        if text is None:
            offenders.append(
                "%s:%d could not be checked: the product did not answer %s for "
                "%r" % (relative.as_posix(), item.line, HELP_FLAG, " ".join(path)))
            continue
        available = _flags_in(text)
        for flag in item.flags:
            if flag not in available:
                offenders.append(
                    "%s:%d passes %s to %r, which does not have it (%s)"
                    % (relative.as_posix(), item.line, flag,
                       " ".join(path) or "the root command", item.source))
    if offenders:
        return _finding(1, Outcome.FAIL, "; ".join(offenders))
    return _finding(1, Outcome.PASS, "")


def _config_finding(documents):
    """Assertion 3: every embedded configuration is one the product accepts.

    Args:
        documents: Relative paths, as :func:`published_docs` returns them.

    Returns:
        Finding: PASS, FAIL naming every rejected configuration, or
        CANNOT_TEST when no workspace could be scaffolded to install them in.
    """
    embedded = _read_configs(documents)
    if not embedded:
        return _finding(2, Outcome.PASS, "")
    root = _seed_workspace()
    if root is None:
        return _finding(
            2, Outcome.CANNOT_TEST,
            "the product's %s did not scaffold a workspace to install the "
            "documented configurations in" % INIT_SUBCOMMAND)
    try:
        offenders = []
        for relative, line, body in embedded:
            verdict = _install_and_probe(root, body, relative, line)
            if verdict:
                offenders.append(verdict)
    finally:
        # Removed here, and the scratch area's own reset is not an argument
        # against it: --reset-env is a recovery an operator runs, not a
        # cleanup this scenario is entitled to defer to. One workspace per
        # run, forever, is the kind of growth nobody notices until a disk
        # does -- and the harness already refuses that bargain for its own
        # temporary directories.
        shutil.rmtree(root, ignore_errors=True)
    if offenders:
        return _finding(2, Outcome.FAIL, "; ".join(offenders))
    return _finding(2, Outcome.PASS, "")


def _read_configs(documents):
    """Every embedded configuration, with where it came from.

    Complexity: ``O(total bytes of the documents)``.

    Args:
        documents: Relative paths.

    Returns:
        list[tuple[pathlib.Path, int, str]]: The document, the 1-based index of
        the fence within it, and the body. The index is what lets a finding
        name WHICH block of a document was rejected.
    """
    collected = []
    root = runs.repo_root()
    for relative in documents:
        try:
            text = (root / relative).read_text(encoding="utf-8")
        except OSError:
            continue
        for index, body in enumerate(extract_configs(text), 1):
            collected.append((relative, index, body))
    return collected


def _seed_workspace():
    """Scaffold one workspace under the scratch area, by cwd and never by flag.

    Returns:
        pathlib.Path | None: The directory holding the new ``.magi/``, or None
        when the product did not create one.
    """
    root = pathlib.Path(
        tempfile.mkdtemp(prefix="s13-", dir=str(runs.scratch_root()))
    )
    seeded = runs.attempt([INIT_SUBCOMMAND], stdin=b"",
                          timeout_s=INIT_TIMEOUT_S, label="s13-init", cwd=root,
                          env={"MAGI_PASSPHRASE": runs.passphrase()})
    if not seeded.ok or not (root / MAGI_DIR_NAME).is_dir():
        return None
    return root


def _install_and_probe(root, body, relative, index):
    """Install one documented configuration and ask the product to load it.

    Args:
        root: The seeded workspace's parent directory.
        body: The configuration text.
        relative: The document it came from.
        index: Which fence of that document it is.

    Returns:
        str: Empty when the product accepted the configuration; otherwise the
        offence, naming the document, the block and what the product said.
    """
    path = root / MAGI_DIR_NAME / MAGI_TOML_NAME
    where = "%s block %d" % (relative.as_posix(), index)
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return "%s could not be installed: %s" % (where, exc)
    attempt = runs.attempt(
        [QUERY_SUBCOMMAND, "--output-format", "json",
         "--timeout", str(PRODUCT_TIMEOUT_S)],
        stdin=PROBE_PROMPT, timeout_s=PROBE_TIMEOUT_S,
        label="s13-config-%s-%d" % (relative.stem.lower(), index), cwd=root,
        env={"MAGI_PASSPHRASE": runs.passphrase()},
    )
    if not attempt.ok:
        return "%s could not be checked: %s" % (where, attempt.failure)
    if attempt.output.exit_code != INVALID_INPUT_EXIT:
        return ""
    return "%s is rejected by the product: %s" % (
        where, _first_line(attempt.output.stderr))


def _first_line(stream: bytes) -> str:
    """The first non-empty line the product printed.

    The whole refusal is several paragraphs of migration guidance, and the
    archived run carries it in full. A finding wants the diagnosis: pasting the
    paragraphs into a one-line detail makes the list of offenders unreadable,
    and the guidance's own semicolons make it unclear where one offender ends
    and the next begins.

    Args:
        stream: Captured bytes.

    Returns:
        str: The first non-empty line, or the empty string when there is none.
    """
    for line in stream.decode("utf-8", errors="replace").splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def _finding(index: int, outcome: Outcome, detail: str) -> Finding:
    """Build one of S13's three findings.

    Args:
        index: Which entry of :data:`S13_ASSERTIONS` this is.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: With no run id, because S13 declares no shared run.
    """
    return Finding(assertion=S13_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)
