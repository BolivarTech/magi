# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Extraction of invocations and configurations from the published guides.

This module parses text and nothing else. It never runs the product and never
decides whether a subcommand exists -- that judgement belongs to the scenario,
which can ask the binary. Keeping the two apart is what lets the parsing rules
be tested in milliseconds against strings, and what stops the verifier growing
its own idea of the product's surface, which is the thing that would age
exactly as the documents did.

The scope is deliberately narrow: an invocation is extracted so the scenario
can ask whether what it names EXISTS. Whether the example does what the
surrounding prose promises is not checked, because that is executing prose, and
it is where a documentation verifier turns brittle and gets switched off.
"""

import dataclasses
import pathlib
import re
import shlex
import subprocess

from smoke.errors import HarnessError

#: The program name a reader would type. Matched on the token's BASENAME, so a
#: path-qualified spelling in a transcript is recognised too.
PROGRAM_NAMES = ("magi-rs", "magi-rs.exe")

#: Fenced-block languages whose body is a candidate ``magi.toml``. A ``bash``
#: or ``sql`` fence is not a configuration however much TOML-shaped text it
#: happens to contain.
CONFIG_LANGUAGES = ("toml",)

#: How ``cargo`` is asked what travels in the package. ``--allow-dirty`` is not
#: optional: the harness runs against a working tree somebody is editing, and
#: without it cargo refuses and the scenario reports the developer's own
#: uncommitted work as a documentation defect.
PACKAGE_COMMAND = ("cargo", "package", "--list", "--allow-dirty")

#: How long cargo is given to answer, in seconds. It reads the manifest and
#: walks the tree; anything past this is hung rather than slow.
PACKAGE_TIMEOUT_SECONDS = 300

#: The one subtree ``cargo package --list`` reports that is NOT documentation.
#: It ships so the crate's own test suite has its inputs. Checking those inputs
#: against today's surface would report defects whose only correct fix is to
#: falsify a record: ``tests/fixtures/v0.11.0/README.md`` documents the
#: published v0.11.0 binary, and every invocation in it is a true statement
#: about that release. The filter names the TREE rather than the file, so a
#: document added there is excluded for the same reason and a document added
#: anywhere else is checked without anyone editing this module.
NON_DOCUMENTATION_TREE = "tests/"

MARKDOWN_SUFFIX = ".md"

#: Shell operators that end one command and start the next. A published guide
#: pipes a prompt into the product, so the program is the first word of a
#: SEGMENT and not of the line.
SEGMENT_SEPARATORS = ("|", "||", "&&", ";", "&")

_FENCE = re.compile(r"^\s*```([A-Za-z0-9_+-]*)\s*$")
_COMMENT = re.compile(r"^\s*[#>]")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_FLAG = re.compile(r"^-{1,2}[A-Za-z0-9]")
_TRAILING_CONTINUATION = re.compile(r"\\\s*$")


@dataclasses.dataclass(frozen=True)
class Invocation:
    """One command line in a published guide that runs the product.

    ``subcommand`` is a PROPERTY rather than a field, and that is the point:
    two attributes holding the same fact drift the first time somebody edits
    one of them, and nothing in Python would say so.

    Attributes:
        words: Every non-flag word after the program name, in order. The whole
            run is kept and not only the first because the product's surface is
            two levels deep -- ``vault set`` owns flags ``vault`` does not, so a
            verifier holding only ``vault`` would look ``-f`` up in the wrong
            help text and report a flag the product has as one it lacks.
        flags: Every flag, normalised: ``--flag=value`` arrives as ``--flag``,
            because what is being asked is whether the flag exists.
        source: The command line as written, for the finding's detail.
        line: The 1-based line the command was written on.
    """

    words: tuple[str, ...]
    flags: tuple[str, ...]
    source: str
    line: int

    @property
    def subcommand(self) -> str:
        """The first subcommand the invocation names.

        Returns:
            str: ``words[0]``, or the empty string when the invocation names
            none. ``magi-rs --version`` naming none is legitimate, so the empty
            string is an answer rather than an error.
        """
        return self.words[0] if self.words else ""


def _fenced_blocks(text: str):
    """Yield each fenced code block with its language and starting line.

    Complexity: ``O(lines)``.

    Args:
        text: The document.

    Yields:
        tuple[str, int, list[str]]: The lowercased language (empty when the
        fence declares none), the 1-based line of the first body line, and the
        body lines. An unclosed fence at the end of the document is yielded
        too: a document ending mid-block still has commands in it.
    """
    language = None
    first_line = 0
    body: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = _FENCE.match(line)
        if match is not None and language is None:
            language = match.group(1).lower()
            first_line = number + 1
            body = []
            continue
        if match is not None:
            yield language, first_line, body
            language = None
            continue
        if language is not None:
            body.append(line)
    if language is not None:
        yield language, first_line, body


def _tokenize(line: str) -> list[str]:
    """Split one command line into tokens, degrading rather than raising.

    Documentation is prose with code in it, not a shell script: a block holding
    an unterminated quote is a real thing to write, and a tokenizer error there
    must not take the scenario down. The fallback splits on whitespace, which
    reads the line less precisely and never fails.

    Args:
        line: The command line, without its fence.

    Returns:
        list[str]: The tokens, with any trailing comment removed.
    """
    stripped = _TRAILING_CONTINUATION.sub("", line)
    try:
        return shlex.split(stripped, comments=True, posix=True)
    except ValueError:
        return stripped.split()


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a token list at shell operators.

    Args:
        tokens: One command line's tokens.

    Returns:
        list[list[str]]: One token list per command in the line.
    """
    found: list[list[str]] = [[]]
    for token in tokens:
        if token in SEGMENT_SEPARATORS:
            found.append([])
        else:
            found[-1].append(token)
    return found


def _is_product(token: str) -> bool:
    """Whether *token* names the product's binary.

    Args:
        token: A shell token.

    Returns:
        bool: True when its basename is one of :data:`PROGRAM_NAMES`. Both
        separators are handled because a Windows transcript writes one and a
        POSIX one writes the other.
    """
    basename = token.replace("\\", "/").rsplit("/", 1)[-1]
    return basename in PROGRAM_NAMES


def _invocation_from(segment: list[str], line: int, source: str):
    """Build an Invocation from one command, or report that it is not one.

    Args:
        segment: The command's tokens.
        line: The line it was written on.
        source: The line as written.

    Returns:
        Invocation | None: The invocation, or None when this command runs
        something other than the product. The program is the FIRST word after
        any environment assignments, so the product named in an argument
        position -- ``cargo install magi-rs`` -- is never read as one.
    """
    index = 0
    while index < len(segment) and _ENV_ASSIGNMENT.match(segment[index]):
        index += 1
    if index >= len(segment) or not _is_product(segment[index]):
        return None
    words: list[str] = []
    flags: list[str] = []
    for token in segment[index + 1:]:
        if _FLAG.match(token):
            flags.append(token.split("=", 1)[0])
        else:
            words.append(token)
    return Invocation(words=tuple(words), flags=tuple(flags), source=source,
                      line=line)


def extract_invocations(text: str) -> list[Invocation]:
    """Every command in *text* that runs the product.

    Only fenced code blocks are read. Naming the binary in a sentence is not
    invoking it, and the distinction matters in both directions: a paragraph
    explaining that a retired flag no longer exists must not be reported as a
    use of it.

    Complexity: ``O(lines)``, one tokenizer pass per fenced line.

    Args:
        text: The document.

    Returns:
        list[Invocation]: In the order they appear.

    Example:
        >>> extract_invocations("```bash\\nmagi-rs vault ls\\n```")[0].subcommand
        'vault'
    """
    found: list[Invocation] = []
    for _language, first_line, body in _fenced_blocks(text):
        for offset, raw in enumerate(body):
            if _COMMENT.match(raw) or not raw.strip():
                continue
            for segment in _segments(_tokenize(raw)):
                invocation = _invocation_from(segment, first_line + offset,
                                              raw.strip())
                if invocation is not None:
                    found.append(invocation)
    return found


def extract_configs(text: str) -> list[str]:
    """Every fenced body in *text* that is a candidate ``magi.toml``.

    Complexity: ``O(lines)``.

    Args:
        text: The document.

    Returns:
        list[str]: The bodies, in the order they appear.
    """
    return [
        "\n".join(body)
        for language, _first_line, body in _fenced_blocks(text)
        if language in CONFIG_LANGUAGES
    ]


def published_docs(repo_root) -> list[pathlib.Path]:
    """Every markdown document that travels inside the published package.

    Derived from ``cargo package --list`` rather than hardcoded. A hardcoded
    list ages, and its failure mode is a document that quietly stops being
    checked -- which is how the guides this scenario exists for drifted in the
    first place.

    Complexity: ``O(files cargo reports)``.

    Args:
        repo_root: The checkout to ask about.

    Returns:
        list[pathlib.Path]: Relative paths, in the order cargo reported them,
        excluding :data:`NON_DOCUMENTATION_TREE`.

    Raises:
        HarnessError: If cargo cannot be run, exits non-zero, or does not
            answer within :data:`PACKAGE_TIMEOUT_SECONDS`. This is the
            harness's own dependency, so the scenario turns it into
            ``CANNOT_TEST`` rather than accusing the product of anything.
    """
    try:
        completed = subprocess.run(
            list(PACKAGE_COMMAND),
            cwd=str(repo_root),
            capture_output=True,
            timeout=PACKAGE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(
            "could not ask cargo which files the package carries: %s" % exc
        ) from exc
    if completed.returncode != 0:
        raise HarnessError(
            "%s exited %d in %s"
            % (" ".join(PACKAGE_COMMAND), completed.returncode, repo_root)
        )
    listed = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        entry = line.strip().replace("\\", "/")
        if not entry.endswith(MARKDOWN_SUFFIX):
            continue
        if entry.startswith(NON_DOCUMENTATION_TREE):
            continue
        listed.append(pathlib.Path(entry))
    return listed
