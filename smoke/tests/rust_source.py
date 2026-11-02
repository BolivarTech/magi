# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-11-02
"""Reading the product's Rust source for the harness's sync guards.

The harness carries copies of the product's constants and wording, and the
guards that keep them true read the Rust tree. A plain ``re.search`` over a
whole file is not such a guard: a value that drifted while its old line
survived in a ``///`` comment above the new one was read from the comment, and
a definition repeated somewhere else was never noticed. Every presence check a
guard makes goes through :func:`one_definition` or :func:`one_occurrence`,
which look only at code lines and insist on exactly one match; a list a guard
collects with ``findall`` is first passed through :func:`code_only`.

A comment line is one whose first non-blank characters are ``//`` (``//``,
``///``, ``//!``). The reading is by line, not a Rust lexer, and the limits
that follow from it are declared rather than handled:

- Block comments (``/* ... */``) are not recognised: a definition inside one
  would count as code. The product does not put one where these guards look
  (an inline ``/* safe */`` argument is ordinary code to them).
- A line inside a multi-line (raw) string literal that starts with ``//`` is
  taken for a comment and skipped.
- Uniqueness is per line: two matches on one physical line count once.
  rustfmt, which the product's CI enforces, puts each item on its own line.
- :func:`one_occurrence` reads a line only up to its first ``//``, whether or
  not that opens a comment, so a needle after a string holding ``//`` (a URL)
  is not found. That fails loudly, never as a false pass.

Example:
    >>> text = "/// was `const N: u8 = 1;`\\nconst N: u8 = 2;\\n"
    >>> one_definition(text, r"const N: u8 = (\\d+);", "n.rs").group(1)
    '2'
"""

import pathlib
import re

#: The repository root, three levels above this file.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: What a comment line starts with, after its indentation.
_COMMENT = "//"

#: Code on one line up to, never across, the start of a trailing comment.
_CODE_BEFORE = r"(?:(?!//)[^\n])*?"


def read_source(relative: str, root: pathlib.Path | None = None) -> str:
    """Read a source file with its line endings normalised.

    rustfmt writes CRLF on Windows and LF elsewhere, so ``\\r`` is stripped
    before any pattern runs. Bytes that are not UTF-8 (a test literal) are
    replaced; nothing a guard reads lives in them.

    Args:
        relative: The path under *root*, with forward slashes.
        root: The repository root; :data:`REPO_ROOT` when ``None``.

    Returns:
        str: The file's text.

    Raises:
        OSError: When the file cannot be read.
    """
    base = REPO_ROOT if root is None else root
    return (base / relative).read_bytes().decode("utf-8", errors="replace").replace("\r", "")


def code_only(text: str) -> str:
    """*text* with every comment line blanked, every other line kept in place.

    A line with code before a trailing comment is kept whole: it is code.

    Complexity: ``O(len(text))``.

    Args:
        text: Source text, ``\\r`` stripped.

    Returns:
        str: The same number of lines; comment lines are empty.

    Example:
        >>> code_only('x = [\\n  "a",\\n  // "b",\\n]\\n')
        'x = [\\n  "a",\\n\\n]\\n'
    """
    return "\n".join("" if line.lstrip().startswith(_COMMENT) else line
                     for line in text.split("\n"))


def _exactly_one(regex: re.Pattern, text: str, where: str) -> re.Match:
    """The single match of *regex* in *text* that does not sit on a comment line.

    Args:
        regex: A pattern anchored at a line start (``re.M``).
        text: Source text.
        where: The file the text came from, for the failure message.

    Returns:
        re.Match: The one match.

    Raises:
        AssertionError: When there is no such match or more than one, naming
            *where*, the pattern and the count.
    """
    found = [match for match in regex.finditer(text)
             if not match.group(0).lstrip().startswith(_COMMENT)]
    if len(found) != 1:
        raise AssertionError("%s: expected exactly one code line matching %r, found %d"
                             % (where, regex.pattern, len(found)))
    return found[0]


def one_definition(text: str, pattern: str, where: str, flags: int = 0) -> re.Match:
    """The one code line that *pattern* starts, after the line's indentation.

    The pattern is anchored with ``^[ \\t]*`` under ``re.M``: it must be the
    first thing on its line, so an old copy left in a comment, or quoted mid
    line, never answers for the real definition. The pattern's own groups are
    available on the result.

    Complexity: ``O(len(text))`` for a pattern without catastrophic
    backtracking.

    Args:
        text: Source text, ``\\r`` stripped.
        pattern: A regular expression for the start of the line.
        where: The file the text came from, for the failure message.
        flags: Extra ``re`` flags, e.g. ``re.S`` for a body spanning lines.

    Returns:
        re.Match: The single match.

    Raises:
        AssertionError: When no code line matches, or more than one does.
    """
    return _exactly_one(re.compile(r"^[ \t]*(?:%s)" % pattern, re.M | flags), text, where)


def one_occurrence(text: str, needle: str, where: str) -> re.Match:
    """The one code line that holds the literal *needle* before the line's first ``//``.

    For wording a guard pins inside a larger line, such as a message in an
    ``#[error(...)]`` attribute. The needle may span lines. The scan stops at
    the first ``//`` on the line even inside a string literal, so a needle
    after a URL on the same line is reported missing (see the module notes).

    Complexity: ``O(len(text) * len(needle))`` in the worst case.

    Args:
        text: Source text, ``\\r`` stripped.
        needle: The exact text to find.
        where: The file the text came from, for the failure message.

    Returns:
        re.Match: The match, from the line start through the needle.

    Raises:
        AssertionError: When no code line holds it, or more than one does.
    """
    regex = re.compile(r"^[ \t]*%s%s" % (_CODE_BEFORE, re.escape(needle)), re.M)
    return _exactly_one(regex, text, where)
