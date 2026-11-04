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
collects with ``findall`` is first passed through :func:`without_comments`,
which also blanks a trailing ``//`` comment.

A comment line is one whose first non-blank characters are ``//`` (``//``,
``///``, ``//!``). The reading is by line, not a Rust lexer, and the limits
that follow from it are declared rather than handled:

- Block comments (``/* ... */``, nested as Rust nests them) are blanked
  before any pattern runs, by a scan that skips string, raw-string and char
  literals so an opener inside one (``"src/**/*.rs"``) opens nothing; a
  lifetime (``'a``) is not taken for a char literal. An unterminated block
  comment fails loudly. [MAGI §6 MS2 loop 28, F1b2a-caspar] So does an
  unterminated string or raw string, which would otherwise hide every block
  comment after it; a byte or C string literal (``b"/*"``, ``c"/*"``,
  ``cr#"/*"#``) is read like the plain or raw one.
  [MAGI §6 MS2 loop 29, F1b2a-caspar, F1b2a-balthasar]
- A guard asking whether a phrase or a key appears in code reads
  :func:`without_comments`, which also blanks a trailing ``//`` comment;
  :func:`code_only` keeps such a line whole. [MAGI §6 MS2 loop 29,
  F1b2a-melchior]
- A line inside a multi-line (raw) string literal that starts with ``//`` is
  taken for a comment and skipped. :func:`one_definition` skips every line
  that starts inside such a literal; :func:`one_occurrence` does not, so a
  needle pinned inside a multi-line string is still found. [MAGI F1b2a]
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

import bisect
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
    base = pathlib.Path(REPO_ROOT if root is None else root).resolve()
    path = (base / relative).resolve()
    # MAGI FIX (S6 loop 28): any relative path was joined to the root, ".."
    # and absolute paths included (F1b2a-balthasar).
    if base not in path.parents:
        raise ValueError("%s is outside %s" % (relative, base))
    return path.read_bytes().decode("utf-8", errors="replace").replace("\r", "")


#: A Rust char literal at the start of the text: one character or one escape
#: (``\\n``, ``\\'``, ``\\x41``, ``\\u{e9}``...), between single quotes. A
#: lifetime (``'a``) has no closing quote right after one character.
_CHAR_LITERAL = re.compile(
    r"'(\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]{1,6}\}|[nrt0\\'\"])|[^\\'\n])'")

#: A raw string's opening: ``r``, ``br`` or ``cr``, any number of ``#``, then
#: ``"``. [F1b2a] Without ``cr`` the ``r`` of a C raw string followed a letter
#: and was not taken for one.
_RAW_STRING = re.compile(r'[bc]?r(#*)"')

#: The largest code point, and the surrogate range a Rust ``char`` cannot hold.
_MAX_CODE_POINT = 0x10FFFF
_SURROGATES = (0xD800, 0xDFFF)

#: The escapes :func:`char_list` unescapes, besides ``\\x`` and ``\\u{}``.
_SIMPLE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": "\0",
                   "\\": "\\", "'": "'", '"': '"'}


def _blank(span: str) -> str:
    """*span* with every character but a newline turned into a space."""
    return re.sub(r"[^\n]", " ", span)


def _block_comment_end(text: str, start: int) -> int:
    """Where the block comment opening at *start* ends, nesting included.

    Args:
        text: Source text.
        start: The index of its ``/*``.

    Returns:
        int: The index just past its closing ``*/``.

    Raises:
        AssertionError: When it is never closed.
    """
    depth, index = 0, start
    while index < len(text):
        if text.startswith("/*", index):
            depth, index = depth + 1, index + 2
        elif text.startswith("*/", index):
            depth, index = depth - 1, index + 2
            if depth == 0:
                return index
        else:
            index += 1
    raise AssertionError("unterminated block comment at offset %d" % start)


def _literal_end(text: str, index: int) -> int:
    """Where the string, raw-string or char literal starting at *index* ends, or *index* if none starts.

    Args:
        text: Source text.
        index: A position in it.

    A byte string, byte char or C string (``b"..."``, ``b'.'``, ``c"..."``)
    is read as its prefix followed by the plain literal, which ends at the
    same place; ``br"..."`` and ``cr"..."`` are raw strings.

    MAGI FIX (S6 loop 29): a string or raw string that never closed ran to the
    end of the file, so no block comment after it was blanked and an old
    definition left in one answered for a real definition (F1b2a-caspar).
    The product compiles, so a literal that does not close means the scan
    misread the file; it now fails loudly.

    Returns:
        int: The index just past the literal; *index* itself when no literal
        starts there (a lifetime's quote included).

    Raises:
        AssertionError: When a string or raw string starting at *index* is
            never closed.
    """
    raw = _RAW_STRING.match(text, index)
    if raw is not None and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")):
        closing = '"' + raw.group(1)
        end = text.find(closing, raw.end())
        if end == -1:
            raise AssertionError("unterminated raw string literal at offset %d" % index)
        return end + len(closing)
    if text.startswith('"', index):
        cursor = index + 1
        while cursor < len(text) and text[cursor] != '"':
            cursor += 2 if text[cursor] == "\\" else 1
        if cursor >= len(text):
            raise AssertionError("unterminated string literal at offset %d" % index)
        return cursor + 1
    char = _CHAR_LITERAL.match(text, index)
    return char.end() if char is not None else index


def _without_block_comments(text: str, line_comments: bool = False, literals: list | None = None) -> str:
    """*text* with every block comment blanked, its newlines kept.

    Line comments, string and char literals are skipped whole, so a ``/*``
    inside one opens nothing.

    Complexity: ``O(len(text))``.

    Args:
        text: Source text.
        line_comments: Whether every ``//`` comment, trailing ones included,
            is blanked too (:func:`without_comments`); kept when ``False``.
        literals: A list that receives the ``(start, end)`` span of every
            string, raw-string and char literal, in order; ``None`` records
            nothing.

    Raises:
        AssertionError: When a block comment or a string literal is never
            closed.
    """
    out, index = [], 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end == -1 else end
            out.append(_blank(text[index:end]) if line_comments else text[index:end])
        elif text.startswith("/*", index):
            end = _block_comment_end(text, index)
            out.append(_blank(text[index:end]))
        else:
            end = _literal_end(text, index)
            if end == index:
                end = index + 1
            elif literals is not None:
                literals.append((index, end))
            out.append(text[index:end])
        index = end
    return "".join(out)


def without_comments(text: str) -> str:
    """*text* with every comment blanked, trailing line comments included, every line kept in place.

    MAGI FIX (S6 loop 29): a guard that asks whether a phrase appears in code
    read :func:`code_only`, which keeps a line whole when code precedes a
    trailing ``//`` comment, so the phrase surviving only in such a comment
    passed (F1b2a-melchior). Here a ``//`` outside a literal starts a comment
    wherever it sits; ``//`` inside a string (a URL) is code.

    Complexity: ``O(len(text))``.

    Args:
        text: Source text, ``\\r`` stripped.

    Returns:
        str: The same text with each comment's characters turned into spaces.

    Raises:
        AssertionError: When a block comment or a string literal is never closed.

    Example:
        >>> without_comments('let u = "a//b"; // note\\n')
        'let u = "a//b";        \\n'
    """
    return _without_block_comments(text, line_comments=True)


def structure_only(text: str) -> str:
    """*text* with every comment and every literal blanked, at the same offsets.

    For a guard that locates structure (a closing brace) by searching the
    text: a brace line inside a multi-line string or a block comment is not
    the code's. Offsets match *text*, so a span found here slices the source.

    Complexity: ``O(len(text))``.

    Args:
        text: Source text, ``\\r`` stripped.

    Returns:
        str: The same text with each comment's and literal's characters,
        newlines excepted, turned into spaces.

    Raises:
        AssertionError: When a block comment or a string literal is never closed.

    Example:
        >>> structure_only('let s = "}";\\n')
        'let s =    ;\\n'
    """
    spans = []
    view = _without_block_comments(text, line_comments=True, literals=spans)
    out, previous = [], 0
    for start, end in spans:
        out.append(view[previous:start])
        out.append(_blank(view[start:end]))
        previous = end
    out.append(view[previous:])
    return "".join(out)


def code_only(text: str) -> str:
    """*text* with every comment line and block comment blanked, every line kept in place.

    A line with code before a trailing ``//`` comment is kept whole: it is
    code. A block comment is blanked wherever it sits
    (:func:`_without_block_comments`).

    Declared limit (MAGI S6 loop 37): a line inside a multi-line string
    literal that starts with ``//`` is blanked too, as if it were a comment.
    Such a line is fixture text, not code, so hiding it is what the pins
    want; a pin that did need it would fail loudly (a false red).

    Complexity: ``O(len(text))``.

    Args:
        text: Source text, ``\\r`` stripped.

    Returns:
        str: The same number of lines; comment lines are empty.

    Raises:
        AssertionError: When a block comment is never closed.

    Example:
        >>> code_only('x = [\\n  "a",\\n  // "b",\\n]\\n')
        'x = [\\n  "a",\\n\\n]\\n'
    """
    return "\n".join("" if line.lstrip().startswith(_COMMENT) else line
                     for line in _without_block_comments(text).split("\n"))


def char_list(text: str) -> list[str]:
    """The characters of a comma-separated list of Rust char literals, escapes resolved.

    MAGI FIX (S6 loop 28): the trim-list pin read ``'(.)'``, which cannot see
    an escaped literal such as ``'\\\\'``, so the guard could read fewer
    characters than the product lists (F1b2a-melchior). Anything in *text*
    that is not a char literal, a comma or whitespace fails the read.

    Args:
        text: The inside of a Rust array, e.g. ``'.', ' '``.

    Returns:
        list[str]: One character per literal, in order.

    Raises:
        AssertionError: When *text* holds no literal, or anything else.

    Example:
        >>> char_list("'/', '\\\\\\\\'")
        ['/', '\\\\']
    """
    chars, index = [], 0
    while index < len(text):
        if text[index].isspace() or text[index] == ",":
            index += 1
            continue
        match = _CHAR_LITERAL.match(text, index)
        if match is None:
            raise AssertionError("not a char literal at %r in %r" % (text[index:index + 12], text))
        body = match.group(1)
        if not body.startswith("\\"):
            chars.append(body)
        elif body[1] == "x":
            chars.append(chr(int(body[2:], 16)))
        elif body[1] == "u":
            # [F1b2a] six hex digits reach past U+10FFFF, and a surrogate is
            # no char in Rust either; chr raised ValueError or accepted one.
            point = int(body[3:-1], 16)
            if point > _MAX_CODE_POINT or _SURROGATES[0] <= point <= _SURROGATES[1]:
                raise AssertionError("not a Unicode scalar value: %r in %r" % (match.group(0), text))
            chars.append(chr(point))
        else:
            chars.append(_SIMPLE_ESCAPES[body[1]])
        index = match.end()
    if not chars:
        raise AssertionError("no char literal in %r" % text)
    return chars


def _starts_inside(spans: list, position: int) -> bool:
    """Whether *position* lies inside one of *spans*, past its first character.

    Args:
        spans: Sorted, disjoint ``(start, end)`` literal spans.
        position: An index in the text.

    Returns:
        bool: ``True`` when some span has ``start < position < end``: the
        position is in the literal's contents or on its closing quote, never
        on its opening.
    """
    index = bisect.bisect_left(spans, (position,)) - 1
    return index >= 0 and position < spans[index][1]


def _exactly_one(regex: re.Pattern, text: str, where: str, outside_literals: bool = False) -> re.Match:
    """The single match of *regex* in *text* that does not sit on a comment line.

    Args:
        regex: A pattern anchored at a line start (``re.M``).
        text: Source text.
        where: The file the text came from, for the failure message.
        outside_literals: Whether a match whose first non-blank character
            lies inside a string or raw-string literal (a line of a
            multi-line fixture) is skipped too.

    Returns:
        re.Match: The one match.

    Raises:
        AssertionError: When there is no such match or more than one, naming
            *where*, the pattern and the count.

    MAGI FIX (S6 loop 34): trailing ``//`` comments stayed in the searched
    view, so a pattern with a leading or middle wildcard read a value from an
    old copy left in such a comment, and a ``re.S`` body capture carried the
    comment's text. Every line comment is now blanked first.
    """
    spans = [] if outside_literals else None
    view = _without_block_comments(text, line_comments=True, literals=spans)
    found = [match for match in regex.finditer(view)
             if not match.group(0).lstrip().startswith(_COMMENT)
             and not (outside_literals and _starts_inside(
                 spans, match.start() + len(match.group(0)) - len(match.group(0).lstrip())))]
    if len(found) != 1:
        raise AssertionError("%s: expected exactly one code line matching %r, found %d"
                             % (where, regex.pattern, len(found)))
    return found[0]


def one_definition(text: str, pattern: str, where: str, flags: int = 0) -> re.Match:
    """The one code line that *pattern* starts, after the line's indentation.

    The pattern is anchored with ``^[ \\t]*`` under ``re.M``: it must be the
    first thing on its line, so an old copy left in a comment, or quoted mid
    line, never answers for the real definition. MAGI FIX (F1b2a): nor does a
    line that starts inside a string or raw-string literal, such as a test
    fixture spread over a multi-line ``r#"..."#``. The match is still read off
    the source with its literals intact, so a pattern may capture a string's
    contents when the literal opens on the definition's own line. The
    pattern's own groups are available on the result.

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
    return _exactly_one(re.compile(r"^[ \t]*(?:%s)" % pattern, re.M | flags), text, where,
                        outside_literals=True)


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
