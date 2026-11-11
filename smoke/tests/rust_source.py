# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
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
that follow from it are declared rather than handled. The trade-off is
deliberate (MAGI §6 MS3 loop11, R7b Balthasar): the harness stays stdlib-only,
and each format drift fails loudly in its sync guard, which then gets one more
case here, a recurring cost accepted over a lexer dependency:

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
  taken for a comment and skipped. :func:`one_definition` and
  :func:`one_occurrence` skip every line that starts inside such a literal,
  and :func:`one_occurrence` also skips a needle that starts inside one.
  [MAGI F1b2a; S6 loops 38 and 44]
- Both skip every test-only item, field, statement or match arm (an outer
  ``#[cfg(...)]`` whose predicate requires ``test``): a fixture there never
  answers for production code. An inner ``#![cfg(test)]`` and
  ``not(not(test))`` are not recognised (declared: a copy there still counts).
  [MAGI S6 loops 44 and 45, F1b2a-melchior, F1b2a-caspar]
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


def _span_at(spans: list, position: int) -> tuple | None:
    """The literal span *position* lies in past its first character (see :func:`_starts_inside`), or ``None``."""
    index = bisect.bisect_left(spans, (position,)) - 1
    return spans[index] if index >= 0 and position < spans[index][1] else None


def _reads_from_a_literal(match: re.Match, spans: list, text: str) -> bool:
    """Whether *match* takes code text from inside a literal that its pattern did not open.

    A group may capture a literal's contents only whole, opened by the pattern's own quote
    (``= "([^"]+)";``); a group that starts elsewhere inside a literal, or a match whose last character is
    in a literal's contents, read a stale copy a wildcard ran into (``let s = "pub const X: u8 = 4;";``).

    Args:
        match: A match over the comment-blanked view, offsets equal to *text*'s.
        spans: The view's sorted literal spans.
        text: The view.

    Returns:
        bool: ``True`` when the match reads from inside a literal.
    """
    last = _span_at(spans, match.end() - 1) if match.end() > match.start() else None
    if last is not None and text[match.end() - 1] != '"':
        return True
    for group in range(1, (match.re.groups or 0) + 1):
        start, end = match.span(group)
        span = _span_at(spans, start) if start >= 0 and end > start else None
        if span is not None and not (text[start - 1] == '"' and end < span[1] and text[end] == '"'):
            return True
    return False


#: An outer attribute's opening, ``#[``; the ``cfg`` one names its predicate (group 1).
_ATTRIBUTE_OPENING = re.compile(r"#\[\s*(cfg\s*\()?")
#: The start of an item that ends at its block's closing ``}`` or at a ``;`` (``const fn`` included).
_BLOCK_ITEM = re.compile(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern|default)\s+)*"
                         r"(?:mod|fn|impl|struct|enum|trait|union|macro_rules!|thread_local!)(?![\w!])")
#: What lets a ``}`` end a non-block item: nothing else on its line, and no ``.`` or ``?`` continuing the
#: expression on the next one. Anything else after it on the same line (``else``, ``=>``, ``if``, ``|``,
#: an operator, a separator) carries the item on. Known, accepted limit: an operator or ``else`` that
#: starts the next line does not (see :func:`without_test_items`).
_ENDS_AFTER_BRACE = re.compile(r"[ \t]*(?:\n|$)(?!\s*[.?])")
#: A ``<`` that opens generic arguments: right after an identifier or ``::``, not ``<<`` nor ``<=``.
_GENERIC_OPENING = re.compile(r"(?<=[\w:])<(?![<=])")
#: A ``>`` that is not part of ``->``, ``=>`` nor ``>=``.
_ANGLE_CLOSING = re.compile(r"(?<![-=])>(?!=)")
#: The brackets an item's end is counted over.
_OPENERS, _CLOSERS = "([{", ")]}"


def _closing(view: str, start: int) -> int:
    """The offset just past the bracket that closes the one right before *start*.

    Raises:
        AssertionError: When it never closes.
    """
    depth = 1
    for index in range(start, len(view)):
        depth += (view[index] in _OPENERS) - (view[index] in _CLOSERS)
        if depth == 0:
            return index + 1
    raise AssertionError("an attribute never closes")


def _arguments(text: str) -> list[str]:
    """*text* split at its top-level commas, blank parts (a trailing comma) dropped."""
    parts, depth, current = [], 0, []
    for char in text:
        depth += (char in _OPENERS) - (char in _CLOSERS)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _requires_test(predicate: str) -> bool:
    """Whether the ``cfg`` *predicate* holds only in a test build.

    Example:
        >>> [_requires_test(p) for p in ("test", "all(unix, test)", "any(test, unix)", "not(test)")]
        [True, True, False, False]
    """
    call = re.fullmatch(r"\s*(all|any|not)\s*\((.*)\)\s*", predicate, re.S)
    if call is None:
        return predicate.strip() == "test"
    arguments = _arguments(call.group(2))
    if call.group(1) == "all":
        return any(_requires_test(argument) for argument in arguments)
    if call.group(1) == "any":
        return bool(arguments) and all(_requires_test(argument) for argument in arguments)
    return False


def _test_item_end(view: str, start: int, block: bool) -> int:
    """Where the test-only code that starts at *start* in *view* ends.

    Args:
        view: :func:`structure_only`'s view of the text.
        start: The offset right after the item's attributes.
        block: Whether the item is a block item (``fn``, ``mod``, ``const fn``...), which ends at the ``}``
            that closes its block or at a ``;``. Anything else (``const``, ``let``, a field, a statement, a
            match arm) ends at a top-level ``,`` or ``;`` outside generic arguments, at a top-level ``}``
            that ends its line (:data:`_ENDS_AFTER_BRACE`), or before the bracket that closes the code
            around it.

    Returns:
        int: The offset just past the item (at the enclosing closer, which is not the item's).

    Raises:
        AssertionError: When the item never ends, or when a non-block item is ambiguous: a top-level ``,``
            after an odd number of ``|`` (a closure's parameters) or after a ``<`` that opens no generic (a
            comparison), or an end reached with generic arguments still open. A guess there either leaves
            test code counted or hides production code, so the guard fails instead.
    """
    # MAGI FIX (S6 loop 47, F1b2a1-caspar, F1b2a1-melchior): a generic argument's comma and a struct
    # pattern's ``}`` ended the item early; each loop found another such token, so what cannot be placed fails.
    depth, angles, pipes, comparison = 0, 0, 0, False
    for index in range(start, len(view)):
        char = view[index]
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth < 0 or (depth == 0 and char == "}" and (block or _ENDS_AFTER_BRACE.match(view, index + 1))):
                if angles and not block:
                    raise AssertionError("an ambiguous test-only item: generic arguments never close")
                return index if depth < 0 else index + 1
        elif depth or block:
            if depth == 0 and char == ";":
                return index + 1
        elif char == "<":
            if _GENERIC_OPENING.match(view, index):
                angles += 1
            elif not (view.startswith(("<=", "<<"), index) or view.startswith("<<", index - 1)):
                comparison = comparison or angles == 0
        elif char == ">" and angles and _ANGLE_CLOSING.match(view, index):
            angles -= 1
        elif char == "|":
            pipes += 1
        elif char in ",;" and not angles:
            if pipes % 2 or (char == "," and comparison):
                raise AssertionError("an ambiguous test-only item: a %r it cannot place" % char)
            return index + 1
        elif char == ";":
            raise AssertionError("an ambiguous test-only item: generic arguments never close")
    raise AssertionError("a test-only item never ends")


def _test_items(view: str):
    """The ``(start, end)`` span of every item an outer ``#[cfg(...)]`` that requires ``test`` marks.

    Args:
        view: :func:`structure_only`'s view of the text.

    Yields:
        tuple[int, int]: From the item's first attribute to its end, in order, not overlapping.
    """
    index = 0
    while True:
        attribute = _ATTRIBUTE_OPENING.search(view, index)
        if attribute is None:
            return
        start, test_only, cursor = attribute.start(), False, attribute.start()
        while (following := _ATTRIBUTE_OPENING.match(view, cursor)) is not None:
            if following.group(1) is not None:
                test_only = test_only or _requires_test(view[following.end():_closing(view, following.end()) - 1])
            cursor = _closing(view, following.start() + 2)
            cursor += len(view[cursor:]) - len(view[cursor:].lstrip())
        if test_only:
            stop = _test_item_end(view, cursor, block=_BLOCK_ITEM.match(view, cursor) is not None)
            yield start, stop
            index = stop
        else:
            index = cursor


def without_test_items(text: str) -> str:
    """*text* with every test-only item blanked, its attributes through its end (:func:`_test_item_end`).

    A guard pins production code; a copy in a test module is a fixture, and with the production line
    gone it would answer for it. Offsets and newlines are kept. Test-only means an outer ``#[cfg(...)]``
    whose predicate requires ``test`` (``test``, ``all(..., test)``; never ``not(test)`` nor an ``any``
    with another branch), on any item, field, statement or match arm. A non-block item whose end the
    delimiter cannot place (a closure's ``|a, b|``, a comparison before a ``,``, generics that never
    close) fails the guard instead of being guessed; an or-pattern arm (``0 | 1 => f(),``) falls under the
    same rule and fails too, toward red. Declared limits: an inner ``#![cfg(test)]`` (the product has none)
    and ``not(not(test))`` are not recognised, so a copy there still counts.

    Known limit, accepted (S6 loop 48, F1b2a1-caspar; user decision 2026-11-04): a non-block item whose
    block expression goes on at the start of the next line with an operator or ``else``
    (``let x = { a }`` then ``+ f("...")`` on the next line) ends at the ``}``, so the rest of it is read as
    production and a copy there counts. Low probability: rustfmt writes ``} else {`` on one line, and the
    36 non-block test items in ``src/`` (``use``, fields, one-line trace calls) have no such shape; it also
    needs a guarded string copied into that remainder while the production line changes. Fix it
    (:data:`_ENDS_AFTER_BRACE` also refusing a next line that starts with an operator, ``else``, ``as`` or
    ``=>``) before adding a test-only statement of that shape.

    Complexity: ``O(len(text))`` per attribute.

    Args:
        text: Source text, ``\\r`` stripped.

    Returns:
        str: The text with each test item's characters, newlines excepted, turned into spaces.

    Raises:
        AssertionError: When a test item never ends or its end is ambiguous (:func:`_test_item_end`).

    Example:
        >>> without_test_items('#[cfg(test)]\\nmod t { fn f() {} }\\nfn g() {}\\n')
        '            \\n                  \\nfn g() {}\\n'
    """
    out, previous = [], 0
    for start, stop in _test_items(structure_only(text)):
        out.append(text[previous:start])
        out.append(_blank(text[start:stop]))
        previous = stop
    out.append(text[previous:])
    return "".join(out)


def _exactly_one(regex: re.Pattern, text: str, where: str, outside_literals: bool = False,
                 needle_length: int = 0) -> re.Match:
    """The single match of *regex* in *text* that does not sit on a comment line nor in a test item.

    Args:
        regex: A pattern anchored at a line start (``re.M``).
        text: Source text.
        where: The file the text came from, for the failure message.
        outside_literals: Whether a match whose first non-blank character
            lies inside a string or raw-string literal (a line of a
            multi-line fixture) is skipped too.
        needle_length: When positive, the length of the needle that ends each match; a match whose
            needle starts inside a literal is skipped too (requires *outside_literals*).

    Returns:
        re.Match: The one match.

    Raises:
        AssertionError: When there is no such match or more than one, naming
            *where*, the pattern and the count.

    MAGI FIX (S6 loop 34): trailing ``//`` comments stayed in the searched
    view, so a pattern with a leading or middle wildcard read a value from an
    old copy left in such a comment, and a ``re.S`` body capture carried the
    comment's text. Every line comment is now blanked first.

    MAGI FIX (S6 loop 44, F1b2a-melchior): a one-line fixture in a test module was a code line, so with
    the production line gone it answered for it; ``#[cfg(test)]`` items are blanked first, and a needle
    found inside some other string no longer counts.
    """
    spans = [] if outside_literals else None
    view = _without_block_comments(without_test_items(text), line_comments=True, literals=spans)
    found = [match for match in regex.finditer(view)
             if not match.group(0).lstrip().startswith(_COMMENT)
             and not (outside_literals and _starts_inside(
                 spans, match.start() + len(match.group(0)) - len(match.group(0).lstrip())))
             and not (needle_length and _starts_inside(spans, match.end() - needle_length))
             # MAGI FIX (S6 loop 46, F1b2a-caspar): only the start was checked, so a wildcard ran into a string
             # and captured a stale copy's value from inside it.
             and not (outside_literals and not needle_length and _reads_from_a_literal(match, spans, view))]
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
    pattern's own groups are available on the result. A group may capture a
    literal's contents only whole, opened by the pattern's own quote, and a
    match may not end inside one: a wildcard that ran into a string never
    reads a value from it. [MAGI S6 loop 46, F1b2a-caspar]

    Known limit, accepted (S6 loop 48, F1b2a2-melchior; user decision
    2026-11-04): "whole" is checked by the quotes around the group, not by the
    literal's real delimiters. A lazy capture such as ``"(.*?)"`` stops at an
    escaped ``\\"`` or at an interior ``"`` of a raw string and returns a
    prefix, and a group may start right after an escaped quote, mid-literal.
    Low probability: of the guards in this harness only three capture a
    literal's contents (``MEMORY_DB_PREFIX`` and ``HARD_LINK_CAUSE`` with
    ``[^"]+`` over values without quotes, and the ``batches leave`` message in
    ``src/review/error.rs``); the prefix read is the product's own text, not a
    stale copy, so the likely outcome is a red, not a false green. Fix it
    (compare the group's ends with the span's real delimiters, raw-string
    hashes and escapes included) before writing a capture over a literal that
    holds a quote.

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
    ``#[error(...)]`` attribute. The needle may span lines. A line that starts
    inside a multi-line string or raw-string literal is not a code line and
    is never counted. The scan stops at
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
    # MAGI FIX (S6 loop 38): a line inside a multi-line literal (a test fixture) was counted, so with the
    # production line gone a stale fixture copy answered for it.
    return _exactly_one(regex, text, where, outside_literals=True, needle_length=len(needle))

#: The product's secret detector, the source :func:`secret_openings` reads.
SECRETS_SOURCE = "src/review/secrets.rs"


def secret_openings(root: pathlib.Path) -> list:
    """Every opening of the product's secret patterns, lowercase, read from its source.

    The token prefixes of ``SECRET_PATTERNS`` (``prefixes: &[...]``), the fixed
    openings of ``NON_TOKEN_PREFIXES`` and ``URL_AUTHORITY_START``, which that
    list names by identifier. [MAGI S6 MS3 loop7, R7b] One reading, so no test
    keeps a hand copy of the table (one missed ``gho_``, ``ghu_``, ``ghs_``,
    ``ghr_``).

    Args:
        root: The repository root.

    Returns:
        list: The openings.

    Raises:
        AssertionError: When the table does not parse (a part matched nothing).
    """
    text = without_comments(read_source(SECRETS_SOURCE, root))
    literal = re.compile(r'"([^"]*)"')
    tokens = [s for group in re.findall(r"prefixes: &\[([^\]]*)\]", text)
              for s in literal.findall(group)]
    fixed = one_definition(text, r"const NON_TOKEN_PREFIXES: &\[&str\] = &\[(.*?)\];",
                           SECRETS_SOURCE, re.S).group(1)
    # MAGI FIX (MS3 §6 loop10): each element is a literal or a `const` named in the file, resolved
    # here; an element that is neither fails, so a new opening is never dropped (R7b Caspar).
    openings = list(tokens)
    for element in (e.strip() for e in fixed.split(",") if e.strip()):
        value = literal.fullmatch(element)
        if value is None and re.fullmatch(r"[A-Z][A-Z0-9_]*", element):
            value = one_definition(text, r'const %s: &str = "([^"]+)";' % element, SECRETS_SOURCE)
        if value is None:
            raise AssertionError("%s: NON_TOKEN_PREFIXES element %r did not parse" % (SECRETS_SOURCE, element))
        openings.append(value.group(1))
    if len(tokens) < 2:
        raise AssertionError("%s: the openings did not parse: %r" % (SECRETS_SOURCE, openings))
    return [o.lower() for o in openings]
