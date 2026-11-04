# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-29
"""The code-review scenarios' copies of the product's numbers and wording.

The harness reads no Rust at run time, so S47-S51 carry their own copies of
the batch constants, the per-batch floor and the texts they parse out of
``stderr``. A copy stops being true silently when the product changes, and the
first sign would be an opaque ``FAIL`` at SMOKE time. These tests read the Rust
source and fail here, at the harness's own gate, the moment the two sides
drift. Each pattern must match: a pattern that matched nothing would let the
comparison below it pass by never running.

Every presence check reads through :mod:`smoke.tests.rust_source`, which looks
only at code lines and insists on exactly one match ([MAGI loop 26]): a plain
search over the file read a drifted constant's old value from a ``///`` comment
left above the new line and stayed green. Absence checks (``assertNotIn``) stay
on the raw text, where a comment can only make them stricter.
"""

import ast
import fractions
import os
import pathlib
import re
import unittest
from unittest import mock

from smoke import logs, runs
from smoke.scenarios import code_review
from smoke.tests.rust_source import (char_list, one_definition, one_occurrence, read_source,
                                     structure_only, without_comments)

#: The repository root, three levels above this file.
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _source(relative: str) -> str:
    """Read a source file under :data:`_ROOT` with ``\\r`` stripped (:func:`read_source`).

    Args:
        relative: The path under the repository root, with forward slashes.

    Returns:
        str: The file's text.
    """
    return read_source(relative, _ROOT)


#: The name the scenario module gives the ``code_review`` summary object.
_SECTION_NAME = "section"

#: The helper that reads a list under a key of the summary, and the parameter
#: its own ``section.get`` takes instead of a literal key (read at its call sites).
_LIST_UNDER = "_list_under"
_LIST_UNDER_PARAMETER = "key"

#: The nodes that open a scope of their own: a ``key`` bound by one of them
#: between a ``section.get(key)`` and ``_list_under`` is not the helper's own.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: Calls that may take ``section`` without reading a key from it.
_KEYLESS_CALLS = frozenset({"isinstance"})

#: The constant the scenario module names the summary's key with, the name of
#: the parsed stdout it reads the summary from, and the key's value.
_SUMMARY_KEY_NAME = "CODE_REVIEW_KEY"
_DOCUMENT_NAME = "document"
_SUMMARY_KEY = code_review.CODE_REVIEW_KEY

#: Builtins that reach a name or a member without writing it: through any of
#: them the summary could be read in a form the guard does not follow.
_REFLECTION_BUILTINS = frozenset({"vars", "locals", "globals", "getattr", "eval", "exec", "compile",
                                  "__import__"})


#: The names of the module that holds the builtins: through it every reflection builtin is an attribute.
_BUILTINS_MODULES = frozenset({"builtins", "__builtins__"})


def _reflection_reached(node) -> str | None:
    """The reflection builtin or builtins module *node* names, ``None`` when it names neither.

    MAGI FIX (S6 loop 38): only bare names were refused, so ``builtins.vars()``,
    ``__builtins__.globals()`` or ``from builtins import vars as v`` reached the
    same functions and read the summary unseen.

    Args:
        node: Any AST node.

    Every path to the module goes through one of these: its name, an import of it, or its name as a string
    (``sys.modules["builtins"]``, ``importlib.import_module("builtins")``). An attribute named like a builtin
    is not refused on its own (``re.compile`` is not the builtin); reaching the module is.

    Returns:
        str | None: The name reached: a bare :data:`_REFLECTION_BUILTINS` name, a
        :data:`_BUILTINS_MODULES` name or string literal, or a ``builtins`` import.

    Example:
        >>> _reflection_reached(ast.parse("builtins.vars()").body[0].value.func.value)
        'builtins'
        >>> _reflection_reached(ast.parse("re.compile").body[0].value) is None
        True
    """
    if isinstance(node, ast.Name) and (node.id in _REFLECTION_BUILTINS or node.id in _BUILTINS_MODULES):
        return node.id
    if _literal_key(node) in _BUILTINS_MODULES:
        return node.value
    if isinstance(node, ast.alias) and node.name.split(".")[0] in _BUILTINS_MODULES:
        return node.name
    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _BUILTINS_MODULES:
        return node.module
    return None


def _is_dunder(name: str) -> bool:
    """Whether *name* is a dunder (``__dict__``): through one, a module's names are reached by string.

    Example:
        >>> _is_dunder("__dict__"), _is_dunder("get")
        (True, False)
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _literal_key(node) -> str | None:
    """The string *node* holds when it is a string literal, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


#: The type-parameter nodes (3.12+) that bind a name; absent on 3.11, where no such syntax exists.
_TYPE_PARAMETER_NODES = tuple(getattr(ast, name) for name in ("TypeVar", "ParamSpec", "TypeVarTuple")
                              if hasattr(ast, name))


def _names_bound_by(node) -> list[str]:
    """The names *node* binds in its scope, by every binding form Python has.

    MAGI FIX (S6 loop 33): only ``Name`` stores and deletes were looked for, so a
    ``match`` capture, an ``except ... as``, an import alias, a nested ``def``
    or ``class``, or a ``global``/``nonlocal`` declaration rebound a name the
    guard trusted.

    Args:
        node: Any AST node.

    Returns:
        list[str]: The names bound: a stored or deleted ``Name``; a function's or
        class's name; an import alias (its ``asname``, else the first dotted part
        of the imported name); a ``match`` capture (``MatchAs``, ``MatchStar``,
        ``MatchMapping.rest``); an exception handler's name; the names a
        ``global`` or ``nonlocal`` declares; a parameter; a type parameter.
        Empty for a node that binds nothing.

    Example:
        >>> _names_bound_by(ast.parse("import os.path as key").body[0].names[0])
        ['key']
        >>> _names_bound_by(ast.parse("x + 1").body[0])
        []
    """
    if isinstance(node, ast.Name):
        return [node.id] if isinstance(node.ctx, (ast.Store, ast.Del)) else []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.alias):
        return [node.asname or node.name.split(".")[0]]
    if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler) + _TYPE_PARAMETER_NODES):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    if isinstance(node, ast.arg):
        return [node.arg]
    return []


def _same_scope_nodes(statements: list):
    """Every node under *statements* that is not inside a nested scope (:data:`_SCOPE_NODES`).

    Args:
        statements: A function's body.

    Yields:
        ast.AST: Each node in the scope the statements open, nested scope nodes excluded with their contents.
    """
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, _SCOPE_NODES):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _is_section_get_of_parameter(node) -> bool:
    """Whether *node* is ``section.get(key)`` with exactly that one argument, ``key`` being the helper's parameter."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == _SECTION_NAME
            and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], ast.Name) and node.args[0].id == _LIST_UNDER_PARAMETER)


def _trusted_list_under(tree: ast.Module) -> ast.FunctionDef | None:
    """The module's ``_list_under`` when its calls and its ``section.get(key)`` can be credited, else ``None``.

    MAGI FIX (S6 loop 33): the helper was trusted by its name, so one whose
    parameters were renamed, a lambda or import bound to the name, or a second
    binding of it read a key the guard never saw; and ``key`` was only checked
    for ``Name`` stores, so a ``match`` capture or an import alias rebound it.

    Args:
        tree: The parsed scenario module.

    Returns:
        ast.FunctionDef | None: The helper, only when the module binds
        ``_list_under`` exactly once (by any form :func:`_names_bound_by`
        knows, anywhere in the module), that binding is an undecorated
        top-level ``def _list_under(section, key)`` taking exactly those two
        positional parameters, nothing in its body binds ``key`` again, and its
        own scope reads ``section.get(key)`` at least once. MAGI FIX (S6 loop
        34): a helper of that shape whose body read nothing (``{}.get(key)``)
        had its calls' keys credited all the same.
    """
    bindings = [node for node in ast.walk(tree) if _LIST_UNDER in _names_bound_by(node)]
    if len(bindings) != 1:
        return None
    helper = bindings[0]
    if not (isinstance(helper, ast.FunctionDef) and helper in tree.body and not helper.decorator_list):
        return None
    arguments = helper.args
    if ([arg.arg for arg in arguments.posonlyargs + arguments.args] != [_SECTION_NAME, _LIST_UNDER_PARAMETER]
            or arguments.vararg or arguments.kwonlyargs or arguments.kwarg):
        return None
    if any(_LIST_UNDER_PARAMETER in _names_bound_by(inner)
           for statement in helper.body for inner in ast.walk(statement)):
        return None
    if not any(_is_section_get_of_parameter(node) for node in _same_scope_nodes(helper.body)):
        return None
    return helper


def _binding_counts(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound anywhere in *tree*, by every form :func:`_names_bound_by` knows.

    Args:
        tree: A parsed module.

    Returns:
        dict[str, int]: The number of bindings of each name bound at least once.

    Example:
        >>> sorted(_binding_counts(ast.parse("a = 1\\ndef f(a):\\n    pass")).items())
        [('a', 2), ('f', 1)]
    """
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        for name in _names_bound_by(node):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _is_summary_definition(node) -> bool:
    """Whether *node* is ``CODE_REVIEW_KEY = "<the summary key>"``: one ``Name`` target, the literal value."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == _SUMMARY_KEY_NAME and _literal_key(node.value) == _SUMMARY_KEY)


def _is_section_assignment(node) -> bool:
    """Whether *node* is exactly ``section = document.get(CODE_REVIEW_KEY)``, the scenario module's one summary read."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == _SECTION_NAME and isinstance(node.value, ast.Call)):
        return False
    call = node.value
    return (isinstance(call.func, ast.Attribute) and call.func.attr == "get"
            and isinstance(call.func.value, ast.Name) and call.func.value.id == _DOCUMENT_NAME
            and len(call.args) == 1 and not call.keywords
            and isinstance(call.args[0], ast.Name) and call.args[0].id == _SUMMARY_KEY_NAME)


def _names_the_summary(node) -> bool:
    """Whether *node* refers to the summary: the name ``CODE_REVIEW_KEY`` in any form, or the key's literal."""
    return ((isinstance(node, ast.Name) and node.id == _SUMMARY_KEY_NAME)
            or (isinstance(node, ast.Attribute) and node.attr == _SUMMARY_KEY_NAME)
            or _SUMMARY_KEY_NAME in _names_bound_by(node)
            or _literal_key(node) == _SUMMARY_KEY)


class _SectionReads:
    """Every use of the name ``section`` in a module, each one either read for its key or refused.

    MAGI FIX (F1b2a): the keys were found with regular expressions, so a legal
    form they did not anticipate (``section . get("k")``, an alias, a helper
    that received ``section``) read a key the guard never saw. The module is
    parsed instead, and every ``Name`` node called ``section`` must be one of
    the forms below; any other use fails loudly.

    Accepted: a binding (an assignment target or a parameter);
    ``section.get("<lit>")`` with exactly one argument;
    ``section.get(key)`` directly inside the module's ``_list_under`` and
    ``_list_under(section, "<lit>")``, both only while
    :func:`_trusted_list_under` returns the helper (otherwise both fail
    closed); ``isinstance(section, ...)``; and ``section`` passed
    positionally to a function defined in the module whose parameter at that
    position is also named ``section``, whose own body the same walk reads.

    MAGI FIX (S6 loop 34): only the name ``section`` was followed, so the
    summary read through any other path (``cr = document.get(CODE_REVIEW_KEY)``,
    a chained ``.get``, ``document["code_review"]``) credited no key and the
    guard stayed green. Every reference to ``CODE_REVIEW_KEY`` and every
    literal equal to the summary key must now be part of the module's single
    top-level ``CODE_REVIEW_KEY = "<key>"`` or its single
    ``section = document.get(CODE_REVIEW_KEY)``; any other one fails closed.

    MAGI FIX (S6 loop 35): a call was credited by the name of a top-level
    ``def``, so a local ``def`` of that name, a later rebinding or a decorator
    ran code the walk never read; only an undecorated top-level ``def`` whose
    name the module binds exactly once is credited now. And the summary could
    be reached without naming its key (a key built at run time, the
    document's values, ``vars()``): every load of ``document`` must now be
    ``document.get(NAME)`` with ``NAME`` a top-level constant bound once to a
    string literal, and no reflection builtin (:data:`_REFLECTION_BUILTINS`)
    may be named. Declared limit: a second name bound to the parsed JSON
    (another ``run.output.json()``) read with a computed key is not followed;
    the guard reads the module's ``document`` only.

    Complexity: ``O(nodes)``.
    """

    def __init__(self, source: str) -> None:
        """Parse *source* and index each node's parent and each module function's parameters.

        Args:
            source: A Python module's source.

        Raises:
            SyntaxError: When it does not parse.
        """
        self._tree = ast.parse(source)
        self._list_under = _trusted_list_under(self._tree)
        self._parents = {}
        counts = _binding_counts(self._tree)
        self._parameters = {node.name: [arg.arg for arg in node.args.posonlyargs + node.args.args]
                            for node in self._tree.body
                            if isinstance(node, ast.FunctionDef) and not node.decorator_list
                            and counts.get(node.name) == 1}
        self._constants = {node.targets[0].id for node in self._tree.body
                           if isinstance(node, ast.Assign) and len(node.targets) == 1
                           and isinstance(node.targets[0], ast.Name) and _literal_key(node.value) is not None
                           and counts.get(node.targets[0].id) == 1}
        for node in ast.walk(self._tree):
            for child in ast.iter_child_nodes(node):
                self._parents[child] = node

    def keys(self) -> set[str]:
        """The literal keys read from ``section``.

        Returns:
            set[str]: Every key named literally in an accepted read.

        Raises:
            AssertionError: Naming the line of the first use the walk cannot follow.
        """
        self._check_summary_references()
        self._check_document_reads()
        keys = set()
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Name) and node.id == _SECTION_NAME:
                if isinstance(node.ctx, ast.Store):
                    continue
                key = self._key_of(node)
                if key is not None:
                    keys.add(key)
        return keys

    def _check_summary_references(self) -> None:
        """Refuse every reference to the summary outside its one definition and its one ``section`` assignment.

        Raises:
            AssertionError: Naming the line of the first other reference, or of a second definition or
                assignment.
        """
        definitions = [node for node in self._tree.body if _is_summary_definition(node)]
        assignments = [node for node in ast.walk(self._tree) if _is_section_assignment(node)]
        allowed = set()
        if len(definitions) == 1:
            allowed.update((definitions[0].targets[0], definitions[0].value))
        if len(assignments) == 1:
            allowed.add(assignments[0].value.args[0])
        for node in ast.walk(self._tree):
            if _names_the_summary(node) and node not in allowed:
                parent = self._parents.get(node)
                raise AssertionError("line %d: a read of the summary the key guard does not follow: %s"
                                     % (getattr(node, "lineno", getattr(parent, "lineno", 0)),
                                        ast.unparse(parent if parent is not None else node)))

    def _check_document_reads(self) -> None:
        """Refuse every reflection builtin and every load of ``document`` but ``document.get(<constant>)``.

        Raises:
            AssertionError: Naming the line of the first reflection builtin, or of the first load of
                ``document`` that is not ``.get`` with one top-level constant bound once to a literal.
        """
        for node in ast.walk(self._tree):
            reached = _reflection_reached(node)
            if reached is not None:
                raise AssertionError("line %d: %s reaches names the key guard does not follow"
                                     % (getattr(node, "lineno", 0), reached))
            # MAGI FIX (S6 loop 40, F1b2a-caspar): the module object (``sys.modules[__name__]``, an
            # ``import_module`` of the scenario) reached ``section`` as an attribute, unseen by the walk.
            if isinstance(node, ast.Attribute) and (node.attr == _SECTION_NAME or _is_dunder(node.attr)):
                raise AssertionError("line %d: a member access the key guard does not follow: %s"
                                     % (node.lineno, ast.unparse(node)))
            if isinstance(node, ast.Name) and node.id == _DOCUMENT_NAME and isinstance(node.ctx, ast.Load):
                attribute = self._parents.get(node)
                call = self._parents.get(attribute)
                if not (isinstance(attribute, ast.Attribute) and attribute.attr == "get"
                        and isinstance(call, ast.Call) and call.func is attribute
                        and len(call.args) == 1 and not call.keywords
                        and isinstance(call.args[0], ast.Name) and call.args[0].id in self._constants):
                    raise AssertionError("line %d: a read of the document the key guard does not follow: %s"
                                         % (node.lineno, ast.unparse(call if call is not None else node)))

    def _key_of(self, node) -> str | None:
        """The key one load of ``section`` reads, ``None`` for a keyless accepted use.

        Raises:
            AssertionError: When the use is none of the accepted forms.
        """
        parent = self._parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = self._parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and len(call.args) == 1 and not call.keywords:
                key = _literal_key(call.args[0])
                if key is not None:
                    return key
                if (isinstance(call.args[0], ast.Name) and call.args[0].id == _LIST_UNDER_PARAMETER
                        and self._reads_the_list_under_parameter(node)):
                    return None
        if isinstance(parent, ast.Call) and node in parent.args and isinstance(parent.func, ast.Name):
            callee, position = parent.func.id, parent.args.index(node)
            # MAGI FIX (S6 loop 33): a _list_under call is credited only through the trusted helper.
            if (callee == _LIST_UNDER and self._list_under is not None and position == 0
                    and len(parent.args) == 2 and not parent.keywords):
                key = _literal_key(parent.args[1])
                if key is not None:
                    return key
            elif callee in _KEYLESS_CALLS:
                return None
            elif callee != _LIST_UNDER:
                parameters = self._parameters.get(callee, [])
                if position < len(parameters) and parameters[position] == _SECTION_NAME:
                    return None
        raise AssertionError("line %d: a use of section the key guard does not follow: %s"
                             % (node.lineno, ast.unparse(parent if parent is not None else node)))

    def _reads_the_list_under_parameter(self, node) -> bool:
        """Whether *node*'s ``section.get(key)`` reads the ``key`` its module-level ``_list_under`` was called with.

        MAGI FIX (F1b2a-c): the innermost function's name alone was checked, so
        a ``key`` rebound in the body, bound by a lambda default or a
        comprehension, or a nested ``_list_under`` read a literal the guard
        never saw.

        Args:
            node: A load of ``section`` whose ``.get`` takes the name ``key``.

        Returns:
            bool: ``True`` only when the nearest enclosing scope (function,
            lambda, comprehension or class) is the helper
            :func:`_trusted_list_under` returned. MAGI FIX (S6 loop 33): the
            rebinding scan now lives there and covers every binding form.
        """
        scope = self._parents.get(node)
        while scope is not None and not isinstance(scope, _SCOPE_NODES):
            scope = self._parents.get(scope)
        return self._list_under is not None and scope is self._list_under


def _summary_keys_read(scenario: str) -> set[str]:
    """The ``code_review`` summary keys a scenario module reads, from every shape it reads them in.

    MAGI FIX (S6 loop 29): the keys were collected from two access shapes, so a
    key read in a third one (brackets, another accessor, a key held in a
    variable) was left out and the fixed-set comparison still passed
    (F1b2a-melchior). MAGI FIX (F1b2a): the shapes are read off the parsed
    module (:class:`_SectionReads`), so spacing, aliases and helpers cannot
    hide a read either.

    Args:
        scenario: The scenario module's source.

    Returns:
        set[str]: The keys read with ``section.get("<key>")`` or
        ``_list_under(section, "<key>")``.

    Raises:
        AssertionError: Naming a use of ``section`` the read does not follow:
            brackets, any accessor but ``.get``, a ``.get`` whose argument is
            neither a literal key nor ``_list_under``'s own parameter, a
            ``_list_under`` call without a literal key, an alias, ``section``
            unpacked or iterated, or passed to a function the walk cannot read;
            or the summary reached by any path but its single
            ``section = document.get(CODE_REVIEW_KEY)``.
    """
    return _SectionReads(scenario).keys()


#: The fewest words a copied stderr phrase may hold: one word ("the") matches
#: nearly any error line, so it would let a scenario pass on unrelated text.
_MIN_PHRASE_WORDS = 2


def _is_word_bounded_phrase(copy: str, text: str) -> bool:
    """Whether *copy* is a phrase of *text* that cannot match unrelated stderr.

    Args:
        copy: The harness's copy of a product phrase.
        text: The product's literal it was copied from.

    Returns:
        bool: ``True`` when *copy* has no surrounding whitespace, holds at least
        :data:`_MIN_PHRASE_WORDS` words, and occurs in *text* with no word
        character right before or after it.

    Example:
        >>> _is_word_bounded_phrase("outside the workspace", "path is outside the workspace: {}")
        True
        >>> _is_word_bounded_phrase("side the work", "path is outside the workspace: {}")
        False
    """
    if copy != copy.strip() or len(re.findall(r"\w+", copy)) < _MIN_PHRASE_WORDS:
        return False
    return re.search(r"(?<!\w)%s(?!\w)" % re.escape(copy), text) is not None


def _usize_const(text: str, name: str) -> int:
    """The value of ``pub const <name>: usize = <n>;`` in *text*.

    Args:
        text: A Rust source file's text.
        name: The constant's name.

    Returns:
        int: Its value, with ``_`` separators removed.

    Raises:
        AssertionError: When no code line declares it, or more than one does.
    """
    match = one_definition(text, r"pub const %s: usize = ([\d_]+);" % re.escape(name),
                           "const %s" % name)
    return int(match.group(1).replace("_", ""))


def _fraction_covers_percent(fraction: float, percent: int) -> bool:
    """Whether the harness's batch *fraction* is at least the product's *percent*, compared exactly.

    MAGI FIX (S6 loop 33): the check was ``round(fraction * 100) >= percent``,
    so 0.495 rounded up to 50 and passed against a product 50 %. Neither
    rounding nor plain float arithmetic is exact (``0.57 * 100`` is
    ``56.99999999999999``), so the fraction is read as the decimal it is
    written as: ``Fraction(str(fraction))`` takes the shortest repr, which for
    a literal such as ``0.60`` is that literal, and compares in rationals.

    Args:
        fraction: The harness's batch window fraction (``runs.BATCH_WINDOW_FRACTION``).
        percent: The product's ``BATCH_WINDOW_FRACTION_PERCENT``.

    Returns:
        bool: ``True`` when ``fraction * 100 >= percent`` holds exactly.

    Example:
        >>> _fraction_covers_percent(0.57, 57)
        True
        >>> _fraction_covers_percent(0.495, 50)
        False
    """
    return fractions.Fraction(str(fraction)) * 100 >= percent


class BatchConstantTests(unittest.TestCase):
    """The batch arithmetic R12's input is sized with (``smoke/runs.py``)."""

    # MAGI FIX (S6 loop 5): the copies had no automated sync (F1-b2).
    def setUp(self) -> None:
        self.window = _source("src/review/window.rs")

    def test_bytes_per_token_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "BYTES_PER_TOKEN"), runs.BYTES_PER_TOKEN)

    def test_the_unmeasured_window_matches_the_product(self) -> None:
        self.assertEqual(_usize_const(self.window, "DEFAULT_UNMEASURED_WINDOW_TOKENS"),
                         runs.UNMEASURED_WINDOW_TOKENS)

    def test_the_harness_sizes_for_a_batch_at_least_as_large_as_the_product_fills(self) -> None:
        # The harness sizes at the larger of the product's two values (50 provisional, 60 measured);
        # a product fraction above the harness's would plan fewer batches than R12's input assumes.
        percent = _usize_const(self.window, "BATCH_WINDOW_FRACTION_PERCENT")
        self.assertTrue(_fraction_covers_percent(runs.BATCH_WINDOW_FRACTION, percent),
                        (runs.BATCH_WINDOW_FRACTION, percent))

    def test_a_harness_fraction_just_under_the_products_fails_the_check(self) -> None:
        # MAGI FIX (S6 loop 33): the check rounded the harness's percent, so 0.495 passed against a product 50 %
        # and the harness planned for a batch smaller than the one the product fills.
        percent = _usize_const(self.window, "BATCH_WINDOW_FRACTION_PERCENT")
        for under in ((percent - 0.5) / 100, (percent - 0.49) / 100, (percent - 0.01) / 100):
            with self.subTest(fraction=under), mock.patch.object(runs, "BATCH_WINDOW_FRACTION", under):
                with self.assertRaises(AssertionError):
                    self.test_the_harness_sizes_for_a_batch_at_least_as_large_as_the_product_fills()
        # The exact value still passes, also where float arithmetic lands just under it (0.57 * 100 < 57).
        for exact, product in ((percent / 100, percent), (0.57, 57), (0.29, 29)):
            with self.subTest(fraction=exact), mock.patch.object(runs, "BATCH_WINDOW_FRACTION", exact), \
                    mock.patch.dict(globals(), {"_usize_const": lambda _text, _name, value=product: value}):
                self.test_the_harness_sizes_for_a_batch_at_least_as_large_as_the_product_fills()

    def test_the_floor_is_the_products_two_minimums(self) -> None:
        text = _source("src/magi/mod.rs")
        found = {name: int(one_definition(
            text, r"const %s: Duration = Duration::from_secs\((\d+)\);" % name, "src/magi/mod.rs").group(1))
            for name in ("MIN_OPERATION_BUDGET", "MIN_CLIENT_TIMEOUT")}
        one_definition(text, r"pub const AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS: u64 =\s*"
                             r"MIN_OPERATION_BUDGET\.as_secs\(\) \+ MIN_CLIENT_TIMEOUT\.as_secs\(\);",
                       "src/magi/mod.rs")
        self.assertEqual(sum(found.values()), code_review.FLOOR_SECS)

    def test_the_least_need_per_batch_is_the_products_lenient_arithmetic(self) -> None:
        # [MAGI §6 MS2 loop 10, F1a-c3] S50's bounds lean on the classifier reserve and the slack.
        text = _source("src/magi/mod.rs")
        for name, copy in (("CLASSIFY_TIMEOUT_SECS", code_review.CLASSIFY_SECS),
                           ("HEADLESS_TIMEOUT_SLACK_PCT", code_review.SLACK_PCT)):
            found = one_definition(text, r"pub const %s: u64 = (\d+);" % name, "src/magi/mod.rs")
            self.assertEqual(copy, int(found.group(1)), name)
        # [MAGI §6 MS2 loop 28, F1b2a-melchior] the 24 was a literal, so a product change to the
        # derivation went unseen once the copies were updated. The derivation is pinned in the product
        # (headless_consult_timeout_secs at the floor, with attempt_factor's most lenient config: no
        # rotation, retry off) and the expected value is computed from the pinned copies.
        for pattern in (r"CLASSIFY_TIMEOUT_SECS\.saturating_add\(\s*configured_ceiling\.saturating_mul\("
                        r"attempt_factor\(max_rotations, retry_disabled\)\) / 100,",
                        r"let attempts_per_model: u64 = if retry_disabled \{ 1 \} else \{ CALLS_PER_MODEL \};",
                        r"let models_per_mage = u64::from\(max_rotations\)\.saturating_add\(1\);",
                        r"\.saturating_mul\(100 \+ HEADLESS_TIMEOUT_SLACK_PCT\)"):
            one_definition(text, pattern, "src/magi/mod.rs")
        lenient_factor = 1 * (0 + 1) * (100 + code_review.SLACK_PCT)
        self.assertEqual(code_review.CLASSIFY_SECS + code_review.FLOOR_SECS * lenient_factor // 100,
                         code_review.MIN_NEED_PER_BATCH_SECS)
        self.assertEqual(24, code_review.MIN_NEED_PER_BATCH_SECS)

    def test_the_trio_copy_of_the_ceiling_arithmetic_is_the_products(self) -> None:
        # [MAGI §6 MS2 loop 28, F1b2b-melchior] smoke.scenarios.trio's attempt_factor and raw_ceiling were
        # copies of src/magi/mod.rs with nothing tying their arithmetic to it: a product change to the
        # truncation or the saturation (a per-batch wall clock below the classifier's slice) went unseen.
        from smoke.scenarios import trio
        text = _source("src/magi/mod.rs")
        body = one_definition(text, r"pub\(crate\) fn raw_ceiling_from_factor\(timeout_secs: u64, factor: u64\)"
                                    r" -> u64 \{(.*?)\n\}", "src/magi/mod.rs", re.S).group(1)
        self.assertEqual(["timeout_secs", ".saturating_sub(CLASSIFY_TIMEOUT_SECS)", ".saturating_mul(100)",
                          ".checked_div(factor)", ".unwrap_or(0)"],
                         [line.strip() for line in without_comments(body).split("\n") if line.strip()])
        one_definition(text, r"pub\(crate\) const CALLS_PER_MODEL: u64 = %d;" % trio.ATTEMPTS_WITH_RETRY,
                       "src/magi/mod.rs")
        self.assertEqual(1, trio.ATTEMPTS_WITHOUT_RETRY)
        # The saturating subtraction: at or below the classifier's slice the raw ceiling is 0.
        for factor in (120, 240, 720):
            for wall in range(0, trio.CLASSIFY_TIMEOUT_SECS + 1):
                self.assertEqual(0, trio.raw_ceiling(wall, factor), (wall, factor))
        # The truncating division, and the factor's three terms.
        self.assertEqual((20 - 6) * 100 // 720, trio.raw_ceiling(20, trio.attempt_factor(2, False)))
        self.assertEqual(2 * 3 * 120, trio.attempt_factor(2, False))
        self.assertEqual(1 * 1 * 120, trio.attempt_factor(0, True))

    def test_the_trio_scenarios_budget_copy_is_the_one_pinned_here(self) -> None:
        # [MAGI §6 MS2 loop 27, F1b2b-balthasar] smoke.scenarios.trio carries its own copy of the budget
        # constants, which the code-review self-tests now reuse; it is tied to the copy the guards above pin
        # against src/magi/mod.rs, so neither can drift alone.
        from smoke.scenarios import trio
        self.assertEqual((code_review.CLASSIFY_SECS, code_review.SLACK_PCT, code_review.FLOOR_SECS),
                         (trio.CLASSIFY_TIMEOUT_SECS, trio.TIMEOUT_SLACK_PCT, trio.CEILING_FLOOR_SECS))

    def test_the_budget_split_and_ceiling_are_the_ones_s50_checks(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-m2] _budget_fields_ok asserts wall == total // batches and a ceiling
        # below the floor: pre_budget's integer split and its RAW (never floor-lifted) refusal ceiling.
        text = _source("src/review/clock.rs")
        for pattern in (r"Ok\(n\) => remaining_secs / n,",
                        r"let per_batch = batch_wall_clock_secs\(total_secs, batches\);",
                        r"per_batch_ceiling_secs: raw_ceiling_from_factor\(\s*per_batch,"):
            one_definition(text, pattern, "src/review/clock.rs")


class StderrWordingTests(unittest.TestCase):
    """The texts S48-S51 parse out of the product's ``stderr``."""

    # MAGI FIX (S6 loop 5): the regexes were tied to the product's wording with
    # nothing to say when that wording moved (F1-b1).
    def setUp(self) -> None:
        self.error = _source("src/review/error.rs")
        self.filter = _source("src/review/filter.rs")

    def test_the_omission_causes_are_the_products(self) -> None:
        found = one_definition(self.filter, r"fn as_str\(self\) -> &'static str \{(.*?)\n    \}\n",
                               "src/review/filter.rs", re.S)
        body = without_comments(found.group(1))
        causes = re.findall(r'Self::\w+ => "([^"]+)"', body)
        # [MAGI §6 MS2 loop 12] an arm may name a constant instead of a literal (HARD_LINK_CAUSE, which
        # src/review/source.rs defines because the explicit-file error shares it); resolve it there.
        source = _source("src/review/source.rs")
        for name in re.findall(r"Self::\w+ => ([A-Z_]+),", body):
            value = one_definition(source, r'pub const %s: &str = "([^"]+)";' % name, "src/review/source.rs")
            causes.append(value.group(1))
        self.assertTrue(causes)
        self.assertEqual(sorted(causes), sorted(code_review.OMISSION_CAUSES))

    def test_missing_is_an_error_cause_never_an_omission_cause(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-c2] "missing" names an explicit file in NotReviewable; it is not a
        # SkipCause, so it can never reach the omission list and OMISSION_CAUSES must not carry it.
        one_definition(self.filter, re.escape('pub(crate) const MISSING: &str = "missing";'),
                       "src/review/filter.rs")
        one_definition(self.error, re.escape('#[error("{} is not reviewable: {cause}", display_path(.path))]'),
                       "src/review/error.rs")
        self.assertNotIn('=> "missing"', self.filter)
        self.assertNotIn("missing", code_review.OMISSION_CAUSES)

    def test_the_omission_list_is_rendered_as_the_patterns_expect(self) -> None:
        one_occurrence(self.filter, 'write!(f, "{} ({})", self.path, self.cause.as_str())', "src/review/filter.rs")
        one_occurrence(self.error, 'format!("; left out: {}{more}", shown.join(", "))', "src/review/error.rs")
        one_occurrence(self.error, 'format!(", and {rest} more")', "src/review/error.rs")
        rendered = "error: x; left out: a b (binary), .magi/c (ignored), and 2 more"
        entries, failure = code_review._omissions(rendered)
        self.assertEqual("", failure)
        self.assertEqual([("a b", "binary"), (".magi/c", "ignored")], entries)

    def test_the_budget_refusal_template_matches_the_pattern(self) -> None:
        template = one_definition(self.error, r'#\[error\(\s*"(\{batches\} batches leave.*?)",\s*budget_advice',
                                  "src/review/error.rs", re.S)
        one_definition(self.error, re.escape('BudgetLimit::Timeout => format!("pass --timeout {min_secs} or more")'),
                       "src/review/error.rs")
        template_text = re.sub(r"\\\n\s*", "", template.group(1))
        # [MAGI §6 MS2 loop 10, F1a-m1] the count is the bare "{batches} batches" at every N: no plural
        # branch exists in the template, so one batch renders "1 batches" and the pattern still matches.
        self.assertTrue(template_text.startswith("{batches} batches leave "), template_text)
        # R15's 20 s under the default config: one batch gets 20 s (raw ceiling 1 s), two get 10 s (0 s).
        for batches, wall, ceiling in (("1", "20", "1"), ("2", "10", "0")):
            values = {"batches": batches, "per_batch_wall_clock_secs": wall, "min_per_batch_secs": "114",
                      "floor_secs": "15", "per_batch_ceiling_secs": ceiling}
            text = template_text
            for name, value in values.items():
                text = text.replace("{%s}" % name, value)
            minimum = str(int(batches) * 114)
            text = text.replace("{}", "pass --timeout %s or more" % minimum)
            match = code_review.BUDGET_PATTERN.fullmatch(text)
            self.assertIsNotNone(match, text)
            self.assertEqual((batches, wall, "114", "15", ceiling, minimum),
                             match.group("batches", "wall", "need", "floor", "ceiling", "minimum"))
            self.assertTrue(code_review._budget_fields_ok(match), text)

    def test_the_fixed_phrases_are_the_products(self) -> None:
        # [MAGI F1b2a-e] a bare assertIn passed an empty or one-letter copy; each copy must be a phrase of
        # at least two words standing at word boundaries of the very literal found in the Rust source.
        for copy, product in ((code_review.OUTSIDE_WORKSPACE_TEXT, "path is outside the workspace: {}"),
                              (code_review.NOTHING_TO_REVIEW_TEXT,
                               "nothing to review: no reviewable file in the input{}")):
            one_occurrence(self.error, '"%s"' % product, "src/review/error.rs")
            self.assertTrue(_is_word_bounded_phrase(copy, product), (copy, product))

    def test_a_trivial_copy_of_a_fixed_phrase_fails_the_check(self) -> None:
        # [MAGI F1b2a-e] assertIn(copy, product_text) held for an empty copy, and S48/S49/S51 then matched
        # any stderr at all: the copy must be a phrase of words, cut at word boundaries of the product's text.
        for name in ("OUTSIDE_WORKSPACE_TEXT", "NOTHING_TO_REVIEW_TEXT"):
            for trivial in ("", " ", "e", "the", "side the work", "nothing to rev"):
                with mock.patch.object(code_review, name, trivial):
                    with self.assertRaises(AssertionError, msg=(name, trivial)):
                        self.test_the_fixed_phrases_are_the_products()


#: The comment in ``src/review/report.rs`` that heads the test's list of every summary key.
_REPORT_KEYS_ANCHOR = "every documented key is present"


def _call_list(body: str, call: str, where: str) -> str:
    """The inside of the one array literal *call* (``.split([``-style, ending at ``[``) takes in *body*.

    Args:
        body: A Rust function body, ``\\r`` stripped.
        call: The call's text up to and including its ``[``.
        where: The function, for the failure message.

    Returns:
        str: The text between the call's ``[`` and the next ``]``.

    Raises:
        AssertionError: When the call does not appear in code exactly once, or its list never closes.

    MAGI FIX (S6 loop 36): the call was found by line, so a line of a multi-line string holding the old
    call answered for code that had drifted. It is now located in :func:`structure_only`'s view, where
    comments and literals are blank at the same offsets, and the list is read from *body* at that offset.
    """
    view = structure_only(body)
    starts = [match.end() for match in re.finditer(re.escape(call), view)]
    if len(starts) != 1:
        raise AssertionError("%s: expected %r once in code, found %d" % (where, call, len(starts)))
    close = view.find("]", starts[0])
    if close < 0:
        raise AssertionError("%s: the list %r opens is never closed" % (where, call))
    return body[starts[0]:close]


def _pinned_report_keys(report: str) -> set[str]:
    """The keys the report's own test lists after :data:`_REPORT_KEYS_ANCHOR`, read from code only.

    Args:
        report: ``src/review/report.rs``, ``\\r`` stripped.

    Returns:
        set[str]: Every quoted key in the list.

    Raises:
        AssertionError: When the anchor is not there exactly once, or not in a comment, or the code right
            after it is not the list.

    MAGI FIX (S6 loop 36): the list was searched in the raw text, so an old copy left in a comment after
    the anchor answered for it. The anchor (itself a comment) is found in the raw text and the list in
    :func:`without_comments`'s view from there; both keep offsets.

    MAGI FIX (S6 loop 40, F1b2a-melchior): that view kept literals, so a fixture after the anchor answered
    for a list that had gone. The list is now located in :func:`structure_only`'s view, as the first code
    after the anchor, and the anchor must be a comment.
    """
    if report.count(_REPORT_KEYS_ANCHOR) != 1:
        raise AssertionError("expected the anchor %r once in src/review/report.rs" % _REPORT_KEYS_ANCHOR)
    if _REPORT_KEYS_ANCHOR in without_comments(report):
        raise AssertionError("the anchor %r is not a comment in src/review/report.rs" % _REPORT_KEYS_ANCHOR)
    view = structure_only(report)
    after = report.index(_REPORT_KEYS_ANCHOR) + len(_REPORT_KEYS_ANCHOR)
    opening = re.compile(r"\s*for k in \[").match(view, after)
    close = view.find("]", opening.end()) if opening is not None else -1
    if close < 0:
        raise AssertionError("the code after the anchor in src/review/report.rs is not the summary key list")
    return set(re.findall(r'"(\w+)"', report[opening.end():close]))


#: The envelope's summary field in ``src/headless/output.rs``: omitted when absent, never renamed.
_ENVELOPE_SUMMARY_FIELD = (r'#\[serde\(skip_serializing_if = "is_absent"\)\]\n[ \t]*(\w+): '
                           r"&'a Option<serde_json::Value>,")
#: The headless envelope's struct in ``src/headless/output.rs``: serde writes its fields as the envelope's keys.
_ENVELOPE_STRUCT = "struct WireOutcome<'a> {"
#: A serde argument that changes a serialized key, on a field (``rename``) or the struct (``rename_all``).
_SERDE_RENAME = re.compile(r"\brename(?:_all)?\s*[=(]")


def _struct_with_attributes(text: str, declaration: str, where: str) -> str:
    """A struct's outer attributes and body, comments blanked: all that names its serialized keys.

    Args:
        text: A Rust source file, ``\\r`` stripped.
        declaration: The struct's declaration line through its ``{``.
        where: The file, for the failure message.

    Returns:
        str: The attribute lines above the struct, up to the previous item's last line (one that ends
        in ``}``, ``;`` or ``{``), then the struct through its closing ``}`` in column one.

    Raises:
        AssertionError: When the struct is not declared exactly once, or never closes.
    """
    view = without_comments(text)
    start = one_definition(text, re.escape(declaration), where).start()
    end = structure_only(text).find("\n}\n", start)
    if end == -1:
        raise AssertionError("%s: %r never closes" % (where, declaration))
    head = []
    for line in reversed(view[:start].split("\n")[:-1]):
        if line.rstrip().endswith(("}", ";", "{")):
            break
        head.append(line)
    return "\n".join(reversed(head)) + "\n" + view[start:end]


def _envelope_summary_key(output: str) -> str:
    """The name the headless envelope serializes the code-review summary under.

    MAGI FIX (S6 loop 36): nothing tied S47's ``"code_review"`` to the product, so renaming the
    envelope's field left every guard green and the drift surfaced only as a FAIL at SMOKE time.

    Args:
        output: ``src/headless/output.rs``, ``\\r`` stripped.

    Returns:
        str: The field's name, which serde writes unchanged: no ``rename`` on any of the struct's
        fields, nor a ``rename_all`` on the struct.

    Raises:
        AssertionError: When the struct renames a key, or no such field is declared in it, or more than one is.
    """
    # MAGI FIX (S6 loop 39, F1b2a-melchior): only the attribute right above the field was read, so a
    # rename on a line of its own, or a rename_all on the struct, moved the key with the guard green.
    struct = _struct_with_attributes(output, _ENVELOPE_STRUCT, "src/headless/output.rs")
    renamed = _SERDE_RENAME.search(struct)
    if renamed is not None:
        raise AssertionError("src/headless/output.rs: the envelope struct renames a key (%r)" % renamed.group(0))
    return one_definition(struct, _ENVELOPE_SUMMARY_FIELD, "WireOutcome in src/headless/output.rs").group(1)


#: The structs whose fields S47 reads off each entry, and those fields' declarations.
_ENTRY_STRUCTS = (("src/review/consolidate.rs", "ConsolidatedFinding",
                   ("pub file: Option<String>,", "pub line: Option<u32>,")),
                  ("src/review/validate.rs", "Discarded", ("pub cause: DiscardCause,",)))
#: The one serde argument the entry structs may carry: it changes a value's form, never its key.
_ENTRY_SERDE_ARGUMENTS = frozenset({"serialize_with"})


def _serde_arguments(view: str) -> list[str]:
    """The name of every top-level argument of every ``serde(...)`` in *view*.

    Args:
        view: Rust code with comments and literals blanked (:func:`structure_only`).

    Returns:
        list[str]: Each argument's leading identifier, in order; ``cfg_attr``'s inner ``serde(...)`` included.

    Raises:
        AssertionError: When a ``serde(`` never closes.

    Example:
        >>> _serde_arguments("#[serde(rename(serialize =    ), skip)]")
        ['rename', 'skip']
    """
    names = []
    for match in re.finditer(r"\bserde\s*\(", view):
        depth, start = 1, match.end()
        for index in range(start, len(view)):
            depth += {"(": 1, ")": -1}.get(view[index], 0)
            if depth == 0:
                break
        else:
            raise AssertionError("a serde( attribute never closes")
        flat, level = [], 0
        for char in view[start:index]:
            level += {"(": 1, ")": -1}.get(char, 0)
            flat.append(char if level == 0 or char in "()" else " ")
        names += re.findall(r"(?:^|,)\s*(\w+)", "".join(flat))
    return names


def _require_entry_fields_serialized(source) -> None:
    """Fail unless every field of :data:`_ENTRY_STRUCTS` is serialized under its own name.

    MAGI FIX (S6 loop 40, F1b2a-melchior): only ``rename`` in the body was refused, so a
    ``skip_serializing``, ``flatten`` or a serde attribute above the struct's derive dropped a key S47
    reads with the guard green. Every serde argument on the struct and its fields is now checked against
    :data:`_ENTRY_SERDE_ARGUMENTS`.

    Args:
        source: Reads a repository path into its text, ``\\r`` stripped.

    Raises:
        AssertionError: When a struct is not declared once, has no serde ``Serialize`` derive, carries a
            serde argument outside the allowed set, or does not declare a field.
    """
    for path, struct, fields in _ENTRY_STRUCTS:
        text = _struct_with_attributes(source(path), "pub struct %s {" % struct, path)
        view = structure_only(text)
        if re.search(r"#\[derive\([^)]*\bserde::Serialize\b", view) is None:
            raise AssertionError("%s in %s: no serde::Serialize derive" % (struct, path))
        refused = [name for name in _serde_arguments(view) if name not in _ENTRY_SERDE_ARGUMENTS]
        if refused:
            raise AssertionError("%s in %s: serde arguments that can move or drop a key: %s" % (struct, path, refused))
        for field in fields:
            one_definition(text, re.escape(field), "%s in %s" % (struct, path))


#: The code lines of the product's ``attempt_factor`` body, comments dropped, the smoke copies' arithmetic.
_ATTEMPT_FACTOR_LINES = ["let attempts_per_model: u64 = if retry_disabled { 1 } else { CALLS_PER_MODEL };",
                         "let models_per_mage = u64::from(max_rotations).saturating_add(1);",
                         "attempts_per_model", ".saturating_mul(models_per_mage)",
                         ".saturating_mul(100 + HEADLESS_TIMEOUT_SLACK_PCT)"]


def _attempt_factor_lines(text: str) -> list[str]:
    """The code lines of ``attempt_factor``'s body in *text* (``src/magi/mod.rs``), stripped, comments dropped.

    MAGI FIX (S6 loop 38): its lines were pinned one by one, so recombining them left the smoke copy stale
    with every guard green; the body is compared whole against :data:`_ATTEMPT_FACTOR_LINES`.

    Raises:
        AssertionError: When the function is not defined exactly once.
    """
    body = one_definition(text, r"pub fn attempt_factor\(max_rotations: u32, retry_disabled: bool\) -> u64 \{(.*?)\n\}",
                          "src/magi/mod.rs", re.S).group(1)
    return [line.strip() for line in without_comments(body).split("\n") if line.strip()]


def _is_document_assignment(node) -> bool:
    """Whether *node* is ``document = <expr>.json()``: one ``Name`` target, a no-argument ``.json()`` call."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == _DOCUMENT_NAME and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "json"
            and not node.value.args and not node.value.keywords)


def _is_empty_fallback(node) -> bool:
    """Whether *node* is ``section = {}``: one ``Name`` target, an empty dict literal."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == _SECTION_NAME and isinstance(node.value, ast.Dict) and not node.value.keys)


def _is_not_a_dict_test(node) -> bool:
    """Whether *node* is exactly ``not isinstance(section, dict)``."""
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Call)
            and isinstance(node.operand.func, ast.Name) and node.operand.func.id == "isinstance"
            and len(node.operand.args) == 2 and not node.operand.keywords
            and isinstance(node.operand.args[0], ast.Name) and node.operand.args[0].id == _SECTION_NAME
            and isinstance(node.operand.args[1], ast.Name) and node.operand.args[1].id == "dict")


def _guarded_fallback(assignment, parents: dict):
    """The one ``section = {}`` the summary read may be followed by, or ``None`` when it has none.

    MAGI FIX (S6 loop 38): any ``section = {}`` in the read's scope was accepted, so an unconditional
    one, or a second one, replaced the summary at run time while the key guard still read the later
    ``.get`` calls. Only the statement right after the read, an ``if not isinstance(section, dict):``
    with no ``else`` whose whole body is ``section = {}``, carries the fallback.

    Args:
        assignment: The module's one ``section = document.get(CODE_REVIEW_KEY)``.
        parents: Each node's parent in the module.

    Returns:
        ast.Assign | None: The fallback assignment, when that ``if`` follows the read.
    """
    block = next((getattr(parents.get(assignment), field) for field in ("body", "orelse", "finalbody")
                  if assignment in getattr(parents.get(assignment), field, [])), [])
    following = block[block.index(assignment) + 1:block.index(assignment) + 2]
    if not following or not isinstance(following[0], ast.If):
        return None
    guard = following[0]
    if (_is_not_a_dict_test(guard.test) and not guard.orelse and len(guard.body) == 1
            and _is_empty_fallback(guard.body[0])):
        return guard.body[0]
    return None


def _require_one_summary_binding(scenario: str) -> None:
    """Fail unless *scenario* binds the summary key, the document and ``section`` only in their accepted forms.

    MAGI FIX (S6 loop 36): :class:`_SectionReads` only allowed the definition and the assignment, so a
    scenario reading another section under ``section`` kept every guard green. Checked on the real module
    only: the synthetic snippets of :class:`SummaryKeyExtractionTests` carry neither.

    MAGI FIX (S6 loop 37): only the summary assignment was counted, so ``section`` rebound inside a helper or
    after the read (``section = document.get(CONSULT_KEY)``), or the document rebound, left the scenario
    reading something else with every guard green. Every binding, by any form :func:`_names_bound_by` knows,
    is now checked.

    Args:
        scenario: The scenario module's source.

    Raises:
        AssertionError: When the key is not defined once at top level or the summary not read once
            (naming both counts); when ``document`` is bound other than once, by ``document = <expr>.json()``;
            or naming the line of a binding of ``section`` that is none of: the summary assignment, the one
            guarded fallback right after it (:func:`_guarded_fallback`), or a parameter of an undecorated
            top-level ``def`` the module binds once.
    """
    tree = ast.parse(scenario)
    definitions = sum(1 for node in tree.body if _is_summary_definition(node))
    assignments = [node for node in ast.walk(tree) if _is_section_assignment(node)]
    if (definitions, len(assignments)) != (1, 1):
        raise AssertionError("expected one %s definition and one section assignment, found %d and %d"
                             % (_SUMMARY_KEY_NAME, definitions, len(assignments)))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    document = [node for node in ast.walk(tree) if _DOCUMENT_NAME in _names_bound_by(node)]
    if len(document) != 1 or not _is_document_assignment(parents.get(document[0])):
        raise AssertionError("expected document bound once, by document = <expr>.json(), found %d binding(s)"
                             % len(document))
    counts = _binding_counts(tree)
    credited = {node for node in tree.body
                if isinstance(node, ast.FunctionDef) and not node.decorator_list and counts.get(node.name) == 1}
    fallback = _guarded_fallback(assignments[0], parents)
    for node in ast.walk(tree):
        if _SECTION_NAME not in _names_bound_by(node):
            continue
        statement = parents.get(node)
        if statement is assignments[0] and node is assignments[0].targets[0]:
            continue
        if statement is fallback and fallback is not None:
            continue
        if isinstance(node, ast.arg) and parents.get(statement) in credited:
            continue
        raise AssertionError("line %d: a binding of section the key guard does not follow: %s"
                             % (getattr(node, "lineno", 0), ast.unparse(statement if statement is not None else node)))


class SyncGuardMutationTests(unittest.TestCase):
    """[MAGI S6 loop 36, F1b2a] guards that stayed green with the guarded thing broken."""

    def test_a_call_only_inside_a_literal_fails_the_call_list_read(self) -> None:
        # one_occurrence did not skip lines inside a multi-line string, so the trim list was read from a fixture
        # while the code had drifted.
        body = ('\n    let fixture = r"\n.trim_end_matches([\'.\', \' \'])\n";\n'
                "    trim_more(path).split(['/', '\\\\'])\n")
        with self.assertRaises(AssertionError):
            _call_list(body, ".trim_end_matches([", "is_never_reviewed")
        self.assertEqual("'.', ' '", _call_list("    path.trim_end_matches(['.', ' '])\n",
                                                ".trim_end_matches([", "is_never_reviewed"))

    def test_a_key_list_read_from_a_comment_fails_the_report_key_read(self) -> None:
        # The list was searched in the raw text, so an old copy in a comment after the anchor answered for it.
        report = ('    // every documented key is present\n    // was: for k in ["batches", "old"]\n'
                  '    for k in ["batches", "new"] {\n')
        self.assertEqual({"batches", "new"}, _pinned_report_keys(report))

    def test_a_key_list_in_a_literal_or_an_anchor_in_code_fails_the_report_key_read(self) -> None:
        # [MAGI S6 loop 40, F1b2a-melchior] the list was searched with literals kept, so a fixture after the
        # anchor answered for the real list; and the anchor itself could be a string, not the test's comment.
        report = _source("src/review/report.rs")
        anchor = "// [CP2 MS2 S2a2 loop 10] every documented key is present, so a dropped field goes red.\n"
        self.assertEqual(1, report.count(anchor))
        for new in (anchor + '        let _fixture = "for k in [\\"hidden\\"]";\n',
                    'let _ = "every documented key is present";\n'):
            with self.subTest(new=new), self.assertRaises(AssertionError):
                _pinned_report_keys(report.replace(anchor, new))

    def test_the_envelope_serializes_the_summary_under_the_key_s47_reads(self) -> None:
        # Nothing tied "code_review" to the product: renaming the envelope field left every test green.
        output = _source("src/headless/output.rs")
        self.assertEqual(code_review.CODE_REVIEW_KEY, _envelope_summary_key(output))
        renamed = output.replace('#[serde(skip_serializing_if = "is_absent")]',
                                 '#[serde(skip_serializing_if = "is_absent", rename = "review_summary")]')
        with self.assertRaises(AssertionError):
            _envelope_summary_key(renamed)

    def test_a_rename_on_its_own_line_or_on_the_struct_fails_the_envelope_key(self) -> None:
        # [MAGI §6 MS2 loop 39, F1b2a-melchior] only the attribute right above the field was read, so a
        # ``rename`` on a line of its own, or a ``rename_all`` on the struct, moved the key with the guard green.
        output = _source("src/headless/output.rs")
        field = '    #[serde(skip_serializing_if = "is_absent")]\n    code_review:'
        struct = "#[derive(Debug, Serialize)]\nstruct WireOutcome<'a> {"
        for old, new in ((field, '    #[serde(rename = "review_summary")]\n' + field),
                         (field, '    #[serde(\n        rename(serialize = "review_summary")\n    )]\n' + field),
                         (struct, "#[derive(Debug, Serialize)]\n#[serde(rename_all = \"camelCase\")]\n"
                                  "struct WireOutcome<'a> {"),
                         (struct, "#[serde(\n    rename_all = \"camelCase\",\n)]\n" + struct)):
            self.assertEqual(1, output.count(old))
            with self.subTest(new=new), self.assertRaises(AssertionError):
                _envelope_summary_key(output.replace(old, new))

    def test_a_scenario_that_reads_another_section_fails_the_binding_check(self) -> None:
        # The definition and the one assignment were only allowed, never required: the real scenario reading
        # another section stayed green.
        _require_one_summary_binding(_source("smoke/scenarios/code_review.py"))
        for scenario in ('CODE_REVIEW_KEY = %r\nCONSULT_KEY = "consult"\ndocument = {}\n'
                         'section = document.get(CONSULT_KEY)\n' % code_review.CODE_REVIEW_KEY,
                         'CONSULT_KEY = "consult"\ndocument = {}\nsection = document.get(CONSULT_KEY)\n'):
            with self.subTest(scenario=scenario), self.assertRaises(AssertionError):
                _require_one_summary_binding(scenario)

    #: The scenario module's shape: one parsed document, the one summary read, the empty fallback in the same
    #: function, and a helper that takes ``section`` as its parameter.
    _SCENARIO = ('CODE_REVIEW_KEY = %r\nCONSULT_KEY = "consult"\n'
                 'def _list_under(section, key):\n    return section.get(key)\n'
                 'def scenario(run):\n    document = run.output.json()\n'
                 '    section = document.get(CODE_REVIEW_KEY)\n'
                 '    if not isinstance(section, dict):\n        section = {}\n'
                 '    return _list_under(section, "findings")\n' % code_review.CODE_REVIEW_KEY)

    def test_a_section_or_document_rebound_fails_the_binding_check(self) -> None:
        # [MAGI S6 loop 37, F1b2a] only the summary assignment was counted, so section rebound inside the helper or
        # after the read, or the document rebound, left the scenario reading something else with every guard green.
        _require_one_summary_binding(self._SCENARIO)
        helper = '    return section.get(key)\n'
        tail = '    return _list_under(section, "findings")\n'
        for mutated in (self._SCENARIO.replace(helper, '    section = {}\n' + helper),
                        self._SCENARIO.replace(helper, '    section = {"findings": []}\n' + helper),
                        self._SCENARIO.replace(tail, '    section = document.get(CONSULT_KEY)\n' + tail),
                        self._SCENARIO.replace('section = {}', 'section = {"findings": []}'),
                        self._SCENARIO.replace(tail, '    for section in ({},):\n        pass\n' + tail),
                        self._SCENARIO.replace(tail, '    f = lambda section: section\n' + tail),
                        self._SCENARIO + 'section = {}\n',
                        self._SCENARIO.replace(tail, '    document = {"code_review": {}}\n' + tail),
                        self._SCENARIO.replace('document = run.output.json()', 'document = {}'),
                        self._SCENARIO.replace(tail, '    document = run.output.json()\n' + tail),
                        # [MAGI S6 loop 38, F1b2a-melchior] an unconditional or a second empty fallback replaced
                        # the summary at run time while the key guard still read the later .get calls.
                        self._SCENARIO.replace('    if not isinstance(section, dict):\n        section = {}\n',
                                               '    section = {}\n'),
                        self._SCENARIO.replace(tail, '    section = {}\n' + tail),
                        self._SCENARIO.replace('if not isinstance(section, dict):', 'if True:'),
                        self._SCENARIO.replace('if not isinstance(section, dict):',
                                               'if not isinstance(section, dict) or True:'),
                        self._SCENARIO.replace('        section = {}\n', '        section = {}\n        pass\n')):
            with self.subTest(scenario=mutated), self.assertRaises(AssertionError):
                _require_one_summary_binding(mutated)

    def test_the_attempt_factor_is_pinned_as_a_whole(self) -> None:
        # [MAGI S6 loop 38, F1b2a-melchior] its three lines were pinned one by one, so recombining them
        # (adding where the product multiplies) kept every guard green with the smoke copy stale.
        text = _source("src/magi/mod.rs")
        self.assertEqual(_ATTEMPT_FACTOR_LINES, _attempt_factor_lines(text))
        recombined = text.replace("        .saturating_mul(models_per_mage)\n", "        .saturating_add(models_per_mage)\n")
        self.assertNotEqual(text, recombined)
        self.assertNotEqual(_ATTEMPT_FACTOR_LINES, _attempt_factor_lines(recombined))


class NeverReviewedTests(unittest.TestCase):
    """The protected names S51 checks are the ones the product never reviews."""

    # MAGI FIX (S6 loop 14): S51 left out the memory database prefix (F1a2-m2).
    def test_the_protected_names_are_the_products_never_reviewed_list(self) -> None:
        source = _source("src/review/source.rs")
        prefix = one_definition(source, r'const MEMORY_DB_PREFIX: &str = "([^"]+)";', "src/review/source.rs")
        listed = one_definition(source, r"pub const NEVER_REVIEWED: &\[&str\] = &\[([^\]]*)\];",
                                "src/review/source.rs")
        self.assertEqual(prefix.group(1), code_review.MEMORY_DB_PREFIX)
        items = without_comments(listed.group(1))
        names = tuple(re.findall(r'"([^"]+)"', items))
        self.assertEqual(set(names), set(code_review.PROTECTED_DIRS))
        # [MAGI §6 MS2 loop 28, F1b2a-melchior] assertIn passed with any number of extra identifiers, whose
        # values escaped the sync; the entries that are not literals are exactly the prefix's constant.
        self.assertEqual(["MEMORY_DB_PREFIX"],
                         [token.strip() for token in re.sub(r'"[^"]*"', "", items).split(",") if token.strip()])

    # MAGI FIX (S6 loop 17): same_product_name copies is_never_reviewed's folding rule with no pin, so
    # a wider product rule would have left S51's candidates behind in silence (F1a1-melchior).
    def test_the_name_folding_is_the_one_is_never_reviewed_applies(self) -> None:
        found = one_definition(_source("src/review/source.rs"),
                               r"pub fn is_never_reviewed\(path: &str\) -> bool \{(.*?)\n\}",
                               "src/review/source.rs", re.S)
        body = found.group(1)
        # [MAGI §6 MS2 loop 28, F1b2a-melchior] '(.)' could not read an escaped literal such as '\\\\', so a
        # character the product trims could be missing from the harness's copy with the guard green.
        chars = char_list(_call_list(body, ".trim_end_matches([", "is_never_reviewed"))
        self.assertEqual(sorted(chars), sorted(runs.TRIMMED_TRAILING))
        self.assertEqual({"/", "\\"}, set(char_list(_call_list(body, ".split([", "is_never_reviewed"))))
        one_occurrence(body, ".split(['/', '\\\\'])", "is_never_reviewed")
        # Every comparison folds ASCII case and nothing else, as same_product_name does.
        self.assertGreater(without_comments(body).count("eq_ignore_ascii_case"), 0)
        self.assertNotIn("to_lowercase", body)
        self.assertNotIn("==", body)


class SummaryKeyExtractionTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1b2a-melchior] a key read in an access shape the guard does not parse fails it.

    The keys were collected from two access shapes only, so a key read with brackets or another accessor
    was missing from the set and the fixed-set comparison still passed.
    """

    _KNOWN = ('a = section.get("batches")\nitems, failure = _list_under(section, "findings")\n'
              'def _list_under(section, key):\n    items = section.get(key)\n')

    def test_the_two_known_shapes_are_read(self) -> None:
        self.assertEqual({"batches", "findings"}, _summary_keys_read(self._KNOWN))

    def test_any_other_access_shape_fails_the_read(self) -> None:
        for shape in ('x = section["new_key"]', "x = section['new_key']", 'x = section.setdefault("k")',
                      'x = section.get("k", 0)', "x = section.get(name)", 'x = _list_under(section, name)',
                      'x = {**section}', "for k in section:\n    pass", "x = getattr(section, 'k')"):
            with self.assertRaises(AssertionError, msg=shape):
                _summary_keys_read(self._KNOWN + shape + "\n")
        # [MAGI F1b2a-c] section.get(key) passed anywhere the innermost def was named _list_under, so a key
        # rebound inside it, bound by a lambda default or a comprehension, or a nested _list_under read a
        # literal the guard never saw.
        for shape in ('def _list_under(section, key):\n    key = "hidden"\n    return section.get(key)',
                      'def _list_under(section, key):\n    f = lambda key="hidden": section.get(key)\n'
                      '    return f()',
                      'def _list_under(section, key):\n    return [section.get(key) for key in ("hidden",)]',
                      'def _outer(section):\n    def _list_under(key="hidden"):\n        return section.get(key)\n'
                      '    return _list_under()\ny = _outer(section)'):
            with self.assertRaises(AssertionError, msg=shape):
                _summary_keys_read(self._KNOWN + shape + "\n")
        # MAGI FIX (S6 loop 34): only the name section was followed, so the summary read through another path
        # (another name, a chained get, brackets on the document) credited no key and the guard stayed green.
        bound = self._BOUND + self._KNOWN
        self.assertEqual({"batches", "findings"}, _summary_keys_read(bound))
        for shape in ('cr = document.get(CODE_REVIEW_KEY)\nz = cr.get("hidden")',
                      'z = document.get(CODE_REVIEW_KEY).get("hidden")',
                      '_bogus = document.get(CODE_REVIEW_KEY, {}).get("no_such_key")',
                      'z = document["code_review"]["hidden"]',
                      'z = document.get("code_review")',
                      'z = code_review.CODE_REVIEW_KEY',
                      'CODE_REVIEW_KEY = "elsewhere"',
                      'section = document.get(CODE_REVIEW_KEY)'):
            with self.subTest(shape=shape), self.assertRaises(AssertionError):
                _summary_keys_read(bound + shape + "\n")

    #: The summary key's definition and the one assignment that reads the summary, as the scenario module writes
    #: them (``smoke/scenarios/code_review.py``).
    _BOUND = ('CODE_REVIEW_KEY = %r\ndocument = {}\nsection = document.get(CODE_REVIEW_KEY)\n'
              % code_review.CODE_REVIEW_KEY)

    #: The calls of :data:`_KNOWN` without its helper, for shapes that define ``_list_under`` themselves.
    _CALLS = 'a = section.get("batches")\nitems, failure = _list_under(section, "findings")\n'

    def test_a_list_under_the_read_cannot_trust_fails_the_read(self) -> None:
        # MAGI FIX (S6 loop 33): _list_under's calls were credited by its name alone, so a helper whose parameters
        # were renamed, a lambda or import bound to the name, or a second binding of it read a key unseen.
        for shape in ('def _list_under(other, key):\n    return other.get("hidden")',
                      'def _list_under(section, name):\n    return section.get(name)',
                      'def _list_under(key, section):\n    return section.get(key)',
                      'def _list_under(section, key, extra=None):\n    return section.get(key)',
                      'def _list_under(section, *, key):\n    return section.get(key)',
                      '_list_under = lambda section, key: section.get("hidden")',
                      'from json import loads as _list_under',
                      'import _list_under',
                      'async def _list_under(section, key):\n    return section.get(key)',
                      # MAGI FIX (S6 loop 34): a trusted-shaped helper credited its call's key while reading nothing.
                      'def _list_under(section, key):\n    return {}.get(key)',
                      'def _list_under(section, key):\n    return (lambda: section.get(key))()'):
            with self.subTest(shape=shape), self.assertRaises(AssertionError):
                _summary_keys_read(self._CALLS + shape + "\n")
        helper = 'def _list_under(section, key):\n    return section.get(key)\n'
        for rebinding in ('_list_under = lambda s, k: s.get("hidden")', 'from json import loads as _list_under',
                          'class _list_under:\n    pass', 'for _list_under in ():\n    pass',
                          'with open("x") as _list_under:\n    pass', 'def _list_under(section, key):\n    pass',
                          'def f():\n    global _list_under\n    _list_under = None',
                          'try:\n    pass\nexcept OSError as _list_under:\n    pass',
                          'match 1:\n    case _list_under:\n        pass', 'del _list_under',
                          '(_list_under := lambda section, key: section.get(key))'):
            with self.subTest(rebinding=rebinding), self.assertRaises(AssertionError):
                _summary_keys_read(self._CALLS + helper + rebinding + "\n")
        # Every form that rebinds key inside the helper.
        for body in ('    match "hidden":\n        case key:\n            pass',
                     '    match ["hidden"]:\n        case [*key]:\n            pass',
                     '    match {"a": "hidden"}:\n        case {**key}:\n            pass',
                     '    import os as key', '    from os import sep as key', '    import key',
                     '    try:\n        pass\n    except OSError as key:\n        pass',
                     '    def key():\n        pass', '    class key:\n        pass',
                     '    def inner():\n        global key', '    def inner():\n        nonlocal key',
                     '    async def key():\n        pass', '    (key := "h")'):
            shape = 'def _list_under(section, key):\n%s\n    return section.get(key)' % body
            with self.subTest(body=body), self.assertRaises(AssertionError):
                _summary_keys_read(self._CALLS + shape + "\n")
        # The trusted helper alone, called with literal keys, is still read.
        self.assertEqual({"batches", "findings"}, _summary_keys_read(self._CALLS + helper))

    def test_a_spaced_get_still_reads_its_key(self) -> None:
        # [MAGI F1b2a] the regex wanted "section.get(" with no space, so a legal "section . get(...)" read a key
        # the fixed-set comparison never saw.
        for shape in ('x = section . get("new_key")', 'x = section.get( "new_key" )', 'x = (section).get("new_key")'):
            self.assertIn("new_key", _summary_keys_read(self._KNOWN + shape + "\n"), shape)

    def test_an_alias_of_section_fails_the_read(self) -> None:
        # [MAGI F1b2a] reading through another name dropped the key from the guard in silence.
        for shape in ('s2 = section\ny = s2.get("new_key")', 's2, s3 = section, 0', 'y = [section][0].get("k")',
                      'y = section if True else None'):
            with self.assertRaises(AssertionError, msg=shape):
                _summary_keys_read(self._KNOWN + shape + "\n")

    def test_passing_section_to_a_function_the_read_cannot_follow_fails_it(self) -> None:
        # [MAGI F1b2a] a helper outside the module, or one whose parameter is named otherwise, reads keys the
        # walk never sees.
        for shape in ('y = _elsewhere(section)', 'y = json.dumps(section)', 'y = _helper(other=section)',
                      'def _renamed(other):\n    return other.get("k")\ny = _renamed(section)',
                      'y = section.get(key)'):
            with self.assertRaises(AssertionError, msg=shape):
                _summary_keys_read(self._KNOWN + shape + "\n")

    def test_a_module_helper_taking_section_by_that_name_has_its_reads_collected(self) -> None:
        shape = 'def _helper(run_id, section):\n    return section.get("inner")\ny = _helper("R12", section)\n'
        self.assertEqual({"batches", "findings", "inner"}, _summary_keys_read(self._KNOWN + shape))
        self.assertEqual({"batches", "findings"},
                         _summary_keys_read(self._KNOWN + "ok = isinstance(section, dict)\nsection = {}\n"))

    #: A module helper that takes ``section`` by that name, and its call.
    _HELPER = 'def _helper(run_id, section):\n    return section.get("inner")\ny = _helper("R12", section)\n'

    def test_a_shadowed_or_rebound_module_helper_fails_the_read(self) -> None:
        # MAGI FIX (S6 loop 35): a call was credited by the name of a top-level def, so a local def of the same
        # name, a later rebinding of it or a decorator that replaces it ran code the walk never read.
        for shape in ('def f():\n    def _helper(run_id, other):\n        return other.get("hidden")\n'
                      '    return _helper("R12", section)',
                      '_helper = lambda run_id, other: other.get("hidden")',
                      'from json import loads as _helper',
                      'class _helper:\n    pass'):
            with self.subTest(shape=shape), self.assertRaises(AssertionError):
                _summary_keys_read(self._KNOWN + self._HELPER + shape + "\n")
        decorated = self._HELPER.replace("def _helper(", "@_wrap\ndef _helper(")
        with self.assertRaises(AssertionError):
            _summary_keys_read(self._KNOWN + decorated)

    def test_the_summary_reached_without_naming_its_key_fails_the_read(self) -> None:
        # MAGI FIX (S6 loop 35): only the key's name and literal were followed, so the summary read with a key
        # built at run time, through the document's values, or by a reflection builtin credited no key.
        bound = self._BOUND + self._KNOWN
        consult = 'CONSULT_KEY = "consult"\nconsult = document.get(CONSULT_KEY)\n'
        self.assertEqual({"batches", "findings"}, _summary_keys_read(bound + consult))
        for shape in ('cr = document.get("code" + "_review")\nz = cr.get("hidden")',
                      'k = "code" + "_review"\ncr = document.get(k)\nz = cr.get("hidden")',
                      'OTHER_KEY = "code" + "_review"\ncr = document.get(OTHER_KEY)',
                      'cr = next(iter(document.values()))\nz = cr.get("hidden")',
                      'alias = document\nz = alias.get("x")',
                      'cr = document.get(CONSULT_KEY, {})',
                      'z = vars()["section"].get("hidden")',
                      'z = globals()["section"].get("hidden")',
                      'z = locals()["section"].get("hidden")',
                      'z = getattr(document, "get")("x")',
                      'z = eval("section").get("hidden")',
                      # [MAGI S6 loop 38, F1b2a-caspar] the ban read bare names only, so the builtins module
                      # reached the same functions as an attribute.
                      'import builtins\nz = builtins.vars()["section"].get("hidden")',
                      'z = __builtins__.vars()["section"].get("hidden")',
                      'from builtins import vars as v\nz = v()["section"].get("hidden")',
                      'import builtins as b\nz = b.globals()["section"].get("hidden")',
                      'import sys\nz = sys.modules["builtins"].vars()["section"].get("hidden")',
                      'import importlib\nz = importlib.import_module("builtins").vars()["section"].get("x")',
                      # [MAGI S6 loop 40, F1b2a-caspar] the module object reached section as an attribute.
                      'import sys\nz = sys.modules[__name__].section.get("hidden")',
                      'import importlib\nz = importlib.import_module("smoke.scenarios.code_review")'
                      '.section.get("hidden")',
                      'import sys\nz = sys.modules[__name__].__dict__["section"].get("hidden")',
                      'CONSULT_KEY = "other"'):
            with self.subTest(shape=shape), self.assertRaises(AssertionError):
                _summary_keys_read(bound + consult + shape + "\n")


class ReportKeyTests(unittest.TestCase):
    """The JSON keys S47 reads exist in the product's report and envelope."""

    # MAGI FIX (S6 loop 6): S47's keys sat outside the drift guard, so a rename
    # would surface as an unexplained FAIL at SMOKE time (F1a-m2).
    def test_every_summary_key_s47_reads_is_one_the_report_pins(self) -> None:
        scenario = _source("smoke/scenarios/code_review.py")
        _require_one_summary_binding(scenario)
        self.assertEqual(code_review.CODE_REVIEW_KEY, _envelope_summary_key(_source("src/headless/output.rs")))
        read = _summary_keys_read(scenario)
        # [MAGI §6 MS2 loop 27, F1b2a-melchior] the keys were only inferred from two access shapes, so an access
        # rewritten in another shape dropped its key from the guard in silence. The set is written out: a key
        # the extraction stops seeing, or a new one, fails here and is reviewed on purpose.
        self.assertEqual({"batches", "batches_completed", "batches_planned", "discarded", "failed_batches",
                          "findings", "unmeasured_models"}, read)
        self.assertEqual(set(), read - _pinned_report_keys(_source("src/review/report.rs")))

    def test_the_entry_fields_s47_reads_are_serialized_under_their_own_names(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-m1] S47 reads file and line off each main finding and cause off each
        # discarded one; a serde rename on either struct would surface as an opaque FAIL at SMOKE time.
        _require_entry_fields_serialized(_source)

    def test_a_skipped_or_reshaped_entry_field_fails_the_entry_field_guard(self) -> None:
        # [MAGI S6 loop 40, F1b2a-melchior] only ``rename`` in the body was refused, so a skip_serializing on a
        # field S47 reads, or a serde attribute on the struct above its derive, dropped the key with the guard green.
        file_field, cause_field = "    pub file: Option<String>,", "    pub cause: DiscardCause,"
        struct = "/// One finding of the consolidated report.\n#[derive(Debug, Clone, serde::Serialize)]"
        for path, old, new in (
                ("src/review/consolidate.rs", file_field,
                 '    #[serde(skip_serializing_if = "Option::is_none")]\n' + file_field),
                ("src/review/consolidate.rs", file_field, "    #[serde(skip)]\n" + file_field),
                ("src/review/consolidate.rs", file_field,
                 '    #[cfg_attr(test, serde(skip_serializing))]\n' + file_field),
                ("src/review/validate.rs", cause_field, "    #[serde(flatten)]\n" + cause_field),
                ("src/review/consolidate.rs", struct, struct.replace("#[derive", '#[serde(into = "X")]\n#[derive')),
                ("src/review/consolidate.rs", struct, struct + '\n#[serde(rename_all = "UPPERCASE")]')):
            self.assertEqual(1, _source(path).count(old), (path, old))

            def mutated(name: str, path=path, old=old, new=new) -> str:
                text = _source(name)
                return text.replace(old, new) if name == path else text

            with self.subTest(new=new), self.assertRaises(AssertionError):
                _require_entry_fields_serialized(mutated)

    def test_the_run_field_the_harness_matches_is_the_one_the_log_writes(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-m1] S50 binds its run-log line to this run through logs.run_needle;
        # the product spells the field once, in src/logging/render.rs.
        one_definition(_source("src/logging/render.rs"),
                       re.escape('pub(crate) const RUN_FIELD: &str = "%s";' % logs.RUN_FIELD),
                       "src/logging/render.rs")
        self.assertEqual(b"run=4242-deadbeef", logs.run_needle("4242-deadbeef"))

    def test_the_stderr_run_id_is_printed_whatever_the_log_level(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-m2/b4] S50 runs at --log-level error and reads `run: <id>` off stderr.
        # The product prints it with eprint_audited, outside the tracing layer, so no level filters it; a
        # move into a tracing macro would make S50's log half unreachable and must fail here first.
        main = _source("src/main.rs")
        needle = 'magi_rs::notices::eprint_audited(&format!("%s{}", magi_rs::logging::run_id()));'
        one_definition(main, re.escape(needle % logs.STDERR_RUN_PREFIX), "src/main.rs")
        self.assertNotRegex(main, r'(?:info|warn|error|debug|trace|event)!\([^;]*"%s\{\}"'
                            % re.escape(logs.STDERR_RUN_PREFIX))

    def test_the_consensus_s47_reads_is_the_one_the_envelope_writes(self) -> None:
        one_definition(_source("src/tools/code_review.rs"),
                       re.escape('"consensus".into(),\n        crate::tools::consult::consensus_json(&c.consensus)'),
                       "src/tools/code_review.rs")
        consult = _source("src/tools/consult.rs")
        for key in code_review.CONSENSUS_TEXT_KEYS:
            one_definition(consult, re.escape('"%s": c.%s,' % (key, key)), "src/tools/consult.rs")


class GitConfigTests(unittest.TestCase):
    """The harness's git settings are a subset of the product's."""

    # MAGI FIX (S6 loop 8): HARNESS_GIT_CONFIG claimed to be a subset of
    # NEUTRAL_GIT_CONFIG with nothing checking it (F1a-m3).
    def test_every_harness_git_setting_is_one_the_product_passes(self) -> None:
        listing = one_definition(_source("src/system/review_git.rs"),
                                 r"pub\(crate\) const NEUTRAL_GIT_CONFIG: &\[&str\] = &\[(.*?)\];",
                                 "src/system/review_git.rs", re.S)
        product = set(re.findall(r'"([^"]*)"', without_comments(listing.group(1))))
        self.assertTrue(runs.HARNESS_GIT_CONFIG)
        self.assertEqual(set(), set(runs.HARNESS_GIT_CONFIG) - product)

    # MAGI FIX (S6 loop 11, F1a-m2): GIT_VISIBLE_QUERY copies the product's
    # enumeration query with nothing checking it; a product query that dropped
    # one of its flags would surface as an opaque SMOKE FAIL.
    def test_the_visibility_query_uses_only_flags_the_products_listing_passes(self) -> None:
        listing = one_definition(_source("src/review/input.rs"), r"const LS_FILES_ALL: &\[&str\] = &\[(.*?)\];",
                                 "src/review/input.rs", re.S)
        product = re.findall(r'"([^"]*)"', without_comments(listing.group(1)))
        self.assertEqual("ls-files", runs.GIT_VISIBLE_QUERY[0])
        self.assertEqual(product[0], runs.GIT_VISIBLE_QUERY[0])
        self.assertEqual(set(), set(runs.GIT_VISIBLE_QUERY) - set(product))

    # MAGI FIX (S6 loop 18, F1a1-melchior): git_env() drops every GIT_* variable
    # on the grounds that none changes what the product's git sees, with
    # nothing checking the product's environment against that claim.
    def test_every_git_variable_the_product_passes_is_one_the_harness_assessed(self) -> None:
        allowlist = one_definition(_source("src/tools/proc_group.rs"),
                                   r"pub\(crate\) const TOOL_ENV_ALLOWLIST: &\[&str\] = &\[(.*?)\];",
                                   "src/tools/proc_group.rs", re.S)
        spawn = one_definition(_source("src/system/review_git.rs"),
                               r"\.env_clear\(\)\s*\.envs\(tool_child_env\(\)\)((?:\s*\.env\([^)]*\))*);",
                               "src/system/review_git.rs")
        inherited = {name for name in re.findall(r'"([^"]*)"', without_comments(allowlist.group(1)))
                     if name.upper().startswith(runs.GIT_ENV_PREFIX)}
        fixed = set(re.findall(r'\.env\("([^"]*)"', without_comments(spawn.group(1))))
        self.assertTrue(fixed, "the git spawn sets no variable of its own: the pattern lost its subject")
        self.assertEqual(set(runs.PRODUCT_GIT_VARIABLES), inherited | fixed)


class SecretPrefixDocTests(unittest.TestCase):
    """The README never names a secret prefix the detector does not check."""

    # MAGI FIX (S6 loop 5): the README's pattern list could drift past
    # src/review/secrets.rs and promise a protection that is not there (F2-b1).
    def test_every_prefix_the_readme_names_is_a_detector_prefix(self) -> None:
        readme = _source("README.md")
        sentence = re.search(r"The patterns are named prefixes and shapes[.:](.*?)Not covered, by design", readme, re.S)
        self.assertIsNotNone(sentence, "the README's secret pattern list was not found")
        # [MAGI §6 MS2 loop 28, F1b2a-melchior] the filter took only lowercase segments after a separator, so
        # a prefix such as GITHUB_PAT_ or one with a digit was never checked. Any letters-and-digits segments
        # ending in a separator count, compared ignoring case: the detector matches prefixes in any case.
        named = {token for token in re.findall(r"`([^`\s]+)`", sentence.group(1))
                 if re.fullmatch(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*[-_]|AKIA|ASIA|AIza", token)}
        self.assertTrue(named)
        self.assertIn("GITHUB_PAT_", named)
        prefixes = set()
        for array in re.findall(r"prefixes: &\[([^\]]*)\]", without_comments(_source("src/review/secrets.rs"))):
            prefixes.update(re.findall(r'"([^"]+)"', array))
        self.assertEqual(set(), {token.lower() for token in named} - {prefix.lower() for prefix in prefixes})


def _locked_version(crate: str) -> str:
    """The version of *crate* that ``Cargo.lock`` resolves.

    Args:
        crate: The package name.

    Returns:
        str: Its locked version.

    Raises:
        AssertionError: When the lockfile has no such package.
    """
    match = re.search(r'\[\[package\]\]\nname = "%s"\nversion = "([^"]+)"' % re.escape(crate),
                      _source("Cargo.lock"))
    if match is None:
        raise AssertionError("%s not found in Cargo.lock" % crate)
    return match.group(1)


def _registry_source(crate: str, relative: str) -> str:
    """Read *relative* from the locked version of *crate* in cargo's registry.

    It assumes cargo's default layout, ``$CARGO_HOME/registry/src/<index>/<crate>-<version>``,
    which a build fills. A vendored or path-overridden dependency is not looked for there, and
    then the guard fails with the message below instead of skipping: a skip would leave the
    probe paths unguarded without a word.

    Args:
        crate: The package name.
        relative: The path inside the package, with forward slashes.

    Returns:
        str: The file's text, ``\\r`` stripped.

    Raises:
        AssertionError: When no registry checkout of that version holds the file.
    """
    home = pathlib.Path(os.environ.get("CARGO_HOME") or pathlib.Path.home() / ".cargo")
    package = "%s-%s" % (crate, _locked_version(crate))
    for index in sorted((home / "registry" / "src").glob("*")):
        candidate = index / package / relative
        if candidate.is_file():
            return candidate.read_bytes().decode("utf-8", errors="replace").replace("\r", "")
    raise AssertionError("%s/%s not found under %s/registry/src; build the product first. A vendored or "
                         "path-overridden magi-core is not looked for: point CARGO_HOME at a cargo home whose "
                         "registry holds this version" % (package, relative, home))


#: How magi-core's ``OllamaProvider`` names a request's path: the segments after the
#: method, joined under the daemon root (``/v1`` is normalised away before this).
_REQUEST_SEGMENTS = re.compile(
    r"\.request\(\s*&self\.client,\s*reqwest::Method::\w+,\s*&\[([^\]]*)\]")


class ProbePathTests(unittest.TestCase):
    """The paths the model counter lets through as the capability probe (``runs.PROBE_PATHS``)."""

    # MAGI FIX (smoke review, Caspar+Melchior): PROBE_PATHS was a hand copy of the probe's
    # requests. The product's probe (src/magi/probe.rs) builds magi-core's OllamaProvider and
    # calls only window() and digest(), whose paths live in magi-core, pinned at Cargo.lock's version.

    def test_the_product_probes_through_the_ollama_provider_of_magi_core(self) -> None:
        probe = _source("src/magi/probe.rs")
        one_definition(probe, re.escape("impl ProbeFactory for OllamaProbeFactory"), "src/magi/probe.rs")
        one_occurrence(probe, "OllamaProvider::new(base_url.as_str(), model)", "src/magi/probe.rs")

    def test_the_probe_paths_match_what_magi_core_requests(self) -> None:
        source = _registry_source("magi-core", "src/providers/ollama.rs")
        start = source.find("impl ProviderProbe for OllamaProvider")
        self.assertNotEqual(-1, start, "the ProviderProbe impl was not found")
        end = source.find("\nimpl ", start + 1)
        block = source[start:] if end == -1 else source[start:end]
        groups = _REQUEST_SEGMENTS.findall(without_comments(block))
        self.assertEqual(2, len(groups), "expected the window and digest requests")
        paths = {"/" + "/".join(s.strip().strip('"') for s in group.split(",")) for group in groups}
        self.assertEqual(set(runs.PROBE_PATHS), paths)


def _rust_fn_body(text: str, signature: str) -> str:
    """The body of the method whose declaration starts with *signature*, up to its closing brace.

    Args:
        text: A Rust source file's text, ``\\r`` stripped.
        signature: The declaration's start, e.g. ``pub fn effective_provider(``.

    Returns:
        str: From the line the signature starts to the first line holding only a closing brace
        (a trailing comment allowed) at the signature's own indentation. rustfmt indents every
        inner block deeper, so that brace is the function's; a top-level function ends at a ``}`` in column one, a method
        at one indented like its ``fn``.

    Raises:
        AssertionError: When no code line starts the declaration, or more than one does, or no
            closing brace at its indentation ends it.
    """
    # MAGI FIX (S6 loop 27): the end was the first "    }" line whatever the
    # signature's indentation, so a top-level function ran on into the next
    # item's method (F1b2a-balthasar).
    match = one_definition(text, re.escape(signature), signature)
    start = match.start()
    indent = match.group(0)[:len(match.group(0)) - len(match.group(0).lstrip(" \t"))]
    # MAGI FIX (F1b2a): searched on the raw text, a "}" line inside a
    # multi-line literal or block comment ended the body early; the view
    # blanks both at the same offsets, and the original is sliced.
    # MAGI FIX (S6 loop 39, F1b2a-melchior): a "} // note" closing line was
    # missed; the view blanks the comment, so trailing blanks are allowed.
    end = re.compile(r"\n%s\}[ \t]*\n" % re.escape(indent)).search(structure_only(text), start)
    # MAGI FIX (S6 loop 28): no closing brace returned the rest of the file,
    # where a pin could match a later item's line (F1b2a-melchior).
    if end is None:
        raise AssertionError("%s: no closing brace at its own indentation" % signature)
    return text[start:end.start()]


class RustFnBodyTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 27, F1b2a-balthasar] ``_rust_fn_body`` ends at its own closing brace."""

    _TEXT = ("pub fn top() -> u8 {\n    0\n}\n\nimpl S {\n    pub fn method(&self) -> u8 {\n"
             "        let x = 1;\n        x\n    }\n}\n")

    def test_a_top_level_function_ends_at_its_unindented_brace(self) -> None:
        body = _rust_fn_body(self._TEXT, "pub fn top(")
        self.assertEqual("pub fn top() -> u8 {\n    0", body)

    def test_a_method_ends_at_the_brace_on_its_own_indentation(self) -> None:
        body = _rust_fn_body(self._TEXT, "pub fn method(")
        self.assertEqual("    pub fn method(&self) -> u8 {\n        let x = 1;\n        x", body)

    def test_a_body_without_its_closing_brace_fails_loudly(self) -> None:
        # [MAGI §6 MS2 loop 28, F1b2a-melchior] it returned the rest of the file instead.
        with self.assertRaisesRegex(AssertionError, "no closing brace"):
            _rust_fn_body("pub fn top() -> u8 {\n    0\n}", "pub fn top(")

    def test_a_brace_line_inside_a_literal_or_comment_does_not_end_the_body(self) -> None:
        # [F1b2a] the end was found on the raw text, so a ``}`` line inside a multi-line raw string or block
        # comment, at the function's indentation, cut the body short of the pinned lines after it.
        text = ('pub fn top() -> &\'static str {\n    let s = r#"\n}\n"#;\n    /*\n}\n    */\n'
                "    s\n}\n\npub fn next() {}\n")
        body = _rust_fn_body(text, "pub fn top(")
        self.assertEqual(text[:text.index("    s\n") + len("    s")], body)

    def test_a_closing_brace_with_a_trailing_comment_ends_the_body(self) -> None:
        # [MAGI §6 MS2 loop 39, F1b2a-melchior] the end was searched as a ``}`` alone on its line, so a
        # ``} // note`` at the function's indentation was missed: a later item's brace ended the body, or none.
        text = "pub fn top() -> u8 {\n    0\n} // end of top\n\npub fn next() -> u8 {\n    1\n}\n"
        self.assertEqual("pub fn top() -> u8 {\n    0", _rust_fn_body(text, "pub fn top("))
        method = "impl S {\n    pub fn m(&self) -> u8 {\n        0\n    } /* m */ \n}\n"
        self.assertEqual("    pub fn m(&self) -> u8 {\n        0", _rust_fn_body(method, "pub fn m("))


class CountedKindTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 25, F1a1-melchior] a refusal run's absent provider kind is the counted one.

    ``endpoints_off_the_counter`` accepts a configuration with no ``provider`` and no ``[magi].kind``,
    which is sound only while the product resolves both, when absent, to ``runs._COUNTED_KIND``: a
    different default would let a refusal run reach a fixed-URL provider the counter never sees.
    """

    def _counted_variant(self) -> str:
        """The ``ProviderKind`` variant ``src/magi/kind.rs`` parses ``runs._COUNTED_KIND`` into."""
        kind = _source("src/magi/kind.rs")
        match = one_definition(kind, r'"%s" => Ok\(Some\(Self::(\w+)\)\)' % re.escape(runs._COUNTED_KIND),
                               "src/magi/kind.rs")
        return match.group(1)

    def test_an_absent_provider_resolves_to_the_counted_kind(self) -> None:
        body = _rust_fn_body(_source("src/config.rs"), "pub fn effective_provider(")
        one_definition(body, re.escape(".unwrap_or(ProviderKind::%s)" % self._counted_variant()), "effective_provider")

    def test_an_absent_magi_kind_inherits_the_provider(self) -> None:
        body = _rust_fn_body(_source("src/config.rs"), "pub fn effective_magi_kind(")
        one_definition(body, re.escape(".unwrap_or_else(|| self.effective_provider())"), "effective_magi_kind")

    def test_the_scaffold_declares_the_counted_kind(self) -> None:
        defaults = _source("src/defaults.rs")
        one_definition(defaults, re.escape('pub const DEFAULT_PROVIDER: &str = "%s";' % runs._COUNTED_KIND),
                       "src/defaults.rs")
        one_definition(defaults, re.escape('writeln!(out, "provider = \\"{}\\"", DEFAULT_PROVIDER)'),
                       "src/defaults.rs")


if __name__ == "__main__":
    unittest.main()
