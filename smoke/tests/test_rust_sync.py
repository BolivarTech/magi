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


def _literal_key(node) -> str | None:
    """The string *node* holds when it is a string literal, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _SectionReads:
    """Every use of the name ``section`` in a module, each one either read for its key or refused.

    MAGI FIX (F1b2a): the keys were found with regular expressions, so a legal
    form they did not anticipate (``section . get("k")``, an alias, a helper
    that received ``section``) read a key the guard never saw. The module is
    parsed instead, and every ``Name`` node called ``section`` must be one of
    the forms below; any other use fails loudly.

    Accepted: a binding (an assignment target or a parameter);
    ``section.get("<lit>")`` with exactly one argument;
    ``section.get(key)`` directly inside the module-level ``_list_under``,
    whose ``key`` nothing rebinds; ``_list_under(section,
    "<lit>")``; ``isinstance(section, ...)``; and ``section`` passed
    positionally to a function defined in the module whose parameter at that
    position is also named ``section``, whose own body the same walk reads.

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
        self._parents = {}
        self._parameters = {node.name: [arg.arg for arg in node.args.posonlyargs + node.args.args]
                            for node in self._tree.body if isinstance(node, ast.FunctionDef)}
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
        keys = set()
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Name) and node.id == _SECTION_NAME:
                if isinstance(node.ctx, ast.Store):
                    continue
                key = self._key_of(node)
                if key is not None:
                    keys.add(key)
        return keys

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
            if callee == _LIST_UNDER and position == 0 and len(parent.args) == 2 and not parent.keywords:
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
            lambda, comprehension or class) is a module-level
            ``_list_under`` that takes ``key`` positionally and never stores
            or deletes it anywhere in its body.
        """
        scope = self._parents.get(node)
        while scope is not None and not isinstance(scope, _SCOPE_NODES):
            scope = self._parents.get(scope)
        if not (isinstance(scope, ast.FunctionDef) and scope.name == _LIST_UNDER
                and self._parents.get(scope) is self._tree):
            return False
        if _LIST_UNDER_PARAMETER not in [arg.arg for arg in scope.args.posonlyargs + scope.args.args]:
            return False
        return not any(isinstance(inner, ast.Name) and inner.id == _LIST_UNDER_PARAMETER
                       and isinstance(inner.ctx, (ast.Store, ast.Del)) for inner in ast.walk(scope))


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
            unpacked or iterated, or passed to a function the walk cannot read.
    """
    return _SectionReads(scenario).keys()


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
        self.assertGreaterEqual(round(runs.BATCH_WINDOW_FRACTION * 100), percent)

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
        one_occurrence(self.error, '"path is outside the workspace: {}"', "src/review/error.rs")
        self.assertIn(code_review.OUTSIDE_WORKSPACE_TEXT, "path is outside the workspace")
        one_occurrence(self.error, '"nothing to review: no reviewable file in the input{}"', "src/review/error.rs")
        self.assertIn(code_review.NOTHING_TO_REVIEW_TEXT, "nothing to review: no reviewable file")

    def test_a_trivial_copy_of_a_fixed_phrase_fails_the_check(self) -> None:
        # [MAGI F1b2a-e] assertIn(copy, product_text) held for an empty copy, and S48/S49/S51 then matched
        # any stderr at all: the copy must be a phrase of words, cut at word boundaries of the product's text.
        for name in ("OUTSIDE_WORKSPACE_TEXT", "NOTHING_TO_REVIEW_TEXT"):
            for trivial in ("", " ", "e", "the", "side the work", "nothing to rev"):
                with mock.patch.object(code_review, name, trivial):
                    with self.assertRaises(AssertionError, msg=(name, trivial)):
                        self.test_the_fixed_phrases_are_the_products()


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
        trimmed = one_occurrence(body, ".trim_end_matches([", "is_never_reviewed")
        rest = body[trimmed.end():]
        chars = char_list(rest[:rest.index("]")])
        self.assertEqual(sorted(chars), sorted(runs.TRIMMED_TRAILING))
        split = one_occurrence(body, ".split([", "is_never_reviewed")
        rest = body[split.end():]
        self.assertEqual({"/", "\\"}, set(char_list(rest[:rest.index("]")])))
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


class ReportKeyTests(unittest.TestCase):
    """The JSON keys S47 reads exist in the product's report and envelope."""

    # MAGI FIX (S6 loop 6): S47's keys sat outside the drift guard, so a rename
    # would surface as an unexplained FAIL at SMOKE time (F1a-m2).
    def test_every_summary_key_s47_reads_is_one_the_report_pins(self) -> None:
        read = _summary_keys_read(_source("smoke/scenarios/code_review.py"))
        # [MAGI §6 MS2 loop 27, F1b2a-melchior] the keys were only inferred from two access shapes, so an access
        # rewritten in another shape dropped its key from the guard in silence. The set is written out: a key
        # the extraction stops seeing, or a new one, fails here and is reviewed on purpose.
        self.assertEqual({"batches", "batches_completed", "batches_planned", "discarded", "failed_batches",
                          "findings", "unmeasured_models"}, read)
        report = _source("src/review/report.rs")
        # The list is found by the comment that names its test, the one place a comment is the anchor;
        # the comment must be unique and the keys are read from code lines only.
        self.assertEqual(1, report.count("every documented key is present"))
        pinned = re.search(r"every documented key is present.*?for k in \[(.*?)\]", report, re.S)
        self.assertIsNotNone(pinned, "the summary key list was not found in src/review/report.rs")
        self.assertEqual(set(), read - set(re.findall(r'"(\w+)"', without_comments(pinned.group(1)))))

    def test_the_entry_fields_s47_reads_are_serialized_under_their_own_names(self) -> None:
        # [MAGI §6 MS2 loop 7, F1a-m1] S47 reads file and line off each main finding and cause off each
        # discarded one; a serde rename on either struct would surface as an opaque FAIL at SMOKE time.
        for path, struct, fields in (("src/review/consolidate.rs", "ConsolidatedFinding",
                                      ("pub file: Option<String>,", "pub line: Option<u32>,")),
                                     ("src/review/validate.rs", "Discarded", ("pub cause: DiscardCause,",))):
            body = one_definition(_source(path),
                                  r"#\[derive\([^)]*serde::Serialize[^)]*\)\]\npub struct %s \{(.*?)\n\}" % struct,
                                  path, re.S).group(1)
            self.assertNotIn("rename", body, struct)
            for field in fields:
                one_definition(body, re.escape(field), "%s in %s" % (struct, path))

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
        at the signature's own indentation. rustfmt indents every inner block deeper, so that
        brace is the function's; a top-level function ends at a ``}`` in column one, a method
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
    end = structure_only(text).find("\n%s}\n" % indent, start)
    # MAGI FIX (S6 loop 28): no closing brace returned the rest of the file,
    # where a pin could match a later item's line (F1b2a-melchior).
    if end == -1:
        raise AssertionError("%s: no closing brace at its own indentation" % signature)
    return text[start:end]


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
