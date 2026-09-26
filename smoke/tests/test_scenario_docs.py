# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S13 scenario's own shape."""

import pathlib
import unittest
from unittest import mock

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import docs  # noqa: F401 - import registers it
from smoke.tests import support

#: A help text in clap's long form: the option spec on its own line, the
#: description indented beneath it. The description deliberately NAMES a flag
#: the command does not have, which is the trap the parser has to avoid.
_LONG_HELP = "\n".join([
    "Run the agent headless over a prompt",
    "",
    "Usage: magi-rs query [OPTIONS]",
    "",
    "Options:",
    "  -i, --input <INPUT>",
    "          Read the prompt from a file",
    "",
    "      --output-format <OUTPUT_FORMAT>",
    "          Output format; omitted means text.",
    "",
    "          The retired --init-config flag used to write one of these, and",
    "          a --release build behaves the same way.",
    "",
    "  -h, --help",
    "          Print help",
])

#: A help text in clap's short form: description on the same line as the spec.
_SHORT_HELP = "\n".join([
    "Usage: magi-rs [OPTIONS] [COMMAND]",
    "",
    "Commands:",
    "  vault    Encrypted secret store",
    "  init     Scaffold a fresh .magi/",
    "  query    Run the agent headless",
    "  help     Print this message",
    "",
    "Options:",
    "  -l, --logout                   Log out",
    "  -p, --passphrase <PASSPHRASE>  Master passphrase",
    "  -h, --help                     Print help",
    "  -V, --version                  Print version",
])


class HelpParsingTests(unittest.TestCase):
    """What the product says about itself, read without embellishment."""

    def test_only_the_leading_option_spec_is_a_flag(self) -> None:
        """A flag NAMED in a description is not a flag the command has.

        This is the dangerous direction: over-collecting makes the check accept
        a flag the product does not carry, so a retired flag mentioned in some
        unrelated help text would silently excuse a document that still uses
        it. ``--init-config`` and ``--release`` appear in the fixture's prose
        and must not be in the answer.
        """
        found = docs._flags_in(_LONG_HELP)
        self.assertEqual({"-i", "--input", "--output-format", "-h", "--help"},
                         found)

    def test_the_short_form_is_parsed_too(self) -> None:
        """clap writes the root help with descriptions on the same line."""
        self.assertEqual({"-l", "--logout", "-p", "--passphrase", "-h",
                          "--help", "-V", "--version"},
                         docs._flags_in(_SHORT_HELP))

    def test_the_commands_section_yields_the_subcommands(self) -> None:
        self.assertEqual({"vault", "init", "query", "help"},
                         docs._subcommands_in(_SHORT_HELP))

    def test_a_help_text_with_no_commands_has_no_subcommands(self) -> None:
        self.assertEqual(set(), docs._subcommands_in(_LONG_HELP))

    def test_a_multi_paragraph_refusal_is_reduced_to_its_diagnosis(self) -> None:
        stream = b"\nerror:  the file   is not compatible\n\nAdd this key:\n"
        self.assertEqual("error: the file is not compatible",
                         docs._first_line(stream))


class SurfaceWalkTests(unittest.TestCase):
    """How deep a subcommand path goes, decided by the product."""

    def setUp(self) -> None:
        self.binary = support.install_fake_runs(self, _help_responder)

    def test_the_walk_stops_at_a_positional_argument(self) -> None:
        """``vault set ANTHROPIC_API_KEY`` names two subcommands and a secret.

        Walking one word further would ask the product for the help of a
        secret's name, which it refuses -- and the scenario would then report
        that it could not check a line that is perfectly fine.
        """
        surface = docs._Surface()
        self.assertEqual(("vault", "set"),
                         surface.resolve(("vault", "set", "ANTHROPIC_API_KEY")))

    def test_a_word_the_product_does_not_list_ends_the_walk(self) -> None:
        surface = docs._Surface()
        self.assertEqual((), surface.resolve(("nonesuch", "set")))

    def test_the_help_of_one_path_is_asked_for_once(self) -> None:
        """A guide naming ``vault set`` twenty times costs one invocation."""
        surface = docs._Surface()
        surface.help_for(("vault", "set"))
        surface.help_for(("vault", "set"))
        asked = [call.args for call in self.binary.calls]
        self.assertEqual(1, asked.count(("vault", "set", "--help")), asked)

    def test_a_path_the_product_refuses_is_remembered_as_refused(self) -> None:
        """A failed answer is cached too, or a document naming a subcommand
        that does not exist pays one invocation per mention."""
        surface = docs._Surface()
        self.assertIsNone(surface.help_for(("nonesuch",)))
        self.assertIsNone(surface.help_for(("nonesuch",)))
        asked = [call.args for call in self.binary.calls]
        self.assertEqual(1, asked.count(("nonesuch", "--help")), asked)


class ScenarioShapeTests(unittest.TestCase):
    """Four assertions are reported, whatever the environment allows."""

    def test_every_assertion_is_reported_against_a_product_that_answers_nothing(self):
        """The default double fails every invocation.

        A scenario that returned early would drop assertions from the report
        rather than saying they could not be evaluated, and the reconciliation
        cannot see the difference: it only knows the scenario spoke at all.
        """
        support.install_fake_runs(self)
        findings = list(docs.the_published_documentation_is_still_true(None))
        self.assertEqual(list(docs.S13_ASSERTIONS),
                         [finding.assertion for finding in findings])
        for finding in findings:
            self.assertNotEqual(Outcome.PASS, finding.outcome, finding)

    def test_it_is_registered_standalone_and_without_the_backend(self) -> None:
        entry = DEFAULT_REGISTRY.get("S13")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)
        self.assertFalse(entry.needs_ambient)


def _help_responder(call: support.Call) -> ProductOutput | None:
    """Answer ``--help`` for the paths the fake product admits to having.

    Args:
        call: What the fake binary was asked to run.

    Returns:
        ProductOutput: The canned help, or None so the caller's failed default
        stands in for a path the product does not have.
    """
    args = list(call.args)
    if args[-1:] != ["--help"]:
        return None
    path = tuple(args[:-1])
    text = {
        (): _SHORT_HELP,
        ("vault",): "Usage: magi-rs vault\n\nCommands:\n  set    Add a secret\n",
        ("vault", "set"): _LONG_HELP,
    }.get(path)
    if text is None:
        return None
    return ProductOutput(stdout=text.encode("utf-8"), stderr=b"", exit_code=0,
                         command=["magi-rs"] + args)



_S13_SCAFFOLD = ('provider = "ollama"\n[magi]\nmelchior_model = "a"\n'
                 '# reasoning = "default"\n# max_tokens = 16384\n'
                 "[[magi.fallback]]\nmodel = \"x\"\nlineage = \"y\"\n")


def _docs_responder(call: support.Call):
    """Answer ``--help`` like the help responder, and ``init`` by scaffolding.

    Args:
        call: What the fake binary was asked to run.

    Returns:
        ProductOutput | None: The canned answer.
    """
    if call.args[:1] == ("init",) and call.cwd:
        root = pathlib.Path(call.cwd)
        (root / ".magi").mkdir(parents=True, exist_ok=True)
        (root / ".magi" / "magi.toml").write_text(_S13_SCAFFOLD,
                                                  encoding="utf-8")
        return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                             command=["magi-rs", "init"])
    return _help_responder(call)


class S13KeyCoverageTests(unittest.TestCase):
    """Every [magi] key the scaffold writes is named in a published guide."""

    def _outcome_of(self, files: dict) -> Outcome:
        """The fourth assertion's outcome for a repository holding *files*.

        Named ``_outcome_of`` rather than ``_outcome``: ``unittest.TestCase``
        sets an instance attribute of that exact name before every test
        method runs, and a same-named method is shadowed by it -- see the
        identical note on ``S11ReasoningVocabularyTests._outcome_of``.

        Args:
            files: Relative path to file contents, seeded into a scratch repo.

        Returns:
            Outcome: What the fourth assertion concluded.
        """
        repo = support.scratch_dir(self)
        for name, body in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(body, encoding="utf-8")
        support.install_fake_runs(self, _docs_responder, repo_root=repo)
        with mock.patch.object(docs, "published_docs",
                               return_value=[pathlib.Path(n) for n in files]):
            findings = list(docs.the_published_documentation_is_still_true(
                None))
        return {f.assertion: f.outcome for f in findings}[docs.S13_ASSERTIONS[3]]

    def test_the_fourth_text_is_the_declared_one(self) -> None:
        self.assertEqual("every [magi] key the scaffold writes is named in a "
                         "published guide", docs.S13_ASSERTIONS[3])

    def test_every_key_named_passes(self) -> None:
        self.assertEqual(Outcome.PASS, self._outcome_of({
            "README.md": "`melchior_model`, `reasoning` and `max_tokens`."}))

    def test_a_commented_key_nobody_names_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._outcome_of({
            "README.md": "`melchior_model` and `reasoning`."}))

    def test_a_key_named_only_in_the_changelog_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._outcome_of({
            "README.md": "`melchior_model` and `reasoning`.",
            "CHANGELOG.md": "Added `max_tokens`."}))

    def test_a_key_named_in_a_guide_under_docs_passes(self) -> None:
        self.assertEqual(Outcome.PASS, self._outcome_of({
            "README.md": "`melchior_model` and `reasoning`.",
            "docs/REASONING-BUDGET.md": "`max_tokens` has no upper bound."}))

    def test_a_prefix_does_not_stand_for_the_key(self) -> None:
        self.assertEqual(Outcome.FAIL, self._outcome_of({
            "README.md": "`melchior_model`, `reasoning_trace`, `max_tokens`."}))

    def test_the_keys_are_read_from_the_scaffold_not_a_documented_config(
            self) -> None:
        """Assertion 3 installs every documented ``magi.toml`` over the seed.

        Assertion 4 must read the table ``init`` wrote, not whichever documented
        configuration assertion 3 left behind: a guide whose last example has
        no ``[magi]`` table would otherwise report that the scaffold has no
        keys at all.
        """
        self.assertEqual(Outcome.PASS, self._outcome_of({
            "README.md": "`melchior_model`, `reasoning` and `max_tokens`.\n\n"
                         "```toml\nprovider = \"ollama\"\n```\n"}))

    def test_a_scaffold_that_cannot_be_seeded_cannot_test_it(self) -> None:
        repo = support.scratch_dir(self)
        (repo / "README.md").write_text("x", encoding="utf-8")
        support.install_fake_runs(self, _help_responder, repo_root=repo)
        with mock.patch.object(docs, "published_docs",
                               return_value=[pathlib.Path("README.md")]):
            findings = list(docs.the_published_documentation_is_still_true(
                None))
        outcome = {f.assertion: f.outcome
                   for f in findings}[docs.S13_ASSERTIONS[3]]
        self.assertEqual(Outcome.CANNOT_TEST, outcome)


class SeedWorkspaceCleanupTests(unittest.TestCase):
    """The directory S13 creates is removed on the path that FAILS too.

    The first cleanup lived in the caller's ``finally``, which covers the
    workspace that was handed back and leaks the one that was not -- and the
    failure path is the likelier of the two to run, because it is the one a
    broken product takes. One directory per failed run, under a scratch area
    an operator is not expected to sweep by hand.
    """

    def test_a_failed_init_leaves_no_directory_behind(self) -> None:
        scratch = support.scratch_dir(self)
        support.install_fake_runs(self)
        with mock.patch.object(docs.runs, "scratch_root",
                               return_value=scratch):
            self.assertIsNone(docs._seed_workspace())
        self.assertEqual([], sorted(scratch.iterdir()),
                         "the workspace it could not scaffold was left behind")

    def test_a_raise_part_way_through_leaves_no_directory_behind(self) -> None:
        scratch = support.scratch_dir(self)
        support.install_fake_runs(self)
        with mock.patch.object(docs.runs, "scratch_root",
                               return_value=scratch):
            with mock.patch.object(docs.runs, "attempt",
                                   side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    docs._seed_workspace()
        self.assertEqual([], sorted(scratch.iterdir()))


if __name__ == "__main__":
    unittest.main()
