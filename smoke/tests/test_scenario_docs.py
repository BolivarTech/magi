# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S13 scenario's own shape."""

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
    """Three assertions are reported, whatever the environment allows."""

    def test_all_three_are_reported_against_a_product_that_answers_nothing(self):
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
