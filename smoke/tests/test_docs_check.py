# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the published-documentation extractor.

The extractor is pure parsing: text in, invocations and configuration bodies
out. It never asks the product anything, which is what lets these tests run in
milliseconds and what keeps the judgement -- does this subcommand exist? -- in
the scenario, where the product can answer it.
"""

import pathlib
import unittest

from smoke.docs_check import Invocation, extract_configs, extract_invocations, published_docs


class ExtractInvocationTests(unittest.TestCase):
    """What counts as an invocation, and what deliberately does not."""

    def test_a_fenced_block_yields_one_invocation_with_its_flags(self) -> None:
        text = "\n".join([
            "prose",
            "```bash",
            "magi-rs query --output-format json --timeout 30",
            "```",
        ])
        found = extract_invocations(text)
        self.assertEqual(1, len(found))
        self.assertEqual("query", found[0].subcommand)
        self.assertEqual(("--output-format", "--timeout"), found[0].flags)
        self.assertEqual(3, found[0].line)
        self.assertEqual("magi-rs query --output-format json --timeout 30",
                         found[0].source)

    def test_several_lines_in_one_block_yield_several_invocations(self) -> None:
        text = "\n".join([
            "```bash",
            "magi-rs vault ls",
            "magi-rs vault rm OPENAI_API_KEY",
            "magi-rs init",
            "```",
        ])
        found = extract_invocations(text)
        self.assertEqual(["vault", "vault", "init"],
                         [item.subcommand for item in found])

    def test_prose_naming_the_product_outside_a_fence_is_not_extracted(self) -> None:
        """Naming the binary in a sentence is not invoking it.

        This is the assertion that stops the verifier punishing the very
        documentation it exists to encourage: a paragraph explaining that
        ``magi-rs --init-config`` was retired must not be read as a use of the
        retired flag.
        """
        text = "Run magi-rs --init-config to scaffold, as older guides said.\n"
        self.assertEqual([], extract_invocations(text))

    def test_a_flag_written_with_an_equals_sign_is_normalised(self) -> None:
        text = "```bash\nmagi-rs query --output-format=json\n```"
        self.assertEqual(("--output-format",), extract_invocations(text)[0].flags)

    def test_a_comment_line_inside_a_fence_is_not_an_invocation(self) -> None:
        """A fenced block is not all commands.

        ``# 3. Run magi-rs -- the startup banner reports the provider`` is a
        comment inside a shell block in the README, and reading it as an
        invocation would extract a subcommand of ``--`` that exists nowhere.
        """
        text = "```bash\n# 3. Run magi-rs and watch the banner\n```"
        self.assertEqual([], extract_invocations(text))

    def test_the_product_named_as_an_argument_is_not_an_invocation(self) -> None:
        """``cargo install magi-rs`` invokes cargo, not the product.

        The program is the first word of a command, so a package name in an
        argument position never becomes one. Without this the extractor would
        read ``--version 0.11.0 --root ... --locked`` as flags of a magi-rs
        invocation and report all four as missing.
        """
        text = "```bash\ncargo install magi-rs --version 0.11.0 --locked\n```"
        self.assertEqual([], extract_invocations(text))

    def test_a_pipeline_segment_is_an_invocation(self) -> None:
        """The published guides pipe a prompt in, so the program is not the
        first word of the LINE -- only of its segment."""
        text = '```bash\n"say ok" | magi-rs query --output-format json\n```'
        found = extract_invocations(text)
        self.assertEqual(1, len(found))
        self.assertEqual("query", found[0].subcommand)

    def test_a_leading_environment_assignment_is_stepped_over(self) -> None:
        text = '```bash\nMAGI_PASSPHRASE="x" magi-rs vault ls\n```'
        self.assertEqual("vault", extract_invocations(text)[0].subcommand)

    def test_every_leading_word_is_kept_not_only_the_first(self) -> None:
        """``vault set`` is two levels deep and ``-f`` belongs to the second.

        A verifier that kept only ``vault`` would look ``-f`` up in the wrong
        help text and report a flag the product has as one it lacks.
        """
        text = "```bash\nmagi-rs vault set OPENAI_API_KEY -f\n```"
        found = extract_invocations(text)[0]
        self.assertEqual(("vault", "set", "OPENAI_API_KEY"), found.words)
        self.assertEqual("vault", found.subcommand)

    def test_an_invocation_with_no_subcommand_reports_an_empty_one(self) -> None:
        """``magi-rs --version`` names no subcommand and is still legitimate,
        so the empty string is the answer rather than an error."""
        found = extract_invocations("```bash\nmagi-rs --version\n```")[0]
        self.assertEqual("", found.subcommand)
        self.assertEqual(("--version",), found.flags)

    def test_an_unbalanced_quote_degrades_instead_of_raising(self) -> None:
        """Documentation is prose with code in it, not a shell script.

        A block holding an unterminated quote is a real thing to write, and a
        tokenizer error there must not take the whole scenario down -- the
        harness would report a product defect where a document merely has an
        odd line.
        """
        text = '```bash\nmagi-rs query --timeout "30\n```'
        found = extract_invocations(text)
        self.assertEqual(1, len(found))
        self.assertEqual("query", found[0].subcommand)


class ExtractConfigTests(unittest.TestCase):
    """Which fenced bodies are candidate configurations."""

    def test_a_toml_fence_is_returned(self) -> None:
        text = "\n".join([
            "prose",
            "```toml",
            "[memory]",
            'mode = "selective"',
            "```",
        ])
        self.assertEqual(['[memory]\nmode = "selective"'], extract_configs(text))

    def test_a_fence_of_another_language_is_not_a_configuration(self) -> None:
        text = "```bash\nmagi-rs init\n```\n```sql\nDROP TABLE memories;\n```"
        self.assertEqual([], extract_configs(text))

    def test_a_document_with_no_fences_yields_nothing(self) -> None:
        self.assertEqual([], extract_configs("plain prose about magi.toml\n"))


class PublishedDocsTests(unittest.TestCase):
    """The list is derived from what cargo packages, never hardcoded."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parent.parent.parent

    def test_the_guides_that_ship_are_listed(self) -> None:
        listed = {path.as_posix() for path in published_docs(self.repo_root)}
        for expected in ("README.md", "docs/OVERVIEW.md", "docs/E2E-TESTING.md",
                         "docs/TIERED-MEMORY.md"):
            self.assertIn(expected, listed)

    def test_the_test_fixtures_are_not_documentation(self) -> None:
        """``cargo package --list`` includes the crate's own test tree.

        ``tests/fixtures/v0.11.0/README.md`` is a provenance record of the
        published v0.11.0 binary, and its invocations are true statements about
        THAT release. Checking it against today's surface would report defects
        whose only correct fix is to falsify the record.
        """
        listed = {path.as_posix() for path in published_docs(self.repo_root)}
        self.assertNotIn("tests/fixtures/v0.11.0/README.md", listed)

    def test_nothing_outside_the_package_is_listed(self) -> None:
        """``smoke/`` is excluded from the package, so this harness's own
        markdown never becomes something the harness checks."""
        for path in published_docs(self.repo_root):
            self.assertFalse(path.as_posix().startswith("smoke/"), path)


class InvocationShapeTests(unittest.TestCase):
    """The type carries no second source of truth."""

    def test_the_subcommand_is_derived_from_the_words(self) -> None:
        """It is a property, not a field, so it cannot disagree with ``words``.

        Two fields holding the same fact drift the first time somebody edits
        one of them, and nothing in Python would say so.
        """
        invocation = Invocation(words=("vault", "set"), flags=("-f",),
                                source="magi-rs vault set -f", line=1)
        self.assertEqual("vault", invocation.subcommand)


if __name__ == "__main__":
    unittest.main()
