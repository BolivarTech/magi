# Claude CLI output fixtures

These fixtures pin the raw output shapes that `parse_agent_output.py`
must keep accepting. They exist because nothing in the unit suite
actually spawns `claude -p` (it needs the CLI and a paid API key), so
without a pinned fixture set a silent contract drift in the upstream
CLI would only surface as a parse failure in production.

Each file is a sample of what a real `claude -p --output-format json`
invocation can look like. The `test_claude_cli_fixture_contract.py`
suite parameterizes over these files and asserts that each one round-
trips through `parse_agent_output` to a valid agent JSON output.

## Adding a fixture

When Anthropic ships a CLI change that alters the wrapper shape, capture
the new raw output into a `.json` file in this directory. The file must:

1. Carry a short, descriptive filename (kebab-case).
2. Contain either: a dict with a `result` key, a dict with a `content`
   list of text blocks, or a plain JSON-encoded string. These are the
   three shapes documented in `parse_agent_output._extract_text`.
3. Wrap a valid agent payload (all required schema keys present).
4. Be committed with a one-line note in the PR description explaining
   which CLI version produced it.

The suite auto-discovers every `.json` file in this directory, so no
test edit is required when adding a new fixture.
