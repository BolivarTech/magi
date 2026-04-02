# MAGI — Multi-Perspective Analysis Plugin for Claude Code

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)](#running-tests)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A Claude Code plugin that implements a **multi-perspective analysis system** inspired by the [MAGI supercomputers](https://evangelion.fandom.com/wiki/Magi) from *Neon Genesis Evangelion*.

Three specialized AI agents independently analyze the same problem from complementary — and deliberately adversarial — perspectives, then synthesize their verdicts via weight-based majority vote.

---

## Why Three Adversarial Perspectives?

### The MAGI in Evangelion

In *Neon Genesis Evangelion* (1995, Hideaki Anno / Gainax), the MAGI are three supercomputers that govern Tokyo-3's critical decisions. Each embodies a different facet of their creator, Dr. Naoko Akagi: **Melchior** (the scientist), **Balthasar** (the mother), and **Caspar** (the woman). Decisions require consensus — no single perspective dominates.

This design reflects a profound insight: **complex decisions benefit from structured disagreement**. A single decision-maker, no matter how capable, carries blind spots. Three independent evaluators with different priorities surface risks, trade-offs, and opportunities that any one of them would miss.

### The Theory in Practice

The adversarial multi-perspective model addresses well-documented cognitive biases in software engineering:

| Bias | How MAGI Mitigates It |
|------|----------------------|
| **Confirmation bias** | Three agents with different evaluation criteria are unlikely to share the same blind spots |
| **Anchoring** | Agents analyze independently — no agent sees the others' output before forming its own verdict |
| **Groupthink** | Caspar (Critic) is designed to be adversarial; its role is to find fault, not agree |
| **Optimism bias** | The weight-based scoring penalizes reject (-1) more heavily than approve (+1), making negative signals harder to override |
| **Status quo bias** | Each agent evaluates from first principles against its own criteria, not against "how things are done" |

The key insight is that **disagreement between agents is a feature, not a failure**. When Melchior (Scientist) approves but Caspar (Critic) rejects, the dissent surfaces a genuine tension between technical correctness and risk tolerance. Unanimous agreement on non-trivial input may indicate insufficiently differentiated prompts, not actual consensus.

In practice, the system works best for decisions with:
- **Genuine uncertainty** — multiple valid approaches exist
- **Significant consequences** — the cost of a wrong decision is high
- **Hidden trade-offs** — benefits and risks are not immediately obvious

For trivial questions with one clear answer, the complexity gate skips the full system and responds directly.

---

## Documentation

For the full technical reference, see [`docs/MAGI-System-Documentation.md`](docs/MAGI-System-Documentation.md).

---

## Agents

| Agent | Codename | Lens | Personality |
|-------|----------|------|-------------|
| **Melchior** | Scientist | Technical rigor and correctness | Precise, evidence-based, favors proven solutions |
| **Balthasar** | Pragmatist | Practicality and maintainability | Grounded, trade-off oriented, advocates for the team |
| **Caspar** | Critic | Risk, edge cases, and failure modes | Adversarial by design, finds what others miss |

---

## Installation

### From GitHub (for users)

```bash
# 1. Add this repo as a marketplace source
/plugin marketplace add BolivarTech/magi

# 2. Install the plugin
/plugin install magi@bolivartech-plugins

# 3. Use it
/magi
```

To update after new versions are published:

```bash
/plugin marketplace update
```

### Local Development

```bash
# Option 1: Plugin flag
claude --plugin-dir /path/to/magi

# Option 2: Symlink for auto-discovery (no flags needed)
mkdir -p .claude/skills
ln -s ../../skills/magi .claude/skills/magi
claude
```

Changes are picked up with `/reload-plugins` without restarting.

---

## Usage

Invoke with `/magi` or natural trigger phrases:

```
MAGI review this code
Give me three perspectives on this design
MAGI analysis of this problem
```

### Modes

| Mode | When to Use | Example |
|------|-------------|---------|
| `code-review` | Reviewing code or diffs | "MAGI review this PR" |
| `design` | Evaluating architecture decisions | "MAGI analyze this migration plan" |
| `analysis` | General problem analysis, trade-offs | "MAGI should we use Redis or Postgres for this?" |

### CLI (Direct Execution)

```bash
python skills/magi/scripts/run_magi.py <mode> <file_or_text> [--model opus] [--timeout 300] [--output-dir <dir>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `opus` | LLM model for all agents (`opus`, `sonnet`, `haiku`) |
| `--timeout` | `300` | Per-agent timeout in seconds |
| `--output-dir` | auto | Directory for agent outputs (default: temp dir) |

---

## How It Works

```
User input
  |
  v
SKILL.md (complexity gate + mode detection)
  |
  v
run_magi.py launches 3x claude -p (parallel, async)
  |               |               |
  v               v               v
Melchior        Balthasar       Caspar
(Scientist)     (Pragmatist)    (Critic)
  |               |               |
  v               v               v
parse_agent_output.py (extract JSON from CLI output)
  |               |               |
  v               v               v
validate.load_agent_output() (schema validation)
  |
  v
consensus.determine_consensus() (weight-based scoring)
  |
  v
reporting.format_report() (banner + markdown report)
```

### Step by Step

1. **Complexity gate** — Simple questions are answered directly without invoking three agents.
2. **Parallel dispatch** — Three sub-agents run concurrently via `asyncio` + `claude -p`, each with a distinct system prompt written to a temp file.
3. **Independent analysis** — Each agent evaluates the same input through its unique lens and produces a structured JSON verdict.
4. **Validation** — Each agent's output is parsed and validated against the [agent JSON schema](#agent-json-schema).
5. **Weight-based vote** — The consensus engine computes a weighted score, deduplicates findings, and generates a consensus report.

### Consensus Rules

Verdicts are weighted: `approve = 1`, `conditional = 0.5`, `reject = -1`.

```
score = sum(weight[verdict] for each agent) / num_agents
```

| Score | Consensus |
|-------|-----------|
| 1.0 (unanimous approve) | **STRONG GO** |
| -1.0 (unanimous reject) | **STRONG NO-GO** |
| > 0 with conditionals | **GO WITH CAVEATS** |
| > 0 without conditionals | **GO (N-M)** |
| <= 0 | **HOLD (N-M)** |

Labels are dynamic: `(N-M)` reflects the actual majority/minority split (e.g., `GO (2-1)` or `HOLD (1-1)` in degraded mode).

### Confidence Formula

```
weight_factor = (abs(score) + 1) / 2    # symmetric for approve and reject
base_confidence = sum(majority_confidence) / num_agents
confidence = base_confidence * weight_factor
```

Using `abs(score)` ensures that both unanimous approve and unanimous reject produce high confidence. At `score = 0` (exact tie), confidence is halved — appropriate for an undecided split.

### Output Example

```
+==================================================+
|          MAGI SYSTEM -- VERDICT                  |
+==================================================+
|  Melchior (Scientist):   APPROVE (90%)           |
|  Balthasar (Pragmatist): CONDITIONAL (85%)       |
|  Caspar (Critic):        REJECT (78%)            |
+==================================================+
|  CONSENSUS: GO WITH CAVEATS                      |
+==================================================+

## Key Findings
[!!!] **[CRITICAL]** SQL injection in query builder _(from melchior, caspar)_
[!!]  **[WARNING]**  Missing retry logic for API calls _(from balthasar)_
[i]   **[INFO]**     Consider adding request timeout _(from caspar)_

## Dissenting Opinion
**Caspar (Critic)**: Risk of data loss outweighs shipping speed...

## Conditions for Approval
- **Balthasar**: Add integration tests before merge

## Recommended Actions
- **Melchior** (Scientist): Fix SQL injection, add parameterized queries
- **Balthasar** (Pragmatist): Ship after adding integration tests
- **Caspar** (Critic): Rework query layer before proceeding
```

### Degraded Mode

When an agent fails (timeout, parse error, validation error):
- Warning printed to stderr identifying the failed agent and reason
- Synthesis proceeds if >= 2 agents succeeded
- Report flagged with `"degraded": true` and `"failed_agents": [...]`

### Fallback Mode

When `claude -p` is unavailable, the skill simulates all three perspectives sequentially within a single response, with **Caspar first** to reduce anchoring bias.

---

## Agent JSON Schema

All agents must produce output matching this schema:

```json
{
  "agent": "melchior | balthasar | caspar",
  "verdict": "approve | reject | conditional",
  "confidence": 0.0-1.0,
  "summary": "One-line verdict summary",
  "reasoning": "Detailed analysis (2-5 paragraphs)",
  "findings": [
    {
      "severity": "critical | warning | info",
      "title": "Short title (non-empty)",
      "detail": "Explanation"
    }
  ],
  "recommendation": "What this agent recommends"
}
```

---

## Project Structure

```
.claude-plugin/
  plugin.json                 -- Plugin manifest (name, version, author, repository)
  marketplace.json            -- Local marketplace config for development
skills/magi/
  SKILL.md                    -- Orchestrator (mode detection, model selection, workflow, fallback)
  agents/
    melchior.md               -- Scientist system prompt
    balthasar.md              -- Pragmatist system prompt
    caspar.md                 -- Critic system prompt (adversarial by design)
  scripts/
    __init__.py               -- Python package marker
    run_magi.py               -- Async orchestrator with --model flag
    synthesize.py             -- Facade: re-exports from validate, consensus, reporting
    validate.py               -- ValidationError + load_agent_output schema validation
    consensus.py              -- VERDICT_WEIGHT + determine_consensus (weight-based scoring)
    reporting.py              -- AGENT_TITLES + format_banner + format_report (ASCII)
    parse_agent_output.py     -- Claude CLI JSON extractor (3 output formats)
tests/
  test_synthesize.py          -- 74 tests: validation, consensus, confidence, dedup, labels
  test_parse_agent_output.py  -- 19 tests: fence stripping, text extraction, pipeline
  test_run_magi.py            -- 16 tests: arg parsing, model flag, orchestration, validation
docs/
  MAGI-System-Documentation.md  -- Full technical reference (Spanish)
pyproject.toml                -- Python >= 3.9, dual license, dev deps, tool config
conftest.py                   -- tdd-guard pytest plugin + sys.path setup
Makefile                      -- verify, test, lint, format, typecheck targets
```

### Module Architecture

The synthesis engine is split into focused, single-responsibility modules:

| Module | Responsibility | Key Exports |
|--------|---------------|-------------|
| `validate.py` | Schema validation | `ValidationError`, `load_agent_output` |
| `consensus.py` | Weight-based scoring | `VERDICT_WEIGHT`, `determine_consensus` |
| `reporting.py` | ASCII banner + markdown report | `format_banner`, `format_report` |
| `synthesize.py` | Facade (re-exports all above) | All public symbols |

**Import convention:** Always import from `synthesize` (the facade), not directly from sub-modules:

```python
from synthesize import load_agent_output, determine_consensus, format_report
```

---

## Running Tests

```bash
# All tests (109 total)
python -m pytest tests/ -v

# Full verification (tests + lint + format + types)
make verify

# Individual checks
make test        # pytest
make lint        # ruff check
make format      # ruff format --check
make typecheck   # mypy
```

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Claude Code CLI (`claude -p`) | For parallel mode | Fallback available without it |
| Python 3.9+ | Yes | Uses `asyncio`, `dict[str, Any]` syntax |

### Dev Dependencies

```bash
pip install pytest pytest-asyncio ruff mypy
```

---

## License

Dual licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE-APACHE), at your option.

---

## Credits

The MAGI concept originates from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995) by Hideaki Anno / Gainax. The three supercomputers — Melchior, Balthasar, and Caspar — govern critical decisions through structured consensus, each embodying a different facet of their creator Dr. Naoko Akagi.

This plugin is a creative adaptation of that multi-perspective decision-making philosophy for software engineering, where the three "facets" become three analytical lenses: technical rigor, pragmatism, and adversarial risk assessment.
