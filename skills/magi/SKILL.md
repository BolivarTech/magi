---
name: magi
description: >
  Multi-perspective analysis system inspired by the MAGI supercomputers from Evangelion.
  Spawns three sub-agents (Melchior, Balthasar, Caspar) that evaluate the same problem
  from different angles and reach a consensus by majority vote. Use this skill for
  decisions with genuine uncertainty, significant consequences, or real trade-offs.
  Trigger phrases: "MAGI", "three perspectives", "multi-perspective analysis",
  "MAGI review", or explicit requests for multi-angle evaluation.
  NOT suitable for trivial questions, simple bugs, or decisions with obvious answers.
---

# MAGI System — Multi-Perspective Analysis Skill

## Overview

The MAGI system uses three specialized sub-agents to analyze problems from
complementary perspectives, then synthesizes their verdicts into a final
consensus. Each agent has a distinct analytical lens:

| Agent        | Codename   | Lens                        |
|------------- |----------- |-----------------------------|
| **Melchior** | Scientist  | Technical rigor & correctness |
| **Balthasar**| Pragmatist | Practicality & maintainability |
| **Caspar**   | Critic     | Risk, edge cases & failure modes |

## Workflow

### Step 1: Evaluate complexity and detect mode

**Complexity gate:** Before launching three sub-agents, assess whether the request
warrants multi-perspective analysis. If the request is simple (single function review,
obvious bug fix, straightforward question with one clear answer), respond directly
without invoking the full MAGI system. MAGI adds value when there is genuine
uncertainty, multiple valid approaches, or significant consequences for a wrong decision.

If the request warrants MAGI, classify into one of three modes:

- **`code-review`** — The user provides code or a diff to evaluate.
- **`design`** — The user asks about architecture, approach selection, or solution design.
- **`analysis`** — General problem analysis, debugging, trade-offs, or decisions.

If ambiguous, default to `analysis`.

### Step 2: Prepare the prompt payload

Construct a single `PROMPT_PAYLOAD` variable containing:

```
MODE: <code-review | design | analysis>
CONTEXT: <user's full question, code, or description>
```

If the user provided files, include their contents (or relevant excerpts) in the CONTEXT block.

### Step 3: Launch the three agents

**Model selection:** The default model for all agents is the **opus** short name.
The Anthropic model ID it resolves to (e.g. `claude-opus-4-7`) is owned by
`skills/magi/scripts/models.py`, so future model bumps are a one-line edit in that
registry — never hardcode the resolved ID here. If the user explicitly requests
a different model in their prompt (e.g., "usa sonnet", "with haiku",
"use sonnet model"), use that short name instead.

Valid short names: `opus`, `sonnet`, `haiku`. If the user requests an unsupported
model (e.g., "use gpt-4"), inform them of the valid options and default to `opus`.

**Parallel mode (preferred):** Use the Bash tool to execute the Python orchestrator.
The orchestrator launches all three agents in parallel, applies timeouts, validates
outputs, and runs synthesis automatically:

    python skills/magi/scripts/run_magi.py <mode> <input_file_or_text> [--model opus] [--timeout 900] [--output-dir <dir>]

Pass `--model sonnet` or `--model haiku` to override the default.

The orchestrator handles everything: agent launching, output parsing, schema validation,
failure alerting, consensus synthesis, and report generation. No additional steps needed.

If a file needs to be analyzed, pass the file path as the second argument.
If analyzing inline text, wrap it in quotes.

**Native sub-agent mode:** If Bash execution is unavailable, use the Agent tool to
launch three sub-agents in parallel, each with its respective system prompt and the
shared PROMPT_PAYLOAD. Pass the selected model via the Agent tool's `model` parameter
(e.g., `"model": "opus"`).

Read each agent's system prompt from the `agents/` directory:
- `agents/melchior.md`
- `agents/balthasar.md`
- `agents/caspar.md`

Each agent must respond with **only** a JSON object matching this schema:

```json
{
  "agent": "melchior | balthasar | caspar",
  "verdict": "approve | reject | conditional",
  "confidence": 0.0-1.0,
  "summary": "One-line verdict summary",
  "reasoning": "Detailed analysis from this agent's perspective (2-5 paragraphs)",
  "findings": [
    {
      "severity": "critical | warning | info",
      "title": "Short title",
      "detail": "Explanation"
    }
  ],
  "recommendation": "What this agent recommends doing"
}
```

### Step 4: Synthesize the consensus (only for native sub-agent mode)

**Skip this step if you used the Python orchestrator in Step 3** — it runs synthesis
automatically and outputs the full canonical report to stdout.

If you used native sub-agent mode, run synthesis manually:

    python skills/magi/scripts/synthesize.py <agent1.json> <agent2.json> [agent3.json] --output report.json

**This JSON report is not the final user-facing output.** It is the structured
input you will use in Step 5 to render the canonical banner + sections.
Never display `report.json` to the user as the final answer — always render
the canonical format first.

The synthesis uses weight-based scoring with `approve=1, conditional=0.5, reject=-1`:

| Score | Condition | Consensus |
|-------|-----------|-----------|
| 1.0 | unanimous approve | **STRONG GO** |
| -1.0 | unanimous reject | **STRONG NO-GO** |
| > 0 | has conditionals | **GO WITH CAVEATS (N-M)** |
| > 0 | no conditionals | **GO (N-M)** |
| 0 | — | **HOLD -- TIE** |
| < 0 | — | **HOLD (N-M)** |

The `(N-M)` suffix reflects the effective majority-minority split: approves
and conditionals count together on the "go" side, rejects on the "no" side.
A unanimous caveats result renders as `GO WITH CAVEATS (3-0)` and a
caveats-with-dissent as `GO WITH CAVEATS (2-1)`.

### Step 5: Present the results

> ## ⚠ MANDATORY FINAL OUTPUT CONTRACT
>
> **Every MAGI invocation, regardless of execution mode, MUST end with the
> canonical output below — byte-for-byte structurally identical.**
>
> - **Parallel mode** (Python orchestrator): copy the stdout of `run_magi.py`
>   verbatim into your reply. Do **not** paraphrase, summarize, reorder, or
>   strip sections. Do **not** prepend a "here are the results" preamble that
>   displaces the banner. Do **not** append additional analysis after
>   `## Recommended Actions` unless the user explicitly asks a follow-up
>   question.
> - **Native sub-agent mode**: after running `synthesize.py`, the JSON report
>   alone is **not** an acceptable final answer. You MUST render the canonical
>   banner + sections from the JSON and emit them to the user with the same
>   alignment, section order, and widths produced by
>   `reporting.format_report()`.
> - **Fallback mode** (no sub-agents, single-model simulation): after the three
>   per-agent JSON blocks and the `## Synthesis` heading, you MUST emit the
>   canonical banner + sections exactly as specified below. The three agent
>   JSON blocks are intermediate scaffolding; the banner + sections are the
>   final deliverable.
>
> **Verification before responding:** re-read your draft reply against the
> template below. If any of the following is missing or misaligned, fix it
> before sending:
>
> 1. Banner border lines match `+` + 50 `=` + `+`.
> 2. Agent verdict rows are column-aligned (all verdicts start at the same
>    column).
> 3. Consensus row uses the `(N-M)` suffix when applicable.
> 4. `## Key Findings` rows have the marker/severity/title column layout.
> 5. Section order is: banner → Key Findings → Dissenting Opinion → Conditions
>    for Approval → Recommended Actions. Optional sections are omitted when
>    empty; never reordered.
> 6. There is **no** `## Consensus Summary` section.

#### Canonical output template

```
+==================================================+
|          MAGI SYSTEM -- VERDICT                  |
+==================================================+
|  Melchior (Scientist):   APPROVE (90%)           |
|  Balthasar (Pragmatist): CONDITIONAL (85%)       |
|  Caspar (Critic):        REJECT (78%)            |
+==================================================+
|  CONSENSUS: GO WITH CAVEATS (2-1)                |
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

#### Format rules (normative)

**Banner:**
- Total width: 52 columns. Border lines: `+` + 50 `=` + `+`.
- Title centered: `|` + `"MAGI SYSTEM -- VERDICT".center(50)` + `|`.
- Agent rows: `|  <label> <VERDICT> (<conf>%)` padded with spaces to 50 inner chars, then `|`.
- Agent labels pad to the longest label so all verdict words start at the same column.
  With the three standard agents, the longest label is `Balthasar (Pragmatist):` (23 chars),
  so padding produces the column alignment shown above.
- `<conf>` is an integer percentage (e.g., `85%`, never `0.85`).
- `<VERDICT>` is uppercase: `APPROVE`, `CONDITIONAL`, or `REJECT`.
- Consensus row: `|  CONSENSUS: <label>` ljust 50, then `|`.

**Key Findings** (section omitted if there are no findings):
- Header: `## Key Findings`
- One line per deduplicated finding. No blank lines between findings. No indented detail line.
- Fixed-width columns:
  - Marker field, width 5, left-justified: `[!!!]`, `[!!] `, `[i]  ` (space-padded).
  - One space separator.
  - Severity label field, width 14, left-justified: `**[CRITICAL]**`, `**[WARNING]** `, `**[INFO]**    `.
  - One space separator.
  - Title starts at column 22.
  - Suffix: ` _(from <agent1>, <agent2>)_` listing every reporting agent.
- Findings are sorted by severity (critical → warning → info).

**Dissenting Opinion** (section omitted if no dissent):
- Header: `## Dissenting Opinion`
- One line per dissenting agent: `**Name (Title)**: <summary>`
- Summary only — do **not** include the full `reasoning` field.
- **Why summary-only**: the Dissenting Opinion section is for at-a-glance
  awareness of the minority position, not the full argument. The complete
  `reasoning` text is preserved in the JSON report on disk and in each agent's
  raw output file under the run's temp directory, so nothing is lost — only
  the console view is truncated. This is intentional.

**Conditions for Approval** (section omitted if no conditionals):
- Header: `## Conditions for Approval`
- Bullet list: `- **Name**: <condition>`  (name only, no role in parentheses).

**Recommended Actions** (always present):
- Header: `## Recommended Actions`
- Bullet list: `- **Name** (Title): <recommendation>` — one per agent, in stable order.

**Consensus Summary is NOT a section.** Do not emit `## Consensus Summary` — the
banner already encodes the verdict and the key findings/dissent sections carry the
substantive content. This is a **breaking change from MAGI 1.0.x**, which had a
`## Consensus Summary` block between the banner and `## Key Findings`. Downstream
consumers that parsed that header must now read `consensus.majority_summary` from
the JSON report instead of grepping the rendered markdown.

## Fallback (no sub-agents available)

If neither `claude -p` nor sub-agent tools are available, simulate all three
perspectives sequentially within a single response.

**Rules for fallback mode:**

1. **Order: Caspar first.** Generate the Critic's perspective first to establish
   risks before the other agents can anchor toward approval.
2. **Independence:** Write each perspective as if it has NOT seen the others.
   Do not reference previous agents' findings in later sections.
3. **Intermediate output:** Present three clearly labeled sections, each
   containing the full JSON object for that agent. These three blocks are
   **intermediate scaffolding**, not the final answer.
4. **Final output is non-negotiable:** After the three per-agent JSON blocks
   and the `## Synthesis` heading, you MUST emit the canonical banner and
   sections from Step 5 — byte-for-byte identical to the format produced by
   `reporting.format_report()` in parallel mode. The same banner width,
   column alignment, section order, finding-row layout, and
   no-`## Consensus Summary` rule apply. See the MANDATORY FINAL OUTPUT
   CONTRACT callout at the top of Step 5.
5. **Acknowledge limitation:** Note in the report that fallback mode was used,
   as a single model generating all three perspectives has inherent anchoring bias.

Example structure:

    ### Caspar (Critic)
    {caspar JSON}

    ### Melchior (Scientist)
    {melchior JSON}

    ### Balthasar (Pragmatist)
    {balthasar JSON}

    ## Synthesis

    +==================================================+
    |          MAGI SYSTEM -- VERDICT                  |
    +==================================================+
    |  Melchior (Scientist):   ...
    ...
    +==================================================+
    |  CONSENSUS: ...                                  |
    +==================================================+

    ## Key Findings
    ...
    ## Dissenting Opinion
    ...
    ## Conditions for Approval
    ...
    ## Recommended Actions
    ...

The banner and five sections here are **the** user-facing answer. The three
agent JSON blocks above are diagnostic scaffolding only.

## Notes

- For code review mode, agents should reference specific line numbers.
- For design mode, agents should consider scalability and migration cost.
- The system is deliberately adversarial — Caspar's job is to find fault.
  This is a feature, not a flaw.
