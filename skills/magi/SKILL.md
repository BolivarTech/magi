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

**Model selection:** The default model for all agents is **opus** (Claude Opus 4.6).
If the user explicitly requests a different model in their prompt (e.g., "usa sonnet",
"with haiku", "use sonnet model"), use that model instead.

Valid models: `opus`, `sonnet`, `haiku`. If the user requests an unsupported model
(e.g., "use gpt-4"), inform them of the valid options and default to `opus`.

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
automatically and outputs the full report.

If you used native sub-agent mode, run synthesis manually:

    python skills/magi/scripts/synthesize.py <agent1.json> <agent2.json> [agent3.json] --output report.json

The synthesis uses weight-based scoring with `approve=1, conditional=0.5, reject=-1`:

| Score | Condition | Consensus |
|-------|-----------|-----------|
| 1.0 | unanimous approve | **STRONG GO** |
| -1.0 | unanimous reject | **STRONG NO-GO** |
| > 0 | has conditionals | **GO WITH CAVEATS** |
| > 0 | no conditionals | **GO (2-1)** |
| 0 | — | **HOLD -- TIE** |
| < 0 | — | **HOLD (2-1)** |

### Step 5: Present the results

Format the output as:

```
+==================================================+
|          MAGI SYSTEM -- VERDICT                   |
+==================================================+
|  Melchior (Scientist):    [VERDICT] ([CONF]%)      |
|  Balthasar (Pragmatist):  [VERDICT] ([CONF]%)      |
|  Caspar (Critic):         [VERDICT] ([CONF]%)      |
+==================================================+
|  CONSENSUS: [FINAL VERDICT]                       |
+==================================================+
```

Where `[CONF]` is the confidence as an integer percentage (e.g., `88%`, not `0.88`).
This matches the format used by `reporting.py` (`:.0%`).

Then present:
1. **Consensus summary** — 2-3 sentences synthesizing the majority position.
2. **Key findings** — Merged and deduplicated from all three agents, sorted by severity.
3. **Dissent** (if any) — The minority agent's core argument, fairly represented.
4. **Conditions** (if any) — What must be addressed before proceeding.
5. **Recommended action** — A single, actionable next step.

## Fallback (no sub-agents available)

If neither `claude -p` nor sub-agent tools are available, simulate all three
perspectives sequentially within a single response.

**Rules for fallback mode:**

1. **Order: Caspar first.** Generate the Critic's perspective first to establish
   risks before the other agents can anchor toward approval.
2. **Independence:** Write each perspective as if it has NOT seen the others.
   Do not reference previous agents' findings in later sections.
3. **Output format:** Present three clearly labeled sections, each containing
   the full JSON object for that agent. Then add a "## Synthesis" section
   applying the same voting rules.
4. **Acknowledge limitation:** Note in the report that fallback mode was used,
   as a single model generating all three perspectives has inherent anchoring bias.

Example structure:

    ### Caspar (Critic)
    {caspar JSON}

    ### Melchior (Scientist)
    {melchior JSON}

    ### Balthasar (Pragmatist)
    {balthasar JSON}

    ## Synthesis
    [Apply voting rules, present banner and report]

## Notes

- For code review mode, agents should reference specific line numbers.
- For design mode, agents should consider scalability and migration cost.
- The system is deliberately adversarial — Caspar's job is to find fault.
  This is a feature, not a flaw.
