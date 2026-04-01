# MAGI Python Migration & Bug Fix — Design Spec

**Date:** 2026-04-01
**Author:** Julian Bolivar
**Status:** Approved

## Goal

Migrate the MAGI plugin from a bash/Python hybrid to pure Python and resolve
all 15 known issues (3 critical, 6 warnings, 6 improvements) documented in
CLAUDE.md. Strict TDD throughout.

---

## 1. Architecture — Python Orchestrator

Replace `run_magi.sh` with `run_magi.py`. Delete the bash script entirely.

### New file structure

```
skills/magi/scripts/
├── __init__.py               # Package marker
├── run_magi.py               # NEW: async orchestrator (replaces run_magi.sh)
├── parse_agent_output.py     # Existing: CLI JSON extractor (add tests)
├── synthesize.py             # Existing: voting engine (fix weight-based scoring)
```

### `run_magi.py` responsibilities

1. Parse CLI args: `mode`, `input`, `--output-dir`, `--timeout 300`
2. Validate prerequisites (`claude` CLI in PATH)
3. Create temp output dir via `tempfile.mkdtemp(prefix="magi-run-")`
4. Launch 3 `claude -p` subprocesses concurrently via `asyncio.create_subprocess_exec`
5. Apply per-agent timeout via `asyncio.wait_for` (default: 300s)
6. Parse each agent's output via `parse_agent_output` (imported as module)
7. Validate parsed output via `load_agent_output` from `synthesize`
8. Alert when < 3 agents succeed (failure alerting)
9. Call `determine_consensus` + `format_report` directly (no shell-out)
10. Print banner + report, save JSON to `--output-dir`

### Key decisions

- `asyncio` for concurrency — naturally solves timeout bug, stdlib only.
- Import `parse_agent_output` and `synthesize` as Python modules — no more
  shelling out to `python3 synthesize.py`.
- `run_magi.sh` is deleted, not deprecated.

---

## 2. Weight-Based Consensus Scoring

Replace the if/elif chain in `determine_consensus()` with numerical scoring
using the existing `VERDICT_WEIGHT` dict.

### Scoring formula

```
weighted_score = sum(VERDICT_WEIGHT[verdict] for each agent) / num_agents
```

Weights: `approve=1, conditional=0.5, reject=-1`.

### Score-to-consensus mapping

| Scenario                          | Score | Consensus        |
|-----------------------------------|-------|------------------|
| 3x approve                        | 1.0   | STRONG GO        |
| 2 approve + 1 reject              | 0.33  | GO (2-1)         |
| 2 approve + 1 conditional         | 0.83  | GO WITH CAVEATS  |
| 1 approve + 2 conditional         | 0.67  | GO WITH CAVEATS  |
| 3x conditional                    | 0.5   | GO WITH CAVEATS  |
| 1 approve + 1 conditional + 1 reject | 0.17 | GO WITH CAVEATS |
| 1 approve + 2 reject              | -0.33 | HOLD (2-1)       |
| 1 conditional + 2 reject          | -0.5  | HOLD (2-1)       |
| 3x reject                         | -1.0  | STRONG NO-GO     |
| 2 approve (2 agents only)          | 1.0   | STRONG GO        |
| 1 approve + 1 reject (2 agents)    | 0.0   | HOLD (2-1)       |
| 1 approve + 1 conditional (2 agents)| 0.75 | GO WITH CAVEATS  |
| 2 conditional (2 agents only)      | 0.5   | GO WITH CAVEATS  |
| 2 reject (2 agents only)           | -1.0  | STRONG NO-GO     |

### Threshold logic

```python
if score == 1.0:                    # unanimous approve
    consensus = "STRONG GO"
elif score == -1.0:                 # unanimous reject
    consensus = "STRONG NO-GO"
elif score > 0 and has_conditions:  # positive with conditionals
    consensus = "GO WITH CAVEATS"
elif score > 0:                     # positive majority, no conditions
    consensus = "GO (2-1)"
else:                               # score <= 0
    consensus = "HOLD (2-1)"
```

### Confidence adjustment

```python
base_confidence = sum(agent.confidence for majority) / num_agents
weight_factor = (score + 1) / 2  # normalize score to 0.0-1.0
confidence = base_confidence * weight_factor
```

- Unanimous approve at high confidence stays high (~0.85).
- 3x conditional at high confidence drops to ~0.5-0.6.
- Dissent naturally reduces confidence through lower weight_factor.

### Impact on existing tests

Several `TestDetermineConsensus` tests assert specific confidence values and
consensus labels. These must be updated in the TDD Red phase to match the new
scoring. Behavioral intent is preserved (dissent lowers confidence, unanimous
is strongest), but exact numbers change.

---

## 3. Output Validation & Failure Alerting

### Output validation (W3 — prompt injection defense)

After `parse_agent_output.py` extracts JSON, validate against the agent schema
by calling `load_agent_output()` from `synthesize.py`. This reuses existing
validation — no new code needed, just wiring it into the pipeline.

In `run_magi.py`, after parsing each agent's raw output:

1. Call `load_agent_output(parsed_file)` to validate schema.
2. If validation fails, log the error and exclude that agent.
3. Catches malformed outputs (confidence > 1.0, unknown verdicts) that could
   result from prompt injection in the analyzed input.

### Failure alerting (W4 — graceful degradation hides failures)

When < 3 agents succeed, the orchestrator now:

- Prints warning to stderr with the specific failure reason (timeout, parse
  error, validation error).
- Includes `"degraded": true` and `"failed_agents": [...]` in the JSON report.
- Still proceeds with synthesis if >= 2 agents (existing behavior, now visible).

Example output:

```
⚠ WARNING: Agent 'caspar' failed (timeout after 300s) — excluded from synthesis
⚠ WARNING: Running synthesis with 2/3 agents — results may be biased
```

### Documentation (W5, W6)

- Document the three expected Claude CLI output formats in
  `parse_agent_output.py` docstrings:
  1. `{"result": "..."}` — standard `--output-format json`
  2. `{"content": [{"type": "text", "text": "..."}]}` — content-block format
  3. Plain string — raw text output
- Troubleshooting guidance in module docstrings and CLAUDE.md, not a separate
  doc file.

---

## 4. Project Configuration & Remaining Fixes

### `pyproject.toml`

```toml
[project]
name = "magi"
version = "1.0.0"
requires-python = ">= 3.9"
description = "MAGI multi-perspective analysis plugin for Claude Code"
license = "MIT OR Apache-2.0"
authors = [{name = "Julian Bolivar"}]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[tool.pytest.ini_options]
testdir = "tests"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.9"
strict = true
```

No runtime dependencies — everything is stdlib.

### `plugin.json` (C1)

Set `repository` to placeholder:

```json
"repository": "https://github.com/OWNER/magi"
```

To be updated when the repository is made public.

### License

Dual license: `MIT OR Apache-2.0` (Rust ecosystem convention). Add
`LICENSE-APACHE` file alongside existing `LICENSE` (MIT).

### File headers (CLAUDE.local.md)

Every new or modified `.py` file gets:

```python
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
```

### SKILL.md update

Replace bash command reference:

```
python scripts/run_magi.py <mode> <input> [--timeout 300] [--output-dir <dir>]
```

### CLAUDE.md update

- Update plugin structure diagram (remove `.sh`, add `.py`).
- Update development commands.
- Move resolved issues out of Known Issues.
- Update dependency table (remove `bash` as required for orchestrator).

### Deletions

- `run_magi.sh` — fully replaced.
- `.claude/skills/magi/scripts/run_magi.sh` — symlinked copy.

### Test structure

```
tests/
├── test_synthesize.py          # Update for weight-based scoring
├── test_parse_agent_output.py  # NEW: 80%+ coverage
├── test_run_magi.py            # NEW: orchestrator tests (mocked subprocess)
```

---

## 5. Issue Resolution Matrix

| #  | Issue                               | Resolution                              |
|----|-------------------------------------|-----------------------------------------|
| C1 | Empty `repository` in plugin.json   | Placeholder URL                         |
| C2 | No tests for parse_agent_output.py  | TDD — new test file, 80%+ coverage      |
| C3 | No timeout in orchestrator          | `asyncio.wait_for` with `--timeout 300` |
| W1 | Unanimous conditional = STRONG GO   | Weight-based scoring                    |
| W2 | Cross-platform `/tmp`               | `tempfile.mkdtemp()` from migration     |
| W3 | Prompt injection guards soft only   | Schema validation in pipeline           |
| W4 | Graceful degradation hides failures | Explicit warnings + degraded flag       |
| W5 | Opaque `claude -p` dependency       | Docstrings documenting output formats   |
| W6 | No troubleshooting guide            | Module docstrings + CLAUDE.md           |
| I1 | Tests for parse_agent_output.py     | Covered by C2                           |
| I2 | `--timeout` flag                    | Covered by C3                           |
| I3 | Fix unanimous-conditional           | Covered by W1                           |
| I4 | `pyproject.toml`                    | Added with Python >= 3.9                |
| I5 | Document failure modes              | Covered by W6                           |
| I6 | Output validation as defense        | Covered by W3                           |

---

## 6. TDD Execution Order

All work follows strict TDD (Red → Green → Refactor) enforced by tdd-guard.

### Phase 1: parse_agent_output.py tests (C2)

- Red: Write `test_parse_agent_output.py` covering all 3 CLI output formats,
  code fence stripping, error cases. Target 80%+ coverage.
- Green: No production changes needed (existing code should pass).
- Refactor: Clean up if needed.

### Phase 2: Weight-based consensus (W1)

- Red: Update `test_synthesize.py` — modify existing consensus tests for new
  scoring, add tests for unanimous-conditional, 2-agent scenarios with weights.
- Green: Rewrite `determine_consensus()` with weight-based logic.
- Refactor: Remove dead code (old if/elif chain).

### Phase 3: Python orchestrator (C3, W2, W3, W4)

- Red: Write `test_run_magi.py` — async orchestration, timeout handling,
  failure alerting, degraded mode flag, cross-platform temp dirs.
- Green: Implement `run_magi.py`.
- Refactor: Extract shared helpers if needed.

### Phase 4: Configuration & cleanup

- Add `pyproject.toml`, `LICENSE-APACHE`.
- Update `plugin.json` repository field.
- Update `SKILL.md`, `CLAUDE.md`.
- Delete `run_magi.sh`.
- Add file headers to all new/modified files.
- Run `make verify` to confirm all checks pass.
