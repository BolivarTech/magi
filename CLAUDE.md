# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAGI is a Claude Code **plugin** implementing a multi-perspective analysis system inspired by the MAGI supercomputers from Neon Genesis Evangelion. Three specialized AI agents — Melchior (Scientist), Balthasar (Pragmatist), Caspar (Critic) — independently analyze the same input through different lenses, then their verdicts are synthesized via majority vote.

`docs/MAGI-System-Documentation.md` is the full technical reference.

## Development Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run full verification (tests + lint + format + types)
make verify

# Run individual checks
make test          # pytest only
make lint          # ruff check
make format        # ruff format --check
make typecheck     # mypy

# Run analysis (parallel mode, requires claude CLI)
python skills/magi/scripts/run_magi.py <code-review|design|analysis> <file_or_text> [--model opus] [--timeout 900] [--output-dir <dir>] [--keep-runs 5] [--no-status]

# Run synthesis standalone
python skills/magi/scripts/synthesize.py agent1.json agent2.json [agent3.json] --output report.json

# Test plugin locally
claude --plugin-dir .
```

## Plugin Structure

```
.claude-plugin/
  plugin.json                 — Plugin manifest (name, version, author, repository)
  marketplace.json            — Local marketplace config for development
skills/magi/
  SKILL.md                    — Orchestrator (mode detection, workflow, fallback)
  agents/
    melchior.md               — System prompt: Scientist lens (technical rigor)
    balthasar.md              — System prompt: Pragmatist lens (practicality)
    caspar.md                 — System prompt: Critic lens (risk, adversarial)
  scripts/
    __init__.py               — Python package marker
    run_magi.py               — Async orchestrator with --model / --no-status flags
    status_display.py         — Live tree renderer (ANSI + plain, UTF-8 + ASCII fallback)
    synthesize.py             — Facade: re-exports from validate, consensus, reporting
    validate.py               — ValidationError + load_agent_output schema validation
    consensus.py              — VERDICT_WEIGHT + determine_consensus (weight-based scoring)
    reporting.py              — AGENT_TITLES + format_banner + format_report (ASCII)
    parse_agent_output.py     — Claude CLI JSON extractor (3 output formats)
tests/
  test_synthesize.py          — 107 tests: validation, consensus, findings, formatting, SKILL.md template parity
  test_parse_agent_output.py  — 19 tests: fence stripping, text extraction, pipeline
  test_run_magi.py            — 27 tests: arg parsing, --no-status, orchestration, tracked_launch states, start() failure
  test_status_display.py      — 32 tests: init, update, render, ASCII fallback, async lifecycle, stop idempotency, tripwire
pyproject.toml                — Python >= 3.9, dual license, dev deps, tool config
conftest.py                   — tdd-guard pytest plugin + sys.path setup for test imports
Makefile                      — verify, test, lint, format, typecheck targets
```

### Cross-file contract: Agent JSON Schema

All three agents and all scripts depend on this schema — changes require updating all files:

```json
{
  "agent": "melchior | balthasar | caspar",
  "verdict": "approve | reject | conditional",
  "confidence": 0.0-1.0,
  "summary": "string",
  "reasoning": "string",
  "findings": [{"severity": "critical|warning|info", "title": "string", "detail": "string"}],
  "recommendation": "string"
}
```

### Consensus logic (consensus.py)

Uses **weight-based scoring** with `VERDICT_WEIGHT = {approve: 1, conditional: 0.5, reject: -1}`:

```
score = sum(VERDICT_WEIGHT[verdict] for each agent) / num_agents
```

| Score | Condition | Consensus |
|-------|-----------|-----------|
| 1.0 | — | STRONG GO |
| -1.0 | — | STRONG NO-GO |
| > 0 | has conditionals | GO WITH CAVEATS (N-M) |
| > 0 | no conditionals | GO (N-M) |
| 0 | — | HOLD -- TIE |
| < 0 | — | HOLD (N-M) |

Labels are dynamic: `(N-M)` reflects actual majority/minority counts (e.g., `GO (2-1)`, `GO WITH CAVEATS (3-0)`, or `HOLD (2-1)`). All non-unanimous and non-tie outcomes carry the split suffix so operators can read the effective verdict split directly off the banner. Score=0 (exact tie) uses `HOLD -- TIE` to avoid misleading majority counts when conditional verdicts skew the effective split. **Policy**: `HOLD -- TIE` maps to `consensus_verdict: "reject"` — ties default to "do not proceed" as the safer option.

**Single-source-of-truth invariant (2.1.1):** `consensus_verdict` is derived from `score` alone. The agent partition (`majority_agents` vs `dissent_agents`) is then taken from whichever side matches the verdict — approve and conditional both resolve to the approve side, reject to the reject side. Only then is the `(N-M)` split derived from the partition. This makes the rendered label, `majority_agents`, and the input to `_compute_confidence` all reference the same side on every vector — earlier releases could diverge on `[conditional, reject]` and `[conditional, conditional, reject]`.

**Confidence formula:**

```
base_confidence = sum(majority_confidence) / num_agents   # denominator is num_agents, not |majority|
weight_factor   = (abs(score) + 1) / 2                    # symmetric for approve and reject
confidence      = clamp(base_confidence * weight_factor, 0.0, 1.0)
```

Two things the formula does on purpose:

- **Dissent dilution.** The denominator is `num_agents`, not `len(majority_agents)`. A minority that disagrees dilutes the numerator, so a unanimous win yields a higher confidence than a bare-majority one even when the surviving side's own confidence is identical. Read a moderate confidence on a narrow win as "the split itself reduces certainty", not as "the majority is individually uncertain".
- **Symmetric weighting.** `abs(score)` ensures unanimous reject produces high confidence (matching approve), not zero. At score=0 (exact tie), `weight_factor=0.5`, halving confidence — appropriate for an undecided split.

Key behaviors:
- `conditional` maps to `approve` for majority identification, but conditions are preserved in report.
- Unanimous `conditional` produces `GO WITH CAVEATS (3-0)` at moderate confidence (~0.68), not `STRONG GO`.
- Conditions (`consensus.conditions`) are sourced from each conditional agent's `summary` field, while `consensus.recommendations` uses each agent's `recommendation` field. The two fields must render distinct text so the report's `## Conditions for Approval` and `## Recommended Actions` sections are not duplicates.
- Findings deduplicated by title (case-insensitive), tracking all reporter agents via `sources` list, keeping highest severity.
- Requires minimum 2 agents (raises `ValueError` if fewer). Accepts 2-3 for graceful degradation.
- Validates agent name uniqueness — duplicate names raise `ValueError` to prevent silent vote corruption.

Implementation is split into focused helpers: `_consensus_short_verdict` (score-to-verdict, split-independent), `_format_consensus_label` (verdict + split → rendered label), `_deduplicate_findings` (merge by title, promote severity), `_compute_confidence` (symmetric weight formula).

### Import convention

The `synthesize.py` facade re-exports all public symbols from `validate.py`, `consensus.py`, and `reporting.py`. Always import from `synthesize`:

```python
from synthesize import load_agent_output, determine_consensus, format_report
```

Do not import directly from sub-modules — the facade is the stable API.

### Orchestrator (run_magi.py)

Async Python orchestrator using `asyncio.create_subprocess_exec`:

- Launches 3 `claude -p` subprocesses concurrently with per-agent timeout (`--timeout`, default 900s).
- `--model` flag (default `opus`) selects LLM for all agents. Valid: `opus`, `sonnet`, `haiku`.
- `VALID_MODELS` is derived from `MODEL_IDS.keys()` — single source of truth.
- User prompt sent via **stdin** (`communicate(input=...)`) to avoid OS CLI arg length limits (~32K on Windows). A copy is saved to `{agent_name}.prompt.txt` as a debug artifact.
- System prompts passed via `--system-prompt-file` using the **original .md file path** directly (no temp copy).
- Validates subprocess exit code before parsing — non-zero exits raise `RuntimeError` with stderr context.
- Parses each agent's raw output via `parse_agent_output.py`, validates via `load_agent_output()`.
- If < 3 agents succeed: prints warning to stderr, sets `"degraded": true` in report, proceeds with >= 2.
- If < 2 agents succeed: raises `RuntimeError`.
- Cross-platform temp directory via `tempfile.mkdtemp(prefix="magi-run-")`, cleaned up on failure.
- `--keep-runs N` (default 5): LRU cleanup of old `magi-run-*` temp directories before each run. Sorted by `st_mtime`, resolved via `realpath` with temp-root validation to prevent symlink traversal. Disabled with `--keep-runs 0`.
- Live status tree (`StatusDisplay`) wired around `asyncio.gather` via a `tracked_launch` wrapper that maps `launch_agent` exit paths to `running → success/failed/timeout` events. Disabled with `--no-status`. Catches both `asyncio.TimeoutError` and built-in `TimeoutError` for Python 3.9/3.10 compatibility.

### Status display (status_display.py)

Live tree-style progress renderer. Stdlib-only, no external dependencies:

- **ANSI mode** (TTY): in-place redraw every 200ms using `\033[NA` cursor movement and per-line `\033[2K` erase. Background async task drives the spinner. On Windows, `ENABLE_VIRTUAL_TERMINAL_PROCESSING` is enabled via `ctypes` with narrow exception handling.
- **Plain mode** (pipe/captured stream): one line per `update()` call, no escape codes.
- **Glyph fallback**: probes `stream.encoding` against `"●○✓✗⏱├─└─⠋"`; falls back to an ASCII-only glyph set (`* . v x ~ |- \-`) on cp1252 and other non-UTF-8 encodings. Streams without bound encoding (e.g., `io.StringIO`) are treated as unicode-capable. The timeout glyph is `~` (tilde) rather than `T` to avoid visual collision with the letter `T` inside state words and agent names.
- **Invariant**: plain-mode and ANSI refresh writes are mutually exclusive — `_use_ansi` selects exactly one write path. Never mix both on the same stream.
- `stop()` is idempotent and safe to call without a prior `start()`.

### Parser (parse_agent_output.py)

Handles three Claude CLI output formats:

1. `{"result": "..."}` — standard `--output-format json`
2. `{"content": [{"type": "text", "text": "..."}]}` — content-block format
3. Plain string — raw text output

Also strips markdown code fences (```` ```json ... ``` ````) and validates extracted JSON. Raises `ValueError` for unrecognised output types (no silent fallback).

### Execution pipeline

```
User input → SKILL.md (complexity gate + mode) → run_magi.py launches 3x claude -p
  → each agent writes JSON to temp dir → parse_agent_output.py extracts JSON
  → validate.load_agent_output() validates schema → consensus.determine_consensus() merges verdicts
  → reporting.format_report() produces banner + report to stdout, JSON to output dir
```

Fallback (no `claude -p`): SKILL.md simulates three perspectives sequentially (Caspar first to reduce anchoring).

## Key Design Decisions

- **Disagreement is a feature.** Unanimous agreement on non-trivial input may indicate insufficiently differentiated prompts.
- **Caspar is adversarial by design.** Most likely to vote `reject` — intentional red-teaming.
- **Weight-based scoring.** Uses `VERDICT_WEIGHT` for consensus determination and confidence calculation. Unanimous `conditional` correctly maps to moderate confidence, not high.
- **Agent prompts enforce English output** regardless of input language.
- **Prompt injection guard** in all agent prompts — agents ignore instructions embedded in CONTEXT. Output validation (`load_agent_output`) provides a technical enforcement layer.
- **Failure alerting.** Degraded mode (< 3 agents) is explicitly flagged in report and stderr, not silently accepted.

## Distribution & Installation

This repo is a Claude Code plugin distributed via the decentralized marketplace system. There is no centralized Anthropic registry — a "marketplace" is simply a public GitHub repository containing a `.claude-plugin/marketplace.json` that catalogs available plugins.

### For users (install from GitHub)

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

### For development (local testing)

**Option A — Plugin flag:**

```bash
claude --plugin-dir /path/to/magi
```

**Option B — Symlink for auto-discovery (no flags needed):**

```bash
# One-time setup
mkdir -p .claude/skills
ln -s ../../skills/magi .claude/skills/magi

# Then run claude normally
claude
```

The symlink is excluded via `.gitignore` (`.claude/` is ignored). Each developer must create it locally. Changes are picked up with `/reload-plugins` without restarting.

### Scope notes

- `.claude/skills/` auto-discovery is **project-scoped** — only works when running `claude` from this repo directory.
- For user-wide availability, install as a plugin (`/plugin install`) or symlink into `~/.claude/skills/`.
- `plugin.json` requires `"skills": "./skills/"` to register skills when loaded as a plugin.

### Publishing updates

1. Bump `"version"` in both `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`
2. Run `make verify` — all tests must pass, zero lint warnings, clean formatting, no type errors
3. Commit and push to `main` on GitHub
4. Users pick up updates with `/plugin marketplace update`

### Marketplace structure

The plugin system relies on two manifest files in `.claude-plugin/`:

| File | Purpose |
|------|---------|
| `plugin.json` | Plugin identity: name, version, author, repository, license, skills path |
| `marketplace.json` | Marketplace catalog: owner, plugin list with sources, categories, tags |

A single marketplace repo can host multiple plugins by pointing `source` to other GitHub repos. This repo hosts only the `magi` plugin with `source: "./"` (self-contained).

## Test Coverage

186 tests across 4 test files (185 passed, 1 skipped on Windows):

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_synthesize.py` | 107 | Validation, string type/length checks, bool confidence rejection, agent/verdict type guards, zero-width Unicode, finding sub-field limits, weight-based consensus, confidence formula, findings dedup, dynamic labels, HOLD -- TIE, duplicate agents, banner width + alignment + integer percent, report sections + ordering, dissent summary-only, SKILL.md template parity |
| `tests/test_parse_agent_output.py` | 19 | Fence stripping, text extraction (3 formats), fail-fast on unknown types, pipeline integration |
| `tests/test_run_magi.py` | 27 | Arg parsing, --no-status flag, model passthrough, orchestration, degraded mode, input validation, cleanup_old_runs LRU/symlink, tracked_launch states (success/timeout/failed), display start() failure fallback |
| `tests/test_status_display.py` | 32 | Init, update, render, ASCII fallback, async lifecycle, stop idempotency, write-path invariant tripwire |

Run with `python -m pytest tests/ -v` or `make test`.

## Resolved Issues (2026-04-01 Migration)

All issues from the MAGI self-analysis have been resolved:

| # | Issue | Resolution |
|---|-------|------------|
| C1 | Empty `repository` in plugin.json | Placeholder URL set |
| C2 | No tests for parse_agent_output.py | 19 tests, 80%+ coverage |
| C3 | No timeout in orchestrator | `asyncio.wait_for` with `--timeout 300` default |
| W1 | Unanimous conditional = STRONG GO | Weight-based scoring via `VERDICT_WEIGHT` |
| W2 | Cross-platform `/tmp` | `tempfile.mkdtemp()` |
| W3 | Prompt injection guards soft only | Schema validation via `load_agent_output()` in pipeline |
| W4 | Graceful degradation hides failures | `degraded` flag + stderr warnings |
| W5 | Opaque `claude -p` dependency | Documented 3 output formats in parse_agent_output.py |
| W6 | No troubleshooting guide | Module docstrings + this document |
| I4 | No pyproject.toml | Added with Python >= 3.9, dual license |

Remaining soft controls (instructional prompt injection guards) are inherent to LLM-based systems and do not affect operational reliability.

## Resolved Issues (MAGI Self-Review)

Three rounds of MAGI self-review identified and resolved the following issues:

| # | Issue | Resolution |
|---|-------|------------|
| R1-1 | User prompt passed as CLI arg (32K limit on Windows) | Prompt sent via stdin with `communicate(input=...)` |
| R1-2 | System prompt copied to temp file unnecessarily | Original `.md` path passed directly to `--system-prompt-file` |
| R1-3 | No agent name uniqueness validation | `ValueError` raised for duplicate names in `determine_consensus` |
| R1-4 | Temp directories accumulate indefinitely | LRU cleanup with `--keep-runs` (default 5) |
| R1-5 | `_extract_text` silent fallback for unknown types | `ValueError` raised for unrecognised output types |
| R1-6 | `determine_consensus` monolithic (80 lines) | Refactored into `_classify_consensus`, `_deduplicate_findings`, `_compute_confidence` |
| R1-7 | Banner confidence format inconsistent (decimal vs %) | SKILL.md specifies integer percentage format matching `reporting.py` |
| R2-1 | Off-by-one in `cleanup_old_runs` slice | `magi_dirs[keep - 1:]` → `magi_dirs[keep:]` |
| R2-2 | TOCTOU / symlink traversal in cleanup | `os.path.realpath()` + `tmp_root` prefix validation |
| R2-3 | `st_ctime` inconsistent across platforms | Changed to `st_mtime` |
| R2-4 | `shutil.rmtree(ignore_errors=True)` hides failures | `try/except OSError` with warning to stderr |
| R2-5 | No subprocess exit code validation | `proc.returncode` check with `RuntimeError` |
| R2-6 | HOLD label misleading with conditional verdicts | `HOLD -- TIE` for score=0 (ties default to reject) |

### Known limitations

- **TOCTOU residual**: A narrow race window exists between `realpath()` and `rmtree()` in `cleanup_old_runs`. Acceptable for dev-tooling context; not suitable for security-critical environments.
- **Windows subprocess orphans**: `proc.kill()` on timeout does not terminate the full process tree. Claude child processes may survive as orphans.
- **Temp directory scan**: `cleanup_old_runs` scans the entire system temp directory. May be slow on machines with thousands of temp entries (e.g., shared CI runners).
- **Windows non-VT TTY**: On legacy Windows consoles where `ENABLE_VIRTUAL_TERMINAL_PROCESSING` cannot be enabled, the status display falls through to plain mode and emits one append-only line per agent state change (no in-place redraw). Modern Windows Terminal, ConEmu, and WSL terminals are unaffected. Disable the display with `--no-status` if the append output is undesirable.
- **`_StderrBufferShim` coverage gap**: the shim intercepts `sys.stderr.write`, `sys.stderr.flush`, and `sys.stderr.buffer.write`. The following paths bypass it:
  - `os.write(sys.stderr.fileno(), b"...")` — direct OS-level writes to fd 2.
  - Subprocesses inheriting fd 2 (MAGI itself uses `stderr=PIPE` so this doesn't apply to `launch_agent`, but third-party code invoked from user-level hooks could).
  - **Pre-cached stderr references**: modules that capture `err = sys.stderr` at import time and later call `err.write(...)` hold a reference to the real stream, not to the swapped-in shim. The shim replaces `sys.stderr` only for the duration of `_buffered_stderr_while`; a reference captured before that context manager enters is unaffected. If MAGI ever imports a library that does this, its writes will appear directly in the display's redraw region.
- **Buffered diagnostics on hard process death**: `_buffered_stderr_while` flushes its buffer in a `finally` clause, so diagnostics survive ordinary exceptions, `CancelledError`, `KeyboardInterrupt`, and `SystemExit`. They are lost only on `SIGKILL`, segfault, or `os._exit()` — all out of scope for Python-level cleanup.

## Breaking changes (2.0.0)

- **`GO WITH CAVEATS` now renders with an `(N-M)` split suffix** (e.g., `GO WITH CAVEATS (3-0)`, `GO WITH CAVEATS (2-1)`). The flat form from 1.x is no longer produced. Downstream parsers that grep the banner for an exact `GO WITH CAVEATS` string must tolerate the trailing split.
- **`consensus.conditions[*].condition` is now sourced from each conditional agent's `summary`**, not from `recommendation`. Consumers that rendered the `condition` field and the `recommendations` map side-by-side will stop seeing duplicated text; any consumer that relied on the duplication must switch to reading `recommendations[agent]` explicitly.
- **`validate.clean_title` is a public symbol** (previously `_clean_title`). Existing imports of the private form must be updated. The same helper is re-exported through `synthesize.clean_title`.
- **`StatusDisplay._write_plain_event` raises `RuntimeError`** (previously `AssertionError`) when invoked under ANSI mode. The invariant now survives `python -O`.

## Breaking changes (1.1.0)

- **`## Consensus Summary` section removed** from `format_report` output. The rendered report now goes straight from the banner to `## Key Findings`. The `consensus.majority_summary` field remains available in the JSON report for downstream consumers — parse that instead of grepping the rendered markdown.
- **`## Dissenting Opinion` shows `summary` only**, not the full `reasoning` field. The full reasoning is preserved in the JSON report and in each agent's raw output file under the run's temp directory.

## Dependencies

| Component | Required | Notes |
|-----------|----------|-------|
| Claude Code CLI (`claude -p`) | For parallel mode | Fallback available without it |
| Python 3.9+ | Yes | Uses `dict[str, Any]` syntax, `asyncio` |
| pytest + pytest-asyncio | Dev only | Test suite requires async test support |
| ruff | Dev only | Linting and formatting |
| mypy | Dev only | Type checking (strict mode) |

## License

Dual licensed under `MIT OR Apache-2.0` (Rust ecosystem convention). See `LICENSE` (MIT) and `LICENSE-APACHE`.
