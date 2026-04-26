# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAGI is a Claude Code **plugin** implementing a multi-perspective analysis system inspired by the MAGI supercomputers from Neon Genesis Evangelion. Three specialized AI agents — Melchior (Scientist), Balthasar (Pragmatist), Caspar (Critic) — independently analyze the same input through different lenses, then their verdicts are synthesized via majority vote.

`docs/MAGI-System-Documentation.md` is the full technical reference.

## Development Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run full verification (lock sync + tests + lint + format + types)
make verify

# Run individual checks
make lockcheck     # uv lock --check (fails if uv.lock drifts from pyproject.toml)
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
    run_magi.py               — Async orchestrator (launch_agent, run_orchestrator, _DisplayLogGate)
    temp_dirs.py              — magi-run-* LRU cleanup + create_output_dir + realpath traversal guard
    subprocess_utils.py       — windows_kill_tree + reap_and_drain_stderr + write_stderr_log + timeouts
    status_display.py         — Live tree renderer (ANSI + plain, UTF-8 + ASCII fallback)
    stderr_shim.py            — _buffered_stderr_while context + stderr shims for display-active runs
    models.py                 — MODEL_IDS + resolve_model + VALID_MODELS
    synthesize.py             — Facade: re-exports from validate, consensus, reporting
    validate.py               — ValidationError + load_agent_output schema validation
    consensus.py              — VERDICT_WEIGHT + determine_consensus (weight-based scoring)
    reporting.py              — AGENT_TITLES + format_banner + format_report (ASCII)
    parse_agent_output.py     — Claude CLI JSON extractor (3 output formats)
tests/
  fixtures/claude-cli-outputs/  — Pinned claude -p output samples auto-discovered by the contract test
  test_synthesize.py          — 142 tests: validation, non-dict top-level guard, consensus, findings, banner verdict-preservation, SKILL.md template parity
  test_parse_agent_output.py  — 27 tests: fence stripping, text extraction, pipeline, claude -p fixture contract
  test_run_magi.py            — 76 tests: arg parsing, --no-status, orchestration, tracked_launch states, temp_dirs LRU, subprocess_utils taskkill order
  test_status_display.py      — 41 tests: init, update, render, ASCII fallback, async lifecycle, stop idempotency, tripwire, refresh-loop Exception resilience
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

### Model selection guidance

The default short name is `opus`, applied uniformly to all three agents — there is no per-agent model assignment, the differentiation comes entirely from each agent's system prompt. Operators can override with `--model sonnet` or `--model haiku`. The registry in `models.py` is the only place where short names map to Anthropic IDs (`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`); bumping a model is a one-line edit there.

**Recommended model per mode** (cost / quality trade-off observed across the captured `magi-report.json` corpus, opus runs ~$0.25/agent ≈ $0.75/run total):

| Mode | Recommended | Rationale |
|------|-------------|-----------|
| `code-review` | `opus` | Dense technical reasoning; correctness depends on subtle interaction tracing where opus pulls ahead. ~$0.75/run is acceptable for PR-blocking decisions. |
| `design` | `opus` | Multi-level abstraction (architecture, scaling cliffs, hidden coupling). The cohort confidence on `design` outputs in the audit corpus drops sharply on smaller models. |
| `analysis` | `sonnet` | Exploratory questions and trade-off framing; sonnet matches opus quality at ~4× lower cost (~$0.20/run total). Reserve opus for cases where the answer drives a hard decision. |
| Smoke / fixtures / contract testing | `haiku` | ~10× cheaper (~$0.07/run); validates the schema and parsing pipeline without burning opus budget on inputs whose verdict you do not act on. |

**Default resolution (2.2.5+)**: when `--model` is omitted, `parse_args` looks the mode up in `MODE_DEFAULT_MODELS` (`models.py`) and resolves to that short name. Explicit `--model X` always wins over the mode default.

**Resolution history**:

* 2.0.x-2.2.2: uniform `opus` default for every mode.
* 2.2.3 (2026-04-25): switched `analysis` default to `sonnet` for cost relief on the assumption that sonnet matched opus quality on exploratory work at ~4× lower cost.
* 2.2.5 (2026-04-26): reverted `analysis` default to `opus`. Production data showed Caspar (most-output agent by design, 4-7K output tokens vs 2-3K for Mel/Bal) failed in ≥33% of sbtdd Loop verifications under sonnet — an order of magnitude above the 3.3% design assumption. Sonnet's ~8K max-output ceiling pressed against Caspar's adversarial-by-design verbosity; 2.2.4 retry could not recover because the failure was structural, not stochastic. The `MODE_DEFAULT_MODELS` plumbing is preserved for future per-mode differentiation; only the `analysis` value flipped back. Operators who want sonnet for analysis can still pass `--model sonnet` explicitly.

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

1. Bump `"version"` in `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` (both occurrences), `pyproject.toml`, and `uv.lock` (the `[[package]] name = "magi"` entry).
2. Run `make verify` — lock sync must pass, all tests must pass, zero lint warnings, clean formatting, no type errors.
3. Commit with message `fix|feat|chore: <short summary> and release <version>` and push to `main` on GitHub.
4. Tag the release commit: `git tag -a v<version> -m "Release <version>: <short summary>"` then `git push origin v<version>`. Every release from 2.1.4 onward carries an annotated tag in `v<MAJOR>.<MINOR>.<PATCH>` form; the tag is non-optional because it is the durable anchor for rollback, changelogs, and GitHub's release UI. Lightweight tags are not acceptable — the annotated form carries tagger identity and a dated message.
5. Users pick up updates with `/plugin marketplace update`.

### Pre-merge MAGI self-review (standard procedure)

For any release that ships behaviour-affecting code — `feat:` commits, `fix:` of medium severity or higher, or any change that touches `tracked_launch` / `launch_agent` / the schema — run MAGI on the branch's diff before merging. This turns the plugin into a self-auditing system and consistently catches gaps that grep + manual review miss (the 2.2.3 lockstep-test gap, the 2.2.4 docstring drift, etc.).

The procedure is fixed so the cadence is repeatable across releases:

1. **Build the diff bundle**. From the feature branch, write a single context file (e.g., `magi-review-<version>-context.md`) containing:
   - One paragraph describing what changed and why.
   - The full `git diff main feature/<branch>` inside a fenced ``diff`` block.
   - 4-6 specific "hot review questions" that direct the agents at the parts of the diff you most want challenged. Avoid leading questions; phrase as "Is X correct?", "Could Y break Z?", "Does the help text leak internal symbols?".
2. **Invoke MAGI** on the bundle: `python skills/magi/scripts/run_magi.py code-review magi-review-<version>-context.md --timeout 900 --no-status`. Wall time is typically 1-3 minutes; cost on opus is ~$0.75 per review.
3. **Read the raw `.json` per agent** in the run's temp directory rather than only the rendered banner. Per-finding `detail` fields carry the reasoning the rendered table truncates to titles.
4. **Run the findings through `superpowers:receiving-code-review`** *before* implementing anything. The skill enforces verify-before-act:
   - Each finding gets a verdict — `valid` (implement), `wrong` (push back with code reference), `style` (push back as preference), `out-of-scope` (already-debated decision), or `YAGNI` (no real consumer).
   - Push-backs require a concrete proof — a file:line reference, a test, or a counter-example. Performative agreement is forbidden by the skill.
5. **Address valid findings** in a follow-up `fix:` commit on the same branch. Push-backs go in the commit message body so the rationale survives later bisects. If the review surfaces an architectural problem (≥ 3 valid findings of the same shape), stop and re-plan before merging.
6. **Delete the context bundle** before merging so it does not enter `main`. The MAGI run's `magi-report.json` on disk is the audit artifact; the bundle is a transient input.
7. **Merge to `main` with `--no-ff`**, tag, push.

**When to skip**: pure documentation commits, pure version bumps, gitignore changes, reformatting, dependency updates with no behaviour change. The cost (~$0.75) is real; do not run MAGI on commits where the diff is mechanical.

**When to escalate beyond MAGI**: if MAGI returns `STRONG NO-GO` or two consecutive iterations of `GO WITH CAVEATS` whose conditions overlap, treat the feature as architecturally suspect and pull in a second human reviewer before merging.

**Yes, this procedure applies to the release that introduces it.** 2.2.4 — the release that ships this section — was itself reviewed under the procedure. The "skip when mechanical" exemption is for diffs that contain no behaviour-affecting code; a release that *introduces* this procedure as a new behavioural commitment for future reviews is not mechanical.

### Marketplace structure

The plugin system relies on two manifest files in `.claude-plugin/`:

| File | Purpose |
|------|---------|
| `plugin.json` | Plugin identity: name, version, author, repository, license, skills path |
| `marketplace.json` | Marketplace catalog: owner, plugin list with sources, categories, tags |

A single marketplace repo can host multiple plugins by pointing `source` to other GitHub repos. This repo hosts only the `magi` plugin with `source: "./"` (self-contained).

## Test Coverage

311 tests across 4 test files (310 passed, 1 skipped on Windows):

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_synthesize.py` | 142 | Validation, string type/length checks, bool confidence rejection, agent/verdict type guards, non-dict top-level JSON (R4-1), zero-width Unicode (incl. U+2060-U+206F word joiner / invisible math operators / tag controls), finding sub-field limits, weight-based consensus, confidence formula, findings dedup, dynamic labels, HOLD -- TIE, duplicate agents, banner width + alignment + integer percent, verdict-suffix preservation under overlong labels (R4-3), report sections + ordering, dissent summary-only, SKILL.md template parity |
| `tests/test_parse_agent_output.py` | 27 | Fence stripping, text extraction (3 formats), fail-fast on unknown types, pipeline integration, pinned claude -p output contract via auto-discovered fixtures (R4-5) |
| `tests/test_run_magi.py` | 96 | Arg parsing, --no-status flag, model passthrough, orchestration, degraded mode, input validation, cleanup_old_runs LRU/symlink (via `temp_dirs` module — R4-4), tracked_launch states (success/timeout/failed), display start() failure fallback, Windows kill-tree order (taskkill before proc.kill, via `subprocess_utils` — R4-4), stderr replay OSError safety, single-shot retry on ValidationError with feedback injection and `retrying` display state (2.2.0), `retried_agents` telemetry field with conditional presence and sorted serialisation (2.2.1), per-mode default model resolution with explicit `--model` override + `MODE_DEFAULT_MODELS` ↔ `VALID_MODES` lockstep invariant (2.2.3), single-shot retry extended to `json.JSONDecodeError` from parse stage with `ValueError` boundary guard (2.2.4) |
| `tests/test_status_display.py` | 46 | Init, update, render, ASCII fallback, async lifecycle, stop idempotency, write-path invariant tripwire, refresh-loop OSError resilience, refresh-loop non-OSError resilience (R4-2), retrying-state glyphs (UTF-8 ↻, ASCII lowercase r), retrying not terminal, unicode probe includes retry glyph, cp1252 fallback safe (2.2.2) |

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
| R3-1 | Windows kill-tree ran *after* `proc.kill()` — parent torn down before `taskkill /T` could enumerate descendants | `_windows_kill_tree(pid)` now runs **before** `proc.kill()`; `proc.kill()` stays as belt-and-suspenders so the asyncio wrapper still observes the exit |
| R3-2 | `_ZERO_WIDTH_RE` skipped `U+2060-U+206F` (word joiner, invisible math operators, deprecated tag controls) — Cf-category invisibles could smuggle dedup-key collisions | Regex widened to `[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff\u00ad]`; covers every Cf-category invisible in the BMP that prior tests touched |
| R3-3 | `_buffered_stderr_while` replay re-raised `OSError` from `saved.write` / `saved.flush`, masking the body's in-flight exception (root cause hidden by a cleanup-only failure) | `finally` wraps the replay in `try/except OSError`; clean-body runs no longer crash on broken pipes, and an in-flight exception always propagates unchanged |
| R3-4 | `StatusDisplay._refresh_loop` and the final `stop()` redraw bubbled `OSError` (e.g. `BrokenPipeError`) out of the background task; `stop()` then re-raised it and discarded gathered agent results | Both call sites wrap `_redraw()` in `try/except OSError`; refresh loop sets `_running = False` and returns silently when the stream dies |
| R4-1 | `validate.load_agent_output` ran `set(data.keys())` without checking `isinstance(data, dict)` — non-object top-level JSON (list, string, null, number) raised `AttributeError` and bypassed the `ValidationError` contract, producing opaque `'list' object has no attribute 'keys'` traces in `asyncio.gather` | Explicit `isinstance(data, dict)` guard after `json.load`; non-dict top-level JSON now raises `ValidationError("Top-level JSON must be an object, got <type>.")` with the filepath preserved |
| R4-2 | `_refresh_loop` and the final `stop()` redraw caught only `OSError` — a `ValueError` (closed `io.StringIO`), `UnicodeEncodeError` (mis-probed encoding), or any future bug in `_redraw` would bubble out of the background task and make `stop()` re-raise on `await self._refresh_task`, discarding gathered agent results the same way R3-4 had for `OSError` | Both handlers widened to `except Exception` with `# noqa: BLE001` and a pointer to `_refresh_loop`'s docstring: *the live display is never allowed to fail the run*. `BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`) still propagate |
| R4-3 | `_fit_content` tail-truncated the banner row; on a pathologically long agent label the verdict+confidence suffix (`APPROVE (85%)`) was erased even though the row stayed width-valid, leaving operators with a structurally correct banner that had lost the one token it exists to communicate | `_fit_content` accepts a `preserve_suffix` keyword; when the suffix fits, truncation eats the label prefix instead. `format_banner` threads `verdict_display (conf_pct)` through as the preserved suffix, falling back to the original tail-cut when no suffix is requested |
| R4-4 | `run_magi.py` had grown to hold arg parsing, LRU cleanup, Windows kill-tree, stderr-shim coordination, display lifecycle, subprocess orchestration, consensus call-out, and report writing — a new maintainer needed four invariants in their head to touch `run_orchestrator` safely | Extracted two pure-filesystem/pure-subprocess modules: `temp_dirs.py` (`MAGI_DIR_PREFIX`, `cleanup_old_runs`, `create_output_dir`, plus TOCTOU/symlink helpers) and `subprocess_utils.py` (`windows_kill_tree`, `reap_and_drain_stderr`, `write_stderr_log`, `format_stderr_excerpt`, plus `TASKKILL_TIMEOUT`/`PROC_WAIT_REAP_TIMEOUT`). `run_magi.py` keeps the orchestration flow and re-exports `cleanup_old_runs`/`create_output_dir`/`MAGI_DIR_PREFIX` for longstanding test imports |
| R4-5 | Nothing in the suite exercised the three `claude -p` output shapes end-to-end because the CLI requires a paid API key; a silent wrapper change at Anthropic would surface only as a production parse failure | Added `tests/fixtures/claude-cli-outputs/` with five pinned captures (`result-shape`, `content-block-shape`, `plain-string-shape`, `result-with-markdown-fences`, `content-block-not-first`) and a parametrized contract test in `test_parse_agent_output.py` that auto-discovers new `.json` files. A guard test asserts the fixture directory is non-empty so a future rename cannot degrade the contract to a vacuous pass |

### Known limitations

- **TOCTOU residual**: A narrow race window exists between `realpath()` and `rmtree()` in `cleanup_old_runs`. Acceptable for dev-tooling context; not suitable for security-critical environments.
- **Windows subprocess orphans (residual)**: the reap path now invokes `taskkill /F /T /PID` *before* `proc.kill()` so the tree is collapsed while the parent-child mapping is still intact. Orphans only survive when `taskkill` is missing from PATH or its `_TASKKILL_TIMEOUT` budget is exceeded — both already emit a "may be orphaned" warning naming the pid so operators can clean up manually.
- **Temp directory scan**: `cleanup_old_runs` scans the entire system temp directory. May be slow on machines with thousands of temp entries (e.g., shared CI runners).
- **Windows non-VT TTY**: On legacy Windows consoles where `ENABLE_VIRTUAL_TERMINAL_PROCESSING` cannot be enabled, the status display falls through to plain mode and emits one append-only line per agent state change (no in-place redraw). Modern Windows Terminal, ConEmu, and WSL terminals are unaffected. Disable the display with `--no-status` if the append output is undesirable.
- **`_StderrBufferShim` coverage gap**: the shim intercepts `sys.stderr.write`, `sys.stderr.flush`, and `sys.stderr.buffer.write`. The following paths bypass it:
  - `os.write(sys.stderr.fileno(), b"...")` — direct OS-level writes to fd 2.
  - Subprocesses inheriting fd 2 (MAGI itself uses `stderr=PIPE` so this doesn't apply to `launch_agent`, but third-party code invoked from user-level hooks could).
  - **Pre-cached stderr references**: modules that capture `err = sys.stderr` at import time and later call `err.write(...)` hold a reference to the real stream, not to the swapped-in shim. The shim replaces `sys.stderr` only for the duration of `_buffered_stderr_while`; a reference captured before that context manager enters is unaffected. If MAGI ever imports a library that does this, its writes will appear directly in the display's redraw region.
- **Buffered diagnostics on hard process death**: `_buffered_stderr_while` flushes its buffer in a `finally` clause, so diagnostics survive ordinary exceptions, `CancelledError`, `KeyboardInterrupt`, and `SystemExit`. They are lost only on `SIGKILL`, segfault, or `os._exit()` — all out of scope for Python-level cleanup.

## Open technical debt

These items are real (not "accepted residuals" like the section above) and have a concrete fix path, but were de-prioritised to avoid bloating an unrelated release. They are documented here so they do not silently age into a third "known limitation" disguised as architecture.

### Backwards-compatibility re-exports in `run_magi.py`

**What**: Lines 56-67 of `run_magi.py` keep `cleanup_old_runs`, `create_output_dir`, and `MAGI_DIR_PREFIX` re-exported from the module's `__all__` so the legacy `from run_magi import cleanup_old_runs` import path continues to work for tests that pre-date the R4-4 split. The accompanying comment explicitly says *"Future code should import from `temp_dirs` directly"* — the re-exports are a transition aid, not a stable API.

**Why it is debt**: every grep for `temp_dirs` symbols has to touch two modules, and the `__all__` of `run_magi.py` carries names that have nothing to do with orchestration. New contributors learn the shortcut and the convention erodes.

**Fix path** (~1-2 hours):
1. `grep -rn "from run_magi import \(cleanup_old_runs\|create_output_dir\|MAGI_DIR_PREFIX\)"` to enumerate the call sites — primarily inside `tests/test_run_magi.py`.
2. Rewrite each import to `from temp_dirs import ...`.
3. Delete the three names from `run_magi.__all__` and the explanatory comment block.
4. Run `make verify`; mypy will flag any miss because the names will no longer be importable from `run_magi`.

**Acceptance**: `git grep "from run_magi import cleanup_old_runs"` returns zero matches. The orchestration symbols (`MODEL_IDS`, `VALID_MODELS`, `resolve_model`) stay re-exported because they straddle modules legitimately — only the `temp_dirs` triple is targeted.

**Why deferred**: pure cleanup, zero behaviour change, zero user impact. Bundle into the next refactor-flavoured release (e.g., 2.3.0) rather than a patch.

### `run_magi.py` size and orchestrator layering

**What**: `run_magi.py` is 628 LOC, the largest module in the project (next is `status_display.py` at 443; the orchestration tier average is ~250). It carries arg parsing, `launch_agent`, the `_DisplayLogGate` class, `_safe_display_update`, `_build_retry_prompt`, `run_orchestrator`, the `tracked_launch` closure (with seven captured variables), and `main`. The R4-4 extraction (`temp_dirs`, `subprocess_utils`) already pulled the pure-filesystem and pure-subprocess pieces out; what remains is the genuinely entangled orchestration core, but it is still bigger than any one reader needs to hold in their head.

**Why it is debt**: every retry-related change so far (2.2.0 retry, 2.2.1 telemetry) had to thread state through the closure rather than through an explicit object. The closure is correct, just not introspectable. A future operator-visible feature like "retry count > 1" would benefit from a real state object instead of a sixth captured variable.

**Fix path** (estimated 3-5 hours, not patch material):
1. Extract `launch_agent` and its timeout / stderr handling into a new `agent_runner.py` module exposing one async function. The function should be a pure I/O operation: subprocess + parse + validate. No display, no retry.
2. Extract the per-agent state machine (running → retrying → terminal) into a `LaunchTracker` class in a new `tracking.py` module. Class instance owns the display-update calls, the retry decision, and the retried/failed sets that today live as closure variables.
3. `run_orchestrator` shrinks to: build display, build trackers for each agent, gather, build report. Probably ~80 LOC instead of ~150.

**Acceptance**: `wc -l skills/magi/scripts/*.py | sort -n` shows no module above ~300 LOC; `tracked_launch` no longer exists as a closure; the tracker class has unit tests independent of the orchestrator.

### `ValueError` boundary conflates parser-shape and config errors

**What**: ``tracked_launch`` (post-2.2.4) does **not** retry on ``ValueError``. The decision is correct in aggregate but lumps two qualitatively different failure modes:

* ``resolve_model("gpt4")`` → ``ValueError("Unknown model 'gpt4'…")``. A configuration error. Retry would re-fail identically; not retrying is correct.
* ``parse_agent_output._extract_text({"weird_shape": …})`` → ``ValueError("Unexpected Claude CLI output type…")``. A structural change in the upstream Anthropic CLI output. Retry *might* recover (the model could roll a recognised shape on the second attempt), but it could also be a sustained CLI change that needs a parser update.

**Why it is debt**: today the wrong-side decision has zero observed cost — there are no production reports of ``_extract_text`` ``ValueError``. But a future Anthropic CLI shape change would surface as a non-recoverable agent loss exactly where 2.2.4 just closed the equivalent gap for ``JSONDecodeError``. The right fix is a custom exception class — e.g., ``ParseShapeError`` — raised by ``_extract_text`` and added to the retry catch alongside ``ValidationError`` and ``json.JSONDecodeError``. The ``resolve_model`` ``ValueError`` keeps its current semantics (no retry).

**Fix path** (estimated 1-2 hours):
1. Introduce ``class ParseShapeError(ValueError)`` in ``parse_agent_output.py``. Subclassing ``ValueError`` preserves backward compatibility for any caller already doing ``except ValueError``.
2. Replace the two ``raise ValueError`` sites in ``_extract_text`` with ``raise ParseShapeError`` (file-too-large stays as ``ValueError``; size limit is a structural failure that retry cannot fix).
3. Extend ``tracked_launch`` retry catch to ``(ValidationError, json.JSONDecodeError, ParseShapeError)``.
4. Add a regression test exercising the shape-fail → retry path; keep the existing ``test_value_error_from_parse_does_not_retry`` test renamed to target ``resolve_model``-style ``ValueError`` (to keep that boundary explicit).

**Acceptance**: a future Anthropic CLI shape change produces a single recoverable failure per agent rather than a 2-of-3 catastrophic loss. The test suite still pins the ``ValueError``-from-config boundary so configuration errors do not silently retry.

**Why deferred**: zero observed cost today. Caspar flagged it during the 2.2.4 self-review (W3) as a known asymmetry; if a production incident surfaces, this becomes the 2.2.5 patch with the same TDD shape as 2.2.4.

**Why deferred**: this is the kind of refactor that warrants a minor release (2.3.0) and a brainstorm session because it touches the hottest async path. Doing it inside a patch would conflate behaviour-preserving work with behaviour-defining work and make any future bisect harder. Telemetry from the 2026-05-15 routine should also land first — it may suggest specific tracker responsibilities (per-agent budget, retry-count) that the refactor should accommodate from day one.

## Post-release hardening

### Agent prompt contract reinforcement (all three agents)

**Context**: A production-log scan across 10 runs × 3 agents (30 extracted outputs, versions 2.0.3 / 2.1.0 / 2.1.1 / 2.1.2 / 2.1.3) found one schema violation: in `magi-run-d48ls1lm` (19-Apr-2026, under 2.1.3), Caspar emitted a well-formed JSON that parsed cleanly with `json.loads` but omitted the required top-level `recommendation` key. The validator correctly raised `ValidationError("Agent output missing keys: ['recommendation']")`, the orchestrator dropped Caspar and set `degraded=true`, and the run completed on Melchior + Balthasar. No downstream corruption; the degraded-mode path worked as designed. Incidence in the sampled corpus: 1/30 ≈ 3.3%, exclusively on Caspar (the most opinionated agent by design, highest output-token pressure).

**Fix**: The single-line `IMPORTANT:` closer in every agent system prompt (`melchior.md`, `balthasar.md`, `caspar.md`) now enumerates all seven required top-level keys and states explicitly that any omission causes the output to be rejected and the agent dropped from consensus. The reinforcement is applied to all three agents — not just Caspar — because the failure mode is LLM-schema-drift rather than agent-specific and the identical closer across prompts is easier to audit than a per-agent divergence. This is a probabilistic mitigation: it raises the cost of omission in the prompt, it does not prove omission impossible.

### Single-shot agent retry on ValidationError (2.2.0)

**Context**: Prompt reinforcement (above) is probabilistic and cannot drive the schema-drift rate to zero. A targeted retry closes the last gap without widening the schema contract and without paying latency cost on the success path — the retry only fires when `load_agent_output` raises.

**Implementation**: `run_orchestrator.tracked_launch` now wraps `launch_agent` in a nested `try` that catches `ValidationError` only. On catch, the closure (a) emits a new `retrying` display state, (b) rebuilds the prompt via `_build_retry_prompt(original_prompt, error)` — original prompt verbatim + `---RETRY-FEEDBACK---` delimiter + the ValidationError message + a restatement of the 7-key schema — and (c) re-invokes `launch_agent` with the rebuilt prompt and the same per-agent `timeout`. Terminal-state emission (`success` / `timeout` / `failed`) remains on the outer handler pair, so the display invariant "exactly one terminal state per agent" survives the retry branch. If the retry raises anything (`ValidationError` again, `TimeoutError`, `RuntimeError`, `CancelledError`), it flows through the outer handler and the agent is dropped into the pre-existing degraded path.

**Scope**: retry triggers **only** on `ValidationError`. `TimeoutError`, subprocess exit errors, cancellation, and signals pass through unchanged so the 2.1.x degraded semantics are preserved verbatim. Retry count is fixed at 1 — a second retry is a separate decision.

**Display**: `StatusDisplay.VALID_STATES` gains `"retrying"`; `_UTF8_GLYPHS.icons` renders it as `↻` and `_ASCII_GLYPHS.icons` as lowercase `r` (lowercase avoids collision with capital `R` in agent/state words — same cosmetic rule the `~`-for-timeout glyph already follows). `_UNICODE_PROBE` was widened to include `↻` so streams that cannot render the retry glyph fall back to the ASCII glyph set **before** the first retry, not on it.

**Budget**: each attempt receives the full `--timeout` ceiling. The retry is not given a reduced budget, and the first attempt's residual time is not carried over. Worst-case wall time per retried agent is therefore `2 × --timeout`; the orchestrator's overall wall time is unchanged for the ~97% of runs where no agent retries.

**Non-goals (2.2.0)**: retry on subprocess timeout, retry on non-schema exceptions, retry count > 1. Tests guard each non-goal so these behaviors cannot regress silently into scope.

### Single-shot retry — JSON parse extension (2.2.4)

**Trigger**: An iter-2 sbtdd Loop 2 catastrophic failure (post-2.2.3) lost two of three agents to ``json.JSONDecodeError`` raised inside ``parse_agent_output`` BEFORE ``load_agent_output`` could wrap the failure into ``ValidationError``. The 2.2.0 retry only caught ``ValidationError``, so both agents were dropped without a second attempt and synthesis aborted on the 2-agent minimum.

**Fix**: ``tracked_launch`` now catches ``(ValidationError, json.JSONDecodeError)`` instead of ``(ValidationError,)``. Both flow through the identical retry path (emit ``retrying``, rebuild prompt with the parser/validator error message, re-launch with a fresh ``--timeout`` budget, fall back to degraded mode if the retry also fails). ``_build_retry_prompt`` was reworded from "failed schema validation" to "rejected by the parsing pipeline" so the corrective message is accurate for both failure classes; the seven-key schema reminder stays because it is useful context regardless of which layer rejected the output.

**Still out of scope** (each pinned by a regression test): ``ValueError`` from ``resolve_model`` (invalid short name — retry would re-fail identically), ``ValueError`` from ``parse_agent_output._extract_text`` (unrecognised CLI shape — needs a parser update, not a retry), ``TimeoutError``, ``RuntimeError``, ``OSError``, ``asyncio.CancelledError``. The retry net is now wide enough to recover from typical LLM output drift (truncation, malformed braces, schema violations) and narrow enough that configuration/structural errors still surface immediately.

**Telemetry**: the 2.2.1 ``retried_agents`` field records every agent that took the retry path, regardless of whether the trigger was ``ValidationError`` or ``JSONDecodeError``. Downstream tooling does not need to know which class triggered the retry — only that one fired.

### Analysis-mode default reverted to opus (2.2.5)

**Trigger**: After 2.2.4 shipped, the user reported a recurring Caspar-drop pattern in sbtdd Loop verifications: each verification runs MAGI three times, and **at least one of the three drops Caspar to a degraded(2,0)** result. The 2.2.0 retry + 2.2.4 JSON-parse extension fired correctly but could not recover Caspar because the failure was structural, not stochastic — the second attempt under the same model hit the same output-token ceiling.

**Why Caspar specifically**: Caspar is adversarial by design (longest reasoning, most findings, densest critique). Across the captured ``magi-report.json`` corpus, Caspar's output token count ranged 3620-6691 — 2-3× larger than Melchior's or Balthasar's. Under the 2.2.3 sonnet default, that put Caspar at 50-87% of the ~8K max-output ceiling for analysis-mode runs. Variance under sbtdd's larger inputs was enough to push Caspar over the cliff while Mel/Bal completed comfortably.

**Why this is not just a degraded-result issue**: losing Caspar specifically biases consensus toward false-positive approval. ``GO_WITH_CAVEATS (2-0)`` from Mel+Bal is structurally different from ``GO_WITH_CAVEATS (2-0)`` with Caspar dissent — the surviving two are the constructive lenses, the missing one is the adversarial one. sbtdd's "F's auto-recovery is for N=1 only" cannot save this because synthesis succeeds with two agents; the only fix that restores quality is preventing the drop in the first place.

**Fix**: ``MODE_DEFAULT_MODELS["analysis"]`` flipped from ``"sonnet"`` back to ``"opus"``. Restores the 32K max-output budget and gives Caspar the room it needs. Cost reverts from ~$0.20/run to ~$0.75/run for bare-``analysis`` invocations; the cost-saving rationale of 2.2.3 was empirically refuted by the failure rate observed in production.

**Tripwire activation**: this revert fired the policy documented in ``memory/routine_telemetry_post_2.2.1.md`` ("Observation policy in effect") by sustained evidence rather than the literal "n=2 iter-2-style" letter. The observed pattern (≥33% Caspar drop per verification, structural cause) was decisive enough to act ahead of the 2026-05-15 telemetry routine.

**MAGI self-review skipped on this commit by design** (precedent for similar reverts): the change restores a configuration that ran successfully through the 0.x.x → 2.2.2 release range, is mechanical from the behaviour-history standpoint, and the decision criteria are baked into the test docstring + the ``models.py`` rationale comment so a future bisector reconstructs the reasoning without the MAGI artifact. The standard "Pre-merge MAGI self-review" procedure exists to catch architectural surprise on forward-going behaviour changes; running it on a documented revert to a known-stable state would be ceremony.

**Still open after 2.2.5**: the cost-saving question for ``analysis``. Sonnet 4.6 may match opus on quality but it cannot fit Caspar's full output today. Two paths the 2026-05-15 telemetry routine should inform:
1. **Per-agent model differentiation**: Caspar on opus, Mel/Bal on sonnet. Saves ~$0.30/run vs full opus while preserving Caspar's output budget. Requires per-agent model plumbing (currently the orchestrator passes one model for all three).
2. **Output-budget plumbing**: a ``--max-tokens`` flag in ``launch_agent`` that overrides the model default. Lets operators raise the sonnet budget on Caspar specifically without changing models. Requires a parser for the Anthropic CLI's ``--max-tokens`` option.

Both are deferred until the routine surfaces enough data to choose between them.

**Telemetry follow-up (2.2.1)**: ``run_orchestrator`` now records every agent that hit the retry path in a closure-captured set and serialises it to the report under the new ``retried_agents`` key. The key follows the same conditional-presence convention as ``degraded`` and ``failed_agents``: omitted entirely on clean runs, sorted alphabetically when present so the JSON is byte-stable. Downstream consumers can compose ``retried_agents - failed_agents`` for the retry-recovered cohort and ``retried_agents & failed_agents`` for retry-also-failed. This closes the 2.2.0 blind spot where successful retries were indistinguishable from clean first-attempt runs and is what makes the post-release fault-rate decision criterion measurable.

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
