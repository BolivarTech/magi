# Headless mode

Magi runs non-interactively as a **CI/CD pipeline step** or as an **AI backend**
for another application. It composes the Unix way: read from stdin / `-i`, write
to stdout / `-o`, send diagnostics to stderr, and return a meaningful exit code.
Running `magi` with no subcommand still launches the TUI, unchanged.

## Subcommands

```bash
# Scaffold the .magi/ state directory (config, encrypted DB, logs).
# With -p / MAGI_PASSPHRASE it also bootstraps an empty vault so later
# headless runs need no interaction. Refuses to overwrite an existing .magi/.
magi init -p "correct horse battery staple"

# Without -p (and without MAGI_PASSPHRASE) it still succeeds and exits 0, but the
# DB is left WITHOUT its encryption envelope — see "Running init without -p" below.
magi init

# Prompt from stdin, streamed text answer to stdout.
echo "explain the retry logic in provider.rs" | magi query

# Rich JSON envelope in, one buffered JSON object out.
magi query -i q.json -o out.json --output-format json

# Force a MAGI 3-perspective consensus on the prompt directly. Omitting --mode
# classifies it first (one extra model call) — see Mode routing above.
magi consult --mode code-review -i decision.txt

# Read-only structural probe of the .magi/ DB — never unlocks, never prints a secret.
magi vault diagnose
```

`magi query` and `magi consult` both accept `--mode <code-review|design|analysis>`
and `--untrusted-content` (see [Mode routing](usage.md#mode-routing)); the JSON envelope
carries the same two as fields.

**`-w` / `--workdir` works on all six subcommands** (v0.13.1; it was `query` and
`consult` only before). It is the base for the `.magi/` walk-up, and on the headless
pair it is also the file-tool sandbox root:

```bash
magi init -w /srv/project           # scaffold .magi/ over there, not in $PWD
magi vault ls -w /srv/project       # and -w may also precede: magi vault -w /srv/project ls
magi query -w /srv/project -i q.txt
```

**On all six subcommands**, a `-w` that is not an existing directory is rejected
up front with **exit code 2**, naming the path, and it is never created. The check
runs before the workspace walk-up, the vault and the input read, so a mistyped path
fails as a mistyped path rather than as a missing workspace.

> **Changed in v0.14.0.** `query` and `consult` used to skip this check: an invalid
> `-w` fell through to `no .magi/ state directory found` at **exit 1**, and the
> unvalidated path then became the file-tool sandbox root, failing a second time
> somewhere unrelated. A script keying on `1` for a bad `-w` on those two
> subcommands needs updating; every other failure class keeps its code.

On `vault` the flag may appear **before or after** the nested subcommand: both
`vault -w <dir> ls` and `vault ls -w <dir>` are the same command. Given once on each
side, the **innermost wins**, the rule `git -C` and `docker` follow; given twice on
the *same* side it is an error. On `init` it may only appear after the subcommand,
and repeating it is an error.

`-w` is **not** a top-level flag the way `-p` is: `magi -w <dir> vault ls` is a usage
error, because the flag belongs to the subcommand rather than to `magi` itself.

A `-w` whose path contains a symlinked component is rejected (exit 2), including
when the target itself is the symlink. This is deliberate hardening, and it is a
real difference from the current directory: reaching the same place with `cd` first
resolves the symlink, so it never triggered.

Without the flag every subcommand keeps using the current directory, exactly as
before. Running `magi` with no subcommand (the TUI) does not take `-w`; `cd` first.

**Input** is auto-detected: a JSON object with a top-level `prompt` string is a
rich **envelope** (`{prompt, system?, model?, provider?, max_tool_calls?,
consult?, mode?, untrusted_content?}`); anything else is a plain-text prompt.
CLI flags win over envelope fields, and the operator's `magi.toml` caps (e.g.
`max_tool_calls`) are a ceiling the envelope cannot exceed. `--input-format` /
`--output-format` force the interpretation.

The envelope is the **primary** surface for `mode` and `untrusted_content`, not
an afterthought: an automated gate piping untrusted content through `magi query
-i` / `magi consult -i` has no human reading the prompt before it reaches the
classifier, so it is exactly this JSON shape (not a CLI flag typed by hand)
that needs to declare both fields. See [Mode routing](usage.md#mode-routing) for what
each one does.

### The envelope `system` field

`system` is ignored unless the operator turns it on, with `--allow-system-override` or
`[headless] allow_system_override = true`. Ignoring it is never silent. Magi prints this on
stderr whatever `--log-level` says, writes a warning to the run log, and sets
`applied_caps.system_ignored` to `true`:

```text
warning: the request's "system" field was ignored; pass --allow-system-override or set [headless] allow_system_override = true to apply it
```

The warning never repeats the ignored text. A `system` that is empty or only whitespace counts
as absent: no warning, no flag, and it does not count toward the size cap.

With the flag on, the two commands differ:

- `magi query` replaces the operator's system prompt with the envelope's `system`.
- `magi consult` adds it to the prompt of each of the three mages, under an
  `## Additional instructions` heading placed just before that prompt's `## Output format`
  section. It refines the mage's role; it does not replace it, and the output format still
  applies. `applied_caps.system_override_applied` is `true` only when the text reached the mages.

Two checks run before any model is called, and both exit with code 2:

- `[headless] max_system_bytes` (default 16384, at least 1) caps the size of `system` in bytes.
  It applies to `query` and `consult`, with or without the flag.
- With `consult` and the flag, a `system` that has a line holding a verdict marker
  (`<MAGI_VERDICT>` or `</MAGI_VERDICT>`) or a line that is `## Output format` (surrounding whitespace ignored) is
  rejected, never rewritten.

Rejecting marker lines stops a forged verdict block, not a `system` that asks a mage for a
verdict. Do not use `--allow-system-override` on runs whose verdict gates anything. The JSON
envelope is only as trusted as the process that writes it.

The applied `system` goes to the three mages, and to any fallback a seat rotates to, exactly as
written. It does not pass through a secret detector, so whoever sends the envelope answers for
what is in it.

## JSON output shape

`--output-format json` writes one buffered JSON object with `schema_version`
always the first physical key, followed (in this exact order) by `response`,
`model`, `provider`, `usage`, `timings`, `stop_reason`, `tool_calls[]`,
`transcript[]`, `consult`, `applied_caps`, `error`.

**Consumer contract while `magi-rs` is `0.x`:** `schema_version` does
**not** move when fields are added; the crate's own `0.x` version is the
compatibility signal instead. A consumer of this JSON **must tolerate new
fields and pin the crate version**; do not treat an unchanged `schema_version`
as a backward-compatibility guarantee.

`consult` is `null` unless the run dispatched a MAGI consensus (via the
`consult` tool, an autonomous route, or `magi consult` directly), in which case
it is an object with these keys, **all always present**:

| Field | Type | Meaning |
|---|---|---|
| `report` | string | the (possibly annotated, possibly truncated) verdict text |
| `degraded` | bool | fewer than 3 agents responded — consensus may be unreliable |
| `mode` | string | the effective mode: `code-review` / `design` / `analysis` |
| `mode_source` | string | how it was resolved: `explicit` / `configured` / `agent-chosen` / `inferred` / `default` (see [Mode routing](usage.md#mode-routing)) |
| `extraction_failures` | object | per-seat list of `{model, attempt, cause}`; an empty object certifies every seat adhered to the verdict contract. The `cause` field uses the crate's own serialized names: `missing-markers`, `unterminated`, `ambiguous`, `invalid-json`, `malformed-object`, `schema`, `echoed-example`, `agent-identity`; the set is open on magi-core's side — a value not listed here is a newer cause, spelled as the crate spells it |
| `input_size` | object | `{estimated_tokens, warn_threshold, exceeded}` — always all three sub-keys, even when unmeasured |
| `report_truncated` | string | `none` / `structural` / `anchored` / `bytes` — which guarantee survived truncation, never a bare boolean |
| `endpoint_divergence` | bool | whether this run's content passed through the principal provider (mode classification) before reaching a trio on a different endpoint |
| `timeout_below_formula` | bool | whether an explicit `--timeout` was below what the derived escalation formula requires |
| `failed_agents` | object | per-seat failure cause, redacted, for a mage that produced no verdict |
| `rotations` | array | one entry per seat **that actually rotated** — from/to lineage, cause, whether that cause was local to the mage or condemned the run (`mage_local`), and the model it ended on. Empty certifies that nobody did; it is a positive statement, not silence. (`magi-core`'s own report carries a row for every seat, rotated or not; this field is the filtered view.) |
| `ran_unmeasured` | array | seats that ran without a measured context window, so their verdict carries that caveat |
| `completions` | object | per seat, one record per completion **attempt**: the model, the cap in force, the token counts, how it finished, its **reasoning** state, and the reasoning **control** it was sent with (v0.20.0). `length` means the model ran out of output budget; a genuinely empty answer looks different, and the two want opposite fixes. `reasoning` is `"NotMeasured"` / `{"Measured": {...}}` / `{"Unsupported": {...}}` (never a bare number), and its `text` is always `null`; an attempt the client timeout killed before it could finish carries no reasoning measurement. See [`docs/REASONING-BUDGET.md`](REASONING-BUDGET.md) |
| `pool_eligibility` | object | per seat, the fallback candidates it could **not** have rotated into and why. Present even when nothing was rejected — an absent map means the snapshot was never computed, which is a different fact from an empty one |

New fields are added to `consult` without a `schema_version` bump: the same
consumer contract above applies to it.

### `--structured-verdicts` (opt-in)

`magi consult --output-format json --structured-verdicts` adds two more keys, for a machine
consumer that wants the trio's verdicts typed rather than rendered:

| Field | Type | Meaning |
|---|---|---|
| `agents` | array | one entry per seat that produced a verdict, each with `agent`, `verdict`, `confidence`, `summary`, `reasoning`, `findings`, `recommendation`; each finding with `severity`, `title`, `detail`, `file`, `line`, `category` (`file`/`line` null when the seat did not locate it) |
| `consensus` | object | magi-core's own `consensus`, `consensus_verdict`, `confidence`, `score`, `agent_count`, `votes`, `dissent`; confidence is the emitted side's confidences summed, divided by the agent count, scaled by `(|score| + 1) / 2`, clamped to `[0, 1]` and rounded to two decimals; dissent is an array of `agent`, `summary`, `reasoning` — the seats whose effective verdict differs from the emitted one; empty when unanimous |

Both are **always present when the flag is passed**, empty array included: an empty `agents`
certifies that no seat completed. Without the flag, neither appears.

**`category` is an OPEN string set, not a closed enum.** magi-core may add a category in a minor
release, so a strict validator should treat an unknown value as a new category rather than a schema
violation. The other enum-valued fields (`agent`, `verdict`, `severity`) are closed by upstream
contract and will not gain values without a breaking change.

**`consensus` is a documented SUBSET**: the seven fields in the table above. `findings`,
`conditions` and `recommendations` are not forwarded, and neither is the deprecated
`majority_summary`: the key exists so a consumer that computes its own consensus can contrast the
headline result, `dissent` carries the contrast data, and the per-finding material is already in
`agents`. Ask if you need them.

**`report_truncated` describes `report` alone.** With the flag on, `report` is still bounded by the
tool-result cap while `agents` and `consensus` are emitted in full, so a `report_truncated` other
than `none` says nothing about the structured keys, which are always complete. Passing the flag
therefore makes stdout unbounded; bound it on your side if that matters.

**The shape varies by FLAG, never by DATA.** That is what keeps the always-present rule intact: a
key that came and went with the outcome would break a strict schema, while a key you asked for by
name gives you the same shape on every run.

The flag exists on `consult` only: `magi query --structured-verdicts` is a parse error, not an
accepted no-op, and it requires `--output-format json`; with text output it exits **2**.

It is opt-in rather than default because the same text already travels rendered in `report`, and
`report` is bounded by the tool-result cap. The agent-facing `/consult` tool never emits these
keys at all: returning the trio's full output there would put it straight into the agent's context
window, which is what that cap exists to prevent.

### `applied_caps`

`applied_caps` reports the limits actually enforced on this run, so a caller can tell an operator
cap from a request that was simply honored as asked. It carries ten keys, all always present:

| Field | Type | Meaning |
|---|---|---|
| `max_tool_calls` | number | the effective tool-invocation cap after applying the operator cap |
| `max_tool_calls_clamped` | bool | `true` if the envelope's `max_tool_calls` was cut down to that cap |
| `timeout_secs` | number \| null | the wall-clock cap in seconds, if one was set |
| `system_override_applied` | bool | `true` if the envelope's `system` prompt was applied (requires the operator flag) |
| `system_ignored` | bool | `true` if the envelope carried a non-blank `system` that was ignored because the operator flag was off. Never `true` together with `system_override_applied` |
| `operation_budget_secs` | number | the per-attempt budget each mage ran under during this run |
| `ceiling_floored` | bool | `true` if the derived ceiling was raised to the 15 s floor. Strictly "raised": a ceiling that lands exactly on the floor was reached, not clamped, and reports `false` |
| `floor_activation_threshold_secs` | number | the smallest `--timeout` that avoids the floor under this run's rotation settings. This is not "the minimum for the run to succeed": a script that retries at this value can still exhaust its budget against a slow model. When it comes back unreasonably large, the fix is to lower `max_rotations`, not to raise `--timeout` |
| `max_rotations_effective` | number | the rotation count the budget was computed with, so a caller can weigh that second lever without redoing the arithmetic |
| `ceiling_above_sanity` | bool | `true` only when an explicit `--timeout` derived a ceiling above the sanity threshold (2400 seconds as of v0.21.0, raised from 600 so the recommended gate `--timeout 16820` — which derives 2335 — does not itself trigger it), which usually means a mistyped `--timeout`. Never `true` on the configured/TUI path, however large `agent_timeout_secs` is: there is no `--timeout` to have mistyped. Worth watching in CI, where stderr is unreliable and this flag is the only signal |

`system_ignored` is new in v0.25.0. The five keys after it are new in v0.15.0. All are additive under the same policy as `consult` above, with no
`schema_version` bump: the tolerate-new-fields rule applies here too.

## Authorization tiers

Secure by default: a read-only CI job cannot mutate or execute:

| Tier | Flag | Tools auto-approved | Notes |
|------|------|---------------------|-------|
| `default` | *(none)* | read-only: `ls` / `view` / `grep` | others are denied; the run continues and records the denial |
| auto | `--auto` | **all** registered tools (`edit`/`bash`/`consult`/`code-review`) | sandboxed |
| full-auto | `--full-auto` | all + elevated `max_tool_calls` (15 → 50) | **prints a WARNING** to stderr and the log; silences the soft repetitive-call guard and the normal cap |

The **hard barriers are always active in every tier**: the `bash` allowlist, the
shell-metacharacter ban, and `PathGuard` sandboxing. No flag relaxes them; even
`--full-auto` rejects `rm -rf /`, `$(...)`, and path traversal.

## State layout & the breaking change

All project state lives in a single **`.magi/`** directory: `magi.toml`, the
encrypted DB (`.magi-rs-memory.db`), and `logs/`. It is discovered by walking up from
the working directory (`-w`/cwd) to the nearest ancestor, `.git`-style.

> **BREAKING (v0.10.0):** loose legacy files in the current directory (a bare
> `.magi-rs-memory.db` / `magi.toml`) are **no longer read**. Migrate with one
> line: run **`magi init`**, then re-add secrets with **`magi vault set <NAME>`**
> (e.g. `ANTHROPIC_API_KEY`). Prior conversation history is not carried over.

> **BREAKING (v0.22.0):** the bundled SQLite engine moved from 3.45.0 to 3.53.2, and a
> `.magi-rs-memory.db` written by 0.21.0 or earlier is **not supported**. Before first use,
> delete or move aside `.magi/.magi-rs-memory.db` with its `-wal` and `-shm` files (keep
> `magi.toml` and `logs/`), then set a new passphrase and re-add secrets with
> **`magi-rs vault set <NAME>`**. History, knowledge, memories and every vault secret are lost;
> the [CHANGELOG](../CHANGELOG.md) has the full list.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | runtime / agent error (incl. `DbCorrupt`, wrong passphrase, timeout) |
| `2` | CLI misuse / invalid input (missing `prompt`, unknown field, over-cap input) |
| `3` | blocked by tier — an essential tool was denied and the run produced no response |

## Output is caller-sensitive

The rich JSON output, and `transcript[]` and `tool_calls[]` in particular, **echoes
tool output**. A `view` of a file that contains a token, or a `bash` command that
prints an environment secret, can surface in that echo. Magi guarantees only that
**its own** managed secrets (the passphrase and stored vault values) never appear;
it **cannot** guarantee the output is free of *non-vault* secrets. Treat the rich
output as sensitive.

> **Passphrase hygiene:** prefer `MAGI_PASSPHRASE` or the interactive prompt over
> `-p <passphrase>`; a value on `argv` is visible via `ps` / `/proc/<pid>/cmdline`
> to the same user (a deliberate, documented trade-off for CI).
