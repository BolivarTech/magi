# Configuration

## API key & model discovery

Resolved in order, first hit wins (`env > vault`; there is no OS keyring or `key.txt` as of v0.9.0):

1. `ANTHROPIC_API_KEY` environment variable
2. The vault entry `ANTHROPIC_API_KEY` (stored via `magi-rs vault set ANTHROPIC_API_KEY`, unlocked by the passphrase)

The model is read from `ANTHROPIC_MODEL`, defaulting to `claude-sonnet-4-6`. With no key found, the agent falls back to `StaticProvider`.

The **OpenAI-compatible key** (`OPENAI_API_KEY`), used by the OpenAI / Ollama / OpenRouter / Groq provider *and* by the embedder, resolves by the same two steps and is stored the same way:

1. `OPENAI_API_KEY` environment variable
2. The vault entry `OPENAI_API_KEY` (stored via `magi-rs vault set OPENAI_API_KEY`)

For a local Ollama daemon neither is needed: magi-rs falls back to a dummy value. Any authenticated endpoint (real OpenAI, OpenRouter, Groq) needs one, and will fail with a clear `401` if it is absent from both places.

**Each section that can have its own endpoint has its own key.** The full rules are in [docs/keys.md](keys.md).

| Section | Key |
|---|---|
| main model | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, depending on `provider` |
| `[magi]` (the trio) | `MAGI_API_KEY` |
| `[embedding]` | `EMBEDDING_API_KEY` |
| `[decision]` | `DECISION_API_KEY` |
| `[code_review]` | `CODE_REVIEW_API_KEY` |

Each one resolves env var first, vault entry second (`magi-rs vault set MAGI_API_KEY`, and so on). When a section has no key of its own, it inherits `OPENAI_API_KEY` only if its endpoint has the same origin as the main one: same scheme, host and port, with the path ignored. A gateway that routes by path counts as one origin and therefore receives the main key; declare the section's own key to avoid that. On any other origin no key is sent, so the main key never reaches a host you did not point it at. An endpoint that needs a key then answers `401` or `403`, and the message names the variable to set.

`MAGI_API_KEY` also applies when `[magi].kind = "anthropic"`: the seats use it if set and `ANTHROPIC_API_KEY` otherwise. It must be a key for the provider of `[magi].kind`.

### Upgrading to 0.24.0

Before 0.24.0 the trio and the embedder always received `OPENAI_API_KEY`, even with a `base_url` pointing elsewhere. Now they receive it only on the main endpoint's origin. After upgrading, a `401` or `403` from the trio or the embedder usually comes from one of three setups:

- The main endpoint is `http://localhost:11434` and `[magi]` or `[embedding]` uses `http://127.0.0.1:11434`, or the other way round.
- The same host on another port, such as a proxy next to the daemon.
- `provider = "anthropic"` with an OpenAI-compatible trio or embedder on an origin other than the root `base_url`.

The message of the error names the variable, and each withheld key leaves an `OPENAI_API_KEY was not sent to ...` entry in the day's log under `.magi/logs/`. To fix a section, store its own key with `magi-rs vault set MAGI_API_KEY` (trio) or `magi-rs vault set EMBEDDING_API_KEY` (embedder). See the [CHANGELOG](../CHANGELOG.md) for the full entry.

**A standard API key is the recommended, supported path.** Create one at [console.anthropic.com](https://console.anthropic.com/) (with billing enabled) and set `ANTHROPIC_API_KEY` or store it with `magi-rs vault set ANTHROPIC_API_KEY`.

> **`/login` (OAuth) is best-effort.** It reuses Anthropic's Claude Code OAuth client to mint a key on your account. That flow is intended for Anthropic's own clients, so it may be **rate-limited, throttled, or blocked** at any time and is not guaranteed. The minted key is written to the vault. Prefer a standard API key.

## The vault & passphrase (v0.9.0)

The DB and every secret are unlocked by a **user passphrase**, resolved as `-p <passphrase>` > `MAGI_PASSPHRASE` env var > interactive hidden prompt. On first run you create one (double entry; `zxcvbn` score ≥ 3 and ≥ 12 chars enforced, no override). **Zero-knowledge: nothing persisted opens the DB without the passphrase. Forget it and the data is unrecoverable.** `-p`/`MAGI_PASSPHRASE` make headless/CI use and moving the `.db` between machines possible.

**The `-p`/`--passphrase` value may begin with a `-` (v0.20.0).** `magi-rs -p --horse-battery-zz vault ls` takes the whole string after `-p` as the passphrase, so a passphrase that happens to start with a hyphen is never mistaken for another flag and never echoed back in an error.

**The strength floor applies to every envelope bootstrap, not only the interactive first run (v0.20.0).** A `.magi/` scaffolded by `magi init` with no `-p` has no envelope yet (see below); the same `zxcvbn` ≥ 3 / ≥ 12-characters floor that guards the interactive first run now also guards that DB's *first* open under any passphrase, on every path: the vault CLI, `--logout`, and headless `query`/`consult` included. A weak passphrase offered there is rejected (`WeakPassphrase`) rather than silently becoming the master secret, and the DB stays envelope-less until a strong one opens it. This does **not** apply to unlocking an *existing* envelope: opening an already-encrypted DB never re-checks strength.

### Running `init` without `-p`

`magi init` **never prompts**. It takes the passphrase from `-p` or `MAGI_PASSPHRASE`,
and *absent* is not an error: it is a valid outcome. Prompting would hang the command
in a Dockerfile or a CI step, waiting on input nobody is there to type.

So without a passphrase the command **succeeds and exits 0**, creating the full `.magi/`
(`magi.toml`, `logs/` and the database with its schema), but it stops short of deriving
the key that wraps the data key. `magi vault diagnose` reports the result:

```
envelope: absent
verdict: fresh
```

The envelope is created on the **first interactive run** instead, which is where you are
asked to choose a passphrase (entered twice; `zxcvbn` score ≥ 3 and ≥ 12 characters, with
no flag to skip it). From then on the DB is encrypted and every later open requires it.

Two consequences worth knowing:

- **An `init` without `-p` is not headless-ready.** A later `magi query` with no TTY and no
  `-p`/`MAGI_PASSPHRASE` fails closed with `PassphraseUnavailable` rather than reading
  stdin: that stream is reserved for a `vault set` *value*, never for the passphrase. For
  an image or a pipeline, use `magi init -p "…"`, which bootstraps the envelope in the same
  step.
- **`envelope: absent` is not corruption.** It is the legitimate `fresh` state. The
  neighbouring state, envelope *present* but not opening, is a different thing entirely:
  it reports `WrongPassphrase`, it is retryable, and it **never wipes the DB**. Under the
  envelope a wrong passphrase and a damaged `wrapped_dek` fail the same GCM-SIV tag, so
  wiping on failure would turn a typo into total data loss.

Secrets live in an encrypted `vault` table, managed by the CLI:

```bash
magi-rs vault ls                       # list secret NAMES + timestamps (never values)
magi-rs vault set ANTHROPIC_API_KEY    # value read from a hidden prompt or stdin, never argv
magi-rs vault set OPENAI_API_KEY       # same, for any OpenAI-compatible backend
magi-rs vault rm  OPENAI_API_KEY       # delete (Y-only confirmation; -f to skip)
magi-rs vault passwd                   # rotate the passphrase (re-wraps the same data key, O(1))
```

`set` **never takes the value as an argument**: it reads it from a hidden prompt
when a TTY is present, and from stdin when there is none. That makes it scriptable
without the secret ever reaching the process table or the shell history:

```bash
echo "$MY_KEY" | magi-rs vault set OPENAI_API_KEY -p "$MAGI_PASSPHRASE"
```

Without a TTY the passphrase must come from `-p` or `MAGI_PASSPHRASE`: stdin is
reserved for the *value*, so it is never read as the passphrase.

**Overwriting an existing name behaves differently in the two modes.** Interactively it
asks for a `Y`-only confirmation. Non-interactively it does not prompt: it **refuses**,
exiting non-zero with `destructive operation requires -f in non-interactive mode`. A
script that re-sets a name it may already hold must therefore pass `-f`:

```bash
echo "$MY_KEY" | magi-rs vault set OPENAI_API_KEY -f -p "$MAGI_PASSPHRASE"
```

There is **no `get`/`cat`/`show` command**. A stored value is never printed, by design.

Every one of them accepts `-w <dir>` to operate on a workspace other than the current
directory, on either side of the subcommand (`vault -w <dir> ls` and `vault ls -w <dir>`
are the same command).

**What `vault passwd` does not do.** It changes the passphrase, but the stored data is not
re-encrypted: the key that encrypts it stays the same, only the passphrase that unlocks that key
changes. That protects you if the old passphrase leaks. It does not protect you if someone already
took a copy of the data key. `magi-rs vault set --help` lists every entry the product reads from the
vault and what each one is for.

## Default backend — Ollama-first (v0.6.0, BREAKING)

> **Breaking change in 0.6.0.** With **no `magi.toml` and no env vars**, Magi now
> defaults to a local **Ollama** backend: `provider = "ollama"` (the now-retired `"openai"`
> value through v0.11.0), `base_url = http://localhost:11434/v1`, model `kimi-k2.6:cloud`, and
> the MAGI trio `glm-5.3:cloud` / `kimi-k2.6:cloud` / `deepseek-v4-pro:cloud`. Previously the
> no-config default was Anthropic.
>
> Melchior was `qwen3.5:397b-cloud` through 0.19.0, until Ollama retired that tag. Balthasar was
> `gpt-oss:120b-cloud` through 0.20.0, until measurement moved it onto the principal
> model, which now also holds that seat.

**To use Anthropic instead**, set `provider = "anthropic"` in `magi.toml` **or**
`MAGI_PROVIDER=anthropic`. The Anthropic Messages API path (key discovery, model
default, `StaticProvider` fallback) is unchanged, just **opt-in** now.

To scaffold a `magi.toml` pre-filled with the built-in Ollama-first defaults, run
**`magi init`** (see [Headless mode](headless.md#headless-mode)); it refuses to overwrite an
existing `.magi/`. There is no other scaffolder: `--init-config` and `/init-config`
were retired in v0.12.0.

## `magi.toml` (optional, multi-backend)

Magi can talk to any **OpenAI-compatible** Chat Completions endpoint: a local
**Ollama** instance (the default), OpenAI itself, Groq, OpenRouter. That is in addition to
the opt-in Anthropic Messages API. The backend and its non-secret settings live in
`.magi/magi.toml`. An annotated reference is committed as
[`docs/magi.toml.example`](magi.toml.example); copy it and edit, or generate a starting
point with `magi init`. `magi.toml` is gitignored.

```toml
provider = "ollama"          # ollama | openai-compat | anthropic

# System-wide default endpoint: used by the main agent, the MAGI trio and the
# embedder unless their own section overrides it.
base_url = "http://localhost:11434/v1"

[openai]
model = "kimi-k2.6:cloud"    # used by BOTH `ollama` and `openai-compat` — they share
                              # the completions protocol; only `ollama` is probeable

[anthropic]
model = "claude-sonnet-4-6"  # optional override of the Anthropic default (opt-in path)
```

**Precedence (per setting): environment variable > `magi.toml` > built-in default.**

| Setting | Env var | `magi.toml` | Default |
|---------|---------|-------------|---------|
| Provider backend (main agent) | `MAGI_PROVIDER` | `provider` | `ollama` |
| System endpoint | `OPENAI_BASE_URL` | `base_url` (root) | `http://localhost:11434/v1` |
| OpenAI/Ollama model | `OPENAI_MODEL` | `[openai].model` | `kimi-k2.6:cloud` |
| Anthropic model | `ANTHROPIC_MODEL` | `[anthropic].model` | see [API key & model discovery](#api-key--model-discovery) |
| MAGI trio backend | *(none)* | `[magi].kind` | inherits `provider` |
| MAGI trio endpoint | *(none)* | `[magi].base_url` | inherits root `base_url` |
| Embedder endpoint | *(none)* | `[embedding].base_url` | inherits root `base_url` |

`[magi].kind` and the two section-level `base_url` overrides have **no dedicated env
var**: they resolve from `magi.toml` only (or inherit), unlike the root-level
settings above. All built-in default literals live in one place:
[`src/defaults.rs`](../src/defaults.rs).

> **Breaking change from v0.11.0.** `base_url` used to live under `[openai].base_url`;
> that key no longer exists, and `provider = "openai"` split into `ollama` (keyless,
> local) and `openai-compat` (authenticated endpoints: OpenAI, Groq, OpenRouter). A
> `magi.toml` written for v0.11.0 or earlier fails to parse; startup prints a guided
> migration error naming every incompatibility in the file, with corrected lines ready
> to paste. Migrating straight from v0.10.x isn't supported; go through v0.11.0 first.

> **Known limitations of the Ollama-first defaults.**
> 1. The built-in defaults assume **Ollama**. If you point `provider = "openai-compat"` at
>    real OpenAI (or another non-Ollama service) **without** setting `OPENAI_MODEL` /
>    `[openai].model` (and the `[magi]` trio), those defaults will not exist there. `kimi-k2.6:cloud` and the
>    `:cloud` trio are Ollama tags. Set them explicitly.
> 2. The default `:cloud` model tags reflect the Ollama catalog at release time and may
>    rot as it changes. They are maintained in one place (`src/defaults.rs`); refresh per
>    release.

**API keys never live in `magi.toml`.** Keys come from env or the vault only. `magi.toml` is non-secret runtime config and is the wrong place for credentials. Specifically:

- **Anthropic key.** `ANTHROPIC_API_KEY` env var, or the vault entry of the same name (see above).
- **OpenAI-compatible key.** `OPENAI_API_KEY` env var, or the vault entry of the same name (`magi-rs vault set OPENAI_API_KEY`). For a local Ollama instance, magi-rs falls back to a dummy value (`"ollama"`) so you can run without setting anything.
- **Trio key.** `MAGI_API_KEY` env var, or the vault entry of the same name. **Embedder key.** `EMBEDDING_API_KEY`, same way. Both fall back to `OPENAI_API_KEY` only on the main endpoint's origin (see [docs/keys.md](keys.md)).
- Placing `api_key` / `OPENAI_API_KEY` (or any other unknown field) inside `magi.toml` is rejected at parse time under `deny_unknown_fields`, not silently dropped.
- **An authenticated `base_url` never carries a literal credential.** Use the
  `[user]`/`[password]` placeholders in its `userinfo`, resolved from the vault at use
  time, never written to disk in the clear:
  ```toml
  base_url = "https://[user]:[password]@host/v1"
  ```
  ```bash
  magi-rs vault set BASE_URL_USER
  magi-rs vault set BASE_URL_PASSWORD
  ```
  (`[magi].base_url` and `[embedding].base_url` resolve their own
  `MAGI_BASE_URL_USER`/`MAGI_BASE_URL_PASSWORD` and
  `EMBEDDING_BASE_URL_USER`/`EMBEDDING_BASE_URL_PASSWORD` vault entries when they
  declare their own endpoint.) A literal credential in `base_url` is rejected at
  startup as a configuration error, naming the fix; a declared placeholder with no
  matching vault entry fails closed, naming the entry and the command to create it.
- **A `magi.toml` that exists but does not parse now stops the process.** This is a
  change from earlier releases, which fell back to defaults with a warning: writing a
  config file is declaring an intent, and continuing on something else when that intent
  can't be honored is worse than stopping. An **absent** file is still a silent default,
  and an **empty** one parses as valid and yields defaults.

### Per-agent MAGI models — `[magi]` (optional)

The three MAGI perspectives (Melchior / Balthasar / Caspar) used by `/consult` and the `consult` tool can each run a **different model**, giving the consensus genuine **lineage diversity**. Different model families reason and fail differently, so agreement across them carries more signal. This is **opt-in**: with no `[magi]` section and no `MAGI_MODEL_*` env vars, all three share the principal model (identical to v0.4.0).

```toml
[magi]
melchior_model    = "qwen3:8b"       # Scientist  — theoretical analysis
melchior_lineage  = "alibaba"
balthasar_model   = "gpt-oss:20b"    # Pragmatist — practical trade-offs
balthasar_lineage = "openai"
caspar_model      = "deepseek-r1:8b" # Critic     — adversarial review
caspar_lineage    = "deepseek"
```

A seat that names a model must also name its lineage, the failure domain you consider that
model to belong to. It is never inferred, and since `enforce_diversity` defaults to `true` the
three have to differ. A seat left on the built-in model inherits the built-in lineage and owes
nothing.

| Setting | Env var | `magi.toml` | Default |
|---------|---------|-------------|---------|
| Melchior model | `MAGI_MODEL_MELCHIOR` | `[magi].melchior_model` | principal model |
| Balthasar model | `MAGI_MODEL_BALTHASAR` | `[magi].balthasar_model` | principal model |
| Caspar model | `MAGI_MODEL_CASPAR` | `[magi].caspar_model` | principal model |

Per-agent overrides change only the model name; the trio's `kind` and `base_url`, by default inherited from the main agent, or diverged with `[magi].kind`/`[magi].base_url` (see the precedence table above), are shared across all three seats. So real cross-family diversity (e.g. Qwen + GPT-OSS + DeepSeek) requires that backend to be an Ollama-style endpoint serving all three families; with an Anthropic backend you can still vary across Anthropic models (tier diversity). See [`docs/magi.toml.example`](magi.toml.example) for the full annotated reference. A blank value is treated as unset. If a seat can't be built (e.g. its backend has no resolvable credential), the trio is unavailable and the startup notice names which seat failed and why (see [Mode routing](usage.md#mode-routing) above).

### Tuning the trio — `[magi]` / `[magi.complexity]` (optional)

A handful of other `[magi]` keys are exposed deliberately narrow: an operator-tunable
subset of what `magi-core`'s builder offers, not the whole surface:

| Key | Purpose |
|-----|---------|
| `agent_timeout_secs` | Per-mage ceiling on the TUI path, and the fallback on the headless path when no explicit `--timeout` is given. Minimum 30 seconds, **no upper bound as of v0.20.0** (a reasoning model can legitimately need minutes); **default 2335 as of v0.21.0**, sized to cover the shipped trio's measured convergence, up from 90. That raises `magi consult`'s own no-`--timeout` deadline to roughly 16818 seconds and makes `magi query --auto`/`--full-auto` warn `below_formula` when no `--timeout` is given either — see [`docs/REASONING-BUDGET.md`](REASONING-BUDGET.md). When `--timeout` is given, headless derives the ceiling from it instead (see [`applied_caps`](headless.md#applied_caps) above). Either way, the two internal timeout layers (retry budget, per-request client timeout) are **derived** from the ceiling, not configured separately: no combination of settings can break the relation between them. |
| `max_query_bytes` | Input cap applied by magi-rs itself, before `magi-core` sees the payload — rejects rather than truncates, since a silently shortened payload would produce a verdict indistinguishable from a legitimate one. Sized for a real review diff (hundreds of KB), not the old 8 KiB limit. |
| `input_warn_tokens` | Threshold for the oversized-input warning. Left unset, it is **measured** by a startup probe against the smallest context window across the trio (only possible when the trio's `kind` is `ollama`, the only measurable one); declaring it overrides the measurement. |
| `retry_disabled` | Disables the trio's inherited retry, for a deployment where 2× the per-mage timeout is unacceptable. |
| `untrusted_content` | See [Mode routing](usage.md#mode-routing) above. |
| `reasoning` (v0.20.0) | `default` \| `disabled` \| `enabled`, whole trio, absent = `default`. See [`docs/REASONING-BUDGET.md`](REASONING-BUDGET.md). |
| `reasoning_spelling` (v0.20.0) | Only for `openai-compat` seats; no default. A rejected spelling returns HTTP 400 and condemns the whole lineage for that run; see the guide above before setting it. |
| `max_tokens` (v0.20.0) | Output cap per completion, per seat; absent = `65536` as of v0.21.0 (was `16384`). No upper bound of magi-rs's own; above the pinned model's own maximum, HTTP 400 condemns the lineage. |
| `reasoning_trace` (v0.20.0) | Off by default. `true` logs a bounded, redacted head/tail of the reasoning text for cut attempts only, to the log file at `INFO`. |

`tool_result_cap_bytes` (root-level, not under `[magi]`) bounds the consult report that
enters the conversation history, on all three routes: the TUI, `magi query` and `magi
consult`. This matters most in an interactive session, where the report is re-sent to
the model on every subsequent turn, so its cost is paid per turn rather than once.

`[magi.complexity]` sets the length thresholds (in **characters**, not bytes) below
which the complexity gate vetoes an autonomous consult; see
[Mode routing](usage.md#mode-routing) above. Absent, the built-in thresholds still apply; a
threshold set to `0` disables the veto for that mode only.

### Quick start — local Ollama (no cost, no rate limits)

```bash
# 1. Install and start Ollama, then pull a model:
ollama pull phi4-mini

# 2. Drop a magi.toml in the workspace:
cp docs/magi.toml.example magi.toml
# (the defaults already point at http://localhost:11434/v1 and phi4-mini)

# 3. Run magi-rs — the startup banner reports the selected provider.
cargo run
```

To use OpenAI instead, edit `magi.toml` (`base_url = "https://api.openai.com/v1"`, pick a `model`) and provide `OPENAI_API_KEY`, either in the environment or in the vault (`magi-rs vault set OPENAI_API_KEY`).

For **OpenRouter**, a verified configuration and the four traps that backend sets are documented in [`docs/OPENROUTER-BACKEND.md`](OPENROUTER-BACKEND.md), with a ready-to-copy [`docs/magi.toml.openrouter.example`](magi.toml.openrouter.example).
