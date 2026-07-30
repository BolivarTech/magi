# Magi Agent — Terminal AI Assistant in Rust

[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![CI](https://github.com/BolivarTech/magi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BolivarTech/magi/actions/workflows/ci.yml)
[![Lints](https://img.shields.io/badge/lints-clippy%20clean-blue.svg)](https://github.com/rust-lang/rust-clippy)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![crates.io](https://img.shields.io/crates/v/magi-rs.svg?label=version)](https://crates.io/crates/magi-rs)

**Magi Agent** (`magi-rs`) is a terminal AI assistant in Rust, modeled on Claude Code. It drives an LLM provider through a multi-turn **tool loop** with sandboxed filesystem and shell access, and persists every conversation to a **locally-encrypted SQLite store**. Nothing leaves your machine except the model API calls you explicitly authorize.

---

## Why Magi?

Most AI coding agents are SaaS-bound and tied to a single vendor. Magi is built for the opposite constraint: **environments where source code cannot leave the machine.**

| Principle | What it means in Magi |
|-----------|-----------------------|
| **Local-first** | A single static Rust binary. Conversation history and project knowledge live in an encrypted SQLite file on disk — never a third-party vault. |
| **Encrypted at rest** | Every stored record is sealed with AES-256-GCM-SIV + error-correcting FEC under a random data key, itself wrapped by an Argon2id-derived key — via the audited [`cryptovault`](https://crates.io/crates/cryptovault) crate. The data key is unlocked by a **user passphrase** (zero-knowledge: nothing persisted opens the DB without it), and a wrong passphrase never wipes the DB. |
| **Sandboxed by default** | Every filesystem tool is confined to the workspace via `PathGuard`; the shell tool runs a strict command allowlist with a hard ban on shell metacharacters. |
| **Human-in-the-loop** | Each tool call must be approved inline in the TUI before it runs. |
| **Fails safe** | If no passphrase is available (headless with none supplied, or entry cancelled), the session degrades to ephemeral (no persistence) rather than ever encrypting with a constant key — the existing on-disk DB is left untouched. |

---

## Features

| Area | Capability |
|------|------------|
| **TUI front-end** | `ratatui` app with **Normal / Selection / Visual** modes, UTF-8-safe cursor handling, system clipboard integration |
| **Agent loop** | Multi-turn orchestration, bounded tool calls (max 15), repetitive-call detection, ANSI/control-char sanitization, interactive approval gate |
| **Provider** | Anthropic **Messages API** with SSE streaming + `tool_use` assembly and 429 retry; `StaticProvider` fallback when no key is configured |
| **Sandboxed tools** | `ls`, `view`, `edit`, `grep`, `bash` (strict allowlist), `project_knowledge` (persistent facts), `consult` (MAGI multi-perspective consensus) |
| **MAGI consult** | The agent can escalate hard, trade-off-heavy decisions to a 3-perspective consensus (Melchior / Balthasar / Caspar) via the `magi-core` crate — invoked autonomously through the tool loop (each call passes the approval gate) or forced with `/consult` |
| **Tiered memory (RAG)** | Embedding-indexed vector store (any openai-compat endpoint, default Ollama + `nomic-embed-text-v2-moe`), composite reranker (similarity + recency + salience), decay/eviction, always-injected preference profile, hard token-budget assembler — context bounded regardless of history depth |
| **Encrypted memory** | SQLite (WAL) with **one cached key derivation per session** (Argon2id + per-DB salt), AES-256-GCM-SIV, Reed-Solomon FEC; text **and** embedding vectors encrypted at rest |
| **OAuth login** | PKCE flow against the Anthropic Console (`/login` in the TUI); the minted key is stored in the vault |
| **Zero-knowledge vault** | A passphrase (never persisted) unlocks the DB; API keys and other secrets live in an encrypted `vault` table managed by `magi-rs vault {ls,set,rm,passwd}` — no OS keyring, no `key.txt` |

---

## Installation

### Prebuilt binaries

Every release ships ready-to-run archives on the [Releases page](https://github.com/BolivarTech/magi/releases) — no Rust toolchain required.

| Archive | Runs on |
|---|---|
| `…-windows-x86_64.zip` (or `.7z`) | Windows 10/11, x64 |
| `…-linux-x86_64.tar.gz` (or `.7z`) | Linux x64 with **glibc ≥ 2.35** — Ubuntu 22.04+, Debian 12+ |
| `…-linux-x86_64-compat.tar.gz` (or `.7z`) | Linux x64 with **any glibc ≥ 2.17** — RHEL/Rocky/Alma/CloudLinux 8 and 9, Debian 10+, older Ubuntu |
| `…-linux-aarch64-rpi5.tar.gz` (or `.7z`) | Raspberry Pi 5 / aarch64, glibc ≥ 2.35 |
| `…-macos-universal2.tar.gz` (or `.7z`) | macOS, Intel and Apple Silicon |

Every platform ships two archives with identical contents. **Prefer `.tar.gz` on
Unix and `.zip` on Windows** — both open with tools the OS already has:

```bash
tar -xzf magi-rs-*.tar.gz          # Linux/macOS, no extra package
```

The `.7z` is smaller but needs `p7zip`, which is *not* installed by default on
RHEL-family distros, minimal server images or most containers.

**Which Linux build?** Check what your system has:

```bash
ldd --version | head -1
```

2.35 or newer → either archive works; prefer `linux-x86_64`, built natively. Anything older → use `linux-x86_64-compat`. The two are the same program from the same source; they differ only in the glibc symbol versions they request, and `-compat` runs everywhere the other does. If you pick the wrong one the binary refuses to start with `GLIBC_2.xx not found` — that is the only symptom, and the fix is to download `-compat`.

Verify what you downloaded against the `SHA256SUMS.txt` published alongside it:

```bash
sha256sum -c --ignore-missing SHA256SUMS.txt      # Linux/macOS
Get-FileHash .\magi-rs-*.zip -Algorithm SHA256    # Windows
```

### Prerequisites

- A Rust toolchain (stable, edition 2021) — install via [rustup](https://rustup.rs/). **Only needed to build from source**; the prebuilt archives above run as-is.
- `ripgrep` (`rg`) on `PATH` for the `grep` tool — required either way.
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com/) (**recommended**), set via `ANTHROPIC_API_KEY` or stored in the vault (`magi-rs vault set ANTHROPIC_API_KEY`). `/login` (OAuth) also works but is **best-effort** — see [Configuration](#configuration).

### Build from source

```bash
git clone https://github.com/BolivarTech/magi.git
cd magi
cargo build --release          # binary at target/release/magi-rs[.exe]
```

### Run

```bash
cargo run                      # launch the TUI (prompts for the vault passphrase)
cargo run -- --logout          # unlock the vault and remove ANTHROPIC_API_KEY, then exit
```

---

## Usage

Launch the TUI and type a prompt. The agent streams its reply, and any tool it wants to run pauses for your approval.

### TUI modes

| Mode | Enter with | Keys |
|------|-----------|------|
| **Normal** | (default) | type to chat · `Ctrl+S` → Selection · `Ctrl+V` paste · `Ctrl+C` copy selection / quit if none |
| **Selection** | `Ctrl+S` | `↑/↓` pick a message · `y` copy whole message · `Enter` → Visual |
| **Visual** | `Enter` (in Selection) | `Shift+←/→` character-level selection · `y` copy |

### Tool approval

When the agent requests a tool, an inline prompt appears: **`y`** approves, **`c`** / **`Esc`** denies. The agent blocks up to 5 minutes waiting for your response.

### Slash commands (handled by the UI, never sent to the model)

| Command | Effect |
|---------|--------|
| `/login` | Start the OAuth (PKCE) login flow — **best-effort**, may be rate-limited (see Configuration); prefer an API key |
| `/logout` | Clear stored API keys |
| `/clear` | Clear the on-screen conversation |
| `/consult <question>` | Force a MAGI 3-perspective consensus on the question (≈ 3 model calls). Blocks the session while it runs, like a normal turn. Requires a configured LLM provider. |
| `/init-config` | Scaffold a `magi.toml` in the workspace pre-filled with the built-in Ollama-first defaults (refuses to overwrite an existing file). Also available as `magi-rs --init-config`. |
| `/help` | Show available commands |
| `/exit`, `/quit` | Leave the app |

### Tools

| Tool | Purpose |
|------|---------|
| `ls` | List directory contents (sandboxed) |
| `view` | Read file contents (sandboxed) |
| `edit` | Create / overwrite a file (sandboxed) |
| `grep` | RipGrep-backed search (sandboxed) |
| `bash` | Run a shell command — **strict allowlist** |
| `project_knowledge` | Persist project facts to encrypted memory |
| `consult` | Run a MAGI 3-perspective consensus on a hard decision (via `magi-core`) |

The `bash` allowlist is: `ls git npm cargo rg cat echo pwd grep mkdir touch find rm diff node python pytest` (with `cargo` restricted to `test` / `build` / `check`), plus a hard ban on the shell metacharacters `` | & ; > < ` $ ( ) { } \ ``.

### MAGI consult (multi-perspective consensus)

For decisions with genuine trade-offs — architecture choices, "should we X vs Y given these constraints?", risk calls — Magi can run a **three-perspective consensus** built on the [`magi-core`](https://crates.io/crates/magi-core) crate: three independent analyst agents (Melchior the scientist, Balthasar the pragmatist, Caspar the critic) evaluate the question and a consensus is synthesized.

- **Automatic (transparent).** The agent decides on its own when a question warrants multi-perspective analysis and invokes the `consult` tool through the normal tool loop. Like any tool call it passes the **inline approval gate** (`y`/`n`) — which is also your cost control, since a consult is ≈ **3 model calls**. Deny it to keep the single-LLM answer.
- **Forced.** Type `/consult <question>` to run the consensus directly, bypassing the router and the approval gate. The session shows `MAGI deliberating — 3 model calls…` and then renders the verbatim report (the three perspectives + the consensus verdict). `/consult` blocks the session while it runs, like a normal turn.
- **Backend.** The consult reuses the **same provider and credentials** already resolved for the agent (Anthropic or any OpenAI-compatible endpoint) — no second config. It is unavailable in `StaticProvider` mode (no API key): `/consult` then reports that a provider is required, and the tool is not registered.
- **Model capability.** Weak / small local models (e.g. Ollama `phi4-mini`) may fail to emit the strict per-agent JSON the consensus requires; the result is then marked `[DEGRADED: …]` (fewer than three agents responded). A capable model is recommended for reliable consensus.

---

## Headless mode

Magi runs non-interactively as a **CI/CD pipeline step** or as an **AI backend**
for another application. It composes the Unix way: read from stdin / `-i`, write
to stdout / `-o`, send diagnostics to stderr, and return a meaningful exit code.
Running `magi` with no subcommand still launches the TUI, unchanged.

### Subcommands

```bash
# Scaffold the .magi/ state directory (config, encrypted DB, logs).
# With -p / MAGI_PASSPHRASE it also bootstraps an empty vault so later
# headless runs need no interaction. Refuses to overwrite an existing .magi/.
magi init -p "correct horse battery staple"

# Prompt from stdin, streamed text answer to stdout.
echo "explain the retry logic in provider.rs" | magi query

# Rich JSON envelope in, one buffered JSON object out.
magi query -i q.json -o out.json --output-format json

# Force a MAGI 3-perspective consensus on the prompt directly.
magi consult -i decision.txt

# Read-only structural probe of the .magi/ DB — never unlocks, never prints a secret.
magi vault diagnose
```

**Input** is auto-detected: a JSON object with a top-level `prompt` string is a
rich **envelope** (`{prompt, system?, model?, provider?, max_tool_calls?,
consult?}`); anything else is a plain-text prompt. CLI flags win over envelope
fields, and the operator's `magi.toml` caps (e.g. `max_tool_calls`) are a ceiling
the envelope cannot exceed. `--input-format` / `--output-format` force the
interpretation.

### Authorization tiers

Secure by default — a read-only CI job cannot mutate or execute:

| Tier | Flag | Tools auto-approved | Notes |
|------|------|---------------------|-------|
| `default` | *(none)* | read-only: `ls` / `view` / `grep` | others are denied; the run continues and records the denial |
| auto | `--auto` | **all** registered tools (`edit`/`bash`/`consult`) | sandboxed |
| full-auto | `--full-auto` | all + elevated `max_tool_calls` (15 → 50) | **prints a WARNING** to stderr and the log; silences the soft repetitive-call guard and the normal cap |

The **hard barriers are always active in every tier** — the `bash` allowlist, the
shell-metacharacter ban, and `PathGuard` sandboxing. No flag relaxes them; even
`--full-auto` rejects `rm -rf /`, `$(...)`, and path traversal.

### State layout & the breaking change

All project state lives in a single **`.magi/`** directory — `magi.toml`, the
encrypted DB (`.magi-rs-memory.db`), and `logs/` — discovered by walking up from
the working directory (`-w`/cwd) to the nearest ancestor, `.git`-style.

> **BREAKING (v0.10.0):** loose legacy files in the current directory (a bare
> `.magi-rs-memory.db` / `magi.toml`) are **no longer read**. Migrate with one
> line: run **`magi init`**, then re-add secrets with **`magi vault set <NAME>`**
> (e.g. `ANTHROPIC_API_KEY`). Prior conversation history is not carried over.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | runtime / agent error (incl. `DbCorrupt`, wrong passphrase, timeout) |
| `2` | CLI misuse / invalid input (missing `prompt`, unknown field, over-cap input) |
| `3` | blocked by tier — an essential tool was denied and the run produced no response |

### Output is caller-sensitive

The rich JSON output — in particular `transcript[]` and `tool_calls[]` — **echoes
tool output**. A `view` of a file that contains a token, or a `bash` command that
prints an environment secret, can surface in that echo. Magi guarantees only that
**its own** managed secrets (the passphrase and stored vault values) never appear;
it **cannot** guarantee the output is free of *non-vault* secrets. Treat the rich
output as sensitive (REQ-H15).

> **Passphrase hygiene:** prefer `MAGI_PASSPHRASE` or the interactive prompt over
> `-p <passphrase>`; a value on `argv` is visible via `ps` / `/proc/<pid>/cmdline`
> to the same user (a deliberate, documented trade-off for CI).

---

## Configuration

### API key & model discovery

Resolved in order, first hit wins (`env > vault` — there is no OS keyring or `key.txt` as of v0.9.0):

1. `ANTHROPIC_API_KEY` environment variable
2. The vault entry `ANTHROPIC_API_KEY` (stored via `magi-rs vault set ANTHROPIC_API_KEY`, unlocked by the passphrase)

The model is read from `ANTHROPIC_MODEL`, defaulting to `claude-sonnet-4-6`. With no key found, the agent falls back to `StaticProvider`. The OpenAI-compatible key (`OPENAI_API_KEY`) follows the same `env > vault` precedence.

**A standard API key is the recommended, supported path.** Create one at [console.anthropic.com](https://console.anthropic.com/) (with billing enabled) and set `ANTHROPIC_API_KEY` or store it with `magi-rs vault set ANTHROPIC_API_KEY`.

> **`/login` (OAuth) is best-effort.** It reuses Anthropic's Claude Code OAuth client to mint a key on your account — a flow intended for Anthropic's own clients — so it may be **rate-limited, throttled, or blocked** at any time and is not guaranteed. The minted key is written to the vault. Prefer a standard API key.

### The vault & passphrase (v0.9.0)

The DB and every secret are unlocked by a **user passphrase** — resolved as `-p <passphrase>` > `MAGI_PASSPHRASE` env var > interactive hidden prompt. On first run you create one (double entry; `zxcvbn` score ≥ 3 and ≥ 12 chars enforced, no override). **Zero-knowledge: nothing persisted opens the DB without the passphrase — forgetting it means the data is unrecoverable.** `-p`/`MAGI_PASSPHRASE` make headless/CI use and moving the `.db` between machines possible.

Secrets live in an encrypted `vault` table, managed by the CLI:

```bash
magi-rs vault ls                       # list secret NAMES + timestamps (never values)
magi-rs vault set ANTHROPIC_API_KEY    # value read from a hidden prompt or stdin, never argv
magi-rs vault rm  OPENAI_API_KEY       # delete (Y-only confirmation; -f to skip)
magi-rs vault passwd                   # rotate the passphrase (re-wraps the same data key, O(1))
```

There is **no `get`/`cat`/`show` command** — a stored value is never printed, by design.

### Default backend — Ollama-first (v0.6.0, BREAKING)

> **Breaking change in 0.6.0.** With **no `magi.toml` and no env vars**, Magi now
> defaults to a local **Ollama** backend (`provider = "openai"`,
> `base_url = http://localhost:11434/v1`, model `kimi-k2.6:cloud`, and the MAGI trio
> `qwen3.5:397b-cloud` / `gpt-oss:120b-cloud` / `deepseek-v4-pro:cloud`). Previously
> the no-config default was Anthropic.

**To use Anthropic instead**, set `provider = "anthropic"` in `magi.toml` **or**
`MAGI_PROVIDER=anthropic` — the Anthropic Messages API path (key discovery, model
default, `StaticProvider` fallback) is unchanged, just **opt-in** now.

To scaffold a `magi.toml` pre-filled with the built-in Ollama-first defaults, run
`magi-rs --init-config` (CLI) or the **`/init-config`** TUI command. Both refuse to
overwrite an existing `magi.toml`.

### `magi.toml` (optional, multi-backend)

Magi can talk to any **OpenAI-compatible** Chat Completions endpoint — a local
**Ollama** instance (the default), OpenAI itself, Groq, OpenRouter — in addition to
the opt-in Anthropic Messages API. The backend and its non-secret settings live in a
workspace-local `magi.toml`. A reference is committed as
[`docs/magi.toml.example`](docs/magi.toml.example); copy it to `magi.toml` and edit, or generate
it with `magi-rs --init-config`. `magi.toml` is gitignored.

```toml
provider = "openai"          # "openai" (default — Ollama) | "anthropic"

[openai]
base_url = "http://localhost:11434/v1"   # Ollama; or https://api.openai.com/v1, Groq, OpenRouter, …
model    = "kimi-k2.6:cloud"             # built-in default; override per endpoint

[anthropic]
model    = "claude-sonnet-4-6"           # optional override of the Anthropic default (opt-in path)
```

**Precedence (per setting): environment variable > `magi.toml` > built-in default.**

| Setting | Env var | `magi.toml` | Default |
|---------|---------|-------------|---------|
| Provider backend | `MAGI_PROVIDER` | `provider` | `openai` (Ollama) |
| OpenAI base URL | `OPENAI_BASE_URL` | `[openai].base_url` | `http://localhost:11434/v1` |
| OpenAI model | `OPENAI_MODEL` | `[openai].model` | `kimi-k2.6:cloud` |
| Anthropic model | `ANTHROPIC_MODEL` | `[anthropic].model` | see [API key & model discovery](#api-key--model-discovery) |

All built-in default literals live in one place — [`src/defaults.rs`](src/defaults.rs).

> **Known limitations of the Ollama-first defaults.**
> 1. The built-in defaults assume **Ollama**. If you point `provider = "openai"` at
>    real OpenAI (or another non-Ollama service) **without** setting `OPENAI_MODEL` /
>    `[openai].model` (and the `[magi]` trio), the defaults — `kimi-k2.6:cloud` and the
>    `:cloud` trio — will not exist there; set them explicitly.
> 2. The default `:cloud` model tags reflect the Ollama catalog at release time and may
>    rot as it changes. They are maintained in one place (`src/defaults.rs`); refresh per
>    release.

**API keys never live in `magi.toml`.** Keys come from env or the vault only — `magi.toml` is non-secret runtime config and is the wrong place for credentials. Specifically:

- **Anthropic key** — `ANTHROPIC_API_KEY` env var, or the vault entry of the same name (see above).
- **OpenAI-compatible key** — `OPENAI_API_KEY` env var, or the vault entry of the same name. For a local Ollama instance, magi-rs falls back to a dummy value (`"ollama"`) so you can run without setting anything.
- Placing `api_key` / `OPENAI_API_KEY` (or any other unknown field) inside `magi.toml` is rejected at parse time under `deny_unknown_fields`, not silently dropped — the file is treated as invalid and magi-rs falls back to defaults with a startup warning.
- A malformed `magi.toml` does not crash magi-rs: it falls back to defaults and surfaces a notice in the TUI on startup.

#### Per-agent MAGI models — `[magi]` (optional)

The three MAGI perspectives (Melchior / Balthasar / Caspar) used by `/consult` and the `consult` tool can each run a **different model**, giving the consensus genuine **lineage diversity** — different model families reason and fail differently, so agreement across them carries more signal. This is **opt-in**: with no `[magi]` section and no `MAGI_MODEL_*` env vars, all three share the principal model (identical to v0.4.0).

```toml
[magi]
melchior_model  = "qwen3:8b"       # Scientist  — theoretical analysis
balthasar_model = "gpt-oss:20b"    # Pragmatist — practical trade-offs
caspar_model    = "deepseek-r1:8b" # Critic     — adversarial review
```

| Setting | Env var | `magi.toml` | Default |
|---------|---------|-------------|---------|
| Melchior model | `MAGI_MODEL_MELCHIOR` | `[magi].melchior_model` | principal model |
| Balthasar model | `MAGI_MODEL_BALTHASAR` | `[magi].balthasar_model` | principal model |
| Caspar model | `MAGI_MODEL_CASPAR` | `[magi].caspar_model` | principal model |

Per-agent overrides **reuse the principal provider's `base_url` + API key** and change only the model name. So real cross-family diversity (e.g. GLM + GPT + DeepSeek) requires the principal backend to be an Ollama-style endpoint serving all three families; with an Anthropic principal you can still vary across Anthropic models (tier diversity). See [`docs/magi.toml.example`](docs/magi.toml.example) for the Light / Balanced / Maximum tier suggestions. A blank value is treated as unset. In `StaticProvider` mode (no API key) the overrides are ignored with a startup notice.

#### Quick start — local Ollama (no cost, no rate limits)

```bash
# 1. Install and start Ollama, then pull a model:
ollama pull phi4-mini

# 2. Drop a magi.toml in the workspace:
cp docs/magi.toml.example magi.toml
# (the defaults already point at http://localhost:11434/v1 and phi4-mini)

# 3. Run magi-rs — the startup banner reports the selected provider.
cargo run
```

To use OpenAI instead, edit `magi.toml` (`base_url = "https://api.openai.com/v1"`, pick a `model`) and set `OPENAI_API_KEY` in the environment.

---

## Tiered Memory (RAG)

Magi's memory subsystem replaces the naive "load the entire history into every prompt"
approach with semantic retrieval and principled forgetting. It is a full
Retrieval-Augmented Generation (RAG) pipeline — embed, retrieve, augment, generate —
implemented entirely in-process with encrypted local storage (no external vector DB).
Three pillars govern how context is built each turn:

| Pillar | What it means |
|--------|---------------|
| **P1 — Storage & retrieval** | Every persisted memory is indexed with an embedding via any OpenAI-compatible embedder (default: `nomic-embed-text-v2-moe:latest` on a local Ollama instance). Retrieval is semantic — top-k by cosine similarity, re-ranked by a weighted combination of similarity, recency, and salience. |
| **P2 — Timely forgetting** | A decay model (wall-clock half-life, access-reinforced with a saturation cap, salience-weighted) identifies obsolete memories. Memories below the strength threshold are archived or deleted; **preferences and high-salience facts are never evicted.** Superseded facts are soft-demoted immediately by the reranker and hard-excluded by the off-hot-path distiller. |
| **P3 — Bounded context recall** | The context assembler packs: system prompt → preference profile (always present) → ranked episodic recalls → current turn, all within a configurable token budget. Context size is bounded regardless of total history depth — no more O(N) prompt growth. |

**Default mode: `selective`** — validated by the built-in benchmark
(`cargo run --release --bin bench_memory`): same recall accuracy as the v0.6.0 "load all"
baseline at roughly **35% of context tokens**, with a lower staleness rate from superseded
facts.

The preference profile is always injected, cross-session: a distiller (driven by the
configured LLM) periodically promotes recurring preferences from episodic memory into a
compact, deduplicated profile with latest-wins semantics. Text **and** embedding vectors
are encrypted at rest via `CryptoVault`; the in-RAM index is never persisted in clear.

For the full technical reference see **[`docs/TIERED-MEMORY.md`](docs/TIERED-MEMORY.md)**.
For hands-on testing against a live backend see **[`docs/E2E-TESTING.md`](docs/E2E-TESTING.md)**.

### Enabling / configuring

The `selective` mode is the default. To tune or disable it, add a `[memory]` section to
your `magi.toml`:

```toml
[memory]
mode = "selective"              # selective (default) | load_all (v0.6.0 behavior)
context_budget_tokens = 8000
top_k = 12
decay_half_life_days = 30.0
distill_every_n_turns = 20

[embedding]
base_url = "http://localhost:11434/v1"            # Ollama default; point elsewhere for cloud
model = "nomic-embed-text-v2-moe:latest"          # dim auto-detected from first response
```

See [`docs/magi.toml.example`](docs/magi.toml.example) for all options with inline documentation.

### Rollback

To revert to the v0.6.0 behavior: set `mode = "load_all"` in `[memory]`. The agent will
stop using embeddings for context assembly and load the full history per turn (existing
`messages` table unchanged). To also purge the tiered-memory table: open
`.magi-rs-memory.db` with any SQLite client and run `DROP TABLE memories;` — this only
removes tiered-memory records; the `sessions`, `messages`, and `knowledge` tables are
unaffected.

---

## How It Works

```
            ┌───────────────────────────────┐
            │  main.rs                       │
            │  config discovery · passphrase │
            │  vault open · tool registration│
            └───────────────┬───────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼            channels           ▼
   ┌──────────────────┐ ◄───────────► ┌──────────────────┐
   │  TUI (ratatui)   │               │  Agent           │
   │  Normal/Sel/Vis  │               │  history · loop  │
   └──────────────────┘               │  approval gate   │
                                      └────────┬─────────┘
            ┌──────────────────────────────────┼──────────────────────────────────┐
            ▼                                  ▼                                   ▼
   ┌──────────────────┐             ┌──────────────────┐               ┌──────────────────┐
   │  Provider        │             │  Tools           │               │ EncryptedSqlite  │
   │  Anthropic (SSE) │             │  ls/view/edit    │               │  CryptoVault     │
   │  Static fallback │             │  grep/bash/know. │               │ Argon2→GCM-SIV→RS│
   └──────────────────┘             └──────────────────┘               └──────────────────┘
```

### The agent loop

1. The user message is appended to history (and persisted to encrypted memory if attached).
2. `provider.stream_messages(...)` streams `TextDelta` chunks to the TUI; each chunk is sanitized of ANSI / control characters.
3. The assembled assistant `Message` is collected on `MessageDone`.
4. If it contains `ToolUse` blocks, each tool is approved, executed, and its result pushed back as a `User`-role `ToolResult` — then the loop repeats.
5. With no tool requested, the final assistant text is returned.

**Invariants:** at most 15 tool calls per query; three consecutive identical tool calls abort ("repetitive tool call detected"); each tool call needs UI approval within 5 minutes or it is denied.

---

## Security Model

- **Encrypted memory (envelope DEK/KEK).** A random 32-byte **data key (DEK)** encrypts every record via the audited [`cryptovault`](https://crates.io/crates/cryptovault) crate (Argon2id → AES-256-GCM-SIV → error-correcting FEC, `#![forbid(unsafe_code)]`). The user **passphrase** plus a per-DB salt derives a **KEK** (Argon2id, OWASP 2025: 64 MiB, t=3, p=4) that **wraps** the DEK; `vault_meta` stores `{salt, wrapped_dek}`, each FEC-encoded. The DEK is unwrapped **once per session** and held **masked in RAM** (`MaskedDek`: XOR mask rotated from `OsRng` on every access, best-effort `mlock`) — no Argon2 runs per record.
- **Zero-knowledge unlock.** Nothing persisted opens the DB without the passphrase — no OS keyring, no cached file, no copy of the DEK on disk (`system::secrets` and the `keyring` dependency were removed in v0.9.0). The passphrase resolves as `-p` > `MAGI_PASSPHRASE` > interactive prompt; a hard strength floor (`zxcvbn` ≥ 3, ≥ 12 chars, no override) guards against offline brute-force of a portable `.db`. Forgetting the passphrase means the data is unrecoverable — there is no backdoor.
- **Never loses your data.** A wrong passphrase or a corrupt wrapped key fails with a typed, **retryable** error and **never wipes** the database — only a brand-new or pre-envelope DB (no wrapped key present) is bootstrapped fresh. `vault_meta` is FEC-protected, so on-disk bit-rot is corrected before the unwrap; corruption beyond the FEC's reach surfaces as a distinct `VaultMetaCorrupt` error, never a silent wrong key.
- **Secrets separation.** The passphrase (which unlocks the DEK) and the stored API keys (entries *inside* the vault) are different secrets in different places: rotating a stored API key never requires re-keying the passphrase, and a wrong API key never invalidates the local conversation DB. `magi-rs vault passwd` rotates the passphrase without re-encrypting any record (it re-wraps the same DEK).
- **Filesystem sandbox.** Every file-touching tool canonicalizes its target and validates it against the workspace root via `PathGuard` (handling Windows `\\?\` verbatim prefixes, null-byte attacks, and lexical normalization).
- **Shell sandbox.** The `bash` tool enforces a per-binary argument allowlist and bans shell metacharacters to prevent subshell injection on both PowerShell and bash.

---

## Project Structure

```
src/
  main.rs              -- config discovery, passphrase resolution / vault open, tool registration, entry
  lib.rs               -- library root (pub mod vault) so fuzz / coverage targets can link
  config.rs            -- MagiConfig (magi.toml load + provider/model resolution)
  defaults.rs          -- single source of truth for all built-in default literals
  agent/
    mod.rs             -- Agent orchestrator: multi-turn tool loop + approval gate
    provider.rs        -- Provider trait; AnthropicProvider (SSE) + OpenAiCompatibleProvider + StaticProvider
  memory/
    mod.rs             -- subsystem facade: MemoryKind, public re-exports (recall, assemble_selective)
    store.rs           -- SqliteVectorStore: encrypted vector store (memories table)
    embedding.rs       -- EmbeddingProvider trait + OpenAiCompatibleEmbedder
    retrieval.rs       -- recall() public API (D-13/B1 seam) + composite reranker
    decay.rs           -- strength model, run_forgetting, enforce_size_cap
    context.rs         -- assemble_selective: token-budget context assembler
    profile.rs         -- preference distiller + render_profile (always-injected)
    clock.rs           -- Clock trait (SystemClock / FixedClock for determinism)
    config.rs          -- MemoryConfig + EmbeddingConfig (deny_unknown_fields)
    salience.rs        -- deterministic salience heuristic at write time
    tokens.rs          -- estimate_tokens, budget_after_margin
    index.rs           -- BruteForceIndex (exact cosine) + InstantDistanceIndex (--features ann)
    error.rs           -- MemoryError + EmbeddingError (thiserror)
  bin/
    bench_memory.rs    -- two-arm benchmark binary (cargo run --bin bench_memory)
    bench_vault_crypto.rs -- FEC/decrypt performance baseline (REQ-V36)
  tools/
    mod.rs             -- Tool trait + ToolError + requires_approval()
    ls.rs read.rs write.rs grep.rs bash.rs knowledge.rs consult.rs
  system/
    database.rs        -- EncryptedSqliteMemory (MemoryStore) over SQLite; opens via the vault envelope
    fs.rs grep.rs path_guard.rs
  vault/
    mod.rs             -- SecretStore boundary + public re-exports (frontier main.rs sees)
    envelope.rs        -- DEK/KEK envelope: bootstrap/open, wrap/unwrap/rekey, keyless FEC over vault_meta
    memguard.rs        -- MaskedDek: DEK masked in RAM (OsRng per-access rotation), harden_process
    master.rs          -- passphrase resolution (-p > env > prompt), zxcvbn strength floor
    store.rs           -- VaultStore: the `vault` table + SecretStore CRUD (one blob per secret)
    cli.rs             -- `magi-rs vault {ls,set,rm,passwd}` (hidden prompt / stdin, Y-only confirm)
    error.rs           -- VaultError (thiserror): WrongPassphrase / VaultMetaCorrupt / ...
  services/
    oauth.rs           -- OAuth PKCE login (callback on 127.0.0.1:54545)
  tui/
    mod.rs             -- ratatui app (Normal / Selection / Visual)
docs/
  OVERVIEW.md          -- what Magi is, magi-core foundation, multi-perspective philosophy
  TIERED-MEMORY.md     -- tiered agnostic memory: full technical reference
  E2E-TESTING.md       -- hands-on testing guide (running application)
  magi.toml.example    -- annotated reference config (committed; magi.toml itself is gitignored)
Cargo.toml             -- edition 2021, MIT OR Apache-2.0
```

---

## Testing

```bash
cargo test                 # all unit + integration tests
cargo nextest run          # same, via nextest (preferred)
cargo test <name>          # single test by substring
cargo test -- --nocapture  # show println! output
```

Full verification (matches the project's per-phase gate):

```bash
cargo nextest run
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo build --release
cargo doc --no-deps
cargo audit
```

> Some tests use real OS resources: the OAuth callback server binds port `54545` — avoid two interactive `/login` flows at once. DB/crypto/vault tests are isolated via `tempfile` plus injected connections and passphrase-prompt doubles — no OS keyring is touched anywhere (`system::secrets` was removed in v0.9.0).

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Rust toolchain (stable, edition 2021) | Yes | via [rustup](https://rustup.rs/) |
| `ripgrep` (`rg`) | For the `grep` tool | on `PATH` |
| Vault passphrase | For persistence | Chosen on first run (or supplied via `-p`/`MAGI_PASSPHRASE`). Without one available the session runs ephemeral (the on-disk DB is left untouched) |
| [Ollama](https://ollama.com/) (default backend) | For live replies (default) | running daemon + a chat model (e.g. `kimi-k2.6:cloud`); run `ollama signin` for `:cloud` tags. Any OpenAI-compatible endpoint works by pointing `base_url` elsewhere |
| Embedding model | For tiered memory (`selective`, the default) | `ollama pull nomic-embed-text-v2-moe:latest`; without it memory degrades gracefully to text-only persistence |
| Anthropic API key | Optional (opt-in) | only with `provider = "anthropic"`; via env var, the vault, or `/login` |

### Default models (Ollama)

These are the built-in defaults used with no `magi.toml` (Ollama-first). The `:cloud`-tagged models run on
Ollama's cloud — run `ollama signin` once (no local weight download); the embedding model is pulled locally.
Override any of them per-section in `magi.toml` (`[openai]`, `[embedding]`, `[magi]`) with local equivalents.

| Role | Default model | Used by |
|------|---------------|---------|
| Chat (principal) | `kimi-k2.6:cloud` | `magi-rs` agent — live replies |
| Embedding | `nomic-embed-text-v2-moe:latest` | `magi-rs` tiered memory (`selective`) — `ollama pull` it |
| Melchior (Scientist) | `qwen3.5:397b-cloud` | `magi-core` multi-perspective consensus (`consult` / `/magi`) |
| Balthasar (Pragmatist) | `gpt-oss:120b-cloud` | `magi-core` multi-perspective consensus (`consult` / `/magi`) |
| Caspar (Critic) | `deepseek-v4-pro:cloud` | `magi-core` multi-perspective consensus (`consult` / `/magi`) |

> The MAGI trio deliberately runs three distinct model families (Alibaba / OpenAI / DeepSeek) for genuine
> cross-lineage diversity. The `consult` tool (and the `/magi` command) only need these when a
> multi-perspective analysis is requested.

---

## Documentation

- [`docs/OVERVIEW.md`](docs/OVERVIEW.md) — what Magi is, the `magi-core` foundation, and the multi-perspective philosophy behind the name.
- [`docs/TIERED-MEMORY.md`](docs/TIERED-MEMORY.md) — full technical reference for the tiered agnostic memory subsystem: RAG pipeline, three pillars, architecture, configuration, benchmark.
- [`docs/E2E-TESTING.md`](docs/E2E-TESTING.md) — hands-on end-to-end testing guide for the tiered memory feature (cross-session recall, preferences, rollback).

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Credits

The **Magi** name comes from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995, Hideaki Anno / Gainax), where three supercomputers — Melchior, Balthasar, and Caspar — govern critical decisions through structured consensus. The same multi-perspective philosophy backs the sibling [`magi-core`](https://crates.io/crates/magi-core) crate, which Magi Agent integrates for its planned multi-perspective `consult` workflow.
