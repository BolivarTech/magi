# Magi Agent — Terminal AI Assistant in Rust

[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-119%20passing-brightgreen.svg)](#testing)
[![Lints](https://img.shields.io/badge/lints-clippy%20clean-blue.svg)](https://github.com/rust-lang/rust-clippy)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Version](https://img.shields.io/badge/version-0.2.1-informational.svg)](CHANGELOG.md)

**Magi Agent** (`magi-rs`) is a terminal AI assistant in Rust, modeled on Claude Code. It drives an LLM provider through a multi-turn **tool loop** with sandboxed filesystem and shell access, and persists every conversation to a **locally-encrypted SQLite store**. Nothing leaves your machine except the model API calls you explicitly authorize.

---

## Why Magi?

Most AI coding agents are SaaS-bound and tied to a single vendor. Magi is built for the opposite constraint: **environments where source code cannot leave the machine.**

| Principle | What it means in Magi |
|-----------|-----------------------|
| **Local-first** | A single static Rust binary. Conversation history and project knowledge live in an encrypted SQLite file on disk — never a third-party vault. |
| **Encrypted at rest** | Every stored record is sealed with Argon2id → AES-256-GCM-SIV → Reed-Solomon FEC. The DB master key lives in the OS keyring, separate from the LLM API key. |
| **Sandboxed by default** | Every filesystem tool is confined to the workspace via `PathGuard`; the shell tool runs a strict command allowlist with a hard ban on shell metacharacters. |
| **Human-in-the-loop** | Each tool call must be approved inline in the TUI before it runs. |
| **Fails safe** | If the OS keyring is unavailable, the session degrades to ephemeral (no persistence) rather than ever encrypting with a constant key. |

---

## Features

| Area | Capability |
|------|------------|
| **TUI front-end** | `ratatui` app with **Normal / Selection / Visual** modes, UTF-8-safe cursor handling, system clipboard integration |
| **Agent loop** | Multi-turn orchestration, bounded tool calls (max 15), repetitive-call detection, ANSI/control-char sanitization, interactive approval gate |
| **Provider** | Anthropic **Messages API** with SSE streaming + `tool_use` assembly and 429 retry; `StaticProvider` fallback when no key is configured |
| **Sandboxed tools** | `ls`, `view`, `edit`, `grep`, `bash` (strict allowlist), `project_knowledge` (persistent facts) |
| **Encrypted memory** | SQLite (WAL) with **one cached key derivation per session** (Argon2id + per-DB salt), AES-256-GCM-SIV, Reed-Solomon FEC, salt-integrity self-heal |
| **OAuth login** | PKCE flow against the Anthropic Console (`/login` in the TUI) |
| **OS keyring** | API key in `magi-rs`, DB master key in `magi-rs-internal` (kept separate); `magi-rust` legacy names auto-migrated |

---

## Installation

### Prerequisites

- A Rust toolchain (stable, edition 2021) — install via [rustup](https://rustup.rs/).
- `ripgrep` (`rg`) on `PATH` for the `grep` tool.
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com/) (**recommended**), set via `ANTHROPIC_API_KEY` or `key.txt`. `/login` (OAuth) also works but is **best-effort** — see [Configuration](#configuration).

### Build from source

```bash
git clone https://github.com/BolivarTech/magi.git
cd magi
cargo build --release          # binary at target/release/magi-rs[.exe]
```

### Run

```bash
cargo run                      # launch the TUI
cargo run -- --logout          # clear stored API keys from the OS keyring and exit
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

The `bash` allowlist is: `ls git npm cargo rg cat echo pwd grep mkdir touch find rm diff node python pytest` (with `cargo` restricted to `test` / `build` / `check`), plus a hard ban on the shell metacharacters `` | & ; > < ` $ ( ) { } \ ``.

---

## Configuration

### API key & model discovery

Resolved in order, first hit wins:

1. `ANTHROPIC_API_KEY` environment variable
2. OS keyring, service `magi-rs`, key `ANTHROPIC_API_KEY`
3. OS keyring, service `magi-rust` (legacy) — auto-migrated to `magi-rs` on read
4. `key.txt` in the working directory: line 1 = API key, line 2 = optional model

The model is read from `ANTHROPIC_MODEL`, then `key.txt` line 2, defaulting to `claude-sonnet-4-6`. With no key found, the agent falls back to `StaticProvider`.

**A standard API key is the recommended, supported path.** Create one at [console.anthropic.com](https://console.anthropic.com/) (with billing enabled) and set `ANTHROPIC_API_KEY` or put it in `key.txt`.

> **`/login` (OAuth) is best-effort.** It reuses Anthropic's Claude Code OAuth client to mint a key on your account — a flow intended for Anthropic's own clients — so it may be **rate-limited, throttled, or blocked** at any time and is not guaranteed. Prefer a standard API key.

> `key.txt` and its variants are gitignored. Never commit a real key.

### `magi.toml` (optional, multi-backend)

Magi can talk to any **OpenAI-compatible** Chat Completions endpoint — OpenAI itself, a local **Ollama** instance, Groq, OpenRouter — in addition to the default Anthropic Messages API. The backend and its non-secret settings live in a workspace-local `magi.toml`. A reference is committed as [`magi.toml.example`](magi.toml.example); copy it to `magi.toml` and edit. `magi.toml` is gitignored.

```toml
provider = "openai"          # "anthropic" (default) | "openai"

[openai]
base_url = "http://localhost:11434/v1"   # Ollama; or https://api.openai.com/v1, Groq, OpenRouter, …
model    = "phi4-mini"                   # REQUIRED when provider = "openai"

[anthropic]
model    = "claude-sonnet-4-6"           # optional override of the Anthropic default
```

**Precedence (per setting): environment variable > `magi.toml` > built-in default.**

| Setting | Env var | `magi.toml` | Default |
|---------|---------|-------------|---------|
| Provider backend | `MAGI_PROVIDER` | `provider` | `anthropic` |
| OpenAI base URL | `OPENAI_BASE_URL` | `[openai].base_url` | `https://api.openai.com/v1` |
| OpenAI model | `OPENAI_MODEL` | `[openai].model` | *(none — required when `provider = "openai"`)* |
| Anthropic model | `ANTHROPIC_MODEL` | `[anthropic].model` | see [API key & model discovery](#api-key--model-discovery) |

**API keys never live in `magi.toml`.** Keys come from env / OS keyring / `key.txt` only — `magi.toml` is non-secret runtime config and is the wrong place for credentials. Specifically:

- **Anthropic key** — `ANTHROPIC_API_KEY` env var, OS keyring (`magi-rs`), or `key.txt` (see above).
- **OpenAI-compatible key** — `OPENAI_API_KEY` env var. For a local Ollama instance, magi-rs falls back to a dummy value (`"ollama"`) so you can run without setting anything.
- Placing `api_key` / `OPENAI_API_KEY` (or any other unknown field) inside `magi.toml` is rejected at parse time under `deny_unknown_fields`, not silently dropped — the file is treated as invalid and magi-rs falls back to defaults with a startup warning.
- A malformed `magi.toml` does not crash magi-rs: it falls back to defaults and surfaces a notice in the TUI on startup.

#### Quick start — local Ollama (no cost, no rate limits)

```bash
# 1. Install and start Ollama, then pull a model:
ollama pull phi4-mini

# 2. Drop a magi.toml in the workspace:
cp magi.toml.example magi.toml
# (the defaults already point at http://localhost:11434/v1 and phi4-mini)

# 3. Run magi-rs — the startup banner reports the selected provider.
cargo run
```

To use OpenAI instead, edit `magi.toml` (`base_url = "https://api.openai.com/v1"`, pick a `model`) and set `OPENAI_API_KEY` in the environment.

---

## How It Works

```
            ┌───────────────────────────────┐
            │  main.rs                       │
            │  config discovery · keyring    │
            │  master-key · tool registration│
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

- **Encrypted memory.** Argon2id (OWASP 2025 parameters: 64 MiB, t=3, p=4) derives a 32-byte key from the DB master password and a **per-database** random salt. The key is derived **once per session** and cached (`Zeroizing`) — no longer per record. AES-256-GCM-SIV (nonce-misuse resistant) provides authenticated encryption with a fresh `OsRng` nonce per record; Reed-Solomon FEC wraps `nonce‖ciphertext` against bit-rot. Blob layout: `[u8 version][u32 LE len][RS(nonce‖ciphertext)]`.
- **Salt integrity & self-heal.** The per-DB salt is stored RS-encoded with a SHA-256 checksum. An absent / corrupt / unrecoverable salt is treated as absent and the store **resets to a fresh start** (no on-disk migration) rather than bricking; minor bit-rot is corrected and prior history survives.
- **Secrets separation.** The LLM API key (keyring `magi-rs`) and the DB master password (keyring `magi-rs-internal`) are kept in **separate** keyring services. Rotating the API key never invalidates the local conversation DB.
- **Filesystem sandbox.** Every file-touching tool canonicalizes its target and validates it against the workspace root via `PathGuard` (handling Windows `\\?\` verbatim prefixes, null-byte attacks, and lexical normalization).
- **Shell sandbox.** The `bash` tool enforces a per-binary argument allowlist and bans shell metacharacters to prevent subshell injection on both PowerShell and bash.

---

## Project Structure

```
src/
  main.rs              -- config discovery, master-key bootstrap, tool registration, entry
  config.rs            -- MagiConfig (magi.toml load + provider/model resolution)
  agent/
    mod.rs             -- Agent orchestrator: multi-turn tool loop + approval gate
    provider.rs        -- Provider trait; AnthropicProvider (SSE) + OpenAiCompatibleProvider + StaticProvider
  tools/
    mod.rs             -- Tool trait + ToolError
    ls.rs read.rs write.rs grep.rs bash.rs knowledge.rs
  system/
    database.rs        -- EncryptedSqliteMemory (MemoryStore) over SQLite + CryptoVault
    fs.rs grep.rs secrets.rs path_guard.rs
  utils/
    crypto.rs          -- CryptoVault: Argon2id → AES-256-GCM-SIV → Reed-Solomon FEC
  services/
    oauth.rs           -- OAuth PKCE login (callback on 127.0.0.1:54545)
  tui/
    mod.rs             -- ratatui app (Normal / Selection / Visual)
docs/                  -- public overview & philosophy (OVERVIEW.md)
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

> Some tests use real OS resources: `system::secrets` tests write to the real OS keyring under service `magi-rs-test-suite`, and the OAuth callback server binds port `54545` — avoid two interactive `/login` flows at once. DB/crypto tests are isolated via `tempfile`.

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Rust toolchain (stable, edition 2021) | Yes | via [rustup](https://rustup.rs/) |
| `ripgrep` (`rg`) | For the `grep` tool | on `PATH` |
| OS keyring | For persistence | Windows Credential Manager / macOS Keychain / Secret Service. Without it the session runs ephemeral |
| Anthropic API key | For live replies | env var, keyring, `key.txt`, or `/login` |

---

## Documentation

- [`docs/OVERVIEW.md`](docs/OVERVIEW.md) — what Magi is, the `magi-core` foundation, and the multi-perspective philosophy behind the name.

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Credits

The **Magi** name comes from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995, Hideaki Anno / Gainax), where three supercomputers — Melchior, Balthasar, and Caspar — govern critical decisions through structured consensus. The same multi-perspective philosophy backs the sibling [`magi-core`](https://crates.io/crates/magi-core) crate, which Magi Agent integrates for its planned multi-perspective `consult` workflow.
