# Magi Agent — Terminal AI Assistant in Rust

[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![CI](https://github.com/BolivarTech/magi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BolivarTech/magi/actions/workflows/ci.yml)
[![Lints](https://img.shields.io/badge/lints-clippy%20clean-blue.svg)](https://github.com/rust-lang/rust-clippy)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![crates.io](https://badgen.net/crates/v/magi-rs?label=version)](https://crates.io/crates/magi-rs)

**Magi Agent** (`magi-rs`) is a terminal AI assistant in Rust, modeled on Claude Code. It drives an LLM provider through a multi-turn **tool loop** with sandboxed filesystem and shell access, and persists every conversation to a **locally-encrypted SQLite store**. Nothing leaves your machine except the model API calls you explicitly authorize.

---

## Why Magi?

Most AI coding agents are SaaS-bound and tied to a single vendor. Magi is built for the opposite constraint: **environments where source code cannot leave the machine.**

| Principle | What it means in Magi |
|-----------|-----------------------|
| **Local-first** | A single static Rust binary. Conversation history and project knowledge live in an encrypted SQLite file on disk, never a third-party vault. |
| **Encrypted at rest** | Every stored record is sealed with AES-256-GCM-SIV + error-correcting FEC under a random data key, itself wrapped by an Argon2id-derived key, via the audited [`cryptovault`](https://crates.io/crates/cryptovault) crate. The data key is unlocked by a **user passphrase** (zero-knowledge: nothing persisted opens the DB without it), and a wrong passphrase never wipes the DB. |
| **Sandboxed by default** | Every filesystem tool is confined to the workspace via `PathGuard`; the shell tool runs a strict command allowlist with a hard ban on shell metacharacters. |
| **Human-in-the-loop** | Each tool call must be approved inline in the TUI before it runs. |
| **Fails safe** | If no passphrase is available (headless with none supplied, or entry cancelled), the session degrades to ephemeral (no persistence) rather than ever encrypting with a constant key. The existing on-disk DB is left untouched. |

---

## Features

| Area | Capability |
|------|------------|
| **TUI front-end** | `ratatui` app with **Normal / Selection / Visual** modes, UTF-8-safe cursor handling, system clipboard integration |
| **Agent loop** | Multi-turn orchestration, bounded tool calls (max 15), repetitive-call detection, ANSI/control-char sanitization, interactive approval gate |
| **Provider** | Anthropic **Messages API** with SSE streaming + `tool_use` assembly and 429 retry; `StaticProvider` fallback when no key is configured |
| **Sandboxed tools** | `ls`, `view`, `edit`, `grep`, `bash` (strict allowlist), `project_knowledge` (persistent facts), `consult` (MAGI multi-perspective consensus) |
| **MAGI consult** | The agent can escalate hard, trade-off-heavy decisions to a 3-perspective consensus (Melchior / Balthasar / Caspar) via the `magi-core` crate. It is invoked autonomously through the tool loop (each call passes the approval gate) or forced with `/consult` |
| **Code review** | `magi-rs code-review`, `/code-review` and the `code-review` tool review a file, a directory, a commit, a range, a branch or the uncommitted changes with the MAGI trio, batch by batch, check every finding against the reviewed code, filter false positives with a decision model and print one report. See [Code review](#code-review-magi-rs-code-review) |
| **Tiered memory (RAG)** | Embedding-indexed vector store (any openai-compat endpoint, default Ollama + `nomic-embed-text-v2-moe`), composite reranker (similarity + recency + salience), decay/eviction, always-injected preference profile, hard token-budget assembler. Context stays bounded regardless of history depth |
| **Encrypted memory** | SQLite (WAL) with **one cached key derivation per session** (Argon2id + per-DB salt), AES-256-GCM-SIV, Reed-Solomon FEC; text **and** embedding vectors encrypted at rest |
| **OAuth login** | PKCE flow against the Anthropic Console (`/login` in the TUI); the minted key is stored in the vault |
| **Zero-knowledge vault** | A passphrase (never persisted) unlocks the DB; API keys and other secrets live in an encrypted `vault` table managed by `magi-rs vault {ls,set,rm,passwd}`. No OS keyring, no `key.txt` |

---

## Installation

### Prebuilt binaries

Every release ships ready-to-run archives on the [Releases page](https://github.com/BolivarTech/magi/releases). No Rust toolchain required.

| Archive | Runs on |
|---|---|
| `…-windows-x86_64.zip` (or `.7z`) | Windows 10/11, x64 |
| `…-linux-x86_64.tar.gz` (or `.7z`) | Linux x64 with **glibc ≥ 2.35** — Ubuntu 22.04+, Debian 12+ |
| `…-linux-x86_64-compat.tar.gz` (or `.7z`) | Linux x64 with **any glibc ≥ 2.17** — RHEL/Rocky/Alma/CloudLinux 8 and 9, Debian 10+, older Ubuntu |
| `…-linux-aarch64-rpi5.tar.gz` (or `.7z`) | Raspberry Pi 5 / aarch64, glibc ≥ 2.35 |
| `…-macos-universal2.tar.gz` (or `.7z`) | macOS, Intel and Apple Silicon |

Every platform ships two archives with identical contents. **Prefer `.tar.gz` on
Unix and `.zip` on Windows.** Both open with tools the OS already has:

```bash
tar -xzf magi-rs-*.tar.gz          # Linux/macOS, no extra package
```

The `.7z` is smaller but needs `p7zip`, which is *not* installed by default on
RHEL-family distros, minimal server images or most containers.

**Which Linux build?** Check what your system has:

```bash
ldd --version | head -1
```

2.35 or newer → either archive works; prefer `linux-x86_64`, built natively. Anything older → use `linux-x86_64-compat`. The two are the same program from the same source; they differ only in the glibc symbol versions they request, and `-compat` runs everywhere the other does. If you pick the wrong one the binary refuses to start with `GLIBC_2.xx not found`. That is the only symptom, and the fix is to download `-compat`.

Verify what you downloaded against the `SHA256SUMS.txt` published alongside it:

```bash
sha256sum -c --ignore-missing SHA256SUMS.txt      # Linux/macOS
Get-FileHash .\magi-rs-*.zip -Algorithm SHA256    # Windows
```

### Prerequisites

- A Rust toolchain (stable, edition 2021), installed via [rustup](https://rustup.rs/). **Only needed to build from source.** The prebuilt archives above run as-is.
- `ripgrep` (`rg`) on `PATH` for the `grep` tool. Required either way.
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com/) (**recommended**), set via `ANTHROPIC_API_KEY` or stored in the vault (`magi-rs vault set ANTHROPIC_API_KEY`). `/login` (OAuth) also works but is **best-effort**. See [Configuration](#configuration).

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

See [docs/usage.md](https://github.com/BolivarTech/magi/blob/main/docs/usage.md).

---

## Headless mode

See [docs/headless.md](https://github.com/BolivarTech/magi/blob/main/docs/headless.md).

### Typed decisions: `magi-rs decide`

See [docs/decide.md](https://github.com/BolivarTech/magi/blob/main/docs/decide.md).

### Code review: `magi-rs code-review`

See [docs/code-review.md](https://github.com/BolivarTech/magi/blob/main/docs/code-review.md).

---

## Configuration

See [docs/configuration.md](https://github.com/BolivarTech/magi/blob/main/docs/configuration.md).

---

## Logging

See [docs/logging.md](https://github.com/BolivarTech/magi/blob/main/docs/logging.md).

---

## Tiered Memory (RAG)

See [docs/TIERED-MEMORY.md](https://github.com/BolivarTech/magi/blob/main/docs/TIERED-MEMORY.md).

---

## How It Works

See [docs/architecture.md](https://github.com/BolivarTech/magi/blob/main/docs/architecture.md#how-it-works).

## Security Model

See [docs/architecture.md](https://github.com/BolivarTech/magi/blob/main/docs/architecture.md#security-model).

## Project Structure

See [docs/architecture.md](https://github.com/BolivarTech/magi/blob/main/docs/architecture.md#project-structure).

---

## Testing

See [docs/testing.md](https://github.com/BolivarTech/magi/blob/main/docs/testing.md).

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Rust toolchain (stable, edition 2021) | Yes | via [rustup](https://rustup.rs/) |
| `ripgrep` (`rg`) | For the `grep` tool | on `PATH` |
| Vault passphrase | For persistence | Chosen on first run (or supplied via `-p`/`MAGI_PASSPHRASE`). Without one available the session runs ephemeral (the on-disk DB is left untouched) |
| [Ollama](https://ollama.com/) (default backend) | For live replies (default) | running daemon + a chat model (e.g. `kimi-k2.6:cloud`); run `ollama signin` for `:cloud` tags. Any OpenAI-compatible endpoint works by pointing `base_url` elsewhere |
| Embedding model | For tiered memory (`selective`, the default) | `ollama pull nomic-embed-text-v2-moe:latest`; without it memory degrades gracefully to text-only persistence |
| Ollama and the decision models | For `magi-rs decide` | The default `clef:cloud` needs Ollama 0.40.5 or later and `ollama signin`; the local alternative `clef-flash` needs 0.35.1 or later. `ollama pull nimble` for the fallback. Without `ollama signin` the daemon refuses `clef:cloud` with HTTP 401 or 403 and the chain moves on to `nimble`; a 401 or 403 on a model that is not `:cloud` still stops the chain. If `nimble` is missing too, the call fails with `decision_unavailable`: the message names `ollama signin`, and the `chain` entry for `nimble` names the missing model (`model_not_found`). If every model is refused, it fails with `missing_key` and the message names `ollama signin`. When no model answers, internal decisions fall back to the main model |
| Anthropic API key | Optional (opt-in) | only with `provider = "anthropic"`; via env var, the vault, or `/login` |

### Default models (Ollama)

These are the built-in defaults used with no `magi.toml` (Ollama-first). The `:cloud`-tagged models run on
Ollama's cloud. Run `ollama signin` once (no local weight download); the embedding model is pulled locally.
Override any of them per-section in `magi.toml` (`[openai]`, `[embedding]`, `[magi]`) with local equivalents.

| Role | Default model | Used by |
|------|---------------|---------|
| Chat (principal) | `kimi-k2.6:cloud` | `magi-rs` agent — live replies |
| Embedding | `nomic-embed-text-v2-moe:latest` | `magi-rs` tiered memory (`selective`) — `ollama pull` it |
| Decision (primary) | `clef:cloud` | `magi-rs decide` — needs Ollama 0.40.5 or later and `ollama signin`, spends the account's quota and sends each decision state to ollama.com. The local alternative is `clef-flash` (`ollama pull clef-flash`, 10.9 GB, Apache-2.0, Ollama 0.35.1 or later) |
| Decision (fallback) | `nimble` | `magi-rs decide` — `ollama pull nimble`; tried when the primary fails or misses its threshold |
| Code-review gate | `clef:cloud` (0.40) → `clef-flash` (0.32) → `nimble` (0.80) | `magi-rs code-review` false-positive gate — `clef:cloud` needs Ollama 0.40.5 or later and `ollama signin`, spends the account's quota and sends each judged finding with its code to ollama.com; the other two run locally |
| Melchior (Scientist) | `glm-5.3:cloud` | `magi-core` multi-perspective consensus (`consult` tool / `/consult`) |
| Balthasar (Pragmatist) | `kimi-k2.6:cloud` | `magi-core` multi-perspective consensus (`consult` tool / `/consult`) |
| Caspar (Critic) | `deepseek-v4-pro:cloud` | `magi-core` multi-perspective consensus (`consult` tool / `/consult`) |

> The MAGI trio deliberately runs three distinct model families (Zhipu / Moonshot / DeepSeek) for genuine
> cross-lineage diversity. The `consult` tool (and the `/magi` command) only need these when a
> multi-perspective analysis is requested.

---

## Documentation

| File | Contents |
|---|---|
| [`docs/OVERVIEW.md`](https://github.com/BolivarTech/magi/blob/main/docs/OVERVIEW.md) | What Magi is, the `magi-core` foundation, and the multi-perspective philosophy behind the name. |
| [`docs/usage.md`](https://github.com/BolivarTech/magi/blob/main/docs/usage.md) | The TUI: modes, tool approval, slash commands, tools, MAGI consult and mode routing. |
| [`docs/headless.md`](https://github.com/BolivarTech/magi/blob/main/docs/headless.md) | `magi-rs init`, `query` and `consult` for CI and for other applications: the JSON output, authorization tiers, state layout and exit codes. |
| [`docs/decide.md`](https://github.com/BolivarTech/magi/blob/main/docs/decide.md) | `magi-rs decide` in full: providers and models, limits, keys, thresholds, the output fields and the error and cause sets. |
| [`docs/code-review.md`](https://github.com/BolivarTech/magi/blob/main/docs/code-review.md) | `magi-rs code-review`: what it reviews and sends, batches, context, the gate, the report and its limits. |
| [`docs/configuration.md`](https://github.com/BolivarTech/magi/blob/main/docs/configuration.md) | API key and model discovery, the vault and passphrase, the default backend and `magi.toml`. |
| [`docs/keys.md`](https://github.com/BolivarTech/magi/blob/main/docs/keys.md) | Which API key each section of `magi.toml` uses. |
| [`docs/logging.md`](https://github.com/BolivarTech/magi/blob/main/docs/logging.md) | The daily log under `.magi/logs/`, what reaches the screen and how to find one run in it. |
| [`docs/TIERED-MEMORY.md`](https://github.com/BolivarTech/magi/blob/main/docs/TIERED-MEMORY.md) | Full technical reference for the tiered agnostic memory subsystem: RAG pipeline, three pillars, architecture, configuration, benchmark. |
| [`docs/E2E-TESTING.md`](https://github.com/BolivarTech/magi/blob/main/docs/E2E-TESTING.md) | Hands-on end-to-end testing guide for the tiered memory feature (cross-session recall, preferences, rollback). |
| [`docs/REASONING-BUDGET.md`](https://github.com/BolivarTech/magi/blob/main/docs/REASONING-BUDGET.md) | The MAGI trio's reasoning controls and output cap: what each `[magi]` key does, what the clock-coverage warning means, and how to read `reasoning`/`control` in a consult report. |
| [`docs/OPENROUTER-BACKEND.md`](https://github.com/BolivarTech/magi/blob/main/docs/OPENROUTER-BACKEND.md) | A verified OpenRouter configuration and the four traps that backend sets. |
| [`docs/architecture.md`](https://github.com/BolivarTech/magi/blob/main/docs/architecture.md) | How the agent loop works, the security model and the project structure. |
| [`docs/testing.md`](https://github.com/BolivarTech/magi/blob/main/docs/testing.md) | How to run the tests and the smoke harness. |
| [`docs/magi.toml.example`](https://github.com/BolivarTech/magi/blob/main/docs/magi.toml.example) | The annotated `magi.toml` reference. |
| [`docs/magi.toml.openrouter.example`](https://github.com/BolivarTech/magi/blob/main/docs/magi.toml.openrouter.example) | A ready-to-copy `magi.toml` for OpenRouter. |

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Credits

The **Magi** name comes from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995, Hideaki Anno / Gainax), where three supercomputers (Melchior, Balthasar, and Caspar) govern critical decisions through structured consensus. The same multi-perspective philosophy backs the sibling [`magi-core`](https://crates.io/crates/magi-core) crate, which Magi Agent integrates for its planned multi-perspective `consult` workflow.
