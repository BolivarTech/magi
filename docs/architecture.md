# Architecture

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
4. If it contains `ToolUse` blocks, each tool is approved, executed, and its result pushed back as a `User`-role `ToolResult`, and then the loop repeats.
5. With no tool requested, the final assistant text is returned.

**Invariants:** at most 15 tool calls per query; three consecutive identical tool calls abort ("repetitive tool call detected"); each tool call needs UI approval within 5 minutes or it is denied.

---

## Security Model

- **Encrypted memory (envelope DEK/KEK).** A random 32-byte **data key (DEK)** encrypts every record via the audited [`cryptovault`](https://crates.io/crates/cryptovault) crate (Argon2id → AES-256-GCM-SIV → error-correcting FEC, `#![forbid(unsafe_code)]`). The user **passphrase** plus a per-DB salt derives a **KEK** (Argon2id, OWASP 2025: 64 MiB, t=3, p=4) that **wraps** the DEK; `vault_meta` stores `{salt, wrapped_dek}`, each FEC-encoded. The DEK is unwrapped **once per session** and held **masked in RAM** (`MaskedDek`: XOR mask rotated from the OS random source (`SysRng`) on every access, best-effort `mlock`). No Argon2 runs per record.
- **Zero-knowledge unlock.** Nothing persisted opens the DB without the passphrase: no OS keyring, no cached file, no copy of the DEK on disk (`system::secrets` and the `keyring` dependency were removed in v0.9.0). The passphrase resolves as `-p` > `MAGI_PASSPHRASE` > interactive prompt; a hard strength floor (`zxcvbn` ≥ 3, ≥ 12 chars, no override) guards against offline brute-force of a portable `.db`. Forgetting the passphrase means the data is unrecoverable. There is no backdoor.
- **Never loses your data.** A wrong passphrase or a corrupt wrapped key fails with a typed, **retryable** error and **never wipes** the database. Only a brand-new or pre-envelope DB (no wrapped key present) is bootstrapped fresh. `vault_meta` is FEC-protected, so on-disk bit-rot is corrected before the unwrap; corruption beyond the FEC's reach surfaces as a distinct `VaultMetaCorrupt` error, never a silent wrong key.
- **Secrets separation.** The passphrase (which unlocks the DEK) and the stored API keys (entries *inside* the vault) are different secrets in different places: rotating a stored API key never requires re-keying the passphrase, and a wrong API key never invalidates the local conversation DB. `magi-rs vault passwd` rotates the passphrase without re-encrypting any record (it re-wraps the same DEK).
- **Filesystem sandbox.** Every file-touching tool canonicalizes its target and validates it against the workspace root via `PathGuard` (handling Windows `\\?\` verbatim prefixes, null-byte attacks, and lexical normalization).
- **Shell sandbox.** The `bash` tool enforces a per-binary argument allowlist and bans shell metacharacters to prevent subshell injection on both PowerShell and bash.
- **No credentials in `magi.toml`.** An authenticated `base_url` carries `[user]`/`[password]` placeholders, not a literal credential: the real value is resolved from the vault in memory at use time and is never written to disk. A URL that does end up with an embedded credential (e.g. copied from an older config) is redacted **by position**, not by content, in every notice, error and report, including a doubly percent-encoded credential and a URL that fails to parse outright (redacted entirely, as the safe failure direction). This now also covers a **connection failure**: the hint that names the endpoint after a refused or unreachable connection redacts the same way (v0.20.0), so a dead backend never echoes its credential back at you.
- **`Authorization: Basic` credentials are masked too (v0.20.0).** The stdout auditor already masked a `Bearer` token; it now also recognizes a `Basic <base64>` header (including the base64 encoding of a `base_url` credential itself) echoed inside a server's error body, and masks it the same way. A memory-distillation failure is reported through the same audited path rather than raw `stderr`, and an OAuth login failure never echoes the authorization code, PKCE verifier, or minted access token a server sent back.

---

## Project Structure

```
src/
  main.rs              -- config discovery, passphrase resolution / vault open, tool registration, entry
  lib.rs               -- library root (pub mod vault, magi, redact, ...) so fuzz / coverage targets can link
  config.rs            -- MagiConfig (magi.toml load + provider/model resolution); config/migrate.rs submodule
  defaults.rs          -- single source of truth for all built-in default literals
  redact.rs            -- URL userinfo redaction by position, never by content
  notices.rs           -- startup notices: per-site level, split by the mouth it reaches
  magi/                -- MAGI trio subsystem: mode resolution, complexity gate, probe (lib, pure)
    mod.rs             -- shared timeout-scale / gate / probe constants
    kind.rs            -- ProviderKind vocabulary (ollama | openai-compat | anthropic)
    mode.rs            -- Mode/ModeSource vocabulary + five-level resolve_mode_guarded
    gate.rs            -- complexity gate: pure predicate over content length + per-mode threshold
    probe.rs           -- model measurement via magi-core's ProviderProbe (composition, not migration)
    endpoint.rs        -- base_url credential-placeholder templates, resolved from the vault
    report_anchors.rs  -- named anchors into magi-core's report markdown, for bounded-output truncation
  agent/
    mod.rs             -- Agent orchestrator: multi-turn tool loop + approval gate
    provider.rs        -- Provider trait; AnthropicProvider (SSE) + OpenAiCompatibleProvider + StaticProvider
    mode_classifier.rs -- mode classification over the main provider (bin)
  memory/
    mod.rs             -- subsystem facade: MemoryKind, public re-exports (recall, assemble_selective)
    store.rs           -- SqliteVectorStore: encrypted vector store (memories table)
    embedding.rs       -- EmbeddingProvider trait + OpenAiCompatibleEmbedder
    retrieval.rs       -- recall() public API + composite reranker
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
    bench_vault_crypto.rs -- FEC/decrypt performance baseline
  tools/
    mod.rs             -- Tool trait + ToolError + requires_approval()
    ls.rs read.rs write.rs grep.rs bash.rs knowledge.rs consult.rs
  system/
    database.rs        -- EncryptedSqliteMemory (MemoryStore) over SQLite; opens via the vault envelope
    fs.rs grep.rs path_guard.rs
  vault/
    mod.rs             -- SecretStore boundary + public re-exports (frontier main.rs sees)
    envelope.rs        -- DEK/KEK envelope: bootstrap/open, wrap/unwrap/rekey, keyless FEC over vault_meta
    memguard.rs        -- MaskedDek: DEK masked in RAM (SysRng per-access rotation), harden_process
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
