# Changelog

All notable changes to **Magi Agent** (`magi-rs`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the **minor** position signals significant or breaking
changes and the **patch** position signals backward-compatible fixes.

## [Unreleased]

### Added
- Per-agent MAGI model selection via `magi.toml` `[magi]` section and
  `MAGI_MODEL_{MELCHIOR,BALTHASAR,CASPAR}` env vars. Opt-in; absent = all three
  perspectives share the principal model (backward compatible). Overrides reuse
  the principal backend's endpoint/key and vary only the model — true
  cross-family lineage diversity requires an Ollama-style multi-family endpoint.

### Deferred (tracked in internal dev-docs)
- **#14** Envelope encryption (key rotation / crypto-shredding / multi-tenancy) — enterprise roadmap.
- **#17** Runtime warning visibility — malformed tool-JSON (#4) and poison recovery (#8) warnings remain stderr-only under the alt-screen; startup/login warnings are already surfaced.
- **#18** Blob version-dispatch / migration — the blob version byte is detection-only; a future format bump still needs a migrate-or-reset path.
- **OpenAI key in keyring / `key.txt`** — `OPENAI_API_KEY` is currently env-only; aligning it with the Anthropic discovery order is a tracked follow-up.

## [0.4.0] - 2026-06-07

First step of the multi-backend MAGI integration: the `magi-core` crate (already
declared but unused) is now wired in, giving the agent a **three-perspective
consensus** capability (Melchior / Balthasar / Caspar). Additive only — the
`Provider` trait, the agent loop, config discovery, and encrypted memory are
unchanged.

### Added
- **`consult` tool** — exposes `magi-core`'s 3-perspective consensus as a tool the main LLM can invoke autonomously when a question has genuine trade-offs. Routing emerges from the existing tool loop (no separate classifier); each call passes the inline approval gate, which doubles as cost control for the ≈ 3 model calls.
- **`/consult <question>` command** — forces a MAGI consensus directly in the TUI, bypassing the router and approval, rendering the verbatim report (three perspectives + verdict). Runs `analyze` in a joined task so a panic in `magi-core` surfaces as a recoverable error instead of killing the runner; blocks the session while it runs, like a normal turn.
- **`MagiCoreProviderAdapter`** (`src/agent/magi_adapter.rs`) — bridges magi-rs's resolved `Provider` to `magi_core::provider::LlmProvider`, so the consensus reuses the same backend + credentials (Anthropic or any OpenAI-compatible endpoint). No second LLM config layer.
- **StaticProvider guard** — `consult` is registered (and `/consult` works) only when a real provider is configured; on the static/no-key path the tool is absent and `/consult` reports that a provider is required.
- **Degraded surfacing** — a consult that completes with fewer than three agents is prefixed `[DEGRADED: …]` so a low-quality consensus is never silent.
- New tests for the adapter (assembled text, role-fold delimiter, error mapping), the tool (contract + consensus + invalid-args + empty-query + oversized-query + backend-error via `magi-core`'s `RoutingMockProvider`), and the `/consult` parser + a full-report render-safety test. Total tests: **147** (was 136).

### Changed
- `magi-core` dependency bumped `1.0` → `1.1` (no features enabled; `reqwest 0.12` is **not** pulled — the adapter reuses magi-rs's existing `reqwest 0.11` stack). `test-utils` enabled as a dev-dependency for `RoutingMockProvider`.

### Security
- The verbatim consult report (LLM-generated) is run through `sanitize_text` before rendering — strips ANSI escapes / control characters, matching the streaming-delta path. Consult input is length-capped (8192 bytes) on both the tool and forced `/consult` paths, and empty queries are rejected before any model call.

### Known limitations
- **System-prompt fold.** `magi-core` differentiates its three personas via distinct *system* prompts, but magi-rs's `Provider` has no system-role channel yet, so the system text is folded into the user turn (behind an explicit delimiter). On weak / small local models this can weaken persona divergence and JSON adherence. Revisit when magi-rs gains a system-prompt channel.
- **`CompletionConfig` not applied.** The adapter does not forward `max_tokens` / `temperature` to the backend (the `Provider` trait exposes no per-call knobs). Deferred.
- **Deferred follow-ups (internal dev-docs):** repetitive-call detection keyed on the consult `query` argument, and an `InputTooLarge` early-rejection UX for oversized `/consult` input.

## [0.3.1] - 2026-05-26

TUI rendering hotfix — two pre-existing UX bugs found during the 0.3.0
end-to-end smoke (long-reply truncation + new-message scroll-off). Both
affect all providers (Anthropic, OpenAI-compatible, Static). Render-only,
no API / trait / contract changes.

### Fixed
- **Conversation pane truncated long messages** at the panel's right border instead of wrapping. New `wrap_message` helper (`src/tui/mod.rs`) word-wraps each message by `chars().count()` (UTF-8 char-safe) to the panel's inner width, preserves existing `\n` as hard breaks, and hard-splits absurdly long words. One message stays one `ListItem` so Selection / Visual navigation (`app.selected_index → app.messages[i]`) is unchanged.
- **Conversation pane did not auto-scroll** when new messages overflowed the bottom. New `effective_selection` / `effective_highlight_symbol` helpers pin the last message as the selected row in Normal mode (ratatui's `List` auto-scrolls to keep the selected item visible) while suppressing the `">> "` highlight prefix so the pin is invisible. Selection / Visual modes still show the prefix and use the user-chosen index.
- **Streaming tail of a tall response stayed off-screen** (and on some terminals "jumped/reset" when the message exceeded the conversation pane). The follow-tail fix now tail-truncates the rendered last message in Normal mode to its trailing `viewport_inner_height` wrapped lines, so streaming visibly scrolls line-by-line. Selection / Visual modes still render every wrapped line so the full message remains reviewable via `Ctrl+S → ↑`.

### Added
- 15 new tests covering the four render-time helpers (`wrap_message`: word wrap normal / multibyte / oversized word / embedded newlines / width-0 / empty; `effective_selection`: follow-tail in Normal mode / chosen index in Selection / Visual / empty messages; `effective_highlight_symbol`: by mode; `tail_lines`: keeps last N / unchanged when max≥len / max=0 no-op / empty / shift-by-one streaming tick). Total tests: **136** (was 121).

### Known limitations
- **Leading indentation and internal whitespace runs are collapsed** when a message is wrapped. `wrap_message` uses `split_whitespace`, which drops leading spaces and multi-space gaps. Markdown bullets like `"  - item"` render as `"- item"`; preformatted blocks lose alignment. Tracked for a follow-up patch (preserve per-line indentation via a different tokenizer).
- **Wrap width is measured in `chars().count()`, not terminal display width.** CJK / emoji glyphs that occupy two cells will wrap one column early; combining marks may wrap one column late. Tracked for a follow-up that switches to `unicode-width`.

## [0.3.0] - 2026-05-25

First multi-backend release. Magi can now talk to any OpenAI-compatible Chat
Completions endpoint — local **Ollama**, OpenAI, Groq, OpenRouter — selected by
a new `magi.toml`, alongside the existing Anthropic Messages API path. No
regression: the Anthropic surface and all its tests are byte-equivalent.

### Added
- **`OpenAiCompatibleProvider`** (`src/agent/provider.rs`) — Chat Completions over `{base_url}/chat/completions` with `stream:true`, a `stream::unfold` SSE state machine that finalizes on `finish_reason` / `[DONE]` / stream-end with an idempotent `done` guard (`MessageDone` emitted exactly once), `data:` prefix tolerant of optional space, malformed lines swallowed, HTTP non-2xx surfaced as `Err`, `MAX_SSE_BUFFER_BYTES` (8 MiB) cap. Constructor takes a named-field `OpenAiSettings` struct (no positional same-type swap).
- **`magi.toml` configuration** (`src/config.rs`) — `MagiConfig` / `OpenAiConfig` / `AnthropicConfig`, `serde(deny_unknown_fields)` so typos (and `api_key`) fail at parse time, **env > TOML > defaults** precedence (`MAGI_PROVIDER` / `OPENAI_BASE_URL` / `OPENAI_MODEL`). `MagiConfig::load` distinguishes `NotFound` (silent default) from other I/O errors (surfaced as a TUI startup notice). Reference `magi.toml.example` is committed; user-local `magi.toml` is gitignored.
- **Coalesced `map_messages` / `map_tools`.** Same-turn parallel `Content::ToolUse` blocks collapse into ONE assistant message with a `tool_calls` array; User `Text`/`ToolResult` are emitted in **content order** (each block as its own OpenAI message). `Tool::input_schema` forwards as `tools:[{type:"function",function:{…}}]`.
- **Bounded tool-call accumulator.** `MAX_TOOL_CALL_SLOTS = 64` caps streamed `tool_calls[].index`; over-cap **warns** (`eprintln`) instead of dropping silently (RF-8). Orphan `arguments` fragments (slot has neither id nor name yet) are skipped with a warning to prevent mis-attribution.
- **`tests-121` (was 95).** 26 new tests — config parsing & precedence; coalesced message mapping; mixed-content User ordering; OpenAI text streaming (with [DONE], without [DONE]/finish_reason, malformed line, HTTP error, stream-end-only finalize); fragmented `tool_calls` assembly; bounded index; post-stop TextDelta suppression; args-before-id skip; `resolve_provider` wiring.

### Security
- **`OPENAI_API_KEY` from environment only.** Never read from `magi.toml`; `deny_unknown_fields` rejects an `api_key` field at parse time. The dummy `"ollama"` fallback for local Ollama is documented inline; real backends fail loudly with 401 if the env var is unset.
- **Keyring separation unchanged.** The Anthropic key (`magi-rs`) and DB master key (`magi-rs-internal`) remain in separate keyring services; `test_agent_history_resilience_to_key_rotation` is untouched.

### Documentation
- **README "Configuration: `magi.toml`" section** — full precedence table for the four settings (`MAGI_PROVIDER`/`OPENAI_BASE_URL`/`OPENAI_MODEL`/`ANTHROPIC_MODEL`), keys-never-in-TOML invariant (and the `deny_unknown_fields` parse-error consequence), Ollama quickstart.
- **`reqwest::Client` no-timeout rationale documented** — local Ollama can spend tens of seconds on cold-load before the first SSE event; a total-request timeout would truncate healthy long streams. Stream-side termination is handled by the three finalize triggers + `MAX_SSE_BUFFER_BYTES`.

## [0.2.1] - 2026-05-25

### Changed
- **Docs/UX:** the README and the static-mode startup hint now recommend a **standard API key** (`ANTHROPIC_API_KEY` / `key.txt`) as the supported path, and mark `/login` (OAuth) as **best-effort** — it reuses Anthropic's Claude Code OAuth client, so it may be rate-limited or blocked. No behavior change.

## [0.2.0] - 2026-05-25

First **public** release — the repository is now open-source (MIT OR Apache-2.0). Functionally identical to 0.1.2, promoted to the semver-correct minor version for the breaking on-disk format change made during the 0.1.x security hardening (a pre-0.1.1 `.magi-rs-memory.db` resets on upgrade). See **[0.1.1]** for the security-audit remediation and **[0.1.2]** for the docs + release-pipeline work this consolidates.

## [0.1.2] - 2026-05-24

### Changed
- **Internal development docs are no longer shipped.** The audit, strategy, and roadmap notes moved to a gitignored `dev-docs/` directory — excluded from the repository and the published crate.

### Added
- **`docs/OVERVIEW.md`** — a public overview of Magi, the `magi-core` foundation, and the multi-perspective (MAGI) philosophy.
- **Windows binary in releases.** Tagged releases now attach `magi-rs-vX.Y.Z-windows-x86_64.zip` (the compiled `magi-rs.exe` + README + licenses) alongside the source archives.

## [0.1.1] - 2026-05-24

First security-hardened release. The internal audit
(8 CRITICAL + 6 WARNING) is fully remediated and all
non-deferred follow-ups are closed. Verified gate: 95/95 tests, `clippy
--all-targets` clean, `fmt`/`build --release`/`doc` clean, `audit` baseline.

### Security
- **Filesystem sandbox closed on all tools.** `write.rs` and `grep.rs` now route through `PathGuard::validate`; the `bash` tool sandboxes every non-flag argument, closing the Windows forward-slash absolute-path escape (`C:/...`).
- **Crypto hardening.** Independent `OsRng` nonce per record; Argon2id pinned to OWASP 2025 parameters (64 MiB, t=3, p=4); a hard cap on the decrypt length-prefix to prevent hostile allocations.
- **Ephemeral fallback (no constant key).** An inaccessible OS keyring degrades the session to in-memory only — it never falls back to a constant passphrase.
- **OAuth callback timeout.** The PKCE callback server is bounded by a timeout; state is enforced (CSRF protection).
- **Secrets separation preserved.** API key (`magi-rs`) and DB master key (`magi-rs-internal`) remain in separate keyring services; rotating one never invalidates the other.

### Changed
- **One key derivation per session (B′).** Argon2 no longer runs per record. A per-DB salt is persisted in a new `vault_meta` table and the 32-byte key is derived once at construction and cached (`Zeroizing`), removing the O(N) cost on history load. Blob layout is now `[u8 version][u32 LE len][RS(nonce‖ciphertext)]`.
- **SSE wire path is byte-buffered.** The streaming reader buffers `Vec<u8>` and decodes only complete `\n\n` blocks, so a multi-byte UTF-8 character split across a network chunk is never decoded mid-character.
- **`tool_use` is assembled from the stream.** `AnthropicProvider` now builds `Content::ToolUse` from `content_block_start`/`input_json_delta`/`stop` (previously tools were inert with the real provider).
- **In-session `/login`.** A successful login rebuilds the running agent's provider without a restart; the canned `StaticProvider` history is cleared **only** when the prior provider was static, and the banner refreshes.
- **Dependency:** `magi-core` bumped `0.6` → `1.0.1`.

### Added
- **Salt integrity & self-heal.** The per-DB salt is RS-encoded with a SHA-256 checksum; an absent/corrupt/unrecoverable salt self-heals to a fresh start instead of bricking, and minor bit-rot is corrected.
- **Poisoned-lock recovery.** A poisoned connection mutex recovers (`into_inner` + warn-once) so persistence continues for the rest of the session.
- **Idempotent salt bootstrap.** Salt read + reset run inside a `BEGIN IMMEDIATE` transaction with an in-tx re-check, closing the concurrent first-open TOCTOU.
- **Blob version byte** (`BLOB_VERSION`), validated before any allocation.
- **TUI startup notices.** Keyring-unavailable (no persistence) and history-reset (fresh start) states surface as startup `Info` messages instead of pre-TUI stderr; the store exposes `was_reset()`.

### Fixed
- Malformed tool-input JSON logs a warning (tool name/id + error) instead of silently degrading to `{}`.
- Default-model fallback handles a blank/whitespace/absent `key.txt` model line.
- Full `clippy --all-targets` and tree-wide `rustfmt` baseline established.

### Notes
- **No on-disk migration (D2/D6 fresh-start).** The blob layout and KDF changed without a migration path: a pre-0.1.1 `.magi-rs-memory.db` deterministically resets to a fresh, empty store on first open. This is intentional for the pre-1.0 single-user threat model.

## [0.1.0] - 2026-05-16

Initial pre-release, published primarily to reserve the `magi-rs` crate name.

### Added
- Async agent loop with the Anthropic Messages API provider.
- Sandboxed tools: `ls`, `view`, `edit`, `grep`, `bash`, `project_knowledge`.
- Encrypted SQLite memory (Argon2id + AES-256-GCM-SIV + Reed-Solomon FEC), WAL mode.
- `ratatui` TUI with Normal / Selection / Visual modes and Unicode-safe input.
- OAuth (PKCE) login and OS keyring integration, with `magi-rust` legacy migration.

[Unreleased]: https://github.com/BolivarTech/magi/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/BolivarTech/magi/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/BolivarTech/magi/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/BolivarTech/magi/releases/tag/v0.2.1
[0.2.0]: https://github.com/BolivarTech/magi/releases/tag/v0.2.0
[0.1.2]: https://github.com/BolivarTech/magi/releases/tag/v0.1.2
[0.1.1]: https://github.com/BolivarTech/magi/releases/tag/v0.1.1
[0.1.0]: https://github.com/BolivarTech/magi/releases/tag/v0.1.0
