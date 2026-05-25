# Changelog

All notable changes to **Magi Agent** (`magi-rs`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the **minor** position signals significant or breaking
changes and the **patch** position signals backward-compatible fixes.

## [Unreleased]

### Deferred (tracked in internal dev-docs)
- **#14** Envelope encryption (key rotation / crypto-shredding / multi-tenancy) — enterprise roadmap.
- **#17** Runtime warning visibility — malformed tool-JSON (#4) and poison recovery (#8) warnings remain stderr-only under the alt-screen; startup/login warnings are already surfaced.
- **#18** Blob version-dispatch / migration — the blob version byte is detection-only; a future format bump still needs a migrate-or-reset path.

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

[Unreleased]: https://github.com/BolivarTech/magi/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/BolivarTech/magi/releases/tag/v0.1.2
[0.1.1]: https://github.com/BolivarTech/magi/releases/tag/v0.1.1
[0.1.0]: https://github.com/BolivarTech/magi/releases/tag/v0.1.0
