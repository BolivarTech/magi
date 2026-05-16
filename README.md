# Magi Agent (`magi-rs`)

Terminal AI assistant in Rust. Drives an LLM provider through a multi-turn
tool loop with sandboxed filesystem and shell access, and persists every
conversation to a locally-encrypted SQLite store.

> **Status: 0.1.0 pre-release.** Published primarily to reserve the crate name.
> See [github.com/BolivarTech/magi](https://github.com/BolivarTech/magi) for
> development status. An internal security audit identified issues that must be
> resolved before this version is recommended for production use.

## Highlights

- **TUI front-end** built on `ratatui` with Normal / Selection / Visual modes.
- **Multi-turn agent loop** with bounded tool calls, repetitive-call detection,
  and an interactive approval gate.
- **Anthropic Messages API provider** with SSE streaming and 429 retry.
- **Sandboxed tools**: `ls`, `view`, `edit`, `grep`, `bash` (strict allowlist),
  and `project_knowledge` (persistent facts).
- **Encrypted memory**: per-record Argon2id key derivation, AES-256-GCM-SIV
  authenticated encryption, Reed-Solomon FEC against bit-rot.
- **OAuth login** via PKCE against the Anthropic Console.
- **OS keyring** integration (`magi-rs` primary, `magi-rust` legacy migration).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
