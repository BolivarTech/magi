# End-to-End Testing — Tiered Memory (running application)

This guide walks through exercising **`magi-rs`** as a running application, focused on the
**tiered agnostic memory** subsystem (`memory.mode = "selective"`, the default). The headline
behavior to demonstrate is **cross-session recall**: the agent remembers facts and preferences
from earlier sessions *without* loading the full history into context.

> Backend is **Ollama-first** by default (`provider = "ollama"`, `http://localhost:11434/v1`).
> Any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, a cloud Ollama, …) works by pointing
> `base_url` at it. Nothing here is specific to a single provider.

---

## 1. Prerequisites (Ollama backend)

```powershell
# Ollama daemon running (separate terminal or as a service)
ollama serve

# Embedding model — REQUIRED for selective mode
ollama pull nomic-embed-text-v2-moe:latest

# Chat model — either local (pull) or cloud (signin, no weight download)
ollama signin                 # for :cloud tags (default chat model kimi-k2.6:cloud)
#   - or -   ollama pull <local-model>   and set [openai].model in magi.toml
```

Confirm Ollama answers and the embedding model is present:

```powershell
curl http://localhost:11434/api/tags    # must list nomic-embed-text-v2-moe:latest
```

## 2. Configuration

With **no `magi.toml`**, the built-in Ollama-first defaults apply (`provider = "ollama"`,
`http://localhost:11434/v1`, `memory.mode = "selective"`). To be explicit or to point at a
different chat model, scaffold a state directory and edit the file it writes:

```powershell
magi-rs init                  # creates .magi/ with a reference magi.toml; refuses to overwrite
# edit [openai].model in .magi/magi.toml if your chat model is not kimi-k2.6:cloud
```

`provider` takes one of `ollama`, `openai-compat` or `anthropic`. The older spelling
`provider = "openai"` is not one of them: a file still carrying it stops the process at startup
with the migration guidance, rather than falling back to a default you did not ask for.

API keys never live in `magi.toml`. For the local Ollama server the dummy value is fine:

```powershell
$env:OPENAI_API_KEY = "ollama"
```

## 3. Build and launch

```powershell
cargo build --release
cargo run --release           # launches the TUI
```

**Watch the startup diagnostics line** (operator visibility):

```
memory: 0 active, 0 archived, 0 pending re-embed (~0 KB index)
```

If Ollama is unreachable, a **startup notice** reports graceful degradation (the memory subsystem
is skipped, text-only persistence continues) — the app still runs.

## 4. Headline test — cross-session recall (selective)

**Session A** — plant facts/preferences:

```
> Remember I prefer Rust over Python for systems programming.
> My project is called Magi and uses an encrypted SQLite store.
> (continue a few turns to exercise the tool loop and the bounded context)
```

Exit with `/exit` → this fires `on_session_close` → **distillation** (recurring preferences are
promoted to the always-injected profile).

**Session B** — restart the app:

```powershell
cargo run --release
```

The diagnostics line now shows `memory: N active` (N > 0). Ask:

```
> Which language do I prefer for systems programming?
```

✅ **Expected:** it answers "Rust", recalled from Session A — **without** loading the full history
(retrieved via the preference profile + budgeted semantic recall). This is the behavior that beats
the `load_all` baseline.

## 5. Automated A/B proof (the measurable result, SC-29)

```powershell
cargo run --release --bin bench_memory
```

Prints the headline **`selective recall ≥ load_all: PASS ✓ (SC-29)`** and writes
`target/bench_memory_report.json` (recall accuracy, mean context tokens/turn, staleness rate per
arm). Deterministic: fixed seed, a deterministic embedder (no network).

## 6. Verify encryption at rest (REQ-03)

```powershell
# Memory text must NOT appear in cleartext in the DB file
Select-String -Path .magi/.magi-rs-memory.db -Pattern "Rust" -SimpleMatch
```

✅ **Expected:** no matches — text *and* embeddings are encrypted via `CryptoVault`
(Argon2id → AES-256-GCM-SIV → Reed-Solomon). The in-RAM ANN index is never persisted in clear.

## 7. Graceful degradation (REQ-29)

With the app running in selective mode, **stop Ollama** and send a turn:

- The write path fails best-effort → `WARN [magi-rs]: memory insert failed (non-fatal)` on stderr,
  but **the turn still completes** (never aborts). When Ollama returns, `on_session_open`
  re-embeds the pending records.

## 8. Compare the baseline (benchmark control)

In `magi.toml`, set `[memory] mode = "load_all"` → reproduces the v0.6.0 "load all history"
behavior. Compare context tokens / latency against `selective`.

## 9. The same walkthrough without the TUI

Everything above drives the interactive app, which is the surface a person uses. Since v0.10.0
there is a second one, and it is the surface a script uses: `magi-rs query` and `magi-rs consult`
run headless, read the prompt from standard input, and answer with a single JSON object.

The passphrase travels in `MAGI_PASSPHRASE`. It is also accepted as `-p`, which is convenient
interactively and wrong in a script: a command line is readable by any process on the machine
while the child lives.

```powershell
$env:MAGI_PASSPHRASE = "…"
"Remember I prefer Rust over Python for systems programming." | magi-rs query --output-format json
```

Section 4's cross-session recall is the same test from here — plant in one invocation, ask in the
next, and read the answer out of the JSON:

```powershell
"Which language do I prefer for systems programming?" | magi-rs query --output-format json
```

Four flags carry most of the weight. `-w <dir>` names the directory the `.magi/` walk-up starts
from, so a script does not depend on where it was launched. `--no-memory` makes the invocation
stateless — nothing is persisted, which is what a control arm needs. `--timeout <seconds>` bounds
the wall clock. `--auto` approves the registered tools without prompting, since there is nobody
there to prompt; the hard barriers stay in place.

```powershell
"Summarise src/agent/mod.rs" | magi-rs query -w . --auto --timeout 300 --output-format json
```

`magi-rs consult` puts the prompt in front of the MAGI trio instead of the agent. Adding
`--structured-verdicts` — which needs `--output-format json`, and says so if you leave it out —
puts the `agents` and `consensus` blocks in the answer, so a script reads each seat's verdict
instead of only the prose. The exit code is part of the contract: 0 when the run completed, 2
when the input or the configuration was rejected.

---

## Rollback

To revert to the legacy behavior, set `mode = "load_all"` in `[memory]`. To also drop the
tiered-memory table: `DROP TABLE memories;` in `.magi/.magi-rs-memory.db` (the `sessions`,
`messages`, and `knowledge` tables are unaffected).
