# OpenRouter as a Magi backend

> How to run `magi-rs` — the agent, its tool loop, the MAGI trio and the memory
> embedder — against [OpenRouter](https://openrouter.ai/), and the four traps
> this backend sets that the Ollama-first defaults do not prepare you for.

**Status: verified working.** On 2026-08-22, `magi-rs` v0.15.0 was exercised
end-to-end against OpenRouter with every subsystem active. Results are in
[Verification record](#verification-record); the configuration those runs used is
tracked as [`magi.toml.openrouter.example`](magi.toml.openrouter.example).

OpenRouter is an aggregator: one OpenAI-shaped endpoint that routes to hundreds
of models from dozens of vendors. That makes it a natural fit for the MAGI trio,
which needs three models from **three independent failure domains** — normally
the hardest part of the configuration to satisfy from a single account.

---

## Quick start

```powershell
# 1. The key travels by environment or vault, NEVER in magi.toml.
$env:OPENAI_API_KEY = "sk-or-v1-..."       # or: magi-rs vault set OPENAI_API_KEY

# 2. Scaffold a workspace and install the OpenRouter profile.
magi-rs init
Copy-Item docs\magi.toml.openrouter.example .magi\magi.toml

# 3. Confirm the transport.
"Reply with one short sentence naming which model you are." | magi-rs query --output-format json
```

`provider = "openai-compat"` and `base_url = "https://openrouter.ai/api/v1"` are
all that selects the backend. `OPENAI_API_KEY` resolves **env > vault**, so an
exported variable wins over a stored one — convenient for a short-lived key.

---

## The four traps

Each of these was hit during the verification study. None of them produces an
error message that names its own cause, which is why they are written down.

### 1. The embedding model in the scaffolded config does not exist here

`magi-rs init` writes `[embedding].model = "nomic-embed-text-v2-moe:latest"`, an
Ollama tag. Against OpenRouter it returns **HTTP 400 `Model does not exist`**,
and the memory subsystem degrades to text-only persistence without failing the
run — so recall quietly stops working while the agent still answers.

Change it. See [Embeddings](#embeddings) for the working set.

### 2. `--timeout` derives the per-mage ceiling *downward*

This is the one that silently degrades a consult, and the direction is
counter-intuitive: passing a **larger** wall clock than you need still yields a
**smaller** per-mage budget than the config asks for.

| invocation | resulting per-mage ceiling | operation budget |
|---|---|---|
| `consult` (no `--timeout`) | **120 s** — taken from `[magi].agent_timeout_secs` | 72 s/attempt |
| `consult --timeout 900` | 124 s — derived | 74 s/attempt |
| `consult --timeout 300` | **40 s** — derived | 24 s/attempt |
| `consult --timeout 180` | **24 s** — derived | 14 s/attempt |

At a 24 s budget the study's first trio run came back **degraded**: two mages
exhausted the budget and rotated, and the third — Balthasar — then found the
fallback pool already consumed by the other two and failed with
`no_fitting_candidate`. A degraded run approves no gate regardless of its
verdict.

`magi-rs` announces which path it took on stderr. Read the line:

```
magi: per-mage ceiling 120s from [magi].agent_timeout_secs (operation budget 72s/attempt)
magi: per-mage ceiling 40s derived from --timeout 300s (operation budget 24s/attempt, max_rotations=2)
```

**Against OpenRouter, prefer omitting `--timeout` entirely.** The default headless
wall clock is already derived from `attempts x (1 + max_rotations) x
agent_timeout_secs` plus slack, so it accommodates the full rotation budget. If
CI demands a hard cap, pass **>= 900 s** and take the warning naming the computed
minimum.

### 3. A 404 that is an account setting, not a missing model

Some models answer:

```
HTTP 404  No endpoints available matching your guardrail restrictions and data policy.
          Configure: https://openrouter.ai/settings/privacy
```

The model exists and is listed in the catalogue; the account's privacy policy
excludes every provider currently serving it. During the study
`qwen/qwen3.7-flash` failed this way while `qwen/qwen3-30b-a3b-instruct-2507`
succeeded — so it is **per-model**, not per-vendor, and cannot be predicted from
the model id. Widen the policy at the linked settings page, or pick another
model.

`magi-rs` classifies this as a `transport` rotation cause and rotates away from
it, which is correct behaviour but spends a rotation on a condition that will
never clear on retry. Prefer removing such a model from the config.

### 4. Free models are unsuitable for the trio

Models with a `:free` suffix draw on a **shared upstream pool**, and under load
return HTTP 429 immediately rather than queueing:

```json
{"error":{"code":429,"metadata":{
  "raw":"z-ai/glm-5.2:free is temporarily rate-limited upstream.",
  "provider_name":"Decart","limit_source":"upstream_provider_shared_pool",
  "retry_after_seconds":5}}}
```

This is independent of the account's own tier — the study's key reported
`is_free_tier: false` and still hit it on the first call. Since a consult fires
three concurrent requests, the free tier is where it is least likely to work.
The paid models in the tracked example cost fractions of a cent per consult; use
them.

---

## Embeddings

**OpenRouter serves `/embeddings`** — worth stating plainly, because the
aggregator's own catalogue suggests otherwise.

The trap is discovery, not availability: **embedding models are absent from the
`/models` endpoint.** Searching that catalogue for `embed` returns zero matches,
and its `output_modalities` field lists only `audio`, `image` and `text`. The
models are reachable but undiscoverable by listing, so the id must be known in
advance. This table is that list, probed directly:

| model | dimensions | measured price | notes |
|---|---|---|---|
| **`baai/bge-m3`** | **1024** | **$0.0002 /Mtok** | recommended — cheapest, smallest vector, multilingual |
| `google/gemini-embedding-001` | 3072 | $0.0000 /Mtok | measured zero; treat as provisional, see below |
| `openai/text-embedding-3-small` | 1536 | $0.0150 /Mtok | |
| `openai/text-embedding-ada-002` | 1536 | — | legacy |
| `openai/text-embedding-3-large` | 3072 | — | |
| `qwen/qwen3-embedding-8b` | 4096 | — | largest vector |
| `qwen/qwen3-embedding-4b` | 2560 | — | |

Prices were obtained by measuring account spend across a fixed batch, not read
from a published table. `gemini-embedding-001` measured **$0.000000** over ~5.9k
tokens — which may be a genuine zero, promotional pricing, or simply below the
sixth decimal the usage endpoint reports. It is not recommended on that basis
alone, and it emits a 3072-dimension vector: three times the on-disk index of
`bge-m3`, for storage `magi-rs` pays on every record.

`baai/bge-m3` is the recommendation on three grounds: 75x cheaper than
`text-embedding-3-small`, the smallest vector in the working set — which is also
the smallest SQLite index — and multilingual, which matters for a store that
retains whatever language the conversation used.

Leave `dim = 0` to autodetect the width from the first response.

### Confirming the embedder actually works

A green transport check does not prove recall. Exercise it across two separate
process invocations, with memory **enabled** (no `--no-memory`):

```powershell
"Remember this fact: the project codename is Aurora Borealis. Reply with just OK." | magi-rs query
"What is the project codename? Answer in one short sentence."                      | magi-rs query
```

Watch the startup notice, which reports the vector store's state:

```
note: memory: 0 active, 0 archived, 0 pending re-embed (~0 KB index)   <- first turn
note: memory: 2 active, 0 archived, 0 pending re-embed (~8 KB index)   <- second turn
```

The second turn should answer *"The project codename is Aurora Borealis."* The
index size is also a cheap confirmation of the autodetected width: 8 KB for two
records is 1024 floats x 4 bytes x 2, i.e. `dim = 1024` as expected.

---

## Choosing models for the trio

Two selection criteria matter more here than price, and neither is obvious.

**Latency is a selection criterion, not a nicety.** All three mages run
concurrently against one derived per-mage budget; a seat that overruns it rotates
away, consuming a fallback and lengthening the run. A model that is cheap per
token but slow per call is a poor seat and a fine fallback.

**A seat must honour the marker protocol.** Each mage is required to emit its
verdict between `<MAGI_VERDICT>` and `</MAGI_VERDICT>`, each alone on its own
line. A model that ignores this fails extraction and burns a rotation without
contributing a verdict — the same cost as being unreachable, but harder to
diagnose because the call itself succeeded.

Measured on an identical code-review payload:

| model | lineage | latency | $/Mtok in | markers |
|---|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | alibaba | 2.9 s | 0.048 | OK |
| `amazon/nova-lite-v1` | amazon | 3.1 s | 0.060 | OK |
| `inclusionai/ling-3.0-flash` | inclusionai | 5.8 s | 0.021 | OK |
| `mistralai/mistral-nemo` | mistral | 8.0 s | 0.019 | OK |
| `openai/gpt-oss-20b` | openai | 21.4 s | 0.030 | OK |
| `z-ai/glm-4.7-flash` | zhipu | 25.8 s | 0.060 | OK |
| `openai/gpt-oss-120b` | openai | 27.2 s | 0.037 | OK |
| `deepseek/deepseek-v4-flash` | deepseek | 42.7 s | 0.060 | OK |
| `google/gemma-3-12b-it` | google | — | 0.050 | **FAILS** |

The tracked example seats the three fastest compliant models from distinct
lineages, and pools the slower compliant ones as fallbacks.
`google/gemma-3-12b-it` is deliberately absent from both: it failed marker
extraction twice in a row during the study, and a model that cannot be extracted
from is worse than one that is merely slow.

**Lineage is declared, never inferred.** OpenRouter's `vendor/model` id looks
like a lineage but is not one — the lineage is *your* declaration of an
independent failure domain, and the same two models may legitimately be one
domain for one operator and two for another. Ensure no fallback shares a lineage
with a seat, so any rotation preserves the three-way diversity that is the whole
point of the mechanism.

---

## What "unmeasured" means here, and why it is not a failure

Every OpenRouter run emits these two notices:

```
<model>: probe: this endpoint does not offer model introspection (not a failure)
notice: lineage diversity was not corroborated against weights digests for every declared
        model, because some digests are unresolved. The declarative checks still applied.
```

Both are expected and neither indicates a problem. Context-window and
weights-digest probing require introspection endpoints that only an Ollama daemon
exposes; `openai-compat` has none, so every model resolves to *not measurable*.

Three consequences follow:

- Consult output reports `ran_unmeasured` listing every mage, and the report
  carries a note that a surviving model ran on an estimated context window.
- `strict_context_guard` is **declined automatically**. `magi-rs` passes it to
  `magi-core` only when at least one candidate has a measured window; enabling it
  with nothing measured would reject every candidate and switch rotation off
  entirely. The decline is announced.
- Lineage diversity is still enforced **declaratively** — three distinct labels
  are checked and a violation is a load error. Only the empirical corroboration
  against weights digests is unavailable, and that path warns rather than blocks.

If you need measured windows, point `[magi].base_url` at an Ollama daemon and
leave the main agent on OpenRouter; the two are configured independently.

---

## Debugging with `curl`: one artifact to expect

On the **non-streaming** path, OpenRouter pads slow generations with keepalive
whitespace lines before the JSON body:

```
$ curl ... -d '{"model":"openai/gpt-oss-120b", ...}' | head -c 40
\n         \n\n         \n\n         \n\n
```

The body is valid JSON after the padding and parses normally, but a naive reader
that inspects the first bytes will misreport a failure. `magi-rs` is unaffected:
it requests `stream: true` and consumes SSE, where the padding does not appear.
Do not read a `curl` artifact as a provider fault.

---

## Verification record

Performed 2026-08-22 against `magi-rs` v0.15.0 (`c9b70d8`), Windows 11.

| # | Surface | Command | Result |
|---|---|---|---|
| 1 | Transport | `query --no-memory --output-format json` | `stop_reason: done`, 373 in / 345 out, **3.0 s** |
| 2 | Tool loop | `query --auto --no-memory` | two calls (`ls` then `view`), correct answer, **6.4 s** |
| 3 | MAGI trio | `consult --mode code-review` | `degraded: false`, `failed_agents: {}`, **0 rotations**, 0 extraction failures, **25.9 s** |
| 4 | Embedder | two `query` turns, memory active | wrote then recalled across processes; index confirmed `dim = 1024` |

Total account spend for the entire study, including every discarded
configuration and all direct probing: **$0.012**.

Run 3 used the tracked example's configuration. An earlier trio attempt under
`--timeout 300` returned `degraded: true`; that run is retained above as trap 2
rather than omitted, because the failure is the useful part.

---

## See also

- [`magi.toml.openrouter.example`](magi.toml.openrouter.example) — the verified configuration
- [`magi.toml.example`](magi.toml.example) — every key documented, against the Ollama default
- [`TIERED-MEMORY.md`](TIERED-MEMORY.md) — what the embedder feeds
- [`E2E-TESTING.md`](E2E-TESTING.md) — exercising the running application
