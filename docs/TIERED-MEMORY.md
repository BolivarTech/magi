# Tiered Agnostic Memory — Reference

> **Comprehensive reference** for the tiered memory subsystem introduced in
> `magi-rs` after v0.6.0. For hands-on testing see [`docs/E2E-TESTING.md`](E2E-TESTING.md).
> For all config knobs see [`docs/magi.toml.example`](magi.toml.example).

---

## What it is

Before this feature, `magi-rs` stored every conversation turn in an encrypted
SQLite file and loaded **the entire history into every prompt** (`load_all` mode —
still available as the benchmark control). Context grew with history: an O(N)
expansion with no principled bound.

The tiered memory subsystem replaces that approach with **semantic retrieval and
principled forgetting**: every persisted memory is indexed with a vector embedding,
recalled by relevance (cosine similarity re-ranked by recency and salience), and
assembled under a **hard token budget**, independent of how large the history grows.
All storage stays **local and encrypted**; only the context budget and the texts to
embed are sent to the configured provider.

Three pillars govern behaviour (detailed below):

| Pillar | Short description |
|--------|--------------------|
| **P1** — Storage & retrieval | Embed at write; exact or ANN cosine search; composite reranker |
| **P2** — Timely forgetting | Decay model; eviction policy; supersession |
| **P3** — Bounded-context recall | Token-budget assembler with fixed priority; never grows with history |

---

## It IS a RAG — why embeddings alone are not enough

A common question when seeing vector embeddings in a memory system: *"Is this RAG,
or just a store?"* The answer: this feature **implements a full RAG pipeline** —
the three steps of Retrieval-Augmented Generation — end to end.

RAG is three chained steps, and the tiered memory covers each one:

| RAG step | What it does | Where it lives here |
|----------|--------------|---------------------|
| **1. Retrieval** | Find relevant content by semantic similarity | Embedding on write (`REQ-01/02`) → encrypted ANN index in RAM (`D-05`) → `recall(query, budget, scope)` (`REQ-04`, `D-13`) |
| **2. Augmentation** | Inject retrieved content into the prompt under a budget | Composite reranker (similarity + recency + salience, `REQ-05`) → token-budget assembler (`REQ-12/13/14`) |
| **3. Generation** | The LLM responds with the enriched context | The existing `Provider` openai-compat path, unchanged |

An embedding stored without retrieval and injection never reaches the model: using
embeddings **requires** the full RAG pipeline, not just the store.

**No external vector database.** A typical RAG stack adds Pinecone, Chroma, or
pgvector. This implementation does not: the vectors are stored **encrypted inside
the existing SQLite file** (`CryptoVault`), decrypted on demand, and an in-RAM
index (exact cosine by default) is rebuilt per session. An external vector DB would
store vectors in the clear and break the privacy guarantee.

**Agentic-memory RAG, not document RAG.** Plain document RAG retrieves from a
static corpus and injects. This system goes further:

- **Forgetting / decay** — obsolete memories lose strength and are evicted (P2).
- **Supersession** — when a fact is updated or contradicted, the old record is
  demoted immediately (reranker soft-supersession) and eventually marked superseded
  by the off-hot-path distiller (hard supersession).
- **Salience + recency** — the reranker weights similarity, wall-clock recency, and
  a per-record salience score; access reinforcement boosts frequently-recalled items
  (bounded to prevent domination).
- **Always-injected preference profile** — a compact distilled profile of durable
  user preferences is included in every context, not just when retrieved by query.
- **Bounded context growth** — the assembler enforces a hard token cap; the context
  never grows with history depth.

---

## Three pillars

### P1 — Efficient storage and retrieval

**Write path.** When a new memory is persisted (turn, tool result, fact), the
`EmbeddingProvider` generates a vector. Both the text and the vector are encrypted
by `CryptoVault` before touching the disk. Salience is assigned **at write time**
by a deterministic heuristic (function of `kind`, salience markers, and structural
signals) — the LLM is never in the write hot path. Preferences (`kind=preference`)
receive the protected floor (`preference_salience = 1.0`).

**Read / index.** At session start, the encrypted vectors are loaded, decrypted, and
an in-RAM index is built. The default index is a brute-force exact cosine
(`BruteForceIndex`); an HNSW ANN variant is available under `--features ann` for
large corpora.

**Provider agnosticism.** The `EmbeddingProvider` trait is implemented by
`OpenAiCompatibleEmbedder` — the same `base_url` / `api_key` pattern as the chat
provider. Pointing `[embedding].base_url` at Ollama (default), OpenAI, Qwen Cloud,
or any compatible endpoint requires **no code change**. Asymmetric task prefixes
(`query_prefix` / `document_prefix`) are applied automatically; they default to the
recommended `nomic-embed-text-v2-moe:latest` values.

**Retrieval (`recall`).**
```
recall(store, embedder, clock, cfg, query, budget, scope) -> Vec<RankedMemory>
```
This is the **stable public API** (design decision D-13): the context assembler
consumes it, and future agent-society consumers can call it directly without pulling
in token-accounting logic. The composite reranker score is:

```
score = w_similarity · cos_sim
      + w_recency    · 0.5^(age_days / decay_half_life_days)
      + w_salience   · salience
```

Weights are configurable (`weight_similarity`, `weight_recency`, `weight_salience`).
Retrieval updates `access_count` and `last_accessed_at` only on returned hits.

**Model / dimension pinning.** Each vector record carries its `model_id` and `dim`.
The retrieval index only scores vectors whose `model_id` matches the active embedder;
vectors from a different model/dimension are excluded and flagged for lazy re-embed.
This allows switching embedding models without corrupting the index.

---

### P2 — Timely forgetting

**Strength model.** Every memory has a deterministic strength score in `[0, 1]`:

```
strength = (w_rec · recency  +  reinforcement  +  w_sal · salience)
           / (w_rec + 1 + w_sal)

recency        = 0.5^(age_days / decay_half_life_days)        # wall-clock decay
reinforcement  = min(access_count, access_saturation_cap)
                 / access_saturation_cap                       # bounded; u64 saturating
salience       = memory.salience                               # [0, 1]
```

All time-dependent quantities use an **injected `Clock`** abstraction — no
`SystemTime::now()` inside the decay logic. This makes decay deterministic under a
fixed clock (benchmark / test) while using the real clock in production.

`access_count` is `u64` incremented with `saturating_add(1)` (no panic, no wrap).
The access saturation cap (`access_saturation_cap = 50`) bounds the reinforcement
contribution so a heavily-used memory cannot dominate the ranking indefinitely.

**Eviction policy.** Memories with `strength < forget_strength_threshold` are
eligible for eviction — **except** preferences and any memory with
`salience >= protect_salience_threshold`. Eviction semantics are controlled by
`evicted_retention_days`:

| Value | Effect |
|-------|--------|
| `-1` (default) | Archive — never hard-delete; evicted memories are excluded from retrieval |
| `0` | Hard-delete immediately |
| `N > 0` | Archive; hard-delete after N days |

A hard size ceiling (`max_records`) evicts the weakest non-protected memories first
when the store exceeds the limit.

**Supersession.** When a new fact updates or contradicts an older one on the same
subject:
- **Soft (immediate, deterministic):** the reranker's recency term makes the newer
  fact rank above the older one. No extra mechanism needed.
- **Hard (off-hot-path, distiller):** the distiller identifies candidate pairs by
  embedding similarity (≥ `supersede_similarity_threshold`) and asks the configured
  LLM whether they contradict. On a positive judgment, the older record's
  `superseded_by` field is set and the record is excluded from future retrieval.
  For preferences, latest-wins applies without any LLM call (deterministic).

---

### P3 — Recall within a bounded context

**Token accounting.** Token counts are estimated by a deterministic heuristic:
`ceil(char_count / chars_per_token)`, default `chars_per_token = 3.5`
(conservative, Spanish-friendly; use `~3.0` for code-heavy sessions, `~2.0` for CJK).
A safety margin (`safety_margin_ratio = 0.1`) is subtracted from the budget before
assembly, guarding against heuristic underestimation.

**Assembler priority (fixed).** `assemble_selective` fills the context in strict
priority order:

```
1. system prompt         — always included (if non-empty)
2. preference profile    — always included; never dropped
3. ranked episodic recalls — greedily filled, highest score first
4. current turn          — always the last message; never silently dropped
```

If the current turn alone exceeds the budget, it is **truncated with an explicit
notice** (`notices` field on `AssembledContext`), preserving system + profile. An
error is returned only when even system + profile cannot fit — which signals a
misconfigured budget.

**Bounded growth.** The assembled context per turn is always `<= budget_after_margin`
regardless of how many sessions or turns are stored. History depth has zero effect on
context size.

**`mode = "load_all"` (v0.6.0 control).** With `memory.mode = "load_all"`, the
agent loads the full `messages` table into context every turn — exactly the pre-tiered
behaviour. This is the benchmark control arm; it is not removed.

---

## Architecture — module map

| Component | Source file | Responsibility |
|-----------|-------------|----------------|
| `EmbeddingProvider` trait + `OpenAiCompatibleEmbedder` | `src/memory/embedding.rs` | `embed(texts) → Vec<Vec<f32>>` via `POST {base_url}/embeddings`. Asymmetric task prefixes (D-04). Typed errors + graceful degradation (REQ-29). |
| `VectorStore` trait + `SqliteVectorStore` | `src/memory/store.rs` | `memories` table (SQLite). Text **and** vectors encrypted via `CryptoVault`. Salience assigned at write. Shares the `Arc<Mutex<Connection>>` and `derived_key` from `EncryptedSqliteMemory`. |
| `MemoryConfig` / `EmbeddingConfig` | `src/memory/config.rs` | Serde structs for `[memory]` and `[embedding]` in `magi.toml`. `deny_unknown_fields`; all defaults in `src/defaults.rs`. |
| `Clock` trait + `SystemClock` / `FixedClock` | `src/memory/clock.rs` | Time abstraction: production = real clock; tests/benchmark = deterministic virtual clock (R-06). |
| `strength` / `run_forgetting` / `enforce_size_cap` / `purge_expired_archives` | `src/memory/decay.rs` | Strength formula (recency + reinforcement + salience); eviction passes with D-07 retention semantics. |
| `BruteForceIndex` / `InstantDistanceIndex` | `src/memory/index.rs` | In-RAM index. Exact cosine (default); HNSW via `--features ann`. Never persisted in clear. |
| `recall` + `RankedMemory` | `src/memory/retrieval.rs` | **Public stable API (D-13/B1)**: composite reranker, lazy re-embed, scope filtering. Independent of the assembler. |
| `assemble_selective` / `AssembledContext` | `src/memory/context.rs` | Token-budget assembler. Consumes `recall`. Fixed priority. Truncation policy. Gate for `mode`. |
| `DistillJudge` trait / `distill` / `render_profile` | `src/memory/profile.rs` | Preference profile (always injected). Off-hot-path distiller with LLM judgment surface isolated behind a trait (mockable in tests). Hard supersession. Boost salience. |
| Salience heuristic | `src/memory/salience.rs` | Deterministic salience assignment at write time (D-11). Configurable `salience_markers`. |
| Token heuristic | `src/memory/tokens.rs` | `estimate_tokens`, `budget_after_margin`. Deterministic; no BPE. |
| `SqliteVectorStore::diagnostics` | `src/memory/store.rs` | Startup notice: active / archived / pending-re-embed counts. |
| Benchmark binary | `src/bin/bench_memory.rs` | Two-arm benchmark (see below). |

---

## Configuration

The `[memory]` and `[embedding]` sections are optional; all values have built-in
defaults. The full reference with inline documentation is
[`docs/magi.toml.example`](magi.toml.example). Key knobs:

```toml
[memory]
mode = "selective"                  # selective (default) | load_all (v0.6.0)
context_budget_tokens = 8000        # token budget for the assembled context
response_headroom_tokens = 1024     # reserved for the model's reply
safety_margin_ratio = 0.1           # fraction of budget held back
chars_per_token = 3.5               # heuristic: es ~3.5, code ~3, CJK ~2
oversized_turn_policy = "truncate"  # truncate | error
top_k = 12                          # retrieval candidates
weight_similarity = 1.0             # reranker weights (relative)
weight_recency = 0.3
weight_salience = 0.5
default_salience = 0.3              # base salience at write
preference_salience = 1.0           # protected floor for preferences
protect_salience_threshold = 0.9    # salience >= this => never evicted
decay_half_life_days = 30.0         # wall-clock recency half-life
access_saturation_cap = 50          # bound on access-reinforcement contribution
forget_strength_threshold = 0.1     # strength below this => eviction candidate
evicted_retention_days = -1         # -1 archive | 0 hard-delete | N>0 after N days
max_records = 50000                 # hard ceiling on active records (0 = opt-out)
supersede_similarity_threshold = 0.85
distill_every_n_turns = 20          # 0 = on-demand / session-close only
distill_on_session_close = true
distill_enabled = true              # false = zero memory egress for distillation
distill_max_batch_tokens = 4000     # max tokens per distillation pass (privacy cap)
profile_max_tokens = 1024           # always-injected preference profile bound
seed = 42                           # index / retrieval determinism seed

[embedding]
provider = "openai"                          # openai-compat (only option)
base_url = "http://localhost:11434/v1"       # Ollama default; any compatible endpoint
model = "nomic-embed-text-v2-moe:latest"    # default; dim auto-detected
dim = 0                                      # 0 = autodetect from first response
query_prefix = "search_query: "             # asymmetric prefix for nomic
document_prefix = "search_document: "
```

**API keys never in `magi.toml`.** The embedding provider key is read from the
`OPENAI_API_KEY` environment variable only. For a local Ollama server, set it to
the dummy `"ollama"` (or leave it unset — the fallback applies). Real cloud
providers will 401 loudly if the key is absent or invalid.

`[memory]` and `[embedding]` use `deny_unknown_fields`: a typo or an `api_key`
field causes a parse error at startup, not silent acceptance.

---

## Provider and backend

Chat and embedding providers follow the same **openai-compat** pattern:

- **Default**: Ollama at `http://localhost:11434/v1`, embedding model
  `nomic-embed-text-v2-moe:latest` (asymmetric prefixes pre-configured).
- **Any cloud**: set `[embedding].base_url` and `[embedding].model`; pass the API
  key via `OPENAI_API_KEY`. Works with OpenAI embeddings, Qwen DashScope, Cohere
  (via compatibility layers), etc.
- **Provider-agnostic**: the `EmbeddingProvider` trait is the only abstraction
  boundary; the vector store and retrieval logic do not know which backend computed
  the vectors.

Failure modes are typed (`EmbeddingError::Auth`, `::RateLimited`, `::Timeout`,
`::BadResponse`, `::Other`) and result in graceful degradation: the failed memory is
persisted text-only with an empty embedding (flagged for lazy re-embed on the next
session), never a panic or data loss.

---

## Encryption at rest

The same `CryptoVault` that protects conversation messages also protects memory:

```
passphrase (chosen by the user; never stored, on disk or anywhere else)
  └─ Argon2id (OWASP 2025: 64 MiB, t=3, p=4) + per-database salt
       └─ key-encryption key
            └─ unwraps the 32-byte data key (derived once per process)
                 ├─ AES-256-GCM-SIV (nonce-misuse resistant, fresh OsRng nonce per record)
                 │    └─ Reed-Solomon FEC (bit-rot recovery)
                 │         └─ encrypted `text_blob` (memory text)
                 └─ same cipher/FEC
                      └─ encrypted `embedding_blob` (float32 vector)
```

The salt and the wrapped data key are the only things the file carries about the key, and
neither opens it. Forgetting the passphrase means the data is gone; that is the point of the
design, not a gap in it.

The **ANN index lives only in RAM** — vectors are decrypted to memory at session
start and the in-RAM index is rebuilt. Nothing in the SQLite file ever contains a
vector or memory text in clear.

---

## Tool approval policy

The `Tool` trait carries a `requires_approval()` method (default `true`).
Tools are grouped as follows:

| Approval | Tools |
|----------|-------|
| Auto-approved (no prompt) | `view`, `ls`, `grep`, `project_knowledge` |
| Requires user approval | `bash`, `edit`, `consult` |

For the MAGI `consult` tool specifically, `[magi].auto_approve = true` (in
`magi.toml`) auto-approves autonomous consensus launches from the agent tool loop.
A TUI notice is emitted before the run so the user knows the multi-model call is in
progress. The explicit `/consult` TUI command is always user-initiated and is never
gated.

---

## Benchmark

Run both arms (baseline `load_all` vs treatment `selective`) against the synthetic
dataset:

```sh
cargo run --release --bin bench_memory
```

Outputs `target/bench_memory_report.json` and a human-readable summary to stdout.

**Headline result (SC-29):** `selective` matches `load_all` recall accuracy at
roughly **35% of context tokens per turn**, with a lower staleness rate from
superseded facts. The run is fully deterministic under `seed = 42` and the fixed
dataset, so re-running produces identical numbers.

Metrics reported: recall accuracy (cross-session), mean context tokens per turn,
staleness rate. The benchmark is self-contained (no encrypted SQLite, no live
embedder) and uses a `DeterministicEmbedder` seeded from `SHA-256(text)` so it
exercises the retrieval/ranking logic without any provider call.

---

## Rollback

To revert to the v0.6.0 "load all history" behavior:

```toml
# in magi.toml
[memory]
mode = "load_all"
```

The agent will load the full `messages` table per turn. The `memories` table (tiered
store) is ignored but left intact.

To also purge the tiered-memory table:

```sql
-- open .magi-rs-memory.db with any SQLite client
DROP TABLE memories;
```

This removes only tiered-memory records. The `sessions`, `messages`, and `knowledge`
tables are unaffected — conversation history and project knowledge are preserved.

---

## End-to-end testing

See [`docs/E2E-TESTING.md`](E2E-TESTING.md) for a step-by-step walkthrough of
cross-session recall, preference persistence, and rollback verification against a
running Ollama backend.
