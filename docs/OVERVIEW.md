# Magi Agent — Overview & Philosophy

> What Magi is, the `magi-core` foundation it builds on, and the multi-perspective
> philosophy that gives the project its name.

---

## What is Magi?

Magi Agent (`magi-rs`) is a **local-first terminal AI assistant written in Rust**. It drives an LLM through a multi-turn tool loop — reading and editing files, running sandboxed shell commands, searching the tree — while keeping the entire conversation in an **encrypted local store**. The guiding constraint is simple: *your code stays on your machine.* The only thing that ever leaves is the model API call you explicitly approve.

That makes Magi a fit for environments where source cannot be shipped to a third-party SaaS — regulated industries, on-prem and air-gapped setups, privacy-conscious developers, and teams running their own models.

Design principles:

- **Local-first** — a single static binary; state lives in an encrypted SQLite file, not a vendor vault.
- **Encrypted at rest** — Argon2id → AES-256-GCM-SIV → Reed-Solomon FEC; the database is unlocked by a passphrase you choose, which is never stored anywhere. Nothing on disk opens it without that passphrase — there is no keyring entry, no cached copy, no recovery path. The LLM API key is a separate secret, kept *inside* the store the passphrase opens.
- **Sandboxed** — filesystem tools are confined to the workspace; the shell tool runs a strict command allowlist.
- **Human-in-the-loop** — every tool call is approved before it runs.

---

## The `magi-core` foundation

Beneath the agent sits [`magi-core`](https://crates.io/crates/magi-core) — a separate, **LLM-agnostic Rust crate** that implements **multi-perspective consensus**. It is the conceptual heart of the project and the reason for the name.

The philosophy is a deliberate rejection of single-oracle thinking: **no single model, and no single perspective, is trusted blindly.** A decision is reached by having several independent evaluators analyze the same problem through different lenses, then reconciling their verdicts. Because `magi-core` is backend-agnostic, those evaluators can be different models, different providers, or the same model under different instructions.

Magi Agent is designed to build on `magi-core` for a multi-perspective **`consult`** workflow — putting a design, a diff, or a decision in front of the panel before you commit to it.

---

## Multi-perspective analysis — the MAGI philosophy

The name comes from the **MAGI** of *Neon Genesis Evangelion* (1995, Hideaki Anno / Gainax): three supercomputers — **Melchior**, **Balthasar**, and **Caspar** — that govern critical decisions, each embodying a different facet of their creator, Dr. Naoko Akagi. No single unit decides; consensus is required.

That fiction encodes a real insight: **complex decisions benefit from structured disagreement.** A single decision-maker, however capable, carries blind spots. Three independent evaluators with different priorities surface risks and trade-offs that any one of them would miss.

### The three lenses

| Agent | Codename | Lens |
|-------|----------|------|
| **Melchior** | Scientist | Technical rigor and correctness |
| **Balthasar** | Pragmatist | Practicality and maintainability |
| **Caspar** | Critic | Risk, edge cases, and failure modes — adversarial by design |

### Why adversarial perspectives

The model directly counters well-documented engineering biases:

| Bias | How multi-perspective mitigates it |
|------|------------------------------------|
| **Confirmation bias** | Three agents with different criteria rarely share the same blind spots |
| **Anchoring** | Each analyzes independently — no agent sees the others before forming its verdict |
| **Groupthink** | Caspar is built to find fault, not to agree |
| **Optimism bias** | Scoring penalizes *reject* more heavily than *approve*, so negative signals are hard to override |
| **Status-quo bias** | Each agent evaluates from first principles, not from "how it's usually done" |

**Disagreement is a feature, not a failure.** When Melchior approves but Caspar rejects, the dissent surfaces a genuine tension between correctness and risk tolerance. Verdicts are reconciled by a weight-based vote (`approve = 1`, `conditional = 0.5`, `reject = -1`) into a single consensus — from **STRONG GO** through **GO WITH CAVEATS** to **HOLD** and **STRONG NO-GO**.

The approach earns its cost on decisions with **genuine uncertainty**, **significant consequences**, or **hidden trade-offs**. For trivial questions with one clear answer, a complexity gate skips the panel and answers directly.

---

## Tiered Memory

Magi's memory subsystem has evolved beyond "persist and reload": it now runs a
full in-process RAG pipeline — embed each memory, retrieve by semantic relevance,
forget the stale, and assemble context within a hard token budget, all with
encrypted local storage. See [`docs/TIERED-MEMORY.md`](TIERED-MEMORY.md) for the
full technical reference.

---

## How it comes together

Magi is two halves of one tool: an **interactive agent** you drive day to day, and a **multi-perspective panel** (via `magi-core`) you call on for the decisions that matter. The agent gets work done while your code never leaves your machine; the panel makes its reasoning auditable — cross-examined by independent perspectives before you act on it.

---

## Credits

The MAGI concept originates from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995) by Hideaki Anno / Gainax, where three supercomputers govern critical decisions through structured consensus, each embodying a facet of their creator, Dr. Naoko Akagi. Magi adapts that multi-perspective decision-making philosophy to software engineering, where the three facets become three analytical lenses: technical rigor, pragmatism, and adversarial risk assessment.
