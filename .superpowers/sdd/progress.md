# SDD Progress Ledger — feat/tiered-memory (Tiered Agnostic Memory)

Plan: planning/claude-plan-tdd.md (+ claude-plan-tdd-org.md task bodies). Approved at Checkpoint 2 (GO WITH CAVEATS, 10 MAGI loops).
Mode: serial + TDD-Guard ON. Per-phase commits (§5). Base commit: 205a5e6.

## Phase MVP (T1–T7), then Forgetting (T8–T9), Context (T10–T12), Learning (T13), Validate (T14), Hardening (T15).

- Task 1: in progress (Red)
- Task 1: COMPLETE (commits c491a2b..4cb1d2d — test/feat/refactor; +tooling fix f4b8600; 191/189+2 green, §0.1 clean)
- Task 2: COMPLETE (commits 526bdaf..c33acdd — test/feat; refactor skipped; 194 green; narrow per-item allows pending T3-7 consumers)
- Task 3: COMPLETE (commits 589ae23..9103788 — test/feat; 199 green; embedder: no timeout, key redacted, 429->RateLimited)
