# SDD Progress Ledger — feat/tiered-memory (Tiered Agnostic Memory)

Plan: planning/claude-plan-tdd.md (+ claude-plan-tdd-org.md task bodies). Approved at Checkpoint 2 (GO WITH CAVEATS, 10 MAGI loops).
Mode: serial + TDD-Guard ON. Per-phase commits (§5). Base commit: 205a5e6.

## Phase MVP (T1–T7), then Forgetting (T8–T9), Context (T10–T12), Learning (T13), Validate (T14), Hardening (T15).

- Task 1: in progress (Red)
- Task 1: COMPLETE (commits c491a2b..4cb1d2d — test/feat/refactor; +tooling fix f4b8600; 191/189+2 green, §0.1 clean)
- Task 2: COMPLETE (commits 526bdaf..c33acdd — test/feat; refactor skipped; 194 green; narrow per-item allows pending T3-7 consumers)
- Task 3: COMPLETE (commits 589ae23..9103788 — test/feat; 199 green; embedder: no timeout, key redacted, 429->RateLimited)
- Task 4: COMPLETE (commits e77db48..7c3f4a2 — test/feat+refactor; 203 green; encrypted store, W12 lock-drop, CP2-B; database.rs +2 accessors only)
- Task 5: COMPLETE (commits 4231a66..fc9c71f — test/feat; 207 green; two-phase encrypt-then-IMMEDIATE-tx, content-hash idempotency, atomic)
- Task 6: COMPLETE (commits 221788a..f0f2c8a — test/feat; 210 green; pure deterministic salience, narrow-allow pending T12 wiring)
- Task 8a (Clock split): COMPLETE (commits 38ac7c4..1a0fd7b — test/feat; 213 green; SystemTime confined to clock.rs)
- Task 7: COMPLETE (commit b7b8df5 — feat; DEVIATION: single commit, no separate test: RED, §5; code correct). 226 default + 228 ann green; instant-distance opt-in (ann feature, not in default tree); cargo audit exit 0 (lru/rustls-pemfile warnings pre-existing from ratatui/reqwest).
- === PHASE MVP (T1-T7 + Clock) COMPLETE ===
- Task 8b (decay split): COMPLETE (commits 48e8e58 test / a66f8d7 feat — SEPARATE per-phase; 231 green; strength() pure normalized [0,1], clock-driven, bounded reinforcement)
- Task 9: COMPLETE (commits 7edd30f test / 45eb388 feat — SEPARATE; 239 green; run_forgetting+enforce_size_cap(CP2-Y)+purge+archived; protection, clock-jump cap, soft-supersession via T7 recency)
- Task 10: COMPLETE (commits 137b00b test / a306380 feat — SEPARATE; 243 green; pure token heuristic estimate_tokens+budget_after_margin)
- Task 11: COMPLETE (commits 1f85815 test / a1aaf95 feat — SEPARATE; 247 green; assemble_selective + AssembledContext; compact_history+test removed, clear_history kept)
- Task 12: COMPLETE (commits 8d003a0 test / a07b301 feat / 2d3ea5a refactor — SEPARATE; 251 green; ACTIVATION: write-path + selective assembler in query_streaming, main.rs wires subsystem; load_all byte-identical)
