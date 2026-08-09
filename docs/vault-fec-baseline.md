# ConcatenatedFec — performance baseline (REQ-V36)

> **Performance history, NOT a decision gate.** The FEC choice is fixed by design (D-V02):
> channel resilience is part of the hardening thesis. This number exists to compare against
> future releases, **not** to reopen the decision. Measured with
> `cargo run --release --bin bench_vault_crypto` against `magi-rs` 0.8.0 (rustc 1.97, Windows MSVC).

## Measurement (2026-07-15)

Corpus: **1000 records × 220 B** of plaintext (a representative size for a conversation message).

| Metric | Value |
|---|---|
| Total plaintext | 220 000 B |
| Total encoded (on disk) | 1 364 000 B |
| **On-disk expansion** | **≈ 6.2×** |
| **Decrypt per record** | **≈ 5.3 ms** |
| Total decrypt (1000 records) | ≈ 5.3 s |

## Honest interpretation

**The real expansion (≈6.2×) is greater than the ≈2.3× the spec estimated.** The ≈2.3× applies to
a *large payload* (RS 1.14× × Viterbi 2×); for **small messages** (220 B) the fixed overheads
dominate — the nonce + AEAD tag (~28 B), the RS(255,223) block padding, Viterbi's duplication, and
the outer base64 are all paid against a small payload, so the ratio climbs.

**The decrypt cost (~5.3 ms/record) is Viterbi's cost.** The previous in-tree crypto was
**RS-only (no Viterbi)**; `ConcatenatedFec` **adds** the Viterbi stage (bit-level error correction
+ coding gain), which is computationally heavy. The migration is, by design, **slower and larger**
in exchange for **more channel resilience**.

## Impact on the hot path (`recall()` / history loading)

- **`selective` mode (default):** `recall()` decrypts ~`top_k` (12) memories/turn ⇒ **~64 ms/turn**
  of FEC-decrypt. Negligible against an LLM turn's latency (seconds).
- **`load_all` mode (the benchmark's control, not the default):** loading N messages costs
  N × 5.3 ms ⇒ **~5.3 s per 1000 messages**. This is the additional reason `selective` is the
  default.
- **Loading a session** (`get_messages`): tens of messages ⇒ hundreds of ms. Perceptible but
  bounded.

## Levers if the cost turns out unacceptable (out of scope for MS1)

FEC is an **injectable strategy** (`ErrorCorrection`). If the cost ever weighs too much, the 0.3.0
crate offers `fec::ReedSolomonCodec` (RS-only, ~1.14×, no Viterbi — like the old crypto) or `NoFec`
(1×, AEAD only). **Changing the FEC changes the on-disk format** (D-V02) ⇒ it would require a
migration/reset, which is why it is not a trivial change. Recorded as a known lever, not as MS1
action.
