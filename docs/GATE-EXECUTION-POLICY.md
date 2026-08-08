# Gate execution policy

**Status:** authorized by the project owner on 2026-08-07, during MS2's §6 gate.

`CLAUDE.local.md` §0.1 lists seven verification gates and requires them green before every commit.
This document records an authorized refinement of **when** each one runs. It does not remove a
gate, lower a threshold, or make any of them optional: every gate still has to be green before a
**round** of work is considered verified, and a red gate still blocks.

## The problem this solves

Measured during MS2's Loop 1, on a 6-physical-core box:

| Fix round | Commits | Wall-clock |
|---|---|---|
| A | 13 | 3 h 26 min |
| CE | 11 | 2 h 58 min |

`cargo nextest run` alone measures **492 s**. Across those 24 commits that is 3 h 17 min of tests,
and adding the other six gates accounts for roughly **88 % of the two rounds' entire wall-clock**.
The time was not spent reasoning about the code — it was spent re-running `cargo`.

The suite got slower for a good reason: closing a real coverage gap in `.config/nextest.toml`'s
`heavy` group took it from 157 to 300 tests, and the file's own measurements record the suite going
275 s → 419 s → 492 s. That price is worth paying once. It is not worth paying 24 times.

## The policy

### Per commit — every commit, no exceptions

```
cargo nextest run
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

These three can each fail from an ordinary code edit, so they gate every commit. `cargo fmt --check`
runs **alone**, never piped: a pipe's exit code masks the failure, and that has bitten this project
before.

### Per round — once, before the round's work is declared verified

```
cargo build --release
cargo doc --no-deps
cargo audit
cargo deny check licenses
```

A "round" is one dispatched unit of work: a fix round, a task's full Red-Green-Refactor cycle, or
any batch of commits reported as complete.

**Why each one moved, individually — the justification is per gate, not a blanket exemption:**

- **`cargo audit` and `cargo deny check licenses`** read `Cargo.toml` and `Cargo.lock`. Nothing else
  can change their result. In fix round CE, **one commit out of eleven** touched `Cargo.toml`; the
  other ten runs could not have returned anything different. This is the clearest case and it is
  pure waste.
- **`cargo build --release`** compiles the same code `cargo nextest run` and
  `cargo clippy --all-targets` have already compiled in debug. A release-only failure needs a
  `cfg`/optimization-dependent difference, which is rare and — importantly — is exactly the kind of
  thing that a once-per-round run still catches before anything is reported as done.
- **`cargo doc --no-deps`** can fail on a broken intra-doc link introduced by an ordinary edit, so
  this is the weakest of the four. It stays per round because `deny(rustdoc::broken_intra_doc_links)`
  is already active in the strict modules, meaning `clippy`/`build` surface most of that class at
  commit time.

### Immediate escalation back to per commit

Run the per-round four **at the commit itself**, not deferred, when that commit:

- touches `Cargo.toml` or `Cargo.lock` — the audit and licence gates are then live again;
- adds or changes a `cfg` attribute, a feature flag, or anything conditional on the build profile;
- is the last commit of the round.

## What this policy does not do

- It does **not** make any gate optional. All seven must be green before work is reported complete.
- It does **not** change any threshold, deny-list, or lint level.
- It does **not** apply to a release. A release runs all seven, from a clean tree, at the tag.
- It does **not** license reporting a round as verified without evidence. The rule that the gates'
  **real output** is read — never "this should pass" — is unchanged.

## The other lever, already applied

`.config/nextest.toml`'s `heavy` group went from `max-threads = 2` to `3` on the same date. That
file already named this as the correct lever and gave the reason: 3 heavy tests × 4 Argon2id lanes
= 12 lanes over 12 logical processors, which is the ceiling before oversubscription returns. The
**filter was not touched** — trimming it is what silently rotted twice before, and the file's
coverage claim has to stay true. If intermittent failures reappear, revert the group to `2` first
and leave the filter alone.

## A third option, deliberately not taken

Running a **filtered** test subset during a mini-cycle and the full suite only at the round's end
was considered and **declined for now**. It is the largest theoretical saving and carries the
matching risk: a fix can break a module the filter excluded, and that would surface only at the end
of the round, after several commits have been built on top of it. Revisit only if the two measures
above prove insufficient — and measure before deciding.
