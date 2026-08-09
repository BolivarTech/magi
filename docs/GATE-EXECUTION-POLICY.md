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
python scripts/scoped_tests.py
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

These three can each fail from an ordinary code edit, so they gate every commit. `cargo fmt --check`
runs **alone**, never piped: a pipe's exit code masks the failure, and that has bitten this project
before.

`scoped_tests.py` runs the touched modules rather than the whole suite — see "The third measure"
below for why, and for the fail-safe that makes it narrow only when it can do so confidently.

### Per round — once, before the round's work is declared verified

```
cargo nextest run          <- the FULL suite, mandatory
cargo build --release
cargo doc --no-deps
cargo audit
cargo deny check licenses
```

A "round" is one dispatched unit of work: a fix round, a task's full Red-Green-Refactor cycle, or
any batch of commits reported as complete.

**The full suite belongs here and is not optional.** It is not "the per-commit run again": the
per-commit run narrows *which tests execute*, and this is the only run that executes them all, so
it is the only thing that catches a behavioural regression in a module the per-commit filter
excluded. If it ever falls out of this list, the split stops being a split and becomes a loss of
coverage.

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

## The third measure, adopted 2026-08-08 on new evidence

Running a **filtered** test subset per commit, with the full suite at the round's end, was declined
when this document was first written, to be revisited "only if the two measures above prove
insufficient". They did prove insufficient — but for a problem that was not visible at the time, so
this is not a reversal of the earlier reasoning.

**The problem the first two measures could not solve.** The full suite runs 400–600 s and the agent
tooling's command ceiling is **600 s**. Agents were therefore backgrounding the run and ending their
turn — which strands them, because a background task only wakes an agent that is still running.
**Seven stalls in one milestone** traced to that single squeeze, each costing a detection turn plus
a resume turn at full context. Raising the concurrency cap and deferring four gates cut the *cost*
of the suite; neither moved it clear of the *ceiling*.

**How it works.** `python scripts/scoped_tests.py` derives a nextest filter from `git diff` and runs
it. Per commit, use it in place of a bare `cargo nextest run`. At the round's close, run the full
suite.

**The gap, stated plainly, and what covers it.** Per commit this narrows *which tests execute*, not
*whether the code compiles*: `cargo clippy --all-targets` and `cargo build` still cover every
target, and they catch the common cross-module break — a changed signature. What it can miss is a
**behavioural** regression in a module the filter excluded. The round-closing full suite is what
catches that, and it is the reason the full suite remains non-negotiable rather than optional.

**Why it is derived and never hand-maintained.** A hand-written list is precisely the failure mode
`.config/nextest.toml` documents twice: a filter that matches nothing raises **no error**, so it
stays declared in the file and silently stops applying. The script maps paths to modules
mechanically, and **every case it cannot map confidently — a crate root, a manifest, a build script,
a shared test helper, an unrecognised path — falls back to the full suite.** The default direction
is always "run more", never "run less". A documentation-only change runs the full suite too, rather
than reporting a vacuous pass over nothing.

**Measured on this repository's last 40 commits:** 28 would run scoped, 11 would fall back to the
full suite, 1 touched documentation only. So the saving applies to roughly seven commits in ten,
and the fallbacks are exactly the commits where narrowing would have been least safe.
