# The smoke harness

This is not part of `cargo test`. It builds the release binary, points it at a real
backend and a real database, and asks whether the product a user installs does what the
documentation says it does.

The unit suite answers a different question, and answers it well: does each piece honour
its own contract? What it cannot see is the defect where no piece violates anything and
the trouble only shows up with all of them running together, in release, against a
backend that can refuse.

Four kinds of defect are invisible from inside the suite. Every one has cost this
project a release or a day:

- **artifact and platform**, a binary that will not start on another distribution;
- **a real backend**, because a mock always answers, and answers correctly;
- **the build profile**, where a `debug_assert!` release removes takes a guard with it;
- **published documentation**, because nothing executes prose.

## What it is not

It does not replace the suite, and it does not repeat the suite outside cargo. What goes
in here is what the suite *cannot* see.

It is also not a design gate. Judgement about architecture belongs to the review gate,
not to a program. And it is deliberately a handful of properties rather than an
exhaustive battery: a harness that grows without limit stops being run, and a gate nobody
runs protects nothing.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | every scenario passed |
| 1 | a scenario did not pass |
| 2 | the preflight cut before anything ran |
| 3 | **the harness itself failed** |

That last one is why the codes are separate. A bug in the harness must never read as a
verdict on magi-rs, so an exception originating here, a failed reconciliation between
what was registered and what reported, and an invalid `smoke.toml` all exit 3. An
exception raised while interpreting the *product's* output is a different event, and it
becomes a `FAIL`.

## Outcomes

Four, not two.

| Outcome | Meaning | Blocks the gate |
|---------|---------|-----------------|
| `PASS` | the assertion held | no |
| `FAIL` | it did not | yes |
| `CANNOT_TEST` | it was promised, and the environment refused to let it run | yes |
| `OUT_OF_SCOPE` | it was never promised | no |

`CANNOT_TEST` exists because without it everything that could not be checked disguises
itself as green. A scenario that could not run is reported as not run.

## Setting up

```bash
cp smoke/smoke.toml.example smoke/smoke.toml   # then fill it in
python -m smoke --init-env
```

`smoke/smoke.toml` holds the passphrase to the harness's own throwaway vault, so the
preflight refuses to start if anyone else can read the file. It is gitignored. The
example beside it is tracked and carries no real values.

You also need the backend credential exported under whatever name `smoke.toml` gives in
`[backend].key_env`, which is `OPENAI_API_KEY` by default. Against a local Ollama daemon
any non-empty value works, since the daemon ignores it. If it does not resolve the
preflight cuts at step 3, before a single backend run has been paid for.

`smoke/env/` is the persistent environment: a real `.magi/` with a real database that
accumulates history across runs. The persistence is the point, not an optimisation. What
a user actually hits is a database that already has data in it, and an environment
rebuilt every run only ever exercises the first launch, which is the one case that never
breaks. Two scenarios say so out loud. Over an empty environment they report
`CANNOT_TEST`, never `PASS`.

`--reset-env` destroys and rebuilds it when a scenario has corrupted it.

## Running it

Three invocations, and they are not interchangeable.

```bash
python -m smoke --smoke-1                              # after the TDD cycle, before the review gate
python -m smoke --smoke-2 --profile smoke/smoke.toml   # iterating after the gate
python -m smoke --smoke-2                              # the run that certifies
```

Smoke 1 runs when the implementation is finished and before the review gate opens.
Spending a gate that costs several model invocations to find out the code does not work
is pure waste. A design review should not be the thing that tells you it is broken.

Smoke 2 runs after the gate closes, because the gate *edits the code*. A change made to
satisfy a reviewer can break what already worked, and without a second pass that
regression reaches the release under a verdict that no longer covers it.

Only the third form emits a certificate: `--smoke-2`, no `--profile`, nothing failed and
nothing left untested. A cheap-profile run never certifies, however green it comes out.

## The cheap profile

`[profile.cheap]` in `smoke.toml.example` is a candidate list. It is not a declaration.

No model has been measured against magi-rs's own prompts yet, so this repository has
never run with `--profile`, and every number below was taken with the product's own
defaults. The rule the project holds itself to is that no cheap model gets declared
without being measured. The honest state of that work is: not done.

The list inherited from MAGI-Core covers verdict-shaped properties only. The main
agent's tool-calling axis, which S5 needs, was never exercised there, so treat the
exclusions below as a starting point:

| Model | Why it was excluded |
|-------|---------------------|
| `gemma4:cloud` | never reasons |
| `gpt-oss:20b-cloud` | ignores `think:false` |
| `nemotron-3-nano:30b-cloud` | truncates its verdict JSON about one run in three |
| `mistral-large-3:675b-cloud` | invalid JSON, and never reasons |

When a profile is finally declared, its expected reds belong here with the written cause
of each. A deterministic red with a reason beside it can be audited. An intermittent one
gets rationalised until somebody deletes the assertion.

## What it costs

Measured on 2026-08-26 against a local Ollama daemon, product defaults, on a green
run: **558 seconds end to end** for the preflight, eight product runs and nineteen
scenarios. Two individual numbers were taken directly, because they are the ones that
decide the total:

| Run | What it does | Wall clock |
|-----|--------------|-----------:|
| R4 | consult, 250 kB payload, structured verdicts | 103 s |
| R6 | query against a backend that answers 401 | 18 s |

The other six are short queries and consults; nothing measures them individually, and a
table of invented per-run numbers would be worse than none.

**R4 no longer reproduces failure #4, and that is a deliberate trade.** The spec picks
S7 as the direct reproduction of a `--timeout`-driven ceiling collapse, and 300 was the
value that collapsed it. At 1800 the arithmetic assertions still check the derived
relation, but `ceiling_floored` is false and the collapse itself is out of reach outside
a hang. What was bought is a run that can complete at all; what was sold is the
reproduction. A cheap standalone assertion on a low-`--timeout` run would buy it back
without a trio invocation, and it is not written yet.

R4 dominates, and its budget is the number worth understanding before you touch it. The
product divides `--timeout` across two attempts of each of three models per mage, so a
healthy run spends a small fraction of what a fully rotating one is allowed. At
`--timeout 300` the derived budget came to 24 seconds per attempt, the 250 kB consult
abandoned after 82 seconds, and eight assertions across four scenarios went red from that
one number. At 1800 the same payload against the same backend finished in 103 seconds
with three real verdicts.

**Four rounds were needed to reach the first green**, and what the three red ones found is
worth more than the green: two guardians that could never have passed, one control that
carried a different prompt from the run it controlled, one detector reading another
scenario's fixtures, and a defect in the product itself. None of them would have turned up
by reading the code.

## Out of scope

```
[OUT_OF_SCOPE] - cross-OS linkage and the published crate (REQ-S26, REQ-S27)
```

Both are properties of artifacts built somewhere else. The harness runs one binary on one
machine, so it can say nothing about how the crate behaves once published, or how it
links on a distribution that is not this one.

Declared rather than omitted, because a green that quietly covers less than a reader
assumes is the exact failure this harness exists to avoid.

## The scenarios

| Id | What it protects |
|----|------------------|
| S1 | the headless output contract (`tests/golden/headless_output_v1.json`) |
| S2 | `magi init` scaffolding, its permissions, and workspace discovery |
| S3 | REQ-V09, the vault stores and never reveals |
| S4 | REQ-V35, a wrong passphrase destroys nothing |
| S5 | `PathGuard`, the tools run and the sandbox holds |
| S6 | the native trio: three verdicts, a consensus, not degraded |
| S7 | REQ-A04, the derived relation between budget, client timeout and ceiling |
| S8 | REQ-S09 and D-16, the large payload arrives whole |
| S9 | `src/memory/` and REQ-29: persist, embed, inject, degrade |
| S10 | `src/redact.rs` and the five leak sites of the v0.12.0 gate |
| S11 | `config/migrate.rs`, a broken config cuts before running |
| S12 | the harness leaves no trace in the tree |
| S13 | every invocation in the published documentation still exists |
| S14 | REQ-S32, the doubly declared `-w` still resolves |
| S15 | `non_blank`, a blank variable is absent and never invalid |
| S16 | rotating a third-party credential costs no local data |
| S17 | REQ-EA01 and REQ-EA06, the structured flag exists only where it should |
| S18 | REQ-EA03, the structured verdict envelope's exact shape |
| S19 | REQ-EA02, the agent's consult cap |

## Adding a scenario

A decorated generator and a registry entry. Nothing else, on purpose: if adding one meant
touching the runner or the report, nobody would add one.

```python
ASSERTIONS = ("the thing that must hold",)

@scenario("S20", assertions=ASSERTIONS, run="R1", needs_backend=True)
def the_thing_that_must_hold(run):
    """..."""
    yield Finding(assertion=ASSERTIONS[0], outcome=Outcome.PASS, detail="",
                  run_id="R1")
```

Import the module in `smoke/scenarios/__init__.py`, because the decorator registers at
import time and a scenario nobody imports is a scenario nobody runs. Then raise
`DECLARED_SCENARIO_COUNT` in `smoke/registry.py`, which is the number the certificate
publishes.

Three rules every scenario obeys:

- **Yield at least one finding.** A scenario that cannot evaluate yields `CANNOT_TEST`
  with the cause. Silence is a harness failure.
- **Reach the product only through `smoke.runs`**, never through `smoke.binary`. A test
  enforces it.
- **Declare every assertion text to the decorator, as the module's own
  constant.** The runner reconciles what arrived against what was promised, in
  both directions: a missing one is coverage lost in silence, and one nobody
  declared reaches the certificate verbatim. Passing a literal instead of the
  constant makes a second copy the check cannot see drifting, and a test
  refuses it.
- **Never assert on what the model answered.** Assert on what the product did: did it
  persist, did it inject, did it redact, what shape is the JSON, how many tokens went in.
  Record the model's answer as an observation with no verdict.

Anchor each scenario to a requirement or a documented invariant. One with no anchor is
surplus. A checkable requirement with no scenario is a declared hole.

## Known limitations

- **`[backend].base_url` can hold exactly one value, and it is not free.** The preflight
  regenerates the environment's `magi.toml` from `magi init` on every run, so what the
  runs use is always the product's default. This setting has to match it or the preflight
  cuts, and the comparison is literal after trimming a trailing slash: `127.0.0.1` and
  `localhost` are different strings here.
- **The probe and the runs must name the same endpoint, and the preflight now checks it.**
  `[backend].base_url` in `smoke.toml` is read by exactly one thing, the reachability
  probe; every run goes to whatever the environment's `magi.toml` declares at its root.
  This repository ran a whole session with the probe aimed at a machine on the LAN and the
  runs going to localhost, and nothing noticed, because both daemons served the same tags.
  Step 8 would have cut on the first discrepancy that mattered: the embedding model exists
  on one and not on the other.
- **A killed run leaves the lock behind.** `smoke/.lock` sits beside `env/` rather than
  inside it, so `--reset-env` does not clear it. Delete it by hand after a run was
  killed.
- **R7 cannot tell you it overwrote a changed `smoke.toml`.** If the file was edited
  between a crash and the restart, the restore writes the new value with no warning that
  it differs from what was there. Comparing the stored value against the file would
  settle it, and REQ-V09 means no stored value is ever printed. The report warns that R7
  overwrote. It cannot say what it overwrote.
- **The round counter lives in `env/`.** `--reset-env` discards it, and the next
  `--smoke-2` counts as the first round of the release.
- **The capability probe measures once per process.** Swap a model underneath a running
  daemon and nothing notices.
- **On Windows the permission check degrades when the ACL cannot be read as SDDL.** It
  reports `CANNOT_TEST` rather than passing, but the protection there is weaker than the
  POSIX bits check on Unix.
- **`env/` is not portable.** It carries absolute paths and a database with
  filesystem-specific state. Copying it to another machine does not work. Run
  `--init-env` there instead.
- **S14's mutation record is borrowed, not its own.** `tests/workdir_flag.rs` documents a
  mutation of the product's `-w` resolution: seven of its eight integration tests go red
  with the flag disabled, and the reasoning sits above `seed_workspace`. That is a
  different artifact guarding a different build profile, and S14 exists precisely because
  those run in debug. S14's own four assertions were mutation-verified at the level of the
  double (`test_a_resolver_that_ignores_the_flag_fails`), not against the release binary.
  Doing it against the binary is open work.
- **Two scenarios can be talked out of their own subject.** S18's key counts and S19's cap
  both read a structure the model has to produce first; if it produces none they report
  `CANNOT_TEST`, which blocks the gate. S5 hit this and was fixed by measuring a prompt
  that makes the product act instead of explain, recorded as `DEFER_TO_THE_GUARD`. The
  other two have no equivalent yet, so the gate's colour still depends on model behaviour
  in two places.

## What the first four rounds found

Kept because the list is the argument for running this at all. The unit suite was green
throughout.

- **A run that hung instead of failing.** R6 pointed at `127.0.0.1:9`, described in the
  code as "the discard service, reserved and unused". Where that service is actually
  running the connection is accepted and never answered. The run was killed with no
  output, and S10 reported `CANNOT_TEST` over three assertions it never got to search.
- **Two guardians that could not pass.** S9 matched a phrase the product emits only when
  the embedder client fails to *construct*, on a probe that makes it fail later; and S8
  read a field `run_consult` documents as always zero. Both had looked fine for as long as
  something else was failing first.
- **A control that controlled nothing.** S9 subtracted two runs carrying different
  prompts, so the difference mixed prompt length with memory injection. Fixed, the honest
  delta turned out to be 33 tokens, which no threshold can sit on.
- **A detector reading someone else's fixtures.** S10 concatenated every archived log,
  including the config examples S13 plants, and reported the product's own documented
  `[user]:[password]` placeholder as a leaked credential.
- **A product defect.** The headless runner consumed `StreamPiece::Notice` and dropped it,
  so REQ-29's "degrade *and say so*" held in the TUI and nowhere else. An operator running
  `magi query` with a failing embedder was told nothing at all.
