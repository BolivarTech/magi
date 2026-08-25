# Changelog

All notable changes to **Magi Agent** (`magi-rs`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the **minor** position signals significant or breaking
changes and the **patch** position signals backward-compatible fixes.

## [Unreleased]

### Fixed

- A present-but-unparseable `magi.toml` now exits **2**, not 1. Exit 2 is the
  published code for invalid input (README, REQ-H23) and the same function
  already used it for an unrecognised envelope `mode`, so one class of failure
  answered two codes and a CI could not tell a malformed config from a corrupt
  database.

### Security

- Update `h2` to 0.4.19, closing RUSTSEC-2026-0258: a peer could send unbounded
  empty DATA frames and force the connection to keep allocating. The crate
  reaches magi-rs transitively through `reqwest`; no magi-rs code changes.

## [0.15.0] - 2026-08-15

Each MAGI mage's per-attempt operation budget used to cap out at 72 s regardless of the run's
`--timeout`. A user reported 2 of 3 mages failing on a 24k-token payload with `--timeout` set
high, and had to read the source to find out why: the budget derived from
`[magi].agent_timeout_secs`, which `config.rs` validates into `30..=120`, so a generous
`--timeout` bought nothing. On the headless path the budget now derives from `--timeout` itself.

### Breaking

- **`TimeoutDecision` is sealed.** A private marker field blocks external construction via a
  struct literal. Replace any such construction with `TimeoutDecision::obeyed(secs)`, a one-line
  change. This seal is why the release takes the minor position rather than a patch.
- **`headless_consult_timeout_secs`'s overflow behaviour changed.** The function is `pub`, and
  folding the slack into `attempt_factor`'s hundredths moved its overflow point roughly 100x
  closer. A `saturating_mul` now guards it, so a caller passing an enormous ceiling gets
  saturation where it previously got a correct result or a panic. Nobody passes a ceiling above
  ~1.8e15 to a timeout calculator, so this is almost certainly inert in practice. Still, it is a
  behavioural change to a published function, and a red build is the wrong place to discover it.

### Added

- **The per-mage operation budget derives from an explicit `--timeout` on the headless path**,
  by inverting the formula that already computed the required wall clock from the ceiling. Both
  directions share one `attempt_factor` function, so they cannot drift apart. Concretely,
  `--timeout 1800` yields a **249 s ceiling and a 149 s per-attempt budget** (the divisor is
  2 attempts x 3 models x 1.2 slack, then x 0.6), against a flat 72 s before this release. The
  TUI path is unchanged: it still reads `agent_timeout_secs` directly.
- **A 15 s floor applies to the derived ceiling** (`AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`). It is
  not defensive padding: the derived path bypasses `config.rs`'s validation, and without a floor
  a small `--timeout` would send both internal layers to their minimums, whose sum exceeds the
  ceiling and breaks the relation between them. When the floor fires, the run starts anyway and
  emits one notice naming both levers: the timeout and the rotation count.
- **No upper bound on the derived ceiling.** A `--timeout` with one extra digit buys hours per
  attempt; a `ceiling_above_sanity` flag reports the situation and the run proceeds regardless.
  This is deliberate: capping the value to protect an operator from their own typo is exactly the
  paternalism this change removes. `--timeout` is also the only knob that scales the per-mage
  budget. A tier default under `--auto`, or `[headless] timeout_secs`, sets the run's wall clock
  without scaling the budget.
- **`applied_caps` gains five fields**: `operation_budget_secs`, `ceiling_floored`,
  `floor_activation_threshold_secs`, `max_rotations_effective`, `ceiling_above_sanity`. See the
  [README](README.md#applied_caps) for the full field table. Additive under the existing envelope
  policy, with no `schema_version` bump.

## [0.14.3] - 2026-08-13

### Added

- **`magi consult --structured-verdicts` adds `agents` and `consensus` to the JSON output.**
  A machine consumer gets the trio's verdicts typed — each seat's `verdict`, `confidence`,
  `summary`, `reasoning`, `recommendation` and per-finding `severity`/`title`/`detail`/`file`/
  `line`/`category` — instead of recovering them from rendered markdown. `consensus` carries
  magi-core's own computation so a consumer that runs its own can **contrast** the two; a
  divergence between independently written formulas is a signal nobody could see if only one
  travelled.

  Opt-in, and deliberately so. The same text already travels rendered in `report`, which is
  bounded by the tool-result cap, so emitting it unconditionally would return the bytes that cap
  exists to bound through a second key. The agent-facing `/consult` tool never emits them at all:
  there the value goes straight into the agent's context window.

  The keys are **always present when the flag is passed**, empty array included — an empty
  `agents` certifies that no seat completed. The output shape varies by flag, never by data, so a
  strict-schema consumer still gets one shape on every run. The flag lives on `consult` alone
  (`magi query --structured-verdicts` is a parse error, not an accepted no-op) and requires
  `--output-format json`; with text output it exits **2** rather than being ignored.

  Requested by `magi-claude` (ref E-A) to delete ~11 Python modules that reimplement what this
  crate already does.

## [0.14.2] - 2026-08-13

### Fixed

- **A model that is both the principal and a fallback candidate keeps its measured window.**
  The magi-rs agent and magi-core's trio/pool are two different levels, so one tag naming both
  is ordinary configuration — and with a handful of good local models it is the common case.
  The startup probe's trio table excludes the principal **on purpose** (a small principal must
  not lower the mage-derived `input_warn_tokens`), but every *per-candidate* decision was asking
  that same table, so a principal measured moments earlier read as unmeasured. One startup said
  both of these about one model in one run:

  ```
  kimi-k2.6:cloud: probe: window 262144 tokens, digest a90cd0d1590c...
  notice: fallback candidate `kimi-k2.6:cloud` has no measured window; it is credited
          with 131072 tokens, the smallest measured this run.
  ```

  Half the real window, on the number that decides whether a rotation into that candidate can
  accept a large prompt. The two views are now distinct values over one probe, and the
  per-candidate one includes the principal.

- **`strict_context_guard = true` is no longer silently downgraded when the only pool candidate
  is the principal.** Same root cause, worse consequence: the guard's fail-safe asks whether any
  *candidate* has a measured window, read "none", and switched the window check off over a pool
  that was in fact eligible. A setting the operator wrote and the system did not apply.

- **The fold is decided on the resolved `(endpoint, model)` pair, not on what the configuration
  declares.** A measurement belongs to the pair, so the principal's measurement may only be
  reused where the pair is genuinely the same one. Deciding that from `[magi].base_url`/`kind`
  being *declared* looks equivalent and is not: writing `kind = "ollama"` under a root that is
  already `ollama` resolves back to the identical endpoint, and would have restored the whole
  defect for that configuration. Endpoints are compared through the same redaction the
  capability cache keys on — a credential difference moves neither a model's window nor its
  digest.

## [0.14.1] - 2026-08-13

> **Supersedes 0.14.0, which was never published.** Its release pipeline failed before the
> publish step on a test whose kill fired before the process it was meant to kill had started
> — a defect in the test's own timing estimate, not in the code under test. Nothing reached
> crates.io under 0.14.0, so its entry is folded here rather than left pointing at a tag that
> does not exist.

### Fixed — release tooling

- **The process-group kill test stopped racing its own worker.** It sizes the kill from a
  measured cold start, and that measurement timed a bare `python` launch while the worker it
  predicts runs through the `bash` tool's shell wrapper — 84 ms against ~900 ms on the
  development box, more than 10x. On a fast machine the floor hid the gap; on the CI runner
  the cheap probe reported 1282 ms while the wrapped path exceeded the floor, so the kill
  landed before the child had written its START marker and the test failed on its own
  precondition. The probe now launches through the same wrapper, and a new test asserts the
  relation between the two rather than a threshold, so it stays honest on any machine.


### Changed — BREAKING (exit code)

- **`query` and `consult` now validate `-w`/`--workdir` like `init` and `vault` do.** A path
  that names no existing directory is rejected before dispatch, **exit 2** (operator misuse)
  with the path in the message. It used to be accepted and fall through to
  `no .magi/ state directory found` at **exit 1** — the misleading message the pre-dispatch
  check was written to remove, still reachable on half the CLI. The quieter half of that bug:
  the unvalidated path also became the file-tool sandbox root, so every later tool call failed
  for a second reason pointing nowhere near the typo.

  The exit code moving `1 -> 2` is the breaking part, and the reason this is a minor. A script
  keying on `1` for a bad `-w` on those two subcommands needs updating; every other failure
  class keeps its code.

### Fixed

- **`magi init` no longer pins the embedding endpoint.** The scaffold emitted
  `[embedding].base_url` **active**, so the embedder overrode the root endpoint instead of
  inheriting it — while the comment three lines above it in the generator said the field is
  left unset precisely *so it can inherit*. The operator-visible cost: run `magi init`, point
  the root `base_url` at a remote host, and the agent and trio move while the embedder keeps
  talking to localhost, with nothing in the file looking wrong because the value it pinned is a
  real endpoint. The key is now emitted **commented**, and two tests lock the behaviour that
  makes omitting it safe: absent inherits the root, declared still overrides.

### Changed

- **The scaffolded rotation pool ships five candidates, up from three**, matching the trio
  configuration this project runs alongside so an operator moving between the two finds the
  same depth. Two of that file's entries could not be copied verbatim — `deepseek-v4-pro` and
  `gpt-oss` are foreign to *its* trio but are two of *this* trio's seats, model and lineage
  both, so copying them would have cut each entry's coverage from three seats to one. The two
  substitutes are the remaining labels from that same file that no seat here holds.
- **`magi init`'s generated header names all three accepted `provider` values.** It mentioned
  only `anthropic`, which reads as a binary choice and hid `openai-compat` entirely — leaving
  an operator pointing at Groq or OpenRouter no way to learn the value from the file they were
  handed, while `deny_unknown_fields` makes a wrong guess fatal rather than a warning.
- **`run-tests.py` now runs doctests and forwards arguments to nextest.** The project copy had
  fallen behind the seed: `cargo nextest` does not execute doctests and `cargo doc` only
  compiles them, so every rustdoc example in this crate sat outside all seven gates. Seven pass
  today. Argument forwarding lets a red-green inner loop be scoped instead of paying eight
  minutes per iteration — the absence of which quietly pushed anyone in a hurry toward plain
  `cargo nextest run`, which is exactly what leaves the TDD-Guard reporter's `test.json` stale.


## [0.13.1] - 2026-08-12

### Added

- **`-w` / `--workdir` now works on `init` and `vault`**, not just `query` and `consult`.
  It is the base for the `.magi/` walk-up, so `magi init -w /srv/project` scaffolds over
  there and `magi vault ls -w /srv/project` reads that workspace, with the current
  directory untouched in both cases. On `vault` the flag parses on either side of the
  nested subcommand — `vault -w <dir> ls` and `vault ls -w <dir>` are the same command.

  A `-w` that is not an existing directory is now rejected before dispatch, **exit code 2**
  (operator misuse, matching this project's existing taxonomy — the same code a symlinked
  `-w` already produced), with the path in the message. That check exists because the same
  typo used to fail two different misleading ways: `init` surfaced a bare I/O error from the
  staging sibling it builds in the target's parent, and `vault` reported `no .magi/ state
  directory found`, which sends the reader looking for a workspace when the problem is the
  path. A missing directory is never created — `init` refuses to nest and refuses to
  overwrite, and quietly building a directory tree does not belong with that.

  On `vault` the flag may appear **before or after** the nested subcommand. Given once on
  each side, the **innermost wins** — the convention `git -C` and `docker` follow; given
  twice on the *same* side it is an error, as it is on `init`, which carries no global. The
  asymmetry comes from clap and is now documented and pinned rather than emergent.

  A `-w` whose path contains a symlinked component is rejected, **including when the target
  itself is the symlink**. This is the pre-existing REQ-H30 hardening applying to the new
  surface, and it is a real difference from the current directory: `getcwd` hands back the
  resolved physical path, so reaching the same directory with `cd` never tripped it.

### Unchanged

- **`query` and `consult` keep their own `-w`**, resolved where it always was, and it is
  **not** covered by the new validation: an invalid path there still surfaces as
  `no .magi/ state directory found` (exit 1). Extending the check to them is deliberately
  left for a minor release rather than done here: it would move that failure from exit 1 to
  exit 2, which a script may key on, and this release is additive.

  The two declarations are separate on purpose: the headless one is also the file-tool
  sandbox root and is resolved after dispatch, while `init`/`vault` need the root *before*
  it, to decide where to look. One `global` flag covering all four is what the code reaches
  for first, and clap rejects it — a global colliding with an arg in its propagation subtree
  trips a `debug_assert!` while the command is built, so the binary panics at startup under
  `debug_assertions` instead of failing to parse.
- **The TUI (no subcommand) still uses the current directory** and takes no `-w`.

## [0.13.0] - 2026-08-12

Lineage rotation: a mage whose model fails now **rotates to a declared fallback of a
different lineage and still emits a verdict**, instead of degrading the whole run. Around
that sit the measurement that decides where a rotation may land, a persistent cache so it
is paid once, and the telemetry that says what actually happened.

### BREAKING — a v0.12.0 `magi.toml` will not start

**The lineage is now mandatory for any seat that declares a model.** A lineage is the
independent failure domain you consider that model to belong to, and it is what decides
every rotation eligibility question. It is never inferred — not from the model name, not
from the endpoint. The same pair of models can legitimately be two lineages for one user
and one for another, so guessing would fabricate a decision that is not ours to make.

```toml
[magi]
melchior_model    = "qwen3.5:397b-cloud"
melchior_lineage  = "alibaba"
balthasar_model   = "gpt-oss:120b-cloud"
balthasar_lineage = "openai"
caspar_model      = "deepseek-v4-pro:cloud"
caspar_lineage    = "deepseek"
```

The startup error names **all** the missing keys at once, with the line to add for each. A
seat left on the built-in model inherits the built-in lineage and owes nothing.

**`enforce_diversity` defaults to `true`**, so the three seats must declare *different*
lineages. Three mages on the same weights add nothing an ensemble can use, and a pool with
no diversity is a safety net the operator believes they have and does not. If you only
have models from one vendor, the better exit is finer labels — by model family, e.g.
`opus`/`sonnet`/`haiku`, which gives you real rotation because they are different weights
with different failure modes. If there is genuinely nothing to diversify with:

```toml
[magi]
enforce_diversity = false
```

**The three v0.11.0 patterns are retired** (`provider = "openai"`, `[openai].base_url`,
`[headless].tool_result_cap_bytes`). A file from that version now receives the v0.13.0
guidance for its missing lineages **plus** generic serde errors for its own stale keys.
That is two versions of distance and it is accepted — but stated here rather than
discovered.

### BREAKING — two observable outputs changed

`kind = "ollama"` now completes through magi-core's `OllamaProvider` rather than the
OpenAI-compatible transport directly. It is not a protocol change — that type delegates to
the same transport internally — but two things you can see are different:

- **`name()` reports `"ollama"`** where it used to report `"openai-compat"`, in errors and
  reports.
- **A `base_url` without `/v1` no longer 404s** under this kind. Both spellings are
  accepted and reach the same endpoint.

### The report's JSON changed shape AND meaning

`schema_version` stays at `1` — it does not move before 1.0, by decision — so this section
is the only channel for both.

**Added, always present:** `rotations` (which mage hopped, from which lineage to which,
why, and the model it ended on) and `ran_unmeasured` (mages that ran without a measured
window, which qualifies confidence in their verdict). Empty is a *positive certificate*
that nobody rotated, not silence — a field that appears and disappears cannot be declared
by a consumer with a strict schema.

**Changed meaning, and this one needs reading:** a reported context window may now be
**assumed** rather than measured. When a candidate could not be measured, it is credited
with the smallest window measured in that run, and the output marks it as assumed. If you
consumed a window on the basis of *"if there is a number, it was measured"*, that
assumption no longer holds.

**A limitation worth knowing before you file it as a bug:** a mage that rotated and whose
task then panicked **or was cancelled** reports the **configured** model, not the one it
ended on, with an empty hop chain. The chain lived on the stack of the task that went
down, either way, and cannot be recovered from outside magi-core. That mage also appears
in `failed_agents`, which is the signal that its rotation entry is not to be trusted.

### Added

- **Rotation.** A shared `[[magi.fallback]]` pool, ordered strongest to weakest, with
  `max_rotations` (default `2`; `0` is a kill-switch restoring v0.12.0 exactly). Shared
  and not per-seat: what consensus needs is that a rotation lands on a lineage the other
  two seats do not hold, which is a property of the pool's diversity rather than of who
  owns the list.
- **Lazy measurement with a persistent cache**, in a new table of the encrypted database.
  Only successful measurements are stored — a cold daemon's silence is transient and must
  never be frozen into a permanent answer — and the key is the `(endpoint, model)` pair,
  because a tag is only unique within an endpoint. Reordering the pool measures nothing:
  the order is rotation preference, not identity.
- **`strict_context_guard`** (default `false`), passed to magi-core **only** when at least
  one candidate has a measured window. A `true` with nothing measured would reject every
  candidate and switch rotation off entirely, which is not a guard.
- **Startup notices** naming a configured model this endpoint could not measure while it
  was measuring others, together with the command that fixes it.

### Changed

- **The headless `--timeout` default now follows `max_rotations` and `retry_disabled`.**
  Both were already exposed, so the worst case is calculated from what you declared rather
  than from a new knob. With the defaults it is ~11 minutes, paid only when something is
  genuinely hung; `--timeout` still overrides it, with a warning naming the computed
  minimum.
- **`input_warn_tokens` stays on the trio's window**, and a pool candidate only lowers it
  when it sits within 10 % of that base. Deriving from the whole pool would let the
  candidate least likely to ever run pull the threshold down on every run, firing the size
  warning on practically every real consult — which does not make it conservative, it
  destroys the signal. Candidates outside the band are reported instead, all in one
  message.


## [0.12.2] - 2026-08-10

Dependency-only patch: `magi-core` moves from 3.1.0 to **3.2.0**. Nothing in this
release uses what that version adds — it is the base the next minor is built on, put in
place on its own so the version bump and the work that depends on it stay separable.

> **0.12.1 does not exist and never will.** Its tag was pushed by hand before the release
> workflow ran. The gate resolves a release by comparing `Cargo.toml`'s version against the
> highest tag already present, so it found `v0.12.1`, concluded the version was already
> released, and skipped the build, the crates.io publish, the tag and the GitHub Release —
> reporting **success**, because declining to release is not a failure. The tag is protected
> by a ruleset and cannot be deleted, so the version number was retired rather than reused.
> `v0.12.1` therefore joins `v0.10.2` as a tag that never reached crates.io.
>
> The release is driven by **pushing a version bump to `main`**, never by pushing a tag; the
> workflow creates the tag last, on purpose, so that a tag always means *this version built,
> published and released*.

### Changed

- **`magi-core` 3.1.0 → 3.2.0**, pinned exactly (`=`) as before. The upstream release is
  purely additive: two new constructors, one new trait method with a defaulting body, and
  a warning on a previously silent condition. No behaviour of this crate changes.

  3.2.0 adds what magi-rs needs for lineage rotation, and adds it in two forms:
  `FallbackPoolBuilder::push_with_probe` / `MagiBuilder::with_agent_and_probe` let a
  rotation probe be declared apart from the provider that serves completions, and
  `OllamaProvider::with_timeout` closes an asymmetry that made that type unusable for
  completions by any consumer sizing its timeout layers against a single ceiling.
  **Neither is used yet.** Wiring them is the next minor's work.

- **The exact pin's stated reason moved to 3.3.0.** It guards the retry defaults, which
  are the numbers the per-mage timeout scale derives from — a `cargo update` that moved
  them would move the floor of a relation the design makes impossible to break from
  configuration. 3.1.0's rustdoc announced that change for 3.2.0; 3.2.0 shipped without
  it. That was **verified before bumping rather than taken from the changelog**: the
  backoff constants and `impl Default for RetryConfig` are byte-identical across the two
  versions. The pin stays, now pointing at the next release.

### Verified

- The full suite passes unchanged at **1088 tests**, the same count as 0.12.0, so the
  upgrade introduces no regression. All seven gates green.

## [0.12.0] - 2026-08-09

Adoption release: magi-rs now uses what `magi-core` 3.1.0 already offered. The MAGI
trio is built on the crate's native providers and the in-tree adapter is gone, consult
modes actually route, the model gets measured when the endpoint allows it, and the
crate's telemetry reaches the output. The dependency itself is unchanged.

Configuration takes a **breaking reshape** in one cut: `base_url` moves to the root as
a system-wide default, `provider = "openai"` splits into two values, and a `magi.toml`
that exists but does not parse now stops the process instead of falling back. Startup
prints a guided migration error naming every incompatibility in the file at once.

### Breaking

- **`base_url` moves to the root of `magi.toml`** and becomes the system-wide default,
  used by the main agent, the MAGI trio and the embedder. **`[openai].base_url` no
  longer exists.** Each consumer may override it with its own key; precedence per
  consumer is its own env var, then its own section, then the root, then the built-in
  default. The previous shape had no way to give the trio or the embedder an endpoint
  of their own, which is exactly what this release needed.
- **`provider = "openai"` splits into `ollama` and `openai-compat`.** They share the
  completions protocol and the same `[openai]` section for the model; they differ in
  that only `ollama` is measurable, because it exposes endpoints for context window and
  weights digest. This is deliberately **not** auto-migrated: `"openai"` was ambiguous,
  that ambiguity is what is being split, and choosing for the user would be guessing.
- **`[headless].tool_result_cap_bytes` moves to the root** as `tool_result_cap_bytes`,
  because it now bounds output on all three routes — the TUI, `magi query` and
  `magi consult` — not just the headless one.
- **A `magi.toml` that exists and does not parse is now fatal.** It used to warn and
  continue on built-in defaults. Writing a config file is declaring an intent; when that
  intent cannot be honoured, carrying on with something else is worse than stopping. An
  **absent** file is still a silent default, and an **empty** one parses as valid and
  yields defaults.
- **Startup emits a guided migration error** that names every incompatibility in the
  file at once, with the corrected lines ready to paste, reporting only what that file
  actually has wrong. When a `base_url` carries embedded credentials the printed line is
  **redacted** and the message says so — redaction wins over paste-ready. Migrating
  straight from 0.10.x is **not supported**: go through 0.11.0 first.
- **`--init-config` and `/init-config` are retired.** `magi init` is the only scaffolder
  and it writes into `.magi/`. The retired flag reports where to go instead of failing
  with an unknown-argument error.
- **The `--output-format json` object of `magi consult` grows fields** — `mode`,
  `mode_source`, `extraction_failures`, `input_size`, `failed_agents`,
  `report_truncated` — while **`schema_version` deliberately stays at 1**. This is a
  contract change even without a schema bump: the project is pre-1.0, so the crate
  version is what signals it instead. **A consumer of this JSON must tolerate new
  fields and pin the crate version** — do not treat an unchanged `schema_version` as a
  backward-compatibility guarantee while magi-rs is `0.x`.
- **`[embedding].base_url` becomes optional and inherits the root `base_url`** when
  absent, instead of always defaulting to `localhost:11434`. Anyone who pointed the
  main agent at a remote endpoint and expected the embedder to stay local now gets
  the remote endpoint for embeddings too, unless `[embedding].base_url` is declared
  explicitly.

### Behaviour Change

Neither of these breaks anything or requires editing `magi.toml` — the agent simply
behaves differently than it did on v0.11.0, and the difference is easy to misread as
a routing regression if it isn't named.

- **The complexity gate ships active by default.** An autonomous consult the agent
  used to run unconditionally can now be vetoed below its mode's threshold before any
  model call happens (see the `### Added` entry below). Explicit `/consult` and
  `magi consult` are never vetoed.
- **`analysis` does not carry a pass-through threshold of `1`.** It has a real veto
  threshold like `code-review` and `design`, and it matters more than the other two
  because `analysis` is the default mode for every invocation that doesn't route
  itself — the path most existing users hit without changing anything.

### Deliberate Behaviours

Three more things worth naming up front, because each one reads like a bug on
first encounter and is not — see the README's "Three behaviours that are
deliberate, not bugs" under [Mode routing](README.md#mode-routing) for the
full explanation of each.

- **A second veto in the same turn is terminal, even for an unrelated
  question.** The complexity gate counts *vetoes*, not content: a second
  autonomous consult attempt in the same turn after a first veto disables
  `consult` for the rest of that turn, whether or not the question is the
  same one. It re-enables on the next turn, and a consult that actually ran
  in between resets the counter.
- **The probe's model measurement can go stale in the dangerous direction.**
  `input_warn_tokens` derives from a context-window measurement taken once at
  startup. Switching the Ollama daemon to a **smaller**-window model while
  magi-rs keeps running does not re-measure until restart, so the stale,
  larger threshold silently stops firing the oversized-input warning right
  when it would matter most. Restart magi-rs after changing the daemon's
  model.
- **Mode inference queries the main agent first when the trio's endpoint
  diverges.** When `[magi].kind`/`[magi].base_url` points the trio at an
  endpoint different from the main agent's — for example, a deliberately more
  restricted network — a consult without a declared mode still sends the
  content to the **main** agent's endpoint for classification before it ever
  reaches the trio. A one-time startup notice flags the divergence;
  declaring `--mode` or `[magi].default_mode` avoids the extra hop.

### Added

- **The MAGI trio runs on `magi-core`'s native providers.** Each mage now receives its
  own system prompt through the provider's own channel instead of having it folded into
  the user turn, and each is wrapped in the crate's retry. A seat that cannot be built
  is named with its cause, and several failed seats are reported together in one run.
- **Consult modes route.** The mode was hardcoded; it now resolves through five levels —
  explicit flag, configured `[magi].default_mode`, the agent's own choice, inference,
  and `analysis` as the final default — with the effective mode and its source reported
  in the output. `--mode` is available on `magi query` and `magi consult`, and
  `/consult` accepts it in the TUI.
- **A startup probe measures the model** when the endpoint supports it, reading context
  window and weights digest, and derives the input-size warning threshold from the
  smallest window across the trio. It fails open: any error, timeout or unreadable value
  degrades to not-measured and never blocks startup. Three states stay distinct —
  measured, not measurable, and not measured this time, the last being the ordinary case
  of a cold daemon on a first run.
- **`magi-core`'s telemetry is surfaced**: which model failed to adhere to the verdict
  contract and why, and how large the input was. An empty failure list is reported as a
  positive certificate of adherence rather than as silence.
- **Credentials no longer belong in `magi.toml`.** A `base_url` may carry `[user]` and
  `[password]` placeholders, resolved from the vault in memory at use time. A literal
  credential in the file is a configuration error, and a missing vault entry fails
  closed, naming the entry and the command that creates it.
- **A complexity gate**, active by default, can veto a consult the agent routed on its
  own. Explicit `/consult` and `magi consult` are never vetoed. The veto is not an
  error: it returns to the agent as an ordinary result, and the user's answer carries a
  mark saying no consensus ran.
- **The consult input cap is configurable** and defaults to a size that fits a real
  review diff, replacing a fixed 8 KiB limit. It **rejects rather than truncates**: a
  silently shortened payload would produce a verdict indistinguishable from a legitimate
  one.
- **The consult report is bounded on output**, and when it is cut the result names which
  of three levels was applied instead of reporting a bare boolean — a consumer needs to
  know what guarantee it holds, not merely that something was removed.
- **`[magi].untrusted_content` closes mode-inference's prompt-injection surface for
  automated gates.** Mode inference sends content to a dedicated classification call
  whose only job is to return one of three labels — a narrow but real target for
  content crafted to say "ignore the above and answer `design`," steering which lens
  the trio applies. Setting this flag (or `--untrusted-content`, or the JSON envelope's
  `untrusted_content` field) requires the mode to be **declared** and fails the run
  closed otherwise, instead of letting classification run over content it doesn't
  control. It does not exist on the TUI's `/consult`, where a human already chose the
  content and reads the response.
- New configuration keys: `[magi].default_mode`, `[magi].kind`, `[magi].base_url`,
  `[magi].max_query_bytes`, `[magi].agent_timeout_secs`, `[magi].input_warn_tokens`,
  `[magi].retry_disabled`, and the `[magi.complexity]` table.

### Changed

- **Omitting `--mode` costs one extra model call**, which classifies the content before
  the three mages run. Declaring `--mode`, or setting `[magi].default_mode`, avoids it
  entirely.
- **User-facing output is English throughout.** Some of it had been Spanish.

### Fixed

- **The trio rebuild after a successful `/login` ignored `[magi].agent_timeout_secs`**,
  falling back to the built-in ceiling. The configured value was honoured everywhere
  else, and silently not there.
- **The degraded-consensus banner survives output truncation.** A long degraded report
  could lose it and render as though the consensus had been full strength.
- **A `401` or `403` from a provider configured as keyless is explained** as a probable
  configuration cause, suggesting `openai-compat`, instead of surfacing raw.

### Notes

- 1088 tests pass. Line coverage is 90.13 %.
- `magi-core` stays at 3.1.0. Lineage rotation, the fallback pool and the context guard
  are the next milestone.

## [0.11.0] - 2026-07-31

Dependency migration: `magi-core` **1.1.1 → 3.1.0**, a jump across three majors.
Scope is deliberately narrow — make magi-rs compile and behave correctly on the new
crate. Adopting what 3.x adds (lineage rotation, provider probe, extraction-failure
telemetry, the richer report) is the next milestone.

### Changed
- **`magi-core` upgraded to 3.1.0.** The two API breaks that reached magi-rs:
  - `ProviderError`'s struct-like variants are `#[non_exhaustive]`, so an external
    `LlmProvider` can no longer build one directly. 3.1.0 restores external
    construction through **`ProviderError::external(message, ExternalErrorKind)`**,
    which produces the dedicated `External { message, kind }` variant rather than
    impersonating an internal one. The MAGI adapter and the consult test doubles now
    use it.
  - **Mage responses must carry the verdict between `<MAGI_VERDICT>` markers.**
    magi-core 3.0.0 replaced the parser that *searched* a response for a
    verdict-shaped JSON object with one that reads only what sits between two
    markers. Searching meant choosing, and choosing could pick wrong — worst case a
    fabricated verdict in the adversarial seat. Every test double that returned a
    bare JSON verdict was updated to wrap it, using the crate's `VERDICT_OPEN` /
    `VERDICT_CLOSE` constants so a future marker change breaks the build instead of
    silently degrading the fixture.
- **`CDLA-Permissive-2.0` added to the license allowlist** for `webpki-root-certs`,
  the Mozilla root bundle pulled in by `rustls-platform-verifier`. It is a data
  licence, permissive, with no copyleft. This gate had been failing since the
  reqwest 0.13 upgrade in 0.10.2; the failure predates this release.

### Notes
- No runtime behaviour of magi-rs itself changed: the TUI, the headless subcommands,
  the vault and the tool sandbox are untouched. 673 tests pass unchanged.
- `reqwest` needed no work: magi-rs and magi-core 3.1.0 are both on 0.13.

## [0.10.4] - 2026-07-30

Release-engineering only — **no changes** to the `magi-rs` runtime, CLI or library
behavior. This release makes the Linux binary run on enterprise distributions and
replaces hand-pushed release tags with a version-driven pipeline.

### Added
- **`linux-x86_64-compat` archive for older Linux distributions.** The existing
  `linux-x86_64` build is linked against the build host's glibc 2.35, which every
  RHEL-family distribution rejects — RHEL/Rocky/Alma/CloudLinux 8 ship glibc 2.28
  and RHEL 9 ships 2.34, so the binary failed to start with
  `version 'GLIBC_2.34' not found`. The new archive is linked against a **glibc
  2.17** baseline via `cargo-zigbuild`; since glibc is forward-compatible it runs
  everywhere the native build does, plus RHEL 7/8/9, Debian 10+ and older Ubuntu.
  Both are published — see the README for which to download. A build step now
  verifies the baseline with `objdump` and fails the release if a binary ever
  exceeds it.
- **`## Prebuilt binaries` section in the README.** Releases have shipped
  ready-to-run archives since v0.10.0 without the README ever mentioning them.
  Documents each archive, how to tell which Linux build you need
  (`ldd --version`), what the wrong choice looks like, and how to verify a
  download against `SHA256SUMS.txt`.
- **Rollback protection on releases.** A version lower than the highest released
  tag aborts the pipeline instead of producing a backwards release.
- **`CHANGELOG.md` entry required** for every new version — release notes are
  extracted from it, so a missing section previously shipped an empty release.
- **`Cargo.lock` consistency guard**, which catches in seconds the drift that
  broke the v0.10.3 release.

### Changed
- **Releases are now driven by the version in `Cargo.toml`, not by a hand-pushed
  tag.** Pushing a version bump to `main` validates it, builds every platform,
  publishes to crates.io and only **then** creates the `vX.Y.Z` tag and the GitHub
  Release. Previously the tag was pushed first, so a failed build left it pointing
  at a broken commit (v0.10.3 had to be deleted and re-pushed; v0.10.2 is tagged
  but never reached crates.io). A tag now means the version built on every target,
  published, and released successfully.
- The checksum asset is named **`SHA256SUMS.txt`** rather than the extensionless
  GNU convention — Magi is Windows-first, and Explorer shows an extensionless file
  as an unopenable "File". `sha256sum -c` is unaffected.
- Version and CHANGELOG are validated on **pull requests**, so a bad bump fails at
  review time instead of after the merge.
- CI actions updated to `@v5` for the Node 24 runtime.

### Fixed
- **Releases failed outright because the committed `Cargo.lock` was stale.** Two
  commits bumped `Cargo.toml` alone, so every `--locked` platform build died with
  `cannot update the lock file` and `cargo publish` aborted on a dirty working
  tree. `--locked` now also runs in CI, so this drift fails on the push that
  introduces it rather than days later during a release.
- Dry-run builds from a branch whose name contains a slash (`ci/…`, `feat/…`)
  failed while packaging, because the branch name was interpolated raw into a
  directory name.

## [0.10.3] - 2026-07-30

Hotfix for a broken release pipeline. Contains everything in 0.10.2, which is the
release that actually reached users.

### Fixed
- **Releases could not be built or published: the committed `Cargo.lock` was
  stale.** The 0.10.2 version bump changed `Cargo.toml` without regenerating the
  lock, so every `cargo build --locked` failed with `cannot update the lock file`
  and `cargo publish` aborted on a dirty working tree.

### Added
- CI guard that re-resolves `Cargo.lock` against `Cargo.toml` and fails in seconds
  rather than five platform builds later.

### Changed
- README version and test badges are now generated from crates.io and the CI
  status instead of being hand-written constants that silently went stale.

## [0.10.2] - 2026-07-30

**Tagged but never published to crates.io** — the release pipeline failed before
the publish step (see 0.10.3). The git tag and its GitHub release exist, which is
why crates.io skips from 0.10.1 to 0.10.3. Everything below shipped in 0.10.3.

### Changed
- **TLS moved from OpenSSL to rustls.** `reqwest` was upgraded 0.11 → 0.13 with
  `default-features = false`, replacing the transitive `native-tls`/`openssl-sys`
  dependency. Prior releases dynamically linked `libssl.so.3` and refused to start
  on any system without OpenSSL 3 (`error while loading shared libraries:
  libssl.so.3`), which includes RHEL/CentOS/CloudLinux 8. No OpenSSL is linked any
  more — the Linux binary now needs only core glibc.
- **Release profile hardened for size and speed:** `lto = true`,
  `codegen-units = 1` and `strip = true`. Note this also sets
  **`panic = "abort"`**, so a panic terminates the process immediately instead of
  unwinding — there is no `catch_unwind` recovery and no unwinding across FFI.

### Added
- Release guard asserting the git tag matches the `Cargo.toml` version, and a
  `SHA256SUMS` asset published alongside the platform archives.

## [0.10.1] - 2026-07-20

Release-tooling only — **no changes** to the `magi-rs` runtime, CLI, or library
behavior. This patch adds prebuilt, ready-to-run binaries to every GitHub release
so users can download and run without a Rust toolchain.

### Added
- **Prebuilt release binaries for all supported platforms.** Every `vX.Y.Z` tag
  now attaches ready-to-run archives to the GitHub release, each built **natively**
  on its platform (avoiding the `openssl-sys`/bundled-SQLite cross-compile issues):
  - **Windows x86_64** — `.7z` and `.zip`
  - **Linux x86_64** — `.7z` (built on glibc 2.35 for wide runtime compatibility)
  - **Raspberry Pi 5 (aarch64)** — `.7z` (glibc 2.35 ≤ Raspberry Pi OS Bookworm's 2.36)
  - **macOS universal2** — `.7z` (a single Intel + Apple Silicon binary via `lipo`)

  Each archive contains the binary plus `README.md` and both licenses — download,
  extract, run. A `workflow_dispatch` trigger lets all targets be dry-run before
  tagging a release.

## [0.10.0] - 2026-07-20

**Headless mode — Magi is now invocable non-interactively** as a CI/CD pipeline
step and as an AI backend for another application, alongside the unchanged TUI.
Three subcommands (`magi init` / `magi query` / `magi consult`) with structured
I/O, a 3-tier tool-authorization policy, and all state unified under `.magi/`.
Delivered across MS1 (unified `.magi/` state + the never-delete ABSOLUTE bootstrap
state machine) and MS2 (subcommands, tiers, rich I/O, `vault diagnose`).

### Added
- **`magi query` / `magi consult` subcommands.** `query` runs the normal agent
  loop (LLM ↔ tools) over a prompt; `consult` forces a MAGI 3-perspective
  analysis of the prompt directly. Input from `-i <file>` or **stdin**; output to
  `-o <file>` or **stdout**; diagnostics/logs always to **stderr**. Input is
  auto-detected as a rich JSON **envelope** (`{prompt, system?, model?, provider?,
  max_tool_calls?, consult?}`) when it is a JSON object with a top-level `prompt`,
  otherwise it is treated as a plain-text prompt. Running `magi` with no subcommand
  still launches the TUI, unchanged.
- **3-tier tool authorization.** `default` (no flag) auto-approves only read-only
  tools (`ls`/`view`/`grep`) and denies the rest; `--auto` auto-approves every
  registered tool (`edit`/`bash`/`consult`), sandboxed; `--full-auto` additionally
  raises `max_tool_calls` (15 → 50) and silences the two **soft** guards
  (3-identical-call detection + the normal cap), printing a **WARNING** to stderr
  and the log at startup. The **hard** barriers (`bash` allowlist, the
  metacharacter ban, `PathGuard::validate`) are **always** active in every tier —
  no flag relaxes them.
- **Rich JSON / text output.** `--output-format json` emits one buffered object
  with `schema_version`, `response`, `model`, `provider`, `usage`, `timings`,
  `stop_reason`, `tool_calls[]`, `transcript[]`, `consult`, `applied_caps` and
  `error`; `--output-format text` (default) streams the response to stdout. Tool
  results are truncated to 64 KiB with a marker.
- **Exit-code taxonomy:** `0` success, `1` runtime/agent error (incl. `DbCorrupt`,
  wrong passphrase), `2` CLI misuse / invalid input (missing `prompt`, unknown
  field, over-cap input), `3` blocked by tier (an essential tool denied and the
  run produced no response).
- **`magi vault diagnose`** — a **read-only** structural probe of the `.magi/` DB
  (envelope present? FEC-decodable? row counts of `vault`/`sessions`/`messages`/
  `knowledge`/`memories`? the §2.1 state-machine verdict) that **never** unlocks,
  mutates, or prints a secret value. `--names` may list vault *names* (already
  non-secret via `vault ls`), never values.
- **`--timeout <secs>`** bounds the whole run's wall-clock: on expiry the run
  aborts cleanly (partial output, `stop_reason = error` / `error.kind = timeout`,
  exit 1), cancelling the in-flight LLM stream and killing any tool subprocess
  tree (POSIX process-group / Windows Job Object). Tiers that execute tools
  (`--auto`+) get a hard 900 s default when no timeout is set.
- **Secret hygiene for headless (REQ-H37).** At startup the process reads and
  **scrubs** `MAGI_PASSPHRASE`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from its
  environment (single-threaded, before the tokio runtime starts), and spawns tool
  subprocesses with an allowlisted environment — so an in-workspace interpreter
  cannot exfiltrate them via `/proc/<pid>/environ`. `.magi/` is created with
  restrictive permissions (`0700` dir / `0600` files; Windows ACL equivalent).

### Changed
- **BREAKING — all state now lives in `.magi/`, discovered by walk-up.** The
  config (`magi.toml`), the encrypted DB (`.magi-rs-memory.db`) and run logs
  (`logs/`) live under a single `.magi/` directory, discovered from the working
  directory (`-w`/cwd) up through the nearest ancestor (`.git`-style), stopping at
  a filesystem boundary and rejecting a symlinked `.magi`. **Loose legacy files in
  the current directory (a bare `.magi-rs-memory.db` / `magi.toml`) are NO LONGER
  READ.** A startup warning flags such leftovers and points at `magi init`. This
  is a deliberate pre-1.0 break with no on-disk migration (see the migration
  one-liner below).
- **`magi init`** scaffolds a fresh `.magi/` (config, `logs/`, and the DB with all
  tables created empty), building it in a temporary sibling and atomically renaming
  it into place; it refuses to overwrite or nest inside an existing `.magi/`. With
  `-p`/`MAGI_PASSPHRASE` it also bootstraps an empty vault (envelope + `vault_meta`
  row) so headless runs need no interaction.
- **Never-delete ABSOLUTE (REQ-H20 / D-H10 / SC-H21).** The DB open/bootstrap
  logic (`src/system/database.rs`) is rewritten to the §2.1 state machine and
  **the last automatic `DELETE` is gone**. Previously a DB whose records had no
  envelope (`wrapped_dek` absent) was treated as a pre-envelope format and
  **wiped** to a fresh start. That path is removed: a DB with data but no
  envelope is now **corruption** (`VaultError::DbCorrupt`, exit 1) and is
  **never** deleted or bootstrapped over — restore a backup or remove `.magi/`
  by hand. The evaluation order is: envelope present ⇒ open (FEC before AEAD;
  FEC-uncorrectable ⇒ `VaultMetaCorrupt`, wrong passphrase ⇒ `WrongPassphrase`,
  never wiped); no envelope + all data tables empty ⇒ bootstrap; no envelope +
  any data (or a missing data table = partial/foreign schema) ⇒ `DbCorrupt`.
- **`EncryptedSqliteMemory::open_with_state_machine`** opens an
  already-initialized `.magi/` DB **without creating any schema**, so a missing
  table surfaces as `DbCorrupt` instead of being silently re-created. The raw
  `EncryptedSqliteMemory::new` path keeps auto-initializing the schema (TUI /
  `vault` CLI) but shares the same never-delete state machine.

### Removed
- **`EncryptedSqliteMemory::was_reset()`** and the "content has been reset"
  startup notice, along with the v0.9.0 fresh-start auto-reset it reported. With
  never-delete absolute there is no reset to report.

### Migration (from v0.9.0)
- Run **`magi init`** to create `.magi/`, then re-add your secrets with **`magi
  vault set <NAME>`** (e.g. `ANTHROPIC_API_KEY`). The loose legacy DB in the
  current directory is not read or migrated (you are warned about it at startup);
  prior conversation history is not carried over (it was local and pre-1.0). This
  is a one-liner, not a migration tool.

## [0.9.0] - 2026-07-17

Vault **MS2 — the vault CLI + zero-knowledge passphrase**. The user-facing surface
promised by MS1's cryptographic foundation: a `magi-rs vault` CLI, a passphrase
in place of the OS keyring, and the DEK protected in RAM rather than held as
plain bytes.

### Added
- **`magi-rs vault {ls,set,rm,passwd}` CLI.** `ls` lists secret names with
  `created_at`/`updated_at` only; `set <name>` reads the value from a hidden
  prompt (TTY) or raw stdin (no TTY), never as a CLI argument, and asks for a
  `Y`-only confirmation before overwriting an existing name (`--force`/`-f`
  skips it); `rm <name>` needs the same confirmation; `passwd` re-keys the
  envelope. There is deliberately no `get`/`cat`/`show <name>` — a stored
  secret's value can never be printed.
- **`-p`/`--passphrase` global flag and `MAGI_PASSPHRASE` env var**, resolved
  with a fixed precedence (`-p` > env > interactive hidden prompt) ahead of
  every TUI launch and every `vault` subcommand. First run asks the user to
  create a passphrase and enforces a hard strength floor (`zxcvbn` score ≥ 3,
  ≥ 12 characters, no composition rules, no override).
- **`MaskedDek` RAM protection** (`src/vault/memguard.rs`). The data key is
  held masked (XOR, `OsRng`-derived mask rotated on every access) rather than
  as plain bytes, backed by best-effort `mlock`/`VirtualLock` (no swap),
  `RLIMIT_CORE=0` (Unix) and `PR_SET_DUMPABLE=0` (Linux) to suppress core
  dumps and same-user `ptrace`. All three layers fail open with a visible
  warning rather than refusing to start.
- **`env > vault` API-key resolution** for both `ANTHROPIC_API_KEY` and
  `OPENAI_API_KEY` — the vault is now the single shared secret store for
  both, closing the previous "OpenAI key is env-only" gap.
- **`passwd`** re-derives the KEK from a new passphrase and re-wraps the
  existing DEK in one crash-safe transaction (openable with the old or new
  passphrase at every point, never bricked) — an O(1) re-key, no record
  re-encryption.
- 2 new fuzz targets (`fuzz_secret_value_roundtrip`, `fuzz_passphrase_input`)
  covering the vault-value and passphrase-input surfaces added in MS2, plus
  `cargo +nightly miri test` coverage of the new core (`memguard`/`store`/
  `master`).

### Changed / BREAKING
- **The OS keyring and `key.txt` are removed entirely.** `system::secrets`
  (`KeyringStore`) is deleted, the `keyring` dependency is gone from
  `Cargo.toml`, and `key.txt`/`key.txt.bak` are no longer read anywhere. The
  DB now opens **only** with the user passphrase.
- **A DB created before v0.9.0** was wrapped with the keyring-era master
  secret; opening it with any passphrase deterministically yields
  `incorrect passphrase`. There is no migration path (pre-1.0, zero installs)
  — delete `.magi-rs-memory.db` manually to start fresh.
- **Every TUI launch now prompts for the passphrase**, and `--logout`/`/logout`
  require unlocking the vault first (they now do `vault rm ANTHROPIC_API_KEY`
  instead of clearing a keyring entry). This is a deliberate UX trade-off —
  daily friction traded for a hard zero-knowledge guarantee (no more silent
  DB unlock from a keyring-cached secret) — and is called out explicitly
  rather than left to be discovered.
- **`/login` (OAuth) writes the minted key to the vault**, not a keyring.

### Security
- **Zero-knowledge, no backdoor.** Nothing persisted anywhere opens the DB
  without the passphrase — no keyring entry, no cached file, no copy of the
  DEK on disk. Forgetting the passphrase means the data is unrecoverable;
  this is stated to the user on first-run passphrase creation and again in
  `vault passwd`.
- **Passphrase strength floor is a hard rejection, not advisory.** `zxcvbn`
  score < 3, under 12 characters, or a blocklist match are all rejected
  outright with no override flag — closing an offline brute-force exposure
  that a portable, keyring-free `.db` file otherwise leaves open.
- **The passphrase is never read from a pipe.** Without a TTY, stdin is
  reserved exclusively for a vault-command *value*; the passphrase must come
  from `-p`/`MAGI_PASSPHRASE` or the command fails closed
  (`VaultError::PassphraseUnavailable`) rather than misreading a piped value
  as the passphrase.
- **The DEK is masked in RAM** (see "Added" above) rather than cached as a
  plain `Vec<u8>`; the mask rotates on every access, even if the operation
  panics.

### Known limitations
- **`passwd` does not re-encrypt records.** Rotating the passphrase re-wraps
  the DEK in O(1) but leaves every record encrypted under the *same* DEK.
  This protects against a compromised passphrase, not against a DEK already
  extracted from a running process's RAM. Documented, not surfaced as a
  runtime warning (would be noise for routine rotation) — see `CLAUDE.md`
  and the `vault passwd` docs.
- **`mlock`/core-dump suppression are best-effort and platform-uneven.**
  Windows cannot prevent a user-initiated process dump from inside the
  program in any language; containers/CI without `CAP_IPC_LOCK` degrade with
  a visible warning instead of refusing to start. None of the three RAM
  layers defeat an attacker with debugger access to the process — the module
  rustdoc says so explicitly.
- **Unlock ergonomics** (avoiding a passphrase prompt on every launch) and
  **hardware-backed key protection** (TPM/Secure Enclave) remain deferred —
  see `dev-docs/PENDING_IMPLEMENTATION.md` §13.3/§13.4.

## [0.8.0] - 2026-07-15

Vault **MS1 — cryptographic foundation**. A pure hardening milestone: no new
user-facing features (the `magi vault` CLI and user passphrase land in MS2). The
crypto is now auditable, the key model is an envelope, and a bad key can no
longer destroy data.

### Changed
- **All at-rest encryption migrated to the audited [`cryptovault`](https://crates.io/crates/cryptovault) 0.3.0 crate.**
  Conversation history, project knowledge, and memory vectors now encrypt through
  the external crate (`#![forbid(unsafe_code)]`, known-answer tests, fuzz targets)
  instead of the in-tree implementation. The primitive pipeline
  (Argon2id → AES-256-GCM-SIV → error-correcting FEC) is unchanged.
- **Envelope key management (DEK/KEK).** A random 32-byte data key (DEK) encrypts
  every record; the DB master secret derives a key-encryption key (KEK, Argon2id)
  that **wraps** the DEK. `vault_meta` stores `{salt, wrapped_dek}`, each
  FEC-encoded so on-disk bit-rot is corrected before the unwrap. The DEK is
  unwrapped once per session and cached — no Argon2 per record — and a future
  master-secret rotation becomes an O(1) re-wrap with no data re-encryption.
- Expensive crypto no longer runs while a database lock is held: knowledge and
  message reads decrypt off-lock, and the bootstrap derives the KEK before taking
  the write lock.

### Added
- `src/vault/` crypto-foundation module: `envelope.rs` (bootstrap/open, wrap/unwrap,
  keyless FEC over `vault_meta`), `error.rs` (`VaultError`), `mod.rs` (public boundary).
- `src/lib.rs` library root so fuzz and coverage targets can link the crate.
- Fuzz targets (`fuzz/`) and a FEC performance-baseline benchmark (`bench_vault_crypto`).
- Mechanical standards enforcement: module lint attributes, `rustfmt.toml`,
  `deny.toml`, and a hardcoded-secret scan test.

### Removed
- In-tree `src/utils/crypto.rs` (1340 lines) and the direct `aes-gcm-siv`,
  `argon2`, and `reed-solomon` dependencies (now transitive via `cryptovault`).

### Fixed
- **A wrong master secret or a corrupt wrapped key no longer wipes the database**
  (REQ-V35). The former "reset on unrecoverable salt" auto-sanitisation is retired:
  an *absent* wrapped key bootstraps fresh (brand-new or pre-envelope DB), but a
  *present-but-unopenable* one fails with a typed, retryable error and leaves all
  data intact. FEC-uncorrectable `vault_meta` surfaces as a distinct
  `VaultMetaCorrupt` error rather than a silent wrong key.
- Concurrent first-open of a fresh DB now yields a single DEK via an atomic
  `BEGIN IMMEDIATE` bootstrap (no double-bootstrap race).

## [0.7.0] - 2026-06-28

### Added
- **Tiered Agnostic Memory subsystem** (`src/memory/`). Replaces the naive
  "load all history into context" approach with a full local RAG pipeline:
  embedding-indexed vector store (any openai-compat endpoint, default
  `nomic-embed-text-v2-moe:latest` on Ollama), composite reranker (similarity +
  recency + salience), decay/eviction with configurable retention semantics, hard
  supersession via an off-hot-path distiller, an always-injected preference
  profile, and a token-budget context assembler with fixed priority
  (system → profile → ranked recalls → current turn). Text **and** vectors are
  encrypted at rest via `CryptoVault`; the ANN index lives only in RAM. The
  v0.6.0 "load all" behavior is preserved as `mode = "load_all"` (benchmark
  control). Built-in two-arm benchmark (`cargo run --bin bench_memory`) confirms
  `selective` matches `load_all` recall at ~35 % of context tokens with lower
  staleness. New `[memory]` and `[embedding]` sections in `magi.toml`. Backward
  compatible: all 387 existing tests remain green; `load_all` reproduces the prior
  behavior exactly. See `docs/TIERED-MEMORY.md` for the full reference and
  `docs/E2E-TESTING.md` for hands-on testing.

- **Configurable auto-approval for the MAGI `consult` tool** (`[magi] auto_approve`).
  When set to `true`, the agent tool loop auto-approves autonomous `consult` launches
  (the main LLM self-routing to the 3-perspective consensus) without prompting. A
  `StreamPiece::Notice` is emitted in the TUI before the tool runs so the user knows
  the potentially long consensus is in progress. Default `false` (opt-in, safe). The
  explicit `/consult` TUI command remains user-initiated and is never gated regardless
  of this flag. Documented in `docs/magi.toml.example` and scaffolded by `--init-config`.

### Fixed
- **Approval-gate spam on sequential safe tool calls.** The approval gate
  previously prompted the user for every tool call uniformly, including safe
  local operations. A model storing N facts via `project_knowledge` produced N
  approval prompts. Added a `requires_approval()` method to the `Tool` trait
  (default `true`, safe-by-default). Read-only and local-memory tools — `view`,
  `ls`, `grep`, and `project_knowledge` — override to `false` and are
  auto-approved without emitting an `ApprovalRequest`. Shell execution (`bash`),
  file writes (`edit`), and multi-model consensus (`consult`) keep the default
  and still require explicit user approval.

### Deferred (tracked in internal dev-docs)
- **#14** Envelope encryption (key rotation / crypto-shredding / multi-tenancy) — enterprise roadmap.
- **#17** Runtime warning visibility — malformed tool-JSON (#4) and poison recovery (#8) warnings remain stderr-only under the alt-screen; startup/login warnings are already surfaced.
- **#18** Blob version-dispatch / migration — the blob version byte is detection-only; a future format bump still needs a migrate-or-reset path.
- **OpenAI key in keyring / `key.txt`** — `OPENAI_API_KEY` is currently env-only; aligning it with the Anthropic discovery order is a tracked follow-up.

## [0.6.0] - 2026-06-09

**BREAKING:** the default backend with no `magi.toml`/env changed from **Anthropic**
to **Ollama** (`http://localhost:11434/v1`, `kimi-k2.6:cloud` + the qwen3.5/gpt-oss/deepseek
trio). Anthropic still works but is now **opt-in** (`provider = "anthropic"` or
`MAGI_PROVIDER=anthropic`).

### Added
- `src/defaults.rs` — single source of truth for the built-in default profile.
- `--init-config` CLI flag and `/init-config` TUI command to scaffold a default `magi.toml`.
- Startup notice (when no `magi.toml`) and an actionable error (when Ollama is unreachable),
  both DRY-interpolated from the default constants.

### Changed
- `resolve_provider`/`resolve_openai_base_url` defaults → Ollama-first.
- `resolve_openai_model` no longer errors when unset — returns the built-in default.
- The MAGI trio defaults to qwen3.5/gpt-oss/deepseek on the openai path when `[magi]` is absent.

### Known limitations
- The built-in defaults assume **Ollama**. If you point `provider=openai` at real OpenAI
  (or another non-Ollama service) WITHOUT setting `OPENAI_MODEL` / `[openai].model` / `[magi]`,
  the defaults (`kimi-k2.6:cloud` + the `:cloud` trio) will not exist there — set them
  explicitly.
- The default `:cloud` model tags reflect the Ollama catalog at release time and may rot over
  time; refresh per release. They live in one place (`src/defaults.rs`) for easy maintenance.

## [0.5.2] - 2026-06-09

Reasoning-model streaming (#24). Reasoning models (e.g. `kimi-k2.6:cloud`,
`deepseek-r1`) stream their chain-of-thought in `delta.reasoning` with empty
`delta.content`; the OpenAI-compatible parser previously ignored it, so the TUI
showed a frozen blank during a long reasoning phase. TUI/provider only — the agent
loop, persistence, crypto, and `/consult` are unchanged.

### Added
- **Live "thinking" feedback for reasoning models.** By default a compact
  `🤔 MAGI Pensando…` indicator with an animated spinner shows while the model
  reasons, instead of a frozen blank. The reasoning text itself is **never
  persisted** to the encrypted store.
- **`/toggle-show-thinking`** switches between the compact indicator (default) and
  a verbose mode that streams the full chain-of-thought inline (useful for
  debugging). Added to `/help`.

### Fixed
- Reasoning-model chats no longer appear frozen: the parser now surfaces
  `delta.reasoning` as a distinct stream instead of dropping it.

### Known limitations
- In the verbose mode (`/toggle-show-thinking`), the reasoning and the answer are
  streamed into the same message bubble and can visually run together; the default
  compact mode is unaffected.

## [0.5.1] - 2026-06-08

TUI usability patch — makes long output (notably the `/consult` report) readable.
TUI-only; the agent loop, providers, persistence, and crypto are unchanged.

### Added
- **Conversation scrollback.** Scroll the history line-by-line with `↑`/`↓`, by page
  with `PgUp`/`PgDn`, and to the top/bottom with `Home`/`End`. The pane follows the
  tail by default and snaps back to the newest content on a new/streaming reply.

### Fixed
- **Tall messages were truncated.** A single message taller than the pane (e.g. a
  MAGI consult report) previously showed only its tail with no way to scroll up; it
  is now fully reachable via the new scrollback.
- **Markdown/table indentation was lost.** `wrap_message` now returns a fitting line
  unchanged, preserving leading indentation and internal alignment spaces (bullets,
  ASCII tables, box-drawing). Wrapping is measured in terminal display columns, so
  CJK/emoji (2-column) glyphs no longer wrap a column early.
- **Long prompts were cut off at the right border.** The input box now grows (up to
  6 rows) and wraps long/multi-line prompts instead of truncating them.

### Known limitations
- `wrap_message` still collapses internal whitespace when reflowing a line that is
  *wider than the terminal* (word-boundary wrap); fitting lines are untouched. Mid-
  prompt cursor placement in an input taller than 6 rows is approximate (exact while
  typing at the end).

## [0.5.0] - 2026-06-08

Per-agent model selection for the three MAGI perspectives (Melchior / Balthasar /
Caspar). Additive and backward-compatible — the `Provider` trait, the agent loop,
config discovery, the `consult` tool, `MagiCoreProviderAdapter`, and encrypted
memory are unchanged; with no `[magi]` configuration the behavior is identical to
v0.4.0.

### Added
- Per-agent MAGI model selection via `magi.toml` `[magi]` section and
  `MAGI_MODEL_{MELCHIOR,BALTHASAR,CASPAR}` env vars. Opt-in; absent = all three
  perspectives share the principal model (backward compatible). Overrides reuse
  the principal backend's endpoint/key and vary only the model — true
  cross-family lineage diversity requires an Ollama-style multi-family endpoint.

## [0.4.0] - 2026-06-07

First step of the multi-backend MAGI integration: the `magi-core` crate (already
declared but unused) is now wired in, giving the agent a **three-perspective
consensus** capability (Melchior / Balthasar / Caspar). Additive only — the
`Provider` trait, the agent loop, config discovery, and encrypted memory are
unchanged.

### Added
- **`consult` tool** — exposes `magi-core`'s 3-perspective consensus as a tool the main LLM can invoke autonomously when a question has genuine trade-offs. Routing emerges from the existing tool loop (no separate classifier); each call passes the inline approval gate, which doubles as cost control for the ≈ 3 model calls.
- **`/consult <question>` command** — forces a MAGI consensus directly in the TUI, bypassing the router and approval, rendering the verbatim report (three perspectives + verdict). Runs `analyze` in a joined task so a panic in `magi-core` surfaces as a recoverable error instead of killing the runner; blocks the session while it runs, like a normal turn.
- **`MagiCoreProviderAdapter`** (`src/agent/magi_adapter.rs`) — bridges magi-rs's resolved `Provider` to `magi_core::provider::LlmProvider`, so the consensus reuses the same backend + credentials (Anthropic or any OpenAI-compatible endpoint). No second LLM config layer.
- **StaticProvider guard** — `consult` is registered (and `/consult` works) only when a real provider is configured; on the static/no-key path the tool is absent and `/consult` reports that a provider is required.
- **Degraded surfacing** — a consult that completes with fewer than three agents is prefixed `[DEGRADED: …]` so a low-quality consensus is never silent.
- New tests for the adapter (assembled text, role-fold delimiter, error mapping), the tool (contract + consensus + invalid-args + empty-query + oversized-query + backend-error via `magi-core`'s `RoutingMockProvider`), and the `/consult` parser + a full-report render-safety test. Total tests: **147** (was 136).

### Changed
- `magi-core` dependency bumped `1.0` → `1.1` (no features enabled; `reqwest 0.12` is **not** pulled — the adapter reuses magi-rs's existing `reqwest 0.11` stack). `test-utils` enabled as a dev-dependency for `RoutingMockProvider`.

### Security
- The verbatim consult report (LLM-generated) is run through `sanitize_text` before rendering — strips ANSI escapes / control characters, matching the streaming-delta path. Consult input is length-capped (8192 bytes) on both the tool and forced `/consult` paths, and empty queries are rejected before any model call.

### Known limitations
- **System-prompt fold.** `magi-core` differentiates its three personas via distinct *system* prompts, but magi-rs's `Provider` has no system-role channel yet, so the system text is folded into the user turn (behind an explicit delimiter). On weak / small local models this can weaken persona divergence and JSON adherence. Revisit when magi-rs gains a system-prompt channel.
- **`CompletionConfig` not applied.** The adapter does not forward `max_tokens` / `temperature` to the backend (the `Provider` trait exposes no per-call knobs). Deferred.
- **Deferred follow-ups (internal dev-docs):** repetitive-call detection keyed on the consult `query` argument, and an `InputTooLarge` early-rejection UX for oversized `/consult` input.

## [0.3.1] - 2026-05-26

TUI rendering hotfix — two pre-existing UX bugs found during the 0.3.0
end-to-end smoke (long-reply truncation + new-message scroll-off). Both
affect all providers (Anthropic, OpenAI-compatible, Static). Render-only,
no API / trait / contract changes.

### Fixed
- **Conversation pane truncated long messages** at the panel's right border instead of wrapping. New `wrap_message` helper (`src/tui/mod.rs`) word-wraps each message by `chars().count()` (UTF-8 char-safe) to the panel's inner width, preserves existing `\n` as hard breaks, and hard-splits absurdly long words. One message stays one `ListItem` so Selection / Visual navigation (`app.selected_index → app.messages[i]`) is unchanged.
- **Conversation pane did not auto-scroll** when new messages overflowed the bottom. New `effective_selection` / `effective_highlight_symbol` helpers pin the last message as the selected row in Normal mode (ratatui's `List` auto-scrolls to keep the selected item visible) while suppressing the `">> "` highlight prefix so the pin is invisible. Selection / Visual modes still show the prefix and use the user-chosen index.
- **Streaming tail of a tall response stayed off-screen** (and on some terminals "jumped/reset" when the message exceeded the conversation pane). The follow-tail fix now tail-truncates the rendered last message in Normal mode to its trailing `viewport_inner_height` wrapped lines, so streaming visibly scrolls line-by-line. Selection / Visual modes still render every wrapped line so the full message remains reviewable via `Ctrl+S → ↑`.

### Added
- 15 new tests covering the four render-time helpers (`wrap_message`: word wrap normal / multibyte / oversized word / embedded newlines / width-0 / empty; `effective_selection`: follow-tail in Normal mode / chosen index in Selection / Visual / empty messages; `effective_highlight_symbol`: by mode; `tail_lines`: keeps last N / unchanged when max≥len / max=0 no-op / empty / shift-by-one streaming tick). Total tests: **136** (was 121).

### Known limitations
- **Leading indentation and internal whitespace runs are collapsed** when a message is wrapped. `wrap_message` uses `split_whitespace`, which drops leading spaces and multi-space gaps. Markdown bullets like `"  - item"` render as `"- item"`; preformatted blocks lose alignment. Tracked for a follow-up patch (preserve per-line indentation via a different tokenizer).
- **Wrap width is measured in `chars().count()`, not terminal display width.** CJK / emoji glyphs that occupy two cells will wrap one column early; combining marks may wrap one column late. Tracked for a follow-up that switches to `unicode-width`.

## [0.3.0] - 2026-05-25

First multi-backend release. Magi can now talk to any OpenAI-compatible Chat
Completions endpoint — local **Ollama**, OpenAI, Groq, OpenRouter — selected by
a new `magi.toml`, alongside the existing Anthropic Messages API path. No
regression: the Anthropic surface and all its tests are byte-equivalent.

### Added
- **`OpenAiCompatibleProvider`** (`src/agent/provider.rs`) — Chat Completions over `{base_url}/chat/completions` with `stream:true`, a `stream::unfold` SSE state machine that finalizes on `finish_reason` / `[DONE]` / stream-end with an idempotent `done` guard (`MessageDone` emitted exactly once), `data:` prefix tolerant of optional space, malformed lines swallowed, HTTP non-2xx surfaced as `Err`, `MAX_SSE_BUFFER_BYTES` (8 MiB) cap. Constructor takes a named-field `OpenAiSettings` struct (no positional same-type swap).
- **`magi.toml` configuration** (`src/config.rs`) — `MagiConfig` / `OpenAiConfig` / `AnthropicConfig`, `serde(deny_unknown_fields)` so typos (and `api_key`) fail at parse time, **env > TOML > defaults** precedence (`MAGI_PROVIDER` / `OPENAI_BASE_URL` / `OPENAI_MODEL`). `MagiConfig::load` distinguishes `NotFound` (silent default) from other I/O errors (surfaced as a TUI startup notice). Reference `docs/magi.toml.example` is committed; user-local `magi.toml` is gitignored.
- **Coalesced `map_messages` / `map_tools`.** Same-turn parallel `Content::ToolUse` blocks collapse into ONE assistant message with a `tool_calls` array; User `Text`/`ToolResult` are emitted in **content order** (each block as its own OpenAI message). `Tool::input_schema` forwards as `tools:[{type:"function",function:{…}}]`.
- **Bounded tool-call accumulator.** `MAX_TOOL_CALL_SLOTS = 64` caps streamed `tool_calls[].index`; over-cap **warns** (`eprintln`) instead of dropping silently (RF-8). Orphan `arguments` fragments (slot has neither id nor name yet) are skipped with a warning to prevent mis-attribution.
- **`tests-121` (was 95).** 26 new tests — config parsing & precedence; coalesced message mapping; mixed-content User ordering; OpenAI text streaming (with [DONE], without [DONE]/finish_reason, malformed line, HTTP error, stream-end-only finalize); fragmented `tool_calls` assembly; bounded index; post-stop TextDelta suppression; args-before-id skip; `resolve_provider` wiring.

### Security
- **`OPENAI_API_KEY` from environment only.** Never read from `magi.toml`; `deny_unknown_fields` rejects an `api_key` field at parse time. The dummy `"ollama"` fallback for local Ollama is documented inline; real backends fail loudly with 401 if the env var is unset.
- **Keyring separation unchanged.** The Anthropic key (`magi-rs`) and DB master key (`magi-rs-internal`) remain in separate keyring services; `test_agent_history_resilience_to_key_rotation` is untouched.

### Documentation
- **README "Configuration: `magi.toml`" section** — full precedence table for the four settings (`MAGI_PROVIDER`/`OPENAI_BASE_URL`/`OPENAI_MODEL`/`ANTHROPIC_MODEL`), keys-never-in-TOML invariant (and the `deny_unknown_fields` parse-error consequence), Ollama quickstart.
- **`reqwest::Client` no-timeout rationale documented** — local Ollama can spend tens of seconds on cold-load before the first SSE event; a total-request timeout would truncate healthy long streams. Stream-side termination is handled by the three finalize triggers + `MAX_SSE_BUFFER_BYTES`.

## [0.2.1] - 2026-05-25

### Changed
- **Docs/UX:** the README and the static-mode startup hint now recommend a **standard API key** (`ANTHROPIC_API_KEY` / `key.txt`) as the supported path, and mark `/login` (OAuth) as **best-effort** — it reuses Anthropic's Claude Code OAuth client, so it may be rate-limited or blocked. No behavior change.

## [0.2.0] - 2026-05-25

First **public** release — the repository is now open-source (MIT OR Apache-2.0). Functionally identical to 0.1.2, promoted to the semver-correct minor version for the breaking on-disk format change made during the 0.1.x security hardening (a pre-0.1.1 `.magi-rs-memory.db` resets on upgrade). See **[0.1.1]** for the security-audit remediation and **[0.1.2]** for the docs + release-pipeline work this consolidates.

## [0.1.2] - 2026-05-24

### Changed
- **Internal development docs are no longer shipped.** The audit, strategy, and roadmap notes moved to a gitignored `dev-docs/` directory — excluded from the repository and the published crate.

### Added
- **`docs/OVERVIEW.md`** — a public overview of Magi, the `magi-core` foundation, and the multi-perspective (MAGI) philosophy.
- **Windows binary in releases.** Tagged releases now attach `magi-rs-vX.Y.Z-windows-x86_64.zip` (the compiled `magi-rs.exe` + README + licenses) alongside the source archives.

## [0.1.1] - 2026-05-24

First security-hardened release. The internal audit
(8 CRITICAL + 6 WARNING) is fully remediated and all
non-deferred follow-ups are closed. Verified gate: 95/95 tests, `clippy
--all-targets` clean, `fmt`/`build --release`/`doc` clean, `audit` baseline.

### Security
- **Filesystem sandbox closed on all tools.** `write.rs` and `grep.rs` now route through `PathGuard::validate`; the `bash` tool sandboxes every non-flag argument, closing the Windows forward-slash absolute-path escape (`C:/...`).
- **Crypto hardening.** Independent `OsRng` nonce per record; Argon2id pinned to OWASP 2025 parameters (64 MiB, t=3, p=4); a hard cap on the decrypt length-prefix to prevent hostile allocations.
- **Ephemeral fallback (no constant key).** An inaccessible OS keyring degrades the session to in-memory only — it never falls back to a constant passphrase.
- **OAuth callback timeout.** The PKCE callback server is bounded by a timeout; state is enforced (CSRF protection).
- **Secrets separation preserved.** API key (`magi-rs`) and DB master key (`magi-rs-internal`) remain in separate keyring services; rotating one never invalidates the other.

### Changed
- **One key derivation per session (B′).** Argon2 no longer runs per record. A per-DB salt is persisted in a new `vault_meta` table and the 32-byte key is derived once at construction and cached (`Zeroizing`), removing the O(N) cost on history load. Blob layout is now `[u8 version][u32 LE len][RS(nonce‖ciphertext)]`.
- **SSE wire path is byte-buffered.** The streaming reader buffers `Vec<u8>` and decodes only complete `\n\n` blocks, so a multi-byte UTF-8 character split across a network chunk is never decoded mid-character.
- **`tool_use` is assembled from the stream.** `AnthropicProvider` now builds `Content::ToolUse` from `content_block_start`/`input_json_delta`/`stop` (previously tools were inert with the real provider).
- **In-session `/login`.** A successful login rebuilds the running agent's provider without a restart; the canned `StaticProvider` history is cleared **only** when the prior provider was static, and the banner refreshes.
- **Dependency:** `magi-core` bumped `0.6` → `1.0.1`.

### Added
- **Salt integrity & self-heal.** The per-DB salt is RS-encoded with a SHA-256 checksum; an absent/corrupt/unrecoverable salt self-heals to a fresh start instead of bricking, and minor bit-rot is corrected.
- **Poisoned-lock recovery.** A poisoned connection mutex recovers (`into_inner` + warn-once) so persistence continues for the rest of the session.
- **Idempotent salt bootstrap.** Salt read + reset run inside a `BEGIN IMMEDIATE` transaction with an in-tx re-check, closing the concurrent first-open TOCTOU.
- **Blob version byte** (`BLOB_VERSION`), validated before any allocation.
- **TUI startup notices.** Keyring-unavailable (no persistence) and history-reset (fresh start) states surface as startup `Info` messages instead of pre-TUI stderr; the store exposes `was_reset()`.

### Fixed
- Malformed tool-input JSON logs a warning (tool name/id + error) instead of silently degrading to `{}`.
- Default-model fallback handles a blank/whitespace/absent `key.txt` model line.
- Full `clippy --all-targets` and tree-wide `rustfmt` baseline established.

### Notes
- **No on-disk migration (D2/D6 fresh-start).** The blob layout and KDF changed without a migration path: a pre-0.1.1 `.magi-rs-memory.db` deterministically resets to a fresh, empty store on first open. This is intentional for the pre-1.0 single-user threat model.

## [0.1.0] - 2026-05-16

Initial pre-release, published primarily to reserve the `magi-rs` crate name.

### Added
- Async agent loop with the Anthropic Messages API provider.
- Sandboxed tools: `ls`, `view`, `edit`, `grep`, `bash`, `project_knowledge`.
- Encrypted SQLite memory (Argon2id + AES-256-GCM-SIV + Reed-Solomon FEC), WAL mode.
- `ratatui` TUI with Normal / Selection / Visual modes and Unicode-safe input.
- OAuth (PKCE) login and OS keyring integration, with `magi-rust` legacy migration.

[Unreleased]: https://github.com/BolivarTech/magi/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/BolivarTech/magi/compare/v0.14.3...v0.15.0
[0.14.3]: https://github.com/BolivarTech/magi/compare/v0.14.2...v0.14.3
[0.14.2]: https://github.com/BolivarTech/magi/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/BolivarTech/magi/compare/v0.13.1...v0.14.1
[0.13.1]: https://github.com/BolivarTech/magi/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/BolivarTech/magi/compare/v0.12.2...v0.13.0
[0.12.2]: https://github.com/BolivarTech/magi/compare/v0.12.0...v0.12.2
[0.12.0]: https://github.com/BolivarTech/magi/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/BolivarTech/magi/compare/v0.10.4...v0.11.0
[0.10.4]: https://github.com/BolivarTech/magi/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/BolivarTech/magi/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/BolivarTech/magi/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/BolivarTech/magi/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/BolivarTech/magi/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/BolivarTech/magi/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/BolivarTech/magi/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/BolivarTech/magi/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/BolivarTech/magi/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/BolivarTech/magi/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/BolivarTech/magi/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/BolivarTech/magi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/BolivarTech/magi/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/BolivarTech/magi/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/BolivarTech/magi/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/BolivarTech/magi/releases/tag/v0.2.1
[0.2.0]: https://github.com/BolivarTech/magi/releases/tag/v0.2.0
[0.1.2]: https://github.com/BolivarTech/magi/releases/tag/v0.1.2
[0.1.1]: https://github.com/BolivarTech/magi/releases/tag/v0.1.1
[0.1.0]: https://github.com/BolivarTech/magi/releases/tag/v0.1.0
