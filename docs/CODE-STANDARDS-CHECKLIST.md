# Code Standards Checklist

Per-file checklist that Loop 1 (`/requesting-code-review`) and MAGI (`/magi:magi`) walk over
every touched file, in addition to `cargo nextest` / `clippy -D warnings` / `fmt --check` /
`build --release` / `doc` / `audit` / `deny check licenses` (`CLAUDE.local.md` §0.1).

## Per file

- [ ] **SRP** — each function/module does exactly one thing; if the purpose needs "and" to be
      described, split it.
- [ ] **DRY** — zero duplicated blocks of 3+ lines; extract to a shared function/constant.
- [ ] **Zero magic numbers** — except `0`, `1`, `-1`; everything else is a named constant
      (`SCREAMING_SNAKE_CASE`).
- [ ] **Useful rustdoc** on every public item — explains the purpose without repeating the
      item's name; includes `# Errors` if the function returns `Result`; includes `# Examples`
      if the usage is not trivial.
- [ ] **Import order** — std → external → crate, each group separated by a blank line.
- [ ] **File header** — every new file opens with `// Author: Julian Bolivar`,
      `// Version: 1.0.0`, `// Date: YYYY-MM-DD`.
- [ ] **Big-O on nested loops** — any loop nesting documents its expected complexity and why
      it is acceptable (or gets refactored if it is not).
- [ ] **Justified and pinned dependencies** — every new dependency has a documented reason
      (commit/PR) and a pinned version in `Cargo.toml`.
- [ ] **Minimum test coverage per public function** — at least one "happy path" case and at
      least one edge case (empty, boundary, error) per new public function.

## Mechanical gates backing this checklist

| Gate | Command | What it certifies |
|---|---|---|
| Compilation | `cargo build --release` | The crate builds without warnings in release mode. |
| Tests | `cargo nextest run` | The whole suite passes; no test broken by the change. |
| Lints | `cargo clippy --all-targets -- -D warnings` | Zero clippy warnings, including
  `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`indexing_slicing`/
  `string_slice` inside `src/vault/` (denied at the module level). |
| Formatting | `cargo fmt --check` | The code follows `rustfmt.toml` (`max_width = 100`). |
| Documentation | `cargo doc --no-deps` | Rustdoc builds without warnings; `missing_docs` **and
  `clippy::missing_docs_in_private_items`** are `deny` inside `src/vault/` (MS2 Task 0) — EVERY
  item, public **or `pub(crate)`/private**, requires rustdoc. Verified 2026-07-17: a
  `pub(crate) fn` without docs breaks the build. |

## New MS2 files (walk checklist (B) for each one)

Every new vault file goes through the "Per file" checklist above at each
`/verification-before-completion` and at the §6 gate:

- [ ] `src/vault/memguard.rs` (Task 1) — `MaskedDek` + `harden_process`
- [ ] `src/vault/store.rs` (Task 2) — `vault` table + `SecretStore` CRUD
- [ ] `src/vault/master.rs` (Task 4) — passphrase resolution + `zxcvbn`
- [ ] `src/vault/cli.rs` (Task 6) — `clap` subcommands `ls`/`set`/`rm`/`passwd`
- [ ] `src/vault/envelope.rs` (Task 5, extension) — `rekey_envelope`
- [ ] `src/vault/error.rs` (Tasks 2/4/6, new variants + English correction of the MS1 ones)

## New MS1 (headless) files — walk checklist (B) for each one

`src/headless/` module (lives in `lib.rs` as `pub mod headless`, same as `vault`, so fuzz/coverage
can link — REQ-H00). The lint attrs of `src/headless/mod.rs` are **identical** to those of
`src/vault/mod.rs` (`deny(missing_docs, missing_docs_in_private_items, unwrap_used[not(test)],
…)`). `cargo llvm-cov` coverage **≥ 90 %** over `src/headless/` **and `src/system/workspace.rs`**
(documented exclusions for pure glue).

- [x] `src/headless/mod.rs` (Task 0) — boundary + lint attrs (REQ-H00) + re-exports
- [x] `src/headless/error.rs` (Task 0) — `HeadlessError` (`thiserror`) + exhaustive `From<VaultError>`
- [x] `src/headless/types.rs` (Task 0) — shared types DECLARING the MS1↔MS2 contract (`pub(crate)`)
- [x] `src/headless/test_support.rs` (Task 0, `#[cfg(test)]`) — generic `with_var` environment helper
- [ ] `src/headless/input.rs` (Tasks 4/5/6) — bounded read + auto-detect + envelope parser + resolution
- [x] `src/headless/output.rs` (Task 7) — rich text/JSON formatting + truncation + error redaction
- [ ] `src/headless/log.rs` (Task 8) — JSONL to `.magi/logs/`, levels, count+size retention, redaction
- [ ] `src/headless/exit.rs` (Task 9) — exit-code taxonomy (0/1/2/3)
- [ ] `src/system/workspace.rs` (Tasks 1/2) — discover/init `.magi/` (walk-up, symlink-reject, perms, atomic)

| Vulnerabilities | `cargo audit` | No known advisories in the dependency tree. |
| Licenses | `cargo deny check licenses` | Only permissive licenses listed in
  `deny.toml`. |
| Secrets | `cargo nextest run --test no_hardcoded_secrets` | No `.rs` file under `src/`
  carries key-shaped hardcoded material (`sk-ant-api...`, `-----BEGIN` blocks), <!-- allow-secret-scan --> and no
  doc or config file (`.md`/`.toml`/`.yml`/`.yaml`/`.example` at the root, `docs/` and `.github/`)
  carries private IPs or absolute user paths either. A line is exempted with the
  `allow-secret-scan` marker, which applies only to that line. <!-- allow-secret-scan --> |

Any finding that does not fit a mechanical-gate category above, but does fit the manual
checklist above, is reported as a review finding (Loop 1 / MAGI) — never silently ignored.

## Miri scope (REQ-V38) — Task 0b spike (2026-07-14)

**Empirically determined:** `cargo +nightly miri test vault::error` runs **clean** in this
environment (Windows MSVC) — all 3 pure `vault::error` tests pass under Miri (0.59s), with no
unsupported operations. Binaries touching SQLite (`rusqlite` bundled, C FFI) and tokio (OS
threads) fall **naturally outside** Miri's scope (their tests are filtered out; Miri cannot
interpret C FFI or the thread runtime).

**REQ-V38 scope (Task 9):** Miri runs over the vault's **pure code** — `error` and `envelope`
(`vault_meta` framing, keyless FEC, wrap/unwrap). **Excluded:** `store`/`database` (SQLite FFI).
**Pending confirmation in Task 2:** whether the crate's *crypto* (`cryptovault`: AES-256-GCM-SIV,
Argon2id) runs under Miri — `aes` may use AES-NI intrinsics that fall back to the portable
backend under Miri (expected via `cpufeatures`), and Argon2 (m=64 MiB) may be slow under
interpretation. If the crypto does not run under Miri, the scope is narrowed to `error` + the
**non-AEAD** framing/FEC of `envelope`, and the exclusion is documented (never claim a pass that
did not happen).

## Milestone hardening gate — result (Task 9, 2026-07-15)

**Miri (REQ-V38) — empirically verified scope:**
- ✅ `cargo +nightly miri test vault::error` runs **clean** (pure domain logic, no UB) — confirmed
  in the Task 0b spike.
- ⚠️ **The envelope's crypto** (`vault::envelope`) **does NOT run under Miri**: it invokes
  `cryptovault` (AES-256-GCM-SIV with possible AES-NI intrinsics + Argon2 at 64 MiB + Viterbi
  FEC), which Miri either cannot interpret or makes impractically slow. **Miri's scope is bounded
  to the pure logic** (`error`, framing/bounds-safety), with the crypto **excluded and
  documented** — a pass that did not happen is never claimed (contingency anticipated by REQ-V38).

**Fuzz (REQ-V39) — targets defined, execution in Linux CI:**
- The 2 targets exist under `fuzz/fuzz_targets/`: `fuzz_vault_meta_decode` (arbitrary bytes →
  `open_envelope`, invariant: never panic or wipe) and `fuzz_vault_blob_decrypt` (arbitrary blob →
  `decrypt_with_key`, handles non-UTF8).
- ⚠️ **`cargo-fuzz`/libFuzzer requires the `-fsanitize=fuzzer` sanitizer (clang/LLVM), NOT
  supported on Windows MSVC** (a known tooling limitation, not a code one). The targets **run in a
  Linux CI job with nightly**: `cargo +nightly fuzz run <target> -- -max_total_time=300`. The
  bounds-safety of the split they would exercise (`fuzz_open_entrypoint`) is additionally covered
  by the `clippy::indexing_slicing` lint (breaks the build) and by
  `test_fuzz_entrypoint_never_panics_on_arbitrary_input` (unit test).

## MS2 hardening gate (REQ-V38 Miri · REQ-V39 fuzz) — Task 8 (2026-07-17)

**Fuzz targets (REQ-V39) — 4 total, wired and unit-smoked:**
- MS1: `fuzz_vault_meta_decode`, `fuzz_vault_blob_decrypt`.
- MS2: `fuzz_secret_value_roundtrip` (arbitrary value → set/get, never panics),
  `fuzz_passphrase_input` (arbitrary lossy passphrase → `check_strength` + KEK derivation, never
  panics).
- Each entrypoint (`magi_rs::vault::fuzz_*_entrypoint`) has a **unit test** exercising it with
  degenerate inputs (empty, non-UTF8, large) under `cargo nextest` — local robustness coverage
  that DOES run on every §0.1.
- **The long run (≥ 30 min/target, coverage-guided) runs in Linux CI with nightly** (a dedicated
  job, not in the RGR loop or the §7 budget — long-running by design, `CLAUDE.local.md` §0.3).
  Consistent with the MS1 note: `cargo +nightly fuzz build` on Windows-MSVC starts the
  instrumentation (ASan/sancov) but linking with libFuzzer is problematic on MSVC (which is why
  the real gate is Linux CI); no Windows pass that was not actually verified is claimed. The
  entrypoints' bounds-safety is additionally guaranteed by `clippy::indexing_slicing` (breaks the
  build) + the unit tests above.

**Miri (REQ-V38) — MS2 scope (extends the Task 0b spike):**
- ✅ Covers the new **pure logic**: `memguard`'s XOR masking (buffer arithmetic, aliasing, init)
  and `master`'s `check_strength` (zxcvbn is pure Rust). `vault::error` remains clean.
- **Excluded** (Miri cannot run them, documented without faking a pass):
  - `store` and the `envelope`/`database` tests that touch **SQLite** (`rusqlite` bundled = C FFI).
  - The **syscalls** of `region::lock`/`mlock` and of `harden_process`
    (`RLIMIT_CORE`/`PR_SET_DUMPABLE`) — skipped under `#[cfg(miri)]`; Miri does not model OS
    syscalls.
  - The **Argon2id** derivation (`m=64 MiB`, very slow under interpretation) and the AES that may
    use AES-NI (falls back to the portable backend via `cpufeatures`, not always under Miri).
- **Runs in CI** alongside the fuzz targets; no pass is claimed for what was not executed.

## MS1-headless hardening gate (REQ-H35 fuzz · REQ-V38 Miri) — Task 10 (2026-07-18)

**Fuzz targets (REQ-H35) — 2 new, wired + built + smoke-tested locally:**
- `fuzz_headless_input` (arbitrary bytes → `read_input_bounded` + `parse_input` across the 3
  `InputFormat` modes; invariant: never panic/UB, no OOM from the bounded read nor stack overflow
  from the bounded JSON depth). Calls the module's `pub fn`s directly.
- `fuzz_sanitize_error` (arbitrary lossy string → `sanitize_error_message` +
  `redact_secret_patterns`; invariant: never panic/UB **and** **idempotent** redaction — a proxy
  for "no key-shaped pattern slips through unredacted"). The entrypoint is
  `magi_rs::headless::fuzz_sanitize_error_entrypoint` (`#[doc(hidden)] pub`, the same convention as
  the 4 vault `fuzz_*_entrypoint`s: exposes the `pub(crate)` boundary to the `fuzz/` crate
  **without** widening the documented public API).
- Each one has a **unit smoke test** under `cargo nextest`
  (`test_parse_input_smoke_never_panics_on_degenerate_bytes` in `input.rs`;
  `test_fuzz_sanitize_error_entrypoint_never_panics_on_arbitrary_input` in `output.rs`) with
  degenerate inputs (empty, non-UTF8, pathologically nested JSON, duplicate keys, a non-string
  `prompt`, strings with embedded `{`/`[`/keys) — robustness coverage that DOES run on every
  §0.1.
- ✅ **`cargo +nightly fuzz build` PASSES on Windows-MSVC** with `cargo-fuzz 0.13.2` + nightly
  `da80ed070` (the MSVC link limitation documented for the vault no longer applies with this
  tooling version). **The instrumented binary requires the ASan runtime on the PATH at runtime**
  (`clang_rt.asan_dynamic-x86_64.dll`, under `…\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\`); without it
  the `.exe` fails with `STATUS_DLL_NOT_FOUND` (0xc0000135) — not a target crash.
- ✅ **60 s local smoke, zero crashes:** `fuzz_headless_input` → **346 653 runs / 61 s**;
  `fuzz_sanitize_error` → **267 007 runs / 61 s** (the redactor's idempotence did not fail across
  ~267k adversarial inputs). The **long coverage-guided run (≥ 30 min/target)** is left for
  CI/§0.3, outside the RGR loop and the §7 budget.

**Miri (REQ-V38) — INFEASIBLE on the current nightly (toolchain regression, NOT a UB finding):**
- ❌ `cargo +nightly miri test headless::{input,output,exit}` **aborts with a rustc ICE**
  (`resolver_for_lowering_raw` panics during the lowering phase, **before** running any test) on
  `rustc 1.99.0-nightly (da80ed070 2026-07-14)`. The ICE happens while compiling the crate under
  Miri, not while running headless code.
- ✅ **Verified it is the toolchain, not the code:** `cargo +nightly miri test vault::error` —
  which ran **clean** under Miri on an earlier nightly (Task 0b spike) — **ICEs identically** on
  this nightly. The cause is the compiler, not the headless modules.
- **Robustness mitigation without Miri, honestly stated:** (a) the crate is
  `#![forbid(unsafe_code)]` crate-wide ⇒ there is no `unsafe` to host UB in (a Miri pass would be
  trivial by construction); (b) the 2 fuzz targets (build + smoke, zero crashes) exercise the
  untrusted parser and the redactor; (c) the unit smoke tests run on every §0.1. **No Miri pass
  that did not happen is claimed.** Re-enabling Miri requires a nightly without the ICE (or
  pinning a previously known-good one).

## MS2-headless hardening gate (REQ-H35 fuzz · REQ-V38 Miri) — Task 10 (2026-07-19)

MS2 scope over the **new pure-logic surface**: the per-tier authorization matrix
(`src/headless/policy.rs` — `Policy::approves`/`silences_soft_guards`/`warnings`). The untrusted
input parser (envelope + bounded read + `sanitize_error_message`) is from MS1 and is already
covered by `fuzz_headless_input` + `fuzz_sanitize_error` (MS1 Task 10). The runner/timeout/consult
touches `Agent` and subprocesses ⇒ **neither pure nor Miri-able**. Verified: MS2 introduced no
**new** untrusted-input surface without a fuzz target.

**Fuzz target (REQ-H35) — 1 new, wired + built + smoke-tested locally:**
- `fuzz_policy` (arbitrary bytes → `(tier_byte, tool_name)` → the whole public surface of
  `Policy`; invariants: **never panics** + **fails closed** — an approval implies a tool name
  known to some tier, an unknown one never returns `true`). The entrypoint is
  `magi_rs::headless::fuzz_policy_entrypoint` (`#[doc(hidden)] pub`, the same convention as the
  vault's/`output`'s `fuzz_*_entrypoint`s: exposes the boundary to the `fuzz/` crate without
  widening the documented public API). Fail-closed is verified with `debug_assert!` (which
  `cargo-fuzz` enables).
- Has a **unit smoke test** under `cargo nextest`
  (`test_fuzz_policy_entrypoint_never_panics_on_arbitrary_input` in `policy.rs`) with degenerate
  inputs (empty, out-of-range tier, non-UTF8 tail, unknown tool) — robustness that DOES run on
  every §0.1.
- ✅ **`cargo +nightly fuzz build` PASSES** on Windows-MSVC (`cargo-fuzz 0.13.2` + nightly
  `da80ed070`); builds all 7 targets. The instrumented `.exe` requires the ASan runtime on the
  PATH (`clang_rt.asan_dynamic-x86_64.dll`, under `…\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\`), same
  as MS1.
- ✅ **60 s local smoke, zero crashes:** `fuzz_policy` → **2 528 273 runs / 61 s**. The
  coverage-guided fuzzer discovered all 7 real tool names by CMP (`ls`/`view`/`grep`/`edit`/`bash`/
  `consult`/`project_knowledge`), exercising the whole matrix and the fail-closed branch without
  panicking. MS1's targets rebuild clean as part of the same `fuzz build`. The long coverage-guided
  run (≥ 30 min) is left for CI/§0.3.

**Miri (REQ-V38) — INFEASIBLE on the current nightly (the SAME MS1 ICE, still unresolved):**
- ❌ `cargo +nightly miri test headless::policy` **aborts with the same rustc ICE**
  (`resolver_for_lowering_raw` panics during the lowering phase, **before** running any test) on
  `rustc 1.99.0-nightly (da80ed070 2026-07-14)`. The ICE happens while compiling the crate under
  Miri, not while running `policy`'s logic.
- ✅ **Re-verified it is the toolchain, not the code:** `cargo +nightly miri test vault::error` —
  which ran **clean** under Miri on an earlier nightly (Task 0b spike) — **ICEs identically** on
  this nightly. The cause is the compiler, not `src/headless/policy.rs`.
- **Robustness mitigation without Miri, honestly stated:** (a) `#![forbid(unsafe_code)]`
  crate-wide ⇒ there is no `unsafe` to host UB in, and `policy` is purely arithmetic/matching logic
  (a Miri pass would be trivial by construction); (b) the `fuzz_policy` target (build + 2.5M runs,
  zero crashes) exercises the whole matrix; (c) the unit smoke test + `policy`'s 6 tests run on
  every §0.1. **No Miri pass that did not happen is claimed** — re-enabling Miri requires a
  nightly without the ICE.

## New MagiCore MS2 (v0.12.0) files — walk checklist (B) for each one

**Not to be confused with the Vault's "MS2" (v0.9.0, section above) — same milestone number,
different project (MagiCore's `sbtdd/spec-behavior.md`).** Eleven new `.rs` files under `src/`
(`git diff --name-status main..HEAD -- 'src/*.rs' 'src/**/*.rs'`, `A` entries), all with the
**same lint-attribute block** as `src/vault/mod.rs`/`src/headless/mod.rs`
(`deny(missing_docs, clippy::missing_docs_in_private_items, rustdoc::broken_intra_doc_links,
clippy::missing_errors_doc, clippy::missing_panics_doc)` + `cfg_attr(not(test), deny(unwrap_used,
expect_used, panic, todo, unimplemented, indexing_slicing, string_slice))`) and the REQ-A00b file
header (`// Author: Julian Bolivar` / `// Version` / `// Date`).

Eight live in the **lib** (`src/magi/`, `src/notices.rs`, `src/redact.rs` — reachable from
`tests/`/`fuzz/`); three in the **bin** (`src/agent/mode_classifier.rs`, `src/config/migrate.rs`)
because they need a bin-only type (`agent::provider::Provider`) or are specific to the TOML's
shape.

- [ ] `src/magi/mod.rs` — subsystem-shared constants (derived timeout scale, gate and probe
      limits) + wiring of the public submodules
- [ ] `src/magi/kind.rs` — `ProviderKind` vocabulary (`ollama` | `openai-compat` | `anthropic`,
      REQ-A01b) + parser
- [ ] `src/magi/mode.rs` — `Mode`/`ModeSource` vocabulary, closed label normalization,
      five-level resolution (`resolve_mode_guarded`, the only public door)
- [ ] `src/magi/gate.rs` — complexity gate: a pure predicate (character length vs. per-mode
      threshold, REQ-A20)
- [ ] `src/magi/probe.rs` — model measurement by composition over magi-core's `ProviderProbe`
      (REQ-A24); a three-state `Measurement`
- [ ] `src/magi/endpoint.rs` — `base_url` templates with credential placeholders
      (`[user]`/`[password]`), resolved from the vault in memory (REQ-A16c)
- [ ] `src/magi/report_anchors.rs` — named anchors into magi-core's report markdown, for bounded
      output truncation (REQ-A11b)
- [ ] `src/agent/mode_classifier.rs` (bin) — mode classification over the PRINCIPAL provider
      (REQ-A07c); implements the pure `magi_rs::magi::mode::ModeClassifier` trait
- [ ] `src/config/migrate.rs` (bin) — a detection pass PRIOR to parsing, for the migration
      patterns of a v0.11.0 `magi.toml` (REQ-A21b), dated technical debt retired in v0.13.0
- [ ] `src/notices.rs` — tiering and ordering of startup notices (`Blocking` < `Resolution` <
      `Info`)
- [ ] `src/redact.rs` — redacts a URL's `userinfo` by POSITION, never by content (REQ-A16)
