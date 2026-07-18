# Code Standards Checklist

Checklist por archivo que Loop 1 (`/requesting-code-review`) y MAGI (`/magi:magi`) recorren
sobre cada archivo tocado, en adición a `cargo nextest` / `clippy -D warnings` / `fmt --check` /
`build --release` / `doc` / `audit` / `deny check licenses` (§0.1 del `CLAUDE.local.md`).

## Por archivo

- [ ] **SRP** — cada función/módulo hace una sola cosa; si el propósito necesita "y" para
      describirse, dividirlo.
- [ ] **DRY** — cero bloques de 3+ líneas duplicados; extraer a una función/constante
      compartida.
- [ ] **Cero magic numbers** — salvo `0`, `1`, `-1`; todo lo demás es una constante nombrada
      (`SCREAMING_SNAKE_CASE`).
- [ ] **Rustdoc útil** en todo ítem público — explica el propósito sin repetir el nombre del
      ítem; incluye `# Errors` si la función devuelve `Result`; incluye `# Examples` si el uso
      no es trivial.
- [ ] **Orden de imports** — std → externos → crate, cada grupo separado por una línea en
      blanco.
- [ ] **Header de archivo** — todo archivo nuevo abre con `// Author: Julian Bolivar`,
      `// Version: 1.0.0`, `// Date: YYYY-MM-DD`.
- [ ] **Big-O en loops anidados** — cualquier anidamiento de bucles documenta su complejidad
      esperada y por qué es aceptable (o se refactoriza si no lo es).
- [ ] **Dependencias justificadas y pineadas** — toda dependencia nueva tiene una razón
      documentada (commit/PR) y una versión pineada en `Cargo.toml`.
- [ ] **Cobertura de test mínima por función pública** — al menos un caso "happy path" y al
      menos un caso borde (vacío, límite, error) por función pública nueva.

## Gates mecánicos que respaldan esta checklist

| Gate | Comando | Qué certifica |
|---|---|---|
| Compilación | `cargo build --release` | El crate compila sin warnings en modo release. |
| Tests | `cargo nextest run` | Toda la suite pasa; ningún test roto por el cambio. |
| Lints | `cargo clippy --all-targets -- -D warnings` | Cero warnings de clippy, incluyendo
  `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`indexing_slicing`/
  `string_slice` dentro de `src/vault/` (denegados a nivel de módulo). |
| Formato | `cargo fmt --check` | El código sigue `rustfmt.toml` (`max_width = 100`). |
| Documentación | `cargo doc --no-deps` | Rustdoc compila sin warnings; `missing_docs` **y
  `clippy::missing_docs_in_private_items`** son `deny` dentro de `src/vault/` (MS2 Task 0) —
  TODO ítem, público **o `pub(crate)`/privado**, exige rustdoc. Verificado 2026-07-17: un
  `pub(crate) fn` sin doc rompe el build. |

## Archivos nuevos de MS2 (recorrer la checklist (B) por cada uno)

Cada archivo nuevo del vault se somete a la checklist "Por archivo" de arriba en cada
`/verification-before-completion` y en el gate §6:

- [ ] `src/vault/memguard.rs` (Task 1) — `MaskedDek` + `harden_process`
- [ ] `src/vault/store.rs` (Task 2) — tabla `vault` + `SecretStore` CRUD
- [ ] `src/vault/master.rs` (Task 4) — resolución de passphrase + `zxcvbn`
- [ ] `src/vault/cli.rs` (Task 6) — subcomandos `clap` `ls`/`set`/`rm`/`passwd`
- [ ] `src/vault/envelope.rs` (Task 5, ampliación) — `rekey_envelope`
- [ ] `src/vault/error.rs` (Tasks 2/4/6, variantes nuevas + corrección a inglés de las de MS1)

## Archivos nuevos de MS1 (headless) — recorrer la checklist (B) por cada uno

Módulo `src/headless/` (vive en `lib.rs` como `pub mod headless`, igual que `vault`, para
que fuzz/coverage linkeen — REQ-H00). Los lint attrs de `src/headless/mod.rs` son **idénticos**
a los de `src/vault/mod.rs` (`deny(missing_docs, missing_docs_in_private_items, unwrap_used[not(test)],
…)`). Cobertura `cargo llvm-cov` **≥ 90 %** sobre `src/headless/` **y `src/system/workspace.rs`**
(exclusiones documentadas para glue puro).

- [x] `src/headless/mod.rs` (Task 0) — frontera + lint attrs (REQ-H00) + re-exports
- [x] `src/headless/error.rs` (Task 0) — `HeadlessError` (`thiserror`) + `From<VaultError>` exhaustivo
- [x] `src/headless/types.rs` (Task 0) — tipos compartidos DECLARADOS del contrato MS1↔MS2 (`pub(crate)`)
- [x] `src/headless/test_support.rs` (Task 0, `#[cfg(test)]`) — helper genérico de entorno `with_var`
- [ ] `src/headless/input.rs` (Tasks 4/5/6) — lectura acotada + auto-detect + parser de envelope + resolución
- [x] `src/headless/output.rs` (Task 7) — formateo texto/JSON rico + truncado + redacción de errores
- [ ] `src/headless/log.rs` (Task 8) — JSONL a `.magi/logs/`, niveles, retención count+size, redacción
- [ ] `src/headless/exit.rs` (Task 9) — taxonomía de exit codes (0/1/2/3)
- [ ] `src/system/workspace.rs` (Tasks 1/2) — descubrir/init `.magi/` (walk-up, symlink-reject, perms, atómico)

| Vulnerabilidades | `cargo audit` | Sin advisories conocidos en el árbol de dependencias. |
| Licencias | `cargo deny check licenses` | Solo licencias permisivas listadas en
  `deny.toml`. |
| Secretos | `cargo nextest run --test no_hardcoded_secrets` | Ningún archivo `.rs` bajo
  `src/` contiene material tipo-clave hardcodeado (`sk-ant-api...`, bloques `-----BEGIN`). |

Todo hallazgo que no encaje en una categoría de la tabla de gates mecánicos, pero sí en la
checklist manual de arriba, se reporta como finding de review (Loop 1 / MAGI) — nunca se
ignora en silencio.

## Alcance de Miri (REQ-V38) — spike de Task 0b (2026-07-14)

**Determinado empíricamente:** `cargo +nightly miri test vault::error` corre **limpio** en este
entorno (Windows MSVC) — los 3 tests puros de `vault::error` pasan bajo Miri (0.59s), sin
operaciones no soportadas. Los binarios que tocan SQLite (`rusqlite` bundled, FFI en C) y tokio
(hilos de SO) quedan **naturalmente fuera** del alcance de Miri (sus tests se filtran; Miri no
puede interpretar FFI en C ni el runtime de hilos).

**Alcance de REQ-V38 (Task 9):** Miri corre sobre el **código puro del vault** — `error` y
`envelope` (framing de `vault_meta`, FEC keyless, wrap/unwrap). **Excluidos:** `store`/`database`
(SQLite FFI). **Pendiente de confirmar en Task 2:** que el *crypto* del crate (`cryptovault`:
AES-256-GCM-SIV, Argon2id) corra bajo Miri — el `aes` puede usar intrínsecos AES-NI que caen al
backend portable bajo Miri (esperado vía `cpufeatures`), y Argon2 (m=64 MiB) puede ser lento bajo
interpretación. Si el crypto no corre bajo Miri, el alcance se acota a `error` + al framing/FEC
**no-AEAD** de `envelope`, y se documenta la exclusión (nunca declarar un pase que no ocurrió).

## Gate de hardening de milestone — resultado (Task 9, 2026-07-15)

**Miri (REQ-V38) — alcance verificado empíricamente:**
- ✅ `cargo +nightly miri test vault::error` corre **limpio** (lógica de dominio pura, sin UB) — confirmado en el spike de Task 0b.
- ⚠️ El **crypto del envelope** (`vault::envelope`) **NO corre bajo Miri**: invoca `cryptovault` (AES-256-GCM-SIV con posibles intrínsecos AES-NI + Argon2 a 64 MiB + FEC Viterbi), que Miri no interpreta / hace impracticablemente lento. **Alcance de Miri acotado a la lógica pura** (`error`, framing/bounds-safety), con el crypto **excluido y documentado** — nunca se declara un pase que no ocurrió (contingencia prevista en REQ-V38).

**Fuzz (REQ-V39) — targets definidos, ejecución en CI Linux:**
- Los 2 targets existen en `fuzz/fuzz_targets/`: `fuzz_vault_meta_decode` (bytes arbitrarios → `open_envelope`, invariante: nunca panic ni borrado) y `fuzz_vault_blob_decrypt` (blob arbitrario → `decrypt_with_key`, maneja no-UTF8).
- ⚠️ **`cargo-fuzz`/libFuzzer requiere el sanitizer `-fsanitize=fuzzer` (clang/LLVM), NO soportado en Windows MSVC** (limitación conocida del tooling, no del código). Los targets se **ejecutan en un CI Linux con nightly**: `cargo +nightly fuzz run <target> -- -max_total_time=300`. El bounds-safety del split que ejercitarían (`fuzz_open_entrypoint`) está además cubierto por el lint `clippy::indexing_slicing` (rompe el build) y por `test_fuzz_entrypoint_never_panics_on_arbitrary_input` (unit test).

## Gate de hardening MS2 (REQ-V38 Miri · REQ-V39 fuzz) — Task 8 (2026-07-17)

**Fuzz targets (REQ-V39) — 4 totales, wired y unit-smoked:**
- MS1: `fuzz_vault_meta_decode`, `fuzz_vault_blob_decrypt`.
- MS2: `fuzz_secret_value_roundtrip` (valor arbitrario → set/get, nunca panic), `fuzz_passphrase_input`
  (passphrase arbitraria lossy → `check_strength` + derive KEK, nunca panic).
- Cada entrypoint (`magi_rs::vault::fuzz_*_entrypoint`) tiene un **test unitario** que lo ejercita
  con entradas degeneradas (vacía, no-UTF8, grande) bajo `cargo nextest` — cobertura de robustez
  local que SÍ corre en cada §0.1.
- **La corrida larga (≥ 30 min/target, coverage-guided) corre en CI Linux con nightly** (job dedicado,
  no en el loop RGR ni en el presupuesto §7 — long-running por diseño, §0.3 de `CLAUDE.local.md`).
  Consistente con la nota de MS1: `cargo +nightly fuzz build` en Windows-MSVC arranca la instrumentación
  (ASan/sancov) pero el enlace con libFuzzer es problemático en MSVC (por eso el gate real es CI Linux);
  no se declara un pase de Windows que no se verificó. El bounds-safety de los entrypoints está además
  garantizado por `clippy::indexing_slicing` (rompe el build) + los tests unitarios de arriba.

**Miri (REQ-V38) — alcance MS2 (extiende el spike de Task 0b):**
- ✅ Cubre la **lógica pura** nueva: el enmascarado XOR de `memguard` (aritmética sobre buffers,
  aliasing, init) y `check_strength` de `master` (zxcvbn es Rust puro). `vault::error` sigue limpio.
- **Excluidos** (Miri no puede ejecutarlos, documentado sin fingir un pase):
  - `store` y las pruebas de `envelope`/`database` que tocan **SQLite** (`rusqlite` bundled = FFI en C).
  - Las **syscalls** de `region::lock`/`mlock` y de `harden_process` (`RLIMIT_CORE`/`PR_SET_DUMPABLE`)
    — se saltean bajo `#[cfg(miri)]`; Miri no modela syscalls del SO.
  - La derivación **Argon2id** (`m=64 MiB`, muy lenta bajo interpretación) y el AES que puede usar
    AES-NI (cae al backend portable vía `cpufeatures`, no siempre bajo Miri).
- **Se corre en CI** junto al fuzz; no se declara un pase de lo que no se ejecutó.

## Gate de hardening MS1-headless (REQ-H35 fuzz · REQ-V38 Miri) — Task 10 (2026-07-18)

**Fuzz targets (REQ-H35) — 2 nuevos, wired + build + smoke ejecutados en local:**
- `fuzz_headless_input` (bytes arbitrarios → `read_input_bounded` + `parse_input` en los 3 modos
  de `InputFormat`; invariante: nunca panic/UB, sin OOM por la lectura acotada ni stack-overflow por
  la profundidad JSON acotada). Llama directo a los `pub fn` del módulo `input`.
- `fuzz_sanitize_error` (string lossy arbitrario → `sanitize_error_message` + `redact_secret_patterns`;
  invariante: nunca panic/UB **y** redacción **idempotente** — proxy de "ningún patrón tipo-clave se
  deja pasar sin redactar"). El entrypoint es `magi_rs::headless::fuzz_sanitize_error_entrypoint`
  (`#[doc(hidden)] pub`, misma convención que los 4 `fuzz_*_entrypoint` del vault: expone la frontera
  `pub(crate)` al crate `fuzz/` **sin** ensanchar la API pública documentada).
- Cada uno tiene un **unit-smoke** bajo `cargo nextest` (`test_parse_input_smoke_never_panics_on_degenerate_bytes`
  en `input.rs`; `test_fuzz_sanitize_error_entrypoint_never_panics_on_arbitrary_input` en `output.rs`)
  con entradas degeneradas (vacía, no-UTF8, JSON patológicamente anidado, dup-key, `prompt` no-string,
  strings con `{`/`[`/claves embebidas) — cobertura de robustez que SÍ corre en cada §0.1.
- ✅ **`cargo +nightly fuzz build` PASA en Windows-MSVC** con `cargo-fuzz 0.13.2` + nightly
  `da80ed070` (la limitación de link MSVC documentada para el vault ya no aplica con esta versión del
  tooling). **El binario instrumentado requiere el runtime ASan en el PATH en runtime**
  (`clang_rt.asan_dynamic-x86_64.dll`, en `…\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\`); sin él, el `.exe`
  falla con `STATUS_DLL_NOT_FOUND` (0xc0000135) — no es un crash del target.
- ✅ **Smoke 60 s local, cero crashes:** `fuzz_headless_input` → **346 653 runs / 61 s**;
  `fuzz_sanitize_error` → **267 007 runs / 61 s** (la idempotencia del redactor no falló en ~267 k
  entradas adversariales). La **corrida larga coverage-guided (≥ 30 min/target)** queda para CI/§0.3,
  fuera del loop RGR y del presupuesto §7.

**Miri (REQ-V38) — INFEASIBLE en el nightly actual (regresión de toolchain, NO un hallazgo de UB):**
- ❌ `cargo +nightly miri test headless::{input,output,exit}` **aborta con un ICE de rustc**
  (`resolver_for_lowering_raw` panickea en la fase de lowering, **antes** de correr cualquier test) en
  `rustc 1.99.0-nightly (da80ed070 2026-07-14)`. El ICE ocurre al compilar el crate bajo Miri, no al
  ejecutar código headless.
- ✅ **Verificado que es toolchain, no código:** `cargo +nightly miri test vault::error` — que corría
  **limpio** bajo Miri en un nightly anterior (spike Task 0b) — **ICEa idéntico** en este nightly. La
  causa es el compilador, no los módulos headless.
- **Mitigación de robustez sin Miri, honesta:** (a) el crate es `#![forbid(unsafe_code)]`
  crate-wide ⇒ no hay `unsafe` donde alojar UB (un pase Miri sería trivial por construcción); (b) los
  2 targets de fuzz (build + smoke, cero crashes) ejercitan el parser no confiable y el redactor; (c)
  los unit-smoke corren en cada §0.1. **No se declara un pase de Miri que no ocurrió.** Re-habilitar
  Miri requiere un nightly sin el ICE (o pinnear uno previo conocido-bueno).
