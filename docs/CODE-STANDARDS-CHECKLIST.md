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
| Documentación | `cargo doc --no-deps` | Rustdoc compila sin warnings; `missing_docs` es
  `deny` dentro de `src/vault/`. |
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
