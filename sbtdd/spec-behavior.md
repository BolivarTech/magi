// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-23

# Spec-Behavior — Fase 0: Remediacion del audit de seguridad

> Especificacion SDD + BDD del feature activo. Output de `/brainstorming` sobre
> `sbtdd/spec-behavior-base.md`. Input a `/writing-plans`.
> Fuente de hallazgos: `docs/AUDIT_2026-05-16.md`. Roadmap: `docs/STRATEGY_2026-05-16.md` §4.

---

## 1. Objetivo

Cerrar los hallazgos bloqueantes del audit del 2026-05-16 para que **ningun
hueco de seguridad activo, panic alcanzable, ni corrupcion silenciosa de datos**
sobreviva antes de iniciar features nuevos (Fase 1+). Al terminar Fase 0:

- `§0.1` (CLAUDE.local.md) pasa completo en verde.
- Una re-corrida del audit reporta **0 CRITICAL** entre los items de alcance y
  los WARNING en alcance cerrados.
- El arbol queda **clippy-clean** (0 warnings) y `nextest` 100% verde.

Fase 0 tiene valor por si sola: aunque el roadmap unificado no continue, deja el
binario actual seguro y verificable.

## 2. Contexto y trazabilidad

- El audit identifico 8 CRITICAL, 27 WARNING, 15+ INFO.
- La mayoria de los fixes **conectan infraestructura que ya existe**: `PathGuard`
  esta completo y testeado pero `#[allow(dead_code)]`; el streaming
  (`query_streaming`) esta construido pero el TUI usa el wrapper `query`.
- Verificacion de codigo (2026-05-23) confirmo: `PathGuard::validate(&self,
  &Path) -> Result<PathBuf>` maneja relativos/absolutos/traversal/null-byte/
  verbatim Windows; `Agent.memory` es `Option<Arc<dyn MemoryStore>>` (el agente
  ya corre sin persistencia); `query_streaming(text, chunk_tx)` es el metodo de
  streaming real.

## 3. Decisiones de diseno (2026-05-23)

| # | Decision | Resolucion |
|---|----------|-----------|
| D1 | Tests rotos preexistentes (provider SSE mocks) | **En alcance** (WU-0), se arreglan primero |
| D2 | Migracion de DBs cifradas tras cambios de cripto | **Fresh-start** (precedente W27): DBs pre-cambio arrancan de cero, **cero codigo de migracion**, se documenta en `CLAUDE.md` |
| D3 | Alcance de Fase 0 | **Top-7 + extras de seguridad** (W1, W4, W8) |
| D4 | Comportamiento ante keyring inaccesible (C4) | **Graceful degrade a efimero**: si no hay master key, no adjuntar memoria + warning ruidoso; nunca usar constante; DB existente intacta |
| D5 | Gate `clippy -D warnings = 0` (Q1) | **Cleanup completo** de todos los warnings de clippy (WU-9), no solo los que bloquean el gate |

## 4. Alcance y secuenciacion

### 4.1 Work units

| WU | Hallazgos | Resumen |
|----|-----------|---------|
| WU-0 | INFO (tests) | Arreglar 2 tests SSE rotos en `provider.rs` |
| WU-9 | INFO (calidad) | Resolver los 26 warnings de clippy del arbol completo |
| WU-1 | C1, C2 | Sandbox: `PathGuard::validate` en `write.rs` y `grep.rs` |
| WU-2 | C3 | `cargo` allowlist sin panic por index OOB |
| WU-3 | C4 | Eliminar fallback `"emergency-key"` → graceful degrade (D4) |
| WU-4 | C8 | Timeout en el servidor de callback OAuth |
| WU-5 | W11, W12 | Mutex poison → `?`; `vault.decrypt` fuera del lock |
| WU-6 | W21 | Conectar `query_streaming` al `response_tx` del TUI |
| WU-7 | C5, C6, C7 | Cripto: nonce independiente + Argon2 OWASP + cap de length prefix (fresh-start) |
| WU-8 | W1, W4, W8 | Extras: cap del buffer SSE; `--%` a banned tokens; zeroize `master_password` |

### 4.2 Restriccion de secuenciacion — "baseline verde" (CRITICA)

`§0.1` se ejecuta **al cierre de cada fase TDD** (incluido Red: el codigo debe
compilar clippy-clean y solo el test nuevo falla). Con `nextest` en rojo (2
tests) y clippy en rojo (26 warnings), la fase Red de la primera tarea de
seguridad **ya falla la verificacion**. Por lo tanto el orden es obligatorio:

```
Fase 0.0 — Baseline verde (prerequisito; bloquea todo lo demas)
   1. WU-0  (arreglar 2 tests SSE)   → nextest = 0 fail
   2. WU-9  (limpiar 26 warnings)    → clippy  = 0 warn
      WU-0 va ANTES de WU-9: arreglar los tests vuelve a usar
      `send_messages` (hoy "never used"), evitando borrar codigo
      que esta por usarse.

Fase 0.1 — Remediacion de seguridad (sobre arbol limpio)
   WU-1 … WU-8  en orden de dependencias (§4.4)
```

### 4.3 Inventario de warnings (WU-9)

clippy reporta 26 warnings en el bin target (+1 `sid` solo en el target de
tests). El gate exige **0**, sin importar el conteo exacto. Inventario por
categoria (lista `archivo:linea` exhaustiva, no subtotales):

- **Imports sin uso:** `tools/knowledge.rs:3` (`anyhow::Result`),
  `services/oauth.rs:4` (`Serialize`), `agent/provider.rs:11` (`mockall::automock`).
- **Dead code:** `provider.rs:21` (variant `ToolUseStart`), `provider.rs:35`
  (`send_messages` — se resuelve solo tras WU-0), `provider.rs:119`
  (`AnthropicResponse`), `provider.rs:141,142,143,144,147,152` (campos never
  read), `agent/mod.rs:20` (field `input`), `agent/mod.rs:74` (`send_info` — solo
  este; `compact_history` no existe en `:74`, corregido tras MAGI Checkpoint 2),
  `tui/mod.rs:194` (`run_tui`), `agent/mod.rs:368` (`sid`, solo tests).
- **Lints de estilo:** `provider.rs:45` (var sin `mut`), `provider.rs:256` y
  `path_guard.rs:118` (`manual_strip` → `strip_prefix`), `agent/mod.rs:122` y
  `tui/mod.rs:105,125,344,357` (`needless_range_loop` → `for`), `bash.rs:115`
  (`collapsible_if` — coincide con el bloque de C3/WU-2), `agent/mod.rs:216`
  (`redundant_pattern_matching` → `is_err()`), `tui/mod.rs:488` (`vec!` macro),
  `main.rs:66` (`get(0)` → `first()`).

**Solapamiento con WU de seguridad:** varios warnings viven en archivos que
WU-1..8 tocaran (`provider.rs`, `tui/mod.rs`, `main.rs`, `bash.rs`,
`path_guard.rs`, `oauth.rs`). WU-9 corre primero y deja el arbol limpio; cada WU
posterior **debe mantener clippy en 0** (no introducir regresiones). Cuando un WU
reescribe un bloque ya limpiado (p.ej. WU-2 reescribe `bash.rs:115`), conserva la
limpieza.

**Riesgo de dead-code:** WU-9 solo elimina codigo **genuinamente** muerto y
resuelve los warnings de campos-no-leidos segun la sugerencia de clippy
(remover campo / `#[allow]` justificado), **sin alterar el happy-path SSE**
(cubierto por los tests de WU-0). El struct de respuesta no-streaming
(`AnthropicResponse`) es candidato a remocion por estar fuera del path real
(SSE), a confirmar en el plan. Si un WU posterior (WU-6/WU-8) necesita un tipo
nuevo, lo agrega en su fase Green con test propio.

### 4.4 Dependencias entre WU (Q2)

- **WU-0 → WU-9**: WU-0 primero (deja `send_messages` usado).
- **WU-9 → {WU-1..8}**: baseline verde antes de seguridad.
- **WU-7 (cripto) ↔ WU-5 (DB)**: capas adyacentes (`database.rs` usa
  `CryptoVault`). WU-5 refactoriza `get_messages`; WU-7 cambia el layout del
  blob. Secuenciar (WU-5 antes que WU-7, o `addBlockedBy`) para evitar conflicto.
- **WU-6 (streaming) ↔ WU-8/W1 (cap buffer SSE)**: ambos tocan `provider.rs` +
  `tui`. Secuenciar WU-6 antes que WU-8 o coordinar el merge.
- El resto (WU-1, WU-2, WU-3, WU-4) son independientes entre si.

### 4.5 Fuera de alcance (diferidos)

W2, W3, W5, W6, W7, W9, W10, W13, W14, W15, W16, W17, W18, W19, W20, W22, W23,
W25, y el resto de INFO de pulido que no sean warnings de clippy. No se tocan
salvo prerequisito mecanico de un fix en alcance.

## 5. Requerimientos (SDD)

### WU-0 — Reparar tests SSE rotos
- **RF-0.1**: `test_anthropic_provider_simple_response` y
  `test_anthropic_provider_tool_use` emiten fixtures **SSE** (`event:
  message_start\ndata: ...\n\n`, `content_block_delta`, `message_stop`),
  alineados con el parser real de `send_messages`/`stream_messages`.
- **RF-0.2**: Tras el fix, `cargo nextest run` reporta **0 fails**.

### WU-9 — Cleanup completo de warnings
- **RF-9.1**: `cargo clippy --all-targets -- -D warnings` retorna sin warnings.
- **RF-9.2**: El cleanup **no cambia comportamiento runtime**: todos los tests
  existentes siguen verdes; no se eliminan rutas de codigo en uso.
- **RF-9.3**: Dead code genuino se elimina; lints de estilo se aplican segun la
  sugerencia de clippy.

### WU-1 — Sandbox de escritura y grep (C1, C2)
- **RF-1.1**: `FileWriteTool` (`write.rs`) valida el path con `PathGuard::validate`
  antes de escribir; rechaza absolutos, `..`, verbatim/UNC Windows (`\\?\`,
  `\\server\share`) y null-bytes. Usa el `PathBuf` canonicalizado que retorna.
- **RF-1.2**: `GrepTool` (`grep.rs`) valida con `PathGuard::validate` (o
  canonicalize + `.starts_with(workspace_root)`), rechazando traversal por symlink.
- **RF-1.3**: Se remueve `#[allow(dead_code)]` de `PathGuard` una vez en uso.

### WU-2 — `cargo` allowlist sin panic (C3)
- **RF-2.1**: `is_command_allowed("cargo")` sin subcomando retorna `false` sin
  indexar un slice vacio (sin panic).
- **RF-2.2**: Solo `cargo test|build|check` permitidos; cualquier otro subcomando
  o ausencia → `false`.
- **RF-2.3**: Documentar el riesgo de `build.rs` (RCE indirecto en compile-time)
  en `CLAUDE.md`.

### WU-3 — Master key sin fallback constante, graceful degrade (C4)
- **RF-3.1**: Se elimina el fallback literal `"emergency-key"`. **Prohibida**
  cualquier passphrase constante.
- **RF-3.2**: Si `discover_or_create_master_key()` retorna `Err`, `main.rs`
  emite un warning visible al usuario y **no adjunta** la memoria cifrada
  (`Agent` corre con `memory = None`, sesion efimera).
- **RF-3.3**: La DB existente nunca se recifra ni se corrompe ante un fallo de
  keyring.
- **RF-3.4**: La separacion de keyrings (`magi-rs` vs `magi-rs-internal`) y la
  migracion legacy dual-read en `main.rs` permanecen intactas.

### WU-4 — Timeout OAuth (C8)
- **RF-4.1**: El servidor de callback OAuth se envuelve en `tokio::time::timeout`
  (600 s); al expirar retorna error y libera el puerto 54545.
- **RF-4.2**: La task del runner que ejecuta `UiEvent::Login` retorna siempre
  (no bloquea `Quit` ni eventos siguientes).

### WU-5 — Concurrencia DB (W11, W12)
- **RF-5.1**: Todo `self.conn.lock().unwrap()` en `database.rs` propaga el error
  (`.map_err(...)?`); un Mutex poisoned **no** panica la runtime.
- **RF-5.2**: `get_messages` colecciona los blobs, **suelta el lock**, y descifra
  (`vault.decrypt`) fuera de la seccion critica.

### WU-6 — Streaming conectado (W21)
- **RF-6.1**: El TUI usa `query_streaming`; el `chunk_tx` se conecta al
  `response_tx` del TUI. El wrapper `query` se elimina o deja de usarse.
- **RF-6.2**: El usuario ve la respuesta del modelo **incrementalmente** (deltas)
  antes de `MessageDone`.
- **RF-6.3**: Toda manipulacion de texto resultante respeta UTF-8 boundary safety
  (`char_indices`/`is_char_boundary`).

### WU-7 — Hardening de cripto (C5, C6, C7) — fresh-start
- **RF-7.1**: El nonce de AES-256-GCM-SIV se deriva **independiente** de la key
  (Argon2 produce 32 bytes de key; el nonce viene de `OsRng`), almacenado en el
  blob antes del ciphertext.
- **RF-7.2**: Argon2 con parametros explicitos OWASP 2025 (`m=65536, t=3, p=4`)
  como constante nombrada con doc-comment.
- **RF-7.3**: El length prefix del blob tiene tope (`MAX_PLAINTEXT_LEN = 50 MiB`);
  un blob malformado retorna `CryptoError` sin `Vec::with_capacity` arbitrario.
- **RF-7.4**: `CLAUDE.md` documenta que las DBs pre-cambio no migran (D2).

### WU-8 — Extras de seguridad (W1, W4, W8)
- **RF-8.1**: El buffer SSE (`provider.rs`) tiene cap (8 MiB); un stream sin
  `\n\n` aborta con error en vez de crecer sin limite (OOM).
- **RF-8.2**: El token PowerShell `--%` se agrega a `dangerous_tokens` del bash
  tool.
- **RF-8.3**: `master_password` en `database.rs` usa `Zeroizing<String>`.

## 6. Escenarios (BDD)

Nombres de tests describen comportamiento, no implementacion (p.ej.
`test_write_rejects_absolute_path_outside_workspace`).

### S-0 (WU-0)
```
Dado el mock HTTP del provider Anthropic emitiendo eventos SSE validos
Cuando send_messages procesa la respuesta
Entonces el Message ensamblado contiene el texto esperado
  Y el test pasa (no "Stream ended without MessageDone")
```

### S-9 (WU-9)
```
Dado el arbol de codigo tras WU-0
Cuando se corre cargo clippy --all-targets -- -D warnings
Entonces no hay warnings
  Y cargo nextest run sigue 100% verde (comportamiento intacto)
```

### S-1 (C1, adversarial)
```
Dado un FileWriteTool con workspace_root = D:\proj
Cuando se invoca con file_path = "C:\\Windows\\System32\\evil.dll"
Entonces la escritura es rechazada con error de sandbox
  Y ningun archivo se crea fuera de D:\proj
```

### S-2 (C2, adversarial)
```
Dado un GrepTool con un symlink workspace\link -> C:\secret
Cuando se busca con path = "link"
Entonces la busqueda es rechazada por estar fuera del workspace
```

### S-3 (C3, adversarial)
```
Dado el allowlist del bash tool
Cuando se valida "cargo" sin subcomando
Entonces retorna false sin panic
  Y "cargo run" retorna false
  Y "cargo test" retorna true
```

### S-4 (C4, graceful degrade)
```
Dado un keyring inaccesible (discover_or_create_master_key retorna Err)
Cuando el binario arranca
Entonces emite un warning visible al usuario
  Y corre con memory = None (sesion efimera, sin persistencia)
  Y no usa ninguna passphrase constante
  Y no recifra ni corrompe la DB existente
```

### S-5 (C8)
```
Dado un flujo OAuth iniciado y luego abandonado
Cuando transcurren 600 segundos
Entonces el servidor de callback expira y libera el puerto
  Y la task del runner retorna un error en vez de bloquear Quit
```

### S-6 (W11)
```
Dado un Mutex de conexion DB envenenado por un panic previo
Cuando se llama un metodo de database.rs
Entonces retorna Err (no panica la runtime)
```

### S-7 (W21)
```
Dado un query del usuario en el TUI
Cuando el provider emite TextDelta incrementales
Entonces el usuario ve la respuesta crecer en pantalla antes de MessageDone
```

### S-8 (C5/C7, adversarial)
```
Dado dos cifrados del mismo plaintext con la misma key derivada (cacheada)
Entonces los nonces difieren (no hay reuse)

Dado un blob con length-prefix = 0xFFFFFFFF
Cuando se intenta descifrar
Entonces retorna CryptoError sin asignar memoria arbitraria
```

### S-8b (W1/W4, adversarial)
```
Dado un stream SSE sin separador "\n\n" que excede 8 MiB
Entonces el stream aborta con error en vez de OOM

Dado un comando bash que contiene "--%"
Entonces es rechazado por token peligroso
```

## 7. Restricciones (invariantes que deben sobrevivir)

- **R-1**: No fusionar los keyrings `magi-rs` y `magi-rs-internal`. El test
  `test_agent_history_resilience_to_key_rotation` sigue verde (WU-3/WU-7 tocan
  cripto y `main.rs`).
- **R-2**: No ampliar `dangerous_tokens` para permitir `$` ni backticks. WU-8
  solo **agrega** `--%`.
- **R-3**: La serializacion de `messages.role` (`"User"`/`"Assistant"`) no cambia
  (sin migracion de tabla `messages`).
- **R-4**: UTF-8 boundary safety en todo cambio de TUI (WU-6).
- **R-5**: La migracion legacy keyring (`magi-rust*` → `magi-rs*`) dual-read en
  `main.rs` permanece intacta (WU-3).
- **R-6**: Toda tool que toque el FS usa `PathGuard::validate` (estandar §0.2).
- **R-7**: Errores de dominio con `thiserror` (`CryptoError`, `ToolError`,
  `FsError`); bordes de aplicacion con `anyhow` + `?`. Sin silent failures.
- **R-8**: Headers `// Author / Version / Date` en archivos nuevos.

## 8. Lo que NO debe hacer (no-goals)

- **NO** agregar features nuevos — Fase 0 es remediacion + cleanup de warnings.
- **NO** escribir codigo de migracion de DB (D2 fresh-start).
- **NO** introducir flags CLI nuevos (p.ej. `--no-memory` es de Fase 2).
- **NO** tocar los WARNING/INFO fuera de alcance (§4.5) salvo que sean un warning
  de clippy (esos entran en WU-9) o prerequisito mecanico de un fix.
- **NO** ampliar el allowlist de binarios del bash tool.
- **NO** mezclar fases TDD ni hallazgos distintos en un mismo commit (§5 git).
- **NO** refactorizar logica no relacionada (WU-9 se limita a lo que clippy marca).

## 9. Criterios de aceptacion (definition of done)

- [ ] `§0.1` completo en verde: `cargo nextest run` (0 fail), `cargo clippy
      --all-targets -- -D warnings` (0 warn), `cargo fmt --check`, `cargo build
      --release` (sin warnings), `cargo doc --no-deps` (sin warnings),
      `cargo audit` (sin vulnerabilidades).
- [ ] Los 2 tests SSE preexistentes pasan (WU-0).
- [ ] 0 warnings de clippy en todo el arbol (WU-9).
- [ ] Cada fix de seguridad tiene un test adversarial (Red prueba el hueco, Green
      lo cierra) — modelo `test_adversarial_bash_injections` / `test_path_validation`.
- [ ] Re-corrida del audit: **0 CRITICAL** (C1-C8) y WARNING en alcance (W1, W4,
      W8, W11, W12, W21) cerrados.
- [ ] `PathGuard` y `query_streaming` ya no estan huerfanos.
- [ ] C4 degrada graceful a efimero ante keyring inaccesible (S-4 verde).
- [ ] `CLAUDE.md` actualizado: fresh-start (D2) y riesgo `build.rs` (RF-2.3).
- [ ] Commits atomicos con prefijos correctos (§5).

## 10. Cuestiones diferidas a `/writing-plans`

- **Q3 (granularidad C5/C6/C7)**: el audit sugiere "un solo PR", pero TDD
  estricto puede pedir separar nonce (C5), params Argon2 (C6) y length cap (C7)
  en tareas distintas. El plan define la granularidad y el orden de fases.
- **Mapeo TDD de WU-9**: el cleanup de warnings no tiene un test de comportamiento
  natural; su "test" es el gate `clippy -D warnings`. El plan decide si se modela
  como tarea refactor con el gate como criterio, o se subdivide por archivo.
- **Marcado `addBlockedBy`**: el plan formaliza las dependencias de §4.4 para el
  fan-out multi-agente (evitar paralelizar WU con archivos compartidos —
  ver CLAUDE.local.md §3 TDD-Guard bajo paralelismo).
