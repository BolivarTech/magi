// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-23

# Spec-Behavior Base — Fase 0: Remediacion del audit de seguridad

> Documento base pre-brainstorming (input a `/brainstorming`). Captura objetivo,
> requerimientos (SDD), escenarios Given/When/Then (BDD), restricciones y
> no-goals de la **Fase 0** del roadmap (`docs/STRATEGY_2026-05-16.md` §4).
> Fuente de los hallazgos: `docs/AUDIT_2026-05-16.md`.

---

## 1. Objetivo

Cerrar los hallazgos bloqueantes del audit del 2026-05-16 para que **ningun
hueco de seguridad activo, panic alcanzable, ni corrupcion silenciosa de datos**
sobreviva antes de empezar cualquier feature nuevo (Fase 1+). Al terminar Fase 0
el codigo debe pasar `§0.1` completo en verde y una re-corrida del audit debe
reportar **0 CRITICAL** entre los items de alcance.

Fase 0 tiene valor por si sola: aunque el roadmap unificado no continue, deja el
binario actual seguro y verificable.

## 2. Contexto y trazabilidad

- El audit identifico 8 CRITICAL, 27 WARNING, 15+ INFO.
- El alcance de Fase 0 fue acordado como el **Top-7** del audit mas extras de
  seguridad de bajo costo/alto valor (decision 2026-05-23).
- La mayoria de los fixes **conectan infraestructura que ya existe** (PathGuard
  esta completo pero `#[allow(dead_code)]`; el streaming esta construido pero
  desconectado).

## 3. Decisiones tomadas (2026-05-23)

| # | Decision | Resolucion | Implicancia |
|---|----------|-----------|-------------|
| D1 | Tests rotos preexistentes (provider SSE mocks) | **En alcance** — se arreglan primero | Habilita el gate `§0.1` (sin esto, ninguna fase TDD cierra en verde) |
| D2 | Migracion de DBs cifradas tras cambios de cripto (C5/C6/C7) | **Fresh-start** (precedente W27) | DBs `.magi-rs-memory.db` pre-cambio arrancan de cero; **cero codigo de migracion**; se documenta en `CLAUDE.md` |
| D3 | Alcance de Fase 0 | **Top-7 + extras de seguridad** (W1, W4, W8) | Ver §4 |

## 4. Alcance (work units)

Cada work unit (WU) mapea a uno o mas hallazgos del audit y sera una o mas
tareas TDD en el plan.

| WU | Hallazgos | Resumen | Esfuerzo audit |
|----|-----------|---------|----------------|
| WU-0 | INFO (tests) | Arreglar 2 tests SSE rotos en `provider.rs` | — (prerequisito) |
| WU-1 | C1, C2 | Sandbox: `PathGuard::validate` en `write.rs` y `grep.rs` | XS |
| WU-2 | C3 | `cargo` allowlist sin panic por index OOB | XS |
| WU-3 | C4 | Eliminar fallback `"emergency-key"` en `main.rs` | XS |
| WU-4 | C8 | Timeout en el servidor de callback OAuth | XS |
| WU-5 | W11, W12 | Mutex poison → `?`; `vault.decrypt` fuera del lock | S |
| WU-6 | W21 | Conectar `query_streaming` al `response_tx` del TUI | M |
| WU-7 | C5, C6, C7 | Cripto: nonce independiente + Argon2 OWASP + cap de length prefix (fresh-start) | M |
| WU-8 | W1, W4, W8 | Extras: cap del buffer SSE; `--%` a banned tokens; zeroize `master_password` | S |

**Fuera de alcance (diferidos a fases posteriores):** W2, W3, W5, W6, W7, W9,
W10, W13, W14, W15, W16, W17, W18, W19, W20, W22, W23, W25 y todos los INFO de
pulido que no bloqueen el gate `§0.1`. No se tocan salvo que sean prerequisito
mecanico de un fix en alcance.

## 5. Requerimientos (SDD)

### WU-0 — Reparar tests SSE rotos
- **RF-0.1**: `test_anthropic_provider_simple_response` y
  `test_anthropic_provider_tool_use` deben emitir fixtures **SSE**
  (`event: message_start\ndata: ...\n\n`, `content_block_delta`, `message_stop`)
  en vez de un body JSON unico, alineados con el parser real de `send_messages`.
- **RF-0.2**: Tras el fix, `cargo nextest run` reporta **0 fails**.

### WU-1 — Sandbox de escritura y grep (C1, C2)
- **RF-1.1**: `FileWriteTool` (`write.rs`) debe validar el path destino con
  `PathGuard::validate` antes de escribir; rechazar absolutos, `..`, prefijos
  verbatim/UNC de Windows (`\\?\`, `\\server\share`), y null-bytes.
- **RF-1.2**: `GrepTool` (`grep.rs`) debe canonicalizar + `.starts_with(workspace_root)`
  (o `PathGuard::validate`), rechazando traversal por symlink.
- **RF-1.3**: Remover `#[allow(dead_code)]` de `PathGuard` una vez en uso.

### WU-2 — `cargo` allowlist sin panic (C3)
- **RF-2.1**: `is_command_allowed("cargo")` sin subcomando debe retornar `false`
  sin indexar un slice vacio (sin panic).
- **RF-2.2**: Solo `cargo test|build|check` permitidos; cualquier otro subcomando
  o ausencia de subcomando → `false`.
- **RF-2.3**: Documentar el riesgo de `build.rs` (RCE indirecto en compile-time)
  en `CLAUDE.md` o restringir a `cargo test --no-run`.

### WU-3 — Master key sin fallback constante (C4)
- **RF-3.1**: `discover_or_create_master_key` debe propagar su error con `?`;
  **prohibido** usar una passphrase literal (`"emergency-key"`) como fallback.
- **RF-3.2**: Si el keyring es inaccesible, el binario falla con mensaje claro al
  usuario; nunca cifra ni intenta descifrar con clave global.

### WU-4 — Timeout OAuth (C8)
- **RF-4.1**: El servidor de callback OAuth debe envolverse en
  `tokio::time::timeout` (10 min); al expirar retorna error y libera el puerto.
- **RF-4.2**: La task del runner que ejecuta `UiEvent::Login` debe retornar
  siempre (no bloquear `Quit` ni eventos siguientes).

### WU-5 — Concurrencia DB (W11, W12)
- **RF-5.1**: Todo `self.conn.lock().unwrap()` en `database.rs` → propagacion de
  error (`.map_err(...)? ` / `anyhow`); un Mutex poisoned no panica la runtime.
- **RF-5.2**: `get_messages` debe coleccionar blobs, **soltar el lock**, y
  descifrar (`vault.decrypt`) fuera de la seccion critica.

### WU-6 — Streaming conectado (W21)
- **RF-6.1**: El TUI debe usar `query_streaming` (no el wrapper `query`
  no-stream); el `chunk_tx` debe conectarse al `response_tx` del TUI.
- **RF-6.2**: El usuario ve la respuesta del modelo **incrementalmente** (deltas),
  no solo al finalizar.
- **RF-6.3**: La manipulacion de texto resultante respeta UTF-8 boundary safety
  (`char_indices`/`is_char_boundary`).

### WU-7 — Hardening de cripto (C5, C6, C7) — fresh-start
- **RF-7.1**: El nonce de AES-256-GCM-SIV se deriva **independiente** de la key
  (32 bytes Argon2 para key + nonce via `OsRng`), almacenado en el blob layout
  antes del ciphertext.
- **RF-7.2**: Argon2 con parametros explicitos OWASP 2025
  (`m=65536, t=3, p=4`) como constante nombrada con doc-comment.
- **RF-7.3**: El length prefix del blob tiene tope superior
  (`MAX_PLAINTEXT_LEN = 50 MiB`); un blob malformado retorna `CryptoError`
  sin `Vec::with_capacity` arbitrario.
- **RF-7.4**: Documentar en `CLAUDE.md` que las DBs pre-cambio no migran
  (fresh-start, D2).

### WU-8 — Extras de seguridad (W1, W4, W8)
- **RF-8.1**: El buffer SSE (`provider.rs`) tiene cap (8 MB); un stream sin
  `\n\n` aborta en vez de crecer sin limite (OOM).
- **RF-8.2**: El token PowerShell `--%` (stop-parsing) se agrega a
  `dangerous_tokens` del bash tool.
- **RF-8.3**: `master_password` en `database.rs` usa `Zeroizing<String>` (no
  sobrevive en heap liberado).

## 6. Escenarios (BDD)

Los nombres de los tests describen comportamiento, no implementacion.

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
Cuando se valida el comando "cargo" sin subcomando
Entonces retorna false sin panic
  Y "cargo run" tambien retorna false
  Y "cargo test" retorna true
```

### S-4 (C4)
```
Dado un keyring inaccesible
Cuando el binario arranca
Entonces falla con un mensaje claro al usuario
  Y no usa ninguna passphrase constante
  Y no corrompe ni recifra la DB existente
```

### S-5 (C8)
```
Dado un flujo OAuth iniciado y luego abandonado por el usuario
Cuando transcurren 10 minutos
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
Dado dos cifrados del mismo plaintext con la misma key derivada cacheada
Entonces los nonces difieren (no hay reuse)

Dado un blob con length-prefix = 0xFFFFFFFF
Cuando se intenta descifrar
Entonces retorna CryptoError sin asignar memoria arbitraria
```

### S-9 (W1/W4, adversarial)
```
Dado un stream SSE sin separador "\n\n" que excede 8 MB
Entonces el stream aborta con error en vez de OOM

Dado un comando bash que contiene "--%"
Entonces es rechazado por token peligroso
```

## 7. Restricciones (invariantes que deben sobrevivir)

- **R-1**: No fusionar los keyrings `magi-rs` y `magi-rs-internal`. El test
  `test_agent_history_resilience_to_key_rotation` debe seguir verde (WU-3/WU-7
  tocan cripto y main.rs — no romper la separacion).
- **R-2**: No ampliar `dangerous_tokens` para permitir `$` ni backticks. WU-8
  solo **agrega** `--%`.
- **R-3**: La serializacion de `messages.role` (`"User"`/`"Assistant"`) no cambia.
- **R-4**: UTF-8 boundary safety en todo cambio de TUI (WU-6).
- **R-5**: La migracion legacy keyring (`magi-rust*` → `magi-rs*`) dual-read en
  `main.rs` permanece intacta (WU-3 toca `main.rs`).
- **R-6**: Toda tool que toque el FS usa `PathGuard::validate` (estandar §0.2).
- **R-7**: Errores de dominio con `thiserror` (`CryptoError`, `ToolError`,
  `FsError`); bordes de aplicacion con `anyhow` + `?`. Sin silent failures.
- **R-8**: Headers `// Author / Version / Date` en archivos nuevos.

## 8. Lo que NO debe hacer (no-goals)

- **NO** agregar features nuevos — Fase 0 es solo remediacion.
- **NO** escribir codigo de migracion de DB (D2 fresh-start).
- **NO** tocar los WARNING/INFO fuera de alcance (§4) salvo prerequisito mecanico.
- **NO** ampliar el allowlist de binarios del bash tool.
- **NO** mezclar fases TDD ni hallazgos distintos en un mismo commit (§5 git).
- **NO** refactorizar codigo no relacionado al hallazgo en curso (evitar scope creep).

## 9. Criterios de aceptacion (definition of done)

- [ ] `§0.1` completo en verde: `cargo nextest run` (0 fail), `cargo clippy
      --tests -- -D warnings` (0 warn), `cargo fmt --check`, `cargo build
      --release` (sin warnings), `cargo doc --no-deps`, `cargo audit`.
- [ ] Los 2 tests SSE preexistentes pasan (WU-0).
- [ ] Cada fix en alcance tiene un test adversarial (Red prueba el hueco, Green
      lo cierra) — modelo `test_adversarial_bash_injections` / `test_path_validation`.
- [ ] Una re-corrida del audit reporta **0 CRITICAL** entre C1-C8 y los WARNING
      en alcance (W11, W12, W21, W1, W4, W8) cerrados.
- [ ] `PathGuard` y `query_streaming` ya no estan huerfanos (`#[allow(dead_code)]`
      removido donde aplique).
- [ ] `CLAUDE.md` actualizado: decision fresh-start (D2) y riesgo `build.rs` (RF-2.3).
- [ ] Commits atomicos con prefijos correctos (§5).

## 10. Riesgos / cuestiones abiertas para `/brainstorming`

- **Q1 (gate clippy)**: `§0.1` exige `clippy -D warnings = 0`, pero hay ~15
  warnings preexistentes. Algunos se resuelven al usar PathGuard (WU-1) y
  streaming (WU-6); los residuales (p.ej. `git.rs` scaffold dead_code W25,
  `f.size()` deprecado en TUI) **bloquean el gate**. Decidir en brainstorming:
  feature-gate/`#[allow]` documentado del scaffold vs. incluir su limpieza como
  WU adicional forzada por el gate.
- **Q2 (orden de WU)**: confirmar dependencias entre WU para el plan multi-agente
  (p.ej. WU-7 cripto y WU-5 DB tocan capas adyacentes; WU-0 es prerequisito de
  todo gate). Marcar `addBlockedBy` donde corresponda.
- **Q3 (granularidad de C5/C6/C7)**: el audit sugiere "un solo PR" pero TDD
  estricto puede pedir separar nonce (C5) de params Argon2 (C6) de length cap
  (C7) en tareas distintas. Definir en el plan.
