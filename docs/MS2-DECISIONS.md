# MS2 — Decisiones verificadas durante la implementación

> Registro de lo que se **comprobó contra el código**, no de lo que se decidió en la spec o en
> el plan. Cada entrada lleva fecha porque una actualización de magi-core puede moverla, y
> porque una verificación sin fecha es indistinguible de una suposición vieja.
>
> Artefactos de proceso: `sbtdd/spec-behavior.md` (el *qué*) · `planning/claude-plan-tdd.md`
> (el *cómo*, aprobado en el Checkpoint 2 del 2026-08-02).

---

## Superficie de API verificada — magi-core 3.1.0

**Fecha de verificación: 2026-08-02.** Guardián permanente: `tests/magi_core_contract.rs`.

### Cómo se verificó

Se escribió un test que **usa** cada símbolo que las Fases 4 a 6 consumen, con **anotaciones
de tipo** en vez de bindings sueltos, y se lo compiló. Un binding suelto (`let _ = &r.campo`)
prueba que el campo existe y nada más; la telemetría de la Fase 6 **itera** esas estructuras,
así que un cambio de forma —`Vec` a `BTreeMap`, `T` a `Option<T>`— pasaría el binding y
rompería la fase entera. Existencia y forma son dos verificaciones distintas.

**Resultado: compila y pasa sin una sola corrección.** La superficie que el plan asume coincide
con el crate real. Eso **no** vuelve inútil al spike — vuelve verificado lo que hasta acá era
una lectura con fecha de vencimiento.

### Las cinco suposiciones que resultaron FALSAS

Encontradas leyendo el crate el 2026-08-01, ya corregidas en el plan y ahora **fijadas por el
compilador**. Se listan porque el patrón importa más que los casos: las cinco eran plausibles,
ninguna la encontró un loop de revisión, y las cinco habrían aparecido en Fase 4 con tres fases
construidas encima.

| # | El plan asumía | La realidad |
|---|---|---|
| 1 | `with_client(...)` para inyectar un `reqwest::Client` propio | **No existe en ningún provider.** El camino real es el constructor `with_timeout`. |
| 2 | `OllamaProvider` sirve para las completions | Su **único** constructor fija un cliente de **300 s sin override**, lo que hace imposible `operation_budget + client_timeout <= techo` (REQ-A04). Queda **solo como sonda** — decisión D-A07. |
| 3 | `RetryConfig { .. }` o `..default()` | Es `#[non_exhaustive]`: fuera del crate **no compila** ni el literal ni el update funcional. El patrón obligado es `default()` mutable. |
| 4 | `ClaudeProvider::new(model, api_key)` | **`api_key` va primero.** Los dos son `impl Into<String>`, así que invertirlos **compila** y falla en runtime con un 401. |
| 5 | `Mode` se parsea con `FromStr`/`from_str` | No hay ninguno. Solo `Display` + serde en kebab-case — por eso MS2 necesita su propio `ModeExt::parse_config_value` (Task 1.0). |

### Una capacidad AUSENTE, y qué se hizo con ella

`MagiReport::window_rejected` **no existe**. Se verificó campo por campo: vive en `rotation.rs`
como estado del registro de rotación, que es MS3. El sustituto es **`failed_agents`**
(`BTreeMap<AgentName, String>`), sobre el que se replantearon REQ-A11d y SC-A11g.

La regla que esto sienta: **una capacidad ausente no se inventa.** Se registra como ausente y
el requerimiento que dependía de ella se replantea con lo que sí hay. Un plan que llama a algo
inexistente no es un plan.

### Formas confirmadas, con su consecuencia de diseño

- **`ProviderProbe` lleva `#[async_trait::async_trait]`** ⇒ es dyn-compatible, y por eso la
  costura de inyección de Task 5.1 puede ser `Arc<dyn ProviderProbe>`. Con `async fn` nativo
  en el trait no lo sería, y la fábrica entera se replantearía.
- **`ProviderProbe` es un trait SEPARADO de `LlmProvider`** ⇒ la composición de REQ-A24 es
  legítima: se construye un `OllamaProvider` solo para sondear, sin usarlo para completions.
- **`extraction_failures` es `BTreeMap<AgentName, Vec<ExtractionFailure>>`** y
  `ExtractionFailure.model` es un `&str` ⇒ REQ-A09 puede nombrar *el modelo*. Sin anotar el
  tipo, eso parecía imposible desde una clave `AgentName`.
- **`input_size` es `Option<InputSize>`** ⇒ el `None` se **mapea** en el JSON de magi-rs, no se
  omite. REQ-A11 exige el campo siempre presente, así que la presencia es una traducción
  nuestra, no un reflejo del reporte.
- **`MagiReport` es `#[non_exhaustive]`** ⇒ los dobles de test no pueden construirlo por
  literal ni hacerle `match` exhaustivo.
- **`Mode` NO es `#[non_exhaustive]`**, y magi-core lo documenta como deliberado: *"a new mode
  should break exhaustive matches so consumers revisit their logic"*. El guardián hace el
  `match` sin brazo `_` para aceptar esa invitación.
- **`OllamaProvider::new` devuelve `Result`** porque **normaliza la URL** (le agrega `/v1` si
  falta). Esa normalización antes ocurría en silencio adentro del provider; REQ-A01b obliga a
  anunciarla en un notice.

### Por qué la versión va fijada con `=`

`magi-core = "=3.1.0"` en `[dependencies]` **y** en `[dev-dependencies]`, con la lista de
features repetida en ambas (`ollama`, `openai-compat`, `claude-api`, más `test-utils` en dev).

**No es higiene, es una razón concreta:** el rustdoc de `RetryConfig` anuncia que **3.2.0 cambia
los defaults de retry**, que son exactamente los números de los que REQ-A04 deriva su escala. Un
caret (`"3.1.0"`) aceptaría 3.2.0 sin que nadie lo decida, moviendo el piso de una relación que
este milestone existe para volver imposible de romper.

Fijarlo **no apaga los guardianes: los vuelve útiles.** Un guardián que se rompe en un
`cargo update` que nadie pidió es ruido; roto en un bump deliberado es la señal para la que se
escribió. El bump a 3.2.0 es trabajo de MS3, con este spike corriendo primero.

**Las features son obligatorias, no una mejora.** Los providers están detrás de gates y el
default no trae ninguno: sin ellas, el trío nativo —el núcleo del milestone— no compila. Lo
encontró `cargo check --example ms2_contracts` en noventa segundos, después de veintisiete loops
de revisión que no lo vieron.

**`claude-cli` NO se habilita.** Lanza el binario `claude` como subproceso, esquivando el modelo
de seguridad del tool `bash` (allowlist, prohibición de metacaracteres, `PathGuard`, grupo de
proceso matable), y además rechaza con `NestedSession` bajo `CLAUDECODE`. Prohibido por §8 de la
spec.

---

## Triage de `plan_contract_check.py` — apertura de la Fase 0 (2026-08-02)

El verificador reporta `FALLA (104 hallazgos bloqueantes)`. La salida se leyó **entera**, no
filtrada — el propio plan registra que filtrarla con un `grep` costó dos hallazgos reales que ya
estaban impresos.

| Chequeo | Hallazgos | Veredicto |
|---|---|---|
| **D3** llamado-y-nunca-definido | 88 | **Falsos positivos, los 88.** La cabecera del script declara que D3 solo garantiza *funciones libres*; 86 son métodos de `std`/tokio/clap (`map`, `unwrap`, `lock`, `arg`). Los dos que parecían reales no lo son: `gate_evaluation` es `self.log.gate_evaluation(..)` con `on_gate_evaluation` definido en el trait, y `resolved_mode` es un método de `env`/`envelope` cuyas funciones libres (`read_resolved_mode`, `inject_resolved_mode`) sí están en el stub. |
| **D8** constante usada y no definida | 3 | **Falsos positivos.** `VERDICT_OPEN`/`VERDICT_CLOSE` se importan de `magi_core::verdict_markers`; `BASE_URL_USER` es un **nombre de entrada de vault** dentro de un literal, no una `const` de Rust. |
| **D4** call site vs firma del stub | 16 | **6 falsos positivos** (`build`/`new` con aridad 0 son `MagiBuilder::build()` de magi-core y `ScriptedTwoToolUseProvider::new`) y **10 reales**. |
| D1, D5 | advisory | No bloquean por diseño. |

**Los diez reales son la deuda registrada que el usuario aprobó**, con remediación asignada a la
fase Red de su tarea dueña:

| Símbolo | Defecto | Tarea dueña |
|---|---|---|
| `effective_base_url`, `effective_magi_base_url`, `effective_embedding_base_url` | devuelven `Result<EndpointTemplate, EndpointError>` y los call sites no lo manejan | 1.1, 1.4 |
| `resolve_mode_guarded` | aridad 1 vs 6, `async` sin `.await`, `Result` sin manejar | 2.4 |
| `build_magi_orchestrator` | devuelve `Result<Arc<Magi>, TrioError>` sin manejar | 4.1, 4.3 |
| `build_native_provider` | devuelve `Result<Arc<dyn LlmProvider>, SeatError>` sin manejar | 4.1 |

**Ninguno pertenece a la Fase 0**, así que la fase abre. Se corrigen **contra el stub primero**
(`examples/ms2_contracts.rs` es normativo para firmas), y recién después se propaga al plan —
nunca al revés.
