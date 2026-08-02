// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Guardián de la superficie de API de `magi-core 3.1.0` (Task 0.0, Fase 0 de MS2).
//!
//! # Qué prueba, y qué NO
//!
//! **No prueba comportamiento.** Prueba que cada símbolo de magi-core que MS2 consume
//! **existe y tipa con la forma que el plan asume**. Si magi-core lo renombra, le cambia la
//! aridad, el orden de argumentos o el tipo de un campo, esto **no compila** — que es
//! exactamente el resultado buscado, y en Fase 0 en vez de en Fase 4.
//!
//! # Por qué existe
//!
//! El plan TDD de MS2 asumió una superficie de API que nadie había verificado, y la primera
//! lectura del crate encontró **cinco** suposiciones falsas de un saque: `with_client` no
//! existe en ningún provider, `OllamaProvider` fija 300 s de timeout de cliente sin override
//! (lo que lo saca del camino de las completions — D-A07), `RetryConfig` es
//! `#[non_exhaustive]`, `ClaudeProvider` toma `api_key` **primero**, y `Mode` no tiene ningún
//! método de parseo. Cinco fallos en un solo pase es la medida de cuánta superficie hay.
//!
//! **La lectura no reemplaza a este archivo, lo justifica**: una lectura envejece en cuanto
//! magi-core publica una versión; el compilador no.
//!
//! # Relación con `examples/ms2_contracts.rs`
//!
//! Son dos archivos con dos vidas. El *example* cruza los contratos **internos** de magi-rs
//! entre sí y **se borra** al cerrar la Fase 6, cuando la implementación real lo reemplaza.
//! Este test cubre la frontera con el **crate externo** y **sobrevive al milestone**: es lo
//! que hace que un bump a magi-core 3.2.0 rompa la suite en vez de derivar en silencio.
//!
//! # Cómo se lee un fallo acá
//!
//! Un error de compilación en este archivo **no se arregla acomodando el test**. Se busca el
//! nombre real en el crate, se corrigen **todas** las apariciones en el plan, y la diferencia
//! se anota en `docs/MS2-DECISIONS.md` con fecha. Si el símbolo no existe en ninguna forma
//! —como le pasó a `MagiReport::window_rejected`— eso **no se inventa**: se registra como
//! capacidad ausente y el requerimiento que dependía de ella se replantea con lo que sí hay.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use magi_core::orchestrator::{MagiBuilder, MagiConfig as CoreMagiConfig};
use magi_core::provider::{LlmProvider, RetryConfig, RetryProvider};
use magi_core::providers::claude::ClaudeProvider;
use magi_core::providers::ollama::OllamaProvider;
use magi_core::providers::openai_compat::OpenAiCompatibleProvider;
use magi_core::reporting::{ExtractionFailure, InputSize, MagiReport};
use magi_core::rotation::ProviderProbe;
use magi_core::schema::{AgentName, AgentOutput, Mode};
use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

/// Dobles compartidos de los tests de integración (Task 0.7). Se declara acá porque este es
/// su primer consumidor: un módulo bajo `tests/` que nadie declara **no es un target de
/// build**, así que ni `cargo check` ni `clippy --all-targets` lo compilarían.
mod support;

/// Endpoint sintáctico para construir providers. **Nunca se contacta**: este archivo no hace
/// I/O, solo type-checking, y un provider se construye sin abrir ninguna conexión.
const SYNTHETIC_BASE_URL: &str = "http://127.0.0.1:11434/v1";

/// Modelo sintético, del mismo carácter que [`SYNTHETIC_BASE_URL`].
const SYNTHETIC_MODEL: &str = "guardian-model";

/// Forma de [`MagiReport`] que la Fase 6 consume, **con tipos anotados**.
///
/// Las anotaciones no son ceremonia: un `let _ = &r.campo` prueba que el campo existe y nada
/// más, y toda la telemetría de la Fase 6 **itera** estas estructuras. Un cambio de forma
/// —`Vec` a `BTreeMap`, `T` a `Option<T>`— compilaría un binding suelto y rompería la fase
/// entera. Existencia y forma son dos verificaciones distintas.
///
/// Nunca se llama: su cuerpo se type-checkea igual, que es todo lo que hace falta.
/// `MagiReport` es `#[non_exhaustive]`, así que fuera del crate no hay forma de construir uno.
fn report_shape(r: &MagiReport) {
    let _: &str = &r.report;
    let _: bool = r.degraded;
    // Por ASIENTO y con un `Vec` adentro. Sin anotar el tipo, "nombrar el modelo que no
    // adhirió" (REQ-A09) parecía imposible desde una clave `AgentName`.
    let _: &BTreeMap<AgentName, Vec<ExtractionFailure>> = &r.extraction_failures;
    // **`Option`**, no un valor. REQ-A11 exige el campo SIEMPRE presente en el JSON de
    // magi-rs, así que el `None` se **mapea**, no se omite: es una traducción nuestra, no un
    // reflejo del reporte.
    let _: &Option<InputSize> = &r.input_size;
    // El sustituto verificado de un `window_rejected` que NO existe en `MagiReport` (vive en
    // `rotation.rs`, que es MS3). REQ-A11d y SC-A11g se replantearon sobre esto.
    let _: &BTreeMap<AgentName, String> = &r.failed_agents;
    // Sostiene SC-A11g: su vacío ES "cero veredictos válidos", que no es un consenso
    // degradado sino la ausencia de consenso.
    let _: &Vec<AgentOutput> = &r.agents;
    // MS3. Acá solo se comprueba que el campo existe; su forma la fija ese milestone.
    let _ = &r.rotations;
}

/// Campos de [`ExtractionFailure`] que REQ-A09 exige surface-ar.
///
/// `model` es el que no puede faltar: con rotación (MS3) la pregunta accionable es *qué
/// modelo* no adhiere, no *qué asiento*.
fn extraction_failure_shape(f: &ExtractionFailure) {
    let _: &str = &f.model;
    let _: u8 = f.attempt;
    let _ = &f.cause;
}

/// Campos de [`InputSize`], que van los tres al JSON de REQ-A11 sin omitir ninguno.
fn input_size_shape(s: &InputSize) {
    let _: usize = s.estimated_tokens;
    let _: usize = s.warn_threshold;
    let _: bool = s.exceeded;
}

/// Métodos de [`MagiBuilder`] que el cableado del trío encadena (Task 4.1).
///
/// Encadenados a propósito: cada uno debe devolver `Self` por valor. Si alguno pasara a
/// `&mut Self`, la cadena deja de compilar acá en vez de en la Fase 4.
fn builder_surface(b: MagiBuilder, p: Arc<dyn LlmProvider>) -> MagiBuilder {
    b.with_timeout(Duration::from_secs(90))
        .with_provider(AgentName::Melchior, p)
        .with_input_warn_tokens(96_000)
        .with_retry_disabled()
}

/// Que un tipo concreto satisfaga [`LlmProvider`], no solo que el trait exista.
fn assert_is_provider<P: LlmProvider + 'static>(_p: &P) {}

/// Ídem para [`ProviderProbe`], que es un trait **separado** — la composición de REQ-A24
/// depende de que se pueda implementar uno sin el otro.
fn assert_is_probe<P: ProviderProbe + 'static>(_p: &P) {}

/// El `match` sobre [`Mode`] es exhaustivo **sin brazo `_`**.
///
/// magi-core documenta el enum como deliberadamente cerrado: *"no `#[non_exhaustive]`: a new
/// mode should break exhaustive matches so consumers revisit their logic"*. Tres funciones de
/// MS2 lo asumen (`GateThresholds::for_mode`, `CliMode::into_mode`, `normalize_label`), así
/// que fijarlo acá es aceptar la invitación: si 3.2.0 agrega un modo, **esto** es lo primero
/// que rompe, en Fase 0, en vez de un `for_mode` devolviendo el umbral equivocado en Fase 3.
fn mode_is_closed(m: Mode) -> &'static str {
    match m {
        Mode::CodeReview => "code-review",
        Mode::Design => "design",
        Mode::Analysis => "analysis",
    }
}

#[test]
fn magi_core_api_surface_is_what_the_plan_assumes() {
    // --- (1) Los tres asientos y los tres modos ------------------------------------------
    let _seats = [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar];
    assert_eq!(mode_is_closed(Mode::CodeReview), "code-review");
    assert_eq!(mode_is_closed(Mode::Design), "design");
    assert_eq!(mode_is_closed(Mode::Analysis), "analysis");

    // --- (2) Propiedades de TIPO de las que cuelga el diseño -----------------------------
    fn assert_clone<T: Clone>() {}
    fn assert_copy_eq<T: Copy + PartialEq>() {}
    // Los tres asientos comparten una config de retry y cada `RetryProvider::with_config` la
    // consume **por valor**: sin `Clone` hay que reconstruirla por asiento y la escala
    // derivada de REQ-A04 deja de ser una sola cosa.
    assert_clone::<RetryConfig>();
    // `GateVerdict::Veto { mode: *mode }` y media docena de `assert_eq!` sobre modos.
    assert_copy_eq::<Mode>();
    // La costura de inyección del probe (Task 5.1) es `Arc<dyn ProviderProbe>`. Si el trait
    // dejara de ser dyn-compatible, la fábrica entera se replantea — mejor saberlo acá.
    let _: Option<Arc<dyn ProviderProbe>> = None;

    // --- (3) `RetryConfig` es `#[non_exhaustive]` ---------------------------------------
    // Fuera del crate NO compila ni el literal `RetryConfig { .. }` ni el update funcional
    // `..default()`. El patrón obligado es `default()` mutable, que es el que magi-core
    // documenta.
    let mut retry = RetryConfig::default();
    retry.operation_budget = Duration::from_secs(54);
    let _: Duration = retry.operation_budget;

    // --- (4) `Mode` NO tiene método de parseo -------------------------------------------
    // Lo que existe es `Display` + serde en kebab-case, y por eso MS2 necesita su propio
    // `ModeExt::parse_config_value` (Task 1.0). Ese trait es de magi-rs y nace en la Fase 1:
    // nombrarlo acá volvería no-compilable justo al spike cuyo trabajo es impedir eso.
    let _: String = Mode::CodeReview.to_string();
    let parsed: Mode = serde_json::from_str(r#""code-review""#).expect("kebab-case");
    assert_eq!(parsed, Mode::CodeReview);

    // --- (5) Constructores, con su ORDEN DE ARGUMENTOS real ------------------------------
    // `api_key` PRIMERO en Claude. Los dos parámetros son `impl Into<String>`, así que
    // invertirlos **compila** y falla en runtime con un 401 — el tipo de defecto que ninguna
    // revisión encuentra.
    let _ = ClaudeProvider::new("api-key", SYNTHETIC_MODEL);
    let _ = ClaudeProvider::with_timeout("api-key", SYNTHETIC_MODEL, Duration::from_secs(27));

    // `Option<String>` en el TERCER parámetro; `None` es el caso Ollama (keyless).
    let openai = OpenAiCompatibleProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL, None)
        .expect("base_url sintáctica válida");
    let _ = OpenAiCompatibleProvider::with_timeout(
        SYNTHETIC_BASE_URL,
        SYNTHETIC_MODEL,
        None,
        Duration::from_secs(27),
    );
    assert_is_provider(&openai);

    // `OllamaProvider` sigue existiendo, pero SOLO como sonda: su único constructor fija un
    // cliente de 300 s sin override, lo que hace imposible la relación de REQ-A04
    // (`operation_budget + client_timeout <= techo`). Devuelve `Result` porque normaliza la
    // URL — y esa normalización es la que REQ-A01b obliga a anunciar en un notice.
    let ollama = OllamaProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL)
        .expect("base_url sintáctica válida");
    assert_is_provider(&ollama);
    assert_is_probe(&ollama);

    // --- (6) `RetryProvider` envuelve un `Arc<dyn LlmProvider>` --------------------------
    // REQ-A03: `MagiBuilder::build()` NO envuelve nada, así que sin esto el trío pierde el
    // reintento que hoy hereda del adapter — una regresión de resiliencia.
    let inner: Arc<dyn LlmProvider> = Arc::new(
        OpenAiCompatibleProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL, None)
            .expect("base_url sintáctica válida"),
    );
    let _ = RetryProvider::with_config(inner, RetryConfig::default());

    // --- (7) Config del orquestador: los dos campos que la escala derivada lee ------------
    let _: Duration = CoreMagiConfig::default().timeout;
    let _: usize = CoreMagiConfig::default().max_input_len;

    // --- (8) Marcadores de veredicto (contrato 3.0.0) ------------------------------------
    // Los dobles de test DEBEN emitir el veredicto entre estos marcadores: magi-core borró su
    // parser de búsqueda, así que un JSON pelado ya no parsea por más válido que sea.
    assert!(!VERDICT_OPEN.is_empty());
    assert!(!VERDICT_CLOSE.is_empty());

    // --- (9) Formas que no se pueden instanciar fuera del crate --------------------------
    // Referenciar el item lo marca como usado; su cuerpo ya se type-checkeó. No hace falta
    // llamarlo, y no hay `#[allow(dead_code)]` que justificar.
    let _ = report_shape;
    let _ = extraction_failure_shape;
    let _ = input_size_shape;
    let _ = builder_surface;
}

/// Contenido con largo suficiente para que el orquestador despache los tres asientos.
///
/// El gate de complejidad de magi-core veta contenido trivial, así que un payload corto haría
/// que estos guardianes midieran **cero llamadas** y pasaran por la razón equivocada.
const DISPATCHABLE_CONTENT: &str =
    "Contenido con largo más que suficiente para que el orquestador despache los tres \
     asientos en vez de vetar la consulta por trivial, que es lo que haría un payload corto.";

/// Asientos que magi-core despacha por consulta: el trío completo.
const EXPECTED_SEATS: usize = 3;

/// Intentos por asiento ante un schema inválido — **medido**, no supuesto.
///
/// Una sonda sobre magi-core 3.1.0 (2026-08-02) observó los tres asientos con exactamente dos
/// llamadas cada uno. Es de donde sale el factor 2 de la fórmula del `--timeout` (REQ-A04), y
/// por eso se afirma el valor exacto en vez de un `>= 2`: ese `>=` pasaba con **un solo**
/// asiento reintentando, o sea que no distinguía el caso sano del degradado.
const ATTEMPTS_PER_SEAT: usize = 2;

/// SC-A04b, primera mitad: un fallo de schema consume **DOS** ventanas de `timeout`.
///
/// De acá sale el factor 2 de la fórmula del `--timeout` headless (REQ-A04). Si magi-core
/// dejara de reintentar ante schema inválido, la escala quedaría sobredimensionada y nadie se
/// enteraría — el consult seguiría funcionando, solo que el `--timeout` derivado pasaría a
/// cubrir el doble de lo necesario.
///
/// **El conteo es POR ASIENTO, y esa es la diferencia entre un guardián y un adorno.** Con un
/// contador global, `total >= 2` pasa con tres mages llamando una vez cada uno — o sea
/// **aunque magi-core no reintente jamás**. El system prompt discrimina asientos porque cada
/// mage recibe el suyo (REQ-A02).
#[tokio::test]
async fn schema_retry_consumes_two_timeout_windows_per_seat() {
    let ceiling = Duration::from_secs(2);
    let provider = Arc::new(support::SchemaFailsOnceProvider::new(
        Duration::from_millis(100),
    ));
    let magi = MagiBuilder::new(provider.clone())
        .with_timeout(ceiling)
        .build()
        .expect("el builder acepta un solo provider compartido");

    let started = Instant::now();
    let _ = magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT).await;
    let elapsed = started.elapsed();

    let by_seat = provider.calls_by_seat();
    // Solo los CONTEOS en los mensajes: las claves son los system prompts completos, o sea
    // ~30 KB que volverían ilegible cualquier fallo. El conteo es el dato; la clave, el medio.
    let counts: Vec<usize> = by_seat.values().copied().collect();

    assert_eq!(
        by_seat.len(),
        EXPECTED_SEATS,
        "se esperaban {EXPECTED_SEATS} asientos con system prompts distintos y hubo \
         {}: o magi-core dejó de despachar el trío completo, o dejó de darle a cada mage su \
         propio system prompt (REQ-A02) — que es lo que hace discriminable este conteo",
        by_seat.len(),
    );
    assert!(
        counts.iter().all(|n| *n == ATTEMPTS_PER_SEAT),
        "cada asiento debe consumir exactamente {ATTEMPTS_PER_SEAT} intentos ante schema \
         inválido; se observaron {counts:?}. Menos ⇒ magi-core dejó de reintentar y el factor \
         2 de REQ-A04 sobredimensiona la escala. Más ⇒ el peor caso ya no es 2× el techo y la \
         fórmula del `--timeout` subestima",
    );
    assert!(
        elapsed < ceiling * 3,
        "el peor caso superó 2× el techo ({elapsed:?} con techo {ceiling:?})",
    );
}

/// SC-A04b, segunda mitad: un provider **colgado** consume UNA sola ventana.
///
/// Es la asimetría que hace correcta la fórmula: un timeout del provider **no** dispara el
/// reintento correctivo de schema, así que ese camino cuesta 1×, no 2×. Si magi-core empezara
/// a reintentar también tras un timeout, el peor caso por mage saltaría de 2× a 4× y el
/// `--timeout` derivado empezaría a cortar consults sanos.
#[tokio::test]
async fn a_hanging_provider_consumes_one_timeout_window() {
    let ceiling = Duration::from_millis(300);
    let magi = MagiBuilder::new(Arc::new(support::HangingProvider))
        .with_timeout(ceiling)
        .build()
        .expect("el builder acepta un solo provider compartido");

    let started = Instant::now();
    let _ = magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT).await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < ceiling * 2,
        "un cuelgue consumió {elapsed:?} con techo {ceiling:?}: magi-core empezó a reintentar \
         tras timeout, y el peor caso de REQ-A04 pasa de 2× a 4×",
    );
}

/// Cuánto permanece «adentro» cada llamada del doble de solapamiento.
///
/// **Medio segundo y no los 150 ms del primer borrador.** El pico solo baja de 3 si un asiento
/// llega a arrancar después de que otro ya salió, o sea si el scheduler se demora más que este
/// valor en despachar tres tareas. Con 150 ms eso es improbable pero alcanzable bajo la carga
/// Argon2 del resto de la suite —este test no está en el grupo `heavy`— y produciría
/// exactamente el fallo intermitente que `.config/nextest.toml` documenta.
///
/// Medio segundo hace la ventana holgada sin debilitar la aserción, que es la dirección
/// correcta del intercambio: se paga medio segundo una vez, no un guardián que a veces miente.
const OVERLAP_DWELL: Duration = Duration::from_millis(500);

/// SC-A04e: los tres mages se ejecutan **solapados**, no en serie.
///
/// Es lo que sostiene el «**NO** se multiplica por 3» de la fórmula del `--timeout` (REQ-A04):
/// con despacho paralelo el peor caso de un consult es el del mage más lento, no la suma de
/// los tres. Si magi-core pasara a despachar en serie, ese peor caso saltaría de 2× a 6× el
/// techo y el `--timeout` derivado empezaría a cortar consults perfectamente sanos — **sin que
/// una sola línea de magi-rs cambiara**, que es exactamente el fallo silencioso que este
/// guardián convierte en una suite rota.
///
/// **El pico se afirma en {EXPECTED_SEATS}, no en `>= 2`.** Dos mages solapados y el tercero
/// en serie ya rompe la fórmula —el peor caso pasa a 4×— y un `>= 2` lo daría por bueno.
#[tokio::test]
async fn the_three_mages_execute_concurrently() {
    let (provider, peak) = support::OverlapCountingProvider::new(OVERLAP_DWELL);

    let magi = MagiBuilder::new(provider)
        .with_timeout(Duration::from_secs(5))
        .build()
        .expect("el builder acepta un solo provider compartido");

    let _ = magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT).await;

    let observed = peak.load(Ordering::SeqCst);
    assert_eq!(
        observed, EXPECTED_SEATS,
        "pico de concurrencia = {observed}, esperado {EXPECTED_SEATS}: magi-core dejó de \
         despachar los tres asientos en paralelo. La fórmula del `--timeout` (REQ-A04) asume \
         solapamiento total y ahora subestima el peor caso",
    );
}
