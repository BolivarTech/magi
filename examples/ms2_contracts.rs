// Author: Julian Bolivar
// Version: 1.1.0
// Date: 2026-08-01

//! Contratos de MS2, verificados por `rustc` en vez de por lectura.
//!
//! # Por qué existe
//!
//! El plan TDD de MS2 fija ~160 firmas en bloques de Rust pegados dentro de un `.md`. Durante
//! el Checkpoint 2 apareció **seis veces** la misma clase de defecto: se cambia una definición
//! y no se propaga a sus consumidores. Sobrevivió a tres contramedidas. El diagnóstico de
//! Caspar (loop 27) es el correcto:
//!
//! > *"prose corrections and code blocks live in the same document and only one of them
//! > compiles"*
//!
//! Este archivo es el que compila.
//!
//! # Regla de autoridad
//!
//! **Para FIRMAS, este archivo es NORMATIVO por encima del plan.** Los bloques de código del
//! plan son ilustrativos: muestran cuerpo, tests y razones. Si un bloque del plan contradice
//! una firma de acá, **el plan está mal** y se corrige hacia acá — nunca al revés sin pasar
//! por este archivo primero. La regla existe porque la sexta recurrencia de la deriva ocurrió
//! *después* de crear este archivo: cerré un `[CRITICAL]` acá y cinco call sites del plan
//! quedaron con la forma vieja, invisibles porque no estaban en [`wiring`].
//!
//! # La lección de la v1: cruzar CONSUMIDORES, no declarar contratos
//!
//! Declarar las firmas no atrapa la deriva — un contrato declarado y uno consumido pueden
//! divergir sin que este archivo se entere. [`wiring`] debe llamar a cada firma **desde el
//! consumidor que el plan declara**: la resolución del vault entrando al trío y al probe,
//! `RunContext::build` con su productor, los contratos de salida de Fase 6, los campos de
//! config de raíz. Cada seam que falte acá es un lugar donde la deriva sigue viva.
//!
//! # Cómo se usa
//!
//! ```text
//! cargo check --example ms2_contracts
//! ```
//!
//! Primer paso de Task 0.0 y precondición de la fase Red de toda tarea.
//!
//! # Ciclo de vida (regla de transición, que faltaba)
//!
//! Cada tarea que implementa un contrato **borra su bloque de acá y poda las líneas de
//! [`wiring`] que lo llamaban, en el mismo commit** — y `cargo check --example ms2_contracts`
//! debe seguir verde después de cada tarea, sobre lo que quede. El archivo se elimina entero
//! al cerrar la Fase 6, cuando el último bloque migró. Si sobreviviera al milestone sería una
//! segunda fuente de verdad, que es el problema que vino a resolver.

#![allow(dead_code, unused_variables, clippy::needless_pass_by_value)]

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use magi_core::orchestrator::{Magi, MagiBuilder, MagiConfig as CoreMagiConfig};
use magi_core::provider::{LlmProvider, RetryConfig, RetryProvider};
use magi_core::reporting::{ExtractionFailure, InputSize, MagiReport};
use magi_core::rotation::ProviderProbe;
use magi_core::schema::{AgentName, AgentOutput, Mode};

// ===========================================================================
// 1. SUPERFICIE DE MAGI-CORE 3.1.0
//
// Verificado por lectura el 2026-08-01 y fijado acá porque la lectura envejece. Cinco
// suposiciones del plan resultaron falsas contra el crate real, y ninguna la encontró un
// loop de revisión: `with_client` no existe en ningún provider, `OllamaProvider` fija 300 s
// de timeout sin override, `RetryConfig` es `#[non_exhaustive]`, `ClaudeProvider` toma
// `api_key` PRIMERO, y `Mode` no tiene método de parseo.
// ===========================================================================

/// Los tres asientos, y que el `match` sobre `Mode` sea exhaustivo **sin brazo `_`**.
///
/// magi-core documenta `Mode` como deliberadamente cerrado: *"a new mode should break
/// exhaustive matches so consumers revisit their logic"*. Si 3.2.0 agrega un modo, **esto**
/// es lo primero que rompe, en Fase 0.
fn core_enums() {
    let _seats = [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar];
    let label = |m: Mode| match m {
        Mode::CodeReview => "code-review",
        Mode::Design => "design",
        Mode::Analysis => "analysis",
    };
    let _ = label(Mode::Analysis);
}

/// Propiedades de tipo de las que cuelga el diseño, no solo existencia de símbolos.
fn core_type_properties() {
    fn assert_clone<T: Clone>() {}
    fn assert_copy_eq<T: Copy + PartialEq>() {}

    // `RetryConfig: Clone` — los tres asientos comparten una config y cada
    // `RetryProvider::with_config` la consume por valor.
    assert_clone::<RetryConfig>();
    // `Mode: Copy + PartialEq` — `GateVerdict::Veto { mode: *mode }` y los `assert_eq!`.
    assert_copy_eq::<Mode>();
    // `dyn ProviderProbe` — la costura de inyección de Task 5.1 es `Arc<dyn ProviderProbe>`.
    let _: Option<Arc<dyn ProviderProbe>> = None;
}

/// `RetryConfig` es `#[non_exhaustive]`: fuera del crate NO compila el literal ni `..default()`.
fn core_retry_config() -> RetryConfig {
    let mut cfg = RetryConfig::default();
    cfg.operation_budget = Duration::from_secs(54);
    cfg
}

/// Constructores, con su **orden de argumentos** real.
fn core_constructors() {
    // OJO: `api_key` PRIMERO. Los dos son `impl Into<String>`, así que invertirlos COMPILA y
    // falla en runtime con un 401.
    let _ = magi_core::providers::claude::ClaudeProvider::with_timeout(
        "api-key",
        "model",
        Duration::from_secs(27),
    );
    // `Option<String>` en el tercer parámetro; `None` es el caso Ollama (keyless).
    let _ = magi_core::providers::openai_compat::OpenAiCompatibleProvider::with_timeout(
        "http://host/v1",
        "model",
        None,
        Duration::from_secs(27),
    );
    // Sigue existiendo, pero SOLO como sonda: su cliente es de 300 s fijos, sin override, y
    // por eso no puede llevar las completions (D-A07).
    let _ = magi_core::providers::ollama::OllamaProvider::new("http://host:11434/v1", "model");
}

/// Métodos del builder que el cableado usa.
fn core_builder(b: MagiBuilder, p: Arc<dyn LlmProvider>) -> MagiBuilder {
    b.with_timeout(Duration::from_secs(90))
        .with_provider(AgentName::Melchior, p)
        .with_input_warn_tokens(96_000)
        .with_retry_disabled()
}

/// Campos de `MagiReport` **con su forma**, no solo su existencia.
///
/// Dos hallazgos que solo aparecieron al anotar tipos: `extraction_failures` va por ASIENTO
/// con un `Vec` adentro (y `ExtractionFailure.model` es lo que REQ-A09 exige nombrar), e
/// `input_size` es **`Option`**, no un valor — REQ-A11 exige el campo siempre presente en
/// NUESTRO json, así que el `None` se mapea, no se omite.
fn core_report_shape(r: &MagiReport) {
    let _: &str = &r.report;
    let _: bool = r.degraded;
    let _: &BTreeMap<AgentName, Vec<ExtractionFailure>> = &r.extraction_failures;
    let _: &Option<InputSize> = &r.input_size;
    let _: &BTreeMap<AgentName, String> = &r.failed_agents;
    // `agents` sostiene SC-A11g: su vacío ES "cero veredictos válidos".
    let _: &Vec<AgentOutput> = &r.agents;

    if let Some((_seat, fs)) = r.extraction_failures.iter().next() {
        if let Some(f) = fs.first() {
            let _: &str = &f.model;
            let _: u8 = f.attempt;
        }
    }
    if let Some(s) = r.input_size.as_ref() {
        let (_, _, _): (usize, usize, bool) = (s.estimated_tokens, s.warn_threshold, s.exceeded);
    }
    let _: Duration = CoreMagiConfig::default().timeout;
    let _: usize = CoreMagiConfig::default().max_input_len;
}

// ===========================================================================
// 2. CONTRATOS DE MAGI-RS
//
// La frontera lib/bin y el acuerdo de firmas entre tareas son nuestros, y son los que se
// rompen: seis de los últimos defectos vivieron acá y ninguno en el crate externo.
// ===========================================================================

/// Vocabulario de backend (Task 1.0/1.1). **Vive en el LIB**: lo toman `probe_models` y
/// `ProbeFactory`, que son puros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Ollama,
    OpenAiCompat,
    Anthropic,
}

/// Error del LIB. No puede ser `ConfigError`, que vive en `config.rs` (bin).
#[derive(Debug)]
pub struct ProviderKindParseError {
    /// El valor que no esta en el vocabulario, **tal cual lo escribio el usuario**.
    ///
    /// Sin el, el error dice "kind no reconocido" y el usuario tiene que ir a buscar cual de
    /// sus dos claves lo trae. REQ-A01b pide un error explicito, y explicito incluye *que*
    /// valor. Es un nombre de provider, no un secreto: no pasa por redaccion.
    pub got: String,
}

impl ProviderKind {
    /// `Ok(None)` = ausente o en blanco. `Err` = presente y no reconocido.
    pub fn parse(raw: &str) -> Result<Option<Self>, ProviderKindParseError> {
        unimplemented!()
    }
    /// Solo `ollama` expone `/api/show` y `/api/tags`.
    pub const fn is_probeable(self) -> bool {
        matches!(self, Self::Ollama)
    }
}

/// De qué nivel salió el modo efectivo (REQ-A08). **CINCO, no cuatro.**
///
/// `AgentChosen` es la corrección del loop 34: el agente eligiendo por el `input_schema` y una
/// llamada de clasificación sobre el contenido compartían la etiqueta `Inferred`, así que
/// ninguna guarda podía distinguirlos — y la de `untrusted_content` terminó bloqueando los dos,
/// matando SC-A07d, que es requerimiento duro del producto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeSource {
    /// Lo declaró un HUMANO: `--mode`, `/consult --mode`, campo del envelope.
    Explicit,
    /// `[magi].default_mode`. Le gana al agente: es la perilla para fijar la lente.
    Configured,
    /// Lo eligió el AGENTE por el `mode` del `input_schema`. Cero llamadas extra.
    /// **No es `Explicit`** — por eso no satisface una guarda que exige declaración humana.
    AgentChosen,
    /// Salió de una llamada de CLASIFICACIÓN sobre el contenido. La que `untrusted_content`
    /// bloquea, porque es la superficie de ataque dedicada.
    Inferred,
    /// `Analysis`, porque no hubo ninguno de los anteriores.
    Default,
}

/// Error del LIB para un valor de config que no nombra un modo.
///
/// **Enum, no unit struct** (corregido en Task 1.0, contra la implementación real): los tests
/// del vocabulario discriminan con `Err(ModeParseError::Unknown { .. })`, y un unit struct no
/// admite ese patrón. El stub es normativo para firmas, así que cuando la implementación y él
/// difieren se corrige ACÁ primero y después se propaga — nunca al revés.
#[derive(Debug)]
pub enum ModeParseError {
    /// El valor tiene contenido y no es una de las tres etiquetas.
    Unknown {
        /// Lo que trajo el archivo.
        got: String,
        /// Los tres aceptados, para que el error sea accionable sin abrir la doc.
        valid: &'static str,
    },
}

/// Extensión de `Mode`: es un tipo foráneo y no admite métodos inherentes.
pub trait ModeExt: Sized {
    fn parse_config_value(raw: &str) -> Result<Option<Self>, ModeParseError>;
}

impl ModeExt for Mode {
    fn parse_config_value(raw: &str) -> Result<Option<Self>, ModeParseError> {
        unimplemented!()
    }
}

/// Normalización CERRADA de tres pasos: trim ASCII, minúsculas ASCII, comparación literal.
pub fn normalize_label(raw: &str) -> Option<Mode> {
    unimplemented!()
}

/// Clasificador de modo. El **trait** es puro y va al lib; `ProviderClassifier`, que necesita
/// el `Provider` del bin, vive en `src/agent/mode_classifier.rs`.
#[async_trait::async_trait]
pub trait ModeClassifier: Send + Sync {
    async fn classify(&self, content: &str) -> Option<Mode>;
}

#[derive(Debug)]
pub enum ModeError {
    UntrustedContentRequiresExplicitMode,
}

/// El resultado COMPLETO de resolver el modo — incluida la señal de privacidad.
///
/// **Decisión de dueño (loop 29, pregunta de Balthasar): el resolutor devuelve si INTENTÓ
/// clasificar, en vez de que cuatro llamadores lo re-deriven.** Re-derivarlo era el origen
/// del falso negativo de `endpoint_divergence`: una clasificación intentada-y-fallida deja
/// `ModeSource::Default`, pero el contenido **ya salió** hacia el provider principal. El
/// único que sabe con certeza si la llamada ocurrió es quien la hizo; ese conocimiento viaja
/// en el retorno o se pierde.
pub struct ModeResolution {
    pub mode: Mode,
    pub source: ModeSource,
    /// `true` si la llamada de clasificación SE HIZO, complete o no. Es la señal que
    /// `RunContext.endpoint_divergence` y `divergence_notice` consumen (REQ-A11d).
    pub classification_attempted: bool,
}

/// **UNA sola definición, `async`, con la clasificación adentro.**
///
/// Llegó a estar escrita tres veces con firmas contradictorias mientras los call sites
/// llamaban a la que reintroducía el bypass de `untrusted_content`. Que la clasificación viva
/// adentro hace inexpresable el orden equivocado: con la marca activa y sin modo declarado,
/// el contenido no sale hacia el provider principal porque el `Err` ocurre antes.
/// Resuelve el modo con la guarda de contenido no confiable (REQ-A07d).
///
/// **`agent_chosen` es un parámetro APARTE de `explicit`, y esa separación es la corrección.**
/// Mientras la elección del agente entraba por `explicit`, satisfacía la guarda por su cuenta
/// — el bypass que Caspar encontró. Y sacarle el campo `mode` al schema, que fue el arreglo
/// anterior, mataba SC-A07d sin comprar seguridad: un agente comprometido al punto de elegir
/// mal la lente puede directamente no consultar, o mentir en el reporte.
///
/// Precedencia: `explicit` > `configured` > `agent_chosen` > clasificación > `Analysis`.
///
/// # Errors
/// [`ModeError::UntrustedContentRequiresExplicitMode`] si la marca está activa y no hay modo
/// **declarado** (humano o config) ni elegido por el agente — o sea, cuando la única salida
/// restante sería la clasificación, que es justo lo que la marca bloquea.
pub async fn resolve_mode_guarded(
    explicit: Option<Mode>,
    configured: Option<Mode>,
    agent_chosen: Option<Mode>,
    untrusted: bool,
    classifier: Option<&dyn ModeClassifier>,
    content: &str,
) -> Result<ModeResolution, ModeError> {
    unimplemented!()
}

/// Umbrales del gate por modo (REQ-A20b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateThresholds {
    pub code_review: usize,
    pub design: usize,
    pub analysis: usize,
}

/// Overrides de `[magi.complexity]`, con NOMBRE por campo.
///
/// Reemplaza a `from_parts(Option<usize>, Option<usize>, Option<usize>)`: tres posicionales
/// del mismo tipo son exactamente el swap silencioso que el rustdoc de `GateThresholds`
/// condena — y que `OpenAiSettings` ya resolvió igual en este árbol.
#[derive(Debug, Clone, Copy, Default)]
pub struct GateOverrides {
    pub code_review: Option<usize>,
    pub design: Option<usize>,
    pub analysis: Option<usize>,
}

impl GateThresholds {
    pub const fn builtin() -> Self {
        Self {
            code_review: 200,
            design: 500,
            analysis: 200,
        }
    }
    /// Tabla ausente ⇒ `GateOverrides::default()` ⇒ built-ins: el gate no se apaga por
    /// omitir una sección. Clave ausente DENTRO de la tabla ⇒ su built-in, no cero.
    pub fn from_overrides(o: GateOverrides) -> Self {
        unimplemented!()
    }
    pub const fn for_mode(&self, mode: &Mode) -> usize {
        match mode {
            Mode::CodeReview => self.code_review,
            Mode::Design => self.design,
            Mode::Analysis => self.analysis,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    Dispatch,
    Veto { mode: Mode },
}

/// Predicado 100 % puro: sin async, sin I/O, sin llamadas al modelo.
pub fn evaluate(content: &str, mode: &Mode, thresholds: &GateThresholds) -> GateVerdict {
    unimplemented!()
}

/// Cómo el par resuelto cruza el trait `Tool`.
///
/// `Tool::execute(&self, args: Value, cancel: &CancellationToken)` no tiene por dónde
/// pasarlo: la firma es del trait y este milestone no la cambia. El embudo del agente
/// resuelve UNA vez y **escribe el resultado en el input** bajo claves reservadas;
/// `ConsultTool::execute` las lee en vez de re-resolver. Prefijo `__` y fuera del
/// `input_schema`: el modelo no las conoce y no puede falsificarlas.
pub const RESOLVED_MODE_KEY: &str = "__resolved_mode";
/// Ver [`RESOLVED_MODE_KEY`].
pub const RESOLVED_MODE_SOURCE_KEY: &str = "__resolved_mode_source";

/// El input MUTABLE que el despacho necesita.
///
/// `for content in &response.content` presta la respuesta de forma inmutable, así que el
/// `&Value` del `ToolUse` no se puede mutar en el lugar (`[WARNING]` del loop 31). Se clona
/// el input —un `Value` de un tool call, decenas de bytes— y se inyecta sobre la copia. El
/// costo está escrito porque la alternativa (recolectar los `ToolUse` antes del bucle) rompe
/// el despacho secuencial del que dependen cuatro contadores por turno.
pub fn input_for_dispatch(input: &serde_json::Value, res: &ModeResolution) -> serde_json::Value {
    unimplemented!()
}

/// Escribe la resolución en el input, justo antes de despachar el tool. Cubre **los dos**
/// despachos: el bucle de `ToolUse` del modelo Y la inyección forzada de
/// `authorize_and_execute_tool` — el forced-consult sin esta llamada era un `[CRITICAL]`.
///
/// **SOBRESCRIBE, nunca fusiona ni respeta un valor previo.** El input viene del modelo, así
/// que puede traer `__resolved_mode` puesto por él: el prefijo `__` y la ausencia del campo
/// en el `input_schema` lo hacen improbable, pero **improbable no es imposible**, y confiar
/// en la oscuridad de un nombre es exactamente el tipo de defensa que este plan rechaza en
/// otros lados. Escribir siempre encima hace que el valor del modelo no pueda sobrevivir.
pub fn inject_resolved_mode(input: &mut serde_json::Value, res: &ModeResolution) {
    unimplemented!()
}

/// El modo que el AGENTE eligió por el `input_schema` — nivel 3, no nivel 1.
///
/// **Ya no toma `untrusted`, y el campo ya no se omite del schema.** El arreglo anterior
/// ignoraba `input["mode"]` con la marca activa; eso cerraba el bypass pero mataba SC-A07d.
/// Lo que cierra el bypass ahora es el TIPO: esto devuelve el nivel `AgentChosen`, que no es
/// `Explicit`, así que no satisface una guarda que exige declaración humana.
pub fn agent_chosen_mode(input: &serde_json::Value) -> Option<Mode> {
    unimplemented!()
}

/// La ausencia es un ERROR TIPADO, no un `Option` — y **no hay fallback a `input["mode"]`**.
///
/// Un embudo que no inyectó es un bug del cableado; re-resolver o leer el campo del modelo
/// "para salir del paso" es lo que permitía que el gate y el consult corrieran con modos
/// distintos, y que el agente satisficiera su propia guarda de `untrusted_content`.
#[derive(Debug)]
pub struct ModeInjectionMissing;

/// Ver [`ModeInjectionMissing`].
pub fn read_resolved_mode(
    input: &serde_json::Value,
) -> Result<(Mode, ModeSource), ModeInjectionMissing> {
    unimplemented!()
}

/// Lo que la TUI arma para CADA turno interactivo — el sitio que faltaba nombrar.
///
/// `[CRITICAL]` del loop 31: el plan nombraba el sitio de `gate_telemetry` y **no** el de
/// `gate_thresholds` ni `mode_config`, así que la superficie de mayor tráfico habría corrido
/// con los built-in ignorando el `magi.toml` — un `code_review = 0` configurado no habría
/// apagado nada en la TUI, en silencio.
pub struct TuiRunConfigParts {
    pub gate_thresholds: GateThresholds,
    pub mode_config: ModeConfig,
    pub gate_telemetry: Arc<dyn GateTelemetry>,
}

/// Config de modo por corrida: lo que el embudo necesita sin volver a leer el TOML.
#[derive(Debug, Clone, Copy)]
pub struct ModeConfig {
    pub default_mode: Option<Mode>,
    pub untrusted_content: bool,
}

/// El sitio de construcción, nombrado: `tui/mod.rs` lo llama al armar la app y el resultado
/// va al `AgentRunConfig` de cada turno.
pub fn tui_run_config_parts(cfg: &MagiConfigStub) -> TuiRunConfigParts {
    unimplemented!()
}

/// Telemetría del gate. **Separada del `RunObserver`**: el observer es `None` en la TUI, que
/// es justo la superficie que más consults autorrutea, así que colgar de él una señal que
/// SC-A20h exige *siempre* la volvía condicional a la superficie.
pub trait GateTelemetry: Send + Sync {
    fn on_gate_evaluation(&self, mode: &Mode, chars: usize, threshold: usize, vetoed: bool);
}

// ---------------------------------------------------------------------------
// Endpoints: el secreto no vive en el TOML (REQ-A16c)
// ---------------------------------------------------------------------------

/// A qué `base_url` corresponde, y por lo tanto qué entradas de vault lee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Root,
    Magi,
    Embedding,
}

/// Lo que el vault sabe hacer, reducido a lo que este contrato necesita.
pub trait SecretLookup {
    fn get(&self, name: &str) -> Option<String>;
}

#[derive(Debug)]
pub enum EndpointError {
    LiteralCredential,
    MissingVaultEntry { entry: &'static str },
    UnknownPlaceholder,
}

/// `base_url` **tal como está en el archivo**: con `[user]`/`[password]`, nunca el secreto.
pub struct EndpointTemplate(String);

/// URL con los placeholders ya sustituidos. **Solo `EndpointTemplate::resolve` la construye**,
/// y resolver exige el vault — por eso un `&str` sin resolver no puede llegar a un provider.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint(String);

impl fmt::Debug for ResolvedEndpoint {
    /// A mano y redactado: un `derive` acá es la forma más fácil de filtrar la credencial.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResolvedEndpoint({})", redact_url(&self.0))
    }
}

impl ResolvedEndpoint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl EndpointTemplate {
    pub fn parse(raw: &str) -> Result<Self, EndpointError> {
        unimplemented!()
    }

    /// El texto de la plantilla — **seguro por construcción, NO necesita redacción**.
    ///
    /// Lo que hay acá es `https://[user]:[password]@host/v1`: por REQ-A16c una credencial
    /// literal es error de configuración, así que la plantilla no puede contener un secreto.
    /// Pasarla por `redact_url` era redundante *y* mal tipado — tres call sites lo hacían.
    ///
    /// El que SÍ necesita redacción es [`ResolvedEndpoint`], que es el de después.
    pub fn as_str(&self) -> &str {
        unimplemented!()
    }
    /// El ÚNICO productor de [`ResolvedEndpoint`].
    pub fn resolve(
        &self,
        vault: &dyn SecretLookup,
        scope: Scope,
    ) -> Result<ResolvedEndpoint, EndpointError> {
        unimplemented!()
    }
}

/// Los tres endpoints del proceso, resueltos de una vez.
///
/// **Decisión de dueño (loop 29): la resolución ocurre UNA vez, en `main.rs`, inmediatamente
/// después de abrir el vault y ANTES de construir el trío o sondear.** No se enhebra el
/// vault dentro de `build_magi_orchestrator`: el builder queda puro (testeable sin vault) y
/// el punto de resolución es un paso nombrado del arranque, no un efecto escondido en un
/// constructor. Es la misma forma que ya tiene el desbloqueo de la DB — un paso, un dueño.
pub struct ResolvedEndpoints {
    pub root: ResolvedEndpoint,
    pub magi: ResolvedEndpoint,
    pub embedding: ResolvedEndpoint,
}

/// El paso de arranque. Falla CERRADO: un placeholder sin entrada de vault detiene el
/// proceso nombrando la entrada y el comando, nunca sustituye vacío (SC-A16f).
pub fn resolve_endpoints(
    cfg: &MagiConfigStub,
    vault: &dyn SecretLookup,
) -> Result<ResolvedEndpoints, EndpointError> {
    unimplemented!()
}

/// Redacción **por posición**, inmune a la codificación del contenido.
pub fn redact_url(raw: &str) -> String {
    unimplemented!()
}

/// Un mensaje de error ya redactado. **Solo [`redact_foreign_error`] lo construye.**
///
/// Convierte la convención en enforcement, que es lo que Caspar pidió: mientras la regla
/// fuera *"acordate de pasar por el redactor"*, un `map_err(|e| e.to_string())` de más la
/// rompía en silencio. Con este newtype, las firmas que aceptan texto de error **piden
/// `SafeErrorText`**, y el único camino a construirlo pasa por el redactor.
///
/// Es la misma disciplina que [`ResolvedEndpoint`]: no se prohíbe la operación peligrosa, se
/// vuelve inexpresable la forma equivocada.
pub struct SafeErrorText(String);

impl fmt::Debug for SafeErrorText {
    /// Necesario porque este tipo va dentro de `SeatError`/`TrioError`, que derivan `Debug`.
    ///
    /// **No es una defensa**: el contenido ya viene redactado —solo `redact_foreign_error` lo
    /// construye— así que un `derive` habria sido igual de seguro. Lo que protege el newtype
    /// es la CONSTRUCCION, no la impresion: no se puede fabricar un `SafeErrorText` desde un
    /// `to_string()` crudo, y las firmas que aceptan texto de error piden este tipo.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SafeErrorText({})", self.0)
    }
}

impl SafeErrorText {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// **Única puerta por la que un error FORÁNEO cruza hacia un mensaje nuestro.**
///
/// Cierra el `[CRITICAL]` del loop 31: la enumeración de cinco caminos cubría los `format!`
/// que escribimos, pero **no** el `to_string()` de un `ProviderError` de magi-core — que
/// construye su propio texto y puede incluir la URL con credenciales que le pasamos. Enumerar
/// nuestros formateadores no alcanza cuando el texto lo redacta otro crate.
///
/// Todo `map_err` que convierta un error de magi-core en `String` pasa por acá. La diferencia
/// con `redact_url` es el dominio: aquélla toma una URL, ésta un mensaje arbitrario donde una
/// URL puede estar embebida en cualquier posición.
pub fn redact_foreign_error(e: &dyn std::error::Error) -> SafeErrorText {
    unimplemented!()
}
// BARRIDO DE URLs EN PROSA — es trabajo INTERNO de `redact_foreign_error`, no una funcion
// aparte del contrato. Estuvo declarada como `redact_urls_in_prose` durante tres loops sin
// que ninguna tarea la implementara: un simbolo del contrato sin duena no lo escribe nadie.
//
// El problema que resuelve sigue en pie y por eso queda escrito aca: `redact_url` toma una
// URL ENTERA y localiza su autoridad por posicion; un error foraneo es PROSA con una URL
// embebida en un lugar cualquiera (`"error sending request for url (https://u:p@host/v1)"`),
// asi que la regla posicional no aplica sin antes ENCONTRAR la URL. El barrido es por
// `scheme://` hasta el primer caracter que no puede pertenecer a una URL, y sobre cada tramo
// hallado se aplica la MISMA regla posicional — no una segunda implementacion.
//
// Es best-effort por naturaleza y se declara: un error que codifique la URL de otra forma se
// le escapa. Por eso la defensa primaria sigue siendo REQ-A16c —el secreto no vive en el
// archivo— y esto es la segunda linea, no la unica.

/// Config del sistema, reducida a lo que este contrato cruza.
pub struct MagiConfigStub;

impl MagiConfigStub {
    /// **Devuelve `EndpointTemplate`, no `&str`** — lo que hay en el archivo es la plantilla.
    ///
    /// Este es el cambio que hace estructural a REQ-A16c: mientras el resolutor devolviera
    /// `&str`, nada obligaba a sustituir antes de llegar a un provider.
    pub fn effective_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        unimplemented!()
    }
    /// Ídem para el trío (`[magi].base_url`) y el embedder.
    pub fn effective_magi_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        unimplemented!()
    }
    pub fn effective_embedding_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        unimplemented!()
    }
    pub fn effective_magi_kind(&self) -> ProviderKind {
        unimplemented!()
    }
    /// `true` si `[magi].base_url` o `[magi].kind` están DECLARADOS — la divergencia se
    /// decide sobre lo declarado, no comparando URLs resueltas.
    pub fn magi_endpoint_diverges(&self) -> bool {
        unimplemented!()
    }
    /// Backend del SISTEMA: `provider` de raiz, o el default built-in.
    ///
    /// Faltaba en el contrato aunque el plan lo llama en cuatro lugares — o sea que su firma
    /// no estaba verificada por nadie. Es el nivel del que `effective_magi_kind` HEREDA.
    pub fn effective_provider(&self) -> ProviderKind {
        unimplemented!()
    }
    /// Cap de ENTRADA de magi-rs, previo al de magi-core (REQ-A11b). Rechaza, nunca trunca.
    ///
    /// Mismo caso que el de arriba: usado por el plan, ausente del contrato.
    pub fn effective_max_query_bytes(&self) -> usize {
        unimplemented!()
    }
    pub fn effective_default_mode(&self) -> Option<Mode> {
        unimplemented!()
    }
    pub fn untrusted_content(&self) -> bool {
        unimplemented!()
    }
    pub fn effective_agent_timeout_secs(&self) -> u64 {
        unimplemented!()
    }
    /// `tool_result_cap_bytes` **de RAÍZ** (subió de `[headless]` en este release, tercer
    /// patrón de migración). Este accessor ES su sitio de declaración en el contrato.
    pub fn effective_tool_result_cap(&self) -> usize {
        unimplemented!()
    }
}

/// Credenciales resueltas `env > vault` (REQ-A12). Reducidas a lo que este contrato cruza.
///
/// Aparte del endpoint y no redundante con él: `ResolvedEndpoint` puede traer `userinfo`
/// (autenticación del proxy o del servidor), mientras que la API key va en un header —
/// `Authorization: Bearer` u `x-api-key`. Dos credenciales, dos destinos; `ollama` usa la
/// primera y ninguna de la segunda.
pub trait Credentials {
    fn openai(&self) -> Option<String>;
    fn anthropic(&self) -> Option<String>;
}

/// Por qué falló un asiento — **tipado, no `String`** (Melchior, loop 32).
///
/// `SeatError` nombra la causa y el plan la consume para reportar los TRES asientos caídos
/// de una vez (REQ-A05b). Con `String` esa distinción se pierde en el borde y el llamador
/// tiene que parsear texto para saber si faltó una credencial o falló el transporte.
#[derive(Debug)]
pub enum SeatError {
    MissingCredential {
        var: &'static str,
    },
    /// Un status HTTP que el asiento devolvio. **Variante propia, no un `Transport` con el
    /// codigo adentro del texto**: `explain_keyless_auth_failure` necesita comparar 401/403
    /// contra el kind, y hacerlo parseando un mensaje seria exactamente el motivo por el que
    /// estos errores dejaron de ser `String`.
    Http {
        status: u16,
    },
    Transport(SafeErrorText),
}

/// Por qué no se pudo construir el trío. Ver [`SeatError`].
#[derive(Debug)]
pub enum TrioError {
    UnknownKind(String),
    /// Ningun asiento declarado. Distinto de `SeatUnbuildable`: aca no fallo ninguno,
    /// simplemente no habia ninguno que construir.
    NoSeats,
    /// TODOS los asientos que fallaron, no el primero: los tres comparten credencial y
    /// endpoint, así que reportar de a uno obliga a tres arranques (REQ-A05b).
    SeatUnbuildable {
        seats: Vec<(AgentName, SeatError)>,
    },
    Builder(SafeErrorText),
}

/// Construye UN provider nativo. **Exige el endpoint RESUELTO.**
pub fn build_native_provider(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    model: &str,
    creds: Option<&dyn Credentials>,
    client_timeout: Duration,
    notices: &mut Vec<Notice>,
) -> Result<Arc<dyn LlmProvider>, SeatError> {
    // El `push` NO es relleno: `&mut Vec` es el contrato porque esta funcion ACUMULA (la
    // normalizacion `/v1` de Ollama sale por aca, REQ-A01b), y un `&mut [Notice]` no puede
    // crecer. Con el cuerpo en `unimplemented!()` a secas, `clippy::ptr_arg` no puede ver eso
    // y propone justamente el cambio que rompe el contrato. Un cuerpo representativo cierra
    // el falso positivo sin un `#[allow]` que despues nadie sepa por que esta.
    notices.push(Notice::info("stub"));
    unimplemented!()
}

/// Construye el trío (Fase 4). Toma los endpoints YA resueltos — el builder no conoce el
/// vault, ver [`ResolvedEndpoints`].
pub fn build_magi_orchestrator(
    cfg: &MagiConfigStub,
    endpoints: &ResolvedEndpoints,
    creds: Option<&dyn Credentials>,
    warn_tokens: Option<usize>,
    notices: &mut Vec<Notice>,
) -> Result<Arc<Magi>, TrioError> {
    // Mismo motivo que en `build_native_provider`: acumula, no reescribe en su lugar.
    notices.push(Notice::info("stub"));
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Probe (REQ-A24)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Measurement {
    Measured {
        window: usize,
        digest: Option<String>,
    },
    NotMeasurable,
    NotMeasuredThisTime,
}

/// Tres estados, no dos: un kind no medible y una sonda que no se pudo construir tienen
/// consecuencias distintas, y colapsarlos afirmaba algo falso sobre el servidor.
pub enum ProbeSeat {
    Ready(Arc<dyn ProviderProbe>),
    NotProbeable,
    Unbuildable(String),
}

/// La costura de inyección: sin ella, todo test del probe necesitaba red real (contra R-A04).
pub trait ProbeFactory: Send + Sync {
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat;
}

/// **Dedup por par (endpoint, modelo): pedir `["a","a","b"]` devuelve DOS entradas.** El
/// mapa colapsa duplicados por construcción — un test que afirme una entrada por modelo
/// PEDIDO afirma una cardinalidad que este tipo no puede producir (hallazgo del loop 29).
pub async fn probe_models(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    models: &[&str],
    factory: &dyn ProbeFactory,
) -> BTreeMap<String, Measurement> {
    unimplemented!()
}

/// Orquesta las sondas del principal y del trío (Fase 5, Step 2a): una tanda si comparten
/// endpoint y kind, dos en `join!` si divergen — y la tabla del trío se RE-PROYECTA para que
/// la ventana del principal jamás contamine `derive_warn_tokens` (SC-A24j).
pub async fn orchestrate_probes(
    cfg: &MagiConfigStub,
    endpoints: &ResolvedEndpoints,
    backend_model: &str,
    trio_models: &[&str],
    factory: &dyn ProbeFactory,
) -> (Option<Measurement>, BTreeMap<String, Measurement>) {
    unimplemented!()
}

/// Del MÍNIMO de las ventanas de los MAGES, no del principal: el payload va a ellos.
pub fn derive_warn_tokens(mages: &BTreeMap<String, Measurement>) -> Option<usize> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Escala de timeouts (REQ-A04) y recorte de salida (REQ-A11b)
// ---------------------------------------------------------------------------

pub const AGENT_TIMEOUT_SECS: u64 = 90;
pub const AGENT_TIMEOUT_MIN_SECS: u64 = 30;
pub const AGENT_TIMEOUT_MAX_SECS: u64 = 120;
pub const CLASSIFY_TIMEOUT_SECS: u64 = 6;
pub const PROBE_TIMEOUT_SECS: u64 = 5;
pub const MAX_QUERY_BYTES: usize = 256 * 1024;
pub const TOOL_RESULT_CAP_BYTES: usize = 64 * 1024;

pub fn derive_operation_budget(ceiling_secs: u64) -> Duration {
    unimplemented!()
}
pub fn derive_client_timeout(ceiling_secs: u64) -> Duration {
    unimplemented!()
}
/// Runtime, NO `const`: se deriva del techo **configurado**, no del default built-in.
pub fn headless_consult_timeout_secs(configured_ceiling: u64) -> u64 {
    unimplemented!()
}

pub struct TimeoutDecision {
    pub effective_secs: u64,
    pub warning: Option<String>,
    pub below_formula: bool,
}

pub fn resolve_run_timeout(asked: Option<u64>, configured_ceiling: u64) -> TimeoutDecision {
    unimplemented!()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationLevel {
    None,
    Structural,
    Anchored,
    Bytes,
}

/// Struct y no tupla: los dos campos se leen por nombre en todas las rutas de salida.
pub struct Truncated {
    pub text: String,
    pub level: TruncationLevel,
}

pub fn truncate_report(report: &str, cap: usize) -> Truncated {
    unimplemented!()
}
/// Texto de la marca de recorte. **Fase 0**, junto a `mark_overhead` que la mide.
pub const TRUNCATION_MARK: &str = "[reporte recortado por límite de tamaño]";

/// Bytes que la marca agrega. **Fase 0**: `validate_output_cap` lo consume en Fase 1, y una
/// constante definida en Fase 6 dejaba a la Fase 1 sin compilar.
pub fn mark_overhead() -> usize {
    unimplemented!()
}
/// Por debajo de esto ni la marca entra, y el recorte deja de aplicarse **en silencio**.
pub fn min_viable_output_cap() -> usize {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Salida por corrida (Fase 6) — el seam que la v1 de wiring() no cruzaba
// ---------------------------------------------------------------------------

/// Señales por corrida que viajan en el JSON de ESA corrida, no en un notice (REQ-A11d).
pub struct RunContext {
    pub endpoint_divergence: bool,
    pub timeout_below_formula: bool,
}

impl RunContext {
    /// Toma la [`ModeResolution`] entera: `classification_attempted` es la señal correcta
    /// para la divergencia — `ModeSource::Inferred` daba falso negativo cuando la
    /// clasificación se intentó y falló, que es la corrida donde declarar la ruta de datos
    /// más importa.
    pub fn build(cfg: &MagiConfigStub, res: &ModeResolution, t: &TimeoutDecision) -> Self {
        unimplemented!()
    }
}

/// Error tipado del caso "los tres fallaron": NUNCA un reporte vacío marcado `degraded`.
#[derive(Debug)]
pub struct AllMagesFailed {
    pub causes: String,
}

/// Predicado sobre `agents.is_empty()` — "cero veredictos válidos", sin importar la vía
/// (fallo de ejecución O de extracción). Su call site es el embudo del consult, ANTES de
/// renderizar nada.
pub fn guard_all_failed(report: &MagiReport) -> Result<(), AllMagesFailed> {
    unimplemented!()
}

/// Por qué la entrada de un consult no es aceptable.
///
/// **Reemplaza a `QueryTooLarge`, que era un struct de un solo caso.** El plan ya rechazaba
/// también la consulta vacía y nombraba `ConsultInputError` en cuatro lugares sin que el tipo
/// existiera en ningún lado — un símbolo referenciado y nunca definido, que es exactamente lo
/// que este archivo existe para que no pase.
#[derive(Debug)]
pub enum ConsultInputError {
    /// Consulta vacía o con solo espacios. Llamar a tres modelos con esto es gasto puro.
    Empty,
    /// Por encima del cap de magi-rs. **Rechaza, NUNCA trunca** (REQ-A11b): un payload
    /// recortado en silencio produce un veredicto indistinguible de uno legítimo.
    TooLarge {
        /// Tamaño recibido, en bytes.
        size: usize,
        /// El cap configurado, para que el mensaje diga por cuánto se pasó.
        cap: usize,
    },
}

/// Rechaza, NUNCA trunca (REQ-A11b). El cap único de las tres rutas de entrada.
///
/// # Errors
/// [`ConsultInputError::Empty`] si la consulta es vacía o solo espacios;
/// [`ConsultInputError::TooLarge`] con el tamaño y el límite si supera el cap.
pub fn check_query_size(query: &str, cap: usize) -> Result<(), ConsultInputError> {
    unimplemented!()
}

/// Fuente única de la forma del JSON (REQ-A11c). Toma el **`Truncated`**, no el reporte más
/// un nivel suelto: texto y nivel viajan juntos o el `report_truncated` del JSON puede
/// mentir sobre el texto que acompaña.
pub fn report_to_consult_json(
    report: &MagiReport,
    truncated: &Truncated,
    res: &ModeResolution,
    ctx: &RunContext,
) -> serde_json::Value {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Notices (Task 1.5)
// ---------------------------------------------------------------------------

/// El orden del enum ES el orden de impresión, así que el `Ord` no es decorativo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NoticeTier {
    Blocking,
    Resolution,
    Info,
}

/// **Toda fuente empuja `Notice`, no `String`** — cuatro tareas empujaban `String` planos
/// mientras la dueña declaraba tiers, o sea que el orden no podía aplicarse a nada.
pub struct Notice {
    pub tier: NoticeTier,
    pub text: String,
}

impl Notice {
    pub fn blocking(text: impl Into<String>) -> Self {
        unimplemented!()
    }
    pub fn resolution(text: impl Into<String>) -> Self {
        unimplemented!()
    }
    pub fn info(text: impl Into<String>) -> Self {
        unimplemented!()
    }
}

pub const NOTICE_MAX_INFO: usize = 5;

pub fn render_notices(notices: Vec<Notice>) -> Vec<String> {
    unimplemented!()
}

/// Aviso de ARRANQUE: dispara cuando la inferencia está ACTIVA (sin `default_mode`), sobre
/// lo declarado. La señal POR CORRIDA es `RunContext.endpoint_divergence`, que sí consume
/// `classification_attempted` — dos momentos, dos señales, y colapsarlas fue un hallazgo.
pub fn divergence_notice(cfg: &MagiConfigStub, inference_active: bool) -> Option<Notice> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// ConsultTool — lo que su schema y su execute exigen que reciba
// ---------------------------------------------------------------------------

/// Lo que `ConsultTool::new` RECIBE, porque su `input_schema` y su `execute` lo consumen.
///
/// El `[CRITICAL]` del loop 29: el plan especificaba que el schema omite `mode` con
/// `untrusted_content` activo, pero `new` nunca recibía la marca — el schema no podía saber
/// qué omitir. `default_mode` NO está acá: el embudo resuelve e inyecta, el tool no
/// re-resuelve.
pub struct ConsultToolCfg {
    // `untrusted_content` YA NO va acá: el `input_schema` es fijo — siempre ofrece `mode`,
    // porque el agente debe poder elegir la lente (SC-A07d). La marca la aplica el resolutor
    // del embudo, no la forma del schema.
    pub max_query_bytes: usize,
    // `gate_thresholds` NO va acá (loop 31): el gate se evalúa en el embudo del agente, que
    // es lo que distingue la ruta autónoma de la explícita (REQ-A20). Un segundo ejemplar
    // dentro del tool era una fuente de verdad sin consumidor — y con el riesgo de divergir
    // del que el embudo aplica, que es el que decide.
}

pub struct ConsultTool {
    magi: Arc<Magi>,
    auto_approve: bool,
    cfg: ConsultToolCfg,
}

impl ConsultTool {
    pub fn new(magi: Arc<Magi>, auto_approve: bool, cfg: ConsultToolCfg) -> Self {
        unimplemented!()
    }
    /// Schema FIJO: siempre ofrece `mode`. Ver [`agent_chosen_mode`] para por qué eso no
    /// reabre el bypass — lo cierra el nivel `AgentChosen`, no la ausencia del campo.
    pub fn input_schema(&self) -> serde_json::Value {
        unimplemented!()
    }
}

// ===========================================================================
// 3. EL CRUCE — cada firma llamada DESDE EL CONSUMIDOR QUE EL PLAN DECLARA
//
// La v1 de esta función declaraba contratos y los llamaba entre sí de forma genérica; la
// deriva sobrevivió en los seams que no cruzaba. Esta versión sigue el ARRANQUE REAL que el
// plan describe, paso a paso, para que cada consumidor declarado quede bajo el compilador.
// ===========================================================================

/// La plantilla que se muestra en un notice — sin redactar, porque no puede tener secreto.
fn tpl_for_display() -> EndpointTemplate {
    unimplemented!()
}

async fn wiring(
    vault: &dyn SecretLookup,
    factory: &dyn ProbeFactory,
    classifier: &dyn ModeClassifier,
    telemetry: &dyn GateTelemetry,
    creds: &dyn Credentials,
    report: &MagiReport,
) -> Result<(), ModeError> {
    let cfg = MagiConfigStub;
    let mut notices: Vec<Notice> = Vec::new();

    // ── ARRANQUE (main.rs) ────────────────────────────────────────────────
    // (1) Vault abierto → resolver los TRES endpoints, una vez, fail-closed.
    let endpoints: ResolvedEndpoints =
        resolve_endpoints(&cfg, vault).expect("placeholder sin entrada ⇒ error, no vacío");

    // (2) Probe ANTES del trío: su salida (`warn_tokens`) es entrada del builder.
    let (principal, trio) = orchestrate_probes(
        &cfg,
        &endpoints,
        "backend-model",
        &["m1", "m2", "m3"],
        factory,
    )
    .await;
    let warn_tokens: Option<usize> = derive_warn_tokens(&trio);
    let _: Option<Measurement> = principal;

    // (3) El trío, con endpoints RESUELTOS — el builder no conoce el vault.
    let magi: Arc<Magi> =
        build_magi_orchestrator(&cfg, &endpoints, Some(creds), warn_tokens, &mut notices)
            .expect("asientos construibles");

    // (3b) Y un asiento suelto, por la misma puerta tipada.
    let _ = build_native_provider(
        cfg.effective_magi_kind(),
        &endpoints.magi,
        "modelo",
        Some(creds),
        derive_client_timeout(cfg.effective_agent_timeout_secs()),
        &mut notices,
    );

    // (4) El tool se registra con lo que su schema y su execute consumen.
    let tool = ConsultTool::new(
        Arc::clone(&magi),
        false,
        ConsultToolCfg {
            max_query_bytes: MAX_QUERY_BYTES,
        },
    );
    let _schema = tool.input_schema();

    // (4b) La TUI arma SUS tres piezas — el sitio que faltaba nombrar. Sin esto, la
    //      superficie de mayor tráfico corría con built-ins ignorando el magi.toml.
    let parts = tui_run_config_parts(&cfg);
    let _: GateThresholds = parts.gate_thresholds;
    let _: ModeConfig = parts.mode_config;
    let _: Arc<dyn GateTelemetry> = parts.gate_telemetry;

    // ── UNA CORRIDA (embudo del agente) ───────────────────────────────────
    // (5) La elección del agente entra por SU PROPIO parámetro, no por `explicit`: es lo que
    //     cierra el bypass sin matar SC-A07d. El schema sigue ofreciendo `mode` siempre.
    let incoming = serde_json::json!({ "query": "contenido", "mode": "design" });
    let chosen = agent_chosen_mode(&incoming);

    // (5b) UNA resolución, con la guarda y la clasificación ADENTRO.
    let res: ModeResolution = resolve_mode_guarded(
        None, // explicito HUMANO: no lo hay en la ruta autonoma
        cfg.effective_default_mode(),
        chosen,
        cfg.untrusted_content(),
        Some(classifier),
        "contenido",
    )
    .await?;

    // (6) La resolución cruza el trait Tool por el input — en el despacho del modelo Y en el
    //     forced-consult. Leerla es Result: sin fallback a `input["mode"]`.
    // El bucle presta la respuesta inmutable: se clona el input y se inyecta sobre la copia.
    let borrowed = serde_json::json!({ "query": "contenido" });
    let mut tool_input = input_for_dispatch(&borrowed, &res);
    inject_resolved_mode(&mut tool_input, &res);
    let _pair: Result<(Mode, ModeSource), ModeInjectionMissing> = read_resolved_mode(&tool_input);

    // (7) El gate usa el MISMO modo, y su telemetría lleva el umbral aplicado.
    let thresholds = GateThresholds::from_overrides(GateOverrides {
        analysis: Some(300),
        ..GateOverrides::default()
    });
    let verdict = evaluate("contenido", &res.mode, &thresholds);
    telemetry.on_gate_evaluation(
        &res.mode,
        "contenido".chars().count(),
        thresholds.for_mode(&res.mode),
        matches!(verdict, GateVerdict::Veto { .. }),
    );

    // (8) Entrada acotada, escala derivada del techo CONFIGURADO.
    check_query_size("contenido", MAX_QUERY_BYTES).expect("bajo el cap");
    let ceiling = cfg.effective_agent_timeout_secs();
    let budget = derive_operation_budget(ceiling);
    let client = derive_client_timeout(ceiling);
    debug_assert!(budget + client <= Duration::from_secs(ceiling), "REQ-A04");
    let decision = resolve_run_timeout(None, ceiling);
    let _min = headless_consult_timeout_secs(ceiling);

    // (9) Salida: guard de cero veredictos → recorte → JSON, texto y nivel JUNTOS.
    guard_all_failed(report).expect("al menos un veredicto válido");
    let cap = cfg.effective_tool_result_cap().max(min_viable_output_cap());
    debug_assert!(
        cap > mark_overhead(),
        "por debajo, el recorte no aplica NADA"
    );
    let truncated: Truncated = truncate_report(&report.report, cap);
    let ctx = RunContext::build(&cfg, &res, &decision);
    let _json: serde_json::Value = report_to_consult_json(report, &truncated, &res, &ctx);

    // (10) Notices: toda fuente empuja `Notice`, la dueña ordena y recorta.
    if let Some(n) = divergence_notice(&cfg, cfg.effective_default_mode().is_none()) {
        notices.push(n);
    }
    // El RESUELTO se redacta; la PLANTILLA no lo necesita (no puede tener un secreto).
    notices.push(Notice::resolution(redact_url(endpoints.root.as_str())));
    notices.push(Notice::info(tpl_for_display().as_str().to_string()));
    // Un error FORÁNEO cruzando a texto nuestro pasa por su propia puerta: `to_string()` de
    // un `ProviderError` arma su mensaje y puede llevar la URL que le dimos. El newtype hace
    // que la firma NO acepte un `String` crudo.
    let foreign: &dyn std::error::Error = &std::fmt::Error;
    let safe: SafeErrorText = redact_foreign_error(foreign);
    notices.push(Notice::info(safe.as_str().to_string()));
    let _rendered = render_notices(notices);

    // (11) Y la superficie de magi-core que todo lo anterior asume.
    core_enums();
    core_type_properties();
    core_constructors();
    core_report_shape(report);
    let _ = RetryProvider::with_config(
        Arc::new(
            magi_core::providers::openai_compat::OpenAiCompatibleProvider::new(
                "http://host/v1",
                "m",
                None,
            )
            .expect("url válida"),
        ) as Arc<dyn LlmProvider>,
        core_retry_config(),
    );

    Ok(())
}

fn main() {
    // No ejecuta nada: el valor de este archivo es que TIPE.
    println!("ms2_contracts: si esto compiló, las firmas del plan son consistentes entre sí");
}
