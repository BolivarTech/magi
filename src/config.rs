// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-25

//! Persistent magi-rs configuration from `magi.toml`. NON-SECRET only — API keys
//! never live here (env/keyring/key.txt).

// Public API of this module is consumed by `main.rs` (Task 6 wiring) and by
// tests; no items here should be flagged dead_code under any cfg.

mod migrate;

use std::path::Path;

use magi_core::schema::{AgentName, Mode};
use magi_rs::magi::endpoint::{EndpointError, EndpointTemplate};
use magi_rs::magi::gate::{GateOverrides, GateThresholds};
use magi_rs::magi::kind::{ProviderKind, ProviderKindParseError};
use magi_rs::magi::mode::{ModeExt, ModeParseError};
use magi_rs::magi::{min_viable_output_cap, AGENT_TIMEOUT_MAX_SECS, AGENT_TIMEOUT_MIN_SECS};
use serde::Deserialize;

/// Errores de configuración de `magi.toml` (Task 1.1, REQ-A01b/A04/A11b/A21b).
///
/// Vive en el **bin** (no en `magi_rs::magi`) porque es específico de la FORMA del TOML;
/// los tipos de error de vocabulario puro (`ProviderKindParseError`, `ModeParseError`)
/// viven en el lib y se absorben acá con `From`, que es la dirección correcta de la
/// dependencia (el lib no puede conocer un tipo del bin).
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// `provider` o `[magi].kind` traen un valor presente y no reconocido.
    #[error("provider desconocido: {got:?} (válidos: {valid})")]
    UnknownProviderKind {
        /// Lo que trajo el archivo.
        got: String,
        /// Los tres valores aceptados, para que el error sea accionable.
        valid: &'static str,
    },

    /// `[magi].default_mode` trae un valor presente y no reconocido.
    #[error("modo desconocido: {got:?} (válidos: {valid})")]
    UnknownMode {
        /// Lo que trajo el archivo.
        got: String,
        /// Las tres etiquetas aceptadas, para que el error sea accionable.
        valid: &'static str,
    },

    /// El archivo trae patrones de migración de v0.11.0 (REQ-A21b). El texto ya viene
    /// renderizado por [`migrate::render_migration_error`] — nombra cada incompatibilidad
    /// y su corrección.
    #[error("{0}")]
    NeedsMigration(String),

    /// El TOML no parsea, o el archivo no se pudo leer.
    #[error("{0}")]
    Parse(String),

    /// `[magi].agent_timeout_secs` cae fuera del rango admisible de §4.9.
    #[error(
        "agent_timeout_secs = {got} fuera de rango [{min}, {max}]: por debajo de {min}s no \
         entra una generación legítima; por encima de {max}s el peor caso de un consult \
         (2 intentos por mage) supera los 4 minutos. No se recorta al extremo — se rechaza."
    )]
    AgentTimeoutOutOfRange {
        /// El valor declarado.
        got: u64,
        /// Piso del rango admisible (§4.9).
        min: u64,
        /// Techo del rango admisible (§4.9).
        max: u64,
    },

    /// `tool_result_cap_bytes` cae por debajo del mínimo viable (REQ-A11b).
    #[error(
        "tool_result_cap_bytes = {got} es menor que el mínimo viable ({min}): por debajo de \
         ese umbral ni la marca de recorte entra, y el cap configurado se ignora en \
         silencio en vez de aplicarse."
    )]
    OutputCapTooSmall {
        /// El valor declarado.
        got: usize,
        /// El mínimo viable ([`magi_rs::magi::min_viable_output_cap`]).
        min: usize,
    },

    /// Una `base_url` (raíz, `[magi]` o `[embedding]`) no es una plantilla válida:
    /// trae una credencial literal en vez de los placeholders `[user]:[password]`, un
    /// placeholder desconocido, o no se pudo recorrer (REQ-A16c, SC-A16d).
    ///
    /// El texto nunca repite el valor ofensor — lo garantiza el `Display` de
    /// [`EndpointError`] en el que se apoya, no este `#[error]`.
    #[error("{0}")]
    Endpoint(#[from] EndpointError),
}

impl From<ProviderKindParseError> for ConfigError {
    fn from(e: ProviderKindParseError) -> Self {
        Self::UnknownProviderKind {
            got: e.got,
            valid: e.valid,
        }
    }
}

impl From<ModeParseError> for ConfigError {
    fn from(e: ModeParseError) -> Self {
        match e {
            ModeParseError::Unknown { got, valid } => Self::UnknownMode { got, valid },
        }
    }
}

/// Extrae un mensaje de error de parseo **sin la línea ofensora** (B11/REQ-A16).
///
/// **`toml::de::Error`'s `Display` reproduce el TEXTO CRUDO del archivo alrededor del
/// error** — para `api_key = "sk-secreto"\n` (rechazado por `deny_unknown_fields`, REQ-A14)
/// el `Display` completo es:
///
/// ```text
/// TOML parse error at line 1, column 1
///   |
/// 1 | api_key = "sk-secreto"
///   | ^^^^^^^
/// unknown field `api_key`, expected `provider`
/// ```
///
/// — el secreto que se está rechazando queda impreso en el propio mensaje de error. Esto
/// se descubrió corriendo `an_api_key_anywhere_in_the_toml_is_a_parse_error`: el
/// `.map_err(|e| ConfigError::Parse(e.to_string()))?` original (calcado del cuerpo que
/// paso a paso pega el brief de esta tarea) usaba el `Display` completo, y el test falló
/// contra el propio código que la spec pidió transcribir — B9/REQ-A00c exige rechazar eso,
/// no transcribirlo. `toml::de::Error::message()` da el mensaje semántico SIN el extracto
/// de la fuente; para el mismo caso es solo `"unknown field \`api_key\`, expected
/// \`provider\`"` — el NOMBRE del campo rechazado, nunca su valor, porque
/// `deny_unknown_fields` rechaza la clave antes de mirar el tipo del valor.
///
/// **La posición SE RECUPERA (fix round 2, I4) — solo el extracto quedaba prohibido.**
/// `message()` también descarta línea/columna, y SC-A21g exige que un error de
/// sintaxis nombre una posición. `toml::de::Error::span()` da el rango de BYTES del
/// error sin ningún contenido — se recorre `raw` contando saltos de línea hasta ese
/// byte (nunca se rebana ni se imprime `raw`), así que la posición no puede
/// reintroducir el leak que esta función existe para evitar.
///
/// Seguridad sigue siendo una de las cinco categorías que nunca se difieren como
/// residual, y una discrepancia de tipo en un campo NO-secreto (p. ej.
/// `agent_timeout_secs = "x"`) también podría, en principio, ecoar un valor en
/// `message()` — cerrar el camino de raíz (nunca mostrar el extracto de fuente) sigue
/// siendo más robusto que enumerar campos "seguros"; eso no cambia acá.
fn safe_parse_error(e: &toml::de::Error, raw: &str) -> String {
    let message = e.message();
    match e.span() {
        Some(span) => {
            let (line, column) = line_col_of(raw, span.start);
            format!("{message} (line {line}, column {column})")
        }
        None => message.to_string(),
    }
}

/// Posición 1-indexada `(línea, columna)` del byte `offset` en `raw`.
///
/// **Nunca devuelve nada de lo que hay EN `raw`** — solo cuenta caracteres y saltos
/// de línea hasta `offset`, así que no puede reintroducir el extracto de fuente que
/// [`safe_parse_error`] existe para suprimir. Sin `indexing_slicing`/`string_slice`:
/// recorre por `char_indices()`, nunca rebana `raw` por posición.
fn line_col_of(raw: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut column = 1usize;
    for (idx, ch) in raw.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiConfig {
    pub provider: Option<String>,
    /// Endpoint por defecto DEL SISTEMA (REQ-A21): lo usan el agente principal, el trío y
    /// el embedder salvo que su propia sección lo overridee. Ausente ⇒ el built-in.
    /// **BREAKING**: hasta v0.11.0 esta clave vivía en `[openai].base_url`, que ya no
    /// existe — ver [`ConfigError::NeedsMigration`].
    pub base_url: Option<String>,
    /// Cap de SALIDA del reporte, en las TRES rutas (TUI, `magi query`, consult headless
    /// — REQ-A11b). Ausente ⇒ [`magi_rs::magi::TOOL_RESULT_CAP_BYTES`].
    pub tool_result_cap_bytes: Option<usize>,
    #[serde(default)]
    pub openai: OpenAiConfig,
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub magi: MagiSectionConfig,
    #[serde(default)]
    pub memory: crate::memory::config::MemoryConfig,
    #[serde(default)]
    pub embedding: crate::memory::config::EmbeddingConfig,
    #[serde(default)]
    pub headless: HeadlessConfig,
}

/// `[headless]` section of `magi.toml` (spec §11). Every field is optional; an
/// unset field falls back to its built-in constant default (see
/// `main.rs::resolve_headless_limits`). Unknown keys are a parse error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadlessConfig {
    /// Cap on `-i`/stdin input bytes (REQ-H29). Overrides `MAX_INPUT_BYTES`.
    pub max_input_bytes: Option<usize>,
    /// Elevated tool-call cap under `--full-auto` (REQ-H08). Overrides `FULL_AUTO_MAX_TOOL_CALLS`.
    pub full_auto_max_tool_calls: Option<u32>,
    /// Keep at most the last N run logs (REQ-H34). Overrides `LOG_RETENTION_RUNS`.
    pub log_retention: Option<usize>,
    /// Total log-dir byte ceiling (REQ-H24). Overrides `LOG_MAX_BYTES`.
    pub log_max_bytes: Option<u64>,
    // `tool_result_cap_bytes` YA NO VIVE ACÁ (Task 1.3, tercer patrón de migración de
    // REQ-A21b): subió al nivel raíz porque bajo `[headless]` cubría el modo por lotes y
    // dejaba suelto el interactivo, que es justo donde el reporte se re-envía en cada turno
    // de una sesión larga. Un cap que protege el caso barato y no el caro protege el caso
    // equivocado. Un archivo que todavía la declare acá recibe el error de migración guiado,
    // no un `unknown field` pelado — ver `detect_migrations`.
    /// Default log level (REQ-H24): `error`|`warn`|`info`|`debug`. Overrides `"info"`.
    pub log_level: Option<String>,
    /// Default wall-clock timeout secs for tool-executing tiers (REQ-H36).
    /// Overrides `FULL_AUTO_TIMEOUT_SECS`.
    pub timeout_secs: Option<u64>,
    /// Whether the envelope may override the operator `system` prompt (REQ-H12b).
    /// Defaults to `false` (the envelope `system` is ignored unless enabled).
    pub allow_system_override: Option<bool>,
}

/// `[openai]` section. Shared by `provider = "ollama"` and `provider = "openai-compat"` —
/// the two providers speak the same Chat-Completions transport and are distinguished only
/// by capability (only `ollama` is probeable, REQ-A24), not by config shape.
///
/// **`base_url` no longer lives here** — it moved to the root of `MagiConfig` (REQ-A21).
/// A `magi.toml` that still declares `[openai].base_url` fails to parse
/// (`deny_unknown_fields`); see [`ConfigError::NeedsMigration`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiConfig {
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConfig {
    pub model: Option<String>,
}

/// Sub-tabla `[magi.complexity]`. Ausente ⇒ built-ins; un modo en `0` ⇒ ese modo nunca se
/// veta; un modo ausente dentro de una tabla presente ⇒ su built-in, **no cero** (REQ-A20b).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexityConfig {
    /// Largo mínimo (caracteres) para despachar un consult autorruteado en `CodeReview`.
    /// Ausente ⇒ el built-in de la librería.
    pub code_review: Option<usize>,
    /// Ver [`Self::code_review`], para `Design`.
    pub design: Option<usize>,
    /// Ver [`Self::code_review`], para `Analysis`. **No hereda el "no vacío" del ejemplo
    /// de magi-core** (REQ-A20): `Analysis` es el default de toda invocación sin modo, así
    /// que un umbral de 1 apagaría el gate en el camino autónomo más común.
    pub analysis: Option<usize>,
}

/// Tabla `[magi]`. Renombrada desde `MagiModelsConfig` porque ya no contiene solo modelos
/// — Task 1.1 le agrega el resto del vocabulario del trío (kind, endpoint, modo, gate de
/// complejidad, timeouts) que la spec de MS2 le atribuye a esta sección.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiSectionConfig {
    /// Override model for Melchior (the Scientist). `None` ⇒ principal model.
    pub melchior_model: Option<String>,
    /// Override model for Balthasar (the Pragmatist). `None` ⇒ principal model.
    pub balthasar_model: Option<String>,
    /// Override model for Caspar (the Critic). `None` ⇒ principal model.
    pub caspar_model: Option<String>,
    /// Auto-approve autonomous MAGI (`consult`) launches when the main LLM
    /// self-routes to the `consult` tool in the agent tool loop. Default `false`
    /// — the agent asks before launching the 3-perspective consensus. `true`
    /// launches without asking, but announces it in the TUI (3 LLM calls take
    /// time). The explicit `/consult` TUI command is NEVER gated — it is always
    /// user-initiated and requires no approval regardless of this flag.
    #[serde(default = "default_auto_approve")]
    pub auto_approve: bool,

    /// Provider del trío; ausente ⇒ **hereda** el de raíz (REQ-A01b). No es una
    /// heurística: la herencia lee un valor declarado, no adivina uno observado.
    pub kind: Option<String>,
    /// Endpoint del trío; ausente ⇒ hereda el de raíz (REQ-A21).
    pub base_url: Option<String>,
    /// Modo fijo para toda invocación sin `--mode`; saltea la inferencia (REQ-A07).
    pub default_mode: Option<String>,
    /// Declara que el contenido bajo análisis NO es confiable (REQ-A07d). Con esto
    /// activo, omitir el modo es **error**, no inferencia.
    pub untrusted_content: Option<bool>,
    /// Cap de entrada DE MAGI-RS, previo a magi-core (REQ-A11b). Ausente ⇒
    /// [`magi_rs::magi::MAX_QUERY_BYTES`].
    pub max_query_bytes: Option<usize>,
    /// Techo por mage; las dos capas internas de reintento se derivan de él
    /// (REQ-A04/A15). Debe caer en `[AGENT_TIMEOUT_MIN_SECS, AGENT_TIMEOUT_MAX_SECS]`.
    pub agent_timeout_secs: Option<u64>,
    /// Umbral de aviso de tamaño; ausente ⇒ se mide por probe, o el default de
    /// magi-core (REQ-A15/A24b).
    pub input_warn_tokens: Option<usize>,
    /// Desactiva el reintento de transporte (REQ-A15).
    pub retry_disabled: Option<bool>,
    /// Umbrales del gate de complejidad por modo; ausente ⇒ built-ins (REQ-A20b).
    pub complexity: Option<ComplexityConfig>,
}

impl MagiSectionConfig {
    /// Los tres asientos con su modelo resuelto: el declarado, o el del backend.
    ///
    /// # Arguments
    /// * `fallback` - Modelo del backend, usado por cualquier asiento sin override.
    #[must_use]
    pub fn seats(&self, fallback: &str) -> Vec<(AgentName, String)> {
        vec![
            (
                AgentName::Melchior,
                self.melchior_model
                    .clone()
                    .unwrap_or_else(|| fallback.into()),
            ),
            (
                AgentName::Balthasar,
                self.balthasar_model
                    .clone()
                    .unwrap_or_else(|| fallback.into()),
            ),
            (
                AgentName::Caspar,
                self.caspar_model.clone().unwrap_or_else(|| fallback.into()),
            ),
        ]
    }

    /// Modelo del **fallback del builder** — el que magi-core usaría para un agente sin
    /// override.
    ///
    /// Es el del backend, no el de ningún mage: elegir el de Melchior lo volvería el
    /// default por accidente. Con los tres asientos overrideados nunca se usa, y por eso
    /// mismo conviene que sea una decisión escrita.
    ///
    /// # Arguments
    /// * `backend_model` - Modelo por defecto del backend resuelto.
    #[must_use]
    pub fn fallback_model<'a>(&self, backend_model: &'a str) -> &'a str {
        backend_model
    }
}

impl Default for MagiSectionConfig {
    fn default() -> Self {
        Self {
            melchior_model: None,
            balthasar_model: None,
            caspar_model: None,
            auto_approve: default_auto_approve(),
            kind: None,
            base_url: None,
            default_mode: None,
            untrusted_content: None,
            max_query_bytes: None,
            agent_timeout_secs: None,
            input_warn_tokens: None,
            retry_disabled: None,
            complexity: None,
        }
    }
}

/// Default value for [`MagiSectionConfig::auto_approve`]: `false` (require
/// explicit approval before each autonomous MAGI consensus launch).
fn default_auto_approve() -> bool {
    false
}

impl MagiConfig {
    /// Parsea un `magi.toml` desde texto, **validando el vocabulario** igual que
    /// [`Self::load`] (REQ-A01b).
    ///
    /// **Valida a propósito, aunque sea el helper que más usan los tests.** Un
    /// `from_toml_str` que deserializa sin validar dejaría que los tests construyan
    /// configuraciones que `load()` jamás aceptaría — la suite ejercitaría un camino que
    /// producción no tiene, la misma clase de brecha que un resolutor `.ok().flatten()`
    /// que se traga un valor inválido.
    ///
    /// # Errors
    /// [`ConfigError::NeedsMigration`] si el archivo trae patrones de v0.11.0 (stub: hasta
    /// Task 1.3 nunca ocurre — ver [`migrate`]); [`ConfigError::Parse`] si el TOML no
    /// parsea; [`ConfigError::UnknownProviderKind`] / [`ConfigError::UnknownMode`] si
    /// `provider`, `[magi].kind` o `[magi].default_mode` traen un valor presente y no
    /// reconocido; [`ConfigError::AgentTimeoutOutOfRange`] /
    /// [`ConfigError::OutputCapTooSmall`] si esos números caen fuera de su rango.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        // La pasada de migración va PRIMERO, igual que en `load()` (Task 1.4). Sin esto
        // los tests ejercitarían un camino de error que producción no tiene: un
        // `magi.toml` de v0.11.0 daría acá el `unknown field` pelado de serde, mientras
        // el usuario real recibe el mensaje guiado.
        let found = migrate::detect_migrations(s);
        if !found.is_empty() {
            return Err(ConfigError::NeedsMigration(
                migrate::render_migration_error(&found),
            ));
        }
        let cfg: Self =
            toml::from_str(s).map_err(|e| ConfigError::Parse(safe_parse_error(&e, s)))?;
        cfg.validate_vocabulary()?;
        Ok(cfg)
    }

    /// Valida TODO el vocabulario del archivo y sus rangos numéricos. Se llama desde
    /// [`Self::from_toml_str`] (y por lo tanto desde `load()`), **antes** de que cualquier
    /// `effective_*()` pueda tragarse un valor inválido y caer al default en silencio.
    ///
    /// # Errors
    /// Ver [`Self::from_toml_str`].
    ///
    /// **Por qué la validación va acá y no en los resolutores.** Un resolutor que hace
    /// `parse(s).ok().flatten().unwrap_or(default)` **se come el error**: `provider =
    /// "banana"` se convertiría en `Ollama` en silencio, que es exactamente el fallback
    /// silencioso que REQ-A01b prohíbe. Validando al cargar, para cuando los resolutores
    /// corren ya no queda nada inválido que tragarse.
    fn validate_vocabulary(&self) -> Result<(), ConfigError> {
        ProviderKind::parse(self.provider.as_deref().unwrap_or_default())?;
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())?;
        <Mode as ModeExt>::parse_config_value(
            self.magi.default_mode.as_deref().unwrap_or_default(),
        )?;
        self.validate_agent_timeout()?;
        self.validate_output_cap()?;
        Ok(())
    }

    /// `agent_timeout_secs` fuera del rango de §4.9 es **error de configuración**.
    ///
    /// # Errors
    /// [`ConfigError::AgentTimeoutOutOfRange`] con el valor, el rango y el porqué.
    ///
    /// **No se recorta al extremo, se rechaza** — mismo criterio que la ventana del probe
    /// (REQ-A16b): recortar convierte un valor que el operador escribió mal en uno
    /// plausible, y después el sistema se comporta distinto de lo que dice el archivo.
    ///
    /// Existe porque sin esto REQ-A04 sería **rompible desde `magi.toml`**: con un techo
    /// por debajo del piso absoluto de la derivación, los pisos internos ganan y la suma
    /// supera el techo. "Imposible por construcción" solo es cierto si el rango de entrada
    /// está acotado.
    fn validate_agent_timeout(&self) -> Result<(), ConfigError> {
        let Some(secs) = self.magi.agent_timeout_secs else {
            return Ok(()); // ausente ⇒ el default built-in, ya válido
        };
        if (AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS).contains(&secs) {
            return Ok(());
        }
        Err(ConfigError::AgentTimeoutOutOfRange {
            got: secs,
            min: AGENT_TIMEOUT_MIN_SECS,
            max: AGENT_TIMEOUT_MAX_SECS,
        })
    }

    /// Rechaza un cap de salida por debajo del mínimo viable (REQ-A11b).
    ///
    /// # Errors
    /// [`ConfigError::OutputCapTooSmall`] con el valor recibido y el mínimo.
    fn validate_output_cap(&self) -> Result<(), ConfigError> {
        let Some(cap) = self.tool_result_cap_bytes else {
            return Ok(());
        };
        let min = min_viable_output_cap();
        if cap < min {
            return Err(ConfigError::OutputCapTooSmall { got: cap, min });
        }
        Ok(())
    }

    /// Provider efectivo del agente principal: clave de raíz, o el default built-in.
    ///
    /// **Infalible por precondición:** [`Self::validate_vocabulary`] ya corrió en
    /// [`Self::from_toml_str`]/`load()`, así que el único `None` posible es el de
    /// ausente-o-vacío.
    ///
    /// Task 4.1: consumida en producción por `resolve_effective_provider_kind` (backend
    /// del agente principal) y por `build_magi_orchestrator`/`effective_magi_kind` (kind
    /// del trío, vía herencia). Cubierta por `blank_string_keys_are_absent_not_invalid` y
    /// `magi_kind_inherits_from_root_provider_when_absent`.
    #[must_use]
    pub fn effective_provider(&self) -> ProviderKind {
        // I5 (review round 2): restored. `MagiConfig`'s fields are `pub` and it
        // derives `Default`, so `MagiConfig { provider: Some("banana".into()),
        // ..Default::default() }` compiles and, without this, would silently
        // return `Ollama` — the precondition this function's own doc calls
        // "infalible por precondición" is exactly what this checks.
        debug_assert!(
            self.validate_vocabulary().is_ok(),
            "load() debe haber validado"
        );
        ProviderKind::parse(self.provider.as_deref().unwrap_or_default())
            .unwrap_or(None)
            .unwrap_or(ProviderKind::Ollama)
    }

    /// Modo declarado en `[magi].default_mode`, o `None` si está ausente/vacío (REQ-A15).
    ///
    /// **Devuelve `Option`, no `Result`, a propósito.** Con `Result`, cada llamador
    /// terminaría escribiendo `.ok().flatten()` y tragándose el `ConfigError` —
    /// convirtiendo un `default_mode = "banana"` en "no hay modo declarado, inferilo". Un
    /// valor inválido muere en `load()`/`from_toml_str()`; para cuando alguien llama acá,
    /// la única respuesta posible ya es "sí, este modo" o "no hay ninguno".
    ///
    /// Misma precondición que [`Self::effective_provider`].
    ///
    /// Consumed in production by `run_consult_subcommand` (`main.rs`, REQ-A15):
    /// the `Configured` level of `resolve_mode_guarded`'s five-level precedence.
    /// Covered by `effective_default_mode_follows_the_same_blank_is_absent_rule`.
    #[must_use]
    pub fn effective_default_mode(&self) -> Option<Mode> {
        // I5 (review round 2): restored — same precondition/rationale as
        // `effective_provider`'s `debug_assert!`.
        debug_assert!(
            self.validate_vocabulary().is_ok(),
            "load() debe haber validado"
        );
        <Mode as ModeExt>::parse_config_value(self.magi.default_mode.as_deref().unwrap_or_default())
            .unwrap_or(None)
    }

    /// `kind` del trío: declarado, o **heredado** del principal (REQ-A01b).
    ///
    /// La herencia NO es heurística: una heurística adivina a partir de un dato observado
    /// (p. ej. el puerto); la herencia lee un valor declarado. No hay nada que adivinar mal.
    ///
    /// Misma precondición que [`Self::effective_provider`].
    ///
    /// Task 4.1: consumida en producción por `build_magi_orchestrator` (`main.rs`),
    /// DESPUÉS de que éste valide por su cuenta el `kind` crudo — el `.unwrap_or(None)`
    /// de acá se traga un valor no reconocido igual que `effective_provider`, así que
    /// `build_magi_orchestrator` no puede depender de este accessor para reportar su
    /// error tipado de kind inválido; lo hace ANTES, con su propio `ProviderKind::parse`.
    /// Covered by `magi_kind_inherits_from_root_provider_when_absent`.
    #[must_use]
    pub fn effective_magi_kind(&self) -> ProviderKind {
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())
            .unwrap_or(None)
            .unwrap_or_else(|| self.effective_provider())
    }

    /// `true` si el trío corre en un endpoint o un kind distintos del principal.
    ///
    /// **Se decide sobre lo DECLARADO, no comparando URLs resueltas.** Dos plantillas
    /// distintas pueden resolver al mismo host —una con credenciales del vault y otra sin
    /// ellas— y comparar el resultado diría "no divergen" sobre una configuración que sí
    /// lo hace. Lo que importa acá es la intención del operador.
    ///
    /// Task 4.4: consumida en producción por `divergence_notice` (`main.rs`, REQ-A07p) —
    /// el `#[allow(dead_code)]` que tenía se retiró acá porque ya hay un llamador real.
    /// Covered by `magi_endpoint_diverges_when_the_trio_declares_its_own_kind_or_base_url`.
    #[must_use]
    pub fn magi_endpoint_diverges(&self) -> bool {
        let declara_url = self
            .magi
            .base_url
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        let declara_kind = self
            .magi
            .kind
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        declara_url || declara_kind
    }

    /// Cap de ENTRADA de magi-rs, previo al `max_input_len` de magi-core (REQ-A11b).
    ///
    /// Criterio del número: **costo, no capacidad**. magi-core ya saltea los modelos donde
    /// el prompt no entra, así que esto no protege al modelo — acota el gasto, y el
    /// payload se paga por tres porque va a los tres mages.
    // Narrow allow: consumed by the consult tool's input cap in a later fase, not this
    // task. Covered by `effective_max_query_bytes_falls_back_to_the_built_in_when_absent`.
    #[allow(dead_code)]
    #[must_use]
    pub fn effective_max_query_bytes(&self) -> usize {
        self.magi
            .max_query_bytes
            .unwrap_or(magi_rs::magi::MAX_QUERY_BYTES)
    }

    /// Cap de SALIDA del reporte, en las TRES rutas (REQ-A11b).
    ///
    /// **Vive en la raíz y no en `[headless]`**: bajo `[headless]` cubriría solo el modo
    /// por lotes y dejaría suelto el interactivo, que es justo donde el reporte se
    /// re-envía en cada turno de una sesión larga. Un cap que protege el caso barato y no
    /// el caro protege el caso equivocado.
    // El `allow(dead_code)` que esto tenía se retiró en Task 1.3: `resolve_headless_limits`
    // ya lo consume. Fase 6 sigue debiendo la aplicación en las OTRAS dos rutas (TUI y el
    // tool loop de `magi query`), que es su alcance real.
    #[must_use]
    pub fn effective_tool_result_cap(&self) -> usize {
        self.tool_result_cap_bytes
            .unwrap_or(magi_rs::magi::TOOL_RESULT_CAP_BYTES)
    }

    /// Endpoint del sistema: raíz declarada, o el default built-in.
    ///
    /// Devuelve la PLANTILLA, no un `&str` ya usable: resolver credenciales exige el
    /// vault, y ese es el único camino a un endpoint utilizable (REQ-A16c).
    ///
    /// # Errors
    /// [`EndpointError`] si el valor declarado no es una plantilla válida (credencial
    /// literal, placeholder desconocido, o URL irrecorrible). Ver
    /// [`magi_rs::magi::endpoint::EndpointTemplate::parse`].
    pub fn effective_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        EndpointTemplate::parse(
            self.base_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL),
        )
    }

    /// Resuelve el override de una sección (`[magi].base_url`/`[embedding].base_url`)
    /// contra el mismo endpoint efectivo del sistema, o hereda si no hay override.
    ///
    /// Compartida por [`Self::effective_magi_base_url`] y
    /// [`Self::effective_embedding_base_url`]: las dos aplican **exactamente** la misma
    /// regla ("propio, vacío-es-ausente, si no herencia") y repetirla en cada una sería
    /// la clase de duplicación que B3 prohíbe — desincronizarlas cambiaría la regla en
    /// una sección sin que nadie lo note en la otra.
    ///
    /// # Errors
    /// Ver [`Self::effective_base_url`].
    fn override_or_inherit_base_url(
        &self,
        own: Option<&str>,
    ) -> Result<EndpointTemplate, EndpointError> {
        match own.map(str::trim).filter(|s| !s.is_empty()) {
            Some(own) => EndpointTemplate::parse(own),
            None => self.effective_base_url(), // herencia, ya validada
        }
    }

    /// Endpoint del trío: override de `[magi].base_url`, o herencia del sistema.
    ///
    /// # Errors
    /// Ver [`Self::effective_base_url`].
    ///
    /// Consumida por [`Self::load`] (Task 1.4) para validar que la plantilla del trío
    /// sea usable ANTES de que el arranque termine — cierra SC-A16d también para
    /// `[magi].base_url`, no solo para la raíz y el embedder. La construcción real del
    /// trío nativo sobre este valor sigue siendo Fase 4; acá solo se valida.
    pub fn effective_magi_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(self.magi.base_url.as_deref())
    }

    /// Endpoint del embedder: override de `[embedding].base_url`, o herencia del sistema
    /// (REQ-A21 — cambio de comportamiento respecto de v0.11.0, ver
    /// [`crate::memory::config::EmbeddingConfig::base_url`]).
    ///
    /// # Errors
    /// Ver [`Self::effective_base_url`].
    pub fn effective_embedding_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(self.embedding.base_url.as_deref())
    }

    /// Carga `magi.toml` desde su **ruta** (no su directorio) — Task 1.4 consume
    /// finalmente `Workspace::config_path()` (REQ-A22b).
    ///
    /// Un archivo **ausente** devuelve los defaults built-in, `Ok`, sin notices. Un
    /// archivo **vacío o con solo espacios** también: todo campo de raíz es opcional, así
    /// que un TOML vacío es un TOML válido que declara cero cosas (SC-A21f).
    ///
    /// **Cambio de comportamiento respecto de v0.11.0 (REQ-A23).** Ahí un archivo roto
    /// producía *warning + defaults* — con `base_url` mudándose a raíz, ese camino
    /// descartaría el archivo entero en silencio y el usuario correría con defaults
    /// creyendo que su config aplica. Un `magi.toml` **presente** que no parsea, que
    /// declara un `provider`/`[magi].kind`/`[magi].default_mode` no reconocido, o que
    /// declara una `base_url` (raíz, `[magi]` o `[embedding]`) con una credencial literal
    /// en vez de los placeholders de REQ-A16c, **termina el proceso** — nunca degrada a
    /// defaults en silencio.
    ///
    /// # Errors
    /// - [`ConfigError::NeedsMigration`] si el archivo trae patrones de v0.11.0.
    /// - [`ConfigError::Parse`] si existe y no parsea, o no se pudo leer.
    /// - [`ConfigError::UnknownProviderKind`] / [`ConfigError::UnknownMode`] si
    ///   `provider`, `[magi].kind` o `[magi].default_mode` traen un valor presente y no
    ///   reconocido.
    /// - [`ConfigError::AgentTimeoutOutOfRange`] / [`ConfigError::OutputCapTooSmall`] si
    ///   esos números caen fuera de su rango.
    /// - [`ConfigError::Endpoint`] si la `base_url` de raíz, `[magi]` o `[embedding]` trae
    ///   una credencial literal, un placeholder desconocido, o no se pudo recorrer
    ///   (SC-A16d) — antes de esto, SOLO el camino del embedder
    ///   (`main.rs::attach_persistent_memory`) veía este error, y lo degradaba a un
    ///   notice + memoria en texto plano en vez de detener el arranque.
    ///
    /// # Arguments
    /// * `path` - Ruta del archivo `magi.toml`. Recomendado absoluta/canónica (p. ej.
    ///   `Workspace::config_path()`) para que la resolución sea reproducible.
    ///
    /// # Returns
    /// `(MagiConfig, Vec<String>)` — la config parseada y los notices de REQ-A12b/A12c
    /// sobre resoluciones que no salieron de lo escrito en el archivo.
    pub fn load(path: &Path) -> Result<(Self, Vec<String>), ConfigError> {
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Self::default(), Vec::new()));
            }
            Err(e) => {
                return Err(ConfigError::Parse(format!(
                    "{}: could not be read ({e})",
                    path.display()
                )))
            }
        };

        // Delega TODA la validación de forma+vocabulario a `from_toml_str` (migración,
        // parseo seguro vía `safe_parse_error`, vocabulario, rangos numéricos) — repetirla
        // acá duplicaría exactamente la lógica que esa función centraliza (B3) y, peor,
        // volvería a filtrar la línea ofensora de un TOML malformado por el `Display`
        // crudo de `toml::de::Error` (ver la doc de `safe_parse_error`).
        let cfg = Self::from_toml_str(&raw).map_err(|e| attach_path(e, path))?;

        // REQ-A16c/SC-A16d: una `base_url` con una credencial literal, un placeholder
        // desconocido, o una URL irrecorrible detiene el arranque ACÁ, en las TRES
        // secciones que pueden declararla. `from_toml_str` NO valida esto a propósito:
        // sus propios tests (`resolve_effective_*_endpoint_propagates_a_malformed_
        // template_error`) construyen una plantilla malformada y observan el error en el
        // punto de RESOLUCIÓN, no en el de parseo — moverlo ahí los rompería. La
        // validación de endpoints vive en el límite de producción, `load()`.
        cfg.effective_base_url()?;
        cfg.effective_magi_base_url()?;
        cfg.effective_embedding_base_url()?;

        let notices = cfg.resolution_notices();
        Ok((cfg, notices))
    }

    /// Notices de REQ-A12b/A12c: toda resolución que no salga de lo escrito en el
    /// archivo se anuncia, para que el usuario no tenga que adivinar qué terminó
    /// aplicando. Llamada únicamente desde [`Self::load`], sobre una config que ya pasó
    /// `validate_vocabulary` y la validación de endpoints — así que sus `effective_*()`
    /// internos nunca pueden fallar acá.
    fn resolution_notices(&self) -> Vec<String> {
        let mut out = Vec::new();

        if self
            .provider
            .as_deref()
            .is_some_and(|s| s.trim().is_empty())
        {
            out.push(format!(
                "notice: `provider` está vacío; se usa el default `{}`",
                // `DEFAULT_PROVIDER` ya es el valor del vocabulario REQ-A01b ("ollama"),
                // el mismo al que cae `effective_provider()` cuando `provider` está
                // ausente/vacío (Task 4.1 colapsó la constante legacy separada).
                crate::defaults::DEFAULT_PROVIDER,
            ));
        }

        // La PLANTILLA se muestra por su texto y **no se redacta**: por REQ-A16c no puede
        // contener un secreto (una credencial literal es error de config, ya rechazado
        // por `load()` antes de llegar acá), así que pasarla por `redact_url` sería
        // redundante *y* mal tipado.
        //
        // m2 (fix round 2, coordinator, 2026-08-03): `if let`, NO `let … else { return
        // out }`. Antes, un `effective_base_url()` fallido cortaba TODA la función —
        // incluidos los dos chequeos de Anthropic de más abajo, que no dependen de
        // esta plantilla — apoyado en que "`load()` ya validó". Esa garantía es real
        // HOY pero vive en `load()`, una función distinta: un futuro llamador de
        // `resolution_notices()`, o un reorden dentro de `load()`, produciría cero
        // avisos de Anthropic en silencio. Acotar el `if let` a este único bloque
        // cuesta cero y elimina el acoplamiento — ver
        // `a_failed_root_base_url_does_not_swallow_the_anthropic_notices`.
        if let Ok(root) = self.effective_base_url() {
            if root.as_str() != crate::defaults::DEFAULT_OPENAI_BASE_URL
                && self.embedding.base_url.is_none()
            {
                out.push(format!(
                    "notice: el embedder hereda `base_url = {}` de la raíz; declaralo en \
                     [embedding] si querés otro",
                    root.as_str(),
                ));
            }
        }

        // REQ-A12c: con `anthropic`, el `base_url` de raíz NO se usa para el agente
        // principal — Anthropic tiene su propio endpoint. Hay DOS sub-casos y ambos
        // avisan, por razones distintas:
        //
        //   (a) el usuario DECLARÓ un base_url  => cree que se usa, y no se usa
        //   (b) quedó el default de Ollama      => parece un olvido de migración
        if self.effective_provider() == ProviderKind::Anthropic {
            let declared = self.base_url.is_some();
            out.push(if declared {
                "notice: con `provider = \"anthropic\"` el `base_url` de raíz NO se usa para el \
                 agente principal (Anthropic usa su propio endpoint); solo aplica a [magi] y \
                 [embedding] si lo heredan"
                    .to_string()
            } else {
                "notice: `provider = \"anthropic\"` con el `base_url` por defecto de Ollama. Ese \
                 valor NO se usa para el agente principal; si querías Ollama, corregí `provider`"
                    .to_string()
            });
        }

        // Misma incoherencia un nivel abajo: el trío en Anthropic con su propio
        // base_url declarado, que tampoco se usa.
        if self.effective_magi_kind() == ProviderKind::Anthropic && self.magi.base_url.is_some() {
            out.push(
                "notice: con `[magi].kind = \"anthropic\"` el `[magi].base_url` NO se usa: \
                 Anthropic usa su propio endpoint"
                    .to_string(),
            );
        }

        out
    }
}

/// Prefija un [`ConfigError::Parse`] con la ruta del archivo ofensor; las demás
/// variantes ya son autocontenidas (nombran el campo, el valor, o el rango) y no
/// necesitan la ruta para ser accionables.
fn attach_path(e: ConfigError, path: &Path) -> ConfigError {
    match e {
        ConfigError::Parse(msg) => ConfigError::Parse(format!("{}: {msg}", path.display())),
        other => other,
    }
}

/// Backend efectivo del agente principal: env `MAGI_PROVIDER` > TOML `provider` >
/// `DEFAULT_PROVIDER` (RF-1, REQ-A01b).
///
/// Task 4.1: retira el shim `legacy_backend_label`/`resolve_provider` que normalizaba el
/// vocabulario nuevo (`ollama`/`openai-compat`/`anthropic`) sobre la etiqueta legacy
/// `"openai"` para que la cadena `provider_kind == "openai"` de `main.rs` siguiera
/// funcionando sin tocarla. Con esa cadena migrada a `ProviderKind` (misma tarea), ya no
/// hay nada que normalizar: el vocabulario es único de punta a punta.
///
/// **`MAGI_PROVIDER` recibe el mismo tratamiento que `provider`/`[magi].kind` en el TOML**:
/// un valor presente y no reconocido es un error explícito (REQ-A01b), no un fallback
/// silencioso — a diferencia del shim retirado, que dejaba pasar cualquier env var vieja
/// sin verificarla. Vacío o en blanco se trata como ausente (REQ-A12).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig` from `magi.toml` (may be default if file absent/invalid).
/// * `env_provider` - Value of `MAGI_PROVIDER` env var, if set.
///
/// # Errors
/// [`ProviderKindParseError`] si `MAGI_PROVIDER` está presente y no es uno de los tres
/// valores del vocabulario.
pub fn resolve_effective_provider_kind(
    config: &MagiConfig,
    env_provider: Option<&str>,
) -> Result<ProviderKind, ProviderKindParseError> {
    if let Some(raw) = env_provider {
        if let Some(kind) = ProviderKind::parse(raw)? {
            return Ok(kind);
        }
    }
    Ok(config.effective_provider())
}

/// Resolves a per-agent MAGI model override. Precedence: env (non-empty) > TOML
/// (non-empty) > `None`. A blank/whitespace value (env or TOML) is treated as
/// unset and falls through to the next level. `None` means the agent uses the
/// backend's model (RF-2, S-4, S-5).
///
/// Restored fix round 1 (coordinator, 2026-08-03): Task 4.1 deleted this along with
/// `agent::magi_wiring` (its only caller, the retired per-agent-adapter machinery) on
/// the reasoning that the native trio's `build_magi_orchestrator` had no env-override
/// parameter in the brief's pasted signature. That reasoning does not survive R-A03:
/// "las únicas rupturas admitidas son las declaradas en REQ-A21, REQ-A22 y REQ-A23" —
/// three, not four — and `MAGI_MODEL_*` appears nowhere in `spec-behavior.md` as an
/// authorized removal. Silence plus R-A03 means the capability stays. `main.rs`'s
/// `build_magi_orchestrator` now takes an env-override parameter and calls this for
/// each seat, layered on top of [`MagiSectionConfig::seats`]'s TOML-or-backend
/// resolution — giving the full `env > TOML > backend's model` chain.
///
/// # Arguments
/// * `toml_model` - The `[magi].<agent>_model` value, if present.
/// * `env_model`  - The `MAGI_MODEL_<AGENT>` env value, if present.
///
/// # Returns
/// `Some(model)` when an effective override exists; `None` otherwise.
pub fn resolve_magi_override(toml_model: Option<&str>, env_model: Option<&str>) -> Option<String> {
    fn non_empty(s: Option<&str>) -> Option<String> {
        s.map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
    non_empty(env_model).or_else(|| non_empty(toml_model))
}

/// env `OPENAI_MODEL` > TOML `[openai].model` > `DEFAULT_OPENAI_MODEL` (RF-3).
/// No longer fallible: the openai path has a built-in default (Ollama-first).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `OPENAI_MODEL` env var, if set.
///
/// # Returns
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_openai_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.openai.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_OPENAI_MODEL.into())
}

/// env `ANTHROPIC_MODEL` > TOML `[anthropic].model` > `DEFAULT_ANTHROPIC_MODEL`.
///
/// Mirrors [`resolve_openai_model`]'s precedence exactly. Fixes a MAGI re-gate
/// WARNING: prior call sites in `main.rs` disagreed on precedence — the
/// headless path checked TOML before env (backwards), and the TUI/other path
/// (`discover_config`) read only env and ignored `[anthropic].model`
/// entirely. Both now route through this single resolver.
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `ANTHROPIC_MODEL` env var, if set.
///
/// # Returns
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_anthropic_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.anthropic.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_ANTHROPIC_MODEL.into())
}

/// Arma los umbrales del gate de complejidad desde `[magi.complexity]` (REQ-A20b).
///
/// **Vive acá, y no en `magi_rs::magi::gate` — moved from Task 1.1 (ver
/// `.superpowers/sdd/claude-plan-tdd/ORDER-FIXES.md`, #1).** `gate.rs` vive en el lib y no
/// puede conocer la forma del TOML; desarmar `[magi.complexity]` en piezas sueltas
/// (`GateOverrides`) es trabajo de este módulo, que ya tiene la tabla en la mano.
///
/// Tabla ausente ⇒ `GateOverrides::default()` ⇒ los tres built-ins de
/// [`GateThresholds::builtin`] (el gate no se apaga por omitir la sección).
// Narrow allow: consumed by the TUI/`magi query`/`magi consult` autonomous-routing
// wiring in Tasks 3.2/3.3, not this task. Covered by
// `gate_thresholds_from_reads_the_complexity_table_and_falls_back_to_builtins`.
#[allow(dead_code)]
#[must_use]
pub fn gate_thresholds_from(config: &MagiConfig) -> GateThresholds {
    let overrides = config
        .magi
        .complexity
        .as_ref()
        .map(|c| GateOverrides {
            code_review: c.code_review,
            design: c.design,
            analysis: c.analysis,
        })
        .unwrap_or_default();
    GateThresholds::from_overrides(overrides)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Task 3.1: `gate_thresholds_from` — desarma `[magi.complexity]` en
    // `GateThresholds` (REQ-A20b). `gate.rs` vive en el lib y no puede conocer la
    // forma del TOML; esta función es la única que la desarma.
    // -------------------------------------------------------------------------

    /// La tabla puebla los umbrales declarados y hereda el built-in en los
    /// ausentes DENTRO de una tabla presente; sin tabla, los tres built-ins.
    #[test]
    fn gate_thresholds_from_reads_the_complexity_table_and_falls_back_to_builtins() {
        let with_table =
            MagiConfig::from_toml_str("[magi.complexity]\ncode_review = 50\nanalysis = 0\n")
                .unwrap();
        let t = gate_thresholds_from(&with_table);
        assert_eq!(t.code_review, 50, "declarado: se usa el valor del archivo");
        assert_eq!(
            t.design,
            GateThresholds::builtin().design,
            "ausente DENTRO de la tabla: su built-in, no cero"
        );
        assert_eq!(
            t.analysis, 0,
            "0 declarado se preserva: es la vía de apagar ESE modo"
        );

        let without_table = MagiConfig::default();
        assert_eq!(
            gate_thresholds_from(&without_table),
            GateThresholds::builtin(),
            "tabla ausente ⇒ built-ins: el gate no se apaga por omitir la sección"
        );
    }

    #[test]
    fn test_parses_full_config() {
        // `provider = "openai"` and `[openai].base_url` are both v0.11.0 shapes (Task
        // 1.1 breaks both, REQ-A21/A01b) — the root-level `base_url` and the
        // `ollama`/`openai-compat`/`anthropic` vocabulary replace them.
        let c = MagiConfig::from_toml_str(
            "provider = \"ollama\"\nbase_url = \"http://localhost:11434/v1\"\n[openai]\nmodel = \"phi4-mini\"\n[anthropic]\nmodel = \"claude-sonnet-4-6\"\n",
        ).unwrap();
        assert_eq!(c.provider.as_deref(), Some("ollama"));
        assert_eq!(c.base_url.as_deref(), Some("http://localhost:11434/v1"));
        assert_eq!(c.openai.model.as_deref(), Some("phi4-mini"));
        assert_eq!(c.anthropic.model.as_deref(), Some("claude-sonnet-4-6"));
    }

    #[test]
    fn test_empty_is_default() {
        assert_eq!(
            MagiConfig::from_toml_str("").unwrap(),
            MagiConfig::default()
        );
    }

    #[test]
    fn test_malformed_is_err() {
        assert!(MagiConfig::from_toml_str("provider = =bad").is_err());
    }

    #[test]
    fn test_unknown_field_is_err() {
        assert!(MagiConfig::from_toml_str("provdier = \"openai\"").is_err());
    }

    // -------------------------------------------------------------------------
    // Task 2: load + resolution tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_missing_file_is_default_no_warning() {
        // Task 1.4: `load` takes a FILE path now, not a directory, and returns
        // `Result<(Self, Vec<String>), ConfigError>`.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        let (c, notices) = MagiConfig::load(&path).unwrap();
        assert_eq!(c, MagiConfig::default());
        assert!(notices.is_empty());
    }

    #[test]
    fn test_load_reads_file() {
        // Task 1.1: `"openai"` is no longer a valid `provider` value (REQ-A01b) —
        // `"anthropic"` exercises the same "a real value round-trips through load()"
        // property without depending on the retired vocabulary.
        //
        // NOT asserting `notices.is_empty()`: `provider = "anthropic"` with no declared
        // `base_url` is exactly SC-A12d's sub-case (b) — the built-in Ollama default is
        // still sitting there, unused by the principal provider, and REQ-A12c requires a
        // notice about it. `silent_resolutions_are_announced_as_notices` and
        // `anthropic_flags_both_the_declared_and_the_defaulted_base_url` cover that
        // notice directly; this test's only job is the round-trip.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        std::fs::write(&path, "provider = \"anthropic\"").unwrap();
        let (c, _notices) = MagiConfig::load(&path).unwrap();
        assert_eq!(c.provider.as_deref(), Some("anthropic"));
    }

    /// Task 4.1: replaces `test_resolve_provider_precedence` (deleted along with the
    /// retired `resolve_provider`/`legacy_backend_label` shim). Same env > TOML > default
    /// precedence, expressed directly in the REQ-A01b vocabulary — no more legacy label
    /// to normalize onto.
    #[test]
    fn test_resolve_effective_provider_kind_precedence() {
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_effective_provider_kind(&c, Some("ollama")).unwrap(),
            ProviderKind::Ollama, // env wins
        );
        assert_eq!(
            resolve_effective_provider_kind(&c, None).unwrap(),
            ProviderKind::Anthropic, // TOML
        );
        // S-1: no config, no env → the built-in default (Ollama-first).
        assert_eq!(
            resolve_effective_provider_kind(&MagiConfig::default(), None).unwrap(),
            ProviderKind::Ollama,
        );
    }

    /// Task 4.1: `MAGI_PROVIDER` gets the SAME explicit-error treatment as `provider`/
    /// `[magi].kind` in the TOML — an unrecognized value is never a silent fallback
    /// (REQ-A01b). The retired shim used to let ANY old env-var value pass through
    /// unchecked; that asymmetry does not survive the migration.
    #[test]
    fn an_unrecognized_env_provider_is_a_configuration_error() {
        let err =
            resolve_effective_provider_kind(&MagiConfig::default(), Some("banana")).unwrap_err();
        assert!(err.to_string().contains("banana"));
    }

    /// SC-A12g / REQ-A12: a blank `MAGI_PROVIDER` is treated as ABSENT, not invalid —
    /// falls through to the TOML/default the same as an unset env var.
    #[test]
    fn a_blank_env_provider_falls_through_to_the_toml_default() {
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_effective_provider_kind(&c, Some("   ")).unwrap(),
            ProviderKind::Anthropic,
        );
    }

    // `test_resolve_openai_base_url_precedence` removed (fix round 3, L1/L2/S1):
    // `resolve_openai_base_url` — the function it tested — bypassed blank-is-absent
    // and credential resolution entirely, and was removed in favor of `main.rs`'s
    // `resolve_effective_principal_endpoint` (which reuses
    // `MagiConfig::effective_base_url` and is covered where it lives, alongside
    // `resolve_effective_embedding_endpoint`).

    #[test]
    fn test_resolve_openai_model_defaults() {
        use crate::defaults::DEFAULT_OPENAI_MODEL;
        // S-2: no env, no TOML → DEFAULT_OPENAI_MODEL (was Err)
        assert_eq!(
            resolve_openai_model(&MagiConfig::default(), None),
            DEFAULT_OPENAI_MODEL
        );
        // S-3: env/TOML still win
        let c = MagiConfig {
            openai: OpenAiConfig {
                model: Some("phi4-mini".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_openai_model(&c, None), "phi4-mini");
        assert_eq!(resolve_openai_model(&c, Some("gpt-4o-mini")), "gpt-4o-mini");
    }

    #[test]
    fn test_resolve_anthropic_model_env_wins_over_toml() {
        // MAGI re-gate WARNING fix: env must win over TOML (not the other way
        // around, which was the bug in the pre-fix headless call site).
        let c = MagiConfig {
            anthropic: AnthropicConfig {
                model: Some("claude-toml-model".into()),
            },
            ..Default::default()
        };
        assert_eq!(
            resolve_anthropic_model(&c, Some("claude-env-model")),
            "claude-env-model"
        );
    }

    #[test]
    fn test_resolve_anthropic_model_toml_when_no_env() {
        let c = MagiConfig {
            anthropic: AnthropicConfig {
                model: Some("claude-toml-model".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_anthropic_model(&c, None), "claude-toml-model");
    }

    #[test]
    fn test_resolve_anthropic_model_default_when_neither() {
        use crate::defaults::DEFAULT_ANTHROPIC_MODEL;
        assert_eq!(
            resolve_anthropic_model(&MagiConfig::default(), None),
            DEFAULT_ANTHROPIC_MODEL
        );
    }

    #[test]
    fn test_load_unreadable_file_is_fatal() {
        // A directory named `magi.toml` makes read_to_string fail with a non-NotFound
        // error → REQ-A23: must be FATAL, never degrade to defaults + warning.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        std::fs::create_dir(&path).unwrap();
        let err = MagiConfig::load(&path).expect_err("unreadable magi.toml must be fatal");
        assert!(err.to_string().contains("magi.toml"));
    }

    // -------------------------------------------------------------------------
    // Task 1: MagiSectionConfig parsing tests (S-1, S-2, S-3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parses_magi_section() {
        // S-1
        let c = MagiConfig::from_toml_str(
            "[magi]\nmelchior_model = \"qwen3:8b\"\ncaspar_model = \"deepseek-r1:32b\"\n",
        )
        .unwrap();
        assert_eq!(c.magi.melchior_model.as_deref(), Some("qwen3:8b"));
        assert_eq!(c.magi.balthasar_model, None);
        assert_eq!(c.magi.caspar_model.as_deref(), Some("deepseek-r1:32b"));
    }

    #[test]
    fn test_absent_magi_section_is_default() {
        // S-2
        let c = MagiConfig::from_toml_str("provider = \"anthropic\"").unwrap();
        assert_eq!(c.magi, MagiSectionConfig::default());
    }

    #[test]
    fn test_unknown_field_in_magi_section_is_err() {
        // S-3
        assert!(MagiConfig::from_toml_str("[magi]\nunknown_field = \"x\"").is_err());
    }

    // ── auto_approve field tests ──────────────────────────────────────────────

    /// Default `[magi]` section (absent or empty) must have `auto_approve = false`.
    ///
    /// Historical: added when `auto_approve` first landed on this section (now `MagiSectionConfig`).
    #[test]
    fn test_magi_auto_approve_defaults_to_false() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert!(
            !c.magi.auto_approve,
            "auto_approve must default to false (opt-in, never silently enabled)"
        );
        // Also check that an explicit [magi] section without the field also defaults.
        let c2 = MagiConfig::from_toml_str("[magi]\nmelchior_model = \"qwen3:8b\"").unwrap();
        assert!(
            !c2.magi.auto_approve,
            "auto_approve must default to false even when [magi] section is present"
        );
    }

    /// `[magi] auto_approve = true` must parse to `true`.
    ///
    /// Historical: added when `auto_approve` first landed on this section (now `MagiSectionConfig`).
    #[test]
    fn test_magi_auto_approve_true_parses() {
        let c = MagiConfig::from_toml_str("[magi]\nauto_approve = true").unwrap();
        assert!(
            c.magi.auto_approve,
            "auto_approve = true in [magi] must parse to true"
        );
    }

    /// `deny_unknown_fields` must still reject genuinely unknown fields even after
    /// adding `auto_approve` (regression guard — field name typos must not silently
    /// apply the default).
    #[test]
    fn test_magi_auto_approve_typo_is_still_rejected() {
        assert!(
            MagiConfig::from_toml_str("[magi]\nauto_approv = true").is_err(),
            "typo 'auto_approv' (missing 'e') must be rejected by deny_unknown_fields"
        );
    }

    // -------------------------------------------------------------------------
    // Task 2 / restored fix round 1: resolve_magi_override precedence tests (S-4, S-5).
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_magi_override_env_wins_over_toml() {
        // S-4: env > TOML
        assert_eq!(
            resolve_magi_override(Some("toml-model"), Some("env-model")),
            Some("env-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_toml_when_no_env() {
        // S-4: TOML when env absent
        assert_eq!(
            resolve_magi_override(Some("toml-model"), None),
            Some("toml-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_none_when_both_absent() {
        // S-4: none ⇒ principal model
        assert_eq!(resolve_magi_override(None, None), None);
    }

    #[test]
    fn test_resolve_magi_override_empty_string_is_unset() {
        // S-5: empty (env or TOML) is treated as unset, falls through precedence
        assert_eq!(
            resolve_magi_override(Some("toml"), Some("   ")),
            Some("toml".to_string())
        );
        assert_eq!(resolve_magi_override(Some(""), None), None);
        assert_eq!(resolve_magi_override(Some(""), Some("")), None);
    }

    // -------------------------------------------------------------------------
    // [headless] section tests (spec §11). `HeadlessConfig` already derives
    // `#[serde(deny_unknown_fields)]`, so these LOCK the existing parsing
    // contract (documenting, not driving it) rather than being a Red/Green
    // pair — they still fail if a future edit silently loosens the section.
    // -------------------------------------------------------------------------

    /// An unknown key inside `[headless]` (e.g. a typo) is a parse ERROR, not
    /// silent acceptance — `deny_unknown_fields` applies to this section like
    /// every other `MagiConfig` sub-table.
    #[test]
    fn test_headless_section_unknown_field_is_err() {
        assert!(MagiConfig::from_toml_str("[headless]\nmax_input_byte = 1024").is_err());
    }

    /// A `[headless]` block with several keys set parses into the matching
    /// `Option` fields; unset keys stay `None` (resolved to their built-in
    /// default elsewhere, `main.rs::resolve_headless_limits`/
    /// `resolve_log_level`/`resolve_allow_system_override`).
    #[test]
    fn test_headless_section_parses_configured_keys() {
        let c = MagiConfig::from_toml_str(
            "[headless]\n\
             max_input_bytes = 2048\n\
             full_auto_max_tool_calls = 30\n\
             log_retention = 7\n\
             log_max_bytes = 1048576\n\
             log_level = \"debug\"\n\
             timeout_secs = 120\n\
             allow_system_override = true\n",
        )
        .unwrap();

        assert_eq!(c.headless.max_input_bytes, Some(2048));
        assert_eq!(c.headless.full_auto_max_tool_calls, Some(30));
        assert_eq!(c.headless.log_retention, Some(7));
        assert_eq!(c.headless.log_max_bytes, Some(1_048_576));
        assert_eq!(c.headless.log_level.as_deref(), Some("debug"));
        assert_eq!(c.headless.timeout_secs, Some(120));
        assert_eq!(c.headless.allow_system_override, Some(true));
    }

    /// An absent `[headless]` section parses to all-`None` (every cap falls
    /// back to its built-in default).
    #[test]
    fn test_headless_section_absent_is_all_none() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert_eq!(c.headless, HeadlessConfig::default());
    }

    // -------------------------------------------------------------------------
    // Task 1.1: base_url to root + unified provider vocabulary (REQ-A01b, A12, A21)
    // -------------------------------------------------------------------------

    use magi_core::schema::Mode;

    #[test]
    fn provider_kind_accepts_the_three_values_and_rejects_the_rest() {
        assert_eq!(
            ProviderKind::parse("ollama").unwrap(),
            Some(ProviderKind::Ollama)
        );
        assert_eq!(
            ProviderKind::parse("openai-compat").unwrap(),
            Some(ProviderKind::OpenAiCompat)
        );
        assert_eq!(
            ProviderKind::parse("anthropic").unwrap(),
            Some(ProviderKind::Anthropic)
        );
        assert!(ProviderKind::parse("banana").is_err());
        assert!(
            ProviderKind::parse("openai").is_err(),
            "el valor viejo ya no es válido"
        );
    }

    /// REQ-A01b: un valor inválido NO se traga — ni siquiera pasando por los resolutores.
    ///
    /// Este test existe porque el test de `ProviderKind::parse` **no alcanza**: prueba la
    /// unidad, y el fallback silencioso vive en el LLAMADOR. Un resolutor con
    /// `.ok().flatten().unwrap_or(default)` deja pasar `"banana"` como `Ollama` mientras el
    /// test de la unidad sigue verde.
    ///
    /// **Corrección de la ruling del coordinador (2026-08-02).** El plan original probaba
    /// esto contra `MagiConfig::load(&path)`, pero `load` en Task 1.1 conserva su firma
    /// externa `(dir: &Path) -> (Self, Option<String>)` (se vuelve falible recién en Task
    /// 1.4, junto con el cableado de `main.rs`/`Workspace::config_path()` que esa tarea
    /// posee). La propiedad que este test defiende — que un valor inválido no se convierte
    /// en un fallback silencioso — se prueba contra `from_toml_str`, que es donde
    /// `validate_vocabulary()` corre de verdad; `load()` la ejercita indirectamente porque
    /// llama a `from_toml_str` sobre el contenido del archivo. Task 1.4 agrega la misma
    /// aserción contra `load()` cuando esa función se vuelve falible.
    #[test]
    fn an_invalid_vocabulary_value_is_rejected_at_parse_not_swallowed_by_a_resolver() {
        for (toml, what) in [
            ("provider = \"banana\"\n", "provider de raíz"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            assert!(
                MagiConfig::from_toml_str(toml).is_err(),
                "{what}: un valor no reconocido debe ser ERROR, nunca un fallback silencioso"
            );
        }
    }

    /// SC-A12g / REQ-A12: regla general — vacío o en blanco es AUSENTE, nunca inválido.
    #[test]
    fn blank_string_keys_are_absent_not_invalid() {
        assert_eq!(ProviderKind::parse("").unwrap(), None);
        assert_eq!(ProviderKind::parse("   ").unwrap(), None);
        let toml = "provider = \"\"\nbase_url = \"  \"\n";
        let cfg = MagiConfig::from_toml_str(toml).expect("vacío no debe romper el parseo");
        assert_eq!(
            cfg.effective_provider(),
            ProviderKind::Ollama,
            "cae al default built-in"
        );
        // `effective_base_url()` devuelve `Result<EndpointTemplate, _>` desde REQ-A16c, así
        // que el test compara el TEXTO de la plantilla.
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            crate::defaults::DEFAULT_OPENAI_BASE_URL
        );
    }

    /// SC-A02b: `[magi].kind` HEREDA de la raíz cuando no se declara.
    #[test]
    fn magi_kind_inherits_from_root_provider_when_absent() {
        let toml = "provider = \"ollama\"\n[magi]\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.effective_magi_kind(), ProviderKind::Ollama);

        let toml = "provider = \"ollama\"\n[magi]\nkind = \"anthropic\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.effective_magi_kind(), ProviderKind::Anthropic);
        assert_eq!(
            cfg.effective_provider(),
            ProviderKind::Ollama,
            "el principal no cambia"
        );
    }

    /// SC-A21c: herencia y override del endpoint.
    #[test]
    fn base_url_inherits_from_root_and_sections_override_it() {
        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        // Texto de la PLANTILLA: `effective_*_base_url()` devuelve `Result<EndpointTemplate,_>`
        // desde REQ-A16c.
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );
        assert_eq!(
            cfg.effective_embedding_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );

        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\nbase_url = \"http://otro/v1\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://otro/v1"
        );
    }

    /// El campo viejo ya no existe: su presencia es campo desconocido.
    #[test]
    fn openai_section_no_longer_accepts_base_url() {
        let toml = "[openai]\nbase_url = \"http://x/v1\"\n";
        assert!(MagiConfig::from_toml_str(toml).is_err());
    }

    /// SC-A12 / REQ-A14: un campo desconocido es ERROR DE PARSEO, no aceptación silenciosa.
    ///
    /// Las API keys NUNCA viven en `magi.toml`, y `deny_unknown_fields` lo hace **mecánico**
    /// en vez de una convención que alguien tiene que recordar. Cierra SC-A12 con el caso que
    /// más importa: el campo mal escrito que además sería un secreto.
    #[test]
    fn an_api_key_anywhere_in_the_toml_is_a_parse_error() {
        for toml in [
            "api_key = \"sk-secreto\"\n",
            "[openai]\napi_key = \"sk-secreto\"\n",
            "[anthropic]\napi_key = \"sk-secreto\"\n",
            "[magi]\napi_key = \"sk-secreto\"\n",
        ] {
            let err = MagiConfig::from_toml_str(toml)
                .expect_err("una api_key en el TOML debe ser ERROR, no aceptación silenciosa");
            assert!(
                !err.to_string().contains("sk-secreto"),
                "y el error NO puede repetir el secreto que se está rechazando"
            );
        }
    }

    /// REQ-A15: `default_mode` se resuelve con la misma regla vacío=ausente.
    ///
    /// **Devuelve `Option<Mode>`, NO `Result`**: la validación vive en `validate_vocabulary`,
    /// que corre en `load()`/`from_toml_str()`. Un resolutor que devuelve `Result` invita a
    /// que el llamador escriba `.ok()` — y eso ya pasó dos veces en este plan.
    #[test]
    fn effective_default_mode_follows_the_same_blank_is_absent_rule() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"code-review\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), Some(Mode::CodeReview));

        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);

        // El valor inválido no llega nunca a este resolutor: muere en el parseo.
        assert!(MagiConfig::from_toml_str("[magi]\ndefault_mode = \"banana\"\n").is_err());
    }

    // -------------------------------------------------------------------------
    // B13: cobertura de las funciones públicas restantes que Task 1.1 produce
    // (`Interfaces > Produces` del brief), sin test explícito en el Step 1.
    // -------------------------------------------------------------------------

    /// `seats()` resuelve cada mage a su modelo declarado, o al fallback del backend.
    #[test]
    fn seats_resolves_each_mage_to_its_override_or_the_backend_fallback() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("custom-melchior".into()),
            ..MagiSectionConfig::default()
        };
        let seats = cfg.seats("backend-default");
        assert_eq!(seats.len(), 3);
        assert_eq!(
            seats[0],
            (AgentName::Melchior, "custom-melchior".to_string())
        );
        assert_eq!(
            seats[1],
            (AgentName::Balthasar, "backend-default".to_string())
        );
        assert_eq!(seats[2], (AgentName::Caspar, "backend-default".to_string()));
    }

    /// `fallback_model()` es el modelo del BACKEND, nunca el de un mage — elegir el de
    /// Melchior lo volvería el default por accidente (ver su rustdoc / Task 4.1).
    #[test]
    fn fallback_model_is_the_backend_model_not_any_seats_override() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("should-not-win".into()),
            ..MagiSectionConfig::default()
        };
        assert_eq!(cfg.fallback_model("backend-default"), "backend-default");
    }

    /// `magi_endpoint_diverges()` es true si el trío declara `kind` o `base_url` propios,
    /// y blanco cuenta como NO declarado (REQ-A12).
    ///
    /// SC-A02c (mitad "ausente"): `kind = ""` se trata como ausente y hereda — el `blank`
    /// de abajo. La otra mitad (`kind = "banana"` ⇒ trío no construible) cierra en Fase 4.
    #[test]
    fn magi_endpoint_diverges_when_the_trio_declares_its_own_kind_or_base_url() {
        assert!(!MagiConfig::default().magi_endpoint_diverges());

        let own_kind = MagiConfig::from_toml_str("[magi]\nkind = \"anthropic\"\n").unwrap();
        assert!(own_kind.magi_endpoint_diverges());

        let own_url =
            MagiConfig::from_toml_str("[magi]\nbase_url = \"http://other/v1\"\n").unwrap();
        assert!(own_url.magi_endpoint_diverges());

        let blank = MagiConfig::from_toml_str("[magi]\nkind = \"\"\n").unwrap();
        assert!(!blank.magi_endpoint_diverges());
    }

    /// `effective_max_query_bytes()`: declarado gana, ausente cae al built-in.
    #[test]
    fn effective_max_query_bytes_falls_back_to_the_built_in_when_absent() {
        assert_eq!(
            MagiConfig::default().effective_max_query_bytes(),
            magi_rs::magi::MAX_QUERY_BYTES
        );

        let declared = MagiConfig::from_toml_str("[magi]\nmax_query_bytes = 999\n").unwrap();
        assert_eq!(declared.effective_max_query_bytes(), 999);
    }

    /// `effective_tool_result_cap()`: declarado gana, ausente cae al built-in.
    #[test]
    fn effective_tool_result_cap_falls_back_to_the_built_in_when_absent() {
        assert_eq!(
            MagiConfig::default().effective_tool_result_cap(),
            magi_rs::magi::TOOL_RESULT_CAP_BYTES
        );

        let above_min = magi_rs::magi::min_viable_output_cap() + 10;
        let declared = MagiConfig {
            tool_result_cap_bytes: Some(above_min),
            ..Default::default()
        };
        assert_eq!(declared.effective_tool_result_cap(), above_min);
    }

    // -------------------------------------------------------------------------
    // Fix round 2 (coordinator review, 2026-08-02): I3/I4/I5/m8 — B13 coverage
    // this task's own new functions shipped without.
    // -------------------------------------------------------------------------

    /// I3: both range boundaries of `agent_timeout_secs` (§4.9) are accepted; one
    /// step outside either end is rejected. `validate_agent_timeout` shipped with
    /// zero tests and an inclusive-both-ends range with nothing pinning the edge.
    #[test]
    fn agent_timeout_secs_accepts_both_boundaries_and_rejects_one_step_outside() {
        for ok in [AGENT_TIMEOUT_MIN_SECS, AGENT_TIMEOUT_MAX_SECS] {
            let toml = format!("[magi]\nagent_timeout_secs = {ok}\n");
            assert!(
                MagiConfig::from_toml_str(&toml).is_ok(),
                "{ok}s is inside [{AGENT_TIMEOUT_MIN_SECS}, {AGENT_TIMEOUT_MAX_SECS}] and must be accepted"
            );
        }
        for bad in [AGENT_TIMEOUT_MIN_SECS - 1, AGENT_TIMEOUT_MAX_SECS + 1] {
            let toml = format!("[magi]\nagent_timeout_secs = {bad}\n");
            assert!(
                MagiConfig::from_toml_str(&toml).is_err(),
                "{bad}s is one step outside the range and must be rejected"
            );
        }
    }

    /// I3: the output-cap floor (`min_viable_output_cap()`) itself is accepted;
    /// one byte below it is rejected. Same "zero tests on a boundary" gap as
    /// `validate_agent_timeout`.
    #[test]
    fn tool_result_cap_bytes_accepts_the_minimum_viable_floor_and_rejects_one_byte_below() {
        let min = magi_rs::magi::min_viable_output_cap();
        let ok_toml = format!("tool_result_cap_bytes = {min}\n");
        assert!(
            MagiConfig::from_toml_str(&ok_toml).is_ok(),
            "the floor itself must be accepted"
        );
        let bad_toml = format!("tool_result_cap_bytes = {}\n", min - 1);
        assert!(
            MagiConfig::from_toml_str(&bad_toml).is_err(),
            "one byte below the floor must be rejected"
        );
    }

    /// I4: `safe_parse_error` keeps line/column (SC-A21g requires a syntax error
    /// to name a position) but never the offending value — only the source
    /// EXCERPT needed suppressing to fix the `api_key` leak (see
    /// `safe_parse_error`'s own doc), not the position.
    #[test]
    fn safe_parse_error_keeps_the_position_but_drops_the_offending_value() {
        // Leading blank line: line 2 (not 1) pins that this is a real computed
        // position, not a hardcoded "line 1, column 1".
        let toml = "\napi_key = \"sk-secreto\"\n";
        let err = MagiConfig::from_toml_str(toml).unwrap_err();
        let msg = err.to_string();
        assert!(!msg.contains("sk-secreto"), "leaked the secret: {msg}");
        assert!(
            msg.contains("line") && msg.contains("column"),
            "lost the position: {msg}"
        );
        assert!(msg.contains("line 2"), "wrong line: {msg}");
    }

    /// I5: `effective_provider` is documented "infalible por precondición" — that
    /// precondition is `validate_vocabulary` having already run. `MagiConfig`'s
    /// fields are `pub` and it derives `Default`, so nothing at the type level
    /// stops a caller from skipping `from_toml_str`/`load()` and constructing an
    /// invalid config directly; the `debug_assert!` is what turns that misuse
    /// into a loud debug-build panic instead of a silent `Ollama` fallback.
    #[test]
    #[should_panic(expected = "validado")]
    fn effective_provider_panics_in_debug_builds_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig {
            provider: Some("banana".into()),
            ..Default::default()
        };
        let _ = cfg.effective_provider();
    }

    /// I5: same precondition, same gap, for `effective_default_mode`.
    #[test]
    #[should_panic(expected = "validado")]
    fn effective_default_mode_panics_in_debug_builds_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig {
            magi: MagiSectionConfig {
                default_mode: Some("banana".into()),
                ..MagiSectionConfig::default()
            },
            ..Default::default()
        };
        let _ = cfg.effective_default_mode();
    }

    /// m8: `[magi].base_url = ""` is blank, not a declared override — it must
    /// inherit the root, not be treated as "the trio declared its own endpoint".
    #[test]
    fn magi_base_url_blank_is_treated_as_absent() {
        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\nbase_url = \"\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://lan:11434/v1",
            "blank must inherit the root, not stay blank"
        );
        assert!(
            !cfg.magi_endpoint_diverges(),
            "a blank override does not count as declaring one (REQ-A12)"
        );
    }

    /// m8: a whitespace-only `default_mode` is blank, not a value — same
    /// blank-is-absent rule as every other vocabulary key.
    #[test]
    fn default_mode_whitespace_only_is_treated_as_absent() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"   \"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);
    }

    /// m8: `[embedding].base_url`'s OVERRIDE winning over the root was untested —
    /// the existing coverage only proved inheritance, never that a declared
    /// embedding-specific endpoint takes precedence over it.
    #[test]
    fn embedding_base_url_override_wins_over_root_inheritance() {
        let toml = "base_url = \"http://lan:11434/v1\"\n\
                     [embedding]\n\
                     base_url = \"http://embedding-only:11434/v1\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_embedding_base_url().unwrap().as_str(),
            "http://embedding-only:11434/v1"
        );
        // The root and [magi] are unaffected by the embedding-only override.
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );
    }

    // -------------------------------------------------------------------------
    // Task 1.4: `load` fallible, resolution notices, `--init-config` retirement
    // -------------------------------------------------------------------------

    /// REQ-A01b a través del camino de PRODUCCIÓN, no solo del parser.
    ///
    /// Complemento obligatorio de
    /// `an_invalid_vocabulary_value_is_rejected_at_parse_not_swallowed_by_a_resolver`
    /// (Task 1.1): aquel prueba `from_toml_str`, éste prueba que `load()` ya no degrada
    /// el error a defaults-más-warning — que es exactamente lo que hacía entre el cierre
    /// de Task 1.1 y el de ésta (hueco intermedio conocido, ver ORDER-FIXES ruptura #8).
    #[test]
    fn an_invalid_vocabulary_value_is_fatal_through_load_not_degraded_to_defaults() {
        for (toml, what) in [
            ("provider = \"banana\"\n", "provider de raíz"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            assert!(
                MagiConfig::load(&path).is_err(),
                "{what}: por load() también debe ser ERROR, nunca defaults + warning"
            );
        }
    }

    /// REQ-A23: presente y no parsea ⇒ FATAL. Ausente ⇒ default silencioso.
    #[test]
    fn a_present_but_broken_config_is_fatal_while_an_absent_one_is_silent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        assert!(
            MagiConfig::load(&path).is_ok(),
            "ausente: default silencioso"
        );

        // m6 (fix round 2, coordinator, 2026-08-03) / SC-A21f: un archivo PRESENTE
        // pero vacío o con solo espacios también es silencioso — todo campo de raíz
        // es opcional, así que un TOML en blanco es un TOML válido que declara cero
        // cosas. `from_toml_str("")` y `detect_migrations("")` ya estaban cubiertas
        // por separado; esto es lo único que faltaba: `load()` de punta a punta
        // contra un ARCHIVO real, que es lo que su propia rustdoc cita como cubierto.
        std::fs::write(&path, "   \n").unwrap();
        let (cfg, notices) = MagiConfig::load(&path).expect("en blanco: default silencioso");
        assert_eq!(cfg, MagiConfig::default());
        assert!(notices.is_empty());

        std::fs::write(&path, "provdier = \"x\"").unwrap();
        let err = MagiConfig::load(&path).expect_err("presente y roto: FATAL");
        assert!(err.to_string().contains("magi.toml"));
    }

    /// SEGUNDA OBLIGACIÓN heredada de Task 1.1: SC-A16d, fallo CERRADO ante una
    /// credencial literal en `base_url` — en las TRES secciones que pueden declararla,
    /// no solo la que el camino del embedder (`attach_persistent_memory`) ya cubría
    /// degradando a un notice + memoria en texto plano (lo que SC-A16d prohíbe).
    ///
    /// Afirma las DOS mitades que el requerimiento pide: que el arranque falla, y que
    /// el mensaje no repite el valor de la credencial encontrada.
    #[test]
    fn a_literal_credential_in_any_base_url_scope_fails_closed_at_load() {
        for (toml, scope) in [
            ("base_url = \"https://alice:s3cr3t@host/v1\"\n", "raíz"),
            (
                "[magi]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[magi]",
            ),
            (
                "[embedding]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[embedding]",
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            let err = match MagiConfig::load(&path) {
                Err(e) => e,
                Ok(_) => {
                    panic!("{scope}: load() debió fallar ante una credencial literal en base_url")
                }
            };
            let msg = err.to_string();
            // m3 (fix round 2, coordinator, 2026-08-03): la contraseña ya estaba
            // cubierta; el USUARIO no lo estaba — y esta prueba vive en `config.rs`,
            // así que una regresión en `EndpointError::LiteralCredential` (que solo
            // lleva `&'static str`, nunca el valor recibido) no la vería el módulo
            // que la obligación nombra si solo pineamos la mitad.
            assert!(
                !msg.contains("s3cr3t"),
                "{scope}: filtró la contraseña: {msg}"
            );
            assert!(!msg.contains("alice"), "{scope}: filtró el usuario: {msg}");
            // Y no alcanza con "no filtra": el mensaje tiene que ser ACCIONABLE —
            // nombrar el placeholder correcto y el comando de vault, no solo decir
            // "credencial inválida".
            assert!(
                msg.contains("[user]") && msg.contains("[password]"),
                "{scope}: no nombra el placeholder: {msg}"
            );
            assert!(
                msg.contains("magi-rs vault set"),
                "{scope}: no nombra el comando para arreglarlo: {msg}"
            );
        }
    }

    /// REQ-A12b: las resoluciones silenciosas se anuncian.
    #[test]
    fn silent_resolutions_are_announced_as_notices() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        std::fs::write(&path, "provider = \"\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("provider")));

        std::fs::write(&path, "base_url = \"http://lan:11434/v1\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            notices.iter().any(|n| n.contains("embedder")),
            "el embedder hereda un base_url NO-default: hay que decirlo"
        );

        std::fs::write(&path, "base_url = \"http://localhost:11434/v1\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            !notices.iter().any(|n| n.contains("embedder")),
            "heredar el DEFAULT no es sorprendente: sería ruido en cada arranque"
        );
    }

    /// SC-A12d: combinación incoherente detectada AL CARGAR, en sus DOS sub-casos.
    #[test]
    fn anthropic_flags_both_the_declared_and_the_defaulted_base_url() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        // (b) default de Ollama sentado ahí — el caso que un guard `is_some()` NO cubría.
        std::fs::write(&path, "provider = \"anthropic\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            notices.iter().any(|n| n.contains("por defecto de Ollama")),
            "sin base_url declarado el default sigue ahí, y parece un olvido de migración"
        );

        // (a) declarado explícitamente — el usuario cree que se usa.
        std::fs::write(
            &path,
            "provider = \"anthropic\"\nbase_url = \"http://x/v1\"\n",
        )
        .unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("NO se usa")));

        // Y el mismo caso un nivel abajo, en el trío.
        std::fs::write(
            &path,
            "[magi]\nkind = \"anthropic\"\nbase_url = \"http://x/v1\"\n",
        )
        .unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("[magi].base_url")));

        // Sin Anthropic no hay nada que avisar.
        std::fs::write(&path, "provider = \"ollama\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(!notices.iter().any(|n| n.contains("no se usa")));
    }

    /// m2 (fix round 2, coordinator, 2026-08-03): un `effective_base_url()` fallido NO
    /// debe silenciar los avisos de incoherencia de Anthropic que le siguen.
    ///
    /// `resolution_notices()` solo corre hoy dentro de `load()`, DESPUÉS de que
    /// `load()` ya validó las tres plantillas — así que en producción
    /// `effective_base_url()` nunca falla acá. Pero esa garantía vive en `load()`, una
    /// función DISTINTA: si `resolution_notices()` alguna vez se llama desde otro
    /// lado, o si `load()` se reordena, el `let Ok(root) = … else { return out }`
    /// original cortaría TODA la función en el primer `?` implícito — incluidos los
    /// dos chequeos de Anthropic que nada tienen que ver con `effective_base_url()` —
    /// sin ninguna señal de que se perdió cobertura.
    ///
    /// Se llama a `resolution_notices()` DIRECTAMENTE (acceso de `mod tests` al
    /// privado del módulo padre), no a través de `load()`: es la única forma de poner
    /// a esta función bajo la precondición que su propio `else` dice que nunca
    /// ocurre, sin duplicar la validación de `load()` en el test.
    #[test]
    fn a_failed_root_base_url_does_not_swallow_the_anthropic_notices() {
        let cfg = MagiConfig::from_toml_str(
            "provider = \"anthropic\"\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
        )
        .unwrap();
        // Precondición del test: la plantilla de raíz SÍ falla (credencial literal),
        // así que `resolution_notices()` corre exactamente bajo la condición que su
        // propio comentario decía "infalible en la práctica".
        assert!(cfg.effective_base_url().is_err());

        let notices = cfg.resolution_notices();
        assert!(
            notices.iter().any(|n| n.contains("NO se usa")),
            "el aviso de incoherencia de Anthropic no debe depender de que la \
             plantilla de raíz haya parseado: {notices:?}"
        );
    }
}
