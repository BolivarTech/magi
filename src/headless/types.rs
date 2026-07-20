// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Tipos compartidos del contrato de salida headless (MS1 ↔ MS2).
//!
//! DEFINIDO acá, no referenciado sin definir — evita el build-break de orden
//! TDD (T6 usa `AppliedCaps` antes de que T7 exista). Los tipos se declaran
//! `pub` con sus campos + derives: T6 consume [`AppliedCaps`]/[`SystemPolicy`];
//! T7 implementa el formateo y el golden sobre [`RunOutcome`] y compañía. MS2
//! **llena** estos tipos con la salida real del `Agent` desde el crate del
//! binario, que solo puede alcanzar API `pub` de la lib (no `pub(crate)`).
//!
//! Los consumidores reales de estos tipos llegan en tareas posteriores del mismo
//! milestone (T6/T7) y en el runner del binario (MS2).

use serde::Serialize;

/// Resultado completo de una corrida headless (`magi query`/`consult`).
///
/// Es la proyección **estable** que el contrato serializa (REQ-H14),
/// desacoplada del `Message` interno del `Agent`: MS2 la llena, no la redefine.
#[derive(Debug, Clone, Serialize)]
pub struct RunOutcome {
    /// Texto de respuesta del agente; `None` si la corrida terminó en error.
    pub response: Option<String>,
    /// Modelo efectivo usado en la corrida.
    pub model: String,
    /// Proveedor efectivo usado en la corrida.
    pub provider: String,
    /// Conteo de tokens de entrada/salida (facturación/observabilidad).
    pub usage: Usage,
    /// Latencias de la corrida (total y, opcionalmente, por turno).
    pub timings: Timings,
    /// Motivo por el que el loop del agente se detuvo.
    pub stop_reason: StopReason,
    /// Registro auditable de cada invocación de tool.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Transcripción normalizada por mensaje (proyección estable).
    pub transcript: Vec<TranscriptEntry>,
    /// Objeto MAGI opaco de una pasada `consult`; `None` si no hubo.
    pub consult: Option<serde_json::Value>,
    /// Límites efectivos tras aplicar techos del operador y clamps del envelope.
    pub applied_caps: AppliedCaps,
    /// Detalle del error si la corrida falló; `None` en éxito.
    pub error: Option<ErrorPayload>,
}

/// Conteo de tokens consumidos por la corrida.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Tokens de entrada (prompt + contexto + tool results).
    pub input_tokens: u64,
    /// Tokens de salida generados por el modelo.
    pub output_tokens: u64,
}

/// Latencias observadas durante la corrida.
#[derive(Debug, Clone, Serialize)]
pub struct Timings {
    /// Duración total de la corrida en milisegundos.
    pub total_ms: u64,
    /// Time-to-first-byte del primer token, si se midió.
    pub ttfb_ms: Option<u64>,
    /// Duración de cada turno del loop, en orden.
    pub per_turn_ms: Vec<u64>,
}

/// Motivo por el que el loop del agente se detuvo.
///
/// Prioridad cuando co-ocurren (REQ-H14): `Error` > `MaxToolCalls` > `Denied` >
/// `Done`. Serializa en `snake_case` (contrato estable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// El agente completó y produjo su respuesta final.
    Done,
    /// Se alcanzó el tope de invocaciones de tool.
    MaxToolCalls,
    /// Un tool esencial fue denegado por el tier y bloqueó la tarea.
    Denied,
    /// La corrida terminó en error (incluye timeout y guarda repetitiva).
    Error,
}

/// Registro auditable de una única invocación de tool.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallRecord {
    /// Nombre del tool invocado.
    pub name: String,
    /// Input JSON con el que se invocó el tool.
    pub input: serde_json::Value,
    /// Resultado del tool (truncado a `TOOL_RESULT_CAP` por el formateador).
    pub result: String,
    /// Duración de la invocación en milisegundos.
    pub ms: u64,
    /// `true` si el tool corrió con éxito; `false` si falló o fue denegado.
    pub ok: bool,
}

/// Una entrada de la transcripción normalizada (una por mensaje).
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptEntry {
    /// Rol normalizado del mensaje (`assistant`/`user`/`tool`).
    pub role: String,
    /// Contenido textual del mensaje.
    pub content: String,
    /// Invocaciones de tool asociadas al mensaje, si las hubo.
    pub tool_calls: Option<Vec<ToolCallRecord>>,
}

/// Límites efectivos aplicados a la corrida (hace visible el clamp de REQ-H12b).
#[derive(Debug, Clone, Serialize)]
pub struct AppliedCaps {
    /// Tope efectivo de invocaciones de tool tras aplicar el techo del operador.
    pub max_tool_calls: u32,
    /// `true` si el `max_tool_calls` del envelope fue recortado al techo.
    pub max_tool_calls_clamped: bool,
    /// Tope de wall-clock en segundos, si se fijó uno.
    pub timeout_secs: Option<u64>,
    /// `true` si se aplicó el `system` del envelope (requiere el flag del operador).
    pub system_override_applied: bool,
}

/// Detalle estructurado de un error de la corrida en la salida JSON.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorPayload {
    /// Mensaje seguro (ya sanitizado; jamás un secreto crudo).
    pub message: String,
    /// Clase del error, para que el caller ruteé sin parsear el mensaje.
    pub kind: ErrorKind,
}

/// Clase de error de una corrida headless (mapea 1:1 a un exit code, REQ-H14).
///
/// Serializa en `snake_case`. Un valor de enum nuevo es **aditivo** (REQ-H14):
/// el consumidor debe tratar un valor desconocido como catch-all, no romper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Entrada inválida / mal uso de la CLI (→ exit 2).
    InputInvalid,
    /// DB corrupta: datos presentes sin envelope (→ exit 1, never-delete).
    DbCorrupt,
    /// Passphrase incorrecta; reintentable, nunca borra (→ exit 1 en headless).
    WrongPassphrase,
    /// No hay passphrase y no hay TTY para pedirla (→ exit 1).
    PassphraseUnavailable,
    /// Un tool esencial fue denegado por el tier y bloqueó la tarea (→ exit 3).
    TierDenied,
    /// Fallo del proveedor LLM (HTTP/red), con el mensaje sanitizado (→ exit 1).
    Provider,
    /// Se venció el `--timeout` de wall-clock (→ exit 1). Clase de primera clase.
    Timeout,
    /// Cualquier otro error de runtime del agente (→ exit 1).
    Runtime,
}

/// System-prompt efectivo de la corrida junto con su **origen** (REQ-H12b).
///
/// El origen es un límite de seguridad: el `system` del envelope solo se honra
/// si el operador lo habilitó; si no, rige el del operador.
#[derive(Debug, Clone)]
pub enum SystemPolicy {
    /// System-prompt fijado por el operador (default; no overridable por el caller).
    Operator(String),
    /// System-prompt del envelope, aceptado porque el operador lo habilitó.
    CallerOverride(String),
}

impl SystemPolicy {
    /// Texto efectivo del system-prompt, sin importar el origen.
    ///
    /// El runner headless lo consume para poblar `AgentRunConfig::system`
    /// (REQ-H12b) — el origen ya quedó registrado en
    /// `AppliedCaps::system_override_applied` en el momento de la resolución.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Operator(s) | Self::CallerOverride(s) => s,
        }
    }
}

// NOTE: `InputFormat` is defined in `input.rs` (parser-local, `pub` for the
// fuzz target), not here — it is a parser input, not part of the output
// contract. The former duplicate declaration in this module was removed.
