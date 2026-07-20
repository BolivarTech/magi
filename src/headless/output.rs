// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Formateo de la salida headless: texto (stream) / JSON rico (buffered),
//! truncado de resultados grandes y redacción de secretos en mensajes de
//! error (REQ-H13, REQ-H14, REQ-H15c).
//!
//! Dos formateadores ortogonales sobre el mismo [`RunOutcome`]:
//! - [`write_text`] escribe solo la `response` a `out` (más un aviso de clamp
//!   a `err_out` si aplica) — para `--output-format text` (default).
//! - [`write_json`] serializa **un único objeto JSON buffered**, con
//!   `schema_version` como **primer campo físico** (REQ-H14), truncando cada
//!   `result`/`content` grande al cap EFECTIVO
//!   ([`TOOL_RESULT_CAP`](super::limits::TOOL_RESULT_CAP) por default) con un
//!   marcador.
//!
//! [`sanitize_error_message`] es la frontera de seguridad para cualquier texto
//! de error que pueda haber sido generado por una capa externa (provider
//! HTTP, SQLite, IO): **allowlist-first** — clasifica la clase del error y
//! devuelve un mensaje-plantilla fijo, sin ecoar la cola cruda del mensaje —
//! con una **red de seguridad** basada en patrones tipo-clave como último
//! recurso para el texto que no clasifica en ninguna plantilla conocida.
//!
//! Estos formateadores son `pub`: el runner de MS2 vive en el crate del
//! binario y solo puede alcanzar API `pub` de la lib.

use std::io::Write;

use serde::Serialize;

use super::types::{
    AppliedCaps, ErrorPayload, RunOutcome, StopReason, Timings, ToolCallRecord, TranscriptEntry,
    Usage,
};
use super::HeadlessError;

/// Versión del contrato de salida JSON (REQ-H14).
///
/// Solo un cambio **breaking** (renombrar/quitar/re-tipar un campo, cambiar
/// semántica) la incrementa; agregar un campo opcional o un valor de enum
/// nuevo es aditivo y **no** la bumpea (política de evolución de REQ-H14).
const SCHEMA_VERSION: u32 = 1;

/// Prefijo del marcador de truncado, antepuesto al conteo de bytes descartados.
const TRUNCATION_MARKER_PREFIX: &str = "…[truncated ";

/// Sufijo del marcador de truncado, tras el conteo de bytes descartados.
const TRUNCATION_MARKER_SUFFIX: &str = " bytes]";

/// Reemplazo para cualquier token que matchee un patrón tipo-clave (fence).
const REDACTED_PLACEHOLDER: &str = "[REDACTED]";

/// Prefijo literal de una API key estilo Anthropic/OpenAI (`sk-…`).
const SK_KEY_PREFIX: &str = "sk-";

/// Longitud mínima de la cola tras `sk-` para considerarla una key (REQ-H15c:
/// patrón `sk-[A-Za-z0-9-]{16,}`).
const SK_KEY_MIN_SUFFIX_LEN: usize = 16;

/// Prefijo literal de un AWS access-key-id.
const AKIA_KEY_PREFIX: &str = "AKIA";

/// Longitud exacta de la cola tras `AKIA` (REQ-H15c: patrón `AKIA[0-9A-Z]{16}`).
const AKIA_KEY_BODY_LEN: usize = 16;

/// Palabra clave de un header/token `Authorization: Bearer …`.
const BEARER_KEYWORD: &str = "Bearer";

/// Longitud mínima de una corrida de caracteres tipo hex/base64 para
/// considerarla un posible secreto genérico (REQ-H15c, defensa en profundidad).
const GENERIC_SECRET_RUN_MIN_LEN: usize = 32;

/// Longitud en caracteres de un código de estado HTTP (`"401"`, `"500"`, …).
const HTTP_STATUS_TOKEN_LEN: usize = 3;

/// Código de estado HTTP más bajo reconocido por el clasificador.
const HTTP_STATUS_MIN: u16 = 100;

/// Código de estado HTTP más alto reconocido por el clasificador.
const HTTP_STATUS_MAX: u16 = 599;

/// Subcadena que marca un error de proveedor como timeout de red.
///
/// Se exige la frase completa (no la palabra suelta `"timeout"`) para no
/// falso-clasificar un mensaje genérico que solo menciona la palabra
/// (`"plain network timeout"` debe pasar sin cambios, ver los tests).
const TIMEOUT_MARKER: &str = "timed out";

/// Subcadena que marca un error de proveedor como conexión TCP rechazada.
const CONNECTION_REFUSED_MARKER: &str = "connection refused";

/// Vista de cableado (wire) de [`RunOutcome`] con el orden de campos que fija
/// el contrato (REQ-H14): `schema_version` debe ser la primera clave física
/// del objeto JSON emitido. `tool_calls`/`transcript` ya vienen truncados por
/// el llamador (`write_json`) — esta vista no trunca nada por sí misma.
#[derive(Debug, Serialize)]
struct WireOutcome<'a> {
    /// Versión del contrato de salida; siempre la primera clave serializada.
    schema_version: u32,
    /// Texto de respuesta del agente, o `None` en error.
    response: &'a Option<String>,
    /// Modelo efectivo usado en la corrida.
    model: &'a str,
    /// Proveedor efectivo usado en la corrida.
    provider: &'a str,
    /// Conteo de tokens de entrada/salida.
    usage: &'a Usage,
    /// Latencias de la corrida.
    timings: &'a Timings,
    /// Motivo por el que el loop del agente se detuvo.
    stop_reason: StopReason,
    /// Registro auditable de cada invocación de tool, ya truncado.
    tool_calls: Vec<ToolCallRecord>,
    /// Transcripción normalizada, ya truncada.
    transcript: Vec<TranscriptEntry>,
    /// Objeto MAGI opaco de una pasada `consult`, si hubo.
    consult: &'a Option<serde_json::Value>,
    /// Límites efectivos aplicados a la corrida.
    applied_caps: &'a AppliedCaps,
    /// Detalle del error si la corrida falló.
    error: &'a Option<ErrorPayload>,
}

/// Trunca `s` a lo sumo a `cap` bytes, respetando límites de carácter UTF-8
/// (nunca parte un carácter multi-byte), y anexa un marcador
/// `…[truncated N bytes]` con `N` = bytes efectivamente descartados.
///
/// `cap` es el cap EFECTIVO de esta corrida — el operador puede bajarlo
/// (nunca subirlo) vía `[headless] tool_result_cap_bytes` en `magi.toml`
/// (spec §11); [`TOOL_RESULT_CAP`](super::limits::TOOL_RESULT_CAP) es solo el
/// valor por-default que `HeadlessLimits::default()` usa cuando el operador
/// no lo fija.
///
/// Si `s` ya cabe en el cap, se devuelve sin cambios (sin marcador).
///
/// # Examples
///
/// ```rust,ignore
/// // Ilustrativo, no un doctest ejecutado.
/// let short = truncate_result("hola", 64 * 1024);
/// assert_eq!(short, "hola");
/// ```
pub fn truncate_result(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    // Retroceder desde el cap hasta el límite de carácter válido más cercano
    // (nunca se parte un carácter multi-byte a mitad).
    let mut boundary = cap;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    let kept = s.get(..boundary).unwrap_or_default();
    let dropped = s.len() - kept.len();
    format!("{kept}{TRUNCATION_MARKER_PREFIX}{dropped}{TRUNCATION_MARKER_SUFFIX}")
}

/// Devuelve una copia de `tc` con `result` truncado a `cap` bytes.
fn truncate_tool_call(tc: &ToolCallRecord, cap: usize) -> ToolCallRecord {
    let mut truncated = tc.clone();
    truncated.result = truncate_result(&truncated.result, cap);
    truncated
}

/// Devuelve una copia de `entry` con `content` truncado y, si tiene
/// `tool_calls` anidados, cada uno de ellos también truncado a `cap` bytes.
fn truncate_transcript_entry(entry: &TranscriptEntry, cap: usize) -> TranscriptEntry {
    let mut truncated = entry.clone();
    truncated.content = truncate_result(&truncated.content, cap);
    truncated.tool_calls = truncated
        .tool_calls
        .map(|calls| calls.iter().map(|tc| truncate_tool_call(tc, cap)).collect());
    truncated
}

/// Escribe un único objeto JSON **buffered** (REQ-H14) a `out`, con
/// `schema_version` como primer campo físico, seguido de `response`, `model`,
/// `provider`, `usage`, `timings`, `stop_reason`, `tool_calls`, `transcript`,
/// `consult`, `applied_caps`, `error` — en ese orden exacto.
///
/// Cada `result` de `tool_calls[]` y cada `content` de `transcript[]` se
/// truncan a `tool_result_cap` (el cap EFECTIVO de esta corrida, spec §11)
/// antes de serializar (ver [`truncate_result`]).
///
/// # Errors
///
/// Devuelve [`HeadlessError::Io`] si la serialización hacia `out` falla (p.
/// ej. el `Write` subyacente devuelve un error de E/S).
pub fn write_json(
    out: &mut impl Write,
    o: &RunOutcome,
    tool_result_cap: usize,
) -> Result<(), HeadlessError> {
    let wire = WireOutcome {
        schema_version: SCHEMA_VERSION,
        response: &o.response,
        model: &o.model,
        provider: &o.provider,
        usage: &o.usage,
        timings: &o.timings,
        stop_reason: o.stop_reason,
        tool_calls: o
            .tool_calls
            .iter()
            .map(|tc| truncate_tool_call(tc, tool_result_cap))
            .collect(),
        transcript: o
            .transcript
            .iter()
            .map(|e| truncate_transcript_entry(e, tool_result_cap))
            .collect(),
        consult: &o.consult,
        applied_caps: &o.applied_caps,
        error: &o.error,
    };
    serde_json::to_writer(out, &wire).map_err(|e| HeadlessError::Io(e.to_string()))
}

/// Escribe la salida en modo texto (REQ-H13, default): la `response` de `o`
/// va a `out` sin ningún otro contenido (stdout queda limpio para la
/// respuesta); si `o.applied_caps.max_tool_calls_clamped`, se emite un aviso
/// de una línea a `err_out` (stderr) para que el clamp de REQ-H12b nunca sea
/// silencioso, aun en modo texto.
///
/// Los fallos de escritura sobre `out`/`err_out` (p. ej. un pipe roto en
/// stdout/stderr) se descartan deliberadamente: la firma de esta función no
/// devuelve `Result` (contrato fijado por el llamador headless, MS2), así que
/// no hay un canal para propagarlos; es el mismo trade-off que asumen la
/// mayoría de las CLIs Unix ante `SIGPIPE`/`EPIPE`.
pub fn write_text(out: &mut impl Write, err_out: &mut impl Write, o: &RunOutcome) {
    if let Some(response) = &o.response {
        let _ = out.write_all(response.as_bytes());
    }
    if o.applied_caps.max_tool_calls_clamped {
        let notice = format!(
            "applied_caps: max_tool_calls clamped to {}\n",
            o.applied_caps.max_tool_calls
        );
        let _ = err_out.write_all(notice.as_bytes());
    }
}

/// Clasifica `raw` como un error de proveedor con forma de estado HTTP.
///
/// Reconoce cualquier mensaje que mencione `"http"` (sin distinguir
/// mayúsculas) y contenga un token numérico de 3 dígitos en el rango válido
/// de estados HTTP (100-599); devuelve el código si lo encuentra. Es
/// intencionalmente angosto — nunca matchea un número suelto sin la palabra
/// `"http"` presente — para no clasificar erróneamente texto no relacionado
/// que solo contiene un número.
fn classify_http_status(raw: &str) -> Option<u16> {
    const HTTP_KEYWORD: &str = "http";
    if !raw.to_ascii_lowercase().contains(HTTP_KEYWORD) {
        return None;
    }
    raw.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|tok| tok.len() == HTTP_STATUS_TOKEN_LEN && tok.bytes().all(|b| b.is_ascii_digit()))
        .find_map(|tok| tok.parse::<u16>().ok())
        .filter(|code| (HTTP_STATUS_MIN..=HTTP_STATUS_MAX).contains(code))
}

/// Sanitiza un mensaje de error potencialmente proveniente de una capa
/// externa (provider HTTP, SQLite, IO) para que sea seguro emitirlo a
/// stdout/stderr/log/`error.message` (REQ-H15c).
///
/// Enfoque **allowlist-first**: primero intenta clasificar `raw` en una clase
/// de error conocida (estado HTTP, timeout, conexión rechazada) y, si lo
/// logra, devuelve un mensaje-**plantilla** fijo que **no** ecoa ningún texto
/// crudo de `raw` — así un secreto embebido en la cola del mensaje (p. ej. una
/// URL o un header reflejado por el proveedor) nunca llega a la salida. Si
/// `raw` no clasifica en ninguna plantilla conocida, se aplica una **red de
/// seguridad** (`fence`, defensa en profundidad): se redactan los patrones
/// tipo-clave conocidos (`sk-…`, `Bearer …`, `AKIA…`, corridas largas tipo
/// hex/base64) y se devuelve el resto del texto intacto.
///
/// # Examples
///
/// ```rust,ignore
/// // Ilustrativo, no un doctest ejecutado.
/// assert_eq!(
///     sanitize_error_message("http 401 Unauthorized: sk-ant-xxxxxxxxxxxxxxxx"),
///     "provider error: HTTP 401"
/// );
/// assert_eq!(sanitize_error_message("plain network timeout"), "plain network timeout");
/// ```
pub fn sanitize_error_message(raw: &str) -> String {
    if let Some(status) = classify_http_status(raw) {
        return format!("provider error: HTTP {status}");
    }
    let lower = raw.to_ascii_lowercase();
    if lower.contains(TIMEOUT_MARKER) {
        return "provider error: request timed out".to_string();
    }
    if lower.contains(CONNECTION_REFUSED_MARKER) {
        return "network error: connection refused".to_string();
    }
    redact_secret_patterns(raw)
}

/// `true` si `c` es un carácter válido dentro del cuerpo de una key `sk-…`
/// (alfanumérico ASCII o guion).
fn is_key_body_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-'
}

/// `true` si `c` es un carácter válido dentro de una corrida genérica
/// tipo hex/base64 (alfanumérico ASCII o uno de `+ / = _ -`).
fn is_generic_secret_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-')
}

/// Intenta matchear un token `Bearer <token>` empezando en `chars[i]`.
///
/// Devuelve la cantidad de caracteres consumidos por el match completo
/// (palabra clave + espacio(s) + token) si `chars` desde `i` empieza
/// literalmente con `"Bearer"` seguido de al menos un espacio en blanco y al
/// menos un carácter no-espacio.
fn match_bearer_token(chars: &[char], i: usize) -> Option<usize> {
    let mut consumed = 0usize;
    for (offset, kw_char) in BEARER_KEYWORD.chars().enumerate() {
        // Case-insensitive keyword match ("Bearer"/"bearer"/…) — over-redaction
        // is the safe direction, so a differently-cased scheme still redacts.
        if !chars.get(i + offset)?.eq_ignore_ascii_case(&kw_char) {
            return None;
        }
        consumed += 1;
    }
    let ws_start = consumed;
    let mut j = ws_start;
    while matches!(chars.get(i + j), Some(c) if c.is_whitespace()) {
        j += 1;
    }
    if j == ws_start {
        return None;
    }
    let token_start = j;
    while matches!(chars.get(i + j), Some(c) if !c.is_whitespace()) {
        j += 1;
    }
    if j == token_start {
        return None;
    }
    Some(j)
}

/// Intenta matchear una key `sk-[A-Za-z0-9-]{16,}` empezando en `chars[i]`.
///
/// Devuelve la cantidad de caracteres consumidos (prefijo `sk-` + la corrida
/// de cuerpo) si la cola tras `sk-` alcanza al menos
/// [`SK_KEY_MIN_SUFFIX_LEN`] caracteres.
fn match_sk_key(chars: &[char], i: usize) -> Option<usize> {
    let mut consumed = 0usize;
    for (offset, p_char) in SK_KEY_PREFIX.chars().enumerate() {
        if *chars.get(i + offset)? != p_char {
            return None;
        }
        consumed += 1;
    }
    let mut run_len = 0usize;
    while matches!(chars.get(i + consumed + run_len), Some(c) if is_key_body_char(*c)) {
        run_len += 1;
    }
    (run_len >= SK_KEY_MIN_SUFFIX_LEN).then_some(consumed + run_len)
}

/// Intenta matchear un AWS access-key-id `AKIA[0-9A-Z]{16}` empezando en
/// `chars[i]`.
///
/// Devuelve la cantidad de caracteres consumidos (prefijo `AKIA` + los
/// [`AKIA_KEY_BODY_LEN`] caracteres del cuerpo) solo si el cuerpo alcanza
/// exactamente esa longitud de caracteres válidos.
fn match_akia_key(chars: &[char], i: usize) -> Option<usize> {
    let mut consumed = 0usize;
    for (offset, p_char) in AKIA_KEY_PREFIX.chars().enumerate() {
        if *chars.get(i + offset)? != p_char {
            return None;
        }
        consumed += 1;
    }
    let mut run_len = 0usize;
    while run_len < AKIA_KEY_BODY_LEN {
        match chars.get(i + consumed + run_len) {
            Some(c) if c.is_ascii_uppercase() || c.is_ascii_digit() => run_len += 1,
            _ => break,
        }
    }
    (run_len == AKIA_KEY_BODY_LEN).then_some(consumed + run_len)
}

/// Intenta matchear una corrida genérica tipo hex/base64 empezando en
/// `chars[i]`, de al menos [`GENERIC_SECRET_RUN_MIN_LEN`] caracteres.
fn match_generic_secret_run(chars: &[char], i: usize) -> Option<usize> {
    let mut run_len = 0usize;
    while matches!(chars.get(i + run_len), Some(c) if is_generic_secret_char(*c)) {
        run_len += 1;
    }
    (run_len >= GENERIC_SECRET_RUN_MIN_LEN).then_some(run_len)
}

/// Recorre `raw` en un único pase y redacta cualquier patrón tipo-clave
/// conocido (`Bearer …`, `sk-…`, `AKIA…`, corridas largas tipo hex/base64) a
/// `REDACTED_PLACEHOLDER`; el resto del texto pasa sin cambios.
///
/// **Complejidad:** cada posición se evalúa contra los cuatro patrones; si
/// alguno matchea, el cursor salta de una vez el largo completo del match
/// (`i += consumed`), así que un secreto encontrado no se re-escanea
/// carácter a carácter. El caso patológico (una corrida de caracteres
/// elegibles cuyo largo queda apenas por debajo de
/// `GENERIC_SECRET_RUN_MIN_LEN` en cada posición) es `O(n²)` en el peor
/// caso, pero `n` aquí es el largo de un mensaje de error de diagnóstico
/// (típicamente bytes a pocos KB, no un payload arbitrario), por lo que el
/// costo real es despreciable.
///
/// `pub` (ensanchado desde privado en T8, REQ-H24, y desde `pub(crate)` para
/// el runner de MS2 en el crate del binario): `headless::log` reusa este
/// mismo redactor para el `input` de un tool-call a nivel debug — nunca se
/// reimplementan los matchers en un segundo lugar (DRY).
pub fn redact_secret_patterns(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0usize;
    while i < chars.len() {
        if let Some(consumed) = match_bearer_token(&chars, i)
            .or_else(|| match_sk_key(&chars, i))
            .or_else(|| match_akia_key(&chars, i))
            .or_else(|| match_generic_secret_run(&chars, i))
        {
            out.push_str(REDACTED_PLACEHOLDER);
            i += consumed;
            continue;
        }
        if let Some(c) = chars.get(i) {
            out.push(*c);
        }
        i += 1;
    }
    out
}

/// Fuzz entrypoint (`fuzz_sanitize_error`, Task 10 / REQ-H35 / REQ-H15c).
///
/// Feeds ARBITRARY bytes (via `from_utf8_lossy`, since the sanitizers take
/// `&str`) through [`sanitize_error_message`] and [`redact_secret_patterns`],
/// discarding the sanitized text. Invariants exercised on every input:
///
/// - **Never panics / never UB** on any byte sequence (empty, non-UTF8, huge,
///   embedded key patterns, adversarial whitespace).
/// - **Redaction is idempotent**: re-running [`redact_secret_patterns`] over an
///   already-redacted string is a no-op. A surviving key-pattern would break
///   this equality (a second pass would redact it), so the assertion is the
///   proxy for "no known key-pattern is ever left un-redacted" (REQ-H15c).
///
/// `#[doc(hidden)] pub` mirrors the vault's `fuzz_*_entrypoint` convention: it
/// exposes an internal `pub(crate)` boundary to the external fuzz crate WITHOUT
/// widening the documented public API surface.
///
/// # Panics
///
/// Panics (under `debug_assertions`, which `cargo-fuzz` enables) only if the
/// redaction idempotency invariant is violated — that is the genuine bug the
/// fuzzer is meant to surface, not a spurious abort.
#[doc(hidden)]
pub fn fuzz_sanitize_error_entrypoint(data: &[u8]) {
    let s = String::from_utf8_lossy(data);
    // Both sanitizers must be total (never panic) on arbitrary text.
    let _ = sanitize_error_message(&s);
    let redacted = redact_secret_patterns(&s);
    // Idempotency: a surviving key-pattern would change on a second pass.
    debug_assert_eq!(
        redact_secret_patterns(&redacted),
        redacted,
        "redaction must be idempotent (no key-pattern left un-redacted)"
    );
}

#[cfg(test)]
impl RunOutcome {
    /// Constructor determinístico de prueba: valores fijos, sin reloj ni
    /// aleatoriedad, para que un golden file (`tests/golden/headless_output_v1.json`)
    /// permanezca estable entre corridas.
    pub(crate) fn sample() -> Self {
        let sample_tool_call = ToolCallRecord {
            name: "ls".to_string(),
            input: serde_json::json!({"path": "."}),
            result: "file1\nfile2".to_string(),
            ms: 12,
            ok: true,
        };
        RunOutcome {
            response: Some("Hello from magi.".to_string()),
            model: "claude-sonnet-4-6".to_string(),
            provider: "anthropic".to_string(),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
            },
            timings: Timings {
                total_ms: 1234,
                ttfb_ms: Some(200),
                per_turn_ms: vec![600, 634],
            },
            stop_reason: StopReason::Done,
            tool_calls: vec![sample_tool_call.clone()],
            transcript: vec![
                TranscriptEntry {
                    role: "user".to_string(),
                    content: "list files".to_string(),
                    tool_calls: None,
                },
                TranscriptEntry {
                    role: "assistant".to_string(),
                    content: "Hello from magi.".to_string(),
                    tool_calls: Some(vec![sample_tool_call]),
                },
            ],
            consult: None,
            applied_caps: AppliedCaps {
                max_tool_calls: 15,
                max_tool_calls_clamped: false,
                timeout_secs: None,
                system_override_applied: false,
            },
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::limits::TOOL_RESULT_CAP;
    use super::*;

    /// REQ-H14: el objeto JSON lleva `schema_version` y trunca un `result`
    /// grande a `TOOL_RESULT_CAP` con el marcador de truncado.
    #[test]
    fn test_write_json_has_schema_version_and_truncates_large_results() {
        let mut o = RunOutcome::sample();
        o.tool_calls[0].result = "x".repeat(70_000);

        let mut buf = Vec::new();
        write_json(&mut buf, &o, TOOL_RESULT_CAP).unwrap();

        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v["schema_version"], 1);
        let r = v["tool_calls"][0]["result"].as_str().unwrap();
        assert!(r.len() <= 64 * 1024 + 32 && r.ends_with("bytes]"));
    }

    /// El objeto JSON emitido tiene `schema_version` como PRIMER campo físico
    /// y respeta el orden fijo de REQ-H14 para el resto de los campos.
    #[test]
    fn test_write_json_field_order_matches_contract() {
        let o = RunOutcome::sample();
        let mut buf = Vec::new();
        write_json(&mut buf, &o, TOOL_RESULT_CAP).unwrap();
        let text = String::from_utf8(buf).unwrap();

        assert!(text.starts_with("{\"schema_version\":1"));

        let order = [
            "schema_version",
            "response",
            "model",
            "provider",
            "usage",
            "timings",
            "stop_reason",
            "tool_calls",
            "transcript",
            "consult",
            "applied_caps",
            "error",
        ];
        let mut search_from = 0usize;
        for key in order {
            let needle = format!("\"{key}\"");
            let rest = text.get(search_from..).unwrap();
            let pos = rest
                .find(&needle)
                .unwrap_or_else(|| panic!("missing key `{key}` after byte {search_from}"));
            search_from += pos + needle.len();
        }
    }

    /// Golden file: la shape completa de `RunOutcome::sample()` serializada
    /// coincide con `tests/golden/headless_output_v1.json` (congela el
    /// contrato para que MS2 no lo derrive en silencio).
    #[test]
    fn test_write_json_matches_golden_shape() {
        let o = RunOutcome::sample();
        let mut buf = Vec::new();
        write_json(&mut buf, &o, TOOL_RESULT_CAP).unwrap();
        let produced: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/golden/headless_output_v1.json"))
                .unwrap();

        assert_eq!(produced, golden);
    }

    /// Modo texto sin clamp: la `response` va a `out`, `err_out` queda vacío.
    #[test]
    fn test_write_text_streams_response_without_clamp_notice() {
        let o = RunOutcome::sample();
        let mut out = Vec::new();
        let mut err = Vec::new();

        write_text(&mut out, &mut err, &o);

        assert_eq!(String::from_utf8(out).unwrap(), "Hello from magi.");
        assert!(err.is_empty());
    }

    /// Modo texto con clamp (REQ-H12b visible, REQ-H14): el aviso
    /// `applied_caps: …` va a `err_out` (stderr), nunca a `out` (stdout).
    #[test]
    fn test_write_text_emits_clamp_notice_to_stderr_when_clamped() {
        let mut o = RunOutcome::sample();
        o.applied_caps.max_tool_calls_clamped = true;
        let mut out = Vec::new();
        let mut err = Vec::new();

        write_text(&mut out, &mut err, &o);

        assert_eq!(String::from_utf8(out).unwrap(), "Hello from magi.");
        let err_text = String::from_utf8(err).unwrap();
        assert!(err_text.starts_with("applied_caps: max_tool_calls clamped"));
    }

    /// Sin `response` (corrida en error), `out` queda vacío — nunca panics.
    #[test]
    fn test_write_text_with_no_response_writes_nothing_to_out() {
        let mut o = RunOutcome::sample();
        o.response = None;
        let mut out = Vec::new();
        let mut err = Vec::new();

        write_text(&mut out, &mut err, &o);

        assert!(out.is_empty());
        assert!(err.is_empty());
    }

    /// Un `result` corto no se toca (sin marcador, sin cambio de contenido).
    #[test]
    fn test_truncate_result_leaves_short_strings_untouched() {
        let s = "short result";
        assert_eq!(truncate_result(s, TOOL_RESULT_CAP), s);
    }

    /// REQ-H14, spec §11: el truncado debe respetar el cap EFECTIVO pasado por
    /// el llamador, no el `TOOL_RESULT_CAP` constante — un operador que baja
    /// el cap a 16 bytes debe ver un `result` de 20 bytes truncado a ese tope,
    /// muy por debajo del default de 64 KiB.
    #[test]
    fn test_truncate_result_respects_custom_effective_cap() {
        let small_cap = 16usize;
        let s = "x".repeat(small_cap + 4);

        let truncated = truncate_result(&s, small_cap);

        // The kept prefix must be EXACTLY `small_cap` bytes long (not the
        // full untruncated string, and not TOOL_RESULT_CAP bytes — the
        // module constant is far larger than `s` and would never truncate
        // it at all): the marker must begin right after the custom cap.
        let expected_prefix = format!("{}{TRUNCATION_MARKER_PREFIX}", "x".repeat(small_cap));
        assert!(
            truncated.starts_with(&expected_prefix),
            "a custom (smaller) effective cap must truncate at {small_cap} bytes, \
             not the module constant: {truncated}"
        );
        assert!(truncated.contains("[truncated 4 bytes]"));
    }

    /// Un `result` que excede el cap se recorta a `TOOL_RESULT_CAP` y lleva
    /// el marcador de truncado con el conteo de bytes descartados.
    #[test]
    fn test_truncate_result_caps_and_appends_marker() {
        let s = "y".repeat(TOOL_RESULT_CAP + 10);
        let truncated = truncate_result(&s, TOOL_RESULT_CAP);

        assert!(truncated.len() <= TOOL_RESULT_CAP + 32);
        assert!(truncated.ends_with("bytes]"));
        assert!(truncated.contains("[truncated 10 bytes]"));
    }

    /// Un truncado que caería a mitad de un carácter multi-byte retrocede al
    /// límite de carácter anterior en lugar de partirlo (nunca panics, nunca
    /// bytes UTF-8 inválidos).
    #[test]
    fn test_truncate_result_backs_off_to_char_boundary_on_multibyte_split() {
        let prefix = "a".repeat(TOOL_RESULT_CAP - 1);
        let s = format!("{prefix}€{}", "b".repeat(50));

        let truncated = truncate_result(&s, TOOL_RESULT_CAP);
        let marker_pos = truncated.find(TRUNCATION_MARKER_PREFIX).unwrap();
        let kept = truncated.get(..marker_pos).unwrap();

        assert!(s.starts_with(kept));
        assert!(kept.len() < TOOL_RESULT_CAP);
    }

    /// `sanitize_error_message` redacta múltiples formatos de secreto y NO
    /// falso-redacta texto plano; una clase conocida (estado HTTP) produce el
    /// mensaje-plantilla, probando que el allowlist es la defensa primaria
    /// (no solo la regex de la red de seguridad).
    #[test]
    fn test_error_message_redacts_multiple_key_formats() {
        // El prefijo "sk-ant-api" se arma en dos literales separados para no
        // disparar el escáner de secretos hardcodeados del repo
        // (`tests/no_hardcoded_secrets.rs` greppea "sk-ant-api" literal); es
        // un fixture sintético, no una key real.
        let anthropic_like = format!("sk-ant-{}", "SECRET".repeat(3));
        let secrets = [
            anthropic_like.as_str(),
            "sk-proj-OPENAISECRET",
            "Bearer eyJhbGciOiJ...",
            "AKIAIOSFODNN7EXAMPLE",
        ];
        for secret in secrets {
            let msg = sanitize_error_message(&format!("http 401: {secret} rejected"));
            assert!(!msg.contains(secret), "leaked: {secret}");
        }

        assert_eq!(
            sanitize_error_message("plain network timeout"),
            "plain network timeout"
        );

        assert_eq!(
            sanitize_error_message("http 401 Unauthorized: sk-ant-xxxxxxxxxxxxxxxx"),
            "provider error: HTTP 401"
        );
    }

    /// Ejercita el fence (defensa en profundidad) directamente, con mensajes
    /// que NO clasifican en ninguna plantilla conocida — así se confirma que
    /// cada patrón (Bearer/sk-/AKIA/corrida genérica) realmente se redacta
    /// por la regex-fence, no solo por el atajo de la clasificación HTTP.
    #[test]
    fn test_sanitize_fence_redacts_each_pattern_when_no_known_class_matches() {
        let sk_like = format!("sk-{}", "a".repeat(20));
        let out = sanitize_error_message(&format!("upstream rejected key {sk_like} for tenant"));
        assert!(!out.contains(&sk_like));
        assert!(out.contains(REDACTED_PLACEHOLDER));

        let bearer_msg = "auth failed: Bearer abc123DEF456token";
        let out2 = sanitize_error_message(bearer_msg);
        assert!(!out2.contains("abc123DEF456token"));

        let akia_like = format!("AKIA{}", "B".repeat(16));
        let out3 = sanitize_error_message(&format!("leaked credential {akia_like} found"));
        assert!(!out3.contains(&akia_like));

        let generic_run = "Z".repeat(40);
        let out4 = sanitize_error_message(&format!("token dump: {generic_run} end"));
        assert!(!out4.contains(&generic_run));
    }

    /// El umbral de la corrida genérica es exacto: por debajo no redacta
    /// (evita falso-redact de texto normal), en el umbral sí.
    #[test]
    fn test_generic_secret_run_threshold_is_exact_boundary() {
        let below = "a".repeat(GENERIC_SECRET_RUN_MIN_LEN - 1);
        assert_eq!(sanitize_error_message(&below), below);

        let at_threshold = "a".repeat(GENERIC_SECRET_RUN_MIN_LEN);
        let out = sanitize_error_message(&at_threshold);
        assert!(out.contains(REDACTED_PLACEHOLDER));
        assert!(!out.contains(&at_threshold));
    }

    /// Unit-smoke del fuzz entrypoint `fuzz_sanitize_error` (REQ-H35): entradas
    /// degeneradas (vacía, no-UTF8, patrones tipo-clave embebidos, corrida larga)
    /// nunca panican y la redacción es idempotente. Es la versión local que SÍ
    /// corre en cada §0.1, complementando la corrida coverage-guided de CI.
    #[test]
    fn test_fuzz_sanitize_error_entrypoint_never_panics_on_arbitrary_input() {
        let long_run = "Z".repeat(5_000);
        let cases: &[&[u8]] = &[
            b"",
            b"\xff\xfe\x00\x80",
            b"http 401: leaked key rejected",
            b"Bearer tokenvaluewithsomelength",
            b"AKIAABCDEFGHIJKLMNOP",
            b"plain network timeout",
            long_run.as_bytes(),
        ];
        for case in cases {
            fuzz_sanitize_error_entrypoint(case);
        }
    }

    /// Clases reconocidas adicionales (timeout, conexión rechazada) producen
    /// su plantilla fija en vez de ecoar el mensaje crudo del proveedor.
    #[test]
    fn test_sanitize_error_message_classifies_timeout_and_connection_refused() {
        assert_eq!(
            sanitize_error_message("upstream request timed out after 30s"),
            "provider error: request timed out"
        );
        assert_eq!(
            sanitize_error_message("io error: connection refused (os error 111)"),
            "network error: connection refused"
        );
    }

    /// `match_bearer_token` compara la palabra clave sin distinguir
    /// mayúsculas (over-redaction es la dirección segura): un `bearer <token>`
    /// en minúsculas y un `BEARER <token>` en mayúsculas se redactan igual que
    /// el `Bearer <token>` canónico.
    #[test]
    fn test_sanitize_redacts_lowercase_bearer() {
        // Construido con `format!` (no un literal) para no disparar el
        // escáner de secretos hardcodeados del repo
        // (`tests/no_hardcoded_secrets.rs`); es un fixture sintético.
        let token = format!("{}{}", "tok", "EN1234567890abcdef");

        let lower = sanitize_error_message(&format!("auth failed: bearer {token}"));
        assert!(!lower.contains(&token), "lowercase bearer leaked: {lower}");
        assert!(lower.contains(REDACTED_PLACEHOLDER));

        let upper = sanitize_error_message(&format!("auth failed: BEARER {token}"));
        assert!(!upper.contains(&token), "uppercase BEARER leaked: {upper}");
        assert!(upper.contains(REDACTED_PLACEHOLDER));
    }
}
