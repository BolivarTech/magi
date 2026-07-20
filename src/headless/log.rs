// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Logs de corrida en JSONL (`./.magi/logs/run-<ts>-<pid>-<rand>.jsonl`,
//! REQ-H24/H34): retención por conteo **y** tamaño ordenada por timestamp de
//! nombre de archivo (nunca por `mtime`), y redacción del `input` crudo de un
//! tool-call según la verbosidad configurada (REQ-H24 — el cap de tamaño y la
//! redacción son controles **distintos**, ninguno sustituye al otro).
//!
//! - [`RunLog::start`] descubre/crea el directorio de logs, poda los runs
//!   viejos (best-effort, tolerante a carreras de `unlink`) y toca el archivo
//!   de esta corrida.
//! - [`RunLog::event`] filtra por [`LogLevel`] y escribe una línea JSON por
//!   [`LogEvent`]: a nivel `info` (o menos verboso) un `ToolCall` solo lleva
//!   `name`/`ok`/`ms`/`input_len`; a nivel `debug` el `input` se **redacta primero**
//!   ([`redact_secret_patterns`], reusado de `output.rs`) y **luego** se **capa**
//!   ([`truncate_result`], reusado de `output.rs`) — este orden evita que un secreto
//!   partido por el límite de truncado filtre un prefijo parcial; nunca se
//!   re-implementan los matchers. El `prompt`/envelope crudo **nunca** se
//!   loguea, a ningún nivel ni en ningún campo.
//!
//! `RunLog`/`LogLevel`/`LogEvent` son `pub`: el runner de MS2 vive en el
//! crate del binario y solo puede alcanzar API `pub` de la lib.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::random;
use serde::Serialize;

use super::output::{redact_secret_patterns, truncate_result};
use super::HeadlessError;

/// Prefijo literal de todo nombre de archivo de log de corrida.
const LOG_FILENAME_PREFIX: &str = "run-";

/// Sufijo literal de todo nombre de archivo de log de corrida.
const LOG_FILENAME_SUFFIX: &str = ".jsonl";

/// Separador entre los tres segmentos (`<ts>-<pid>-<rand>`) del nombre.
const LOG_FILENAME_SEPARATOR: char = '-';

/// Ancho, en dígitos decimales, del `<ts>` (epoch-millis) embebido en el
/// nombre del archivo. `u64::MAX` tiene 20 dígitos decimales, así que este
/// ancho garantiza que el orden lexicográfico del nombre coincida con el
/// orden cronológico indefinidamente (REQ-H24).
const TIMESTAMP_WIDTH: usize = 20;

/// Ancho, en dígitos hexadecimales, del sufijo aleatorio del nombre (`u32` =
/// 4 bytes = 8 dígitos hex) — evita colisiones entre corridas iniciadas
/// dentro del mismo milisegundo, incluso con el mismo PID (REQ-H24).
const RAND_SUFFIX_HEX_WIDTH: usize = 8;

/// Valor del campo `"kind"` para una línea [`LogEvent::Message`].
const EVENT_KIND_MESSAGE: &str = "message";

/// Valor del campo `"kind"` para una línea [`LogEvent::ToolCall`].
const EVENT_KIND_TOOL_CALL: &str = "tool_call";

/// Verbosidad configurada de un [`RunLog`], ordenada de menos a más
/// detallada (`Error < Warn < Info < Debug`, REQ-H24 `--log-level`).
///
/// Un [`LogEvent`] cuya propia severidad es **más verbosa** que la
/// verbosidad configurada del [`RunLog`] se filtra y nunca se escribe
/// (pasa el filtro sii `evento.level() <= configurado`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    /// Solo fallos irrecuperables.
    Error,
    /// Condiciones recuperables pero dignas de aviso (p. ej. un clamp, una
    /// denegación por tier).
    Warn,
    /// Avisos operacionales normales (default, REQ-H24) — incluye metadata
    /// de tool-call pero **nunca** el `input` crudo (redacción REQ-H24).
    Info,
    /// Máxima verbosidad — incluye el `input` crudo de un tool-call, capado
    /// y redactado (nunca sin pasar por el redactor).
    Debug,
}

/// Un registro de un archivo `run-*.jsonl` descubierto durante un escaneo de
/// retención (REQ-H24/H34).
struct LogFileEntry {
    /// Ruta completa del archivo.
    path: PathBuf,
    /// Timestamp epoch-millis parseado del nombre; `None` si el nombre no
    /// matchea el patrón esperado — se trata como el **más viejo** posible
    /// (máxima prioridad de poda, ver [`prune_retention`]).
    ts: Option<u64>,
    /// Tamaño en bytes del archivo (o `0` si no se pudo leer su metadata).
    size_bytes: u64,
}

/// Una ocurrencia a registrar como una línea JSON en el log de la corrida.
///
/// Cada variante lleva su propia severidad ([`LogLevel`]) para el filtro de
/// verbosidad; es independiente de la verbosidad **configurada** del
/// [`RunLog`], que en cambio gobierna la redacción del `input` de un
/// `ToolCall` (REQ-H24).
#[derive(Debug, Clone)]
pub enum LogEvent<'a> {
    /// Un diagnóstico de texto libre (avisos de arranque, clamps, notices).
    /// **Nunca** debe ser el `prompt`/envelope crudo del caller (REQ-H11/H24).
    Message {
        /// Severidad de esta ocurrencia.
        level: LogLevel,
        /// Texto legible del mensaje.
        text: &'a str,
    },
    /// El registro de una única invocación de tool.
    ToolCall {
        /// Severidad de esta ocurrencia.
        level: LogLevel,
        /// Nombre del tool invocado.
        name: &'a str,
        /// `true` si la invocación tuvo éxito.
        ok: bool,
        /// Duración de la invocación en milisegundos.
        ms: u64,
        /// Input crudo del tool (JSON serializado u otro texto); solo se
        /// incluye en claro (capado + redactado) si la verbosidad
        /// **configurada** del `RunLog` es [`LogLevel::Debug`] — en caso
        /// contrario solo se registra su longitud en bytes (REQ-H24).
        input: &'a str,
    },
}

impl LogEvent<'_> {
    /// Severidad de esta ocurrencia, usada para filtrar contra la
    /// verbosidad configurada de un [`RunLog`].
    fn level(&self) -> LogLevel {
        match self {
            LogEvent::Message { level, .. } | LogEvent::ToolCall { level, .. } => *level,
        }
    }

    /// Renderiza esta ocurrencia como una línea JSON, dada la verbosidad
    /// **configurada** de un [`RunLog`] (gobierna si el `input` de un
    /// `ToolCall` va crudo-capado-y-redactado o solo como su longitud) y el
    /// cap EFECTIVO `tool_result_cap` (spec §11) aplicado al `input` redactado.
    ///
    /// # Errors
    ///
    /// Devuelve [`HeadlessError::Io`] si la serialización a JSON falla (en la
    /// práctica, nunca para estas estructuras — no llevan claves de mapa
    /// no-string ni floats no-finitos).
    fn render(
        &self,
        configured_level: LogLevel,
        tool_result_cap: usize,
    ) -> Result<String, HeadlessError> {
        let ts = current_epoch_millis();
        let serialized = match self {
            LogEvent::Message { level, text } => serde_json::to_string(&MessageLine {
                ts,
                level: *level,
                kind: EVENT_KIND_MESSAGE,
                message: text,
            }),
            LogEvent::ToolCall {
                level,
                name,
                ok,
                ms,
                input,
            } => {
                let (input_field, input_len_field) = if configured_level == LogLevel::Debug {
                    // Redact BEFORE truncating: redacting the full input first means a
                    // secret straddling the truncation boundary cannot be split into an
                    // un-matchable partial that leaks; then truncate the secret-free result.
                    (
                        Some(truncate_result(
                            &redact_secret_patterns(input),
                            tool_result_cap,
                        )),
                        None,
                    )
                } else {
                    (None, Some(input.len()))
                };
                serde_json::to_string(&ToolCallLine {
                    ts,
                    level: *level,
                    kind: EVENT_KIND_TOOL_CALL,
                    name,
                    ok: *ok,
                    ms: *ms,
                    input: input_field,
                    input_len: input_len_field,
                })
            }
        };
        serialized.map_err(|e| HeadlessError::Io(e.to_string()))
    }
}

/// Shape JSON de una línea [`LogEvent::Message`].
#[derive(Debug, Serialize)]
struct MessageLine<'a> {
    /// Epoch-millis de la escritura de esta línea.
    ts: u64,
    /// Severidad de la ocurrencia.
    level: LogLevel,
    /// Discriminador de tipo de línea (`"message"`).
    kind: &'static str,
    /// Texto del mensaje.
    message: &'a str,
}

/// Shape JSON de una línea [`LogEvent::ToolCall`].
#[derive(Debug, Serialize)]
struct ToolCallLine<'a> {
    /// Epoch-millis de la escritura de esta línea.
    ts: u64,
    /// Severidad de la ocurrencia.
    level: LogLevel,
    /// Discriminador de tipo de línea (`"tool_call"`).
    kind: &'static str,
    /// Nombre del tool invocado.
    name: &'a str,
    /// `true` si la invocación tuvo éxito.
    ok: bool,
    /// Duración de la invocación en milisegundos.
    ms: u64,
    /// `input` crudo (capado + redactado), presente **solo** a nivel debug.
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    /// Longitud en bytes del `input`, presente cuando `input` está ausente
    /// (niveles distintos de debug) — REQ-H24: el cap no sustituye la
    /// redacción, así que a niveles no-debug directamente no se emite nada
    /// del contenido, solo su tamaño.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_len: Option<usize>,
}

/// Log de una corrida headless, en JSONL (REQ-H24/H34).
#[derive(Debug)]
pub struct RunLog {
    /// Ruta del archivo JSONL de esta corrida.
    path: PathBuf,
    /// Verbosidad configurada: eventos más verbosos que este nivel se filtran.
    level: LogLevel,
    /// Cap EFECTIVO (spec §11) aplicado al `input` redactado de un
    /// `ToolCall` a nivel debug (`[headless] tool_result_cap_bytes`).
    tool_result_cap: usize,
}

impl RunLog {
    /// Arranca el log de una corrida nueva bajo `logs_dir`: crea el
    /// directorio si falta, poda los runs viejos (best-effort, REQ-H34) hasta
    /// los caps EFECTIVOS `retention_runs`/`max_log_bytes` (spec §11, un
    /// operador puede bajarlos vía `[headless] log_retention`/`log_max_bytes`),
    /// y toca el archivo `run-<ts>-<pid>-<rand>.jsonl` de esta corrida.
    /// `tool_result_cap` es el cap EFECTIVO aplicado al `input` redactado de
    /// un `ToolCall` logueado a nivel debug (`[headless] tool_result_cap_bytes`).
    ///
    /// # Errors
    ///
    /// Devuelve [`HeadlessError::Io`] si no se puede crear `logs_dir` o abrir
    /// el archivo de esta corrida. La poda de retención en sí **nunca**
    /// produce un error propagado — es best-effort (REQ-H34).
    pub fn start(
        logs_dir: &Path,
        level: LogLevel,
        retention_runs: usize,
        max_log_bytes: u64,
        tool_result_cap: usize,
    ) -> Result<Self, HeadlessError> {
        fs::create_dir_all(logs_dir).map_err(|e| HeadlessError::Io(e.to_string()))?;
        prune_retention(logs_dir, retention_runs, max_log_bytes);

        let path = logs_dir.join(generate_log_file_name());
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| HeadlessError::Io(e.to_string()))?;

        Ok(Self {
            path,
            level,
            tool_result_cap,
        })
    }

    /// Registra `ev` como una línea JSON si pasa el filtro de verbosidad
    /// (`ev.level() <= self.level`); un evento filtrado es un no-op exitoso.
    ///
    /// `&mut self`: la firma sigue la interfaz fijada en el plan; hoy ningún
    /// campo muta (cada llamada abre el archivo en modo append y lo cierra),
    /// lo que mantiene la implementación simple para un proceso de una sola
    /// corrida y deja lugar para un writer bufferizado más adelante sin
    /// romper la firma pública.
    ///
    /// # Errors
    ///
    /// Devuelve [`HeadlessError::Io`] si serializar el evento o escribir al
    /// archivo del log falla.
    pub fn event(&mut self, ev: &LogEvent<'_>) -> Result<(), HeadlessError> {
        if ev.level() > self.level {
            return Ok(());
        }
        let line = ev.render(self.level, self.tool_result_cap)?;
        self.append_line(&line)
    }

    /// Anexa `line` (sin salto de línea) más un `\n` al archivo de esta
    /// corrida, abriéndolo en modo append.
    fn append_line(&self, line: &str) -> Result<(), HeadlessError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| HeadlessError::Io(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| HeadlessError::Io(e.to_string()))
    }
}

/// Epoch-millis actual, robusto ante un reloj de sistema anterior a la época
/// Unix (degrada a `0` en vez de entrar en pánico — caso extremo que no
/// debería ocurrir en la práctica, pero nunca debe abortar la corrida).
fn current_epoch_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    u64::try_from(millis).unwrap_or(u64::MAX)
}

/// Genera el nombre `run-<ts>-<pid>-<rand>.jsonl` de esta corrida (REQ-H24):
/// `<ts>` epoch-millis con ancho fijo [`TIMESTAMP_WIDTH`] (orden lexicográfico
/// == orden cronológico), `<pid>` del proceso actual, `<rand>` hex de
/// [`RAND_SUFFIX_HEX_WIDTH`] dígitos (evita colisiones bajo paralelismo de CI
/// con corridas sub-segundo, incluso con reuse de PID).
fn generate_log_file_name() -> String {
    let ts_ms = current_epoch_millis();
    let pid = process::id();
    let rand_suffix: u32 = random();
    format!(
        "{LOG_FILENAME_PREFIX}{ts_ms:0ts_width$}-{pid}-{rand_suffix:0rand_width$x}{LOG_FILENAME_SUFFIX}",
        ts_width = TIMESTAMP_WIDTH,
        rand_width = RAND_SUFFIX_HEX_WIDTH,
    )
}

/// Parsea el `<ts>` epoch-millis embebido en un nombre
/// `run-<ts>-<pid>-<rand>.jsonl`. Devuelve `None` si el nombre no matchea el
/// prefijo/sufijo esperado o si el primer segmento no es un `u64` válido —
/// los llamadores tratan `None` como el candidato **más viejo** posible
/// (ver [`prune_retention`]), nunca como un panic ni un salto silencioso.
fn parse_log_timestamp(file_name: &str) -> Option<u64> {
    let stem = file_name
        .strip_prefix(LOG_FILENAME_PREFIX)?
        .strip_suffix(LOG_FILENAME_SUFFIX)?;
    let ts_part = stem.split(LOG_FILENAME_SEPARATOR).next()?;
    ts_part.parse::<u64>().ok()
}

/// Lista los archivos `run-*.jsonl` de `dir` con su timestamp parseado
/// (o `None`) y tamaño. Un `dir` que no se puede listar (p. ej. no existe
/// todavía) produce una lista vacía en lugar de propagar un error — la
/// retención es best-effort (REQ-H34).
fn list_run_logs(dir: &Path) -> Vec<LogFileEntry> {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return Vec::new();
    };
    read_dir
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|entry| {
            let file_name_os = entry.file_name();
            let file_name = file_name_os.to_str()?;
            if !file_name.starts_with(LOG_FILENAME_PREFIX)
                || !file_name.ends_with(LOG_FILENAME_SUFFIX)
            {
                return None;
            }
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            Some(LogFileEntry {
                path: entry.path(),
                ts: parse_log_timestamp(file_name),
                size_bytes,
            })
        })
        .collect()
}

/// Intenta borrar `path`; una carrera donde el archivo ya no existe
/// (`NotFound`, p. ej. otra corrida concurrente ya lo podó) cuenta como éxito
/// — el objetivo es el estado final del directorio, no quién lo borró
/// (REQ-H34). Cualquier otro error (permisos, etc.) se ignora también: la
/// poda es siempre best-effort y nunca aborta la corrida.
fn try_prune_one(path: &Path) -> bool {
    match fs::remove_file(path) {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(_) => false,
    }
}

/// Poda los `run-*.jsonl` de `dir` hasta que queden a lo sumo
/// `retention_runs` **y** su tamaño combinado sea a lo sumo `max_bytes`
/// (REQ-H24/H34). Ambos son los caps EFECTIVOS de esta corrida (spec §11,
/// `[headless] log_retention`/`log_max_bytes`) —
/// [`LOG_RETENTION_RUNS`](super::limits::LOG_RETENTION_RUNS) y
/// [`LOG_MAX_BYTES`](super::limits::LOG_MAX_BYTES) son solo los defaults que
/// `HeadlessLimits::default()` usa cuando el operador no los fija.
///
/// Eviction oldest-first: un nombre cuyo timestamp no parsea ordena como el
/// más viejo (`None < Some`, `Option::Ord`), así que un nombre ajeno/corrupto
/// siempre es el primer candidato a poda en lugar de sobrevivir
/// indefinidamente. Best-effort: cualquier fallo de listado o de borrado
/// (incluida una carrera de `unlink` con otra corrida) se ignora — la poda
/// **nunca** aborta la corrida ni contamina stdout.
///
/// **Complejidad:** `O(n log n)` por el `sort_by_key` más `O(n)` por el
/// recorrido de poda, con `n` = cantidad de archivos `run-*.jsonl` en el
/// directorio — en la práctica acotado por `retention_runs` más lo acumulado
/// desde la última poda.
fn prune_retention(dir: &Path, retention_runs: usize, max_bytes: u64) {
    let mut entries = list_run_logs(dir);
    entries.sort_by_key(|e| e.ts);

    let mut count = entries.len();
    let mut total_bytes: u64 = entries.iter().map(|e| e.size_bytes).sum();

    for entry in &entries {
        if count <= retention_runs && total_bytes <= max_bytes {
            break;
        }
        if try_prune_one(&entry.path) {
            count = count.saturating_sub(1);
            total_bytes = total_bytes.saturating_sub(entry.size_bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::super::limits::{LOG_MAX_BYTES, LOG_RETENTION_RUNS, TOOL_RESULT_CAP};
    use super::{
        current_epoch_millis, parse_log_timestamp, prune_retention, LogEvent, LogLevel, RunLog,
        LOG_FILENAME_PREFIX, LOG_FILENAME_SUFFIX, TIMESTAMP_WIDTH,
    };

    /// Crea un archivo `run-*.jsonl` fake con un timestamp parseable
    /// estrictamente creciente en `i`, para ejercitar la retención sin
    /// depender de reloj real ni de `RunLog::start`.
    fn touch_log(dir: &Path, i: usize) -> PathBuf {
        let ts: u64 = 1_000_000_000_000 + i as u64;
        let name = format!(
            "{LOG_FILENAME_PREFIX}{ts:0width$}-9999-{i:08x}{LOG_FILENAME_SUFFIX}",
            width = TIMESTAMP_WIDTH
        );
        let path = dir.join(name);
        fs::write(&path, b"{}\n").expect("test fixture write must succeed");
        path
    }

    /// REQ-H24/H34: la retención poda los más viejos hasta quedar en el tope
    /// de conteo (+1 por el archivo de la corrida actual que crea `start`).
    #[test]
    fn test_log_retention_prunes_oldest_beyond_count_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..(LOG_RETENTION_RUNS + 5) {
            touch_log(dir.path(), i);
        }

        let _log = RunLog::start(
            dir.path(),
            LogLevel::Info,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start must succeed");

        let n = fs::read_dir(dir.path()).expect("read_dir").count();
        assert!(
            n <= LOG_RETENTION_RUNS + 1,
            "expected pruned count, got {n}"
        );
    }

    /// REQ-H24/H34, spec §11: la retención debe respetar el cap EFECTIVO
    /// (`[headless] log_retention`) pasado a `start`, no el
    /// `LOG_RETENTION_RUNS` constante — un operador que baja el tope a 3 debe
    /// ver la poda quedarse en ese número, muy por debajo del default de 50.
    #[test]
    fn test_run_log_start_respects_custom_effective_retention_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let small_retention = 3usize;
        for i in 0..(small_retention + 5) {
            touch_log(dir.path(), i);
        }

        let _log = RunLog::start(
            dir.path(),
            LogLevel::Info,
            small_retention,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start must succeed");

        let n = fs::read_dir(dir.path()).expect("read_dir").count();
        assert!(
            n <= small_retention + 1,
            "expected pruning down to the custom (smaller) retention cap, got {n} files"
        );
    }

    /// La poda conserva los archivos más recientes (mayor `<ts>`), no un
    /// subconjunto arbitrario.
    #[test]
    fn test_log_retention_keeps_the_newest_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut paths = Vec::new();
        for i in 0..(LOG_RETENTION_RUNS + 3) {
            paths.push(touch_log(dir.path(), i));
        }

        prune_retention(dir.path(), LOG_RETENTION_RUNS, LOG_MAX_BYTES);

        // The newest entry (highest i) must have survived the prune.
        let newest = paths.last().expect("at least one path");
        assert!(
            newest.exists(),
            "newest log file must survive retention pruning"
        );
        // The oldest entry (i = 0) must have been pruned first.
        let oldest = paths.first().expect("at least one path");
        assert!(!oldest.exists(), "oldest log file must be pruned first");
    }

    /// Race tolerance (REQ-H34): un archivo que otra corrida ya borró antes
    /// de que esta poda lo alcance no hace fallar `start` (simulado borrando
    /// manualmente uno de los candidatos más viejos antes de podar).
    #[test]
    fn test_start_succeeds_when_a_prune_candidate_already_vanished() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut paths = Vec::new();
        for i in 0..(LOG_RETENTION_RUNS + 5) {
            paths.push(touch_log(dir.path(), i));
        }
        // Simulate a concurrent run already having pruned the oldest file.
        fs::remove_file(&paths[0]).expect("simulated race removal");

        let result = RunLog::start(
            dir.path(),
            LogLevel::Info,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        );
        assert!(
            result.is_ok(),
            "start must tolerate a vanished prune candidate"
        );
    }

    /// Un nombre cuyo `<ts>` no parsea se trata como el más viejo posible:
    /// se poda primero y nunca hace crashear la retención.
    #[test]
    fn test_unparseable_timestamp_filename_is_pruned_first_without_crashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let garbage_name = format!("{LOG_FILENAME_PREFIX}not-a-timestamp-1{LOG_FILENAME_SUFFIX}");
        let garbage_path = dir.path().join(&garbage_name);
        fs::write(&garbage_path, b"{}\n").expect("write garbage fixture");

        for i in 0..(LOG_RETENTION_RUNS + 2) {
            touch_log(dir.path(), i);
        }

        prune_retention(dir.path(), LOG_RETENTION_RUNS, LOG_MAX_BYTES);

        assert!(
            !garbage_path.exists(),
            "unparseable-ts filename must be pruned first"
        );
    }

    /// `parse_log_timestamp` es `None` para un nombre que no matchea el
    /// patrón, y `Some` con el valor correcto para uno bien formado.
    #[test]
    fn test_parse_log_timestamp_none_for_malformed_some_for_wellformed() {
        assert_eq!(parse_log_timestamp("not-even-close.txt"), None);
        assert_eq!(parse_log_timestamp("run-abc-123-deadbeef.jsonl"), None);
        assert_eq!(
            parse_log_timestamp("run-00000000000001234567-123-deadbeef.jsonl"),
            Some(1_234_567)
        );
    }

    /// Un evento a nivel `debug` con un secreto tipo `sk-ant-...` en el
    /// `input` NO aparece en claro en la línea escrita — se reusa el
    /// redactor de `output.rs` (REQ-H24, T7).
    #[test]
    fn test_debug_tool_call_redacts_secret_in_written_log_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Construido con `format!` para no disparar el escáner de secretos
        // hardcodeados del repo (`tests/no_hardcoded_secrets.rs`); es un
        // fixture sintético, no una key real.
        let secret = format!("sk-ant-{}", "SECRET".repeat(3));
        let input = format!("{{\"token\":\"{secret}\"}}");

        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Debug,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::ToolCall {
            level: LogLevel::Info,
            name: "bash",
            ok: true,
            ms: 5,
            input: &input,
        })
        .expect("event must write");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(
            !contents.contains(&secret),
            "secret leaked in clear: {contents}"
        );
        assert!(
            contents.contains("[REDACTED]"),
            "expected redaction marker: {contents}"
        );
    }

    /// A nivel `info` (no-debug), un `ToolCall` registra `name`/`ok`/`ms` y
    /// la longitud del `input`, pero **nunca** el `input` crudo — ni siquiera
    /// uno sin secretos (REQ-H24: cap ≠ redacción, la ausencia es la regla).
    #[test]
    fn test_info_level_tool_call_omits_raw_input_and_logs_length_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = "plain non-secret argument";

        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Info,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::ToolCall {
            level: LogLevel::Info,
            name: "ls",
            ok: true,
            ms: 3,
            input,
        })
        .expect("event must write");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(
            !contents.contains(input),
            "raw input must not appear at info level"
        );
        assert!(contents.contains("\"input_len\":25"));
        assert!(contents.contains("\"name\":\"ls\""));
    }

    /// Un secreto `sk-…` posicionado para que el límite de truncado
    /// (`TOOL_RESULT_CAP`) caiga a mitad de su cuerpo NO deja un prefijo
    /// parcial en claro: con el orden redact-then-truncate el secreto
    /// COMPLETO se redacta sobre el input sin truncar, así que ni el secreto
    /// completo ni un prefijo partido sobreviven en la línea escrita. (Bajo
    /// el orden viejo truncate-then-redact, el corte partiría el cuerpo del
    /// secreto en un remanente de 12 caracteres — por debajo del mínimo de 16
    /// que exige el patrón `sk-` — dejando ese prefijo sin matchear y filtrado
    /// en claro; este test falla si esa regresión reaparece.)
    #[test]
    fn test_debug_input_redacts_secret_straddling_truncation_boundary() {
        use crate::headless::limits::TOOL_RESULT_CAP;

        // Fixture construido con `format!` (no un literal) para no disparar
        // el escáner de secretos hardcodeados del repo
        // (`tests/no_hardcoded_secrets.rs`).
        let body = "SECRET".repeat(4); // 24 caracteres alnum — cuerpo válido del patrón `sk-`.
        let key = format!("sk-{body}");

        // Cuántos caracteres del CUERPO de la key quedan retenidos del lado
        // "kept" del corte de truncado: por debajo de `SK_KEY_MIN_SUFFIX_LEN`
        // (16), así que si el input llegara sin redactar hasta el truncado,
        // el remanente quedaría sin matchear por el patrón `sk-`.
        const KEPT_BODY_LEN: usize = 12;
        let prefix_len = TOOL_RESULT_CAP - "sk-".len() - KEPT_BODY_LEN;
        let input = format!("{}{key}", "x".repeat(prefix_len));
        assert!(
            input.len() > TOOL_RESULT_CAP,
            "fixture must straddle the truncation cap"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Debug,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::ToolCall {
            level: LogLevel::Info,
            name: "bash",
            ok: true,
            ms: 5,
            input: &input,
        })
        .expect("event must write");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(
            !contents.contains(&key),
            "full secret leaked in clear: {contents}"
        );

        // El prefijo que el orden viejo (truncate-then-redact) habría dejado
        // sin matchear tampoco debe sobrevivir en claro.
        let partial_prefix = key
            .get(.."sk-".len() + KEPT_BODY_LEN)
            .expect("prefix within key bounds");
        assert!(
            !contents.contains(partial_prefix),
            "partial secret prefix leaked in clear (split-secret regression): {contents}"
        );
    }

    /// El `prompt`/envelope crudo nunca se loguea: un `Message` a nivel debug
    /// solo lleva el texto explícito que el llamador pasó (no hay una vía
    /// para que el prompt entre por otro campo).
    #[test]
    fn test_message_event_never_carries_a_raw_prompt_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Debug,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::Message {
            level: LogLevel::Info,
            text: "startup notice",
        })
        .expect("event must write");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(contents.contains("startup notice"));
        assert!(!contents.contains("\"prompt\""));
    }

    /// Un evento más verboso que la verbosidad configurada se filtra: no se
    /// escribe ninguna línea (el archivo, tocado por `start`, queda vacío).
    #[test]
    fn test_event_more_verbose_than_configured_level_is_filtered_out() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Warn,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::Message {
            level: LogLevel::Debug,
            text: "should not appear",
        })
        .expect("filtered event must still return Ok");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(
            contents.is_empty(),
            "filtered event must not be written: {contents}"
        );
    }

    /// Un evento a la misma verbosidad configurada (o menos verboso) pasa el
    /// filtro y se escribe.
    #[test]
    fn test_event_at_or_below_configured_level_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RunLog::start(
            dir.path(),
            LogLevel::Warn,
            LOG_RETENTION_RUNS,
            LOG_MAX_BYTES,
            TOOL_RESULT_CAP,
        )
        .expect("start");
        log.event(&LogEvent::Message {
            level: LogLevel::Warn,
            text: "a warning",
        })
        .expect("event must write");

        let contents = fs::read_to_string(&log.path).expect("read log file");
        assert!(contents.contains("a warning"));
    }

    /// El orden de severidad es `Error < Warn < Info < Debug` (usado por el
    /// filtro de verbosidad).
    #[test]
    fn test_log_level_ordering_from_least_to_most_verbose() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
    }

    /// `current_epoch_millis` nunca entra en pánico y devuelve un valor
    /// mayor que cero para un reloj de sistema normal.
    #[test]
    fn test_current_epoch_millis_is_nonzero_under_a_normal_clock() {
        assert!(current_epoch_millis() > 0);
    }
}
