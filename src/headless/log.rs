// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-18

//! Run logs in JSONL (`./.magi/logs/run-<ts>-<pid>-<rand>.jsonl`, REQ-H24/H34): retention by
//! count **and** size ordered by file-name timestamp (never by `mtime`), and redaction of a
//! tool-call's raw `input` according to configured verbosity (REQ-H24 — the size cap and
//! redaction are **distinct** controls, neither replaces the other).
//!
//! - [`RunLog::start`] discovers/creates the logs directory, prunes old runs
//! old (best-effort, tolerant of `unlink` races) and touches this run's file.
//! - [`RunLog::event`] filters by [`LogLevel`] and writes one JSON line per
//! [`LogEvent`]: at `info` level (or less verbose) a `ToolCall` only carries
//! `name`/`ok`/`ms`/`input_len`; at `debug` level the `input` is **redacted first**
//! ([`redact_secret_patterns`], reused from `output.rs`) and **then** **capped**
//! ([`truncate_result`], reused from `output.rs`) — this order prevents a secret split by the
//! truncation limit from leaking a partial prefix; matchers are never re-implemented. The raw
//! `prompt`/envelope is **never** logged, at any level or in any field.
//!
//! `RunLog`/`LogLevel`/`LogEvent` are `pub`: the MS2 runner lives in the binary crate and can
//! only reach `pub` APIs of the lib.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::random;
use serde::Serialize;

use super::output::{redact_secret_patterns, truncate_result};
use super::HeadlessError;

/// Literal prefix of every run-log file name.
const LOG_FILENAME_PREFIX: &str = "run-";

/// Literal suffix of every run-log file name.
const LOG_FILENAME_SUFFIX: &str = ".jsonl";

/// Separator between the three name segments (`<ts>-<pid>-<rand>`).
const LOG_FILENAME_SEPARATOR: char = '-';

/// Width, in decimal digits, of the `<ts>` (epoch-millis) embedded in the file name. `u64::MAX`
/// has 20 decimal digits, so this width ensures the lexicographic order of the name matches
/// chronological order indefinitely (REQ-H24).
const TIMESTAMP_WIDTH: usize = 20;

/// Width, in hexadecimal digits, of the name's random suffix (`u32` = 4 bytes = 8 hex digits) —
/// avoids collisions between runs started within the same millisecond, even with the same PID
/// (REQ-H24).
const RAND_SUFFIX_HEX_WIDTH: usize = 8;

/// Value of the `"kind"` field for a [`LogEvent::Message`] line.
const EVENT_KIND_MESSAGE: &str = "message";

/// Value of the `"kind"` field for a [`LogEvent::ToolCall`] line.
const EVENT_KIND_TOOL_CALL: &str = "tool_call";

/// Configured verbosity of a [`RunLog`], ordered from least to most detailed (`Error < Warn <
/// Info < Debug`, REQ-H24 `--log-level`).
///
/// A [`LogEvent`] whose own severity is **more verbose** than the configured verbosity of the
/// [`RunLog`] is filtered and never written (passes the filter iff `event.level() <=
/// configured`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    /// Only irrecoverable failures.
    Error,
    /// Recoverable but noteworthy conditions (e.g. a clamp, a tier denial).
    Warn,
    /// Normal operational notices (default, REQ-H24) — includes tool-call metadata but
    /// **never** the raw `input` (redaction REQ-H24).
    Info,
    /// Maximum verbosity — includes a tool-call's raw `input`, capped and redacted (never
    /// without going through the redactor).
    Debug,
}

impl std::str::FromStr for LogLevel {
    type Err = HeadlessError;

    /// Parses the value of `[headless] log_level` (spec §11): exactly `"error"`, `"warn"`,
    /// `"info"` or `"debug"` — the same four literals accepted by `--log-level` (case-
    /// sensitive, no normalization, so that a typo in `magi.toml` is a clear error instead of
    /// an accidental match).
    ///
    /// # Errors
    ///
    /// Returns [`HeadlessError::InputInvalid`] with the received value if `s` is not one of the
    /// four recognized literals.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "error" => Ok(LogLevel::Error),
            "warn" => Ok(LogLevel::Warn),
            "info" => Ok(LogLevel::Info),
            "debug" => Ok(LogLevel::Debug),
            other => Err(HeadlessError::InputInvalid(format!(
                "invalid [headless] log_level {other:?} (expected one of \
                 error|warn|info|debug)"
            ))),
        }
    }
}

/// A record of a `run-*.jsonl` file discovered during a retention scan (REQ-H24/H34).
struct LogFileEntry {
    /// Full path of the file.
    path: PathBuf,
    /// Epoch-millis timestamp parsed from the name; `None` if the name does not match the
    /// expected pattern — treated as the **oldest** possible (highest prune priority, see
    /// [`prune_retention`]).
    ts: Option<u64>,
    /// Size in bytes of the file (or `0` if its metadata could not be read).
    size_bytes: u64,
}

/// An occurrence to log as one JSON line in the run log.
///
/// Each variant carries its own severity ([`LogLevel`]) for the verbosity filter; it is
/// independent of the **configured** verbosity of the [`RunLog`], which instead governs the
/// redaction of a `ToolCall`'s `input` (REQ-H24).
#[derive(Debug, Clone)]
pub enum LogEvent<'a> {
    /// A free-text diagnostic (startup notices, clamps, notices).
    /// **Never** must it be the caller's raw `prompt`/envelope (REQ-H11/H24).
    Message {
        /// Severity of this occurrence.
        level: LogLevel,
        /// Human-readable message text.
        text: &'a str,
    },
    /// The record of a single tool invocation.
    ToolCall {
        /// Severity of this occurrence.
        level: LogLevel,
        /// Name of the invoked tool.
        name: &'a str,
        /// `true` if the invocation succeeded.
        ok: bool,
        /// Duration of the invocation in milliseconds.
        ms: u64,
        /// Raw tool input (serialized JSON or other text); only included in plaintext (capped +
        /// redacted) if the verbosity
        /// **configured** of the `RunLog` is [`LogLevel::Debug`] — otherwise
        /// only its byte length is logged (REQ-H24).
        input: &'a str,
    },
}

impl LogEvent<'_> {
    /// Severity of this occurrence, used to filter against the configured verbosity of a
    /// [`RunLog`].
    fn level(&self) -> LogLevel {
        match self {
            LogEvent::Message { level, .. } | LogEvent::ToolCall { level, .. } => *level,
        }
    }

    /// Renders this occurrence as one JSON line, given the verbosity
    /// **configured** of a [`RunLog`] (governs whether a `ToolCall`'s `input` goes
    /// raw-capped-and-redacted or only as its length) and the EFFECTIVE cap `tool_result_cap`
    /// (spec §11) applied to the redacted `input`.
    ///
    /// # Errors
    ///
    /// Returns [`HeadlessError::Io`] if JSON serialization fails (in practice, never for these
    /// structures — they carry no non-string map keys nor non-finite floats).
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
                    // Redact BEFORE truncating: redacting the full input first means a secret
                    // straddling the truncation boundary cannot be split into an un-matchable
                    // partial that leaks; then truncate the secret-free result.
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

/// JSON shape of a [`LogEvent::Message`] line.
#[derive(Debug, Serialize)]
struct MessageLine<'a> {
    /// Epoch-millis of this line's writing.
    ts: u64,
    /// Severity of the occurrence.
    level: LogLevel,
    /// Line-type discriminator (`"message"`).
    kind: &'static str,
    /// Message text.
    message: &'a str,
}

/// JSON shape of a [`LogEvent::ToolCall`] line.
#[derive(Debug, Serialize)]
struct ToolCallLine<'a> {
    /// Epoch-millis of this line's writing.
    ts: u64,
    /// Severity of the occurrence.
    level: LogLevel,
    /// Line-type discriminator (`"tool_call"`).
    kind: &'static str,
    /// Name of the invoked tool.
    name: &'a str,
    /// `true` if the invocation succeeded.
    ok: bool,
    /// Duration of the invocation in milliseconds.
    ms: u64,
    /// Raw `input` (capped + redacted), present **only** at debug level.
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    /// Byte length of the `input`, present when `input` is absent (levels other than debug) —
    /// REQ-H24: the cap does not replace redaction, so at non-debug levels nothing of the
    /// content is emitted directly, only its size.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_len: Option<usize>,
}

/// Log of a headless run, in JSONL (REQ-H24/H34).
#[derive(Debug)]
pub struct RunLog {
    /// Path of this run's JSONL file.
    path: PathBuf,
    /// Configured verbosity: events more verbose than this level are filtered.
    level: LogLevel,
    /// EFFECTIVE cap (spec §11) applied to the redacted `input` of a `ToolCall` at debug level
    /// (`[headless] tool_result_cap_bytes`).
    tool_result_cap: usize,
}

impl RunLog {
    /// Starts the log for a new run under `logs_dir`: creates the directory if missing, prunes
    /// old runs (best-effort, REQ-H34) down to the EFFECTIVE caps
    /// `retention_runs`/`max_log_bytes` (spec §11, an operator may lower them via `[headless]
    /// log_retention`/`log_max_bytes`), and touches this run's `run-<ts>-<pid>-<rand>.jsonl`
    /// file. `tool_result_cap` is the EFFECTIVE cap applied to the redacted `input` of a
    /// `ToolCall` logged at debug level (`[headless] tool_result_cap_bytes`).
    ///
    /// # Errors
    ///
    /// Returns [`HeadlessError::Io`] if `logs_dir` cannot be created or this run's file cannot
    /// be opened. Retention pruning itself **never** produces a propagated error — it is best-
    /// effort (REQ-H34).
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

    /// Logs `ev` as one JSON line if it passes the verbosity filter (`ev.level() <=
    /// self.level`); a filtered event is a successful no-op.
    ///
    /// `&mut self`: the signature follows the interface fixed in the plan; today no field
    /// mutates (each call opens the file in append mode and closes it), which keeps the
    /// implementation simple for a single-run process and leaves room for a buffered writer
    /// later without breaking the public signature.
    ///
    /// # Errors
    ///
    /// Returns [`HeadlessError::Io`] if serializing the event or writing to the log file fails.
    pub fn event(&mut self, ev: &LogEvent<'_>) -> Result<(), HeadlessError> {
        if ev.level() > self.level {
            return Ok(());
        }
        let line = ev.render(self.level, self.tool_result_cap)?;
        self.append_line(&line)
    }

    /// Appends `line` (without newline) plus a `\n` to this run's file, opening it in append
    /// mode.
    fn append_line(&self, line: &str) -> Result<(), HeadlessError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| HeadlessError::Io(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| HeadlessError::Io(e.to_string()))
    }
}

/// Current epoch-millis, robust against a system clock earlier than the Unix epoch (degrades to
/// `0` instead of panicking — an extreme case that should not happen in practice, but must
/// never abort the run).
fn current_epoch_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    u64::try_from(millis).unwrap_or(u64::MAX)
}

/// Generates this run's `run-<ts>-<pid>-<rand>.jsonl` name (REQ-H24): `<ts>` epoch-millis with
/// fixed width [`TIMESTAMP_WIDTH`] (lexicographic order == chronological order), current
/// process `<pid>`, [`RAND_SUFFIX_HEX_WIDTH`]-digit hex `<rand>` (avoids collisions under CI
/// parallelism with sub-second runs, even with PID reuse).
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

/// Parses the `<ts>` epoch-millis embedded in a `run-<ts>-<pid>-<rand>.jsonl` name. Returns
/// `None` if the name does not match the expected prefix/suffix or if the first segment is not
/// a valid `u64` — callers treat `None` as the **oldest** possible candidate (see
/// [`prune_retention`]), never as a panic or a silent skip.
fn parse_log_timestamp(file_name: &str) -> Option<u64> {
    let stem = file_name
        .strip_prefix(LOG_FILENAME_PREFIX)?
        .strip_suffix(LOG_FILENAME_SUFFIX)?;
    let ts_part = stem.split(LOG_FILENAME_SEPARATOR).next()?;
    ts_part.parse::<u64>().ok()
}

/// Lists the `run-*.jsonl` files of `dir` with their parsed timestamp (or `None`) and size. A
/// `dir` that cannot be listed (e.g. does not exist yet) yields an empty list instead of
/// propagating an error — retention is best-effort (REQ-H34).
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

/// Attempts to delete `path`; a race where the file no longer exists (`NotFound`, e.g. another
/// concurrent run already pruned it) counts as success — the goal is the final state of the
/// directory, not who deleted it (REQ-H34). Any other error (permissions, etc.) is also
/// ignored: pruning is always best-effort and never aborts the run.
fn try_prune_one(path: &Path) -> bool {
    match fs::remove_file(path) {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(_) => false,
    }
}

/// Prunes the `run-*.jsonl` files of `dir` until at most `retention_runs` remain **and** their
/// combined size is at most `max_bytes` (REQ-H24/H34). Both are the EFFECTIVE caps of this run
/// (spec §11, `[headless] log_retention`/`log_max_bytes`) —
/// [`LOG_RETENTION_RUNS`](super::limits::LOG_RETENTION_RUNS) and
/// [`LOG_MAX_BYTES`](super::limits::LOG_MAX_BYTES) are only the defaults
/// `HeadlessLimits::default()` uses when the operator does not set them.
///
/// Eviction oldest-first: a name whose timestamp does not parse sorts as the oldest (`None <
/// Some`, `Option::Ord`), so a foreign/corrupt name is always the first prune candidate instead
/// of surviving indefinitely. Best-effort: any listing or deletion failure (including an
/// `unlink` race with another run) is ignored — pruning
/// **never** aborts the run nor contaminates stdout.
///
/// **Complexity:** `O(n log n)` from `sort_by_key` plus `O(n)` from the
/// prune traversal, with `n` = number of `run-*.jsonl` files in the directory — in practice
/// bounded by `retention_runs` plus whatever accumulated since the last prune.
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
        current_epoch_millis, parse_log_timestamp, prune_retention, HeadlessError, LogEvent,
        LogLevel, RunLog, LOG_FILENAME_PREFIX, LOG_FILENAME_SUFFIX, TIMESTAMP_WIDTH,
    };

    /// Creates a fake `run-*.jsonl` file with a strictly increasing parseable timestamp in `i`,
    /// to exercise retention without depending on the real clock or `RunLog::start`.
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

    /// REQ-H24/H34: retention prunes the oldest until the count cap remains (+1 for the current
    /// run's file created by `start`).
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

    /// REQ-H24/H34, spec §11: retention must respect the EFFECTIVE cap (`[headless]
    /// log_retention`) passed to `start`, not the `LOG_RETENTION_RUNS` constant — an operator
    /// who lowers the cap to 3 must see pruning settle at that number, well below the default
    /// of 50.
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

    /// Pruning keeps the most recent files (highest `<ts>`), not an arbitrary subset.
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

    /// Race tolerance (REQ-H34): a file another run already deleted before this prune reaches
    /// it does not make `start` fail (simulated by manually deleting one of the oldest
    /// candidates before pruning).
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

    /// A name whose `<ts>` does not parse is treated as the oldest possible: it is pruned first
    /// and never makes retention crash.
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

    /// `parse_log_timestamp` is `None` for a name that does not match the pattern, and `Some`
    /// with the correct value for a well-formed one.
    #[test]
    fn test_parse_log_timestamp_none_for_malformed_some_for_wellformed() {
        assert_eq!(parse_log_timestamp("not-even-close.txt"), None);
        assert_eq!(parse_log_timestamp("run-abc-123-deadbeef.jsonl"), None);
        assert_eq!(
            parse_log_timestamp("run-00000000000001234567-123-deadbeef.jsonl"),
            Some(1_234_567)
        );
    }

    /// A `debug`-level event with a secret like `sk-ant-...` in the `input` does NOT appear in
    /// plaintext in the written line — the redactor from `output.rs` is reused (REQ-H24, T7).
    #[test]
    fn test_debug_tool_call_redacts_secret_in_written_log_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Built with `format!` so as not to trigger the repo's hardcoded-secret scanner
        // (`tests/no_hardcoded_secrets.rs`); it is a synthetic fixture, not a real key.
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

    /// At `info` level (non-debug), a `ToolCall` logs `name`/`ok`/`ms` and the `input` length,
    /// but **never** the raw `input` — not even one without secrets (REQ-H24: cap ≠ redaction,
    /// absence is the rule).
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

    /// A `sk-…` secret positioned so the truncation limit (`TOOL_RESULT_CAP`) falls in the
    /// middle of its body does NOT leave a partial plaintext prefix: with the redact-then-
    /// truncate order the COMPLETE secret is redacted over the untruncated input, so neither
    /// the complete secret nor a split prefix survives in the written line. (Under the old
    /// truncate-then-redact order, the cut would split the secret body into a 12-character
    /// remainder — below the 16-character minimum required by the `sk-` pattern — leaving that
    /// prefix un-matched and leaked in plaintext; this test fails if that regression
    /// reappears.)
    #[test]
    fn test_debug_input_redacts_secret_straddling_truncation_boundary() {
        use crate::headless::limits::TOOL_RESULT_CAP;

        // Fixture built with `format!` (not a literal) so as not to trigger the repo's
        // hardcoded-secret scanner (`tests/no_hardcoded_secrets.rs`).
        let body = "SECRET".repeat(4); // 24 caracteres alnum — cuerpo válido del patrón `sk-`.
        let key = format!("sk-{body}");

        // How many characters of the key BODY remain retained on the "kept" side of the
        // truncation cut: below `SK_KEY_MIN_SUFFIX_LEN` (16), so if the input reached
        // truncation un-redacted, the remainder would fail to match the `sk-` pattern.
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

        // The prefix that the old order (truncate-then-redact) would have left un-matched must
        // also not survive in plaintext.
        let partial_prefix = key
            .get(.."sk-".len() + KEPT_BODY_LEN)
            .expect("prefix within key bounds");
        assert!(
            !contents.contains(partial_prefix),
            "partial secret prefix leaked in clear (split-secret regression): {contents}"
        );
    }

    /// The raw `prompt`/envelope is never logged: a `Message` at debug level only carries the
    /// explicit text the caller passed (there is no path for the prompt to enter through
    /// another field).
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

    /// An event more verbose than the configured verbosity is filtered: no line is written (the
    /// file, touched by `start`, remains empty).
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

    /// An event at the same configured verbosity (or less verbose) passes the filter and is
    /// written.
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

    /// The severity order is `Error < Warn < Info < Debug` (used by the verbosity filter).
    #[test]
    fn test_log_level_ordering_from_least_to_most_verbose() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
    }

    /// REQ-H24, spec §11: `[headless] log_level` parses exactly the four documented literals to
    /// their `LogLevel` variant.
    #[test]
    fn test_log_level_from_str_parses_all_four_literals() {
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
        assert_eq!("warn".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
    }

    /// An unrecognized `[headless] log_level` string is a clear typed error, never a silent
    /// fallback to a default.
    #[test]
    fn test_log_level_from_str_rejects_unknown_value() {
        assert!(matches!(
            "verbose".parse::<LogLevel>(),
            Err(HeadlessError::InputInvalid(_))
        ));
        assert!(matches!(
            "".parse::<LogLevel>(),
            Err(HeadlessError::InputInvalid(_))
        ));
        // Case-sensitive: no silent normalization of a near-miss.
        assert!(matches!(
            "Debug".parse::<LogLevel>(),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// `current_epoch_millis` never panics and returns a value greater than zero for a normal
    /// system clock.
    #[test]
    fn test_current_epoch_millis_is_nonzero_under_a_normal_clock() {
        assert!(current_epoch_millis() > 0);
    }
}
