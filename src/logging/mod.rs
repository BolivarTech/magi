// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! Server-grade logging for magi-rs: a daily file rotated in UTC, `.xz`
//! compression with retention, and an auditor that redacts secrets before they
//! reach any output.
//!
//! # The shape, and why it is this shape
//!
//! Everything decidable is a **pure function that returns a decision**;
//! a thin shim executes it. Rotation, retention, chunking and rendering never
//! touch the filesystem or read a clock. That is what makes "on day 8 it is
//! compressed and on day 31 it is deleted" testable with two dates instead of
//! thirty-one days of real files.
//!
//! # Lint policy, which is stricter here than in most of the crate
//!
//! This module is held to the same bar as `vault`: panicking constructs are
//! **denied**, not discouraged. The reason is specific rather than general —
//! this is the subsystem you read when everything else has already failed, so
//! a panic inside it takes the diagnostic channel down at the exact moment it
//! is needed. Fallible operations return `Result` and degrade to a documented
//! best effort; they never abort the process that was trying to log.
//!
//! The denials are lifted under `cfg(test)`, where `unwrap` on a literal date
//! is clarity rather than risk.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![cfg_attr(not(test), deny(clippy::todo))]
#![cfg_attr(not(test), deny(clippy::unimplemented))]
#![deny(missing_docs)]

use std::path::PathBuf;

/// Everything this subsystem can fail at.
///
/// **Defined here because no task owned it.** Three tasks of the plan return
/// `Result<_, LoggingError>` — the compressor, the retention executor and
/// `init_logging` — and none declared the type. `mod.rs` is the subsystem's API
/// surface and is already one of the milestone's files, so putting it here
/// keeps the file count honest.
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    /// The log directory could not be created.
    #[error("cannot create the log directory {path}: {source}")]
    DirCreate {
        /// Directory that could not be created.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// A write, create or rename failed.
    #[error("cannot write {path}: {source}")]
    Write {
        /// File the operation targeted.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// Compression, its read-back, or the comparison failed.
    #[error("cannot compress {path}: {source}")]
    Compress {
        /// File being compressed, or the staged temporary.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// An operator-supplied filter directive could not be parsed.
    #[error("invalid filter directive {directive:?}: {reason}")]
    FilterInvalid {
        /// The directive as written.
        directive: String,
        /// Why it was rejected.
        reason: String,
    },
}

/// Delivers an audited line to a screen.
///
/// **Declared here, in the library, and not as `NoticeSink`.** The sink trait
/// lives in the binary crate, which this module cannot see; inverting the
/// dependency would make `logging` depend on the agent. The binary's sink is
/// adapted to this in MS2, which is also where the screen branch is wired.
pub trait NoticeDelivery: Send + Sync {
    /// Shows one audited line. **Consults no filter**: exemption from the
    /// filters is what the alarm path buys.
    fn deliver(&self, line: &auditor::Audited);
}

/// The level the screen branch is wired at (REQ-L19).
///
/// `ERROR` and `WARN` reach the screen; `INFO` and below go only to the file.
/// **Not operator-configurable, and that is decided rather than pending**: the
/// `[logging].tui_filter` key was removed once it was found to be accepted by
/// serde and read by nothing, so the policy lives here as one constant rather
/// than as a setting whose effect nobody could observe.
pub const SCREEN_LEVEL: tracing::Level = tracing::Level::WARN;

/// What both shapes of the recovery-detection warning say.
///
/// One constant so the emitted form and the collected form cannot drift into
/// two different sentences about one condition — and so the guard in
/// `health.rs` matches the text the operator actually reads.
const RECOVERY_DETECTION_OFF: &str =
    "health recovery detection is off: file_filter excludes info-level events, \
     so a degradation is never seen to recover";

/// Warns, once at startup, when recovery detection cannot work.
///
/// # Parameters
///
/// * `file_filter` — the file branch's filter.
/// * `screen_level` — the screen branch's level, when one is wired.
///
/// # Why this is a notice and not an exemption in `enabled`
///
/// The success events that health recovery is detected from are `INFO`-level.
/// When the layer's `enabled` admits no `INFO` they never reach it, so a
/// degradation is still shown and is never seen to recover. Carving an
/// exception into `enabled` for cause-bearing events would be a filter that
/// lies about what it filters — an operator who raised the threshold asked for
/// fewer events. So the behaviour stands and the consequence is named instead.
///
/// # Only ONE of the two inputs is operator-settable
///
/// `enabled` is the union of the file branch's filter and the screen branch's
/// level, but the screen branch is the fixed [`SCREEN_LEVEL`] constant, which
/// never admits `INFO` — the `[logging].tui_filter` key was removed once it was
/// found to be read by nothing. So the union collapses to a question about
/// `file_filter` alone, and the notice below names that key rather than "both
/// filters": there is no second one for an operator to go looking for.
///
/// # Why it is emitted HERE and not inside [`init_logging`]
///
/// `init_logging` mounts the layer; it knows nothing about recovery detection,
/// and teaching it would put a screen policy inside the plumbing. The caller
/// invokes this right after it holds the handle, where the filter and the
/// screen level are already resolved and the sink already exists.
///
/// # The terminal surface uses [`recovery_detection_notice`] instead
///
/// Emitting in place is right for a surface that announces as it goes, which
/// headless does. It is wrong for the TUI, whose screen sink is unreachable
/// until `run_tui_ext` attaches it: anything emitted before that reaches the
/// primary buffer, which `EnterAlternateScreen` then covers for the whole
/// session.
///
/// # Complexity
///
/// `O(k)` over the filter's per-target overrides.
pub fn warn_if_recovery_detection_is_off(
    file_filter: &filter::Filter,
    screen_level: Option<tracing::Level>,
) {
    if health::recovery_detection_is_off(file_filter, screen_level) {
        tracing::warn!(
            target: health::HEALTH_TARGET,
            "{message}",
            message = RECOVERY_DETECTION_OFF,
        );
    }
}

/// [`warn_if_recovery_detection_is_off`], as a COLLECTED notice rather than an
/// immediate emission.
///
/// # Parameters
///
/// * `file_filter` — the file branch's filter.
/// * `screen_level` — the screen branch's level, when one is wired.
///
/// # Returns
///
/// `Some(Notice::warn(..))` when recovery detection cannot work; `None` when it
/// can, so a caller can `extend` its startup list unconditionally.
///
/// # Why the terminal surface needs this shape
///
/// The TUI's screen sink only becomes reachable once `run_tui_ext` attaches it
/// to the response channel, which happens long after `init_logging` returns.
/// Announcing there would take the sink's unattached branch and write to the
/// primary buffer, which is swapped out a moment later — the operator would be
/// told that recovery detection is off in text they never see. So this rides
/// with the rest of the startup notices and is announced inside the one window
/// where the sink is live and the frame is not yet up.
///
/// # The cost, stated
///
/// A collected notice is announced under `magi_rs::startup` rather than under
/// the health target [`warn_if_recovery_detection_is_off`] uses, so a
/// per-target directive naming health does not address the terminal's copy of
/// this one line. That is the flattening every
/// other startup notice already accepts (D-L11's execution is centralised), and
/// the alternative is a line an operator can filter but never read.
///
/// # Complexity
///
/// `O(k)` over the filter's per-target overrides.
pub fn recovery_detection_notice(
    file_filter: &filter::Filter,
    screen_level: Option<tracing::Level>,
) -> Option<crate::notices::Notice> {
    health::recovery_detection_is_off(file_filter, screen_level)
        .then(|| crate::notices::Notice::warn(RECOVERY_DETECTION_OFF))
}

/// A delivery that shows nothing, for the tests and for MS1's absent screen.
pub struct DiscardDelivery;

impl NoticeDelivery for DiscardDelivery {
    fn deliver(&self, _line: &auditor::Audited) {}
}

pub mod appender;
pub mod auditor;
pub mod chunk;
pub mod filter;
pub mod health;
pub mod magi_layer;
pub mod render;
pub mod retention;
pub mod rotation;
pub mod sweep;
pub mod xz;

#[cfg(test)]
pub(crate) mod testutil;

/// Configuration of the logging subsystem.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Where the daily files live.
    pub log_dir: std::path::PathBuf,
    /// Which events reach the file (REQ-L30): a level, or per-target
    /// directives. Parsed by the caller so an invalid one is a LOAD error
    /// (REQ-L31) rather than something this function discovers too late.
    pub file_filter: filter::Filter,
}

/// What a caller keeps after initialising.
///
/// Cheap to clone: the second `init_logging` call returns the same handle rather
/// than a second subsystem.
#[derive(Clone)]
pub struct LoggingHandle {
    appender: std::sync::Arc<appender::DailyAppender>,
    /// The layer's health reporter, shared rather than owned.
    ///
    /// Exposed only through [`LoggingHandle::health_tick`] and
    /// [`LoggingHandle::health_flush`], so no caller decides when to lock.
    health: std::sync::Arc<magi_layer::HealthReporter>,
}

impl LoggingHandle {
    /// Waits for the queue to empty, up to `budget` (REQ-L54).
    ///
    /// # Parameters
    ///
    /// * `budget` — the longest this is allowed to hold up the exit.
    ///
    /// # Returns
    ///
    /// Bytes still queued when it gave up; `0` when everything reached the
    /// file.
    ///
    /// # Why the process needs this
    ///
    /// The writer is a detached thread. Without a wait, the process exits and
    /// takes it down with whatever it had not written yet, and the events lost
    /// are the LAST ones -- which on a run that ended badly are the only ones
    /// worth having.
    ///
    /// # Why it is bounded
    ///
    /// A writer that is stuck must not hold the exit open. The budget makes the
    /// wait a best effort with a stated cost, and the return value lets the
    /// caller say how much did not make it rather than pretend it did.
    ///
    /// # Complexity
    ///
    /// `O(budget / poll interval)`.
    #[must_use]
    pub fn drain(&self, budget: std::time::Duration) -> usize {
        /// How often the queue is re-checked while draining.
        const POLL: std::time::Duration = std::time::Duration::from_millis(20);
        let deadline = std::time::Instant::now() + budget;
        loop {
            let left = self.appender.queued_bytes();
            if left == 0 || std::time::Instant::now() >= deadline {
                return left;
            }
            std::thread::sleep(POLL);
        }
    }

    /// Advances the health window, showing any transition it expires.
    ///
    /// # Parameters
    ///
    /// * `now` — a monotonic instant. Passed in rather than read here so the
    ///   tracker stays deterministic under test.
    ///
    /// # Who calls it
    ///
    /// The TUI's event loop, which already wakes on its own `poll` timeout;
    /// and headless once per agent turn, which is the only natural cadence a
    /// mode without an event loop has. The consequence is declared rather than
    /// hidden: in headless the window is really "until the next turn", and a
    /// run shorter than that relies on [`Self::health_flush`] instead.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of subsystems observed so far.
    pub fn health_tick(&self, now: std::time::Instant) {
        self.health.tick(now);
    }

    /// Shows every pending health transition, window or no window (SC-L90).
    ///
    /// Called at close, where there is no "later" left for a state to settle
    /// in: the choice is between showing a pending transition now and losing
    /// it. Neither `SIGKILL` nor an unhandled `SIGTERM` reaches this, and MS2
    /// installs no handler — a container stopped that way loses the pending
    /// transition, with the same consolation as the appender's own drain, that
    /// what already reached the file is complete.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of subsystems observed so far.
    pub fn health_flush(&self) {
        self.health.flush();
    }

    /// How many events the appender has dropped.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.appender.dropped()
    }
}

/// Permission bits the log's own files carry on Unix (REQ-L65).
///
/// `0600` on a file, `0700` on the directory: a log is a transcript of what the
/// agent did, and on a shared machine the process umask commonly leaves it
/// `0644` — world-readable — which is the state this constant exists to prevent.
pub(crate) const OWNER_ONLY_FILE_MODE: u32 = 0o600;
/// The directory half of [`OWNER_ONLY_FILE_MODE`].
pub(crate) const OWNER_ONLY_DIR_MODE: u32 = 0o700;

/// Restricts `path` to its owner.
///
/// # Parameters
///
/// * `path` — an existing file or directory.
/// * `mode` — [`OWNER_ONLY_FILE_MODE`] or [`OWNER_ONLY_DIR_MODE`].
///
/// # Errors
///
/// [`LoggingError::Write`] if the permissions cannot be set.
///
/// # Windows
///
/// A no-op, and that is a decision rather than a gap. The log lives inside
/// `.magi/logs/`, and `magi init` already restricts `.magi/` by ACL to the
/// current user (REQ-H38), which a file created underneath inherits. The
/// Unix-side umask has no equivalent there, so there is no window this would
/// close that the directory's ACL leaves open.
///
/// # Complexity
///
/// `O(1)`.
pub(crate) fn restrict(path: &std::path::Path, mode: u32) -> Result<(), LoggingError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).map_err(|e| {
            LoggingError::Write {
                path: path.to_path_buf(),
                source: e,
            }
        })?;
    }
    #[cfg(not(unix))]
    let _ = (path, mode);
    Ok(())
}

/// The auditor the installed layer uses.
///
/// One per process, because the registered secrets have to be the SAME set for
/// every mouth: two auditors would mean two different ideas of what must be
/// redacted, and the mouth with the emptier one is the leak.
///
/// **Public because the layer is not the only consumer.** The mode classifier
/// and the headless runtime redact with an auditor too, and each used to build
/// its own -- next to a comment saying there is one per process. A registry
/// filled by `register_process_secrets` was invisible to both, so the exact
/// pass covered the log and nothing else.
pub fn process_auditor() -> &'static std::sync::Arc<auditor::Auditor> {
    static A: std::sync::OnceLock<std::sync::Arc<auditor::Auditor>> = std::sync::OnceLock::new();
    A.get_or_init(|| std::sync::Arc::new(auditor::Auditor::new()))
}

/// Registers the secrets this process resolved, so both mouths mask them.
///
/// # Parameters
///
/// * `secrets` — name and value pairs. **The caller composes the variants** it
///   wants covered; this does not derive them, because the encoder that produced
///   a value inside a URL lives on the resolving side, not here.
///
/// # Returns
///
/// The names whose value was too short for the exact pass, so the caller can
/// warn. They are registered regardless — pass 1 still covers them.
///
/// # Complexity
///
/// `O(total bytes)`.
pub fn register_process_secrets(
    secrets: &[(auditor::SecretName, &str)],
) -> Vec<auditor::SecretName> {
    let mut short = Vec::new();
    for (name, value) in secrets {
        // **Three variants, and the third is the MAIN case, not an extra.**
        //
        // *Raw* — the value as it appears in free text.
        //
        // *Escaped* — still needed with the JSON mode deferred, and it is easy
        // to believe otherwise: in text mode a `{:?}` on a `String` escapes too,
        // so the raw literal can be absent from a line with no JSON in sight.
        //
        // *Percent-encoded* — a password with reserved characters **never**
        // appears raw inside a `base_url`, which is exactly where credentials
        // live. Registering only the raw form is blindness in the one place
        // that matters.
        //
        // The variants are composed HERE and the auditor keeps only
        // `(length, hash)` of each, so the plaintext copies live exactly as long
        // as this call.
        // **One quote off each end, not every quote.** `{:?}` wraps the value
        // in exactly one pair, so `trim_matches` -- which strips as many as it
        // finds -- ate the value's own trailing quote too: a credential ending
        // in `"` registered its escaped form as `abc\` instead of `abc\"`, and
        // the exact pass then scanned for a string that never appears. Three
        // reviewers named this independently, and while the raw and encoded
        // variants still cover the value, a wrong variant in the auditor is not
        // something to carry forward.
        let quoted = format!("{value:?}");
        let escaped = quoted
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(quoted.as_str());
        let encoded = crate::encoding::percent_encode(value);
        if !process_auditor().register_secret(*name, &[value, escaped, encoded.as_str()]) {
            short.push(*name);
        }
    }
    short
}

/// This process's run identifier, minted once.
///
/// **Discoverable, not merely existent** (REQ-L63). Telling an operator to
/// "filter by run" is useless if the job cannot say which run is its own, and
/// the per-run file that used to be the implicit answer is exactly what the
/// JSONL retirement removed. The value is emitted in the output envelope AND on
/// stderr, so a CI job can capture it without parsing the log.
///
/// Same shape and the same reasons as an event id: `<pid>-<16 hex>`, 64 bits,
/// because a counter or 32 bits collide across concurrent CI runs — which is
/// precisely when isolating one matters.
///
/// # Complexity
///
/// `O(1)` after the first call.
#[must_use]
pub fn run_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(|| chunk::EventId::new().render())
}

/// Emits the run's first event: what was invoked, and where.
///
/// # Parameters
///
/// * `command` — the subcommand the user ran, e.g. `"query"`.
/// * `workspace` — the resolved workspace root.
///
/// # Why this is not enough on its own, said where a reader will look
///
/// REQ-L63 is explicit that this event **helps a reader orient and does not
/// make the file self-sufficient**. With several runs writing concurrently the
/// daily file holds many "first events" interleaved, and none of them says
/// which one is *yours*. That answer comes from the envelope and from the
/// stderr line; this is the third piece, not a replacement for either.
///
/// # Complexity
///
/// `O(1)`.
pub fn announce_run(command: &str, workspace: &std::path::Path) {
    tracing::info!(
        target: "magi_rs::headless",
        command = command,
        workspace = %workspace.display(),
        "run start"
    );
}

/// The single global handle.
///
/// **A `OnceLock`, never a `Once`, and the difference is not style.**
/// `std::sync::Once` **poisons** if its closure panics: every later
/// `call_once` panics too, forever. On a path where creating the directory can
/// fail, that turns a recoverable failure into a dead process on the second
/// attempt — REQ-L35 violated by the very mechanism chosen to protect it. A
/// `OnceLock` does not poison: a failed attempt leaves the cell empty and the
/// next call tries again.
static HANDLE: std::sync::OnceLock<LoggingHandle> = std::sync::OnceLock::new();

/// The handle this process installed, if logging came up at all.
///
/// The exit path needs it and does not own it: `init_logging` hands its handle
/// to a caller that may drop it long before the process ends, and the writer
/// lives in the static either way.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn installed() -> Option<&'static LoggingHandle> {
    HANDLE.get()
}

/// Brings the subsystem up. **Idempotent by construction.**
///
/// # Parameters
///
/// * `cfg` — where and at what level to write.
/// * `sink` — **built by the caller**, never here: a failure before it existed
///   would have nowhere to report and would end in an `eprintln!` that writes
///   over the TUI's alternate screen.
/// * `tui` — the screen branch. `None` in MS1; the parameter exists from MS1 so
///   MS2 does not reopen this signature.
///
/// # Returns
///
/// The handle. A second call returns the **same** handle and `Ok`.
///
/// # A second call with a DIFFERENT configuration is ignored, and says so
///
/// The new `log_dir` or level is not applied and the caller gets no error. That
/// is deliberate — reconfiguring a global subscriber hot is a mechanism MS1 does
/// not have and nobody asked for — but the warning **names that the
/// configuration was discarded**, not merely that a second call happened.
/// Without that, an operator changes `log_dir`, sees a generic notice, and hunts
/// for a file that was never moved.
///
/// Returning `Err` would force every caller to decide whether that is fatal, and
/// the answer is always no: logging is already running, which is what they
/// wanted.
///
/// # Errors
///
/// [`LoggingError::DirCreate`] if the log directory cannot be created.
///
/// # Complexity
///
/// `O(1)` after the first call.
pub fn init_logging(
    cfg: &LoggingConfig,
    sink: std::sync::Arc<dyn NoticeDelivery>,
    tui: Option<(magi_layer::TuiSink, tracing::Level)>,
) -> Result<LoggingHandle, LoggingError> {
    if let Some(existing) = HANDLE.get() {
        // A second call would otherwise reach `set_global_default`, which
        // second call — a test that starts twice, a `main` that retries, a new
        // surface that does not know another already initialised — would abort
        // the process. REQ-L35 says logging never aborts.
        let (line, _) = auditor::Auditor::new().audit(
            "notice: logging was already initialised; this call's configuration \
             (log directory and level) was DISCARDED and the running one kept",
            "magi_rs::logging",
            None,
            0,
        );
        sink.deliver(&line);
        return Ok(existing.clone());
    }

    // Step 2: resolve and create the directory, open the file.
    let appender = std::sync::Arc::new(appender::DailyAppender::new(&cfg.log_dir)?);

    // Step 3: resolve the zone offset. Constant in MS1 (UTC) and cannot fail.
    // The step keeps its number because THE ORDER IS THE CONTRACT.

    // Step 4: build the auditor and register the environment's secrets.
    let audit = std::sync::Arc::clone(process_auditor());

    // Step 5: mount the layer and install the subscriber.
    let mut layer = magi_layer::MagiLayer::new(
        magi_layer::FileSink::new(std::sync::Arc::clone(&appender)),
        cfg.file_filter.clone(),
        std::sync::Arc::clone(&audit),
        std::sync::Arc::clone(&sink),
    );
    // Taken before installation, for the same reason `with_tui` is called
    // before it: installation consumes the layer, and the exit path still has
    // to be able to flush what the tracker is holding.
    let health = layer.health();
    if let Some((tui_sink, tui_level)) = tui {
        // Called HERE, inside, before the subscriber is installed: installation
        // consumes the layer, so afterwards a `self -> Self` builder has nothing
        // left to apply to.
        layer = layer.with_tui(tui_sink, tui_level);
    }

    let handle = LoggingHandle {
        appender: std::sync::Arc::clone(&appender),
        health,
    };
    // **The race is resolved, not discarded.** Two callers can both pass the
    // `get` above, and both then build an appender — each with its own writer
    // thread on the same file. Throwing the `set` result away left the loser
    // holding a handle that is not the installed one: `installed()` would find
    // the winner's, the exit drain would wait on that, and the loser's thread
    // would keep writing with nobody accounting for it. The loser takes the
    // winner's handle and lets its own drop, which closes its channels and ends
    // its writer.
    if HANDLE.set(handle.clone()).is_err() {
        if let Some(winner) = HANDLE.get() {
            return Ok(winner.clone());
        }
    }

    use tracing_subscriber::layer::SubscriberExt as _;
    let subscriber = tracing_subscriber::registry().with(layer);
    // Deliberately not `set_global_default`, which panics on a second install:
    // this path is already guarded by the `OnceLock` above, and the result is
    // discarded so a race between two first-callers degrades to one of them
    // winning rather than to a panic.
    let _ = tracing::subscriber::set_global_default(subscriber);

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_event_names_the_command_and_the_workspace() {
        // REQ-L63's third piece. A reader who opens the file cold has to be
        // able to tell what produced these lines.
        let line = testutil::capture(|| {
            announce_run("consult", std::path::Path::new("/srv/project"));
        });
        assert!(
            line.contains("consult"),
            "the event does not name the command: {line}"
        );
        assert!(
            line.contains("/srv/project"),
            "the event does not name the workspace: {line}"
        );
    }

    #[test]
    fn a_different_invocation_produces_a_different_first_event() {
        // Without this, an event that hardcodes one command satisfies the test
        // above and stops describing the run it announces.
        let one = testutil::capture(|| {
            announce_run("query", std::path::Path::new("/a"));
        });
        let two = testutil::capture(|| {
            announce_run("consult", std::path::Path::new("/b"));
        });
        assert_ne!(one, two, "the first event ignores what it was told");
    }
}
