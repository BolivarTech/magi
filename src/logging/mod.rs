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
//! a thin shim executes it. Rotation, retention and chunking take the date, the
//! clock reading and the file list as PARAMETERS and touch nothing: that is what
//! makes "on day 8 it is compressed and on day 31 it is deleted" testable with
//! two dates instead of thirty-one days of real files.
//!
//! **Rendering is the exception, and it is one on purpose.** `render_event`
//! reads `OffsetDateTime::now_utc()` itself, because the stamp an event carries
//! must be when it was EMITTED; taking it from a parameter filled in further
//! down would record when the writer got to it, which on a queue that is
//! draining is a different and less useful time. Its escapers and its header
//! composition stay pure. `EventId::new` is the other clock reader here, and
//! only on the fallback branch where the OS random source refuses.
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
/// **Defined here because no task owned it.** It is returned from all over the
/// subsystem — the compressor, the retention executor, the appender, the filter
/// parser, the permission shim and `init_logging` — and no single task declared
/// it. `mod.rs` is the subsystem's API surface and is already one of the
/// milestone's files, so putting it here keeps the file count honest.
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

/// A delivery that shows nothing.
///
/// **Half its stated reason expired in MS2 and the other half is load-bearing.** "For
/// MS1's absent screen" is gone: both surfaces now wire a real sink, so no production
/// path constructs this. What is left is not "for the tests" in the loose sense that
/// would make it public surface with no consumer — it is consumed from OUTSIDE the
/// crate, by `tests/ms1_interface_probe.rs`, `tests/prompt_never_logged.rs` and
/// `tests/tui_buffer_audited_only.rs`, which are separate binaries linking `magi_rs`
/// the way a downstream user would.
///
/// That is why it is `pub` and must stay `pub`. Gating it `#[cfg(test)]` was tried and
/// fails to compile all three of those binaries: `cfg(test)` covers the lib's own unit
/// tests and nothing else. A reviewer grepping only `src/` sees eleven unit-test call
/// sites and concludes this is deletable; the three integration consumers are the part
/// that is not visible from there.
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
/// **What `Ok` promises, and what it does not.** It promises what this function
/// owns: the log directory exists, the appender and its writer thread are
/// running, and this handle is the installed one. It does **not** promise that
/// any event will reach them — mounting the layer needs the process's global
/// subscriber, and another installer may already hold it. That case still
/// returns `Ok`, deliberately (REQ-L35: logging never aborts the process that
/// was trying to log), and is announced through `sink`, because it is a state
/// an operator can act on and a caller cannot.
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
        // A second call — a test that starts twice, a `main` that retries, a
        // new surface that does not know another already initialised — takes
        // this branch and reports, rather than building a second appender with
        // a second writer thread on the same file.
        //
        // **Through the PROCESS auditor, and escaped for the mouth it goes to.**
        // A fresh `Auditor` starts with nothing registered, so its exact pass
        // does nothing and only the pattern pass stands between this line and a
        // screen — the same defect `main.rs` already guards itself against, in
        // the one file that guard does not read. The screen escaper rather than
        // the file one because `sink` is a screen: doubling a backslash here
        // would show a path nobody can paste.
        //
        // **The alarm is forwarded**, the same as on every other audited path
        // in this subsystem. What stood here instead was a justification for
        // dropping it — the text below is a literal with no interpolation, so
        // there is nothing in it for the passes to find — and only its first
        // half is true. Fixed prose does carry no runtime value, which is why
        // an interpolated `log_dir` or error is where a credential would
        // ORDINARILY arrive. But the exact pass matches REGISTERED VALUES, not
        // shapes, and nothing stops a registered one from occurring inside our
        // own prose: this module's `..._still_raises_the_alarm` test registers
        // a phrase of the very sentence below and watches it get masked. So the
        // conclusion did not follow, and a false justification is worse than a
        // bare gap — it tells the next reader there is nothing here to look at.
        let (line, alarm) = process_auditor().audit(
            "notice: logging was already initialised; this call's configuration \
             (log directory and level) was DISCARDED and the running one kept",
            "magi_rs::logging",
            None,
            0,
        );
        // Capped like every other screen delivery. The text above is a literal
        // and cannot reach the cap today, which is exactly why the omission
        // was invisible; the cap is a property of the MOUTH, so the next author
        // who interpolates a `log_dir` into this notice inherits it instead of
        // having to remember it.
        sink.deliver(
            &line
                .map_line(render::escape_for_screen)
                .truncate_for_display(magi_layer::TUI_PAYLOAD_MAX_BYTES),
        );
        if let Some(alarm) = alarm {
            // The handle's own appender, which is the installed one: this
            // branch exists precisely because a second must not be built.
            //
            // **Settled but not reported, the asymmetry `HealthReporter::show`
            // already carries.** No `Reporter` is reachable from here — the
            // layer owns one by value and was consumed by installation — and
            // `settle_alarm` is what keeps the omission honest: a refused alarm
            // gives its latch back, so the next masking raises it again instead
            // of the finding being lost.
            let outcome = existing.appender.submit(
                auditor::Queued::Alarm(alarm.clone()),
                appender::Priority::High,
                magi_layer::NO_RESERVATION,
            );
            magi_layer::settle_alarm(process_auditor(), &alarm, outcome);
        }
        return Ok(existing.clone());
    }

    // Step 2: resolve and create the directory, open the file.
    let appender = std::sync::Arc::new(appender::DailyAppender::new(&cfg.log_dir)?);

    // Step 3: resolve the zone offset. Constant in MS1 (UTC) and cannot fail.
    // The step keeps its number because THE ORDER IS THE CONTRACT.

    // Step 4: take the PROCESS auditor. It is neither built nor filled here --
    // `process_auditor` owns the single instance and `register_process_secrets`
    // fills it, called by whichever surface resolved the secrets. Both happen
    // outside this function, and the step number is kept because THE ORDER IS
    // THE CONTRACT.
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
    // **The annotation IS the sole-layer assertion.** `Layer::enabled` is
    // evaluated for the whole subscriber, not per layer: a `false` from
    // `MagiLayer` disables the event for every layer beneath it. That is inert
    // while it is the only one and silently wrong the day it is not — a second
    // layer would start losing events with nothing to say so. Adding a
    // `.with(..)` here changes this expression's type and stops the build,
    // which is the earliest a reader can be told; a test could only observe it
    // afterwards, and only if somebody wrote one.
    let subscriber: tracing_subscriber::layer::Layered<
        magi_layer::MagiLayer,
        tracing_subscriber::Registry,
    > = tracing_subscriber::registry().with(layer);
    // **`set_global_default` RETURNS an error on a second install; it does not
    // panic.** The comment that stood here said the opposite, and the
    // difference decides how this line has to be written: the panicking
    // spellings are `SubscriberInitExt::init` and an `expect` on this result,
    // and both are what REQ-L35 forbids.
    //
    // **The error is read rather than discarded, and the difference is what
    // `Ok` then means.** Discarding it covered two unlike cases with one
    // silence. A race between two first-callers past the `OnceLock` above is
    // benign — the loser is already holding the winner's handle by the time it
    // reaches this line. Another INSTALLER owning the process's subscriber is
    // not: the directory exists, the writer thread is running, `installed()`
    // answers with a handle, and no event ever reaches any of it. The return
    // stays `Ok` because REQ-L35 says a caller must never be aborted over
    // logging, so the mouth is the only place left that can say so.
    if tracing::subscriber::set_global_default(subscriber).is_err() {
        // Audited, escaped and capped exactly like the already-initialised
        // notice above, and for the same reasons: the process auditor rather
        // than a fresh one, the SCREEN escaper because `sink` is a screen, and
        // the cap because it belongs to the mouth rather than to the site.
        let (line, alarm) = process_auditor().audit(
            "warning: another subscriber is already installed for this process, \
             so events will not reach the log; the log directory was created \
             and stays empty",
            "magi_rs::logging",
            None,
            0,
        );
        sink.deliver(
            &line
                .map_line(render::escape_for_screen)
                .truncate_for_display(magi_layer::TUI_PAYLOAD_MAX_BYTES),
        );
        if let Some(alarm) = alarm {
            // Settled but not reported, the asymmetry the other three audited
            // paths in this subsystem already carry: no `Reporter` is reachable
            // here, and `settle_alarm` gives a refused alarm its latch back so
            // the finding is raised again rather than lost.
            let outcome = appender.submit(
                auditor::Queued::Alarm(alarm.clone()),
                appender::Priority::High,
                magi_layer::NO_RESERVATION,
            );
            magi_layer::settle_alarm(process_auditor(), &alarm, outcome);
        }
    }

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The environment variable `cargo-nextest` exports into every test process.
    const NEXTEST_MARKER: &str = "NEXTEST";

    /// This suite requires a process-per-test runner, and says so once instead of
    /// five times in a row.
    ///
    /// # What actually happens under `cargo test`
    ///
    /// The subsystem installs itself into a `OnceLock` and raises the process's global
    /// `LevelFilter`, both of which are per-PROCESS. `cargo test` runs the whole lib in
    /// ONE process, so the first test to call `init_logging` wins for every test after
    /// it, and the ones asserting on the *uninstalled* state lose — a handful spread
    /// across `notices` and this module, with nothing in the output saying the runner
    /// is the cause. **The count is left unstated on purpose**: it moves with every
    /// test added on either side, and a number here would go stale without going red.
    ///
    /// # Why this is a pointer and not the only defence
    ///
    /// Those tests are already self-guarding: each opens by asserting
    /// `LevelFilter::current() == OFF` with "something installed one". So the shared
    /// process produces a RED, never a false green, which is the safe direction and was
    /// deliberate. What it does not produce is an explanation — a scatter of
    /// unrelated-looking failures sends a reader after a product defect that is not
    /// there. This turns them into one failure that names the cause.
    ///
    /// It is deliberately NOT a check on ordering or a `serial_test` lock: serialising
    /// would make the assertions pass in one arrangement and hide that the state is
    /// global. The requirement is the runner, so the runner is what is asserted.
    #[test]
    fn the_suite_requires_a_process_per_test_runner() {
        assert!(
            std::env::var_os(NEXTEST_MARKER).is_some(),
            "this suite must be run with `cargo nextest run`, which gives every test its \
             own process. Under `cargo test` the whole library shares one, so the \
             `OnceLock` subscriber and the global `LevelFilter` leak between tests and \
             five of them fail for a reason that has nothing to do with the code under \
             test. CI runs nextest in both workflows; run it locally too."
        );
    }

    /// Keeps every line a mouth was handed.
    #[derive(Default)]
    struct RecordingSink {
        /// One entry per delivered line, in delivery order.
        lines: std::sync::Mutex<Vec<String>>,
    }

    impl NoticeDelivery for RecordingSink {
        fn deliver(&self, line: &auditor::Audited) {
            if let Ok(mut l) = self.lines.lock() {
                l.push(line.as_str().to_string());
            }
        }
    }

    /// A phrase that appears VERBATIM in the already-initialised notice.
    ///
    /// Registering it as a secret is what turns "which auditor audits that
    /// line" into something a test can observe: the process auditor masks it,
    /// a freshly built one has nothing registered and cannot.
    const PHRASE_INSIDE_THE_NOTICE: &str = "already initialised";

    #[test]
    fn the_already_initialised_notice_goes_through_the_process_auditor() {
        // The notice is the subsystem talking about itself, and it used to be
        // audited by an `Auditor::new()` built on the spot. A fresh auditor has
        // no registered secrets, so its exact pass does nothing and only the
        // pattern pass stands between that line and a mouth -- the same defect
        // `main.rs`'s `no_surface_builds_an_auditor_of_its_own` was written for,
        // in the one file that guard does not read.
        //
        // Isolated by `cargo nextest`'s one-process-per-test model: both
        // `HANDLE` and the process auditor are process-global, so a second test
        // sharing this process would share them too.
        //
        // Mutation-verified: with `Auditor::new().audit(..)` back in
        // `init_logging`, the phrase comes through in the clear and this fails.
        assert!(
            PHRASE_INSIDE_THE_NOTICE.len() >= auditor::MIN_SECRET_BYTES,
            "a phrase below the floor never enters the exact pass, so this \
             would hold for free"
        );
        process_auditor().register_secret(
            auditor::SecretName::new("NOTICE_PROBE"),
            &[PHRASE_INSIDE_THE_NOTICE],
        );

        let dir = tempfile::tempdir().expect("a temp dir");
        let cfg = LoggingConfig {
            log_dir: dir.path().to_path_buf(),
            file_filter: filter::Filter::parse("info").expect("valid"),
        };
        let sink = std::sync::Arc::new(RecordingSink::default());
        let first = init_logging(&cfg, sink.clone(), None);
        assert!(first.is_ok(), "the first call must bring the subsystem up");
        let _second =
            init_logging(&cfg, sink.clone(), None).expect("a second call is not an error");

        let said = sink.lines.lock().expect("not poisoned").join("\n");
        assert!(
            said.contains("DISCARDED"),
            "the fixture produced no already-initialised notice: {said:?}"
        );
        assert!(
            !said.contains(PHRASE_INSIDE_THE_NOTICE),
            "the notice was audited by an auditor that knows nothing this \
             process registered: {said:?}"
        );
    }

    /// A second phrase from the same notice, registered by the test below.
    ///
    /// Distinct from [`PHRASE_INSIDE_THE_NOTICE`] so the two tests cannot pass
    /// on each other's registration if they ever share a process.
    const PHRASE_THE_ALARM_MUST_NAME: &str = "log directory and level";

    /// Reads the day's file until `needle` shows up, or the deadline passes.
    ///
    /// The writer is a detached thread, so a bare read races it. A generous
    /// FAILURE deadline keeps the wait meaningful under load without turning
    /// the test into a guess about how fast the thread is.
    fn wait_for_log(dir: &std::path::Path, needle: &str) -> String {
        let path = dir.join(rotation::file_name(time::OffsetDateTime::now_utc().date()));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let written = std::fs::read_to_string(&path).unwrap_or_default();
            if written.contains(needle) || std::time::Instant::now() >= deadline {
                return written;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    #[test]
    fn the_already_initialised_notice_that_masks_a_secret_still_raises_the_alarm() {
        // The ninth instance of one class, and the last one in this module:
        // the auditor's contract is "mask AND say so -- both, never one", and
        // this branch took the first half. `Reporter::announce`,
        // `HealthReporter::show` and `on_event` all forward their alarm; this
        // one dropped it, behind a comment claiming the literal text has
        // nothing in it for the passes to find.
        //
        // That claim is what this fixture disproves. The exact pass matches
        // REGISTERED VALUES, not shapes, and nothing stops a registered
        // credential from occurring inside our own prose -- so a masking
        // happens here and, without the forward, nobody is told.
        //
        // Isolated by `cargo nextest`'s one-process-per-test model: `HANDLE`
        // and the process auditor are both process-global.
        assert!(
            PHRASE_THE_ALARM_MUST_NAME.len() >= auditor::MIN_SECRET_BYTES,
            "a phrase below the floor never enters the exact pass, so this \
             would hold for free"
        );
        process_auditor().register_secret(
            auditor::SecretName::new("ALARM_PROBE"),
            &[PHRASE_THE_ALARM_MUST_NAME],
        );

        let dir = tempfile::tempdir().expect("a temp dir");
        let cfg = LoggingConfig {
            log_dir: dir.path().to_path_buf(),
            file_filter: filter::Filter::parse("info").expect("valid"),
        };
        let sink = std::sync::Arc::new(RecordingSink::default());
        let _first = init_logging(&cfg, sink.clone(), None).expect("the subsystem comes up");
        let _second =
            init_logging(&cfg, sink.clone(), None).expect("a second call is not an error");

        let said = sink.lines.lock().expect("not poisoned").join("\n");
        assert!(
            !said.contains(PHRASE_THE_ALARM_MUST_NAME),
            "the notice was not masked at all, so there is no alarm to forward \
             and the rest of this proves nothing: {said:?}"
        );
        let written = wait_for_log(dir.path(), "SECURITY:");
        assert!(
            written.contains("SECURITY:") && written.contains("ALARM_PROBE"),
            "the already-initialised branch masked its own notice and told \
             nobody: {written:?}"
        );
    }

    #[test]
    fn the_installed_layer_publishes_its_hint_as_the_global_level_filter() {
        // The hint's VALUE is pinned next door, in
        // `magi_layer::tests::the_level_hint_is_the_maximum_of_both_branches`.
        // What nothing pinned is that it ARRIVES: `tracing` reads
        // `Layer::max_level_hint` once, at installation, and publishes it as
        // `LevelFilter::current()` — which is what every `event!` callsite in
        // this binary AND in every dependency consults before it builds
        // anything at all. Without the hint reaching there, `current()` stays
        // at `TRACE`, every callsite in the tree is enabled, and the whole
        // reason the static hint exists is paid for and not collected.
        //
        // Mutation-verified: with `max_level_hint` deleted from the `Layer`
        // impl this reads `TRACE` instead of `WARN`.
        //
        // Isolated by `cargo nextest`'s one-process-per-test model — the
        // global subscriber and its published filter are process-wide.
        use tracing::level_filters::LevelFilter;
        assert_ne!(
            LevelFilter::current(),
            LevelFilter::WARN,
            "the fixture starts at the level it means to prove was installed, \
             so the assertion below would hold for free"
        );

        let dir = tempfile::tempdir().expect("a temp dir");
        let cfg = LoggingConfig {
            log_dir: dir.path().to_path_buf(),
            // Chosen because it is neither the uninstalled default nor the
            // filter's own default: only the installed hint can produce it.
            file_filter: filter::Filter::parse("warn").expect("valid"),
        };
        let _handle = init_logging(&cfg, std::sync::Arc::new(DiscardDelivery), None)
            .expect("the subsystem comes up");

        assert_eq!(
            LevelFilter::current(),
            LevelFilter::WARN,
            "the layer's hint never reached the global filter, so every \
             callsite in the dependency tree is still enabled"
        );
    }

    #[test]
    fn two_first_callers_end_up_holding_the_same_installed_subsystem() {
        // Two callers can both pass the `HANDLE.get()` check and both build an
        // appender, each with its own writer thread on the same file. Throwing
        // the `HANDLE.set` result away left the loser holding a handle that is
        // not the installed one: `installed()` would find the winner's, the
        // exit drain would wait on that, and the loser's thread would keep
        // writing with nobody accounting for it.
        //
        // **Deterministic, and honest about what that buys.** The assertions
        // hold whichever order the two threads take, so this can never fail
        // spuriously — a flaky race test is worse than none. What it cannot do
        // is GUARANTEE the interleaving: if one thread finishes before the
        // other starts, the second takes the already-initialised branch and
        // reaches the same conclusion by the other road. The barrier is what
        // makes the overlap likely; the assertions are what make the test safe
        // either way.
        //
        // Separate directories on purpose: a second call whose configuration
        // differs is the case an operator actually produces, and it is the one
        // where a loser keeping its own appender would go unnoticed.
        let a_dir = tempfile::tempdir().expect("a temp dir");
        let b_dir = tempfile::tempdir().expect("a temp dir");
        let gate = std::sync::Arc::new(std::sync::Barrier::new(2));

        let start = |dir: std::path::PathBuf, gate: std::sync::Arc<std::sync::Barrier>| {
            std::thread::spawn(move || {
                let cfg = LoggingConfig {
                    log_dir: dir,
                    file_filter: filter::Filter::parse("info").expect("valid"),
                };
                gate.wait();
                init_logging(&cfg, std::sync::Arc::new(DiscardDelivery), None)
            })
        };
        let first = start(a_dir.path().to_path_buf(), std::sync::Arc::clone(&gate));
        let second = start(b_dir.path().to_path_buf(), gate);

        let a = first.join().expect("no panic").expect("a handle");
        let b = second.join().expect("no panic").expect("a handle");
        let installed = installed().expect("something was installed");

        for (which, handle) in [("first", &a), ("second", &b)] {
            assert!(
                std::sync::Arc::ptr_eq(&handle.appender, &installed.appender),
                "the {which} caller kept its OWN appender: a second writer \
                 thread is on the day's file and the exit drain waits on the \
                 other one"
            );
            assert!(
                std::sync::Arc::ptr_eq(&handle.health, &installed.health),
                "the {which} caller kept its OWN health reporter, so the exit \
                 flush shows transitions nobody recorded"
            );
        }
    }

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

    /// A phrase the not-installed notice must carry.
    ///
    /// The words an operator would search for, not the whole sentence: a
    /// verbatim copy of the text would pass while saying anything at all.
    const PHRASE_IN_THE_NOT_INSTALLED_NOTICE: &str = "will not reach";

    #[test]
    fn a_subscriber_that_could_not_be_installed_is_announced() {
        // `set_global_default` fails when something else already holds the
        // process's global subscriber. Its `Result` was discarded, so the
        // caller got an `Ok` describing a state that does not exist: the
        // directory is created, the writer thread is running, `installed()`
        // answers with a handle -- and NOT ONE EVENT reaches any of it, because
        // the layer was never mounted.
        //
        // REQ-L35 keeps the return `Ok`: logging must never abort the process
        // that was trying to log. So the failure has to reach the operator
        // through the mouth instead, which is what this pins.
        assert!(
            tracing::subscriber::set_global_default(tracing_subscriber::registry()).is_ok(),
            "this test owns its process's global subscriber"
        );

        let dir = tempfile::tempdir().expect("a temp dir");
        let cfg = LoggingConfig {
            log_dir: dir.path().to_path_buf(),
            file_filter: filter::Filter::parse("info").expect("valid"),
        };
        let sink = std::sync::Arc::new(RecordingSink::default());
        let handle = init_logging(&cfg, sink.clone(), None);

        assert!(
            handle.is_ok(),
            "REQ-L35: a subsystem that cannot come up still never aborts its caller"
        );
        let said = sink.lines.lock().expect("not poisoned").join("\n");
        assert!(
            said.contains(PHRASE_IN_THE_NOT_INSTALLED_NOTICE),
            "the caller was told logging was up while nothing was mounted: {said:?}"
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
