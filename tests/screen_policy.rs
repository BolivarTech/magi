// Author: Julian Bolivar
// Version: 0.18.1
// Date: 2026-08-31

//! The screen policy, driven through the REAL dispatcher.
//!
//! # What these hold down
//!
//! `INFO` and below go only to the file (REQ-L19); the filters are **per
//! layer**, so nothing global may discard an event before the layer sees it
//! (REQ-L24); and a health transition still serving its window is shown when
//! the run closes rather than lost (SC-L90).
//!
//! # Why an integration binary
//!
//! `init_logging` installs a global subscriber behind a `OnceLock`, so a
//! process gets exactly one. `cargo nextest` runs each test in its own process,
//! which is what makes several of them here legitimate; under plain
//! `cargo test` they would share one subscriber and only the first would mean
//! anything.
//!
//! # The trap these are written to avoid
//!
//! Every one emits with the `tracing` macros and asserts on what came out of a
//! mouth. A test that built the screen line by hand would exercise the renderer
//! and stay green through exactly the mutation it exists to catch — switching
//! the per-layer filters for a global one, which drops the event before the
//! layer runs at all.

use std::sync::{Arc, Mutex};

use magi_rs::logging::auditor::Audited;
use magi_rs::logging::magi_layer::TuiSink;
use magi_rs::logging::{init_logging, LoggingConfig, NoticeDelivery, SCREEN_LEVEL};

/// Captures everything a mouth was handed.
#[derive(Default)]
struct CapturingSink {
    /// One entry per delivered line, in delivery order.
    lines: Mutex<Vec<String>>,
}

impl NoticeDelivery for CapturingSink {
    fn deliver(&self, line: &Audited) {
        if let Ok(mut l) = self.lines.lock() {
            l.push(line.as_str().to_string());
        }
    }
}

impl CapturingSink {
    /// Everything delivered so far, as one blob to grep.
    fn joined(&self) -> String {
        self.lines
            .lock()
            .map(|l| l.join("\n"))
            .unwrap_or_else(|p| p.into_inner().join("\n"))
    }
}

/// The day's log file name, which is REQ-L23's third part.
///
/// Derived rather than written down: the message names the file the run is
/// actually writing, and a literal would pass on one day and fail on the next.
fn todays_log_file_name() -> String {
    magi_rs::logging::rotation::file_name(time::OffsetDateTime::now_utc().date())
}

/// Everything the file mouth wrote, once `needle` appears or the deadline
/// passes.
///
/// **A poll on the condition, never a fixed delay.** The writer is a detached
/// thread, so how long it takes to flush is a property of the machine, not of
/// the product: a flat sleep is a guess that is too long on an idle box and too
/// short on a loaded one, which is this repository's documented flake recipe.
/// The discriminating property is that the line *arrives*, never that it
/// arrives inside some number of milliseconds, so the deadline is a FAILURE
/// bound set far past any legitimate flush.
///
/// # Returns
///
/// The file contents — the needle's presence is left to the caller to assert,
/// so a timeout reports what was actually written instead of a bare "false".
fn wait_for_file(dir: &std::path::Path, needle: &str) -> String {
    /// Far past any legitimate flush; only a real defect reaches it.
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
    /// How often the file is re-read while waiting.
    const POLL: std::time::Duration = std::time::Duration::from_millis(20);
    let start = std::time::Instant::now();
    loop {
        let written = everything_written(dir);
        if written.contains(needle) || start.elapsed() >= DEADLINE {
            return written;
        }
        std::thread::sleep(POLL);
    }
}

/// Everything the file mouth wrote.
fn everything_written(dir: &std::path::Path) -> String {
    let mut all = String::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return all;
    };
    for entry in entries.flatten() {
        all.push_str(&std::fs::read_to_string(entry.path()).unwrap_or_default());
    }
    all
}

/// Brings the subsystem up with both mouths pointed at one capturing sink.
///
/// The screen branch is wired at [`SCREEN_LEVEL`], which is the production
/// policy — a test that widened it would report on a screen nobody ships.
fn start(
    dir: &std::path::Path,
    file_filter: &str,
) -> (magi_rs::logging::LoggingHandle, Arc<CapturingSink>) {
    let screen = Arc::new(CapturingSink::default());
    let cfg = LoggingConfig {
        log_dir: dir.to_path_buf(),
        file_filter: magi_rs::logging::filter::Filter::parse(file_filter).expect("a valid filter"),
    };
    let handle = init_logging(
        &cfg,
        screen.clone(),
        Some((TuiSink::new(screen.clone()), SCREEN_LEVEL)),
    )
    .expect("logging comes up");
    (handle, screen)
}

#[test]
fn test_recovery_is_shown_and_fails_if_filters_become_global() {
    // SC-L17, and REQ-L24 is what makes it possible. The success event that
    // proves a subsystem recovered is `INFO`-level on purpose, while the screen
    // is wired at `WARN`. It reaches the layer because the layer's `enabled` is
    // the UNION of both branches; the TRANSITION the tracker returns is not an
    // event and is delivered straight to the sink, so the screen level never
    // gets to suppress it.
    //
    // **Mutation #5 of the milestone: put a global filter in front of the
    // layer** — `registry().with(LevelFilter::WARN).with(layer)` in
    // `init_logging`. The `info!` below is then dropped before `on_event`, no
    // recovery is ever pending, and the assertion on the check mark fails.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "embedder",
        cause.name = "unreachable",
        "embedding request failed"
    );
    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::INFO,
        cause.subsystem = "embedder",
        cause.name = "unreachable",
        "embedding request ok"
    );

    // The recovery serves the 30 s window, and no test may wait that long. The
    // close path is what a short run has instead (SC-L90).
    handle.health_flush();

    let shown = screen.joined();
    assert!(
        shown.contains("memory: retrieval unavailable"),
        "the degradation must reach the screen, or the recovery assertion below \
         holds against a pipeline that shows nothing at all: {shown}"
    );
    assert!(
        shown.contains("✓ memory: retrieval restored"),
        "the recovery never reached the screen. Its source event is INFO-level, \
         so anything that filters globally instead of per layer kills it: {shown}"
    );
    assert!(
        shown.contains(&todays_log_file_name()),
        "REQ-L23's third part -- where to read more -- is missing: {shown}"
    );
}

#[test]
fn a_pending_recovery_reaches_the_screen_only_when_the_run_closes() {
    // The guardian the two call sites of task 1.2 were added without: while
    // `health_flush` was a no-op, any test of it could only have asserted that
    // nothing happened. It has a body now, so this asserts the thing that
    // matters at `main`'s exit -- a transition still serving its window is
    // shown, not swallowed.
    //
    // The discriminating half is the FIRST assertion: without it this passes
    // against an implementation that shows every transition immediately, which
    // is the flapping SC-L70 forbids.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "embedder",
        cause.name = "http_error",
        "embedding request failed"
    );
    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::INFO,
        cause.subsystem = "embedder",
        cause.name = "http_error",
        "embedding request ok"
    );

    let before = screen.joined();
    assert!(
        !before.contains("✓ memory: retrieval restored"),
        "the window has not elapsed, so the recovery must still be pending: {before}"
    );

    handle.health_flush();

    let after = screen.joined();
    assert!(
        after.contains("✓ memory: retrieval restored"),
        "closing the run must not swallow the pending transition: {after}"
    );
}

#[test]
fn test_info_never_reaches_the_screen_but_reaches_the_file() {
    // REQ-L19: the screen carries what is actionable and the file carries the
    // diagnosis. This is the whole objective of the milestone reduced to one
    // event of each level.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    tracing::info!(target: "magi_rs::agent", "context assembled from 12 memories");
    tracing::warn!(target: "magi_rs::agent", "the embedder answered 500");

    drop(handle);

    // The screen is asserted with no wait at all: delivery to the sink happens
    // inside `on_event`, on the emitting thread. Only the FILE goes through the
    // writer thread, so only the file is waited on.
    let shown = screen.joined();
    assert!(
        shown.contains("the embedder answered 500"),
        "a WARN must reach the screen, or the assertion below holds for a screen \
         branch that was never wired: {shown}"
    );
    assert!(
        !shown.contains("context assembled"),
        "an INFO event reached the screen: {shown}"
    );

    let written = wait_for_file(dir.path(), "context assembled");
    assert!(
        written.contains("context assembled"),
        "and the file is where it must have gone instead: {written}"
    );
}

#[test]
fn one_tick_shows_every_transition_whose_window_has_elapsed() {
    // `HealthTracker::tick` hands back ONE transition per call while `flush`
    // hands back a `Vec`, so the wrapper around `tick` has to supply the loop.
    // Without it a cascade shows one recovery now and holds the other until
    // the next call -- a whole agent turn in headless, which can be minutes.
    //
    // The window is expired by ARITHMETIC, not by waiting: `health_tick` takes
    // the instant as a parameter precisely so this is deterministic.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    for subsystem in ["embedder", "provider"] {
        tracing::event!(
            target: "magi_rs::memory",
            tracing::Level::WARN,
            cause.subsystem = subsystem,
            cause.name = "unreachable",
            "subsystem call failed"
        );
    }
    for subsystem in ["embedder", "provider"] {
        tracing::event!(
            target: "magi_rs::memory",
            tracing::Level::INFO,
            cause.subsystem = subsystem,
            cause.name = "unreachable",
            "subsystem call ok"
        );
    }

    handle.health_tick(
        std::time::Instant::now()
            + std::time::Duration::from_secs(magi_rs::logging::health::HEALTH_MIN_STABLE_SECS + 1),
    );

    let shown = screen.joined();
    assert!(
        shown.contains("✓ memory: retrieval restored"),
        "the first elapsed recovery is missing: {shown}"
    );
    assert!(
        shown.contains("✓ provider: reachable again"),
        "the SECOND elapsed recovery is missing, so one tick showed only one \
         of two transitions that were both due: {shown}"
    );
}

#[test]
fn the_log_path_in_a_screen_notice_can_be_pasted_as_it_is_shown() {
    // REQ-L23's third part is "where to read more", which is only useful if
    // the user can open what they are shown. The file's escaper doubles the
    // backslash so the file stays parseable, and applying it here turned a
    // Windows path into `C:\Users\...\magi-<date>.log` -- a string that
    // opens nothing.
    //
    // The DIRECTORY is what discriminates: the file name alone has no
    // separator in it, so asserting on the file name passes either way.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "vault",
        cause.name = "locked",
        "the vault is locked"
    );
    drop(handle);

    let shown = screen.joined();
    let expected = dir.path().join(todays_log_file_name());
    let expected = expected.display().to_string();
    assert!(
        shown.contains("vault: locked"),
        "the fixture produced no screen notice: {shown}"
    );
    assert!(
        shown.contains(&expected),
        "the notice does not carry the path as the filesystem spells it.\n\
         expected to find: {expected}\n\
         shown: {shown}"
    );
}

#[test]
fn a_screen_notice_is_capped_like_every_other_screen_line() {
    // The health notice was the ONE screen entry that skipped
    // `TUI_PAYLOAD_MAX_BYTES`, and its variable part is not only the operator's
    // `log_dir`: an undeclared cause falls through to the "internal error"
    // branch, which quotes the cause key back -- and a cause key is runtime
    // text the EMITTER chooses, with nothing bounding its length.
    //
    // The filler carries SPACES, and that is the fixture rather than styling:
    // a long unbroken run is what the auditor's generic-secret pass treats as
    // a secret. The first draft used `"cause-part-"` and the whole payload
    // came back as `***` -- a 178-byte notice that passed the cap assertion
    // while proving nothing. The lower bound below is what caught it.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    let huge = "cause part ".repeat(magi_rs::logging::magi_layer::TUI_PAYLOAD_MAX_BYTES / 4);
    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "nonesuch",
        cause.name = huge.as_str(),
        "a subsystem with no declared message table row"
    );
    drop(handle);

    let notice = screen
        .lines
        .lock()
        .map(|l| l.clone())
        .unwrap_or_else(|p| p.into_inner().clone())
        .into_iter()
        .find(|l| l.starts_with("internal error: no screen message is declared"))
        .expect("the undeclared cause must still produce a notice");

    assert!(
        notice.len() > TRUNCATION_MARKER_SLACK,
        "the fixture produced a trivially short notice, so the cap is untested: {}",
        notice.len()
    );
    assert!(
        notice.len()
            <= magi_rs::logging::magi_layer::TUI_PAYLOAD_MAX_BYTES + TRUNCATION_MARKER_SLACK,
        "a screen notice escaped the payload cap at {} bytes",
        notice.len()
    );
}

/// Room for the marker `truncate_for_display` appends past the cap.
const TRUNCATION_MARKER_SLACK: usize = 128;

/// The seven diagnostic lines a clean startup produces today (SC-L14).
///
/// Shortened stand-ins for the real texts, one per production site the inventory of task 3.1
/// classified as `info`: the provider banner, the no-config default, the principal probe
/// result, the memory diagnostics, the log sweep summary, the `base_url` normalisation, and
/// the per-mage budget derived from `--timeout`.
const CLEAN_STARTUP_DIAGNOSTICS: [&str; 7] = [
    "Ollama (kimi-k2.6:cloud)",
    "no magi.toml found; using the Ollama-first defaults",
    "kimi-k2.6:cloud: measured window 262144 tokens",
    "memory: 0 active, 0 archived, 0 pending re-embed (~0 KB index)",
    "logs: 2 compressed, 0 abandoned temporaries removed",
    "notice: `base_url` had no `/v1` suffix and one was added",
    "magi: per-mage ceiling 249s derived from --timeout 1800s",
];

#[test]
fn test_a_clean_startup_shows_nothing_on_screen() {
    // SC-L14, and the scenario that measures whether this milestone met its objective. A
    // startup with nothing degraded puts NOT ONE line on the screen, while the file keeps
    // every one of them.
    //
    // **The second half is what makes this a guardian.** A test that only asserted the screen
    // is empty would pass just as well against a screen branch that was never wired at all --
    // which is what production looks like until the sink is swapped in. So the same live
    // sink is shown, at the end, to be capable of delivering: one `warn` goes through it and
    // must appear. Without that, "the screen is clean" and "there is no screen" read alike.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    magi_rs::notices::emit_notices(
        CLEAN_STARTUP_DIAGNOSTICS
            .iter()
            .map(|t| magi_rs::notices::Notice::info(*t))
            .collect(),
    );

    let shown = screen.joined();
    assert!(
        shown.is_empty(),
        "a clean startup put something on the screen: {shown}"
    );

    let written = wait_for_file(dir.path(), CLEAN_STARTUP_DIAGNOSTICS[6]);
    for line in CLEAN_STARTUP_DIAGNOSTICS {
        assert!(
            written.contains(line),
            "the diagnostics must still be in the day's file; {line:?} is missing from: \
             {written}"
        );
    }

    magi_rs::notices::emit_notices(vec![magi_rs::notices::Notice::warn(
        "the vault could not be opened",
    )]);
    assert!(
        screen.joined().contains("the vault could not be opened"),
        "the screen mouth was never live, so the emptiness above proves nothing"
    );

    drop(handle);
}

/// Comfortably past the cap that used to trim at five.
const NOTICES_PAST_THE_RETIRED_CAP: usize = 20;

#[test]
fn test_no_notice_truncation_line_is_ever_emitted() {
    // REQ-L20/D-L12: `NOTICE_MAX_INFO` and the "… N more diagnostic notice(s) omitted" line
    // are gone rather than raised. The cap existed because there was nowhere to put what it
    // discarded, and it produced the worse of both outcomes -- the reader got five lines of
    // noise AND the rest was destroyed instead of filed.
    //
    // Asserting the ABSENCE of the omitted line is not enough on its own: it holds against an
    // implementation that emits nothing at all. The positive half -- all twenty reached the
    // file -- is what discriminates.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    let notices: Vec<magi_rs::notices::Notice> = (0..NOTICES_PAST_THE_RETIRED_CAP)
        .map(|i| magi_rs::notices::Notice::info(format!("startup diagnostic number {i}")))
        .collect();
    magi_rs::notices::emit_notices(notices);

    let written = wait_for_file(
        dir.path(),
        &format!(
            "startup diagnostic number {}",
            NOTICES_PAST_THE_RETIRED_CAP - 1
        ),
    );
    for i in 0..NOTICES_PAST_THE_RETIRED_CAP {
        assert!(
            written.contains(&format!("startup diagnostic number {i}")),
            "notice {i} was trimmed instead of filed: {written}"
        );
    }
    assert!(
        !written.contains("omitted"),
        "the truncation line survived into the file: {written}"
    );
    assert!(
        !screen.joined().contains("omitted"),
        "the truncation line reached the screen: {}",
        screen.joined()
    );

    drop(handle);
}

/// Turns of the failing embedder SC-L15 describes.
const CONSECUTIVE_FAILING_TURNS: usize = 5;

/// The health notice the embedder's `http_error` cause renders to.
///
/// The notice, not the failure event: the two are different lines produced by
/// two different parts of one `on_event`, and telling them apart is the whole
/// point of counting below.
const EMBEDDER_HTTP_ERROR_NOTICE: &str = "memory: retrieval failing";

/// The failure record itself, as the emitter words it.
const EMBEDDER_FAILURE_RECORD: &str = "embedding request failed";

#[test]
fn five_consecutive_failures_show_one_notice_and_leave_five_records() {
    // SC-L15, whose two halves ARE its content: ONE notice on the screen and
    // FIVE records in the file, out of the SAME five failures. Each half was
    // already held down alone -- the tracker's dedup in `logging::health`, the
    // one-event-per-call rule in `memory::embedding` -- and neither can see the
    // pairing, which is what the scenario is about. A dedup that also reached
    // the file, or a file that kept five while the screen kept one for the
    // wrong reason, passes both of those and fails this.
    //
    // Driven through the real dispatcher and read off both real mouths: the
    // file half comes from the day's file, not from a count of what was
    // emitted, because "five were emitted" is not "five were written".
    //
    // # What "un aviso" counts, and why six lines is the correct total
    //
    // SC-L15's subject is the DEGRADATION NOTICE -- the scenario's own title
    // says so -- and there is exactly one of those. The five WARN records reach
    // the screen too, so the total is six lines, and that is REQ-L19 working
    // rather than a leak. Both facts that make it so are named here, because a
    // reader who re-derives this from the scenario text alone will see a
    // contradiction and reach for a "fix" that breaks the tracker:
    //
    //   * REQ-L19 (spec:2099): `ERROR` and `WARN` reach the screen; `INFO` and
    //     below go ONLY to the file. A failure record is `WARN`, so it is on
    //     the screen by requirement.
    //   * `ok` is derived from the LEVEL -- `ok_from_level` is
    //     `level > tracing::Level::WARN` (`src/memory/embedding.rs:1325`,
    //     mirroring the layer's own derivation). So a failure emitted below
    //     `WARN` to keep it off the screen would be read by the tracker as a
    //     SUCCESS and drive it to `Restored`: the exact opposite of what it
    //     reports.
    //
    // Together those make "five failures, one line on screen" unreachable by
    // construction, not merely unimplemented. The count of records on screen is
    // asserted below rather than left to this comment, because it is true and
    // worth pinning -- and if it ever moves, that is a decision about REQ-L19,
    // not a refactor.
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, screen) = start(dir.path(), "info");

    for turn in 1..=CONSECUTIVE_FAILING_TURNS {
        tracing::event!(
            target: "magi_rs::memory",
            tracing::Level::WARN,
            cause.subsystem = "embedder",
            cause.name = "http_error",
            "embedding request failed: HTTP 500 on turn {}",
            turn
        );
    }

    // The screen is read before the handle is dropped: closing flushes any
    // PENDING transition (SC-L90), and a notice that only arrives at close is
    // not the one this scenario counts.
    let shown = screen
        .lines
        .lock()
        .map(|l| l.clone())
        .unwrap_or_else(|p| p.into_inner().clone());
    let notices = shown
        .iter()
        .filter(|l| l.contains(EMBEDDER_HTTP_ERROR_NOTICE))
        .count();
    assert_eq!(
        notices, 1,
        "SC-L15: five identical failures owe the screen exactly ONE notice, \
         and it got {notices}: {shown:?}"
    );
    let records_on_screen = shown
        .iter()
        .filter(|l| l.contains(EMBEDDER_FAILURE_RECORD))
        .count();
    assert_eq!(
        records_on_screen, CONSECUTIVE_FAILING_TURNS,
        "REQ-L19 puts every WARN on the screen, so the records belong there \
         alongside the one notice. Moving this number is a decision about \
         REQ-L19 and about what `ok` is derived from, not a refactor: {shown:?}"
    );

    drop(handle);

    let written = wait_for_file(dir.path(), &format!("turn {CONSECUTIVE_FAILING_TURNS}"));
    for turn in 1..=CONSECUTIVE_FAILING_TURNS {
        assert!(
            written.contains(&format!("HTTP 500 on turn {turn}")),
            "SC-L15: the file owes one record per failure, and turn {turn} is \
             missing from: {written}"
        );
    }
}

#[test]
fn a_transition_flushed_at_close_reaches_the_file() {
    // What the close-ordering guard in `main.rs` is guarding, and the half a
    // reader is most likely to doubt. `health_flush` is usually described by
    // what it puts on the SCREEN, and on that reading where it sits at exit
    // would not matter. It has a FILE-bound half too: the transition's text
    // carries the operator's own `log_dir` and the emitter's cause key, both
    // runtime data, so the auditor masks what it recognises and the alarm that
    // says masking happened goes to the appender -- a detached writer thread
    // the exit waits for exactly once.
    //
    // # Why the fixture has this shape
    //
    // A subsystem's FIRST transition is immediate, so the only thing a flush
    // can be holding is a later one: here a CAUSE CHANGE inside a subsystem
    // already degraded. And the secret is registered BETWEEN the two events,
    // because the alarm latches per `(secret, target)` -- had the first line
    // carried it, the flushed one would raise nothing and this would pass over
    // a flush that did no work.
    //
    // # What makes it discriminating
    //
    // The needle is the alarm's HEALTH target, not the secret's name. The
    // second event's own file line also carries the value and raises its own
    // alarm under `magi_rs::memory`, so a check for the name alone is green
    // with `health_flush` deleted.
    const SECRET_VALUE: &str = "flushed-transition-secret-value-long-enough";
    const SECRET: &str = "FLUSHED_TRANSITION_SECRET";
    let dir = tempfile::tempdir().expect("a temp dir");
    let (handle, _screen) = start(dir.path(), "info");

    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "nonesuch",
        cause.name = "first_cause",
        "a subsystem with no declared message table row"
    );

    assert!(
        magi_rs::logging::process_auditor().register_secret(
            magi_rs::logging::auditor::SecretName::new(SECRET),
            &[SECRET_VALUE]
        ),
        "the value must be long enough for the exact pass, or nothing is masked, no \
         alarm is raised, and this test proves nothing"
    );

    tracing::event!(
        target: "magi_rs::memory",
        tracing::Level::WARN,
        cause.subsystem = "nonesuch",
        cause.name = SECRET_VALUE,
        "the cause changed inside a subsystem that is already degraded"
    );

    let before = health_alarm_lines(&everything_written(dir.path()), SECRET);
    assert!(
        before.is_empty(),
        "the health target already raised an alarm before the close, so the assertion          below cannot tell the flush from what preceded it: {before:?}"
    );

    handle.health_flush();

    let written = wait_for_file(dir.path(), HEALTH_TARGET);
    let alarms = health_alarm_lines(&written, SECRET);
    assert!(
        !alarms.is_empty(),
        "the transition flushed at close produced no file-bound event, so ordering it \
         against the exit drain would be guarding nothing: {written}"
    );
    assert!(
        !written.contains(SECRET_VALUE),
        "the alarm quoted the value it was raised for: {alarms:?}"
    );
}

/// The target a health transition's alarm names, which `logging` keeps `pub(crate)`.
///
/// Written out rather than imported, and the duplication is the point: this test
/// asserts on what an OPERATOR reads out of the file, so a rename of the constant
/// that changed what they read has to fail here rather than follow along silently.
const HEALTH_TARGET: &str = "magi_rs::logging::health";

/// Every written line that is an alarm raised by `secret` under the health target.
///
/// Both halves are required together: the secret's name alone also appears on the
/// alarm the emitting event raises under its own target, which is present whether or
/// not the close flushed anything.
fn health_alarm_lines(written: &str, secret: &str) -> Vec<String> {
    written
        .lines()
        .filter(|l| l.contains(secret) && l.contains(HEALTH_TARGET))
        .map(ToString::to_string)
        .collect()
}
