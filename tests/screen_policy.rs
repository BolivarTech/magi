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
    std::thread::sleep(std::time::Duration::from_millis(300));

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

    let written = everything_written(dir.path());
    assert!(
        written.contains("context assembled"),
        "and the file is where it must have gone instead: {written}"
    );
}
