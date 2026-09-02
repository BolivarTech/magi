// Author: Julian Bolivar
// Version: 0.18.1
// Date: 2026-09-02

//! MS2 task 0.1: does MS1's plumbing actually work when something drives it?
//!
//! # Why this exists
//!
//! MS2 builds on four things MS1 declares but never exercises in production:
//! `CauseKey` propagating through `Audited`, `NoticeSink::emit`'s
//! non-deduplicating delivery, `MagiLayer::with_tui`, and
//! `LoggingHandle::health_tick`/`health_flush`. In MS1 all four are no-ops by
//! construction — nothing ever builds a `CauseKey`, nothing ever calls
//! `with_tui` — so MS1's own gate could pass green with any one of them
//! broken underneath. Finding that out after `health.rs` and the TUI wiring
//! are already built on top costs both of those phases; finding it here costs
//! one file.
//!
//! # Where the fourth test lives, and why not here
//!
//! `NoticeSink::emit` is declared in `src/agent/mode_classifier.rs`, which is
//! part of the **binary** crate (`mod agent;` in `main.rs`), not the library
//! (`src/lib.rs` has no `pub mod agent`). An integration test under `tests/`
//! links only against the library crate `magi_rs`, so it cannot name
//! `NoticeSink` at all — this is the same lib/bin split `CLAUDE.md` documents
//! for the headless surfaces, discovered here by trying to write the test the
//! plan describes. `test_emit_delivers_without_deduplicating` therefore lives
//! in `src/agent/mode_classifier.rs`'s own `#[cfg(test)] mod tests`, which is
//! the only place inside the crate that can see the trait it probes.
//!
//! # Isolation
//!
//! Two of the three tests here install a **global** `tracing` subscriber
//! (`test_health_tick_and_flush_are_reachable_from_the_handle`, through
//! `init_logging`) or would collide with one if run twice in the same
//! process. `cargo nextest` gives each test its own process, so that is safe;
//! plain `cargo test` — which shares one process across every test in this
//! binary — is not the runner this file is written for, matching the existing
//! precedent in `tests/init_logging.rs` and `tests/canary_both_mouths.rs`.

use std::sync::{Arc, Mutex};

use tracing_subscriber::layer::SubscriberExt as _;

use magi_rs::logging::appender::DailyAppender;
use magi_rs::logging::auditor::{Audited, Auditor, CauseKey};
use magi_rs::logging::filter::Filter;
use magi_rs::logging::magi_layer::{FileSink, MagiLayer, TuiSink};
use magi_rs::logging::{init_logging, DiscardDelivery, LoggingConfig, NoticeDelivery};

/// Records every [`Audited`] line a sink was handed, in order.
///
/// Kept as `Audited` clones rather than `String`s: the round-trip test needs
/// `Audited::cause()`, which a rendered string throws away.
#[derive(Default)]
struct CapturingSink {
    lines: Mutex<Vec<Audited>>,
}

impl NoticeDelivery for CapturingSink {
    fn deliver(&self, line: &Audited) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(line.clone());
        }
    }
}

/// SC's plumbing check for `CauseKey`: an event that declares one must come
/// out the other side of the auditor still carrying it.
///
/// Uses the TUI branch to observe the `Audited` directly — that sink's
/// `deliver` is the only place in the layer that hands one back to the
/// caller, since the file branch only ever sees a queued byte count.
#[test]
fn test_a_cause_key_survives_the_round_trip_through_audited() {
    let dir = tempfile::tempdir().expect("tempdir");
    let appender = Arc::new(DailyAppender::new(dir.path()).expect("appender"));
    let tui_sink = Arc::new(CapturingSink::default());

    let layer = MagiLayer::new(
        FileSink::new(appender),
        Filter::parse("trace").expect("valid filter"),
        Arc::new(Auditor::new()),
        Arc::new(DiscardDelivery),
    )
    .with_tui(
        TuiSink::new(Arc::clone(&tui_sink) as Arc<dyn NoticeDelivery>),
        tracing::Level::WARN,
    );

    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        tracing::warn!(
            cause.subsystem = "embedder",
            cause.name = "http_500",
            "embedding request failed"
        );
    });

    let lines = tui_sink.lines.lock().expect("not poisoned");
    let audited = lines.first().expect("the WARN must have reached the sink");
    assert_eq!(
        audited.cause(),
        Some(CauseKey::new("embedder", "http_500")),
        "the cause key must survive the round trip through Audited, \
         not come back as None"
    );
}

/// `MagiLayer::with_tui` actually wires the second mouth: `Some(...)` reaches
/// both branches, and with no `with_tui` call at all the file branch keeps
/// working on its own.
#[test]
fn test_with_tui_wires_the_second_sink() {
    let dir = tempfile::tempdir().expect("tempdir");
    let appender = Arc::new(DailyAppender::new(dir.path()).expect("appender"));
    let auditor = Arc::new(Auditor::new());
    let file_filter = Filter::parse("trace").expect("valid filter");

    // Some((tui, level)): a WARN reaches both the file branch and the screen.
    let tui_sink = Arc::new(CapturingSink::default());
    let with_tui = MagiLayer::new(
        FileSink::new(Arc::clone(&appender)),
        file_filter.clone(),
        Arc::clone(&auditor),
        Arc::new(DiscardDelivery),
    )
    .with_tui(
        TuiSink::new(Arc::clone(&tui_sink) as Arc<dyn NoticeDelivery>),
        tracing::Level::WARN,
    );

    let subscriber = tracing_subscriber::registry().with(with_tui);
    tracing::subscriber::with_default(subscriber, || {
        tracing::warn!(target: "magi_rs::tests", "wired warning");
    });

    assert_eq!(
        tui_sink.lines.lock().expect("not poisoned").len(),
        1,
        "with_tui must wire the screen branch"
    );
    assert!(
        appender.queued_bytes() > 0,
        "the file branch must still receive the event"
    );

    // None: no `with_tui` call at all, so there is no second mouth to reach.
    let unattached = Arc::new(CapturingSink::default());
    let without_tui = MagiLayer::new(
        FileSink::new(Arc::clone(&appender)),
        file_filter,
        auditor,
        Arc::new(DiscardDelivery),
    );
    let queued_before = appender.queued_bytes();
    let subscriber = tracing_subscriber::registry().with(without_tui);
    tracing::subscriber::with_default(subscriber, || {
        tracing::warn!(target: "magi_rs::tests", "file only warning");
    });

    assert!(
        appender.queued_bytes() > queued_before,
        "the file branch must still work when with_tui was never called"
    );
    assert!(
        unattached.lines.lock().expect("not poisoned").is_empty(),
        "a sink that was never wired in must never receive anything"
    );
}

/// `LoggingHandle::health_tick`/`health_flush` are no-ops in MS1, but the
/// probe's job is to confirm they are reachable and inert, never a panic —
/// MS2's TUI event loop and headless turn boundary call both unconditionally.
#[test]
fn test_health_tick_and_flush_are_reachable_from_the_handle() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_filter: Filter::parse("info").expect("valid filter"),
    };
    let handle = init_logging(&cfg, Arc::new(DiscardDelivery), None).expect("init");

    handle.health_tick(std::time::Instant::now());
    handle.health_flush();
}
