// Author: Julian Bolivar
// Version: 0.18.1
// Date: 2026-08-31

//! SC-L32: the screen branch delivers audited output and nothing else.
//!
//! # Why this is a SECURITY test and not a formatting one
//!
//! In the TUI's Selection mode the `y` key copies a whole message to the system
//! clipboard. So an unredacted credential on screen is not merely visible, it is
//! one keystroke from leaving the machine — which makes the second mouth every
//! bit as dangerous as the file, and it is the mouth MS1 never wired.
//!
//! # Why it has a binary to itself (R-L52b)
//!
//! It emits through the real dispatcher and therefore needs its own global
//! subscriber and its own process auditor registry. `cargo nextest` gives each
//! test its own process, so isolation comes from the runner.
//!
//! # Mutation
//!
//! Remove the auditor from the screen branch of `MagiLayer::on_event` — deliver
//! the rendered text rather than the audited line — and this goes red.

use std::sync::{Arc, Mutex};

use magi_rs::logging::auditor::{Audited, SecretName};
use magi_rs::logging::magi_layer::TuiSink;
use magi_rs::logging::{init_logging, DiscardDelivery, LoggingConfig, NoticeDelivery};

/// The password inside the credentialled endpoint, caught by shape alone.
const PASSWORD: &str = "hunter2-and-then-some";
/// A credentialled endpoint, exactly as an operator would configure one.
const ENDPOINT: &str = "https://bob:hunter2-and-then-some@example.com/v1";
/// A secret with NO recognisable shape, and the fixture depends on that.
///
/// The URL password is found by pass 1 on position alone, so a screen branch
/// that skipped the auditor entirely but still went through some pattern scan
/// would hide behind it. Only registration finds this one, so it is the
/// assertion that proves the auditor ran on THIS branch.
const OPAQUE: &str = "correct horse battery staple";

/// Captures what the screen branch was handed.
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

#[test]
fn the_tui_buffer_receives_audited_output_only() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let screen = Arc::new(CapturingSink::default());
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_filter: magi_rs::logging::filter::Filter::parse("trace").expect("a valid filter"),
    };
    // The notice sink is DISCARDING on purpose: everything asserted below has
    // to have arrived through the screen branch, not through the layer's own
    // announcement channel.
    let _handle = init_logging(
        &cfg,
        Arc::new(DiscardDelivery),
        Some((TuiSink::new(screen.clone()), tracing::Level::TRACE)),
    )
    .expect("logging comes up");

    magi_rs::logging::register_process_secrets(&[(SecretName::new("BASE_URL_PASSWORD"), OPAQUE)]);

    tracing::error!(
        target: "magi_rs::agent",
        "GET {ENDPOINT} failed for session {OPAQUE}"
    );

    let shown = screen
        .lines
        .lock()
        .map(|l| l.join("\n"))
        .unwrap_or_else(|p| p.into_inner().join("\n"));

    assert!(
        shown.contains("failed for session"),
        "the fixture delivered nothing, so every assertion below holds for free: {shown}"
    );
    assert!(
        !shown.contains(PASSWORD),
        "the URL password reached the screen, and `y` copies it: {shown}"
    );
    assert!(
        !shown.contains(OPAQUE),
        "the shapeless secret reached the screen -- only the exact pass finds \
         this one, so this is what proves the auditor ran on this branch: {shown}"
    );
    assert!(
        shown.contains("***"),
        "and it was redacted rather than dropped: {shown}"
    );
}
