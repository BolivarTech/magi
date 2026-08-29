// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! `init_logging` is idempotent, and the second call says what it discarded.
//!
//! An integration binary because it installs a **global** subscriber: `cargo
//! nextest` gives each test its own process, so one install is one process's.
//! A unit test would fight every other test in the same binary for the global.

use std::sync::{Arc, Mutex};

use magi_rs::logging::auditor::Audited;
use magi_rs::logging::{init_logging, LoggingConfig, NoticeDelivery};

/// Records what the sink was handed.
#[derive(Default)]
struct Recording {
    lines: Mutex<Vec<String>>,
}

impl NoticeDelivery for Recording {
    fn deliver(&self, line: &Audited) {
        if let Ok(mut l) = self.lines.lock() {
            l.push(line.as_str().to_string());
        }
    }
}

#[test]
fn a_second_call_returns_the_same_handle_and_names_what_it_discarded() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(Recording::default());

    let cfg = LoggingConfig {
        log_dir: first_dir.path().to_path_buf(),
        file_level: tracing::Level::INFO,
    };
    let first = init_logging(&cfg, sink.clone(), None).expect("first init");

    // A SECOND call with a DIFFERENT directory. `set_global_default` panics on a
    // second install, so without idempotence this aborts the process — REQ-L35
    // violated by the mechanism meant to protect it.
    let other = LoggingConfig {
        log_dir: second_dir.path().to_path_buf(),
        file_level: tracing::Level::DEBUG,
    };
    let second = init_logging(&other, sink.clone(), None).expect("second init is Ok, not Err");

    // Returning `Err` would force every caller to decide whether that is fatal,
    // and the answer is always no: logging is already running.
    assert_eq!(
        first.dropped(),
        second.dropped(),
        "the same subsystem, not a second one"
    );

    let said = sink.lines.lock().unwrap().join("\n");
    assert!(
        said.contains("DISCARDED"),
        "the notice must name that the CONFIGURATION was dropped, not merely \
         that a second call happened — otherwise an operator changes log_dir, \
         reads a generic notice, and hunts for a file that never moved: {said}"
    );
    assert!(
        !second_dir.path().join("magi-2026-01-01.log").exists(),
        "and nothing was written to the directory that was ignored"
    );
}

#[test]
fn the_run_id_is_stable_within_a_process_and_has_the_documented_shape() {
    let a = magi_rs::logging::run_id();
    let b = magi_rs::logging::run_id();
    assert_eq!(a, b, "one run, one id");

    let (pid, hex) = a.split_once('-').expect("<pid>-<hex16>");
    assert!(pid.parse::<u32>().is_ok(), "the pid half: {a}");
    assert_eq!(hex.len(), 16, "64 bits, not 32: {a}");
    assert!(
        hex.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "lowercase hex: {a}"
    );
}

/// SC-L79: filtering the daily file by a run returns **only** that run's lines.
///
/// The file is shared by every concurrent invocation -- nine runs landed in one
/// file during this milestone's own smoke pass -- so the run id on the line is
/// the whole of what replaces the per-run file REQ-H24 retired. Retiring that
/// contract without this leaves a CI consumer a log it cannot filter.
///
/// **Every line, including continuations.** A payload past the 4096-byte
/// threshold is split, and a continuation line missing the id is a line the
/// filter drops -- which is the same defect, only harder to see.
#[test]
fn every_line_of_the_daily_file_carries_the_run_id() {
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(Recording::default());
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_level: tracing::Level::INFO,
    };
    let handle = init_logging(&cfg, sink, None).expect("init");

    tracing::info!(target: "magi_rs::agent", "a short one");
    // Comfortably past MAX_LINE_BYTES, so this event becomes several lines.
    //
    // **Prose, not padding.** A long run of one repeated character is exactly
    // what the auditor's shape pass calls a secret, so `"x".repeat(12_000)`
    // arrives as `***` -- one short line, and the split this test exists to
    // exercise never happens. That masking already produced one ineffective
    // guardian in this milestone; the fixture has to survive the redactor.
    let long = "the quick brown fox jumps over the lazy dog ".repeat(300);
    tracing::info!(target: "magi_rs::agent", "{long}");

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(300));

    let mut written = String::new();
    for entry in std::fs::read_dir(dir.path()).expect("read_dir").flatten() {
        written.push_str(&std::fs::read_to_string(entry.path()).unwrap_or_default());
    }

    let lines: Vec<&str> = written.lines().filter(|l| !l.trim().is_empty()).collect();
    // Without this the loop below is vacuously true over an empty file, which
    // is this repository's most frequent way of shipping a guardian that
    // guards nothing. Two events, one of them split, is more than three lines.
    assert!(
        lines.len() > 3,
        "the fixture produced {} lines, so the assertion below proves nothing: {written}",
        lines.len()
    );

    let needle = format!("run={}", magi_rs::logging::run_id());
    for line in &lines {
        assert!(
            line.contains(&needle),
            "a line the run filter would drop: {line}"
        );
    }
}

/// A backslash survives as ONE escape, not two.
///
/// Escaping is stage 3 of REQ-L64 and runs in the layer. The file sink then
/// escaped the layer's result a second time on its way into the chunker.
/// Nothing a test could see broke -- the line is still one line, the auditor
/// still ran first -- but the CONTENT is wrong: every Windows path in the log
/// read `C:\\Users` where it should read `C:\Users`, and a message already
/// escaped to `\n` arrived as `\\n`.
///
/// The two arms of the sink disagreed, which is why it went unnoticed. An alarm
/// is rendered inside the sink and escaped there exactly once, correctly; only
/// the line arm, which arrives pre-escaped, was escaped twice. A guardian over
/// the alarm path would have stayed green forever.
#[test]
fn a_backslash_is_escaped_once_and_not_twice() {
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(Recording::default());
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_level: tracing::Level::INFO,
    };
    let handle = init_logging(&cfg, sink, None).expect("init");

    // A real Windows path, which is where this shows up in practice.
    tracing::info!(target: "magi_rs::agent", "path {} here", r"C:\Users\jb");

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(300));

    let mut written = String::new();
    for entry in std::fs::read_dir(dir.path()).expect("read_dir").flatten() {
        written.push_str(&std::fs::read_to_string(entry.path()).unwrap_or_default());
    }

    // Without this the two assertions below hold over an empty file.
    assert!(
        written.contains("here"),
        "the fixture produced nothing: {written}"
    );
    // One escape doubles each backslash, so this is what lands on the wire.
    assert!(
        written.contains(r"C:\\Users\\jb"),
        "the path is not singly escaped: {written}"
    );
    assert!(
        !written.contains(r"C:\\\\Users"),
        "the path was escaped twice: {written}"
    );
}
