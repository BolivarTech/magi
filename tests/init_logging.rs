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
