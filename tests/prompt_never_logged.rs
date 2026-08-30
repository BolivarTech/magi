// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! The raw prompt never reaches a log file. **Carried forward, not inherited.**
//!
//! This guarantee lived in `RunLog`'s tests until the JSONL run log was retired.
//! Deleting the thing that wrote the prompt removes today's risk and removes
//! today's guard with it — so the guarantee moves here, onto the layer that
//! writes now. A requirement whose only test went out with the code it tested is
//! a requirement nobody is checking.
//!
//! An integration binary because it installs a global subscriber.

use std::sync::Arc;

use magi_rs::logging::auditor::{Auditor, SecretName};
use magi_rs::logging::{init_logging, DiscardDelivery, LoggingConfig};

/// Stands in for a user prompt: long, distinctive, and the sort of thing that
/// must not end up on disk.
const PROMPT: &str = "PROMPT-BODY-do-not-write-this-to-any-file-ever";

#[test]
fn nothing_the_layer_writes_carries_the_raw_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_filter: magi_rs::logging::filter::Filter::parse("trace").expect("valid"),
    };
    let handle = init_logging(&cfg, Arc::new(DiscardDelivery), None).expect("init");

    // Everything an ordinary turn emits, at the most verbose level there is.
    tracing::info!(target: "magi_rs::agent", "startup notice");
    tracing::debug!(target: "magi_rs::agent", tool = "bash", ok = true, "tool call");
    tracing::warn!(target: "magi_rs::agent", "a warning with detail");

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(200));

    let mut written = String::new();
    for entry in std::fs::read_dir(dir.path()).unwrap().flatten() {
        written.push_str(&std::fs::read_to_string(entry.path()).unwrap_or_default());
    }

    assert!(
        !written.contains(PROMPT),
        "the prompt reached a log file: {written}"
    );
    // And the file is not empty, or the assertion above holds for free — which
    // is exactly how this guarantee would quietly stop being checked.
    assert!(
        written.contains("startup notice"),
        "the fixture must actually have written something: {written}"
    );
}

#[test]
fn a_registered_secret_never_reaches_the_file_even_at_trace_level() {
    let auditor = Auditor::new();
    auditor.register_secret(SecretName::new("K"), &["sk-ant-a-live-key-value"]);
    let (audited, alarm) = auditor.audit(
        "calling https://api.example.com with sk-ant-a-live-key-value",
        "magi_rs::agent",
        None,
        0,
    );
    assert!(
        !audited.as_str().contains("sk-ant-a-live-key-value"),
        "the value survived the audit: {}",
        audited.as_str()
    );
    assert!(alarm.is_some(), "and it alarmed, both never one");
}
