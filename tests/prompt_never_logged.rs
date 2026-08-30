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

/// No emission site in the product hands a prompt to `tracing`.
///
/// **This is the guarantee, and it is a property of the EMITTERS, not of the
/// layer.** The layer masks registered secrets and things that look like
/// credentials; a user's prompt is neither, so if any call site passed one it
/// would be written out verbatim and nothing downstream would stop it. The
/// protection is that no call site does.
///
/// The behavioural test below cannot see this: it emits three events that never
/// contained the prompt and then finds the prompt absent, which holds whether or
/// not anything guards it. Deleting the auditor entirely leaves it green. It is
/// kept because it pins the shape of what IS written, but the requirement needs
/// this one.
#[test]
fn no_call_site_in_the_product_logs_a_prompt_or_a_user_message() {
    // The fields a prompt would arrive under. Narrow on purpose: a broad match
    // on "prompt" hits doc comments and constant names, and a guardian that
    // cries wolf gets relaxed until it stops speaking.
    const FORBIDDEN: &[&str] = &[
        "prompt = ",
        "prompt =%",
        "prompt = %",
        "prompt = ?",
        "user_message = ",
        "envelope = ",
    ];

    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            scanned += 1;
            // **The whole invocation, not one line.** `tracing::info!(` sits on
            // its own line and the fields follow it, so a line-based scan looks
            // at the macro name and the field separately and matches neither.
            // The first version of this test did exactly that, and its mutation
            // -- a real call site given a `prompt` field -- stayed green.
            for (start, _) in text.match_indices("tracing::") {
                let Some(rest) = text.get(start..) else {
                    continue;
                };
                // `event!` is the general form the five level macros expand to,
                // and a call site can use it directly -- `render_fixture!` does.
                // Listing only the five leaves the one that covers them all.
                let is_emit = ["info!", "debug!", "warn!", "error!", "trace!", "event!"]
                    .iter()
                    .any(|m| rest.starts_with(&format!("tracing::{m}")));
                if !is_emit {
                    continue;
                }
                let Some(open) = rest.find('(') else { continue };
                // `skip(open)` would be wrong here and was: `char_indices`
                // counts CHARACTERS while `open` is a BYTE offset, so the walk
                // started somewhere else entirely and met a `)` before any `(`.
                let Some(from_open) = rest.get(open..) else {
                    continue;
                };
                let mut depth = 0usize;
                let mut end = open;
                for (i, c) in from_open.char_indices() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                end = open + i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(call) = rest.get(open..=end) else {
                    continue;
                };
                if FORBIDDEN.iter().any(|f| call.contains(f)) {
                    let line = text.get(..start).map_or(0, |p| p.lines().count());
                    offenders.push(format!("{}:{}", path.display(), line + 1));
                }
            }
        }
    }

    // Without this the loop above proves nothing when the walk finds no files,
    // which is how a path typo turns a guardian into a formality.
    assert!(scanned > 50, "the source walk found only {scanned} files");
    assert!(
        offenders.is_empty(),
        "a call site passes a prompt to the log, where nothing downstream will          mask it: {offenders:?}"
    );
}

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
