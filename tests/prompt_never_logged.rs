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
                //
                // **`span!` is deliberately absent, and the test below is why.**
                // A span field would leak only if something READ it, and the
                // layer implements `on_event` alone -- no `on_new_span`, no
                // `on_record`, no `on_enter` -- so a span's fields are never
                // visited, formatted or written. Scanning for `span!` would fail
                // a call site that cannot leak. That exemption is an assumption
                // about another file, which is how a documented exemption
                // quietly becomes a hole, so
                // `the_layer_reads_no_span_fields_which_is_why_spans_are_exempt`
                // pins it: add a span callback and it goes red.
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
        "a call site passes a prompt to the log, where nothing downstream will mask it: {offenders:?}"
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

/// The auditor masks a registered value AND alarms — both, never one.
///
/// **Renamed, because the old name promised a file.** It read
/// `a_registered_secret_never_reaches_the_file_even_at_trace_level` and touched
/// no file: no directory, no subscriber, no emission, no trace level. It called
/// `Auditor::audit` and read the returned string. Two reviewers reached the same
/// conclusion independently, and they were right — the name described coverage
/// nobody had here.
///
/// The property the old name claimed IS guarded, in
/// `tests/canary_both_mouths.rs`: four tests there install the layer with
/// `Filter::parse("trace")`, emit through the real dispatcher and read the file
/// back. Writing a second fixture for it would duplicate that for nothing. What
/// was wrong was the name, so the name is what changed.
#[test]
fn the_auditor_both_masks_a_registered_value_and_alarms_on_it() {
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

/// The span exemption above is a fact about `magi_layer.rs`, so it is checked.
///
/// The scanner deliberately ignores `tracing::span!` because a span's fields are
/// never read: the layer implements `on_event` and nothing else. That is true
/// today and nothing makes it stay true. Adding `on_new_span` or `on_record`
/// would start recording span fields and silently open the exact hole the
/// scanner exists to close, with no test anywhere going red.
///
/// So this one does. It is a source check rather than a behavioural one because
/// the absence of a trait method is not observable at runtime — the default
/// implementation does nothing, which is indistinguishable from not reading the
/// fields.
///
/// Complexity: `O(bytes of magi_layer.rs)`.
#[test]
fn the_layer_reads_no_span_fields_which_is_why_spans_are_exempt() {
    let source = std::fs::read_to_string("src/logging/magi_layer.rs")
        .expect("the layer's source is part of the tree this test runs in");
    for callback in [
        "fn on_new_span",
        "fn on_record",
        "fn on_enter",
        "fn on_close",
    ] {
        assert!(
            !source.contains(callback),
            "magi_layer.rs now implements `{callback}`, so span fields reach the log and the prompt scanner above must stop exempting `span!`. Add the span macros to its list, then delete this assertion's entry."
        );
    }
    // Without this the loop is vacuously true over an unreadable or moved file,
    // which is the shape of guardian this milestone has now produced eight
    // times.
    assert!(
        source.contains("fn on_event"),
        "magi_layer.rs no longer implements `on_event`, so this test is reading the wrong file and proves nothing"
    );
}
