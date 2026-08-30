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
    // The field NAMES a prompt would arrive under. Names, not spellings: the
    // first version listed `"prompt = "` and its sigil variants, which missed
    // `tracing`'s shorthand entirely. `info!(target: "t", prompt)` passes a
    // field called `prompt` carrying the local of that name, and `%prompt` and
    // `?prompt` do the same through Display and Debug. All three write the
    // value; none contains an `=`. A reviewer found this, and it is the one
    // finding in this round that could actually leak.
    //
    // Narrow is still the goal — a guardian that cries wolf gets relaxed until
    // it stops speaking — so `is_field_use` requires the name to sit where a
    // field sits, which keeps `system_prompt`, `prompt_tokens` and the word
    // inside a string literal out.
    const FORBIDDEN: &[&str] = &["prompt", "user_message", "envelope"];

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
                if FORBIDDEN.iter().any(|f| is_field_use(call, f)) {
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

/// Pins the SHAPE of what the layer writes — not the prompt guarantee.
///
/// **Known and accepted: this one cannot fail for the property its name
/// suggests.** It emits events that never contained the prompt and then finds
/// the prompt absent, which holds whether or not anything guards it; deleting
/// the auditor leaves it green. The prompt guarantee is carried by
/// `no_call_site_in_the_product_logs_a_prompt_or_a_user_message` above, which
/// checks the emitters, where the property actually lives. This is kept for
/// what it does prove: that the fixture writes, and what it writes has the
/// shape the rest of these tests read.
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

/// The span exemption above is a fact about every `Layer` under `src/`.
///
/// The scanner deliberately ignores `tracing::span!` because a span's fields are
/// never read: no layer magi-rs installs implements a span callback. That is
/// true today and nothing makes it stay true. Adding `on_new_span` or
/// `on_record` would start recording span fields and silently open the exact
/// hole the scanner exists to close, with no test anywhere going red.
///
/// So this one does. It is a source check rather than a behavioural one because
/// the absence of a trait method is not observable at runtime — the default
/// implementation does nothing, which is indistinguishable from not reading the
/// fields.
///
/// **It scans every `impl Layer` under `src/`, not one named file.** The first
/// version read `src/logging/magi_layer.rs` and nothing else, which a reviewer
/// caught: a second `impl Layer` in any other file would have carried span
/// callbacks straight past it, and moving the existing one would have left the
/// check reading a file that no longer implements anything. Finding the impls
/// rather than naming their file removes both.
///
/// **The impl is found by shape, not by the generic's spelling.** A first
/// version matched the literal `Layer<S> for`, which three reviewers flagged
/// together: `impl<Sub> Layer<Sub> for` is the same impl and the same risk, and
/// the search would have walked straight past it. The walk now reads `Layer<`,
/// tracks angle-bracket depth to the matching `>` so a nested argument does not
/// end it early, and accepts what follows only if it is `for`.
///
/// Complexity: `O(bytes under src/)`.
#[test]
fn no_layer_in_the_tree_reads_span_fields_which_is_why_spans_are_exempt() {
    const CALLBACKS: [&str; 4] = [
        "fn on_new_span",
        "fn on_record",
        "fn on_enter",
        "fn on_close",
    ];

    let mut layers = Vec::new();
    // `CARGO_MANIFEST_DIR`, not a relative `"src"`: a relative path resolves
    // against the process's working directory, which the test runner owns. The
    // prompt scanner above already roots itself this way; this one claimed to
    // and did not.
    let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
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
            if declares_a_layer_impl(&text) {
                layers.push((path, text));
            }
        }
    }

    // Two guards against a vacuous pass, which is the shape of defect this
    // milestone has now produced eight times. The first catches a walk that
    // found nothing; the second catches a walk that found only doubles, after
    // the production layer was renamed or moved out from under it.
    assert!(
        !layers.is_empty(),
        "the walk found no `impl Layer` at all, so the loop below proves nothing"
    );
    assert!(
        layers.iter().any(|(p, _)| p.ends_with("magi_layer.rs")),
        "the product's own layer is not among the {} found, so this test is no longer reading it: {:?}",
        layers.len(),
        layers.iter().map(|(p, _)| p).collect::<Vec<_>>()
    );

    for (path, text) in &layers {
        for callback in CALLBACKS {
            assert!(
                !text.contains(callback),
                "{} now implements `{callback}`, so span fields reach the log and the prompt scanner above must stop exempting `span!`. Add the span macros to its list, then delete this assertion's entry.",
                path.display()
            );
        }
    }
}

/// Whether *text* contains an `impl … Layer<…> for …` header.
///
/// Matching the literal `Layer<S> for` would have missed `impl<Sub> Layer<Sub>
/// for`, which is the same impl written with a different generic name. This
/// reads the trait name, walks its generic arguments by bracket depth — so a
/// nested `Layer<Foo<Bar>>` does not end at the inner `>` — and accepts the
/// match only when `for` follows.
///
/// # Parameters
///
/// * `text` — the whole contents of one source file.
///
/// # Returns
///
/// `true` if any such header is present.
///
/// # Complexity
///
/// `O(n)` in the file's length: each `Layer<` scans forward only to its own
/// matching `>`, and those spans do not overlap.
fn declares_a_layer_impl(text: &str) -> bool {
    const TRAIT_OPEN: &str = "Layer<";
    for (at, _) in text.match_indices(TRAIT_OPEN) {
        let Some(rest) = text.get(at + TRAIT_OPEN.len()..) else {
            continue;
        };
        if !impl_precedes(text, at) {
            continue;
        }
        let mut depth = 1usize;
        let mut close = None;
        let mut previous = ' ';
        for (i, c) in rest.char_indices() {
            match c {
                '<' => depth += 1,
                // `->` inside a generic argument is a return arrow, not a
                // closing bracket. `Layer<fn() -> T>` would otherwise close at
                // the arrow and the header would go unrecognised — a false
                // NEGATIVE, which is the direction that matters here.
                '>' if previous != '-' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
            previous = c;
        }
        let Some(close) = close else { continue };
        let Some(after) = rest.get(close + 1..) else {
            continue;
        };
        if after.trim_start().starts_with("for ") {
            return true;
        }
    }
    false
}

/// Whether *args* uses *name* in the position a `tracing` field occupies.
///
/// `tracing` accepts four spellings for the same leak — `name = value`,
/// `name` (shorthand for the local of that name), `%name` and `?name` — and
/// only the first contains an `=`. A scanner that looks for `"name = "`
/// therefore misses three of the four, which is the gap this closes.
///
/// The position test is what keeps it narrow. The name must begin a field, so
/// what precedes it (past an optional `%` or `?` sigil and any spaces) is the
/// opening paren, a comma or a brace — never a letter, which excludes
/// `system_prompt`, and never a quote, which excludes the word inside a string
/// literal. What follows must not continue the identifier, which excludes
/// `prompt_tokens`.
///
/// # Parameters
///
/// * `args` — the macro invocation's argument list, parens included.
/// * `name` — the field name to look for.
///
/// # Returns
///
/// `true` if the invocation passes a field by that name.
///
/// # Complexity
///
/// `O(n)` in the argument list's length.
fn is_field_use(args: &str, name: &str) -> bool {
    for (at, _) in args.match_indices(name) {
        let after_ok = args
            .get(at + name.len()..)
            .and_then(|s| s.chars().next())
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if !after_ok {
            continue;
        }
        let before = args.get(..at).unwrap_or_default();
        let mut chars = before.chars().rev().peekable();
        if matches!(chars.peek(), Some('%' | '?')) {
            chars.next();
        }
        while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
            chars.next();
        }
        if matches!(chars.next(), Some('(' | ',' | '{')) {
            return true;
        }
    }
    false
}

/// Whether an `impl` opens the header that reaches *at*.
///
/// Without this the walk matched any `Layer<…> for` — including one quoted in a
/// doc comment, which is a false positive, and this very file's own prose,
/// which is worse because it is guaranteed. It looks back over the header,
/// which may wrap across lines, and ignores comment lines so that describing an
/// impl is not the same as declaring one.
///
/// # Parameters
///
/// * `text` — the whole file.
/// * `at` — the byte offset of the trait name.
///
/// # Returns
///
/// `true` if the enclosing statement is an `impl`.
///
/// # Complexity
///
/// `O(1)` — the look-back is bounded by [`HEADER_LOOKBACK_BYTES`].
fn impl_precedes(text: &str, at: usize) -> bool {
    /// How far back an `impl` header may start. Three wrapped lines is more
    /// than rustfmt produces for one.
    const HEADER_LOOKBACK_BYTES: usize = 240;

    let from = at.saturating_sub(HEADER_LOOKBACK_BYTES);
    let Some(window) = text.get(from..at) else {
        return false;
    };
    // A `;` or `}` ends the previous item, so anything before it belongs to
    // something else.
    let header = window
        .rsplit_once([';', '}'])
        .map_or(window, |(_, tail)| tail);
    header
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .any(|l| l.contains("impl"))
}
