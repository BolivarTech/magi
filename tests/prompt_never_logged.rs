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
    {
        for (path, text) in source_files() {
            scanned += 1;
            // **The whole invocation, not one line.** `tracing::info!(` sits on
            // its own line and the fields follow it, so a line-based scan looks
            // at the macro name and the field separately and matches neither.
            // The first version of this test did exactly that, and its mutation
            // -- a real call site given a `prompt` field -- stayed green.
            // The path's own spacing is removed first. `tracing :: info!` is
            // the same call site and the anchor is a literal, so without this
            // it matched nothing at all -- found by running it, after a
            // reviewer named it. Offsets below index the FLATTENED text, and
            // only the reported line number is derived from it, so the shift
            // costs an approximate line rather than a wrong verdict.
            // Comments come out of the WHOLE file, not just the extracted
            // call. `tracing::/*c*/info!` is a call site the anchor could not
            // see at all -- the same trivia family, one level further out. The
            // reported line number was already approximate, so nothing else
            // changes.
            let text = tighten_paths(&without_comments(&text));
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
                // `info![...]` and `info!{...}` are the same invocation under
                // the other two delimiters rustc accepts, and anchoring on `(`
                // alone let both through untouched.
                let Some(open) = rest.find(['(', '[', '{']) else {
                    continue;
                };
                // `skip(open)` would be wrong here and was: `char_indices`
                // counts CHARACTERS while `open` is a BYTE offset, so the walk
                // started somewhere else entirely and met a `)` before any `(`.
                let Some(from_open) = rest.get(open..) else {
                    continue;
                };
                // No second comment pass: the text this slices was stripped
                // above, so a comment cannot reach the predicates from here.
                let Some(call) = argument_list(from_open)
                    .and_then(|len| rest.get(open..open + len))
                    .map(as_parenthesised)
                else {
                    continue;
                };
                if FORBIDDEN
                    .iter()
                    .any(|f| is_field_use(&call, f) || is_value_use(&call, f))
                {
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

    // Rooted at `CARGO_MANIFEST_DIR` by `source_files`, not at a relative
    // "src": a relative path resolves against the process's working directory,
    // which the test runner owns.
    let layers: Vec<_> = source_files()
        .into_iter()
        .filter(|(_, text)| declares_a_layer_impl(text))
        .collect();

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

/// *call* with its outer delimiters rewritten to parentheses.
///
/// **The extraction learned `[` and `{`; the position predicates did not.** A
/// reviewer raised it as critical and was right: `info![prompt]` extracted
/// correctly and then matched nothing, because the name sat against a `[` that
/// no boundary set contained. Worse, the two mutation rows that were supposed to
/// cover this put the name after a comma, so they proved the extraction and
/// stayed silent about the matching — the guardian's own harness hiding the
/// hole, which is the defect this milestone exists to keep finding.
///
/// Normalising here rather than widening both predicates keeps every position
/// rule written once, the braced field block `{ prompt = p }` included: that
/// brace is INTERNAL and is left alone.
///
/// # Parameters
///
/// * `call` — the extracted invocation, delimiters included.
///
/// # Returns
///
/// The same text with the first and last characters as `(` and `)`.
///
/// # Complexity
///
/// `O(n)` in the call's length.
fn as_parenthesised(call: &str) -> String {
    let mut chars: Vec<char> = call.chars().collect();
    if matches!(chars.first(), Some('[' | '{')) {
        chars[0] = '(';
    }
    if let Some(last) = chars.last_mut() {
        if matches!(last, ']' | '}') {
            *last = ')';
        }
    }
    chars.into_iter().collect()
}

/// *call* with every comment span replaced by a single space.
///
/// **The walk skipped comments; the predicates still read them.** A reviewer
/// worked out the consequence and it was the opposite of what the previous
/// docstring claimed: `info!(target: "t", /* x */ prompt)` leaves `*/` in front
/// of the name, no boundary set contains it, and the field goes unseen. That is
/// a silent miss, not the false positive I declared — verified by running both
/// spellings in isolation, after a first attempt put two invocations in one
/// file and let the detected one hide the undetected one.
///
/// A space rather than nothing, so `a/*x*/b` cannot fuse into one identifier.
///
/// # Parameters
///
/// * `call` — the extracted invocation.
///
/// # Returns
///
/// The same text with comments blanked.
///
/// # Complexity
///
/// `O(n)` in the call's length.
fn without_comments(call: &str) -> String {
    let chars: Vec<char> = call.chars().collect();
    let mut out = String::with_capacity(call.len());
    let mut i = 0usize;
    while i < chars.len() {
        if let Some((skipped, _)) = literal_at(&chars, i) {
            out.extend(chars.get(i..i + skipped).unwrap_or_default());
            i += skipped;
            continue;
        }
        if let Some((skipped, _)) = comment_at(&chars, i) {
            out.push(' ');
            i += skipped;
            continue;
        }
        if let Some(c) = chars.get(i) {
            out.push(*c);
        }
        i += 1;
    }
    out
}

/// *text* with whitespace removed around `::` and before a macro's `!`.
///
/// The anchor is the literal `tracing::info!`, so any whitespace the compiler
/// tolerates inside that path is an evasion. The first version replaced three
/// spellings built from U+0020; a reviewer pointed out that a tab or a newline
/// does the same job, and so does a space before the bang. This removes the
/// whitespace itself rather than enumerating where it can sit.
///
/// # Parameters
///
/// * `text` — a source file.
///
/// # Returns
///
/// The text with those runs closed up.
///
/// # Complexity
///
/// `O(n)` in the text's length.
fn tighten_paths(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            // Look past this run of whitespace: if a `::` or a `!` follows, or
            // a `::` precedes, the run is inside a path and comes out.
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let follows_sep = out.ends_with("::");
            let precedes_sep = chars.get(j) == Some(&':') && chars.get(j + 1) == Some(&':');
            let precedes_bang = chars.get(j) == Some(&'!');
            if follows_sep || precedes_sep || precedes_bang {
                i = j;
                continue;
            }
            out.push(c);
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Whether *args* uses *name* in the position a `tracing` field occupies.
///
/// `tracing` accepts five spellings for the same leak — `name = value`,
/// `name` (shorthand for the local of that name), `%name`, `?name` and the raw
/// identifier `r#name` — and only two of them contain an `=`. A scanner that looks for `"name = "`
/// therefore misses three of the four, which is the gap this closes.
///
/// The position test is what keeps it narrow. The name must begin a field, so
/// what precedes it — past any sigils, with whitespace anywhere between them —
/// is the opening paren, a comma or a brace — never a letter, which excludes
/// `system_prompt`, and never a quote, which excludes the word inside a string
/// literal. What follows must not continue the identifier, which excludes
/// `prompt_tokens`.
///
/// **The brace earns its place**: `info!(target: "t", { prompt = p }, "msg")`
/// is a braced field block and it compiles, which was checked rather than
/// assumed when a reviewer asked whether any syntax needs it. It costs a false
/// positive on a struct literal written inside macro arguments — `Foo { prompt:
/// p }` — and that is the right trade: a false positive fails the build and
/// someone looks, a false negative writes a prompt to disk and nobody does.
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
        // Walk back past any number of sigils with whitespace anywhere between
        // them. The first version peeked for the sigil BEFORE skipping
        // whitespace, so `%prompt` matched and `% prompt` did not -- a reviewer
        // found it, and the compiler makes no such distinction. Both are one
        // field carrying the same value.
        //
        // A raw identifier is the same field under a spelling the parser
        // tolerates: `r#prompt = p` compiles and writes the value. Checked
        // rather than assumed -- a dotted name in this position does NOT
        // compile (`local ambiguity when calling macro`), so that one is not a
        // form to cover.
        loop {
            while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
                chars.next();
            }
            match chars.peek() {
                Some('#') => {
                    chars.next();
                    if matches!(chars.peek(), Some('r')) {
                        chars.next();
                    }
                }
                Some('%' | '?') => {
                    chars.next();
                }
                _ => break,
            }
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
    // Line comments and block comments both describe rather than declare. Two
    // reviewers asked for the second: `/* impl<S> Layer<S> for X */` would
    // otherwise be read as an impl header, which is the same false positive the
    // line-comment filter already prevents, arriving through the other syntax.
    let without_blocks = strip_block_comments(header);
    without_blocks
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .any(|l| l.contains("impl"))
}

/// Nothing brings an emit macro into scope, which is what makes the anchor sound.
///
/// The scanner above finds call sites by looking for `tracing::` followed by an
/// emit macro. A reviewer put the hole plainly: `use tracing::info;` and then a
/// bare `info!(prompt)` carries the value and matches nothing. No file does that
/// today, so the scanner would simply go quiet the first time one did — the
/// failure this repository keeps rediscovering, where the guard stops speaking
/// and nothing says so.
///
/// Widening the scan to a bare `info!(` was the other option and it is worse: a
/// bare `info!` may belong to any crate, so it would report call sites it cannot
/// judge, and a guardian that cries wolf gets relaxed until it stops speaking.
///
/// **Declared limit: a re-export defeats this too.** The filter looks for the
/// literal `tracing` in the item, so an emit macro re-exported under another
/// crate's path — `use some_wrapper::info;` where `some_wrapper` re-exports
/// `tracing::info` — names nothing this guard recognises. Widening the filter
/// to every crate would report items it cannot judge; the boundary is recorded
/// instead.
///
/// **Declared limit: a Cargo.toml dependency rename defeats this.**
/// `tracing = { package = "tracing", ... }` under another key makes the crate
/// reachable by a name no source file mentions, and there is no textual trace in
/// `src/` for either guard to find. Detecting it means reading the manifest,
/// which is a different check with a different failure mode; it is recorded here
/// as a limit rather than left to be discovered.
///
/// **It reads `use` ITEMS, not lines, because the first version read lines.** A
/// reviewer listed what that missed and every entry was real: a braced group
/// wrapped across lines, `pub use`, `#[macro_use] extern crate tracing`, and a
/// path spaced as `tracing :: info`. The commit that introduced it claimed the
/// anchor was now true, which was stronger than what the assertion enforced —
/// its own kind of defect. An item runs to its `;`, however many lines that is.
///
/// Complexity: `O(bytes under src/)`.
#[test]
fn nothing_brings_an_emit_macro_into_scope_so_the_scanner_sees_every_call_site() {
    const EMIT: [&str; 6] = ["info", "warn", "error", "debug", "trace", "event"];

    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for (path, text) in source_files() {
        scanned += 1;
        // The 2015-edition route, which imports every macro at once without
        // naming any of them.
        // `#[macro_use] extern crate tracing` is the 2015-edition route and
        // imports every macro at once. `extern crate tracing as t` is the other
        // half of the same evasion: it names no macro, but it makes `t::info!`
        // spellable and the anchor looks for `tracing::`.
        let flat_file = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat_file.contains("extern crate tracing")
            && (text.contains("macro_use") || flat_file.contains("extern crate tracing as "))
        {
            offenders.push(format!("{}: extern crate tracing", path.display()));
        }
        for (at, _) in text.match_indices("use ") {
            let Some(rest) = text.get(at..) else { continue };
            let Some((item, _)) = rest.split_once(';') else {
                continue;
            };
            // A `use` written inside a comment has no `;` of its own, so without
            // a bound the split runs to the next semicolon anywhere in the file.
            if item.len() > MAX_USE_ITEM_BYTES || !item.contains("tracing") {
                continue;
            }
            let flat = item.split_whitespace().collect::<Vec<_>>().join(" ");
            // A glob brings every macro in at once and an alias renames the
            // crate, so neither names a macro that `names_item` could find. A
            // reviewer listed both after the item rewrite, which is the second
            // time the enforced set was narrower than the claim.
            // Every spelling that brings the crate in wholesale or under
            // another name. A reviewer listed three more after the first pass
            // closed two, which is the third time the enforced set has come up
            // narrower than the claim -- so this matches the SHAPE (the crate
            // renamed, or a glob over it) rather than a list of literals.
            // `tracing :: info` flattens to one space between tokens, not to
            // none, so the shape checks below run on a form with the path
            // separator's spacing removed as well.
            // The separator's spacing AND the group's: `use tracing::{ * }` and
            // `use tracing::{ self as t }` are the same items with a space inside
            // the braces, which a reviewer found still slipping through.
            let tight = flat
                .replace(" :: ", "::")
                .replace(":: ", "::")
                .replace(" ::", "::")
                .replace("{ ", "{")
                .replace(" }", "}");
            let renamed = tight.contains("tracing as ") || tight.contains("tracing::{self as ");
            let globbed = tight.contains("tracing::*") || tight.contains("tracing::{*");
            let wholesale = renamed || globbed;
            if wholesale || EMIT.iter().any(|m| names_item(&flat, m)) {
                offenders.push(format!("{}: {flat}", path.display()));
            }
        }
    }

    // The same vacuity guard the scanner above carries: without it an empty or
    // unreadable walk passes silently.
    assert!(
        scanned > 50,
        "the source walk found only {scanned} files, so the assertion below proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "an emit macro is in scope, so a bare call to it carries a field the prompt scanner cannot see. Call it as `tracing::info!(...)` instead: {offenders:?}"
    );
}

/// How long a `use` item may be before it is taken for something else.
const MAX_USE_ITEM_BYTES: usize = 400;

/// Whether the whitespace-flattened `use` item *flat* names macro *name*.
///
/// The name must sit where an item sits, so `tracing::info` and
/// `tracing::{info, warn}` match while `tracing::info_span` and a module called
/// `information` do not.
///
/// # Parameters
///
/// * `flat` — the item with runs of whitespace collapsed to one space.
/// * `name` — the macro name.
///
/// # Returns
///
/// `true` if the item brings that macro into scope.
///
/// # Complexity
///
/// `O(n)` in the item's length.
fn names_item(flat: &str, name: &str) -> bool {
    for (at, _) in flat.match_indices(name) {
        let before = flat.get(..at).unwrap_or_default().trim_end();
        let opens = before.ends_with("::") || before.ends_with('{') || before.ends_with(',');
        let after = flat.get(at + name.len()..).unwrap_or_default().trim_start();
        let closes = after.is_empty()
            || after.starts_with(',')
            || after.starts_with('}')
            || after.starts_with("as ");
        if opens && closes {
            return true;
        }
    }
    false
}

/// Every `.rs` file under `src/`, read once.
///
/// Three copies of this walk had accumulated, which a reviewer counted before I
/// did. A read that fails is skipped rather than reported, and that is safe only
/// because every caller pairs it with a count assertion — a walk that silently
/// found nothing would otherwise pass every check built on it.
///
/// # Returns
///
/// Each file's path and contents.
///
/// # Complexity
///
/// `O(bytes under src/)`.
fn source_files() -> Vec<(std::path::PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    out.push((path, text));
                }
            }
        }
    }
    out
}

/// The byte length of the delimited argument list starting at *from_open*.
///
/// **Every literal is skipped, because a `)` inside one is not a delimiter.**
/// Three reviewers found this in two rounds and each time the missing form was
/// narrower than the last: an ordinary string first, then raw strings and
/// character literals. A truncated extraction is a SILENT miss -- the field this
/// scanner exists to catch simply falls outside the call -- so the walk now
/// models all three rather than the common one.
///
/// The limit that used to be declared here -- an opening delimiter written in a
/// comment ahead of the real argument list -- is closed rather than declared as
/// of the commit that strips comments from the whole file before the anchor
/// search. A reviewer caught the stale text; a declared limit that no longer
/// applies is as misleading as an undeclared one.
///
/// # Parameters
///
/// * `from_open` - text beginning at the opening delimiter.
///
/// # Returns
///
/// The length through the matching `)`, or `None` when it is unbalanced.
///
/// # Complexity
///
/// `O(n)` in the argument list's length.
fn argument_list(from_open: &str) -> Option<usize> {
    let chars: Vec<char> = from_open.chars().collect();
    let (open, close) = match chars.first() {
        Some('[') => ('[', ']'),
        Some('{') => ('{', '}'),
        _ => ('(', ')'),
    };
    let mut depth = 0usize;
    let mut i = 0usize;
    // Byte offset kept alongside the character index, because the caller slices
    // by bytes. Conflating the two is a mistake this file has already made once.
    let mut at = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if let Some((skipped, bytes)) = literal_at(&chars, i) {
            i += skipped;
            at += bytes;
            continue;
        }
        // A comment inside the arguments is the fourth member of the same
        // family as the three literals: a `)` written in one is not a
        // delimiter, and reading it as one truncates the call. It was found
        // after the literals were closed, which is why it is closed rather
        // than declared -- the family is the defect, not any one spelling.
        if let Some((skipped, bytes)) = comment_at(&chars, i) {
            i += skipped;
            at += bytes;
            continue;
        }
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(at + c.len_utf8());
            }
        }
        i += 1;
        at += c.len_utf8();
    }
    None
}

/// How many characters and bytes the literal starting at *i* occupies.
///
/// Handles the three forms whose contents must not be read as code: `"..."`
/// with escapes, `r"..."` / `r#"..."#` with a matching hash count, and `'c'`.
///
/// # Parameters
///
/// * `chars` - the text as characters.
/// * `i` - where to look.
///
/// # Returns
///
/// `Some((characters, bytes))` if a literal starts there, else `None`.
///
/// # Complexity
///
/// `O(n)` in the literal's length.
fn literal_at(chars: &[char], i: usize) -> Option<(usize, usize)> {
    let measure = |from: usize, to: usize| -> (usize, usize) {
        let bytes = chars
            .get(from..to)
            .map_or(0, |s| s.iter().map(|c| c.len_utf8()).sum());
        (to - from, bytes)
    };

    // Raw string: `r`, any number of `#`, then the quote. The same count of `#`
    // closes it, which is the whole point of the form -- a quote inside does not
    // end it.
    if chars.get(i) == Some(&'r') {
        let mut hashes = 0usize;
        while chars.get(i + 1 + hashes) == Some(&HASH) {
            hashes += 1;
        }
        if chars.get(i + 1 + hashes) == Some(&QUOTE) {
            let mut j = i + 2 + hashes;
            while j < chars.len() {
                if chars[j] == QUOTE {
                    let closed = (0..hashes).all(|k| chars.get(j + 1 + k) == Some(&HASH));
                    if closed {
                        return Some(measure(i, j + 1 + hashes));
                    }
                }
                j += 1;
            }
            return Some(measure(i, chars.len()));
        }
    }

    // Ordinary string.
    if chars.get(i) == Some(&QUOTE) {
        let mut j = i + 1;
        while j < chars.len() {
            if chars[j] == ESCAPE {
                j += 2;
                continue;
            }
            if chars[j] == QUOTE {
                return Some(measure(i, j + 1));
            }
            j += 1;
        }
        return Some(measure(i, chars.len()));
    }

    // Character literal. A lifetime is written the same way up to the quote --
    // `'a` -- so this accepts only a form that actually closes, which a lifetime
    // never does within the two characters that follow.
    if chars.get(i) == Some(&TICK) {
        let escaped = chars.get(i + 1) == Some(&ESCAPE);
        let close = if escaped { i + 3 } else { i + 2 };
        if chars.get(close) == Some(&TICK) {
            return Some(measure(i, close + 1));
        }
    }

    None
}

/// How many characters and bytes the comment starting at *i* occupies.
///
/// A line comment runs to the newline; a block comment nests, so the depth is
/// tracked rather than stopping at the first `*/`.
///
/// # Parameters
///
/// * `chars` - the text as characters.
/// * `i` - where to look.
///
/// # Returns
///
/// `Some((characters, bytes))` if a comment starts there, else `None`.
///
/// # Complexity
///
/// `O(n)` in the comment's length.
fn comment_at(chars: &[char], i: usize) -> Option<(usize, usize)> {
    let measure = |from: usize, to: usize| -> (usize, usize) {
        let bytes = chars
            .get(from..to)
            .map_or(0, |s| s.iter().map(|c| c.len_utf8()).sum());
        (to - from, bytes)
    };

    if chars.get(i) == Some(&SLASH) && chars.get(i + 1) == Some(&SLASH) {
        let mut j = i + 2;
        while j < chars.len() && chars[j] != NEWLINE {
            j += 1;
        }
        return Some(measure(i, j));
    }

    if chars.get(i) == Some(&SLASH) && chars.get(i + 1) == Some(&STAR) {
        let mut depth = 1usize;
        let mut j = i + 2;
        while j < chars.len() {
            if chars.get(j) == Some(&SLASH) && chars.get(j + 1) == Some(&STAR) {
                depth += 1;
                j += 2;
            } else if chars.get(j) == Some(&STAR) && chars.get(j + 1) == Some(&SLASH) {
                depth -= 1;
                j += 2;
                if depth == 0 {
                    return Some(measure(i, j));
                }
            } else {
                j += 1;
            }
        }
        return Some(measure(i, chars.len()));
    }

    None
}

/// Opens a comment, and closes a block one.
const SLASH: char = '/';
/// Pairs with a slash to delimit a block comment.
const STAR: char = '*';
/// Ends a line comment.
const NEWLINE: char = '\n';

/// Opens and closes a string literal.
const QUOTE: char = '"';
/// Escapes the next character inside a string literal.
const ESCAPE: char = '\\';
/// Delimits a character literal.
const TICK: char = '\'';
/// Pads a raw string's delimiter.
const HASH: char = '#';

/// Whether *args* passes *name* as a field's VALUE.
///
/// `info!(target: "t", key = prompt)` writes the prompt under a field named
/// something else, so the name-position test above sees nothing at all. This
/// covers the pass-through forms: `= name`, `= &name`, the sigils `%` and `?`,
/// the raw identifier `r#name`, and the identity methods that hand the same
/// bytes on under another expression -- `clone`, `to_owned`, `as_str`,
/// `to_string`.
///
/// **What it does NOT cover, stated rather than implied.** This is text
/// matching, not Rust, so a prompt reaching a field through a local rename, a
/// struct field, a function's return value or a format argument is invisible to
/// it. That is an accepted residual and the reason the scanner is one of two
/// guards rather than the guarantee: the auditor masks at the layer, and this
/// catches the emitter forms a person actually writes. A previous docstring
/// justified the exclusions by `prompt.len()` alone, which did not describe what
/// the predicate really left out -- a reviewer said so twice.
///
/// `prompt.len()` stays excluded on purpose: a derived quantity is not the
/// prompt, and a guardian that cries wolf gets relaxed until it stops speaking.
///
/// **Declared, not closed: concatenation.** `key = prompt.to_owned() + x`
/// compiles and writes the prompt inside a longer string, and this predicate
/// does not see it. Closing it means treating `+` as a terminator, which then
/// has to handle the operand on either side and starts matching the shape `a +
/// b` generally — a guardian that cries wolf, for the one spelling of an
/// expression that could be written a dozen other ways. It is recorded beside
/// the `prompt.len()` precedent rather than chased.
///
/// # Parameters
///
/// * `args` - the macro invocation's argument list.
/// * `name` - the local whose value must not be logged.
///
/// # Returns
///
/// `true` if the invocation passes that local as a value.
///
/// # Complexity
///
/// `O(n)` in the argument list's length.
fn is_value_use(args: &str, name: &str) -> bool {
    /// Expressions that hand the same bytes on under another spelling.
    const IDENTITY: [&str; 4] = [".clone()", ".to_owned()", ".as_str()", ".to_string()"];

    for (at, _) in args.match_indices(name) {
        // The method is compared against a tail with ALL its whitespace
        // squeezed out, not merely trimmed at the front. Trimming closed
        // `prompt .clone()` and left `prompt. clone()` and `prompt.clone ()`
        // open -- the family closed one boundary deep, as a reviewer put it.
        // Squeezing closes every spelling of the same call at once, and it is
        // safe because what follows the method only has to be a delimiter.
        let tail = args.get(at + name.len()..).unwrap_or_default();
        let squeezed: String = tail.chars().filter(|c| !c.is_whitespace()).collect();
        // Stripped in a LOOP, because they chain: `prompt.clone().to_owned()`
        // hands on the same bytes twice and one strip left the second in place,
        // where it read as an unrelated expression. `prompt.len()` still stays
        // green -- `.len()` is not in the list, so the loop stops at it.
        let mut rest: &str = &squeezed;
        while let Some(shorter) = IDENTITY.iter().find_map(|m| rest.strip_prefix(m)) {
            rest = shorter;
        }
        // The mirror of the walk-back's wrapper set, and it had the same gap.
        // `]` and `}` close an argument list as much as `)` does; `;` ends the
        // element of a repeat array, `?[prompt; 3]`; and `[` opens a slice,
        // `&prompt[..]` and `?prompt[0..]`. All three of those compile and all
        // three passed green. `prompt[0..]` unsigilled does NOT compile, which
        // is why each spelling was built before it was covered.
        if !(rest.is_empty() || rest.starts_with([',', ')', ']', '}', ';', '['])) {
            continue;
        }
        // Walk back past any number of prefixes, whitespace between them
        // included. `r#prompt` is the same local under a spelling the parser
        // tolerates, and `(&prompt)` is the same value behind one more layer.
        //
        // `[` and `{` are wrappers for the same reason `(` is. The one that
        // matters is `key = ?[prompt]`: the sigil supplies Debug, which an
        // array does implement, so it compiles and it writes the prompt. The
        // unsigilled `key = [prompt]` does NOT compile -- `[&str; 1]` is not a
        // `tracing::Value` -- which is why the pair had to be checked and not
        // assumed. None of the three makes a false positive on `f(prompt)` or
        // `v[prompt]`, because what precedes that bracket is a name, not an `=`.
        let mut before = args.get(..at).unwrap_or_default().trim_end();
        loop {
            let stripped = before
                .strip_suffix("r#")
                .or_else(|| before.strip_suffix(['%', '?', '&', '(', '[', '{']));
            match stripped {
                Some(head) => before = head.trim_end(),
                None => break,
            }
        }
        if before.ends_with('=') && !before.ends_with("==") {
            return true;
        }
    }
    false
}

/// *text* with `/* ... */` comments removed, nesting included.
///
/// Rust's block comments nest, so a naive split on the first `*/` reopens the
/// code at a point that is still inside a comment. A reviewer raised it as a
/// remark rather than a defect and it is one line of state either way, so it is
/// modelled rather than noted.
///
/// # Parameters
///
/// * `text` — the window to clean.
///
/// # Returns
///
/// The text outside every block comment.
///
/// # Complexity
///
/// `O(n)` in the text's length.
fn strip_block_comments(text: &str) -> String {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let two = (bytes.get(i), bytes.get(i + 1));
        if two == (Some(&'/'), Some(&'*')) {
            depth += 1;
            i += 2;
        } else if two == (Some(&'*'), Some(&'/')) && depth > 0 {
            depth -= 1;
            i += 2;
        } else {
            if depth == 0 {
                if let Some(c) = bytes.get(i) {
                    out.push(*c);
                }
            }
            i += 1;
        }
    }
    out
}
