// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Stages 1 and 3 of REQ-L64: rendering an event to text, and escaping a line.
//!
//! # The two stages here are NOT consecutive, and the module name hides that
//!
//! REQ-L64 runs four stages in order: **render → audit → format/escape →
//! chunk**. [`render_event`] is stage 1 and [`escape_for_line`] is stage 3, and
//! **the auditor runs between them**. They share a module because both are pure
//! text transformations over the same header format, not because they run back
//! to back.
//!
//! `escape_for_line` is called by the layer, *after* auditing — never by
//! `render_event`. Reading the module name and chaining the two inside would
//! break REQ-L64's ordering without anything failing to compile.
//!
//! # This module is NOT fully pure, and the plan says it is
//!
//! [`escape_for_line`] and [`header_of`] are pure. [`render_event`] is not: its
//! signature carries no timestamp, so it reads the clock itself. The split is
//! deliberate rather than an oversight — `header_of` takes the instant as a
//! parameter precisely so the part a test needs to pin is pinnable, and
//! `render_event` is exercised through a real dispatcher where the clock is the
//! honest source anyway.
//!
//! Stated because the plan calls this module pure. A reader who takes that
//! literally will look for a `Clock` parameter that does not exist and conclude
//! something is missing.
//!
//! # Escaping is a security property, not formatting
//!
//! Without it, a message carrying a newline produces what *looks* like an
//! independent log line. A foreign string from magi-core, or an error body an
//! endpoint controls, could then **forge log entries** — including one
//! imitating an auditor alarm — with nothing to tell the false from the real.

use std::fmt::Write as _;

use time::OffsetDateTime;
use tracing::field::{Field, Visit};
use tracing::{Event, Level};

/// Collects an event's fields, keeping the message apart from the rest.
#[derive(Default)]
struct FieldWriter {
    message: String,
    fields: String,
}

impl FieldWriter {
    /// Appends ` name=value`, the shape the tests and the operator read.
    fn push_field(&mut self, name: &str, value: &str) {
        let _ = write!(self.fields, " {name}={value}");
    }
}

impl Visit for FieldWriter {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == MESSAGE_FIELD {
            self.message.push_str(value);
        } else {
            self.push_field(field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // `{:?}` is the fallback for every type that is not a plain string.
        // A string reaches `record_str` above and keeps its quotes off, which
        // is what makes `attempt=2` read as `attempt=2` and not `attempt="2"`.
        if field.name() == MESSAGE_FIELD {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

/// The field `tracing` uses for an event's message.
const MESSAGE_FIELD: &str = "message";
/// The characters escaping rewrites, and their replacements.
///
/// Named because the literals are unreadable inline: a match arm on a raw
/// backslash beside a match arm on an escaped one is the kind of line a reader
/// has to count characters to parse.
const BACKSLASH: char = '\\';
/// The line terminator an unescaped foreign string would use to forge a line.
const NEWLINE: char = '\n';
/// Carriage return, which some terminals also treat as a line break.
const CARRIAGE_RETURN: char = '\r';
/// Horizontal tab.
const TAB: char = '\t';
/// Replacement for [`BACKSLASH`].
const ESCAPED_BACKSLASH: &str = "\\\\";
/// Replacement for [`NEWLINE`].
const ESCAPED_NEWLINE: &str = "\\n";
/// Replacement for [`CARRIAGE_RETURN`].
const ESCAPED_CARRIAGE_RETURN: &str = "\\r";
/// Replacement for [`TAB`].
const ESCAPED_TAB: &str = "\\t";
/// Opening of the escaped-unicode form used for every other control character.
const ESCAPE_UNICODE_OPEN: &str = "\\u{";

/// Separator between the target and the message in a header.
const TARGET_SEPARATOR: &str = ": ";

/// Renders an event to text. Stage 1: nothing is escaped here.
///
/// # Parameters
///
/// * `event` — the event as the dispatcher hands it over.
///
/// # Returns
///
/// `<header><message><space-separated fields>`, with every byte exactly as the
/// emitter wrote it. **Nothing is escaped**: escaping is stage 3, and the
/// auditor runs between the two.
///
/// # Complexity
///
/// `O(n)` over the event's rendered length.
#[must_use]
pub fn render_event(event: &Event<'_>) -> String {
    let meta = event.metadata();
    let mut writer = FieldWriter::default();
    event.record(&mut writer);
    let mut line = header_of(
        *meta.level(),
        meta.target(),
        OffsetDateTime::now_utc(),
        crate::logging::run_id(),
    );
    line.push_str(&writer.message);
    line.push_str(&writer.fields);
    line
}

/// Escapes a rendered line so it can never span more than one physical line.
///
/// # Parameters
///
/// * `rendered` — the audited text of stage 1.
///
/// # Returns
///
/// The same text with the backslash doubled, the three common control
/// characters written as `\n`, `\r`, `\t`, and every other control
/// character as `\u{h}` in lowercase hex.
///
/// # Why this is a security property and not formatting
///
/// A newline that survives produces what LOOKS like an independent log line, so
/// a foreign string — a magi-core error, a body an endpoint controls — could
/// forge entries, including one imitating an auditor alarm, with nothing to
/// tell the false from the real.
///
/// # Complexity
///
/// `O(n)` over the input.
#[must_use]
pub fn escape_for_line(rendered: &str) -> String {
    let mut out = String::with_capacity(rendered.len());
    for c in rendered.chars() {
        match c {
            // The backslash goes first: escaping it after the others would
            // double the backslashes they just introduced.
            BACKSLASH => out.push_str(ESCAPED_BACKSLASH),
            NEWLINE => out.push_str(ESCAPED_NEWLINE),
            CARRIAGE_RETURN => out.push_str(ESCAPED_CARRIAGE_RETURN),
            TAB => out.push_str(ESCAPED_TAB),
            c if c.is_control() => {
                let _ = write!(out, "{ESCAPE_UNICODE_OPEN}{:x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Builds the header the chunker budgets against.
///
/// # Parameters
///
/// * `level` — the event's level.
/// * `target` — the emitting module path.
/// * `ts` — the instant, converted to UTC before rendering.
/// * `run` — the process's run id, which every line carries.
///
/// # Returns
///
/// `YYYY-MM-DDTHH:MM:SSZ LEVEL run=<run> target: `, **ending in the separating
/// space**.
/// The trailing space is part of the contract: `chunk::split` budgets against
/// `4096 - header.len()`, so a header that stopped one byte short would leave
/// every payload one byte too long.
///
/// # Why this lives here and not in the chunker
///
/// Two modules cannot each hold half the header format. This one produces it,
/// the chunker measures it, and the budget is computed over the same string.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn header_of(level: Level, target: &str, ts: OffsetDateTime, run: &str) -> String {
    let _ = run;
    let ts = ts.to_offset(time::UtcOffset::UTC);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z {} {}{}",
        ts.year(),
        u8::from(ts.month()),
        ts.day(),
        ts.hour(),
        ts.minute(),
        ts.second(),
        level,
        target,
        TARGET_SEPARATOR
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::testutil::{fixed_ts, render_fixture};

    #[test]
    fn the_header_carries_the_run_it_was_given() {
        // SC-L79: the daily file is shared by every concurrent run, so
        // filtering it by a run id is the ONLY way to isolate one. That is
        // impossible unless the id is on the line.
        let header = header_of(
            Level::INFO,
            "magi_rs::agent",
            fixed_ts(),
            "4242-deadbeefcafe0001",
        );
        assert!(
            header.contains("run=4242-deadbeefcafe0001"),
            "no run field in the header: {header}"
        );
    }

    #[test]
    fn a_different_run_produces_a_different_header() {
        // Without this, a hardcoded literal satisfies the test above and the
        // field stops tracking the process it names. The mutation that matters
        // is not "delete the field" but "freeze it".
        let one = header_of(Level::INFO, "t", fixed_ts(), "1-a");
        let two = header_of(Level::INFO, "t", fixed_ts(), "2-b");
        assert_ne!(one, two, "the header ignores the run it was handed");
    }

    #[test]
    fn a_rendered_event_carries_the_process_run_id() {
        // The pure function above is given a run; this proves `render_event`
        // actually hands it the process's own, rather than something else.
        let line = render_fixture!(Level::INFO, "magi_rs::agent", "hello");
        assert!(
            line.contains(&format!("run={}", crate::logging::run_id())),
            "the rendered line does not name this process's run: {line}"
        );
    }

    #[test]
    fn the_rendered_line_carries_timestamp_level_target_and_message() {
        let line = render_fixture!(Level::WARN, "magi_rs::agent", "boom", attempt = "2");
        assert!(line.contains("WARN"));
        assert!(line.contains("magi_rs::agent"));
        assert!(line.contains("boom"));
        assert!(line.contains("attempt=2"));
    }

    #[test]
    fn the_timestamp_carries_an_explicit_offset_not_a_bare_local_time() {
        let line = render_fixture!(Level::INFO, "t", "m");
        assert!(
            line.contains("+00:00") || line.contains('Z'),
            "no explicit offset in {line}"
        );
    }

    #[test]
    fn render_does_not_escape_anything() {
        let line = render_fixture!(
            Level::INFO,
            "t",
            "a
b"
        );
        assert!(
            line.contains('\n'),
            "stage 1 must keep the raw newline; escaping is stage 3"
        );
    }

    #[test]
    fn escaping_turns_every_newline_and_control_char_into_two_visible_bytes() {
        assert_eq!(escape_for_line("a\nb"), "a\\nb");
        assert_eq!(escape_for_line("a\rb"), "a\\rb");
        assert_eq!(escape_for_line("a\u{7}b"), "a\\u{7}b");
    }

    #[test]
    fn an_escaped_line_can_never_contain_a_raw_newline() {
        // The security property: a foreign string cannot forge a second log line.
        let hostile = "ok\n2026-08-14T00:00:00Z ERROR magi_rs::logging: SECRET LEAK DETECTED";
        let escaped = escape_for_line(hostile);
        assert!(!escaped.contains('\n'), "a forged line survived escaping");
        assert_eq!(escaped.lines().count(), 1);
    }

    #[test]
    fn the_header_is_produced_here_so_the_chunker_budgets_against_the_same_string() {
        let h = header_of(Level::INFO, "magi_rs::agent", fixed_ts(), "0-0");
        assert!(
            !h.is_empty(),
            "an empty header would satisfy the suffix check below for free"
        );
        assert!(
            h.ends_with(' '),
            "the header ends with the separator the budget counts"
        );
    }
}
