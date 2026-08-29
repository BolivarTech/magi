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
//! # Escaping is a security property, not formatting
//!
//! Without it, a message carrying a newline produces what *looks* like an
//! independent log line. A foreign string from magi-core, or an error body an
//! endpoint controls, could then **forge log entries** — including one
//! imitating an auditor alarm — with nothing to tell the false from the real.

use time::OffsetDateTime;
use tracing::{Event, Level};

/// Renders an event to text. Stage 1: nothing is escaped here.
#[must_use]
pub fn render_event(_event: &Event<'_>) -> String {
    String::new() // Red-phase stub
}

/// Escapes a rendered line so it can never span more than one physical line.
#[must_use]
pub fn escape_for_line(_rendered: &str) -> String {
    String::new() // Red-phase stub
}

/// Builds the header the chunker budgets against.
#[must_use]
pub fn header_of(_level: Level, _target: &str, _ts: OffsetDateTime) -> String {
    String::new() // Red-phase stub
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::testutil::{fixed_ts, render_fixture};

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
        let h = header_of(Level::INFO, "magi_rs::agent", fixed_ts());
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
