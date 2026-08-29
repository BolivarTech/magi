// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Splitting an oversized event into bounded, UTF-8-safe lines.

/// Maximum bytes a single written line may occupy, terminator included.
pub const MAX_LINE_BYTES: usize = 4096;

/// Bytes the line terminator occupies, and the reason it is named.
///
/// REQ-L06 counts the terminator inside the threshold, because the threshold
/// bounds the `write` call and that call carries it. Written as a bare
/// `+ 1`, clippy offers to collapse `len() + 1 <= MAX` into `len() < MAX` —
/// arithmetically identical, and it deletes the requirement from the code: the
/// comparison stops saying anything about a terminator and becomes an arbitrary
/// strict inequality. Naming the byte keeps the reason visible and the lint
/// quiet, in that order of importance.
pub const NEWLINE_BYTES: usize = 1;

/// Identifier correlating the chunks of one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventId {
    pid: u32,
    rand: u64,
}

impl EventId {
    /// Mints an identifier for one event.
    #[must_use]
    pub fn new() -> Self {
        Self { pid: 0, rand: 0 }
    }

    /// Renders the identifier as `<pid>-<16 hex digits>`.
    #[must_use]
    pub fn render(&self) -> String {
        String::new() // Red-phase stub
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the indented header the continuation chunks carry.
#[must_use]
pub fn cont_header_for(_id: &EventId) -> String {
    String::new() // Red-phase stub
}

/// Splits an already-escaped event into lines that fit the threshold.
#[must_use]
pub fn split(_event: &str, _first_header: &str, _cont_header: &str, _id: EventId) -> Vec<String> {
    Vec::new() // Red-phase stub
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload of a produced line, with its header and marker stripped.
    ///
    /// A chunked line is `<header>id=<pid>-<hex16> n/N <payload>`, so the
    /// payload is what follows the marker's two tokens. An unchunked line
    /// carries no marker at all (REQ-L11), and there the header ends at the
    /// first space.
    fn payload_of(line: &str) -> &str {
        match line.find("id=") {
            Some(i) => line[i..].splitn(3, ' ').nth(2).unwrap_or(""),
            None => line.split_once(' ').map(|(_, p)| p).unwrap_or(line),
        }
    }

    #[test]
    fn every_written_line_stays_within_the_threshold_including_the_newline() {
        let event = "x".repeat(50 * 1024);
        let id = EventId::new();
        // A LITERAL header, not `header_of` from task 1.4: that function does
        // not exist yet, and this test is about the threshold, not about where
        // the header comes from.
        let header = "2026-08-14T00:00:00Z INFO magi_rs::agent: ";
        let lines = split(&event, header, &cont_header_for(&id), id);
        // Without this the assertion below is vacuously true on an empty
        // result: a 50 KiB event MUST chunk, so an unchunked answer is the
        // failure, not a pass.
        assert!(
            lines.len() > 1,
            "50 KiB must be chunked, got {}",
            lines.len()
        );
        for line in &lines {
            assert!(
                line.len() + NEWLINE_BYTES <= MAX_LINE_BYTES,
                "line of {} bytes",
                line.len()
            );
        }
    }

    #[test]
    fn a_short_event_carries_no_chunk_marker() {
        let id = EventId::new();
        let lines = split("short", "H ", &cont_header_for(&id), id);
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].contains("id="));
        assert!(!lines[0].contains("1/1"));
    }

    #[test]
    fn cuts_land_on_character_boundaries_and_every_chunk_is_valid_utf8() {
        let event = "日".repeat(4000); // 3 bytes each
        let id = EventId::new();
        let lines = split(&event, "H ", &cont_header_for(&id), id);
        let joined: String = lines.iter().map(|l| payload_of(l)).collect();
        assert_eq!(joined, event);
    }

    #[test]
    fn the_marker_count_matches_the_number_of_lines_actually_produced() {
        let event = "日".repeat(20_000);
        let id = EventId::new();
        let lines = split(&event, "H ", &cont_header_for(&id), id);
        let n = lines.len();
        assert!(n > 1, "a 60 KB event must produce several chunks, got {n}");
        for (i, line) in lines.iter().enumerate() {
            assert!(
                line.contains(&format!("{}/{}", i + 1, n)),
                "line {} claims a wrong N",
                i + 1
            );
        }
    }

    #[test]
    fn budgets_are_sized_for_the_widest_marker_not_the_first() {
        // 512 chunks: the marker grows from "1/512" to "512/512".
        let event = "x".repeat(512 * 4000);
        let id = EventId::new();
        let lines = split(&event, "H ", &cont_header_for(&id), id);
        // The point of this test is a marker that WIDENS from `1/N` to a
        // three-digit numerator; over an empty or short result there is no
        // widening to observe and every assertion below holds for free.
        assert!(
            lines.len() > 100,
            "the marker must reach three digits for this test to mean anything, got {}",
            lines.len()
        );
        for line in &lines {
            assert!(line.len() + NEWLINE_BYTES <= MAX_LINE_BYTES);
        }
    }

    #[test]
    fn the_two_budgets_differ_because_the_continuation_header_is_shorter() {
        // REQ-L08: chunk 1 carries the full header, 2..N only the indented
        // marker. A single budget would size both against the first chunk and
        // waste the rest.
        let long_header = "2026-08-14T00:00:00Z INFO magi_rs::agent::very::long::target: ";
        let id = EventId::new();
        let event = "x".repeat(40_000);
        let lines = split(&event, long_header, &cont_header_for(&id), id);
        let first_payload = payload_of(&lines[0]).len();
        let cont_payload = payload_of(&lines[1]).len();
        assert!(
            cont_payload > first_payload,
            "the continuation header is shorter, so its payload budget is larger: \
             {cont_payload} vs {first_payload}"
        );
        for line in &lines {
            assert!(line.len() + NEWLINE_BYTES <= MAX_LINE_BYTES);
        }
    }

    #[test]
    fn the_event_id_is_sixty_four_bits_rendered_as_pid_dash_sixteen_hex() {
        let id = EventId::new().render();
        let (pid, hex) = id.split_once('-').expect("pid-hex form");
        assert!(pid.parse::<u32>().is_ok());
        assert_eq!(hex.len(), 16);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
