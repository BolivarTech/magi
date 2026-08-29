// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Splitting an oversized event into bounded, UTF-8-safe lines.

use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::OsRng;
use rand::RngCore;

/// Maximum bytes a single written line may occupy, terminator included.
pub const MAX_LINE_BYTES: usize = 4096;

/// Bytes the line terminator occupies, and the reason it is named.
///
/// REQ-L06 counts the terminator inside the threshold, because the threshold
/// bounds the `write` call and that call carries it. Written as a bare
/// `+ 1 <= MAX`, clippy offers to collapse `len() + 1 <= MAX` into `len() < MAX` —
/// arithmetically identical, and it deletes the requirement from the code: the
/// comparison stops saying anything about a terminator and becomes an arbitrary
/// strict inequality. Naming the byte keeps the reason visible and the lint
/// quiet, in that order of importance.
pub const NEWLINE_BYTES: usize = 1;

const ID_PREFIX_BYTES: usize = 3;
const SPACE_AFTER_ID_BYTES: usize = 1;
const SLASH_BYTES: usize = 1;
const SPACE_BYTES: usize = 1;
const MIN_CHUNK_COUNT: usize = 1;
const ONE_BASED_OFFSET: usize = 1;
const MAX_SIZING_ITERATIONS: usize = 4;
const COUNTER_INITIAL: u64 = 0;

static FALLBACK_COUNTER: AtomicU64 = AtomicU64::new(COUNTER_INITIAL);

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
        let pid = process::id();
        let mut bytes = [0u8; 8];
        let mut rng = OsRng;
        let rand = match rng.try_fill_bytes(&mut bytes) {
            Ok(()) => u64::from_ne_bytes(bytes),
            Err(_) => {
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let count = FALLBACK_COUNTER.fetch_add(1, Ordering::SeqCst);
                (pid as u64).wrapping_mul(seed).wrapping_add(count)
            }
        };
        Self { pid, rand }
    }

    /// Renders the identifier as `<pid>-<16 hex digits>`.
    #[must_use]
    pub fn render(&self) -> String {
        format!("{}-{:016x}", self.pid, self.rand)
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Number of decimal digits needed to write `n`.
///
/// Time complexity: O(log n), space complexity: O(1).
fn decimal_digits(mut n: usize) -> usize {
    if n == 0 {
        return MIN_CHUNK_COUNT;
    }
    let mut digits = 0;
    while n > 0 {
        digits += 1;
        n /= 10;
    }
    digits
}

/// Byte width of the worst-case marker `n/N ` when `N` has `digit_count` digits.
///
/// Time complexity: O(1), space complexity: O(1).
const fn marker_width(digit_count: usize) -> usize {
    digit_count + SLASH_BYTES + digit_count + SPACE_BYTES
}

/// Line payload budget for a given header/marker overhead.
///
/// Time complexity: O(1), space complexity: O(1).
fn line_budget(overhead: usize) -> usize {
    MAX_LINE_BYTES
        .saturating_sub(NEWLINE_BYTES)
        .saturating_sub(overhead)
}

/// Ceiling division without floating point.
///
/// Time complexity: O(1), space complexity: O(1).
fn ceil_div(a: usize, b: usize) -> usize {
    let decrement = b.saturating_sub(1);
    a.saturating_add(decrement) / b
}

/// Estimates the chunk count from the event length and the two budgets.
///
/// Time complexity: O(1), space complexity: O(1).
fn estimate_chunk_count(event_len: usize, first_budget: usize, cont_budget: usize) -> usize {
    if event_len <= first_budget {
        return MIN_CHUNK_COUNT;
    }
    let tail = event_len - first_budget;
    if cont_budget == 0 {
        return MIN_CHUNK_COUNT + tail;
    }
    MIN_CHUNK_COUNT + ceil_div(tail, cont_budget)
}

/// Cuts `event` into payloads that respect UTF-8 boundaries and the per-line budgets.
///
/// Time complexity: O(|event|), space complexity: O(N) for the returned references.
fn cut_payloads(event: &str, first_budget: usize, cont_budget: usize) -> Vec<&str> {
    if event.is_empty() {
        return vec![event];
    }

    let mut payloads = Vec::new();
    let mut pos = 0;
    let mut budget = first_budget;

    while pos < event.len() {
        let raw_end = pos.saturating_add(budget).min(event.len());
        let mut end = raw_end;
        while end > pos && !event.is_char_boundary(end) {
            end = end.saturating_sub(1);
        }
        if end == pos {
            let advance = event
                .get(pos..)
                .unwrap_or("")
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            if advance == 0 {
                break;
            }
            end = pos.saturating_add(advance);
        }
        payloads.push(event.get(pos..end).unwrap_or(""));
        pos = end;
        budget = cont_budget;
    }

    payloads
}

/// Builds the indented header the continuation chunks carry.
#[must_use]
pub fn cont_header_for(id: &EventId) -> String {
    format!("  id={} ", id.render())
}

/// Splits an already-escaped event into lines that fit the threshold.
#[must_use]
pub fn split(event: &str, first_header: &str, cont_header: &str, id: EventId) -> Vec<String> {
    let rendered_id = id.render();
    let rendered_id_len = rendered_id.len();

    let mut digit_count = MIN_CHUNK_COUNT;
    for _ in 0..MAX_SIZING_ITERATIONS {
        let mw = marker_width(digit_count);
        let overhead_first =
            first_header.len() + ID_PREFIX_BYTES + rendered_id_len + SPACE_AFTER_ID_BYTES + mw;
        let overhead_cont = cont_header.len() + mw;
        let first_budget = line_budget(overhead_first);
        let cont_budget = line_budget(overhead_cont);
        let estimated_n = estimate_chunk_count(event.len(), first_budget, cont_budget);
        let new_digit_count = decimal_digits(estimated_n);
        if new_digit_count == digit_count {
            break;
        }
        digit_count = new_digit_count;
    }

    let mut mw = marker_width(digit_count);
    let mut overhead_first =
        first_header.len() + ID_PREFIX_BYTES + rendered_id_len + SPACE_AFTER_ID_BYTES + mw;
    let mut overhead_cont = cont_header.len() + mw;
    let mut first_budget = line_budget(overhead_first);
    let mut cont_budget = line_budget(overhead_cont);
    let mut payloads = cut_payloads(event, first_budget, cont_budget);
    let mut n = payloads.len();

    let real_digit_count = decimal_digits(n);
    if real_digit_count > digit_count {
        digit_count = real_digit_count;
        mw = marker_width(digit_count);
        overhead_first =
            first_header.len() + ID_PREFIX_BYTES + rendered_id_len + SPACE_AFTER_ID_BYTES + mw;
        overhead_cont = cont_header.len() + mw;
        first_budget = line_budget(overhead_first);
        cont_budget = line_budget(overhead_cont);
        payloads = cut_payloads(event, first_budget, cont_budget);
        n = payloads.len();
    }

    let mut lines = Vec::with_capacity(n);
    if n == MIN_CHUNK_COUNT {
        let payload = payloads.first().copied().unwrap_or("");
        lines.push(format!("{}{}", first_header, payload));
        return lines;
    }

    for (i, payload) in payloads.iter().enumerate() {
        let line_no = i.saturating_add(ONE_BASED_OFFSET);
        if i == 0 {
            lines.push(format!(
                "{}id={} {}/{} {}",
                first_header, rendered_id, line_no, n, payload
            ));
        } else {
            lines.push(format!("{}{}/{} {}", cont_header, line_no, n, payload));
        }
    }

    lines
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
