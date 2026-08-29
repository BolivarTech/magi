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

/// Byte length of the literal `id=` that precedes the identifier on chunk one.
const ID_PREFIX_BYTES: usize = 3;
/// The single space between the identifier and the `n/N` marker.
const SPACE_AFTER_ID_BYTES: usize = 1;
/// The `/` separating numerator from denominator in the marker.
const SLASH_BYTES: usize = 1;
/// The space between the marker and the payload.
const SPACE_BYTES: usize = 1;
/// Rounds allowed for the budget/marker-width fixed point.
///
/// Four is generous: the width only grows with the digit count of `N`, so the
/// loop settles in one or two rounds for any event a process can hold in
/// memory. The bound exists so a pathological header cannot spin forever.
const MAX_SIZING_ITERATIONS: usize = 4;

/// Counter behind the identifier's degraded path when the OS RNG refuses.
static FALLBACK_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    // Zero is written with one digit. The first version returned a constant
    // named for a chunk count here, which happened to hold the right number
    // and said the wrong thing.
    if n == 0 {
        return 1;
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

/// Ceiling division without floating point, total in `b`.
///
/// A zero divisor answers `a` rather than dividing: the caller's guard already
/// makes that unreachable today, but a helper that panics on an input its own
/// signature accepts is a landmine in a module whose whole point is that it
/// must not panic while everything else is failing.
///
/// Time complexity: O(1), space complexity: O(1).
fn ceil_div(a: usize, b: usize) -> usize {
    if b == 0 {
        return a;
    }
    a.saturating_add(b - 1) / b
}

/// Estimates the chunk count from the event length and the two budgets.
///
/// Time complexity: O(1), space complexity: O(1).
fn estimate_chunk_count(event_len: usize, first_budget: usize, cont_budget: usize) -> usize {
    if event_len <= first_budget {
        return 1;
    }
    let tail = event_len - first_budget;
    if cont_budget == 0 {
        // One line per byte is the worst the caller can force with a header
        // that leaves no room; it is an estimate, and the real cut decides.
        return 1 + tail;
    }
    1 + ceil_div(tail, cont_budget)
}

/// The two payload budgets for a marker of `digit_count` digits.
///
/// Extracted because the expression appeared **three times** in `split` — in
/// the sizing loop, after it, and again in the re-cut — and three copies of an
/// arithmetic definition is three chances for one of them to drift.
///
/// Returns `(first_budget, cont_budget)`.
///
/// Time complexity: O(1), space complexity: O(1).
fn budgets(
    first_header: &str,
    cont_header: &str,
    rendered_id_len: usize,
    digit_count: usize,
) -> (usize, usize) {
    let mw = marker_width(digit_count);
    let first = first_header.len() + ID_PREFIX_BYTES + rendered_id_len + SPACE_AFTER_ID_BYTES + mw;
    (line_budget(first), line_budget(cont_header.len() + mw))
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
pub fn cont_header_for(id: &EventId, run: &str) -> String {
    format!(
        "  {}{} id={} ",
        crate::logging::render::RUN_FIELD,
        run,
        id.render()
    )
}

/// Splits an already-escaped event into lines that fit the threshold.
#[must_use]
pub fn split(event: &str, first_header: &str, cont_header: &str, id: EventId) -> Vec<String> {
    let rendered_id = id.render();
    let id_len = rendered_id.len();

    // The marker widens with the digit count of `N`, and `N` depends on the
    // budget the marker leaves — so the two are solved by iterating to a fixed
    // point rather than in one pass.
    let mut digits = 1;
    for _ in 0..MAX_SIZING_ITERATIONS {
        let (first, cont) = budgets(first_header, cont_header, id_len, digits);
        let next = decimal_digits(estimate_chunk_count(event.len(), first, cont));
        if next == digits {
            break;
        }
        digits = next;
    }

    // `N` comes from the REAL cut, never from the estimate: retracting each cut
    // to a character boundary gives back up to three bytes per chunk, the
    // deficit accumulates, and the tail lands past the predicted last line.
    let (mut first, mut cont) = budgets(first_header, cont_header, id_len, digits);
    let mut payloads = cut_payloads(event, first, cont);

    // If the real count needs a wider marker than the budgets allowed for, cut
    // once more with the corrected width. Emitting a marker wider than its own
    // budget is exactly the overflow this task exists to prevent.
    if decimal_digits(payloads.len()) > digits {
        digits = decimal_digits(payloads.len());
        (first, cont) = budgets(first_header, cont_header, id_len, digits);
        payloads = cut_payloads(event, first, cont);
    }

    let n = payloads.len();
    if n == 1 {
        // A line with no marker IS a complete line (REQ-L11).
        return vec![format!(
            "{}{}",
            first_header,
            payloads.first().copied().unwrap_or("")
        )];
    }

    payloads
        .iter()
        .enumerate()
        .map(|(i, payload)| {
            if i == 0 {
                format!("{first_header}id={rendered_id} 1/{n} {payload}")
            } else {
                format!("{cont_header}{}/{n} {payload}", i + 1)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::testutil::payload_of;

    /// REQ-L06's threshold, pinned here as a LITERAL rather than read from
    /// [`MAX_LINE_BYTES`].
    ///
    /// Asserting against the constant **cannot fail**: it is the same value the
    /// implementation sizes its budgets with, so raising it moves the produced
    /// lines and the allowed ceiling together and the guarantee evaporates in
    /// silence. The plan's own mutation for this task — "raise the budget above
    /// 4096, the threshold test must go red" — was run and stayed green for
    /// exactly that reason.
    ///
    /// The requirement is 4096 bytes. A test that guards it has to say 4096.
    const REQUIRED_MAX_LINE_BYTES: usize = 4096;

    #[test]
    fn the_constant_still_matches_the_requirement_it_is_supposed_to_encode() {
        // If someone changes MAX_LINE_BYTES deliberately, this is the test that
        // says so — instead of every other assertion quietly following it.
        assert_eq!(MAX_LINE_BYTES, REQUIRED_MAX_LINE_BYTES);
    }

    #[test]
    fn a_continuation_header_carries_the_run() {
        // Chunks 2..N are lines too, and SC-L79 says filtering by a run
        // returns ONLY its lines -- which means ALL of them. A continuation
        // line without the run is a line that filter drops.
        let id = EventId::new();
        let cont = cont_header_for(&id, "4242-deadbeefcafe0001");
        assert!(
            cont.contains("run=4242-deadbeefcafe0001"),
            "no run field in the continuation header: {cont}"
        );
        assert_ne!(
            cont_header_for(&id, "1-a"),
            cont_header_for(&id, "2-b"),
            "the continuation header ignores the run it was handed"
        );
    }

    #[test]
    fn every_written_line_stays_within_the_threshold_including_the_newline() {
        let event = "x".repeat(50 * 1024);
        let id = EventId::new();
        // A LITERAL header, not `header_of` from task 1.4: that function does
        // not exist yet, and this test is about the threshold, not about where
        // the header comes from.
        let header = "2026-08-14T00:00:00Z INFO magi_rs::agent: ";
        let lines = split(&event, header, &cont_header_for(&id, "0-0"), id);
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
                line.len() + NEWLINE_BYTES <= REQUIRED_MAX_LINE_BYTES,
                "line of {} bytes",
                line.len()
            );
        }
    }

    #[test]
    fn a_short_event_carries_no_chunk_marker() {
        let id = EventId::new();
        let lines = split("short", "H ", &cont_header_for(&id, "0-0"), id);
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].contains("id="));
        assert!(!lines[0].contains("1/1"));
    }

    #[test]
    fn cuts_land_on_character_boundaries_and_every_chunk_is_valid_utf8() {
        let event = "日".repeat(4000); // 3 bytes each
        let id = EventId::new();
        let lines = split(&event, "H ", &cont_header_for(&id, "0-0"), id);
        let joined: String = lines.iter().map(|l| payload_of(l)).collect();
        assert_eq!(joined, event);
    }

    #[test]
    fn the_marker_count_matches_the_number_of_lines_actually_produced() {
        let event = "日".repeat(20_000);
        let id = EventId::new();
        let lines = split(&event, "H ", &cont_header_for(&id, "0-0"), id);
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
        let lines = split(&event, "H ", &cont_header_for(&id, "0-0"), id);
        // The point of this test is a marker that WIDENS from `1/N` to a
        // three-digit numerator; over an empty or short result there is no
        // widening to observe and every assertion below holds for free.
        assert!(
            lines.len() > 100,
            "the marker must reach three digits for this test to mean anything, got {}",
            lines.len()
        );
        for line in &lines {
            assert!(line.len() + NEWLINE_BYTES <= REQUIRED_MAX_LINE_BYTES);
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
        let lines = split(&event, long_header, &cont_header_for(&id, "0-0"), id);
        let first_payload = payload_of(&lines[0]).len();
        let cont_payload = payload_of(&lines[1]).len();
        assert!(
            cont_payload > first_payload,
            "the continuation header is shorter, so its payload budget is larger: \
             {cont_payload} vs {first_payload}"
        );
        for line in &lines {
            assert!(line.len() + NEWLINE_BYTES <= REQUIRED_MAX_LINE_BYTES);
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
