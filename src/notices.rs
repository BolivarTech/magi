// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! Startup notices and the level each one is announced at.
//!
//! # Why it lives in the LIB and not under `system/`/`tui/`
//!
//! It is pure: no I/O, no network, no state. `main.rs` consumes it to assemble the list of
//! notices a startup announces, then hands the list to [`emit_notices`].
//!
//! # One axis, not two (D-L11)
//!
//! The tier this module used to carry encoded **severity** and **visibility** at once, because
//! a cap on how many notices survived meant something had to say "do not trim this one". With
//! the file as a destination there is no cap, and the only question left — screen or file — is
//! severity. So a notice carries a `tracing::Level`, and the layer decides the mouth: `ERROR`
//! and `WARN` reach the screen, `INFO` goes only to the file (REQ-L19).

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::collections::HashSet;

/// Target every startup notice is emitted under.
///
/// Fixed rather than per-source so an operator can raise or lower the whole startup
/// announcement with one `[logging].file_filter` directive.
pub const NOTICE_TARGET: &str = "magi_rs::startup";

/// A startup notice, with the level that decides which mouth it reaches.
///
/// **Every source pushes `Notice`, not `String`** — a bare string carries no level, so the
/// decision of screen-versus-file would fall to whoever collected it rather than to the site
/// that knows what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    /// Level — `ERROR` and `WARN` reach the screen, `INFO` only the file (REQ-L19).
    pub level: tracing::Level,
    /// Text to display, already formatted by whoever built it.
    pub text: String,
}

impl Notice {
    /// Builds an `ERROR` notice: something the user asked for is not available.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            level: tracing::Level::ERROR,
            text: text.into(),
        }
    }

    /// Builds a `WARN` notice: a capability is gone, or something works worse without failing.
    pub fn warn(text: impl Into<String>) -> Self {
        Self {
            level: tracing::Level::WARN,
            text: text.into(),
        }
    }

    /// Builds an `INFO` notice: diagnostic, never urgent.
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            level: tracing::Level::INFO,
            text: text.into(),
        }
    }

    /// Transitional alias for [`Notice::warn`], removed by this task's sweep.
    ///
    /// It exists only so the call sites still compile while each one is being classified
    /// individually. Mapping the whole family to `WARN` is deliberately the WRONG answer —
    /// it is the bulk translation D-L11 forbids, and the classification tests fail against it.
    pub fn resolution(text: impl Into<String>) -> Self {
        Self::warn(text)
    }
}

/// Announces every notice through `tracing`, at its own level.
///
/// # Parameters
///
/// * `notices` — everything a startup collected, in discovery order.
///
/// # Contract
///
/// - **Order**: `ERROR` → `WARN` → `INFO`. `sort_by_key` is stable, so two notices of the same
///   level keep the order in which they were passed.
/// - **Dedup**: two notices with the same `text` collapse into one — the trio emits the same
///   `base_url` normalization notice once per seat, and it is one fact. Applied AFTER sorting,
///   so a text emitted at two levels survives at the more severe one.
/// - **No cap**: every notice is announced (REQ-L20). The cap and its
///   `… N more diagnostic notice(s) omitted` line are gone, because the file has room for all
///   of them and the screen no longer sees `INFO` at all.
///
/// # Complexity
///
/// `O(n log n)` for the sort plus `O(n)` for the dedup, over the notices of ONE startup.
pub fn emit_notices(notices: Vec<Notice>) {
    let _ = notices;
}

/// How many `INFO`s survive the [`render_notices`] cap.
///
/// **Transitional, removed by this task** (REQ-L20): with `INFO` off the screen there is no
/// noise left for a cap to bound. It survives only until the two call sites of
/// [`render_notices`] move onto [`emit_notices`].
pub const NOTICE_MAX_INFO: usize = 5;

/// Sorts by level, deduplicates by exact text, and trims the `INFO`s past the cap.
///
/// **Transitional, removed by this task** (REQ-L20/D-L12) — see [`emit_notices`], which is what
/// replaces it.
///
/// # Complexity
///
/// `O(n log n)` for the sort plus `O(n)` for the dedup.
pub fn render_notices(notices: Vec<Notice>) -> Vec<String> {
    let mut sorted = notices;
    sorted.sort_by_key(|n| n.level);

    let mut seen_text = HashSet::with_capacity(sorted.len());
    let deduped = sorted
        .into_iter()
        .filter(|n| seen_text.insert(n.text.clone()));

    let mut info_seen = 0usize;
    let mut dropped = 0usize;
    let mut out = Vec::new();
    for n in deduped {
        if n.level == tracing::Level::INFO {
            info_seen += 1;
            if info_seen > NOTICE_MAX_INFO {
                dropped += 1;
                continue;
            }
        }
        out.push(n.text);
    }
    if dropped > 0 {
        out.push(format!("… {dropped} more diagnostic notice(s) omitted"));
    }
    out
}

/// How much of an error message survives for display.
pub const ERROR_DISPLAY_CAP: usize = 240;

/// Marker shown where an error's scaffolding was dropped.
const HEAD_DROPPED: &str = "…";

/// Formats an error for a user-facing notice.
///
/// # Parameters
///
/// * `prefix` — the notice's own lead-in, e.g. `"could not open the database"`.
/// * `err` — the error's own text.
/// * `cap` — how many bytes of `err` may survive.
///
/// # The two fixes this exists for
///
/// **P-L01 — the prefix is not repeated.** An error whose own `Display` already
/// opens with the caller's lead-in used to be rendered as that lead-in twice,
/// so the first line of the notice said nothing.
///
/// **P-L02 — truncation drops the HEAD, not the tail.** An error chain puts the
/// scaffolding first and the root cause LAST: `could not open the encrypted
/// database (opening …/.magi/state.db failed: llama-server binary not found)`.
/// Cutting from the tail — which is right for a tool RESULT and is what
/// `truncate_result` does — throws away the only part anyone needed. Eighty
/// characters of scaffolding and no diagnosis is the failure that motivated this
/// whole feature.
///
/// # Complexity
///
/// `O(n)` over the message.
#[must_use]
pub fn error_for_display(prefix: &str, err: &str, cap: usize) -> String {
    let body = err.strip_prefix(prefix).map_or(err, str::trim_start);
    let body = body.strip_prefix(": ").unwrap_or(body);
    if body.len() <= cap {
        return format!("{prefix}: {body}");
    }
    // Keep the TAIL: step FORWARD to a character boundary from the cut point.
    let mut start = body.len() - cap;
    while start < body.len() && !body.is_char_boundary(start) {
        start += 1;
    }
    let tail = body.get(start..).unwrap_or(body);
    format!("{prefix}: {HEAD_DROPPED}{tail}")
}

#[cfg(test)]
mod tests {

    /// P-L01: an error that already opens with the caller's lead-in is not
    /// prefixed with it twice.
    #[test]
    fn the_prefix_is_not_repeated_when_the_error_already_carries_it() {
        let shown = error_for_display(
            "could not open the database",
            "could not open the database: file is locked",
            ERROR_DISPLAY_CAP,
        );
        assert_eq!(shown, "could not open the database: file is locked");
        assert_eq!(
            shown.matches("could not open the database").count(),
            1,
            "saying it twice makes the first line of the notice say nothing"
        );
    }

    /// P-L02: the ROOT CAUSE is at the tail of an error chain, so truncation
    /// drops the head.
    #[test]
    fn a_long_error_is_truncated_at_the_head_so_the_cause_survives() {
        let scaffolding = "opening the encrypted store failed: ".repeat(10);
        let cause = "llama-server binary not found";
        let shown = error_for_display("memory", &format!("{scaffolding}{cause}"), 60);

        assert!(
            shown.contains(cause),
            "the diagnosis must survive; that is the whole point: {shown}"
        );
        assert!(
            !shown.contains(&scaffolding),
            "and the scaffolding is what goes: {shown}"
        );
    }

    #[test]
    fn a_short_error_is_shown_whole_without_a_marker() {
        let shown = error_for_display("memory", "disk full", ERROR_DISPLAY_CAP);
        assert_eq!(shown, "memory: disk full");
    }
    use super::*;

    /// The actionable items first, regardless of the order in which they were discovered.
    #[test]
    fn notices_are_ordered_by_level_not_by_discovery() {
        let out = render_notices(vec![
            Notice::info("measured window: 128k"),
            Notice::error("the trio is not buildable: missing OPENAI_API_KEY"),
            Notice::resolution("`[embedding].base_url` inherited the root"),
        ]);
        assert!(
            out[0].contains("not buildable"),
            "the one demanding action goes first"
        );
        assert!(out[1].contains("inherited"));
        assert!(out[2].contains("measured window"));
    }

    /// The cap trims NOISE, never signals.
    #[test]
    fn the_cap_truncates_info_only_and_says_how_many_it_dropped() {
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO + 3)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        v.push(Notice::error("b1"));
        v.push(Notice::resolution("r1"));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains("b1")),
            "Blocking is NEVER trimmed"
        );
        assert!(
            out.iter().any(|n| n.contains("r1")),
            "Resolution is not trimmed either"
        );
        assert_eq!(
            out.iter().filter(|n| n.starts_with('d')).count(),
            NOTICE_MAX_INFO
        );
        assert!(
            out.last().unwrap().contains('3'),
            "says how many it dropped"
        );
    }

    /// Two sources can produce the SAME warning: the three seats with the same `base_url` emit
    /// the `/v1` normalization.
    #[test]
    fn identical_notices_are_emitted_once() {
        let n = "notice: `base_url` de Ollama sin sufijo `/v1`";
        let out = render_notices(vec![
            Notice::resolution(n),
            Notice::resolution(n),
            Notice::resolution(n),
        ]);
        assert_eq!(out.len(), 1, "three seats, one notice");
    }

    /// Empty edge case (B13): nothing to sort, deduplicate, or trim — never panics, and with no
    /// `Info` to trim there is no "omitted" line.
    #[test]
    fn empty_input_renders_to_an_empty_list() {
        let out = render_notices(vec![]);
        assert!(out.is_empty());
    }

    /// Exact cap boundary: `info_seen > NOTICE_MAX_INFO` is strict, so exactly
    /// `NOTICE_MAX_INFO` `Info` notices do not trigger ANY trimming. Only the above-the-cap
    /// case was covered before this test; the off-by-one at the boundary is the classic defect
    /// of this kind of guard.
    #[test]
    fn exactly_the_cap_worth_of_info_drops_nothing() {
        let v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        let out = render_notices(v);
        assert_eq!(
            out.len(),
            NOTICE_MAX_INFO,
            "none is trimmed at the exact boundary"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "with no trimming there is no omitted line: {out:?}"
        );
    }

    /// The signal-vs-noise property the module exists to guarantee: same text, different tiers
    /// — the more severe one (`Blocking`) survives, not the `Info`.
    ///
    /// With IDENTICAL text, which one survived cannot be read directly from the output `String`
    /// (they are the same string). It is tested by its EFFECT on the cap: exactly
    /// `NOTICE_MAX_INFO` distinct filler `Info`s are added, which on their own do not trigger
    /// any trimming (see the exact boundary test above). If the duplicate survived as `Info`
    /// instead of `Blocking`, it would add one more `Info` and WOULD trigger the trim. That it
    /// does not trigger it, and that the duplicate text is still present, is the proof that the
    /// `Blocking` survived — which never counts against the cap.
    #[test]
    fn cross_level_duplicate_text_keeps_the_more_severe_level() {
        let dup_text = "the trio is not buildable: missing OPENAI_API_KEY";
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("filler{i}")))
            .collect();
        v.push(Notice::info(dup_text));
        v.push(Notice::error(dup_text));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains(dup_text)),
            "the duplicate must survive (under the Blocking tier): {out:?}"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "if the Info survived, it would exceed the cap and something would get \
             trimmed: {out:?}"
        );
    }
}
