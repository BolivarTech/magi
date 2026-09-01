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
///
/// **The cost of one target, stated because someone will need it:** collecting the emission
/// in one place flattens the module target for all of it, so a REQ-L30/L31 per-target
/// directive can address the startup as a unit and not a subsystem within it. What it does
/// not cost is provenance — each event still carries its own file, line and span.
const NOTICE_TARGET: &str = "magi_rs::startup";

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
    emit_notices_into(notices, &mut std::io::stderr().lock());
}

/// [`emit_notices`], with the last-resort mouth supplied.
///
/// # Parameters
///
/// * `notices` — everything a startup collected.
/// * `fallback` — where a notice goes when no subscriber was ever installed. Headless passes
///   `stderr` (via [`emit_notices`]); a test passes a buffer, which is what makes the
///   no-layer branch observable without capturing a process's own file descriptors.
///
/// # Why the terminal passes something other than `stderr`
///
/// A TUI session started outside a `.magi/` workspace installs no layer at all —
/// `init_logging` is guarded on a discovered workspace — so its startup notices take the
/// branch below. Written to `stderr` they land on the PRIMARY buffer, which
/// `EnterAlternateScreen` swaps out a moment later, and the first-run "no `.magi/` state
/// directory found — run `magi init`" is hidden for the whole session. So `run_tui_ext`
/// supplies a fallback that writes into the transcript instead, which is why this is `pub`
/// and not the private helper it started as.
///
/// # Complexity
///
/// [`emit_notices`]'s, plus `O(n)` writes on the fallback path.
pub fn emit_notices_into(notices: Vec<Notice>, fallback: &mut dyn std::io::Write) {
    let ordered = ordered_for_emission(notices);

    // **`OFF` is the no-subscriber state, and it is the right question to ask.** The global
    // maximum starts at `OFF` and is only raised when a subscriber is installed, so this
    // answers "would `announce` reach anything at all?" rather than the narrower "was a
    // dispatcher ever set". Our own layer hints `max(file_filter, SCREEN_LEVEL)`, which is
    // never `OFF`, so an installed layer never takes the fallback.
    if tracing::level_filters::LevelFilter::current() != tracing::level_filters::LevelFilter::OFF {
        for notice in &ordered {
            announce(notice);
        }
        return;
    }

    // No layer, so no file either: `init_logging` is guarded on a discovered `.magi/`
    // workspace and this function is not. The screen policy still decides who speaks —
    // writing every line here would put the whole diagnostic list in front of the user in
    // exactly the case SC-L14 exists to keep quiet — and what is left with nowhere to go is
    // the `INFO` half, whose declared destination is a file that does not exist. The first
    // `WARN` in this situation says why there is no workspace, which is also why there is no
    // log.
    for notice in ordered.iter().filter(|n| n.level <= SCREEN_LEVEL) {
        // **Audited, because this is an output** (REQ-L48). The layer path reaches a mouth
        // through `announce`, and therefore through the auditor; this path reaches one
        // directly. REQ-L48's guarantee is that the auditor is the only route to any output
        // and that the COMPILER enforces it — every sink takes an `Audited`, so handing one a
        // raw `String` does not compile — and a `&str` written straight to a `Write` is the
        // one shape that escapes the type. Nothing here is static: a startup notice carries a
        // resolved `base_url`, an error chain or a vault entry name, and each of those sites
        // redacting on its own is the convention REQ-L48 exists to replace.
        //
        // The PROCESS auditor, not a local one: the registered secrets have to be the same set
        // for every mouth, and a second auditor is a second, emptier idea of what to mask.
        let (audited, alarm) = crate::logging::process_auditor().audit(
            &notice.text,
            NOTICE_TARGET,
            None,
            notice.text.len(),
        );
        // A failed write to stderr has nowhere to be reported, so it is dropped rather than
        // escalated: this is already the last resort.
        let _ = writeln!(fallback, "{}", audited.as_str());
        // Masking and the alarm that says masking happened travel together — both, never one.
        // The appender is what carries the pair on the layer path; with no layer there is no
        // appender, so the alarm goes to the same last-resort mouth. It quotes neither the
        // secret nor the line (REQ-L50).
        //
        // **The alarm is ROUTED, not merely rendered** (REQ-L48). `render_alarm` promises not
        // to quote the secret, but that promise is a convention and REQ-L48's whole point is
        // that a convention is not what stands between a runtime-composed string and a mouth.
        // The text it does interpolate — a secret's NAME and a target — is composed here, and
        // a name is operator-chosen: nothing stops one from being another secret's value.
        //
        // **The loop terminates, and by the auditor's own bookkeeping rather than a cap.**
        // `Auditor::alarm` latches `(secret, target)`; the target is fixed at `NOTICE_TARGET`
        // for every pass, so each iteration must find a secret not yet latched at it. The
        // registered set is finite, so the chain is bounded by its size and needs no counter
        // to say so.
        let mut pending = alarm;
        while let Some(raised) = pending {
            let rendered = crate::logging::auditor::render_alarm(&raised);
            let (audited_alarm, next) = crate::logging::process_auditor().audit(
                &rendered,
                NOTICE_TARGET,
                None,
                rendered.len(),
            );
            let _ = writeln!(fallback, "{}", audited_alarm.as_str());
            pending = next;
        }
    }
}

/// The level at and above which a notice reaches a human.
///
/// The same constant the layer's screen branch is wired at (REQ-L19), referenced rather than
/// repeated: two copies would let the fallback and the layer disagree about what a user sees,
/// and the disagreement would only show up in the case where there is no layer to compare
/// against.
const SCREEN_LEVEL: tracing::Level = crate::logging::SCREEN_LEVEL;

/// Puts the notices in the order they are announced in, with the duplicates gone.
///
/// Split from [`emit_notices`] so the decision — order and dedup — stays a pure function a
/// test can read, and only the `tracing` call needs a subscriber. It is the same division the
/// rest of the logging subsystem is built on.
///
/// # Complexity
///
/// `O(n log n)` for the sort plus `O(n)` for the dedup.
fn ordered_for_emission(notices: Vec<Notice>) -> Vec<Notice> {
    let mut sorted = notices;
    // `tracing::Level` orders ERROR < WARN < INFO, and `sort_by_key` is stable, so this is
    // "most severe first, discovery order within a level".
    sorted.sort_by_key(|n| n.level);

    let mut seen_text = HashSet::with_capacity(sorted.len());
    sorted
        .into_iter()
        .filter(|n| seen_text.insert(n.text.clone()))
        .collect()
}

/// Emits one notice at its own level.
///
/// # Why a chain of comparisons and not a `match`
///
/// `tracing::Level`'s constants are associated constants of a struct, which cannot appear in a
/// pattern. The levels are also compile-time literals in the macro rather than a value it
/// accepts, so the three arms have to be written out.
///
/// # Complexity
///
/// `O(n)` over the text.
fn announce(notice: &Notice) {
    let text = notice.text.as_str();
    if notice.level == tracing::Level::ERROR {
        tracing::event!(target: NOTICE_TARGET, tracing::Level::ERROR, "{text}");
    } else if notice.level == tracing::Level::WARN {
        tracing::event!(target: NOTICE_TARGET, tracing::Level::WARN, "{text}");
    } else {
        tracing::event!(target: NOTICE_TARGET, tracing::Level::INFO, "{text}");
    }
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
    use super::*;

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

    /// C1: a notice must not vanish when there is no layer to route it.
    ///
    /// `init_logging` is guarded on a discovered `.magi/` workspace on both surfaces, and
    /// `emit_notices` is not. So a run started in a directory with no workspace has no
    /// subscriber at all, and every notice — `WARN` and `ERROR` included — used to be a
    /// no-op. What that cost the user was the one message that explains the situation they
    /// are in: "no .magi/ state directory found — run `magi init`", followed by a session
    /// that quietly saves nothing.
    ///
    /// The `INFO` line is asserted ABSENT rather than present, and that is the half worth
    /// reading twice: with no layer there is no file either, so a fallback that printed
    /// everything would put the whole diagnostic list on the screen — defeating SC-L14 in
    /// exactly the production case this fixes, while SC-L14's own test stayed green.
    #[test]
    fn a_notice_still_reaches_the_user_when_no_layer_was_installed() {
        assert_eq!(
            tracing::level_filters::LevelFilter::current(),
            tracing::level_filters::LevelFilter::OFF,
            "this test is about the NO-subscriber path, and something installed one"
        );

        let mut out = Vec::new();
        emit_notices_into(
            vec![
                Notice::info("memory: 0 active, 0 archived"),
                Notice::warn("no .magi/ state directory found"),
                Notice::error("the trio is not buildable"),
            ],
            &mut out,
        );

        let shown = String::from_utf8(out).expect("the fallback writes UTF-8");
        assert!(
            shown.contains("no .magi/ state directory found"),
            "the warning that explains the run vanished: {shown:?}"
        );
        assert!(
            shown.contains("the trio is not buildable"),
            "an error vanished: {shown:?}"
        );
        assert!(
            !shown.contains("memory: 0 active"),
            "a diagnostic reached the screen: {shown:?}"
        );
    }

    /// REQ-L48: the last-resort mouth is an output, so it goes through the auditor too.
    ///
    /// The requirement's guarantee is structural — `Audited` has one constructor and every
    /// output takes only an `Audited`, so handing a sink a raw `String` does not compile. This
    /// path took a `&str` and wrote it, which is the one shape the type system cannot refuse,
    /// and so the guarantee held everywhere except here.
    ///
    /// **A notice is not static text.** The startup list carries resolved `base_url`s, error
    /// chains and vault entry names, all composed at runtime, and every one of them is `WARN`
    /// or `ERROR` — the levels this branch prints. Each site redacts on its own today, which
    /// is a convention, and a convention is what REQ-L48 exists to replace.
    ///
    /// The secret below is registered with the PROCESS auditor, which is the same instance the
    /// layer uses, and it is placed in free text rather than inside a URL authority on
    /// purpose: pass 1 would catch a `user:pass@host` by shape whether or not anything was
    /// registered, so a URL would let this test pass against an auditor that never ran the
    /// exact pass.
    #[test]
    fn the_no_layer_fallback_audits_before_it_writes() {
        assert_eq!(
            tracing::level_filters::LevelFilter::current(),
            tracing::level_filters::LevelFilter::OFF,
            "this test is about the NO-subscriber path, and something installed one"
        );

        // Not key-shaped, so no pattern matcher claims it: the exact pass is the only thing
        // that can redact this, and the exact pass is the auditor.
        let value = "correct-horse-battery-staple-42";
        crate::logging::register_process_secrets(&[(
            crate::logging::auditor::SecretName::new("A_FALLBACK_GUARD_ONLY_SECRET"),
            value,
        )]);

        let mut out = Vec::new();
        emit_notices_into(
            vec![Notice::warn(format!(
                "the vault could not be opened with {value}"
            ))],
            &mut out,
        );

        let shown = String::from_utf8(out).expect("the fallback writes UTF-8");
        assert!(
            !shown.contains(value),
            "a registered secret reached the fallback in the clear, so this mouth is outside \
             the audit: {shown:?}"
        );
        assert!(
            shown.contains(crate::logging::auditor::REDACTED),
            "the line must arrive masked rather than dropped: {shown:?}"
        );
        assert!(
            shown.contains("the vault could not be opened"),
            "and the rest of the notice must survive, or the fix traded a leak for a silence: \
             {shown:?}"
        );
    }

    /// REQ-L48, the half the previous pass left open: the ALARM is an output too.
    ///
    /// The notice text was routed through the auditor and the alarm raised about it was
    /// not, so one of the two lines this branch writes still reached a mouth without
    /// passing the only thing allowed to hand a mouth anything. What kept it safe was
    /// `render_alarm`'s promise never to quote the secret — a convention, which is exactly
    /// what REQ-L48 exists to replace with a route. The class had already produced a raw
    /// `writeln!` and a fresh, empty `Auditor` in this same milestone, so "the invariant
    /// says it cannot happen" is not evidence; the write is.
    ///
    /// **How the mutation is made to bite.** `render_alarm` interpolates the secret's NAME,
    /// so a second secret whose VALUE is the first one's name puts a registered value into
    /// the alarm text by construction. Routed, it is masked; written raw, it ships. Nothing
    /// about the assertion depends on `render_alarm`'s wording.
    #[test]
    fn the_no_layer_fallback_audits_the_alarm_it_writes() {
        assert_eq!(
            tracing::level_filters::LevelFilter::current(),
            tracing::level_filters::LevelFilter::OFF,
            "this test is about the NO-subscriber path, and something installed one"
        );

        // The trigger's name is the guard's value, which is what puts a registered value
        // inside the rendered alarm.
        const TRIGGER_NAME: &str = "MS2_ALARM_ROUTE_TRIGGER";
        let trigger_value = "alarm-route-trigger-value-77";
        crate::logging::register_process_secrets(&[
            (
                crate::logging::auditor::SecretName::new(TRIGGER_NAME),
                trigger_value,
            ),
            (
                crate::logging::auditor::SecretName::new("MS2_ALARM_TEXT_GUARD"),
                TRIGGER_NAME,
            ),
        ]);

        let mut out = Vec::new();
        emit_notices_into(
            vec![Notice::warn(format!(
                "the vault could not be opened with {trigger_value}"
            ))],
            &mut out,
        );

        let shown = String::from_utf8(out).expect("the fallback writes UTF-8");
        assert!(
            shown.contains("SECURITY:"),
            "the alarm must still be delivered — routing it must not silence it: {shown:?}"
        );
        assert!(
            !shown.contains(TRIGGER_NAME),
            "the alarm line carried a registered value in the clear, so this write is \
             outside the audit route: {shown:?}"
        );
    }

    /// The actionable items first, regardless of the order in which they were discovered.
    #[test]
    fn notices_are_ordered_by_level_not_by_discovery() {
        let out = ordered_for_emission(vec![
            Notice::info("measured window: 128k"),
            Notice::error("the trio is not buildable: missing OPENAI_API_KEY"),
            Notice::warn("the vault could not be opened"),
        ]);
        assert!(
            out[0].text.contains("not buildable"),
            "the one demanding action goes first"
        );
        assert!(out[1].text.contains("vault"));
        assert!(out[2].text.contains("measured window"));
    }

    /// How many notices a run can pile up before anyone would have thought about a cap.
    const MORE_THAN_ANYONE_READS: usize = 20;

    /// REQ-L20/D-L12: the cap is gone, so nothing is trimmed and there is no line saying
    /// anything was.
    ///
    /// A cap is a count standing in for a policy, and it produced the worse of both outcomes:
    /// the reader still got five lines of noise, and the rest was destroyed rather than filed.
    /// With `INFO` off the screen there is nothing left for it to protect anyone from.
    #[test]
    fn every_notice_survives_now_that_the_cap_is_gone() {
        let v: Vec<Notice> = (0..MORE_THAN_ANYONE_READS)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        let out = ordered_for_emission(v);
        assert_eq!(out.len(), MORE_THAN_ANYONE_READS, "nothing may be trimmed");
        assert!(
            !out.iter().any(|n| n.text.contains("omitted")),
            "the truncation line must not exist at all: {out:?}"
        );
    }

    /// Two sources can produce the SAME warning: the three seats with the same `base_url` emit
    /// the `/v1` normalization.
    #[test]
    fn identical_notices_are_emitted_once() {
        let n = "notice: `base_url` had no `/v1` suffix";
        let out = ordered_for_emission(vec![Notice::info(n), Notice::info(n), Notice::info(n)]);
        assert_eq!(out.len(), 1, "three seats, one notice");
    }

    /// Empty edge case (B13): nothing to sort or deduplicate, and it never panics.
    #[test]
    fn an_empty_list_emits_nothing() {
        assert!(ordered_for_emission(vec![]).is_empty());
    }

    /// The signal-vs-noise property the module exists to guarantee: same text at two levels —
    /// the more severe one survives, so the text still reaches the screen.
    ///
    /// It is the dedup's ORDER that makes this true, and the reason the sort comes first: with
    /// dedup before the sort, whichever copy was discovered earlier would win, and the level a
    /// notice ends up at would depend on the order in which two unrelated subsystems happened
    /// to run.
    #[test]
    fn cross_level_duplicate_text_keeps_the_more_severe_level() {
        let dup_text = "the trio is not buildable: missing OPENAI_API_KEY";
        let out = ordered_for_emission(vec![Notice::info(dup_text), Notice::error(dup_text)]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].level,
            tracing::Level::ERROR,
            "the copy that reaches the screen must be the one that survived"
        );
    }
}
