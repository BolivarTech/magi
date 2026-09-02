// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! Startup notices and the level each one is announced at.
//!
//! # Why it lives in the LIB and not under `system/`/`tui/`
//!
//! Because both surfaces need it and neither owns it: `main.rs` assembles the list a startup
//! announces and hands it to [`emit_notices`], and the TUI supplies its own mouth to
//! [`emit_notices_into`]. Nothing here reaches the network or the filesystem, and no state is
//! kept in the module.
//!
//! **It is not pure, and the sentence that used to say so was wrong.** [`emit_notices`] writes
//! to `stderr`, [`emit_notices_into`] writes to whatever mouth it is handed, `announce` goes
//! through the global `tracing` dispatcher, and `write_audited` reads the PROCESS auditor —
//! shared, mutable, and deliberately so, because a second auditor is a second and emptier idea
//! of what to mask. The part that really is pure is `ordered_for_emission`, which is why the
//! order-and-dedup decision was split out of the emitting function in the first place.
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
/// **The three no-layer tests below need a process each, so run them under
/// `cargo nextest`, never plain `cargo test`.** Each asserts
/// `LevelFilter::current() == OFF` as its precondition, and one registers a
/// process secret; a shared process in which any other test installs a
/// subscriber makes them fail on that assertion. They fail loudly rather than
/// silently, which is why no serialisation crate is pulled in for them: the
/// runner this repository already uses gives each its own process, and the
/// precondition names the collision when it does not.
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
        write_audited(fallback, &notice.text);
    }
}

/// Announces one already-composed line on the process's last-resort mouth, audited.
///
/// # Parameters
///
/// * `text` — the line to show, without a trailing newline.
///
/// # Why a startup's fatal errors need this and cannot use [`emit_notices`]
///
/// A `Notice` is something a run collected and hands over; the two funnels this serves are
/// the opposite — the process is ending, there is no list and frequently no layer, and the
/// text is an error chain composed from types this crate does not own. `eprintln!` is the one
/// mouth REQ-L48's type-level guarantee cannot reach, because it takes a format string rather
/// than an [`Audited`][a]: nothing about it can be made not to compile. So the route is
/// supplied instead of enforced, and `no_fatal_error_is_announced_outside_the_audit_route`
/// in `main.rs` is what keeps the two funnels on it.
///
/// [a]: crate::logging::auditor::Audited
///
/// # Complexity
///
/// The auditor's, over `text`, plus one write per alarm the pass raises.
pub fn eprint_audited(text: &str) {
    write_audited(&mut std::io::stderr().lock(), text);
}

/// Audits `text` and writes it — with every alarm it raises — to `mouth`.
///
/// # Parameters
///
/// * `mouth` — the last-resort destination.
/// * `text` — runtime-composed text; nothing here may assume it is static.
///
/// # Complexity
///
/// The auditor's, over `text`, plus one pass per alarm.
fn write_audited(mouth: &mut dyn std::io::Write, text: &str) {
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
    let (audited, alarm) =
        crate::logging::process_auditor().audit(text, NOTICE_TARGET, None, text.len());
    // A failed write to stderr has nowhere to be reported, so it is dropped rather than
    // escalated: this is already the last resort.
    let _ = writeln!(mouth, "{}", audited.as_str());
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
    drain_alarms(alarm, mouth);
}

/// Routes an alarm chain to `mouth`, audited, until the chain is exhausted.
///
/// # Parameters
///
/// * `first` — the alarm the caller's own audit raised, if any.
/// * `mouth` — where the alarms go.
///
/// # Complexity
///
/// `O(a)` passes for `a` alarms, each the auditor's own over a short line.
fn drain_alarms(
    first: Option<crate::logging::auditor::AuditExempt>,
    mouth: &mut dyn std::io::Write,
) {
    let mut pending = first;
    while let Some(raised) = pending {
        let rendered = crate::logging::auditor::render_alarm(&raised);
        let (audited_alarm, next) =
            crate::logging::process_auditor().audit(&rendered, NOTICE_TARGET, None, rendered.len());
        let _ = writeln!(mouth, "{}", audited_alarm.as_str());
        pending = next;
    }
}

/// Audits `text` and RETURNS it, rather than writing it to a mouth.
///
/// # Why a returning variant exists at all
///
/// The headless surface composes a structured payload -- a JSON envelope a CI job
/// parses -- so its error text is not a line to print but a FIELD to fill. The
/// auditor still has to see it: the payload reaches stdout, and REQ-L48's rule is
/// about reaching an output, not about the shape of the write.
///
/// **This is not a second redaction.** A pattern matcher (`sanitize_error_message`)
/// masks what LOOKS like a credential; the exact pass masks what this run actually
/// registered, which no composition site knows about. A vault-substituted password
/// that is not key-shaped -- an ordinary passphrase -- passes every pattern and only
/// the exact pass catches it. Both, in that order.
///
/// # Parameters
///
/// * `text` — runtime-composed text bound for a structured field.
///
/// # Returns
///
/// The masked text. Any alarm the pass raised goes to `stderr`, which is where the
/// headless surface's own diagnostics go and is never the payload: putting an alarm
/// inside the envelope would corrupt a contract a consumer parses.
///
/// # Complexity
///
/// The auditor's, over `text`, plus one pass per alarm.
#[must_use]
pub fn audited_field(text: &str) -> String {
    audited_field_at(text, NOTICE_TARGET)
}

/// [`audited_field`], attributed to `target`.
///
/// # Why the target is a parameter and not a constant
///
/// The auditor's alarm latch is keyed on `(secret, target)`. A surface
/// that borrows another's target therefore shares its latch: a startup notice that
/// already alarmed for a secret **silently suppresses** this surface's alarm for the
/// same one. The masking still happens; the notice that it happened disappears, which
/// is the half an operator acts on (MS2 gate, integration pass, Caspar).
///
/// # Parameters
///
/// * `text` — runtime-composed text bound for a structured field.
/// * `target` — the surface this mask belongs to. One per surface, `'static`.
///
/// # Returns
///
/// The masked text. Any alarm goes to `stderr`, never into the caller's payload.
///
/// # Complexity
///
/// The auditor's, over `text`, plus one pass per alarm.
#[must_use]
pub fn audited_field_at(text: &str, target: &'static str) -> String {
    let (audited, alarm) = crate::logging::process_auditor().audit(text, target, None, text.len());
    drain_alarms(alarm, &mut std::io::stderr().lock());
    audited.as_str().to_string()
}

/// Splits notices by the mouth their level sends them to (REQ-L19).
///
/// # Parameters
///
/// * `notices` — the notices to split, in discovery order.
///
/// # Returns
///
/// `(screen, file)` — `ERROR` and `WARN` in the first, everything else in the second.
/// Discovery order is preserved within each half; [`emit_notices`] sorts, so partitioning
/// is all this has to do.
///
/// # Why a startup needs the two halves apart
///
/// A startup collects notices before the logging layer exists and keeps collecting after.
/// A screen-bound notice has a mouth the whole time — [`emit_notices_into`]'s fallback —
/// while a file-bound one has none until the layer is up, so announcing the whole list
/// early destroys the `INFO` half and carrying the whole list late loses everything to any
/// path that returns in between. Emitting each half where its mouth exists is the only
/// arrangement that loses neither.
///
/// # Complexity
///
/// `O(n)`.
#[must_use]
pub fn partition_by_mouth(notices: impl IntoIterator<Item = Notice>) -> (Vec<Notice>, Vec<Notice>) {
    notices.into_iter().partition(|n| n.level <= SCREEN_LEVEL)
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
    //
    // **Its `Ord` is INVERTED relative to its discriminants, and reading only one of the two
    // halves gives the opposite answer.** `LevelInner` is declared `Trace = 0 … Error = 4`, so
    // the numbers ascend from trace to error; `impl Ord for Level` then compares
    // `(other as usize).cmp(&(self as usize))` — operands swapped — so the ORDER ascends the
    // other way. "Greater" means MORE VERBOSE, which is also why `<= SCREEN_LEVEL` selects the
    // severe half and `> Level::WARN` reads as "informational, therefore a success".
    //
    // A reviewer has already asked for this sort and those comparisons to be reversed "to match
    // standard `tracing::Level` ordering". Reversing them sends `ERROR` to the file and `INFO`
    // to the screen. `tracing_level_orders_most_severe_first` asserts the ordering so the next
    // reader can check it rather than take this paragraph on trust.
    sorted.sort_by_key(|n| n.level);

    let mut seen_text = HashSet::with_capacity(sorted.len());
    sorted
        .into_iter()
        .filter(|n| seen_text.insert(n.text.clone()))
        .collect()
}

/// Emits one notice at its own level.
///
/// # Why the three arms are written out
///
/// `tracing::event!` builds a `static __CALLSITE` around the level, so the level has to be a
/// compile-time constant: passing `notice.level` is `error[E0435]: attempt to use a
/// non-constant value in a constant`. One arm per level is the only shape that gives the macro
/// a literal.
///
/// **What this is NOT is a limit on patterns**, and the sentence that used to say so was
/// wrong. `tracing::Level`'s constants match perfectly well — the compiler even reasons about
/// exhaustiveness through them, answering a two-arm `match` with `Level(LevelInner::Trace)`,
/// `Level(LevelInner::Debug)` and `Level(LevelInner::Info)` not covered. Both alternatives
/// were compiled rather than reasoned about, which is what separated the true half from the
/// false one. A `match` is therefore available; it would just carry the same three literal
/// levels inside its arms, so it buys nothing over the comparisons.
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

    /// REQ-L19: the split follows the level, because the level is what names the mouth.
    ///
    /// The headless startup is the consumer. It collects its configuration notices before
    /// the layer exists, so announcing them there sends the whole list through the
    /// fallback, which shows what the screen policy admits and DROPS the rest — the `INFO`
    /// half, on a completely successful run, with no file yet for it to go to.
    #[test]
    fn notices_are_split_by_the_mouth_their_level_reaches() {
        let (screen, file) = partition_by_mouth(vec![
            Notice::info("the embedder inherits the root endpoint"),
            Notice::warn("`provider` is blank, so the default is in force"),
            Notice::error("the trio is not buildable"),
            Notice::info("`base_url` had no `/v1` suffix"),
        ]);

        assert_eq!(
            screen.len(),
            2,
            "WARN and ERROR still reach a user with no layer installed: {screen:?}"
        );
        assert!(
            screen.iter().all(|n| n.level <= SCREEN_LEVEL),
            "the screen half must hold exactly what the screen policy admits: {screen:?}"
        );
        assert_eq!(
            file.len(),
            2,
            "the diagnostics must survive the split rather than be dropped by it: {file:?}"
        );
        assert!(
            file.iter().all(|n| n.level == tracing::Level::INFO),
            "an INFO has nowhere to go until the layer is up, which is why it waits: {file:?}"
        );
        assert_eq!(
            file[0].text, "the embedder inherits the root endpoint",
            "discovery order must survive within a half"
        );
    }

    /// The ordering every predicate in this module and in `logging` is built on, pinned.
    ///
    /// **`tracing::Level`'s `Ord` is INVERTED relative to its discriminants**, and reading
    /// only one of the two halves is how a reviewer arrives at the opposite conclusion.
    /// `LevelInner` really is declared `Trace = 0 … Error = 4`, so the numbers ascend from
    /// trace to error; `impl Ord for Level` then compares `(other as usize).cmp(&(self as
    /// usize))`, swapping the operands, so the ORDER ascends the other way:
    /// `ERROR < WARN < INFO < DEBUG < TRACE`. "Greater" means MORE VERBOSE.
    ///
    /// Everything downstream follows from that one fact and reads backwards without it:
    /// `sort_by_key(level)` puts `ERROR` first, `level <= SCREEN_LEVEL` selects `ERROR` and
    /// `WARN` for the screen, and `embedding.rs`'s `level > Level::WARN` calls an `INFO` a
    /// success. Inverting any one of them would send `ERROR` to the file and `INFO` to the
    /// screen, which is the inversion this milestone exists to prevent — so the fact is
    /// asserted here rather than left to a comment a future reader has to trust.
    #[test]
    fn tracing_level_orders_most_severe_first() {
        let mut all = vec![
            tracing::Level::TRACE,
            tracing::Level::DEBUG,
            tracing::Level::INFO,
            tracing::Level::WARN,
            tracing::Level::ERROR,
        ];
        all.sort();
        assert_eq!(
            all,
            vec![
                tracing::Level::ERROR,
                tracing::Level::WARN,
                tracing::Level::INFO,
                tracing::Level::DEBUG,
                tracing::Level::TRACE,
            ],
            "ascending order is most-severe-first; `sort_by_key` in this module depends on it"
        );
        assert!(
            tracing::Level::ERROR < tracing::Level::WARN,
            "an ERROR sorts before a WARN"
        );
        assert!(
            tracing::Level::INFO > tracing::Level::WARN,
            "`>` means more verbose, which is why `ok_from_level` reads `level > WARN`"
        );
        assert!(
            tracing::Level::ERROR <= SCREEN_LEVEL && tracing::Level::WARN <= SCREEN_LEVEL,
            "`<= SCREEN_LEVEL` must select the two levels a human sees"
        );
        assert!(
            !(tracing::Level::INFO <= SCREEN_LEVEL),
            "and must exclude the diagnostic half"
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
