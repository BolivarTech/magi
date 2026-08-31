// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! One `tracing` layer, carrying both filters and both sinks.
//!
//! # Why one layer and not two
//!
//! Two `tracing` layers have **no point of convergence** between the layers and
//! their sinks, so "one auditor before the fan-out" cannot exist with two: it
//! either runs twice — duplicating the scan on the hot path — or runs before
//! filtering and audits events nobody will emit.
//!
//! # The order inside, which is a security order
//!
//! render (stage 1) → **audit** (stage 2) → escape (stage 3) → ask each filter
//! → fan out. Budgets are measured on the escaped text, because that is what
//! gets written.
//!
//! # Auditing comes before truncation, and inverting it leaks
//!
//! The screen branch caps what it shows. Truncating first would split a secret
//! that straddles the cap: the exact pass looks for the whole value, finds
//! **neither** half, and the half on the visible side ships unredacted. The cost
//! of the right order — auditing 100 MiB to show 64 KiB — is accepted and named:
//! an event that size is rare, and the alternative is a leak.

use std::sync::Arc;

use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use crate::logging::appender::{DailyAppender, Priority, Submitted};
use crate::logging::auditor::{Auditor, Queued};
use crate::logging::render::{escape_for_line, render_event};

/// Cap on what the screen branch displays.
///
/// Without it a 100 MiB `ERROR` reaches the TUI's message list, which caps
/// nothing, and from there the clipboard with one keystroke.
pub const TUI_PAYLOAD_MAX_BYTES: usize = 64 * 1024;

/// Writes to the appender's queue.
pub struct FileSink {
    appender: Arc<DailyAppender>,
}

impl FileSink {
    /// Wraps an appender as the file branch of the layer.
    #[must_use]
    pub fn new(appender: Arc<DailyAppender>) -> Self {
        Self { appender }
    }
}

/// Pushes to the caller's notice sink. **Wired in MS2.**
pub struct TuiSink {
    sink: Arc<dyn crate::logging::NoticeDelivery>,
}

impl TuiSink {
    /// Wraps a delivery as the screen branch of the layer.
    #[must_use]
    pub fn new(sink: Arc<dyn crate::logging::NoticeDelivery>) -> Self {
        Self { sink }
    }
}

/// A credential-free announcement the layer makes about ITSELF.
///
/// Separate from the screen branch on purpose: the screen branch is a filter
/// over ordinary events and MS2 decides what passes it, while this is the
/// subsystem reporting that it has stopped working. Routing the second through
/// the first would let a filter silence the one message that must never be
/// silenced.
struct Reporter {
    sink: Arc<dyn crate::logging::NoticeDelivery>,
    /// The process auditor, shared with the layer that owns this reporter.
    auditor: Arc<Auditor>,
    /// Latched so the notice is emitted ONCE, not per discarded event.
    ///
    /// The failure modes here are all high-frequency by nature — a full channel
    /// is full for every event that follows — so an unlatched notice would turn
    /// one problem into a flood that hides it.
    /// Latched for the DEGRADED tier: events are being lost, the branch lives.
    degraded: std::sync::atomic::AtomicBool,
    /// Latched for the STOPPED tier: the branch is gone for the rest of the run.
    ///
    /// **Two latches and not one.** With a single one, a moment of congestion
    /// fires the notice and the writer's death is then silent forever: the
    /// operator is told events are being discarded and never told the log
    /// stopped. Those are different things to know and different things to do
    /// about, so the worse of the two must always be able to speak.
    stopped: std::sync::atomic::AtomicBool,
}

impl Reporter {
    /// Announces `text` once for its tier.
    ///
    /// # Parameters
    ///
    /// * `latch` — the tier's latch; a tier speaks once and no more.
    /// * `text` — what to say.
    fn announce_once(&self, latch: &std::sync::atomic::AtomicBool, text: &str) {
        use std::sync::atomic::Ordering;
        if latch
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        // Through the PROCESS auditor, not a fresh one, and escaped like every
        // other line that reaches a mouth. The text is ours and carries no
        // credential today, so neither step changes what is delivered -- which
        // is exactly why it would be easy to skip, and why the next author who
        // interpolates a path or an error into this message would find the one
        // path that had been left exempt.
        let (line, _) = self
            .auditor
            .audit(text, "magi_rs::logging", None, text.len());
        self.sink.deliver(&line.map_line(escape_for_line));
    }

    /// Turns a submission outcome into the notice it deserves, if any.
    fn report(&self, outcome: Submitted) {
        match outcome {
            Submitted::Queued => {}
            Submitted::DroppedFull => self.announce_once(
                &self.degraded,
                "warning: the log queue is full and events are being discarded; \
                 the session continues and the log is now incomplete",
            ),
            Submitted::DroppedOversized => self.announce_once(
                &self.degraded,
                "warning: an event was too large for the log queue and was \
                 discarded; the session continues and the log is now incomplete",
            ),
            Submitted::WriterGone => self.announce_once(
                &self.stopped,
                "warning: the log writer has stopped; this session continues \
                 WITHOUT a log file",
            ),
            Submitted::WriterHung => self.announce_once(
                &self.stopped,
                "warning: the log writer stopped responding; this session \
                 continues WITHOUT a log file",
            ),
        }
    }
}

/// The only layer.
pub struct MagiLayer {
    file: FileSink,
    file_filter: crate::logging::filter::Filter,
    tui: Option<(TuiSink, tracing::Level)>,
    auditor: Arc<Auditor>,
    reporter: Reporter,
}

impl MagiLayer {
    /// Builds the layer with its file branch.
    #[must_use]
    pub fn new(
        file: FileSink,
        file_filter: crate::logging::filter::Filter,
        auditor: Arc<Auditor>,
        notices: Arc<dyn crate::logging::NoticeDelivery>,
    ) -> Self {
        Self {
            file,
            file_filter,
            tui: None,
            reporter: Reporter {
                sink: notices,
                auditor: Arc::clone(&auditor),
                degraded: std::sync::atomic::AtomicBool::new(false),
                stopped: std::sync::atomic::AtomicBool::new(false),
            },
            auditor,
        }
    }

    /// Attaches the screen branch and its level.
    ///
    /// **Never called in MS1**, and the parameter exists from MS1 anyway. What
    /// is reserved is the connection MECHANISM, not invented content: without
    /// it MS2 would have to change `new`'s signature, reopening the interface
    /// MS1 freezes with every call site behind it.
    ///
    /// It is called by `init_logging` **inside**, before the subscriber is
    /// installed — installation consumes the layer, so afterwards a
    /// `self -> Self` builder has nothing left to apply to.
    #[must_use]
    pub fn with_tui(mut self, tui: TuiSink, tui_level: tracing::Level) -> Self {
        self.tui = Some((tui, tui_level));
        self
    }

    /// The maximum of both branches' levels.
    ///
    /// Without this, `LevelFilter::current()` stays at `TRACE` and **every
    /// callsite in the dependency tree is enabled** — the cost of the static
    /// level hint is the whole reason it exists.
    fn max_level(&self) -> tracing::Level {
        let file = self.file_filter.max_level();
        match self.tui.as_ref() {
            Some((_, tui)) => file.max(*tui),
            None => file,
        }
    }
}

impl<S: Subscriber> Layer<S> for MagiLayer {
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(self.max_level().into())
    }

    /// The UNION of both branches.
    ///
    /// **Careful when a second layer appears:** `Layer::enabled` is evaluated
    /// globally, so `false` here disables the event for the whole subscriber.
    /// Harmless while `MagiLayer` is the only one.
    fn enabled(&self, meta: &tracing::Metadata<'_>, _: Context<'_, S>) -> bool {
        meta.level() <= &self.max_level()
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        // Stage 1: render. Nothing is escaped yet.
        let rendered = render_event(event);
        let reserved = rendered.len();
        let target = event.metadata().target();

        // Stage 2: audit, over the WHOLE line, before anything is cut.
        let (audited, alarm) = self.auditor.audit(
            &rendered,
            // `tracing` emits the target as a literal, so it is already
            // `'static` — the same reason `SecretName` is.
            leak_target(target),
            None,
            reserved,
        );

        // Stage 3: escape, then fan out.
        let escaped = audited.map_line(escape_for_line);
        // **What is reserved is what the writer will release.** `map_line`
        // above recomputed the measure from the escaped text -- which is what
        // the queue actually holds, and can be severalfold the rendered length
        // once control characters take their escaped form. Passing anything
        // else here would reserve one number and release another, which is the
        // leak that made the byte budget a lifetime quota.
        let reserved = escaped.reserved_len();
        let level = *event.metadata().level();

        if level <= self.file_filter.level_for(target) {
            let priority = if level <= tracing::Level::WARN {
                Priority::High
            } else {
                Priority::Low
            };
            // **The outcome is read, not discarded.** Every variant other than
            // `Queued` means the log is losing events or has stopped, and a
            // subsystem whose whole purpose is diagnosis must not be the one
            // thing that fails without saying so.
            self.reporter.report(self.file.appender.submit(
                Queued::Line(escaped.clone()),
                priority,
                reserved,
            ));
        }
        if let Some((tui, tui_level)) = self.tui.as_ref() {
            if level <= *tui_level {
                tui.sink
                    .deliver(&escaped.truncate_for_display(TUI_PAYLOAD_MAX_BYTES));
            }
        }
        if let Some(alarm) = alarm {
            // The alarm consults NO filter: exemption is from the filters, not
            // from congestion.
            self.reporter.report(self.file.appender.submit(
                Queued::Alarm(alarm),
                Priority::High,
                0,
            ));
        }
    }
}

/// Interns a target so it can live in a `SecretName`-shaped `'static` slot.
///
/// `tracing` builds each callsite's metadata as a `static`, so every target
/// **is** `'static` — but the borrow checker cannot see that through
/// `Metadata::target()`, which hands back a shorter lifetime. Interning is the
/// cheap way to recover it, and the set is bounded by the number of callsites in
/// the program, not by anything an input controls.
///
/// # Complexity
///
/// `O(log n)` over the distinct targets seen.
fn leak_target(target: &str) -> &'static str {
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<BTreeSet<&'static str>>> = Mutex::new(None);
    let Ok(mut guard) = SEEN.lock() else {
        return "magi_rs::unknown";
    };
    let set = guard.get_or_insert_with(BTreeSet::new);
    if let Some(found) = set.get(target) {
        return found;
    }
    let leaked: &'static str = Box::leak(target.to_string().into_boxed_str());
    set.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::auditor::SecretName;
    use tempfile::tempdir;

    /// Records what the reporter delivered.
    #[derive(Default)]
    struct RecordingSink {
        lines: std::sync::Mutex<Vec<String>>,
    }

    impl crate::logging::NoticeDelivery for RecordingSink {
        fn deliver(&self, line: &crate::logging::auditor::Audited) {
            if let Ok(mut l) = self.lines.lock() {
                l.push(line.as_str().to_string());
            }
        }
    }

    #[test]
    fn a_transient_notice_does_not_silence_the_permanent_one() {
        // With one latch for every outcome, a moment of congestion spoke first
        // and the writer's death was then silent for the rest of the run: the
        // operator was told events were being discarded and never told the log
        // had stopped. Those are different things to know -- one says the file
        // is incomplete, the other says there is no file -- and different
        // things to do about.
        let sink = Arc::new(RecordingSink::default());
        let reporter = Reporter {
            sink: sink.clone(),
            auditor: Arc::new(Auditor::new()),
            degraded: std::sync::atomic::AtomicBool::new(false),
            stopped: std::sync::atomic::AtomicBool::new(false),
        };

        // The transient one first, twice, so its own latch is proven to hold.
        reporter.report(Submitted::DroppedFull);
        reporter.report(Submitted::DroppedFull);
        // Then the permanent one.
        reporter.report(Submitted::WriterGone);
        reporter.report(Submitted::WriterHung);

        let said = sink.lines.lock().unwrap().clone();
        assert_eq!(
            said.iter().filter(|l| l.contains("discarded")).count(),
            1,
            "the transient tier must speak once: {said:?}"
        );
        assert_eq!(
            said.iter()
                .filter(|l| l.contains("WITHOUT a log file"))
                .count(),
            1,
            "and the permanent one must still be able to speak at all: {said:?}"
        );
    }

    #[test]
    fn the_level_hint_is_the_maximum_of_both_branches() {
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::new(crate::logging::DiscardDelivery),
        );
        assert_eq!(layer.max_level(), tracing::Level::INFO);

        // Without the maximum, `LevelFilter::current()` stays at TRACE and every
        // callsite in the dependency tree is enabled.
        let with_screen = layer.with_tui(
            TuiSink::new(Arc::new(crate::logging::DiscardDelivery)),
            tracing::Level::DEBUG,
        );
        assert_eq!(
            with_screen.max_level(),
            tracing::Level::DEBUG,
            "the more verbose branch decides the hint"
        );
    }

    #[test]
    fn a_target_interns_to_the_same_static_instead_of_leaking_per_event() {
        let a = leak_target("magi_rs::agent");
        let b = leak_target("magi_rs::agent");
        assert!(
            std::ptr::eq(a, b),
            "a per-event leak would grow without bound on a long run"
        );
        assert_ne!(leak_target("magi_rs::other"), a);
    }

    #[test]
    fn an_audited_line_truncated_for_display_is_still_an_audited() {
        // The screen cap is a TRANSFORMATION, never a second constructor: it
        // takes `self` by value, so it can only be applied to something that
        // already went through the auditor. If it produced a `String`, the TUI
        // branch would hold text outside the type that proves it was audited —
        // in the exact mouth that copies to the clipboard.
        const SECRET: &str = "a-live-secret-value";
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &[SECRET]);

        // **The secret STRADDLES the cap**, and that is the whole fixture. Put
        // it entirely past the cut and truncating first simply removes it, so
        // the test passes either way -- which it did, until this was run as a
        // mutation. Ten of its bytes land on the visible side.
        //
        // The padding carries SPACES, never one unbroken run: 64 KiB of the
        // same character is what `match_generic_secret_run` treats as a secret,
        // and redacting the padding would take the straddling half with it,
        // masking the very leak this test exists to observe. That masked it too.
        let straddle = 10;
        let unit = "filler word ";
        let pad = TUI_PAYLOAD_MAX_BYTES - straddle;
        let mut padding = unit.repeat(pad / unit.len() + 1);
        padding.truncate(pad);
        let long = format!("{padding}{SECRET}");
        let visible_half = SECRET.get(..straddle).expect("ascii");

        let (audited, _) = auditor.audit(&long, "magi_rs::tests", None, long.len());
        let shown = audited.truncate_for_display(TUI_PAYLOAD_MAX_BYTES);

        assert!(
            shown.as_str().len() <= TUI_PAYLOAD_MAX_BYTES + 128,
            "the cap bounds what reaches the screen"
        );
        assert!(
            !shown.as_str().contains(visible_half),
            "audited BEFORE the cut, so the half on the visible side is already \
             redacted. Cut first and the exact pass looks for the WHOLE value, \
             finds neither half, and this prefix ships"
        );
    }
}
