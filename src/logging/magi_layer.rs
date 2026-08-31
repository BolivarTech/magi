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

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use crate::logging::appender::{DailyAppender, Priority, Submitted};
use crate::logging::auditor::{Audited, Auditor, CauseKey, Queued};
use crate::logging::health::{render_transition, HealthTracker, Transition, HEALTH_TARGET};
use crate::logging::render::{escape_for_line, render_event};

/// Field name an emitter uses to declare the subsystem half of a [`CauseKey`]
/// (task 3.3's convention: `cause.subsystem = "…", cause.name = "…"`).
const CAUSE_SUBSYSTEM_FIELD: &str = "cause.subsystem";
/// Field name an emitter uses to declare the cause half of a [`CauseKey`].
const CAUSE_NAME_FIELD: &str = "cause.name";

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

/// Feeds the health tracker, and shows what it decides is worth showing.
///
/// **Held behind an `Arc` by two owners with different jobs.** The layer
/// observes, on whichever thread emitted; the [`LoggingHandle`] ticks the
/// window and flushes at close, on the thread that is ending the run. Neither
/// owns the other, and the handle deliberately exposes only `health_tick` and
/// `health_flush` rather than the tracker itself: handing out an
/// `Arc<Mutex<HealthTracker>>` would leave every caller deciding when to lock,
/// which is exactly the decision this repository already centralised for the
/// DEK.
///
/// [`LoggingHandle`]: crate::logging::LoggingHandle
pub(crate) struct HealthReporter {
    /// The tracker itself, which is `!Sync` by design — see its own rustdoc.
    tracker: Mutex<HealthTracker>,
    /// Where a transition is shown. **Not the screen BRANCH**: a transition is
    /// not an event and must not be filtered again, the same exemption the
    /// auditor's alarm has and for the same reason. This is the sink the
    /// caller built, which is the TUI's message channel in the terminal and
    /// stderr headless.
    sink: Arc<dyn crate::logging::NoticeDelivery>,
    /// The process auditor, shared with the layer that owns this reporter.
    auditor: Arc<Auditor>,
    /// Consulted only for its directory, to name the day's file.
    appender: Arc<DailyAppender>,
}

impl HealthReporter {
    /// Builds a reporter over an empty tracker.
    fn new(
        sink: Arc<dyn crate::logging::NoticeDelivery>,
        auditor: Arc<Auditor>,
        appender: Arc<DailyAppender>,
    ) -> Self {
        Self {
            tracker: Mutex::new(HealthTracker::new()),
            sink,
            auditor,
            appender,
        }
    }

    /// Feeds one audited event to the tracker, and shows what comes back.
    ///
    /// # Parameters
    ///
    /// * `line` — the audited event, which carries the emitter's cause key.
    /// * `level` — the event's level, which is the ONLY thing `ok` is derived
    ///   from. Deriving it from the text would be what R-L13 forbids for the
    ///   key itself: every HTTP 500 with a fresh request id would read as new.
    ///
    /// # Why the lock is taken only when there is a cause
    ///
    /// magi-core's 46 uninstrumented call sites carry no key and are most of
    /// the volume. Testing `cause` first trades one comparison for a mutex
    /// acquisition per foreign event on the hot path.
    ///
    /// # Complexity
    ///
    /// `O(1)` for an event with no cause; otherwise one lock plus the
    /// tracker's own amortised `O(1)`.
    fn observe(&self, line: &Audited, level: tracing::Level) {
        let Some(cause) = line.cause() else {
            return;
        };
        // ERROR and WARN are a failure; INFO and below carrying a key are the
        // success event recovery is detected from (task 3.3).
        let ok = level > tracing::Level::WARN;
        let decided = {
            let Ok(mut tracker) = self.tracker.lock() else {
                return;
            };
            tracker.observe(Some(cause), ok, Instant::now())
        };
        if let Some(transition) = decided {
            self.show(&transition);
        }
    }

    /// Expires the stability window without a new event, showing everything
    /// that has come due.
    ///
    /// # Why this LOOPS while [`Self::flush`] does not
    ///
    /// The asymmetry is in the pure tracker, not here: `tick` hands back one
    /// transition per call and `flush` hands back a `Vec`. So the wrapper has
    /// to supply the loop that `flush` gets for free. Without it a cascade —
    /// two subsystems whose windows elapse together — shows one now and holds
    /// the other until the next call, which in headless is a whole agent turn
    /// away and may be minutes of `consult`. Each pass consumes one pending,
    /// and pendings are bounded by the number of subsystems, so it terminates.
    ///
    /// # Complexity
    ///
    /// `O(s²)` worst case in the number of subsystems observed so far, which
    /// is a handful: `O(s)` per pass over at most `s` pendings.
    pub(crate) fn tick(&self, now: Instant) {
        while let Some(transition) = self.tracker.lock().ok().and_then(|mut t| t.tick(now)) {
            self.show(&transition);
        }
    }

    /// Shows every still-pending transition, window or no window (SC-L90).
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of subsystems observed so far.
    pub(crate) fn flush(&self) {
        let decided = {
            let Ok(mut tracker) = self.tracker.lock() else {
                return;
            };
            tracker.flush()
        };
        for transition in &decided {
            self.show(transition);
        }
    }

    /// Puts one transition on the screen.
    ///
    /// # Why an audited line and not a plain `String`
    ///
    /// The text carries RUNTIME data: REQ-L23's third part is the day's log
    /// path, composed from whatever the operator put in `log_dir` — which can
    /// hold a credential if somebody puts one there. The fixed words and the
    /// cause key have nothing to audit; the path does. The cause is passed as
    /// `None` on purpose: what leaves here is the SCREEN MESSAGE, not the
    /// event that caused it, and handing the key back would re-inject it into
    /// the tracker.
    ///
    /// # Complexity
    ///
    /// `O(k*n)` — the auditor's, over a short line.
    fn show(&self, transition: &Transition) {
        let text = render_transition(transition, &self.todays_log_path());
        // The alarm is discarded exactly as `Reporter::announce_once` discards
        // its own: this is the layer talking about itself, and the redaction
        // that matters has already happened inside `audit`.
        let (line, _alarm) = self.auditor.audit(&text, HEALTH_TARGET, None, text.len());
        self.sink.deliver(&line.map_line(escape_for_line));
    }

    /// The file today's events are being written to.
    ///
    /// Composed per delivery rather than cached, because a session that
    /// crosses midnight UTC rolls onto a new file and a cached path would send
    /// the reader to yesterday's.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn todays_log_path(&self) -> PathBuf {
        self.appender
            .dir()
            .join(crate::logging::rotation::file_name(
                time::OffsetDateTime::now_utc().date(),
            ))
    }
}

/// The only layer.
pub struct MagiLayer {
    file: FileSink,
    file_filter: crate::logging::filter::Filter,
    tui: Option<(TuiSink, tracing::Level)>,
    auditor: Arc<Auditor>,
    reporter: Reporter,
    health: Arc<HealthReporter>,
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
        let health = Arc::new(HealthReporter::new(
            Arc::clone(&notices),
            Arc::clone(&auditor),
            Arc::clone(&file.appender),
        ));
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
            health,
        }
    }

    /// The health reporter this layer feeds.
    ///
    /// `init_logging` takes a handle on it BEFORE installing the subscriber,
    /// because installation consumes the layer and the exit path still has to
    /// be able to flush.
    pub(crate) fn health(&self) -> Arc<HealthReporter> {
        Arc::clone(&self.health)
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
            cause_from_event(event),
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

        // **Health is observed here: after the audit, before the fan-out.**
        // This is the only point that sees EVERY event, including the ones the
        // screen filter is about to discard -- and whether a subsystem is
        // healthy must not depend on what happens to be displayed. It is also
        // where the `Audited` carrying the cause key already exists.
        self.health.observe(&escaped, level);

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

/// One interning cache: the leaked values [`intern`] has produced so far,
/// and whether its once-only capacity warning has already fired.
///
/// The warning is latched **per cache**, not globally — [`leak_target`]'s
/// cache and [`cause_from_event`]'s cache never share one flag, so one
/// reaching its cap cannot suppress the other's warning.
#[derive(Default)]
struct InternCache {
    /// Every distinct value interned into this cache so far.
    seen: BTreeSet<&'static str>,
    /// Whether the capacity warning already fired for this cache.
    warned: bool,
}

/// Interns `value` into a process-lifetime slice, deduplicated against
/// `cache`: a repeated value returns the same `'static` pointer instead of a
/// fresh leak. Past `max_entries` DISTINCT values, a new one is never
/// leaked — `fallback` is returned instead, and a warning fires once for
/// that cache (mirroring `MAX_TOOL_CALL_SLOTS` in `agent::provider`: cap and
/// warn, never an unbounded grow; latched rather than per-call because the
/// failure mode is high-frequency by nature, the same argument [`Reporter`]
/// makes for its own two latches above).
///
/// **`max_entries` is not one-size-fits-all — see the callers.**
/// [`leak_target`] passes `usize::MAX` (its input is compile-time bounded, so
/// there is nothing to cap); [`cause_from_event`] passes a real ceiling
/// (its input is runtime text an emitter chooses, which is not compile-time
/// bounded at all).
///
/// # Complexity
///
/// `O(log n)` over the distinct values already interned into `cache`.
fn intern(
    cache: &Mutex<Option<InternCache>>,
    value: &str,
    fallback: &'static str,
    max_entries: usize,
) -> &'static str {
    let Ok(mut guard) = cache.lock() else {
        return fallback;
    };
    let entry = guard.get_or_insert_with(InternCache::default);
    if let Some(found) = entry.seen.get(value) {
        return found;
    }
    if entry.seen.len() >= max_entries {
        if !entry.warned {
            entry.warned = true;
            eprintln!(
                "WARNING: an interning cache reached its cap of {max_entries} \
                 distinct values; further values collapse to {fallback:?} \
                 instead of leaking indefinitely"
            );
        }
        return fallback;
    }
    let leaked: &'static str = Box::leak(value.to_string().into_boxed_str());
    entry.seen.insert(leaked);
    leaked
}

/// Interns a target so it can live in a `SecretName`-shaped `'static` slot.
///
/// `tracing` builds each callsite's metadata as a `static`, so every target
/// **is** `'static` — but the borrow checker cannot see that through
/// `Metadata::target()`, which hands back a shorter lifetime. Uncapped: a
/// target is a module path fixed at COMPILE TIME by this program's own
/// `tracing::warn!`/`info!`/... call sites, so the set of distinct values is
/// bounded by the binary itself — nothing an attacker or a bug can grow at
/// runtime. Compare [`cause_from_event`], whose input is NOT compile-time
/// bounded and therefore does need a real cap.
///
/// # Complexity
///
/// `O(log n)` over the distinct targets seen.
fn leak_target(target: &str) -> &'static str {
    static SEEN: Mutex<Option<InternCache>> = Mutex::new(None);
    intern(&SEEN, target, "magi_rs::unknown", usize::MAX)
}

/// Upper bound on distinct `cause.subsystem`/`cause.name` values this
/// process will intern.
///
/// Unlike [`leak_target`]'s targets, a cause field is text the EMITTER
/// chooses at runtime (`cause.subsystem = "embedder"`), so nothing bounds
/// how many distinct values a buggy or hostile emitter could mint — restating
/// "the process bounds it" here would describe a hope, not enforce one. Past
/// this many distinct values, a new one degrades to
/// [`CAUSE_INTERN_CAP_REACHED`] instead of growing the cache further. Task
/// 3.3's own message table lists four causes total, so 256 is headroom for
/// legitimate growth, not a tuned measurement — the number only has to be
/// far enough past realistic use that it never fires in ordinary operation.
const MAX_INTERNED_CAUSE_VALUES: usize = 256;

/// Substituted for a cause-field value once [`MAX_INTERNED_CAUSE_VALUES`]
/// distinct values are already interned.
const CAUSE_INTERN_CAP_REACHED: &str = "cause-intern-cap-reached";

/// Reads the `cause.subsystem`/`cause.name` field pair off an event, when the
/// emitter declared both.
///
/// # Why "both, or none" (task 3.3's convention)
///
/// A `CauseKey` with an invented half would match neither the failure nor the
/// success event it was supposed to pair with, and the health tracker (MS2)
/// would read it as a cause that never recovers. Accepting one field alone
/// would force inventing the other, so an event carrying only one is treated
/// exactly like one carrying neither.
///
/// # Why this interns rather than borrows
///
/// [`Visit::record_str`] hands back a value bound to the call, even though
/// every field an emitter writes as `cause.subsystem = "embedder"` is a
/// `'static` string literal in practice — the trait's signature cannot see
/// that. [`CauseKey::new`] requires `&'static str`, so the value is interned
/// — capped at [`MAX_INTERNED_CAUSE_VALUES`], unlike [`leak_target`]'s
/// uncapped cache, because this input is runtime text an emitter chooses,
/// not a compile-time-bounded set of module paths.
///
/// # Only `&str`-valued fields are recognised
///
/// The `Visit` trait requires a `record_debug` method with no default body,
/// so this type still implements it — but its body does nothing for the two
/// cause field names. Task 3.3's only declared convention is a string
/// literal on both halves; special-casing a `Debug`-formatted value that
/// nothing today produces would be speculative surface with no consumer
/// (G2), and it would feed a wider, less predictable set of strings into the
/// capped cache above than the convention actually allows. If a future task
/// needs a non-string cause field, that task adds the handling with its
/// consumer.
///
/// # Complexity
///
/// `O(k)` over the event's field count, plus `O(log n)` per interned value.
fn cause_from_event(event: &Event<'_>) -> Option<CauseKey> {
    /// Collects the two cause fields, if present, off one event.
    #[derive(Default)]
    struct CauseFields {
        /// The `cause.subsystem` field's value, once seen.
        subsystem: Option<String>,
        /// The `cause.name` field's value, once seen.
        cause: Option<String>,
    }
    impl Visit for CauseFields {
        fn record_str(&mut self, field: &Field, value: &str) {
            match field.name() {
                CAUSE_SUBSYSTEM_FIELD => self.subsystem = Some(value.to_string()),
                CAUSE_NAME_FIELD => self.cause = Some(value.to_string()),
                _ => {}
            }
        }

        fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {
            // Deliberately empty — see this function's rustdoc: only
            // `&str`-valued cause fields are recognised.
        }
    }

    static SEEN: Mutex<Option<InternCache>> = Mutex::new(None);

    let mut fields = CauseFields::default();
    event.record(&mut fields);
    match (fields.subsystem, fields.cause) {
        (Some(subsystem), Some(cause)) => Some(CauseKey::new(
            intern(
                &SEEN,
                &subsystem,
                CAUSE_INTERN_CAP_REACHED,
                MAX_INTERNED_CAUSE_VALUES,
            ),
            intern(
                &SEEN,
                &cause,
                CAUSE_INTERN_CAP_REACHED,
                MAX_INTERNED_CAUSE_VALUES,
            ),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::auditor::SecretName;
    use tempfile::tempdir;
    use tracing_subscriber::layer::SubscriberExt as _;

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

    /// Task 0.1 fix-round Finding 2: distinct cause-field values past
    /// [`MAX_INTERNED_CAUSE_VALUES`] must never keep growing the cache — they
    /// collapse to [`CAUSE_INTERN_CAP_REACHED`] instead of leaking
    /// indefinitely.
    ///
    /// Drives real events through the real dispatcher (`cause_from_event`
    /// takes an `Event`, which has no public constructor outside one) with
    /// EVERY subsystem and cause value distinct, so each of the first
    /// `MAX_INTERNED_CAUSE_VALUES / 2` events consumes exactly two new cache
    /// slots. That makes the crossover exact rather than approximate: the
    /// cache fills to precisely the cap after that many events, and every
    /// event after it — however many more are sent — must keep coming back
    /// as the SAME fallback pair, which is the only way this test could
    /// observe "does not keep growing" without a private accessor into the
    /// cache's size.
    ///
    /// Isolated by relying on `cargo nextest`'s one-process-per-test model
    /// (documented elsewhere in this file's tests and in
    /// `tests/canary_both_mouths.rs`): the cache is a function-local
    /// `static`, so a second test sharing this process would share it too.
    #[test]
    fn interning_past_the_cap_falls_back_instead_of_growing_forever() {
        struct CaptureCause(Arc<Mutex<Vec<Option<CauseKey>>>>);
        impl<S: Subscriber> Layer<S> for CaptureCause {
            fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
                let key = cause_from_event(event);
                if let Ok(mut captured) = self.0.lock() {
                    captured.push(key);
                }
            }
        }

        let captured: Arc<Mutex<Vec<Option<CauseKey>>>> = Arc::new(Mutex::new(Vec::new()));
        let fills_the_cap = MAX_INTERNED_CAUSE_VALUES / 2;
        let past_the_cap = 40;

        let subscriber = tracing_subscriber::registry().with(CaptureCause(Arc::clone(&captured)));
        tracing::subscriber::with_default(subscriber, || {
            for i in 0..(fills_the_cap + past_the_cap) {
                let subsystem = format!("cause-cap-subsystem-{i}");
                let cause = format!("cause-cap-cause-{i}");
                tracing::warn!(
                    cause.subsystem = subsystem.as_str(),
                    cause.name = cause.as_str(),
                    "distinct cause value for the cap test"
                );
            }
        });

        let captured = captured.lock().expect("not poisoned");
        assert_eq!(captured.len(), fills_the_cap + past_the_cap);

        for (i, key) in captured[..fills_the_cap].iter().enumerate() {
            let key = key.expect("both fields were declared");
            assert_ne!(
                key.subsystem(),
                CAUSE_INTERN_CAP_REACHED,
                "event {i} is within the cap and must intern normally"
            );
        }

        let fallback = CauseKey::new(CAUSE_INTERN_CAP_REACHED, CAUSE_INTERN_CAP_REACHED);
        for (i, key) in captured[fills_the_cap..].iter().enumerate() {
            assert_eq!(
                *key,
                Some(fallback),
                "event {} is past the cap; a growing cache would keep minting \
                 distinct values instead of always returning the same fallback",
                fills_the_cap + i
            );
        }
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
