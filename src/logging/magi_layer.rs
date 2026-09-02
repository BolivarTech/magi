// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! One `tracing` layer, carrying both sinks with the level each is gated at.
//!
//! The two gates are **not** two filters: the file branch takes a configurable
//! [`crate::logging::filter::Filter`], the screen branch a fixed
//! [`crate::logging::SCREEN_LEVEL`]. Only the first is operator-settable.
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
use crate::logging::auditor::{AuditExempt, Audited, Auditor, CauseKey, Queued};
use crate::logging::health::{render_transition, HealthTracker, Transition, HEALTH_TARGET};
use crate::logging::render::{escape_for_line, escape_for_screen, render_event};

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
    /// Where an alarm raised by this reporter's OWN text goes.
    ///
    /// Not for the notices themselves — those go to `sink`, which consults no
    /// filter. It is here so that a redaction the auditor performs on this path
    /// is announced like any other, instead of being the one masking nobody is
    /// told about.
    appender: Arc<DailyAppender>,
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
        self.announce(text);
    }

    /// Announces `text`, audited, escaped, and with its alarm forwarded.
    ///
    /// # Parameters
    ///
    /// * `text` — what to say. The caller owns whether it is latched.
    ///
    /// # Why the alarm is FORWARDED rather than discarded
    ///
    /// The auditor's contract is "mask AND say so — both, never one". This path
    /// used to take the first half and drop the second, so a redaction here was
    /// invisible. The texts are ours and carry no credential today, which is
    /// exactly why it was easy to skip and why the next author who interpolates
    /// a path or an error would have found the one exempt path in the
    /// subsystem.
    ///
    /// # Why re-entering [`Self::report`] terminates
    ///
    /// A refused alarm submission is reported, and `report` announces through a
    /// latch, so the second pass finds its tier already spoken for and returns.
    /// The recursion is two deep at most, and no lock is held across it.
    ///
    /// # Complexity
    ///
    /// `O(k*n)` — the auditor's, over a short line.
    fn announce(&self, text: &str) {
        let (line, alarm) = self.auditor.audit(text, REPORTER_TARGET, None, text.len());
        // The SCREEN escaper: `sink` is a screen, and the file's doubling would
        // damage any path or error a future author interpolates here. And the
        // cap belongs to the MOUTH, not to the site: every other screen
        // delivery in this module applies it, and the one that did not is the
        // hole rather than the saving.
        self.sink.deliver(
            &line
                .map_line(escape_for_screen)
                .truncate_for_display(TUI_PAYLOAD_MAX_BYTES),
        );
        if let Some(alarm) = alarm {
            let outcome =
                self.appender
                    .submit(Queued::Alarm(alarm.clone()), Priority::High, NO_RESERVATION);
            settle_alarm(&self.auditor, &alarm, outcome);
            self.report(outcome);
        }
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

/// Target the subsystem's announcements about ITSELF are attributed to.
const REPORTER_TARGET: &str = "magi_rs::logging";

/// What an alarm reserves on the queue: nothing.
///
/// The alarm is exempt from the filters, not from the byte budget — it simply
/// has no rendered length to reserve, because the writer renders it.
///
/// `pub(super)` because `init_logging`'s already-initialised branch forwards an
/// alarm too, and a second `0` written out there would be the same rule in two
/// places with nothing tying them together.
pub(super) const NO_RESERVATION: usize = 0;

/// Gives an alarm's latch back when the queue refused it.
///
/// # Parameters
///
/// * `auditor` — the one that raised the alarm, and holds its latch.
/// * `alarm` — the finding that was submitted.
/// * `outcome` — what the appender did with it.
///
/// # Why the rule lives here rather than inline at the two call sites
///
/// The appender refuses a zero-byte priority submission only when its 2048
/// slots are exhausted or its writer has died, and neither is arrangeable in a
/// test. Taking the outcome as a parameter is what makes the RULE — "the latch
/// means delivered, never merely considered" — drivable, so the guard is a
/// behavioural test rather than a reading of two call sites.
///
/// # Complexity
///
/// `O(log n)`, the auditor's own.
///
/// `pub(super)` so `init_logging` uses this rule rather than restating it: it
/// is the fourth alarm-forwarding site in the subsystem, and the first three
/// all go through here.
pub(super) fn settle_alarm(auditor: &Auditor, alarm: &AuditExempt, outcome: Submitted) {
    if outcome != Submitted::Queued {
        auditor.retract_alarm(alarm);
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
        let decided = self.tracker().observe(Some(cause), ok, Instant::now());
        if let Some(transition) = decided {
            self.show(&transition);
        }
    }

    /// The tracker, **recovered rather than abandoned** if the lock is
    /// poisoned.
    ///
    /// # The invariant every caller owes: drop the guard before delivering
    ///
    /// Nothing this returns may still be alive when [`Self::show`] runs. Under
    /// `show` are an [`Auditor::audit`] pass over the whole line, which takes
    /// the auditor's own lock, and then `NoticeDelivery::deliver`, which in the
    /// terminal takes the sink's mutex, may take a second one and may
    /// `tokio::spawn`. Holding this across that nests three more locks and a
    /// channel send under a mutex that the emitting thread needs for every
    /// cause-bearing event, and opens a self-deadlock: a `tracing` event
    /// carrying `cause.*` emitted from anywhere beneath `show` re-enters
    /// [`Self::observe`] and blocks here forever, on a non-reentrant
    /// `std::sync::Mutex`, with the alternate screen held.
    ///
    /// So every call site binds the guard in a `let` STATEMENT, which drops it
    /// at the semicolon. This was stated as a property of the type before it
    /// was one: [`Self::tick`] drained the tracker with a `while let`, whose
    /// scrutinee temporary outlives the loop body, and the rule held for
    /// `observe` and `flush` alone. The guard that holds it for all three now
    /// is in this module's test list, under a name beginning "the tracker is
    /// unlocked while a ticked transition"; it probes the lock from inside a
    /// delivery, so a regression to a temporary shows up as a count rather
    /// than as a code review.
    ///
    /// # Why recovery and not a silent skip
    ///
    /// Returning early on a `PoisonError` — which is what all three call sites
    /// used to do — turns one panic anywhere in the process into health
    /// tracking that is dead for the rest of the run and never says so. That
    /// is a silent failure (G4) in the subsystem whose entire job is
    /// announcing failure, and it is the one failure this subsystem could
    /// never report, because the reporting is what would be broken.
    ///
    /// Recovery is safe here for a specific reason rather than a general one:
    /// [`HealthTracker`] is a plain state machine with **no panicking
    /// operation** — no indexing, no unwrap, no allocation-dependent
    /// invariant that a partial write could leave half-applied. A poisoned
    /// lock therefore means some *other* code panicked while holding this
    /// guard, and the state behind it is still structurally whole.
    ///
    /// # The consequence, stated
    ///
    /// If a future change gives the tracker an operation that CAN panic
    /// mid-update, this starts handing out a half-updated state machine
    /// instead of no state machine. The worst it can produce is a wrong
    /// screen notice — never a crash, and never a lost log line, because the
    /// file branch does not pass through here.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn tracker(&self) -> std::sync::MutexGuard<'_, HealthTracker> {
        self.tracker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
    ///
    /// # Why `loop` + `let ... else` and NOT `while let`
    ///
    /// The obvious spelling is `while let Some(t) = self.tracker().tick(now)`,
    /// and it is wrong: a temporary created in a `while let` scrutinee lives to
    /// the end of the LOOP BODY, so the guard [`Self::tracker`] returns is
    /// still held while [`Self::show`] runs its audit pass and calls into the
    /// sink. Binding it in a `let` statement drops it at the semicolon, which
    /// is the shape [`Self::observe`] and [`Self::flush`] already have — and
    /// the reason all three respect the invariant on [`Self::tracker`] rather
    /// than only two of them.
    pub(crate) fn tick(&self, now: Instant) {
        loop {
            let Some(transition) = self.tracker().tick(now) else {
                break;
            };
            self.show(&transition);
        }
    }

    /// Shows every still-pending transition, window or no window (SC-L90).
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of subsystems observed so far.
    pub(crate) fn flush(&self) {
        let decided = self.tracker().flush();
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
        // **The alarm is forwarded, not discarded**, and this is the path where
        // it matters most: the text carries the operator's own `log_dir`, so it
        // is the one self-referential line with runtime data in it. Masking
        // without announcing would be the auditor keeping half its contract in
        // the subsystem whose job is announcing.
        let (line, alarm) = self.auditor.audit(&text, HEALTH_TARGET, None, text.len());
        if let Some(alarm) = alarm {
            // **The outcome is settled but not reported, and the asymmetry with
            // `Reporter::announce` is deliberate.** No reporter is reachable
            // from here -- the layer owns one by value and this is held behind
            // an `Arc` by the exit path as well -- and reaching one would nest
            // an announcement inside a health transition. `settle_alarm` is
            // what keeps that honest: a refused alarm gives its latch back, so
            // the next transition raises it again rather than the finding being
            // lost.
            let outcome =
                self.appender
                    .submit(Queued::Alarm(alarm.clone()), Priority::High, NO_RESERVATION);
            settle_alarm(&self.auditor, &alarm, outcome);
        }
        self.sink.deliver(
            &line
                .map_line(escape_for_screen)
                .truncate_for_display(TUI_PAYLOAD_MAX_BYTES),
        );
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
    /// Where [`cause_from_event`] interns this layer's cause halves.
    ///
    /// **Owned by the layer rather than a function-local `static`**, and the
    /// difference is testability, not lifetime: a `static` is reachable only
    /// from whichever test happens to run first in the process, so "a causeless
    /// event interns nothing" could be asserted about a detached cache and
    /// never about the one `on_event` actually uses. One layer is installed per
    /// process, so this is the same single cache either way.
    cause_cache: Arc<Mutex<Option<InternCache>>>,
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
        // Taken before `file` is moved into the struct below: both the health
        // reporter and the plain one need somewhere to put an alarm raised by
        // their own text, and the file branch owns the only appender.
        let appender = Arc::clone(&file.appender);
        let health = Arc::new(HealthReporter::new(
            Arc::clone(&notices),
            Arc::clone(&auditor),
            Arc::clone(&appender),
        ));
        Self {
            file,
            file_filter,
            tui: None,
            reporter: Reporter {
                sink: notices,
                auditor: Arc::clone(&auditor),
                appender,
                degraded: std::sync::atomic::AtomicBool::new(false),
                stopped: std::sync::atomic::AtomicBool::new(false),
            },
            auditor,
            health,
            cause_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// The cache this layer interns cause-field halves into.
    ///
    /// Taken before installation, which consumes the layer.
    #[cfg(test)]
    fn cause_cache(&self) -> Arc<Mutex<Option<InternCache>>> {
        Arc::clone(&self.cause_cache)
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
        // **Annotated, because the annotation is the guard.** `Event::metadata`
        // hands back a `&'static Metadata<'static>`, so the target is ALREADY
        // `&'static str`: interning it manufactured a lifetime it already had,
        // and paid a global mutex per event to do it — on the one path every
        // event crosses, including the foreign call sites that carry no cause
        // and are the bulk of the volume. If a future `tracing` shortens either
        // lifetime this line stops compiling, rather than quietly needing the
        // mutex back.
        let target: &'static str = event.metadata().target();

        // Stage 2: audit, over the WHOLE line, before anything is cut.
        let (audited, alarm) = self.auditor.audit(
            &rendered,
            target,
            cause_from_event(&self.cause_cache, event, &self.reporter),
            reserved,
        );

        // Stage 3: escape, then fan out. **Each mouth gets its own escaper**,
        // so one audited copy is kept back before the file's is made. The file
        // is grepped and parsed, so it needs the backslash doubled; the screen
        // is read by a person, and there the doubling turns a path REQ-L23
        // means them to open into one that pastes nowhere. The health path was
        // given `escape_for_screen` for exactly this reason and the ordinary
        // screen branch was left on the file's.
        //
        // The clone is not new cost: the file branch used to clone for its own
        // submission, and now moves instead.
        //
        // **Made only when the screen branch will take it.** It used to be
        // unconditional, which meant a full copy of every audited line for a
        // branch that may not be wired at all — the headless surface wires
        // none — and, when it is, for the `INFO` and below that
        // [`SCREEN_LEVEL`] never admits. The predicate is exactly the one the
        // delivery below applies, so nothing about WHAT reaches a screen
        // changes.
        //
        // [`SCREEN_LEVEL`]: crate::logging::SCREEN_LEVEL
        let level = *event.metadata().level();
        let for_screen = self
            .tui
            .as_ref()
            .is_some_and(|(_, tui_level)| level <= *tui_level)
            .then(|| audited.clone());
        let escaped = audited.map_line(escape_for_line);
        // **What is reserved is what the writer will release.** `map_line`
        // above recomputed the measure from the escaped text -- which is what
        // the queue actually holds, and can be severalfold the rendered length
        // once control characters take their escaped form. Passing anything
        // else here would reserve one number and release another, which is the
        // leak that made the byte budget a lifetime quota.
        let reserved = escaped.reserved_len();

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
                Queued::Line(escaped),
                priority,
                reserved,
            ));
        }
        // The level test lives in the copy above, so `for_screen` is `Some`
        // only on the events that pass it; the `tui` arm here is just how the
        // sink is reached, not a second decision.
        //
        // **`reserved` above stays the FILE branch's measure** and is
        // deliberately not recomputed here: it is what the appender's
        // accounting reserves and what its writer gives back, and the screen
        // queues nothing.
        if let (Some(for_screen), Some((tui, _))) = (for_screen, self.tui.as_ref()) {
            tui.sink.deliver(
                &for_screen
                    .map_line(escape_for_screen)
                    .truncate_for_display(TUI_PAYLOAD_MAX_BYTES),
            );
        }
        if let Some(alarm) = alarm {
            // The alarm consults NO filter: exemption is from the filters, not
            // from congestion.
            let outcome = self.file.appender.submit(
                Queued::Alarm(alarm.clone()),
                Priority::High,
                NO_RESERVATION,
            );
            settle_alarm(&self.auditor, &alarm, outcome);
            self.reporter.report(outcome);
        }
    }
}

/// One interning cache: the leaked values [`intern`] has produced so far,
/// and whether its once-only capacity warning has already fired.
///
/// The warning is latched **per cache**, not globally, so a second cache added
/// later cannot have its warning suppressed by the first one reaching its cap.
/// [`cause_from_event`]'s is the only cache today.
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
/// **`max_entries` is a parameter and not a constant** because what bounds the
/// input is a property of the caller, not of the interning. Its one caller,
/// [`cause_from_event`], passes a real ceiling: the input is runtime text an
/// emitter chooses, so nothing else bounds it.
///
/// # Complexity
///
/// `O(log n)` over the distinct values already interned into `cache`.
fn intern(
    cache: &Mutex<Option<InternCache>>,
    value: &str,
    fallback: &'static str,
    max_entries: usize,
    reporter: &Reporter,
) -> &'static str {
    // **The decision is taken under the guard; the announcement is not.**
    // Speaking here cost three things at once: an I/O syscall while the mutex
    // every cause-bearing event needs was held; a panicking construct under
    // that mutex, which would poison the cache for the rest of the run; and a
    // raw `eprintln!`, which in the terminal writes over the alternate screen
    // the frame is drawn on. The latch is still SET inside, so two threads
    // crossing the cap together still produce one warning.
    let (interned, warn) = {
        let Ok(mut guard) = cache.lock() else {
            return fallback;
        };
        let entry = guard.get_or_insert_with(InternCache::default);
        if let Some(found) = entry.seen.get(value) {
            return found;
        }
        if entry.seen.len() >= max_entries {
            let first = !entry.warned;
            entry.warned = true;
            (fallback, first)
        } else {
            let leaked: &'static str = Box::leak(value.to_string().into_boxed_str());
            entry.seen.insert(leaked);
            (leaked, false)
        }
    };
    if warn {
        reporter.announce(&format!(
            "warning: an interning cache reached its cap of {max_entries} \
             distinct values; further values collapse to {fallback:?} instead \
             of leaking indefinitely"
        ));
    }
    interned
}

/// Upper bound on distinct `cause.subsystem`/`cause.name` values this
/// process will intern.
///
/// A cause field is text the EMITTER chooses at runtime
/// (`cause.subsystem = "embedder"`), so nothing bounds how many distinct
/// values a buggy or hostile emitter could mint — restating
/// "the process bounds it" here would describe a hope, not enforce one. Past
/// this many distinct values, a new one degrades to
/// [`CAUSE_INTERN_CAP_REACHED`] instead of growing the cache further. Task
/// 3.3's own message table lists four causes total, so 256 is headroom for
/// legitimate growth, not a tuned measurement — the number only has to be
/// far enough past realistic use that it never fires in ordinary operation.
///
/// # What firing costs, downstream
///
/// Past the cap, every further distinct value collapses onto
/// [`CAUSE_INTERN_CAP_REACHED`], and that key is what the health tracker
/// then receives. Because the tracker keys its state on the SUBSYSTEM half,
/// two unrelated subsystems that both land on the fallback share one health
/// state: one's degradation cancels the other's, and a recovery of either
/// reads as a recovery of both. The key also has no row in
/// `health::render_transition`'s message table, so whatever does survive
/// that reaches a screen as the no-row defect line rather than as a message.
/// Both are the intended trade — a bounded cache degrading loudly beats an
/// unbounded one — but neither is visible from this constant's own site.
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
/// — capped at [`MAX_INTERNED_CAUSE_VALUES`], because this input is runtime
/// text an emitter chooses. The event's TARGET needs none of this:
/// `Event::metadata` already hands back a `&'static Metadata<'static>`, so
/// interning it bought a lifetime it already had at the price of a mutex on
/// every event.
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
///
/// # Parameters
///
/// * `cache` — the interning cache, owned by the layer. **Reached only when
///   both halves are present**: an event without them returns before any lock
///   is taken, which is what keeps the mutex off the path of the call sites
///   that carry no key at all.
/// * `event` — the event whose fields are read.
/// * `reporter` — where the cache's once-only capacity warning is announced.
fn cause_from_event(
    cache: &Mutex<Option<InternCache>>,
    event: &Event<'_>,
    reporter: &Reporter,
) -> Option<CauseKey> {
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

    let mut fields = CauseFields::default();
    event.record(&mut fields);
    match (fields.subsystem, fields.cause) {
        (Some(subsystem), Some(cause)) => Some(CauseKey::new(
            intern(
                cache,
                &subsystem,
                CAUSE_INTERN_CAP_REACHED,
                MAX_INTERNED_CAUSE_VALUES,
                reporter,
            ),
            intern(
                cache,
                &cause,
                CAUSE_INTERN_CAP_REACHED,
                MAX_INTERNED_CAUSE_VALUES,
                reporter,
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

    /// Waits until today's log file contains `needle`, and returns what it
    /// held when the wait ended.
    ///
    /// **A condition with a generous failure deadline, never a fixed sleep.**
    /// Under the `heavy` nextest group a flat wait is a guess; the
    /// discriminating property is "the writer got to it at all", not "within
    /// 100 ms".
    fn wait_for_log(dir: &std::path::Path, needle: &str) -> String {
        let path = dir.join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let written = std::fs::read_to_string(&path).unwrap_or_default();
            if written.contains(needle) || std::time::Instant::now() >= deadline {
                return written;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// A phrase that appears VERBATIM in the `DroppedFull` notice.
    const PHRASE_IN_THE_REPORTER_NOTICE: &str = "being discarded";
    /// A phrase that appears VERBATIM in a `memory` degradation notice.
    const PHRASE_IN_THE_HEALTH_NOTICE: &str = "retrieval unavailable";

    #[test]
    fn a_reporter_notice_that_masks_a_secret_still_raises_the_alarm() {
        // The auditor's contract is "mask AND say so, both, never one". The
        // subsystem's own announcements took the first half and threw the
        // second away: `announce_once` discarded its alarm, so a redaction on
        // the one path that will grow runtime data was invisible.
        //
        // Driven through `Reporter::report`, not a hand-built line, so the
        // fixture cannot pass while the production path drops the alarm.
        assert!(
            PHRASE_IN_THE_REPORTER_NOTICE.len() >= crate::logging::auditor::MIN_SECRET_BYTES,
            "a phrase below the floor never enters the exact pass"
        );
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let auditor = Arc::new(Auditor::new());
        auditor.register_secret(SecretName::new("RPT"), &[PHRASE_IN_THE_REPORTER_NOTICE]);
        let sink = Arc::new(RecordingSink::default());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::clone(&auditor),
            sink.clone(),
        );

        layer.reporter.report(Submitted::DroppedFull);

        let said = sink.lines.lock().unwrap().join("\n");
        assert!(
            !said.contains(PHRASE_IN_THE_REPORTER_NOTICE),
            "the notice was not audited at all: {said:?}"
        );
        let written = wait_for_log(dir.path(), "SECURITY:");
        assert!(
            written.contains("SECURITY:") && written.contains("RPT"),
            "the reporter masked its own notice and told nobody: {written:?}"
        );
    }

    #[test]
    fn a_health_notice_that_masks_a_secret_still_raises_the_alarm() {
        // Same hole, on the path that ALREADY carries runtime data: a health
        // transition names the day's log file, composed from whatever the
        // operator put in `log_dir`. `show` audited it and dropped the alarm.
        assert!(
            PHRASE_IN_THE_HEALTH_NOTICE.len() >= crate::logging::auditor::MIN_SECRET_BYTES,
            "a phrase below the floor never enters the exact pass"
        );
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let auditor = Arc::new(Auditor::new());
        auditor.register_secret(SecretName::new("HLT"), &[PHRASE_IN_THE_HEALTH_NOTICE]);
        let sink = Arc::new(RecordingSink::default());
        let reporter = HealthReporter::new(sink.clone(), Arc::clone(&auditor), appender);

        let cause = CauseKey::new("embedder", "unreachable");
        let (event, _) = auditor.audit(
            "embedding request failed",
            "magi_rs::memory",
            Some(cause),
            0,
        );
        reporter.observe(&event, tracing::Level::WARN);

        let said = sink.lines.lock().unwrap().join("\n");
        assert!(
            !said.contains(PHRASE_IN_THE_HEALTH_NOTICE),
            "the transition was not audited at all: {said:?}"
        );
        let written = wait_for_log(dir.path(), "SECURITY:");
        assert!(
            written.contains("SECURITY:") && written.contains("HLT"),
            "the health path masked its own notice and told nobody: {written:?}"
        );
    }

    #[test]
    fn a_refused_alarm_gives_its_latch_back() {
        // `Auditor::alarm` deduplicates by `(secret, target)`: it inserts the
        // pair -- latching -- and only THEN hands the alarm to a caller that
        // has yet to submit it. A submission the appender refuses therefore
        // spends the one alarm that pair will ever produce on a queue entry
        // that never existed, and the operator is never told.
        //
        // The rule is tested with the outcome supplied, because the appender
        // refuses a zero-byte priority submission only when its 2048 slots are
        // exhausted or its writer has died -- neither of which a test can
        // arrange deterministically. The call sites pass this the outcome they
        // got.
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["a-registered-secret-value"]);
        let line = "url=https://x/a-registered-secret-value";

        // **One target per outcome, because the latch is keyed on
        // `(secret, target)`.** Reusing one target would make each assertion's
        // own `audit` call re-latch the pair and starve the next iteration --
        // and the obvious repair, retracting between rounds, would put the
        // mechanism under test into the fixture that sets it up.
        let cases = [
            (Submitted::DroppedFull, "magi_rs::case_full"),
            (Submitted::DroppedOversized, "magi_rs::case_oversized"),
            (Submitted::WriterGone, "magi_rs::case_gone"),
            (Submitted::WriterHung, "magi_rs::case_hung"),
        ];
        for (refused, target) in cases {
            let (_, alarm) = auditor.audit(line, target, None, 0);
            let alarm = alarm.unwrap_or_else(|| panic!("{refused:?}: the fixture raised no alarm"));
            settle_alarm(&auditor, &alarm, refused);
            assert!(
                auditor.audit(line, target, None, 0).1.is_some(),
                "{refused:?} burned the latch: the alarm is marked raised and \
                 nothing was ever queued"
            );
        }

        // And the other half: a submission that WAS accepted must still
        // deduplicate, or every line carrying the secret raises again.
        let accepted = "magi_rs::case_queued";
        let (_, alarm) = auditor.audit(line, accepted, None, 0);
        let alarm = alarm.expect("a target with no history alarms on first sight");
        settle_alarm(&auditor, &alarm, Submitted::Queued);
        assert!(
            auditor.audit(line, accepted, None, 0).1.is_none(),
            "an accepted alarm must keep its latch, or one secret floods the log"
        );
    }

    #[test]
    fn a_transient_notice_does_not_silence_the_permanent_one() {
        // With one latch for every outcome, a moment of congestion spoke first
        // and the writer's death was then silent for the rest of the run: the
        // operator was told events were being discarded and never told the log
        // had stopped. Those are different things to know -- one says the file
        // is incomplete, the other says there is no file -- and different
        // things to do about.
        let dir = tempdir().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let reporter = Reporter {
            sink: sink.clone(),
            auditor: Arc::new(Auditor::new()),
            appender: Arc::new(DailyAppender::new(dir.path()).unwrap()),
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
    fn a_per_target_override_survives_the_hint_and_still_binds_the_other_targets() {
        // `Filter::max_level` folds the per-target overrides into the hint, and
        // that fold is pinned only where it is DEFINED. Here is where it is
        // CONSUMED: the hint is a global pre-cut, so a fold that returned the
        // bare default would make `enabled` reject every event more verbose
        // than it -- and the override an operator wrote would be silently dead
        // before `level_for` ever got a chance to honour it.
        //
        // Load-bearing beyond the filter itself: `recovery_detection_is_off`
        // asks the same question of the same fold.
        //
        // Driven through the real dispatcher rather than by calling
        // `max_level()`, because a unit assertion on the fold stays green
        // through exactly the breakage this exists to catch.
        //
        // Mutation-verified: with `max_level` reading the filter's default
        // instead of the fold, the admitted event never reaches the file.
        const ADMITTED: &str = "the override admits this one";
        const EXCLUDED: &str = "the default still refuses this one";

        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            // ERROR everywhere, INFO for one target: the hint has to end at
            // INFO or the override cannot be reached at all.
            crate::logging::filter::Filter::parse("error,magi_rs::memory=info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::new(crate::logging::DiscardDelivery),
        );

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            // The excluded one FIRST. Both are INFO, so both take the same
            // priority queue and it is FIFO: once the second has arrived, the
            // first has had its chance, and the absence below is a decision
            // rather than a race with the writer thread.
            tracing::info!(target: "magi_rs::agent", "{EXCLUDED}");
            tracing::info!(target: "magi_rs::memory", "{ADMITTED}");
        });

        let written = wait_for_log(dir.path(), ADMITTED);
        assert!(
            written.contains(ADMITTED),
            "the per-target override never reached the file: the hint cut the \
             event off before `level_for` could honour it: {written:?}"
        );
        assert!(
            !written.contains(EXCLUDED),
            "the hint became the decision: raising it for ONE target admitted \
             every target, which is not what the operator wrote: {written:?}"
        );
    }

    #[test]
    fn health_tracking_continues_after_its_lock_is_poisoned() {
        // I3: `HealthReporter::tracker` recovers a `PoisonError` with
        // `into_inner` instead of returning early. The pattern, and the reason
        // for it, are already in this repository —
        // `system::database::tests::test_poisoned_lock_recovers_and_continues`
        // makes the identical claim about persistence: recover so the
        // subsystem keeps working, rather than fail closed for the session.
        //
        // **The assertion is that health tracking CONTINUES**, not that the
        // calls return without panicking. A skip-on-poison implementation also
        // returns without panicking; it just never shows anything again, which
        // is the silent failure this guards. Mutation-verified: with the three
        // call sites back on `let Ok(..) else { return; }` the sink comes back
        // EMPTY and this fails at 0 of 2.
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let shown = Arc::new(RecordingSink::default());
        let auditor = Arc::new(Auditor::new());
        let reporter = Arc::new(HealthReporter::new(
            shown.clone(),
            Arc::clone(&auditor),
            appender,
        ));

        // Poison it exactly as the precedent does: panic while holding it.
        let poisoner = Arc::clone(&reporter);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.tracker.lock().unwrap();
            panic!("intentional poison");
        })
        .join();
        assert!(
            reporter.tracker.is_poisoned(),
            "the fixture must actually poison the lock, or everything below \
             holds for free"
        );

        // One audited event per step, since `observe` takes what the layer
        // already built. `Auditor::audit` is the only constructor of `Audited`.
        let cause = CauseKey::new("embedder", "unreachable");
        let event = |text: &str| auditor.audit(text, "magi_rs::memory", Some(cause), 0).0;

        // 1. `observe` still works: a subsystem's first failure is immediate.
        reporter.observe(&event("embedding request failed"), tracing::Level::WARN);
        // 2. `tick` still works: the recovery is windowed, so expire it.
        reporter.observe(&event("embedding request ok"), tracing::Level::INFO);
        reporter.tick(Instant::now() + std::time::Duration::from_secs(WELL_PAST_THE_WINDOW_SECS));
        // 3. `flush` still works: degrade again, then leave a recovery pending.
        reporter.observe(&event("embedding request failed"), tracing::Level::WARN);
        reporter.observe(&event("embedding request ok"), tracing::Level::INFO);
        reporter.flush();

        let said = shown.lines.lock().unwrap().clone();
        let degradations = said
            .iter()
            .filter(|l| l.contains("memory: retrieval unavailable"))
            .count();
        let recoveries = said
            .iter()
            .filter(|l| l.contains("✓ memory: retrieval restored"))
            .count();
        assert_eq!(
            degradations, 2,
            "`observe` stopped reporting after the poison: {said:?}"
        );
        assert_eq!(
            recoveries, 2,
            "`tick` and `flush` must each still deliver their transition after \
             the poison; got {recoveries} of 2: {said:?}"
        );
    }

    /// Comfortably past [`crate::logging::health::HEALTH_MIN_STABLE_SECS`], so
    /// the windowed transition in the test above is unambiguously due.
    const WELL_PAST_THE_WINDOW_SECS: u64 = crate::logging::health::HEALTH_MIN_STABLE_SECS + 1;

    /// A delivery that reports whether the tracker was still locked while it
    /// ran.
    ///
    /// The `Weak` is the only way to ask: a sink is handed to
    /// [`HealthReporter::new`], so it cannot hold the reporter at construction
    /// and is pointed back at it afterwards. `try_lock` is the whole
    /// measurement — a `std::sync::Mutex` is not reentrant, so the thread that
    /// already holds the guard gets `WouldBlock` here exactly as a second
    /// thread would.
    #[derive(Default)]
    struct TrackerProbe {
        /// The reporter whose tracker this probes, set right after it exists.
        reporter: std::sync::OnceLock<std::sync::Weak<HealthReporter>>,
        /// How many deliveries found the tracker still locked.
        locked: std::sync::atomic::AtomicUsize,
        /// How many deliveries happened at all.
        delivered: std::sync::atomic::AtomicUsize,
    }

    impl crate::logging::NoticeDelivery for TrackerProbe {
        fn deliver(&self, _line: &Audited) {
            use std::sync::atomic::Ordering;
            self.delivered.fetch_add(1, Ordering::SeqCst);
            if let Some(reporter) = self.reporter.get().and_then(std::sync::Weak::upgrade) {
                if reporter.tracker.try_lock().is_err() {
                    self.locked.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }

    #[test]
    fn the_tracker_is_unlocked_while_a_ticked_transition_is_delivered() {
        // I1: `tick` used to drain the tracker with
        // `while let Some(t) = self.tracker().tick(now)`. A temporary created
        // in a `while let` scrutinee lives until the end of the LOOP BODY, so
        // the guard was still held while `show` ran -- a full `Auditor::audit`
        // pass, two allocations, and then `NoticeDelivery::deliver`, which in
        // the TUI takes the sink's own mutex, may take a second one and may
        // `tokio::spawn`. Three more locks and a channel send nested under the
        // first, on the thread that had just drawn the frame.
        //
        // The consequence today is contention: a worker thread emitting a
        // cause-bearing WARN blocks in `observe` for the whole delivery. The
        // consequence waiting to happen is a self-deadlock -- any `tracing`
        // event carrying `cause.*` emitted from under `show` re-enters
        // `on_event`, reaches `observe`, and blocks forever on a non-reentrant
        // mutex with the alternate screen held. Nothing but `observe`'s
        // `cause.is_none()` early return stands between the two, and that is a
        // guard on the PAYLOAD, not on the lock discipline.
        //
        // `observe` and `flush` bind the guard in a `let` statement and have
        // never had this, so the probe below watches both of them too: their
        // deliveries pass through the same counter, and a regression that
        // moved either onto a temporary would raise it as well.
        use std::sync::atomic::Ordering;

        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let auditor = Arc::new(Auditor::new());
        let probe = Arc::new(TrackerProbe::default());
        let reporter = Arc::new(HealthReporter::new(
            Arc::clone(&probe) as Arc<dyn crate::logging::NoticeDelivery>,
            Arc::clone(&auditor),
            Arc::clone(&appender),
        ));
        probe
            .reporter
            .set(Arc::downgrade(&reporter))
            .expect("the probe is pointed at its reporter exactly once");

        let cause = CauseKey::new("embedder", "unreachable");
        let event = |text: &str| auditor.audit(text, "magi_rs::memory", Some(cause), 0).0;

        // The first failure is shown immediately, by `observe`.
        reporter.observe(&event("embedding request failed"), tracing::Level::WARN);
        // The success behind it is windowed, so only a `tick` past the window
        // can drain it -- which is the path under test.
        reporter.observe(&event("embedding request ok"), tracing::Level::INFO);
        let before_tick = probe.delivered.load(Ordering::SeqCst);
        reporter.tick(Instant::now() + std::time::Duration::from_secs(WELL_PAST_THE_WINDOW_SECS));

        assert!(
            probe.delivered.load(Ordering::SeqCst) > before_tick,
            "the tick delivered nothing, so this measured no delivery at all"
        );

        // **And `flush`, which the paragraph above claimed to watch and this
        // test never called.** Its deliveries went through the same counter in
        // principle and through nothing in practice, so the sentence describing
        // the coverage was the only thing holding the third path down. `flush`
        // is the one that runs at close, on the thread that is ending the run
        // and still holds the alternate screen. Verified by mutation: with
        // `flush` binding its guard in a `let` that outlives the delivery, this
        // reports 1 of 4 deliveries locked.
        reporter.observe(&event("embedding request failed"), tracing::Level::WARN);
        reporter.observe(&event("embedding request ok"), tracing::Level::INFO);
        let before_flush = probe.delivered.load(Ordering::SeqCst);
        reporter.flush();
        assert!(
            probe.delivered.load(Ordering::SeqCst) > before_flush,
            "the flush delivered nothing, so it measured no delivery at all"
        );

        assert_eq!(
            probe.locked.load(Ordering::SeqCst),
            0,
            "the tracker was still locked during {} of {} deliveries: the \
             delivery path runs nested under the tracker's own mutex",
            probe.locked.load(Ordering::SeqCst),
            probe.delivered.load(Ordering::SeqCst)
        );
    }

    /// Builds a reporter over `sink`, with a real appender under `dir`.
    fn reporter_over(
        dir: &std::path::Path,
        sink: Arc<dyn crate::logging::NoticeDelivery>,
    ) -> Reporter {
        Reporter {
            sink,
            auditor: Arc::new(Auditor::new()),
            appender: Arc::new(DailyAppender::new(dir).unwrap()),
            degraded: std::sync::atomic::AtomicBool::new(false),
            stopped: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// A delivery that reports whether the interning cache was still locked
    /// while it ran.
    ///
    /// `try_lock` is the whole measurement: a `std::sync::Mutex` is not
    /// reentrant, so the thread that already holds the guard gets `WouldBlock`
    /// here exactly as a second thread would.
    struct CacheProbe {
        /// The cache this probes, shared with the caller of `intern`.
        cache: Arc<Mutex<Option<InternCache>>>,
        /// How many deliveries found the cache still locked.
        locked: std::sync::atomic::AtomicUsize,
        /// Everything delivered, in order.
        lines: Mutex<Vec<String>>,
    }

    impl crate::logging::NoticeDelivery for CacheProbe {
        fn deliver(&self, line: &Audited) {
            use std::sync::atomic::Ordering;
            if self.cache.try_lock().is_err() {
                self.locked.fetch_add(1, Ordering::SeqCst);
            }
            if let Ok(mut l) = self.lines.lock() {
                l.push(line.as_str().to_string());
            }
        }
    }

    #[test]
    fn the_interning_cap_warning_speaks_through_a_mouth_and_not_under_the_guard() {
        // Three defects in one `eprintln!`: it ran I/O while holding the
        // interning mutex, it put a panicking construct under that mutex --
        // poisoning the cache for the rest of the run -- and it wrote raw to
        // stderr, which in the TUI lands on top of the alternate screen the
        // frame is drawn on.
        //
        // Both halves are asserted, because either alone passes the wrong fix:
        // moving the write out from under the guard leaves it on stderr, and
        // routing it through a mouth while still holding the guard leaves the
        // lock nesting.
        let dir = tempdir().unwrap();
        let cache: Arc<Mutex<Option<InternCache>>> = Arc::new(Mutex::new(None));
        let probe = Arc::new(CacheProbe {
            cache: Arc::clone(&cache),
            locked: std::sync::atomic::AtomicUsize::new(0),
            lines: Mutex::new(Vec::new()),
        });
        let reporter = reporter_over(
            dir.path(),
            Arc::clone(&probe) as Arc<dyn crate::logging::NoticeDelivery>,
        );

        const FALLBACK: &str = "probe-cap-reached";
        assert_eq!(
            intern(&cache, "first-value", FALLBACK, 1, &reporter),
            "first-value",
            "the fixture must fill the cap before it can be exceeded"
        );
        assert_eq!(
            intern(&cache, "second-value", FALLBACK, 1, &reporter),
            FALLBACK,
            "past the cap a new value collapses onto the fallback"
        );
        // A THIRD, so the latch has something to hold back. With two calls the
        // count below is satisfied by a latch that works and by one that was
        // never consulted, because only one call was ever past the cap.
        assert_eq!(
            intern(&cache, "third-value", FALLBACK, 1, &reporter),
            FALLBACK,
            "and every further value keeps collapsing onto it"
        );

        let lines = probe.lines.lock().unwrap().clone();
        let said = lines.join("\n");
        // **Exactly one, never `>= 1`.** A latch that fires on every event past
        // the cap is as broken as one that never fires -- the failure mode is
        // high-frequency by nature, so an unlatched notice turns one problem
        // into the flood that hides it -- and an at-least-one assertion cannot
        // tell the two apart.
        assert_eq!(
            lines.iter().filter(|l| l.contains(FALLBACK)).count(),
            1,
            "the cap warning must be latched to exactly one announcement, and \
             must reach a mouth at all rather than stay a raw write to a \
             terminal the TUI may own: {said:?}"
        );
        assert_eq!(
            probe.locked.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the warning was delivered with the interning guard still held"
        );
    }

    #[test]
    fn a_poisoned_interning_cache_degrades_to_the_fallback_and_never_panics() {
        // The `let Ok(mut guard) = cache.lock() else` in `intern` was the one
        // degradation in this module with nothing driving it, so nothing said
        // whether the subsystem survives a poisoned cache or takes the process
        // down with it -- inside the layer that must never panic, on a path
        // every cause-bearing event runs.
        //
        // **It degrades rather than recovering, unlike `HealthReporter::
        // tracker`, and the asymmetry is not an oversight.** The tracker's
        // state is structurally whole after a foreign panic; this cache's
        // entries are `Box::leak`ed pointers, and handing back a fallback loses
        // nothing but the deduplication. The cost is stated where it bites: for
        // the rest of the run every cause value collapses onto one key, so the
        // health tracker merges the subsystems that land on it -- the same
        // consequence `MAX_INTERNED_CAUSE_VALUES` already documents for its own
        // cap.
        let dir = tempdir().unwrap();
        let reporter = reporter_over(dir.path(), Arc::new(crate::logging::DiscardDelivery));
        let cache: Arc<Mutex<Option<InternCache>>> = Arc::new(Mutex::new(None));

        // Poisoned exactly as the health-tracker precedent does it: panic while
        // holding the guard.
        let poisoner = Arc::clone(&cache);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("intentional poison");
        })
        .join();
        assert!(
            cache.is_poisoned(),
            "the fixture must actually poison the lock, or the assertion below \
             holds for free"
        );

        const FALLBACK: &str = "poisoned-cache-fallback";
        assert_eq!(
            intern(
                &cache,
                "a-value-worth-interning",
                FALLBACK,
                usize::MAX,
                &reporter
            ),
            FALLBACK,
            "a poisoned cache must degrade, not panic inside the layer"
        );
    }

    #[test]
    fn an_event_target_is_already_static_and_needs_no_interning() {
        // The layer used to intern the target through a global mutex, on the
        // reasoning that `Metadata::target()` hands back a shorter lifetime.
        // It does not: `Event::metadata` returns `&'static Metadata<'static>`,
        // so `target()` is `&'static str` and there was nothing to manufacture.
        //
        // A compile-level assertion, which is the strongest shape available
        // here: the binding below only type-checks while the property holds, so
        // a `tracing` upgrade that shortened either lifetime turns this red at
        // build time rather than letting the per-event mutex quietly return.
        struct CaptureTarget(Arc<Mutex<Vec<&'static str>>>);
        impl<S: Subscriber> Layer<S> for CaptureTarget {
            fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
                let target: &'static str = event.metadata().target();
                if let Ok(mut seen) = self.0.lock() {
                    seen.push(target);
                }
            }
        }

        let seen: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureTarget(Arc::clone(&seen)));
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(target: "magi_rs::agent", "an event with a fixed target");
        });

        assert_eq!(
            seen.lock().expect("not poisoned").as_slice(),
            ["magi_rs::agent"],
            "the target reached the layer unchanged, with no interning between"
        );
    }

    /// How many distinct values `cache` has interned so far.
    ///
    /// A cache still `None` was never even locked-and-initialised, and an empty
    /// one interned nothing; both are zero, which is what the guardian below
    /// means by "interns nothing".
    fn interned_values(cache: &Mutex<Option<InternCache>>) -> usize {
        cache
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.seen.len()))
            .unwrap_or(0)
    }

    #[test]
    fn an_event_with_no_cause_fields_interns_nothing() {
        // The layer took an interning mutex for EVERY event: the target was
        // interned before the cause was so much as looked at, and the call
        // sites that carry no cause are the bulk of the volume. A counting
        // probe rather than a lock probe, because "did not lock" and "did not
        // intern" are one question here, and a count also goes red if a future
        // author routes something else through the same cache.
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let layer = MagiLayer::new(
            FileSink::new(appender),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::new(crate::logging::DiscardDelivery),
        );
        // Taken before installation, which consumes the layer.
        let cache = layer.cause_cache();

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(detail = "no cause fields here", "a foreign call site");
            assert_eq!(
                interned_values(&cache),
                0,
                "a causeless event reached the interning cache"
            );
            tracing::warn!(
                cause.subsystem = "embedder",
                cause.name = "unreachable",
                "an instrumented call site"
            );
            assert_eq!(
                interned_values(&cache),
                2,
                "a cause-bearing event must intern exactly its two halves"
            );
        });
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
    /// Needs no runner-level isolation any more: the cache is the layer's,
    /// so this drives one it owns rather than a process-wide `static` a second
    /// test in the same process would share.
    #[test]
    fn interning_past_the_cap_falls_back_instead_of_growing_forever() {
        struct CaptureCause(
            Arc<Mutex<Vec<Option<CauseKey>>>>,
            Reporter,
            Mutex<Option<InternCache>>,
        );
        impl<S: Subscriber> Layer<S> for CaptureCause {
            fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
                let key = cause_from_event(&self.2, event, &self.1);
                if let Ok(mut captured) = self.0.lock() {
                    captured.push(key);
                }
            }
        }

        let dir = tempdir().unwrap();
        let captured: Arc<Mutex<Vec<Option<CauseKey>>>> = Arc::new(Mutex::new(Vec::new()));
        let fills_the_cap = MAX_INTERNED_CAUSE_VALUES / 2;
        let past_the_cap = 40;

        let subscriber = tracing_subscriber::registry().with(CaptureCause(
            Arc::clone(&captured),
            reporter_over(dir.path(), Arc::new(crate::logging::DiscardDelivery)),
            Mutex::new(None),
        ));
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

    /// A path shaped like the ones an operator actually reads off a screen.
    ///
    /// The separators are what discriminates: a message with no backslash in it
    /// renders identically under both escapers, so a fixture without one would
    /// pass whichever is wired.
    const A_WINDOWS_PATH: &str = r"C:\Users\op\logs\magi.log";

    #[test]
    fn the_screen_gets_screen_escaping_while_the_file_still_gets_the_file_kind() {
        // Both mouths shared one `escape_for_line`, which is the FILE escaper:
        // it doubles the backslash so the file stays parseable. On a screen that
        // doubling is pure damage -- a Windows path arrives as
        // `C:\\Users\\...`, which opens nothing and pastes nowhere -- and the
        // health path had already been given its own escaper for exactly this
        // reason while the ordinary screen branch was left behind.
        //
        // Driven through the real dispatcher, because a hand-built line would
        // exercise the renderer and stay green through the very swap this
        // exists to catch.
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let screen = Arc::new(RecordingSink::default());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::new(crate::logging::DiscardDelivery),
        )
        .with_tui(
            TuiSink::new(Arc::clone(&screen) as Arc<dyn crate::logging::NoticeDelivery>),
            tracing::Level::WARN,
        );

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(target: "magi_rs::logging", "the log is at {A_WINDOWS_PATH}");
        });

        let shown = screen.lines.lock().unwrap().join("\n");
        assert!(
            shown.contains("the log is at"),
            "the fixture reached the screen branch not at all: {shown:?}"
        );
        assert!(
            shown.contains(A_WINDOWS_PATH),
            "the screen carries file escaping: a path shown like this cannot be \
             pasted anywhere: {shown:?}"
        );

        let written = wait_for_log(dir.path(), "the log is at");
        assert!(
            written.contains(r"C:\\Users\\op\\logs\\magi.log"),
            "the FILE must keep its own escaping -- it is grepped and parsed, \
             and a lone backslash there is ambiguous: {written:?}"
        );

        // And the budget still follows the FILE branch's measure: the writer
        // releases what the line carries, so reserving anything else makes the
        // channel's counter drift until it refuses events with capacity spare.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while appender.reserved(Priority::High) != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(
            appender.reserved(Priority::High),
            0,
            "reserved and released disagree, so the two branches' measures were \
             crossed"
        );
    }

    /// A secret with NO recognisable shape, and the canary depends on that.
    ///
    /// A key-shaped value, or one sitting in a URL authority, is masked by pass
    /// 1 alone — so a canary built from one stays green against an auditor that
    /// knows nothing this process registered, which is precisely the defect
    /// this gate found six times in six files. Only registration finds this
    /// one, so only the exact pass can produce a green here.
    const SHAPELESS: &str = "wooden table lamp shade";

    #[test]
    fn the_reporters_own_announcement_is_capped_like_every_other_screen_line() {
        // The cap was applied per SITE rather than at the mouth: `show` and
        // `on_event`'s screen branch capped, `Reporter::announce` did not. All
        // three deliver to the same sink, which in the terminal is a message
        // list that caps nothing and copies to the clipboard with one
        // keystroke, so the odd one out is a hole rather than a saving.
        //
        // Today's texts are literals, which is why nothing had noticed. The
        // contract on `announce` already invites the next one not to be:
        // "anyone who interpolates a path or an error into it owes the
        // forwarding", and a path or an error is exactly what has no bound.
        let dir = tempdir().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let reporter = reporter_over(
            dir.path(),
            Arc::clone(&sink) as Arc<dyn crate::logging::NoticeDelivery>,
        );

        reporter.announce(&filler(TUI_PAYLOAD_MAX_BYTES * 2));

        let said = sink.lines.lock().unwrap().join("\n");
        assert!(
            said.contains("filler word"),
            "the fixture announced nothing: {}",
            said.len()
        );
        assert!(
            said.len() <= TUI_PAYLOAD_MAX_BYTES + 128,
            "the subsystem's own announcement reaches a screen uncapped: it \
             was handed {} bytes",
            said.len()
        );
    }

    #[test]
    fn the_alarm_the_file_mouth_writes_names_the_secret_and_never_its_value() {
        // **The appender's alarm arm is the one write in this subsystem that
        // reaches a mouth without passing the auditor**, and following it is
        // what this test records. It is exempt by construction rather than by
        // omission: what it writes is the auditor's OWN finding, whose type
        // carries a `SecretName` and a target and nothing else (REQ-L50), and
        // auditing it would mask the very name the alarm exists to publish.
        //
        // So the property is pinned from the outside instead: with a shapeless
        // secret registered, the file mouth must end up holding the alarm, the
        // name, and no occurrence of the value -- on the ordinary line or on
        // the alarm line.
        //
        // Mutation-verified twice. Building the layer on a fresh
        // `Auditor::new()` -- the six-times defect -- ships the value on the
        // ordinary line; deleting the alarm submission from `on_event` leaves
        // the masking done and unannounced, which is the auditor keeping half
        // its contract.
        assert!(
            SHAPELESS.len() >= crate::logging::auditor::MIN_SECRET_BYTES,
            "a value below the floor never enters the exact pass, so this \
             would hold for free"
        );
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let auditor = Arc::new(Auditor::new());
        let name = SecretName::new("SHAPELESS_CANARY");
        auditor.register_secret(name, &[SHAPELESS]);
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::clone(&auditor),
            Arc::new(crate::logging::DiscardDelivery),
        );

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "magi_rs::tests", "configured with {SHAPELESS}");
        });

        let written = wait_for_log(dir.path(), "SECURITY:");
        assert!(
            written.contains("configured with"),
            "the fixture reached the file mouth not at all: {written:?}"
        );
        assert!(
            written.contains("SECURITY:"),
            "a value was masked and the file was never told: {written:?}"
        );
        assert!(
            written.contains(name.as_str()),
            "the alarm names no secret, so it sends a reader nowhere: \
             {written:?}"
        );
        assert!(
            !written.contains(SHAPELESS),
            "the shapeless secret reached the file mouth: only the exact pass \
             can catch this one, so this is the assertion that proves the \
             auditor ran on this branch at all: {written:?}"
        );
    }

    #[test]
    fn the_submission_outcome_reaches_the_reporter_from_on_event() {
        // The RULE -- which outcome deserves which notice -- is pinned by
        // `a_transient_notice_does_not_silence_the_permanent_one`, which calls
        // `Reporter::report` directly. What nothing pinned is the WIRING:
        // that `on_event` hands `submit`'s answer to it. Replace the call with
        // `let _ = ...submit(..)` and every notice test stays green while the
        // subsystem whose whole purpose is diagnosis becomes the one thing in
        // the process that fails without saying so.
        //
        // **Drivable after all, and this is the closure Caspar asked for.**
        // The `DroppedFull`, `WriterGone` and `WriterHung` outcomes need 2048
        // exhausted slots or a dead writer, neither of which a test can
        // arrange; `DroppedOversized` needs only an event larger than the
        // channel's byte budget, which is arithmetic. So the wiring is driven
        // through the real dispatcher rather than recorded as a residual.
        //
        // `INFO` puts it on the ORDINARY channel, whose budget is a third of
        // the priority one's -- the cheapest event that can be refused.
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let notices = Arc::new(RecordingSink::default());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::clone(&notices) as Arc<dyn crate::logging::NoticeDelivery>,
        );

        let payload = filler(crate::logging::appender::LOG_CHANNEL_LOW_BYTES + 1024);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "magi_rs::tests", "{payload}");
        });

        let said = notices.lines.lock().unwrap().join("\n");
        assert!(
            said.contains("too large"),
            "the appender refused the event and `on_event` threw the refusal \
             away: the log is losing events and nobody is told: {}",
            said.len()
        );
        assert!(
            said.contains("the log is now incomplete"),
            "the refusal was reported as something other than a discard: {said}"
        );
    }

    /// Prose of `bytes` length, never one unbroken run of a character.
    ///
    /// A long run of the same byte is what `match_generic_secret_run` treats as
    /// a secret, so padding built that way arrives redacted and every length
    /// assertion downstream measures the redaction instead of the payload.
    fn filler(bytes: usize) -> String {
        let unit = "filler word ";
        let mut padding = unit.repeat(bytes / unit.len() + 1);
        padding.truncate(bytes);
        padding
    }

    #[test]
    fn an_oversized_event_is_cut_by_the_screen_branch_and_not_by_the_file() {
        // The truncation RULE is pinned next door, in
        // `an_audited_line_truncated_for_display_is_still_an_audited`, which
        // calls `truncate_for_display` itself. What nothing pinned is the
        // WIRING: that `on_event`'s screen branch applies it. Delete the call
        // there and the rule stays green while a 100 MiB `ERROR` reaches the
        // TUI's message list -- which caps nothing -- and from there the
        // clipboard with one keystroke.
        //
        // Both directions are asserted, because either alone passes a wrong
        // fix: capping the whole pipeline would satisfy the screen assertion
        // while silently amputating the file, which is the one mouth that must
        // hold the event entire.
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());
        let screen = Arc::new(RecordingSink::default());
        let layer = MagiLayer::new(
            FileSink::new(Arc::clone(&appender)),
            crate::logging::filter::Filter::parse("info").expect("valid"),
            Arc::new(Auditor::new()),
            Arc::new(crate::logging::DiscardDelivery),
        )
        .with_tui(
            TuiSink::new(Arc::clone(&screen) as Arc<dyn crate::logging::NoticeDelivery>),
            tracing::Level::WARN,
        );

        // Comfortably past the cap, and a tail that only survives uncut.
        const TAIL: &str = "the-tail-only-an-uncut-line-carries";
        let payload = format!("{}{TAIL}", filler(TUI_PAYLOAD_MAX_BYTES * 2));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(target: "magi_rs::logging", "oversized {payload}");
        });

        let shown = screen.lines.lock().unwrap().join("\n");
        assert!(
            shown.contains("oversized"),
            "the fixture reached the screen branch not at all: {}",
            shown.len()
        );
        assert!(
            shown.len() <= TUI_PAYLOAD_MAX_BYTES + 128,
            "the screen branch applied no cap: it was handed {} bytes",
            shown.len()
        );
        assert!(
            !shown.contains(TAIL),
            "the tail of an oversized event reached the screen, so nothing cut it"
        );

        // And the FILE keeps the whole event: the cap is a screen policy, not a
        // pipeline one. **Measured in bytes, not by grepping for the tail**,
        // because the file mouth chunks at `chunk::MAX_LINE_BYTES` and a
        // continuation header lands wherever the arithmetic puts it -- which
        // may be inside any given marker. Length survives that; a needle does
        // not, and a needle that happens to straddle a boundary would report a
        // truncation that never happened.
        let written = wait_for_log(dir.path(), "oversized");
        assert!(
            written.contains("oversized"),
            "the fixture reached the file mouth not at all"
        );
        assert!(
            written.len() > payload.len(),
            "the file holds {} bytes of a {}-byte event: the cut happened \
             before the fan-out, so the one mouth that must carry the event \
             entire was amputated by a screen policy",
            written.len(),
            payload.len()
        );
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
        let long = format!("{}{SECRET}", filler(TUI_PAYLOAD_MAX_BYTES - straddle));
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
