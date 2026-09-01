// Author: Julian Bolivar
// Version: 0.18.1
// Date: 2026-08-31

//! Transition-only health tracking (D-L13, SC-L14...SC-L119).
//!
//! This module decides *when* a subsystem's health is worth putting on
//! screen, and *what* that says. It is pure: it emits nothing, touches no
//! filesystem and knows no TUI. `observe` takes whatever the caller already
//! knows about one event and returns a [`Transition`] only when that
//! transition is worth showing; everything else is silence by design, because
//! "everything else" is exactly the noise this feature exists to remove.
//!
//! # What lives here and what does not
//!
//! [`HealthTracker`] holds the state machine; [`render_transition`] holds the
//! message table, because a `CauseKey` and what it means to a user are two
//! halves of one piece of knowledge and splitting them across a module
//! boundary is how they drift apart. Both are functions of their arguments:
//! the day's log path is **received**, not looked up, and `Instant` is
//! **passed in**, not read.
//!
//! Delivery is somebody else's job. The layer is what feeds this and what
//! puts the result on screen, so nothing here decides which mouth a
//! transition reaches or when the window is expired.

use std::path::Path;
use std::time::{Duration, Instant};

use tracing::Level;

use crate::logging::auditor::CauseKey;
use crate::logging::filter::Filter;

/// How long a state has to hold before its transition reaches the screen
/// (R-L13d). **Chosen: 30 s, not measured.**
pub const HEALTH_MIN_STABLE_SECS: u64 = 30;

/// A health change worth showing on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// A subsystem went from healthy to degraded, or changed cause while
    /// already degraded.
    Degraded(CauseKey),
    /// A subsystem returned to healthy.
    Restored(CauseKey),
}

/// Health state of a single subsystem. `None` means healthy; `Some(cause)`
/// means degraded, with the cause currently in force.
type HealthState = Option<CauseKey>;

/// A candidate transition that has been observed but has not yet held for
/// [`HEALTH_MIN_STABLE_SECS`].
#[derive(Debug, Clone)]
struct PendingTransition {
    /// The transition to emit once the window elapses.
    transition: Transition,
    /// The health state this transition moves the subsystem to.
    target: HealthState,
    /// When the candidate state was first observed.
    since: Instant,
}

/// Per-subsystem bookkeeping: the state last shown on screen, and any
/// candidate transition still serving its window.
#[derive(Debug, Clone, Default)]
struct SubsystemState {
    /// The health state last actually reported to the caller.
    shown: HealthState,
    /// A transition holding for the stability window, if any.
    pending: Option<PendingTransition>,
}

/// Decides when a degradation or recovery is worth showing on screen.
///
/// State is kept **per subsystem** ([`CauseKey::subsystem`]), never per
/// cause: two causes of one subsystem share a single health state, with the
/// cause riding along to say why. This is what lets a subsystem's first
/// failure show immediately (SC-L71) while a later cause change inside the
/// same subsystem still serves the window (SC-L76).
///
/// Not thread-safe by itself, deliberately: `observe`, `tick` and `flush`
/// all take `&mut self`; whoever shares one instance across threads supplies
/// its own `Mutex`, the same trade-off `SqliteVectorStore` already makes in
/// this repository.
///
/// # Examples
///
/// ```
/// use std::time::Instant;
/// use magi_rs::logging::auditor::CauseKey;
/// use magi_rs::logging::health::{HealthTracker, Transition};
///
/// let mut tracker = HealthTracker::new();
/// let cause = CauseKey::new("embedder", "unreachable");
///
/// // A subsystem's first-ever failure is shown right away (SC-L71).
/// let first = tracker.observe(Some(cause), false, Instant::now());
/// assert!(matches!(first, Some(Transition::Degraded(_))));
///
/// // A repeat of the same failure is not news: nothing more to show.
/// assert!(tracker.observe(Some(cause), false, Instant::now()).is_none());
/// ```
#[derive(Debug, Clone, Default)]
pub struct HealthTracker {
    /// One entry per subsystem, in the order the subsystems were first
    /// OBSERVED -- which is what gives [`Self::flush`] its order. An entry is
    /// created by the first event naming that subsystem whatever the event
    /// said, so this is not an order of degradation: a subsystem whose first
    /// event was a success still ranks ahead of one that degraded later.
    states: Vec<(&'static str, SubsystemState)>,
}

impl HealthTracker {
    /// Builds an empty tracker: every subsystem starts healthy and unseen.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds a subsystem's bookkeeping, creating it (healthy, unseen) the
    /// first time it is referenced.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of distinct subsystems seen so far: a linear scan
    /// of `states`, every call. `s` is a handful (embedder, provider,
    /// vault, ...), which is why a scan is the right structure here -- and it
    /// is what gives [`Self::flush`] its first-observed order for free. It is
    /// linear all the same, not amortised constant.
    fn state_mut(&mut self, subsystem: &'static str) -> &mut SubsystemState {
        match self.states.iter().position(|(name, _)| *name == subsystem) {
            Some(idx) => &mut self.states[idx].1,
            None => {
                self.states.push((subsystem, SubsystemState::default()));
                let last = self.states.len() - 1;
                &mut self.states[last].1
            }
        }
    }

    /// Observes one event. Returns a transition only when the new state has
    /// already held for [`HEALTH_MIN_STABLE_SECS`] (R-L13d) -- **except** a
    /// subsystem's very first degradation, which is immediate (SC-L71).
    ///
    /// `cause` is `None` for the call sites that carry no cause key; such an
    /// event is ignored rather than keyed off its text (R-L13). `ok` is the
    /// caller's own classification of the event's level -- this function
    /// never inspects text to derive either value.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of distinct subsystems seen so far: the lookup
    /// into `states` is a linear scan, run on every call, and everything
    /// after it is constant time. `s` is a handful, so the scan stays
    /// cheaper than the map that would replace it.
    pub fn observe(
        &mut self,
        cause: Option<CauseKey>,
        ok: bool,
        now: Instant,
    ) -> Option<Transition> {
        let cause = cause?;
        let candidate: HealthState = if ok { None } else { Some(cause) };
        let state = self.state_mut(cause.subsystem());

        // Rule: same state as what is shown -- nothing new to report, and
        // any transition that had been waiting on a DIFFERENT candidate is
        // no longer relevant.
        if candidate == state.shown {
            state.pending = None;
            return None;
        }

        // Rule (SC-L71): a subsystem's first-ever degradation is immediate.
        if state.shown.is_none() && candidate.is_some() {
            let transition = Transition::Degraded(cause);
            state.shown = candidate;
            // **A structural no-op, kept as a statement of the invariant.**
            // Reaching this arm with a pending is unreachable, and the proof is
            // short enough to write down: a pending is only ever created in the
            // final arm below, which needs `candidate != shown` and NOT
            // (`shown.is_none() && candidate.is_some()`) -- and since
            // `candidate != shown`, a `shown` of `None` forces `candidate` to be
            // `Some`, which is this arm. So a pending exists only while `shown`
            // is `Some`. `shown` returns to `None` only in `tick` and `flush`,
            // and both `take()` the pending in the same step without putting it
            // back. Hence `shown.is_none()` implies `pending.is_none()`, which
            // is exactly this arm's guard.
            //
            // It is therefore deletable with the suite green, and deleting it is
            // still the wrong move: what the line says is that an immediate
            // emission leaves nothing waiting. Were the arm ever widened -- a
            // second immediate case, a `shown` written somewhere else -- the
            // reader would need that decided here, and an absent line decides
            // nothing. A test cannot hold this down either: no input reaches the
            // arm with a pending, so any test would pass with or without it.
            state.pending = None;
            return Some(transition);
        }

        // Everything else -- a recovery, or a cause change inside an
        // already-degraded subsystem -- serves the window.
        let transition = match candidate {
            None => Transition::Restored(cause),
            Some(_) => Transition::Degraded(cause),
        };

        match &state.pending {
            // Already pending toward this exact candidate: keep its
            // original `since` rather than restarting the window.
            Some(pending) if pending.target == candidate => {}
            _ => {
                state.pending = Some(PendingTransition {
                    transition,
                    target: candidate,
                    since: now,
                });
            }
        }

        None
    }

    /// Expires the window without a new event. The caller invokes this
    /// periodically -- the TUI event loop's own `poll` timeout, or once per
    /// agent turn in headless mode -- without it, a pending transition whose
    /// subsystem stops producing events -- exactly what happens when
    /// something goes fully dark -- would never be emitted. What runs at
    /// shutdown is [`Self::flush`], not this: `tick` still respects the
    /// window and a closing process has no "later" left for the window to
    /// elapse in.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of distinct subsystems observed so far.
    pub fn tick(&mut self, now: Instant) -> Option<Transition> {
        let window = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        for (_, state) in &mut self.states {
            if let Some(pending) = state.pending.take() {
                if now.saturating_duration_since(pending.since) >= window {
                    state.shown = pending.target;
                    return Some(pending.transition);
                }
                state.pending = Some(pending);
            }
        }
        None
    }

    /// Emits every pending transition, ignoring whether its window has
    /// elapsed. Used at shutdown (SC-L90): a short headless run would
    /// otherwise end without ever showing its only pending signal.
    ///
    /// Returns a `Vec`, not an `Option`, because state is per subsystem: a
    /// cascading failure can leave one pending transition per subsystem,
    /// and an `Option` would show one and silently drop the rest.
    ///
    /// The order is **first-observed**: subsystems come out in the order this
    /// tracker first saw an event naming each of them, degraded or not. That
    /// is not the same as earliest-degradation-first, and nothing here sorts
    /// by when a pending transition started serving its window.
    ///
    /// # Complexity
    ///
    /// `O(s)` in the number of distinct subsystems observed so far.
    pub fn flush(&mut self) -> Vec<Transition> {
        let mut out = Vec::new();
        for (_, state) in &mut self.states {
            if let Some(pending) = state.pending.take() {
                state.shown = pending.target;
                out.push(pending.transition);
            }
        }
        out
    }
}

/// Target every line this module puts on screen is attributed to.
///
/// Fixed rather than inherited from the event that caused it: the alarm path
/// carries a target so an operator knows where to go look, and a transition's
/// origin is this module, not the subsystem it is reporting on.
///
/// `pub(crate)` because its only consumers are the layer and `warn_if_
/// recovery_detection_is_off`, both inside this crate — a public constant
/// nothing outside can use is surface without a consumer (G2).
pub(crate) const HEALTH_TARGET: &str = "magi_rs::logging::health";

/// The subsystem half of the causes the message table below declares.
const SUBSYSTEM_EMBEDDER: &str = "embedder";
/// The subsystem half of the provider's causes.
const SUBSYSTEM_PROVIDER: &str = "provider";
/// The subsystem half of the vault's causes.
const SUBSYSTEM_VAULT: &str = "vault";
/// A subsystem that could not be reached at all.
const CAUSE_UNREACHABLE: &str = "unreachable";
/// A subsystem that answered, badly.
const CAUSE_HTTP_ERROR: &str = "http_error";
/// A vault that is closed.
const CAUSE_LOCKED: &str = "locked";

/// How [`render_transition`] opens when a cause has no row below.
///
/// A named constant so the guard that looks for it matches the text the
/// function actually emits: a copy in a test is a copy that can drift, and the
/// drift would leave the guard passing over the very branch it exists to
/// forbid.
const NO_ROW_PREFIX: &str = "internal error: no screen message is declared for cause";

/// The screen texts one cause owes: degradation first, recovery second.
///
/// **The table is the contract, not a suggestion.** What a user reads must not
/// be left to whoever implements the call site, so each pair is written once,
/// here, and every message carries REQ-L23's first two parts — what broke, and
/// what it means for this session. [`render_transition`] adds the third.
///
/// # Two rows have no emitter yet
///
/// `provider`/`unreachable` and `vault`/`locked` are written but **nothing
/// emits them**: task 3.3 instrumented the embedder only, so they are absent
/// from [`CauseKey::ALL`] and cannot reach a screen today. They are kept rather
/// than deleted — this function is private, so an unused arm is not public
/// surface, and the row is what the task that instruments each subsystem will
/// need. A cause with no row is the loud case, not this one.
///
/// # Returns
///
/// `None` for a cause with no declared row, which is a defect rather than a
/// runtime case — see [`render_transition`].
///
/// # Complexity
///
/// `O(1)`: a match over a fixed set of literals.
fn screen_messages(key: CauseKey) -> Option<(&'static str, &'static str)> {
    match (key.subsystem(), key.cause()) {
        (SUBSYSTEM_EMBEDDER, CAUSE_UNREACHABLE) => Some((
            "memory: retrieval unavailable — answers will not use past context",
            "✓ memory: retrieval restored",
        )),
        (SUBSYSTEM_EMBEDDER, CAUSE_HTTP_ERROR) => Some((
            "memory: retrieval failing — answers will not use past context",
            "✓ memory: retrieval restored",
        )),
        (SUBSYSTEM_PROVIDER, CAUSE_UNREACHABLE) => Some((
            "provider: unreachable — the turn cannot complete",
            "✓ provider: reachable again",
        )),
        (SUBSYSTEM_VAULT, CAUSE_LOCKED) => Some((
            "vault: locked — stored credentials are unavailable",
            "✓ vault: unlocked",
        )),
        _ => None,
    }
}

/// Turns a transition into the line the user reads.
///
/// # Parameters
///
/// * `t` — the transition the tracker produced.
/// * `log_path` — the day's log file. **Received, never looked up**: this
///   module is pure, and REQ-L23's third part is a path only the caller knows.
///
/// # Returns
///
/// One screen line carrying REQ-L23's three parts: what broke, what it means
/// for this session, and where to read more.
///
/// # A cause with no declared row is a DEFECT, and the line says so
///
/// Substituting generic runtime text would satisfy the type and defeat the
/// requirement: the message exists to be specific, and the omission would stay
/// invisible until the cause fires in production, which is precisely when
/// somebody needs the message. Naming it instead makes the missing row
/// findable. It does not panic — this module is the one a reader reaches for
/// when everything else has already failed.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use magi_rs::logging::auditor::CauseKey;
/// use magi_rs::logging::health::{render_transition, Transition};
///
/// let key = CauseKey::new("provider", "unreachable");
/// let line = render_transition(
///     &Transition::Degraded(key),
///     Path::new(".magi/logs/magi-2026-08-14.log"),
/// );
/// assert!(line.starts_with("provider: unreachable"));
/// assert!(line.contains("magi-2026-08-14.log"));
/// ```
///
/// # Complexity
///
/// `O(n)` over the rendered line.
#[must_use]
pub fn render_transition(t: &Transition, log_path: &Path) -> String {
    let (key, text) = match t {
        Transition::Degraded(key) => (key, screen_messages(*key).map(|(d, _)| d)),
        Transition::Restored(key) => (key, screen_messages(*key).map(|(_, r)| r)),
    };
    match text {
        Some(text) => format!("{text} (see {})", log_path.display()),
        None => format!(
            "{NO_ROW_PREFIX} {}/{}; \
             the task that introduced the cause owes one (see {})",
            key.subsystem(),
            key.cause(),
            log_path.display()
        ),
    }
}

/// Whether the layer admits no `INFO` event.
///
/// The union is taken over the file branch's **filter** and the screen
/// branch's **level** — two different things, and only the first is
/// operator-settable. The screen branch is `SCREEN_LEVEL`, a constant that
/// never admits `INFO`, so in production this is a question about
/// `file_filter` alone. `screen_level` stays a parameter because MS1's screen
/// branch can be absent, which is what `None` means; it is not a second
/// configurable filter.
///
/// # Parameters
///
/// * `file_filter` — the file branch's filter, the only settable half.
/// * `screen_level` — the screen branch's fixed level, or `None` when no
///   screen branch is wired.
///
/// # Returns
///
/// `true` when no `INFO` event can reach the layer, which is the condition
/// under which recovery detection silently stops working: the success events a
/// recovery is derived from are `INFO`-level.
///
/// # Complexity
///
/// `O(k)` over the filter's per-target overrides.
pub(crate) fn recovery_detection_is_off(file_filter: &Filter, screen_level: Option<Level>) -> bool {
    let union = match screen_level {
        Some(screen) => file_filter.max_level().max(screen),
        None => file_filter.max_level(),
    };
    union < Level::INFO
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Every `CauseKey` below pairs a real subsystem with a real cause of it.
    // The state machine never consults the message table -- it keys on the
    // subsystem half alone -- so a key whose two halves were the same string
    // would still exercise it, but it would also assert that a "subsystem"
    // named `embedder_http_500` is a thing this program has, which it is not.

    #[test]
    fn the_first_degradation_is_immediate_and_does_not_wait_for_the_window() {
        // SC-L71: making the first "something broke" notice wait 30 s turns
        // the anti-flapping defence into a delay on the signal that matters
        // most.
        let mut h = HealthTracker::new();
        let c = CauseKey::new("embedder", "http_error");
        assert!(matches!(
            h.observe(Some(c), false, Instant::now()),
            Some(Transition::Degraded(_))
        ));
    }

    #[test]
    fn emits_only_on_transition_not_on_every_observation() {
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let c = CauseKey::new("embedder", "http_error");
        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        // Every further failing observation is silent: same state, no news.
        assert!(h
            .observe(Some(c), false, t0 + Duration::from_secs(1))
            .is_none());
        assert!(h
            .observe(Some(c), false, t0 + Duration::from_secs(2))
            .is_none());
        // The recovery is a SUBSEQUENT transition, so it serves the window.
        let t1 = t0 + Duration::from_secs(2);
        assert!(
            h.observe(Some(c), true, t1).is_none(),
            "not the first transition: windowed"
        );
        assert!(matches!(h.tick(t1 + w), Some(Transition::Restored(_))));
    }

    #[test]
    fn a_flapping_service_shows_one_degradation_and_nothing_more() {
        // SC-L70: down/up six times in 20 s. The FIRST degradation shows;
        // every later flip is pending and cancelled before its window
        // elapses, so the screen never flaps. The file still records all
        // twelve occurrences.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let c = CauseKey::new("embedder", "unreachable");
        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        for i in 1..=6u64 {
            let up = t0 + Duration::from_secs(i * 3);
            let down = t0 + Duration::from_secs(i * 3 + 1);
            assert!(
                h.observe(Some(c), true, up).is_none(),
                "recovery pending, not shown"
            );
            assert!(
                h.observe(Some(c), false, down).is_none(),
                "flip cancels the pending one"
            );
        }
        assert!(
            h.tick(t0 + Duration::from_secs(20)).is_none(),
            "nothing ever settled"
        );
    }

    #[test]
    fn one_endpoint_alternating_between_two_causes_shows_exactly_one_transition() {
        // SC-L76: one subsystem alternating http_500 <-> connection_refused
        // five times in 20 s. The screen sees ONE transition: the first.
        // Changing cause inside an already-degraded subsystem is not news,
        // and the window covers Degraded(A) -> Degraded(B).
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let a = CauseKey::new("embedder", "http_500");
        let b = CauseKey::new("embedder", "connection_refused");
        assert!(
            matches!(h.observe(Some(a), false, t0), Some(Transition::Degraded(_))),
            "the subsystem's first degradation is immediate"
        );
        for i in 1..=5u64 {
            let t = t0 + Duration::from_secs(i * 4);
            assert!(
                h.observe(Some(b), false, t).is_none(),
                "cause change: goes through the window"
            );
            assert!(h
                .observe(Some(a), false, t + Duration::from_secs(1))
                .is_none());
        }
        assert!(
            h.tick(t0 + Duration::from_secs(30)).is_none(),
            "no cause held for the window, so there is no second transition"
        );
    }

    #[test]
    fn two_different_subsystems_each_get_an_immediate_first_degradation() {
        // The cascade case, the other side of the same rule.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let emb = CauseKey::new("embedder", "unreachable");
        let prov = CauseKey::new("provider", "unreachable");
        assert!(matches!(
            h.observe(Some(emb), false, t0),
            Some(Transition::Degraded(_))
        ));
        assert!(
            matches!(
                h.observe(Some(prov), false, t0 + Duration::from_secs(600)),
                Some(Transition::Degraded(_))
            ),
            "another subsystem: its first notice is also immediate"
        );
    }

    #[test]
    fn a_cancelled_recovery_followed_by_a_cause_change_shows_nothing() {
        // The edge that combines both mechanisms: a pending recovery gets
        // cancelled, then a cause change inside the same subsystem. Neither
        // is news, so the screen sees nothing -- but it is the one path
        // where the pending transition changes CLASS (Restored -> Degraded)
        // without ever being emitted.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let a = CauseKey::new("embedder", "http_500");
        let b = CauseKey::new("embedder", "connection_refused");
        assert!(matches!(
            h.observe(Some(a), false, t0),
            Some(Transition::Degraded(_))
        ));
        assert!(
            h.observe(Some(a), true, t0 + Duration::from_secs(5))
                .is_none(),
            "recovery pending"
        );
        assert!(
            h.observe(Some(b), false, t0 + Duration::from_secs(10))
                .is_none(),
            "cancels it and leaves a cause change pending"
        );
        assert!(
            h.tick(t0 + Duration::from_secs(15)).is_none(),
            "the window restarted"
        );
        let got = h.tick(t0 + Duration::from_secs(10 + HEALTH_MIN_STABLE_SECS));
        assert!(
            matches!(got, Some(Transition::Degraded(_))),
            "and once it elapses, it is a degradation"
        );
    }

    #[test]
    fn alternating_between_two_failing_causes_is_not_a_recovery() {
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let a = CauseKey::new("embedder", "cause_a");
        let b = CauseKey::new("embedder", "cause_b");
        // The subsystem's first degradation is IMMEDIATE (SC-L71)...
        assert!(matches!(
            h.observe(Some(a), false, t0),
            Some(Transition::Degraded(_))
        ));
        // ...so the later tick has nothing pending to emit.
        assert!(h
            .tick(t0 + Duration::from_secs(HEALTH_MIN_STABLE_SECS))
            .is_none());
        // Changing cause inside the same degraded subsystem goes through the
        // window, and once it elapses it is still a degradation: never a
        // recovery.
        let t1 = t0 + Duration::from_secs(HEALTH_MIN_STABLE_SECS + 1);
        assert!(
            h.observe(Some(b), false, t1).is_none(),
            "still pending, not immediate"
        );
        let got = h.tick(t1 + Duration::from_secs(HEALTH_MIN_STABLE_SECS));
        assert!(matches!(got, Some(Transition::Degraded(_))));
        assert!(
            !matches!(got, Some(Transition::Restored(_))),
            "never a recovery"
        );
    }

    #[test]
    fn a_revert_discards_the_pending_transition_instead_of_letting_it_elapse() {
        // R-L13d's discard half. A revert to the state already on screen is
        // not merely "no news": it must DROP the transition that was serving
        // its window, or that transition survives and a later `tick` emits a
        // change the subsystem has since undone.
        //
        // The flapping test above cannot see this: it ticks at 20 s, before
        // any of its cancelled windows could have elapsed. The tick here is
        // deliberately placed PAST the discarded recovery's original window.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let c = CauseKey::new("embedder", "http_error");

        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        // A recovery starts serving its window at t0 + 1 s.
        let recovered_at = t0 + Duration::from_secs(1);
        assert!(
            h.observe(Some(c), true, recovered_at).is_none(),
            "a recovery is never immediate: it serves the window"
        );
        // The subsystem fails again before that window elapses, reverting to
        // the state already shown. That discards the pending recovery.
        assert!(
            h.observe(Some(c), false, t0 + Duration::from_secs(2))
                .is_none(),
            "reverting to the shown state is not news"
        );

        // The subsystem stays degraded from here on, so nothing may come out
        // at the discarded recovery's own deadline...
        assert!(
            h.tick(recovered_at + w).is_none(),
            "the discarded recovery must not surface at its original deadline"
        );
        // ...nor at any point after it.
        assert!(
            h.tick(recovered_at + w + w).is_none(),
            "and it must not surface later either: it was discarded, not delayed"
        );
    }

    #[test]
    fn repeating_a_pending_candidate_runs_its_window_from_the_first_observation() {
        // The keep-original-`since` half of the same rule. A candidate that is
        // already pending must not restart its window every time the same
        // state is observed again: a subsystem that keeps reporting success
        // once a second would then never reach its deadline, and the recovery
        // would never be shown at all.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let c = CauseKey::new("embedder", "http_error");

        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));

        // The recovery's window starts at this first observation of it.
        let first_seen = t0 + Duration::from_secs(1);
        assert!(h.observe(Some(c), true, first_seen).is_none(), "pending");

        // The same candidate again, once a second, all of it well inside the
        // window. Every one of these takes the keep-original-`since` arm.
        for i in 1..HEALTH_MIN_STABLE_SECS {
            assert!(
                h.observe(Some(c), true, first_seen + Duration::from_secs(i))
                    .is_none(),
                "still the same candidate: nothing to show yet"
            );
        }

        // At FIRST observation + window it must emit. Were the window
        // restarted on each observation, `since` would be the last of them and
        // this tick would land a whole window early with nothing to give.
        assert!(
            matches!(h.tick(first_seen + w), Some(Transition::Restored(_))),
            "the window runs from the first observation of the candidate, \
             not from the most recent one"
        );
    }

    #[test]
    fn a_repeating_new_cause_runs_its_window_from_the_first_observation_too() {
        // The DEGRADED half of the keep-original-`since` rule, and the half the
        // test above cannot reach: its pending candidate is a recovery
        // (`target == None`), so a keep arm narrowed to `candidate.is_none()`
        // leaves it green while every cause change inside an already-degraded
        // subsystem silently restarts its window on each observation.
        //
        // That is the production case, not a curiosity: a subsystem already
        // shown as degraded whose NEW cause fires once a second would then
        // never reach a deadline, and the screen would never be told the cause
        // changed at all.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let first = CauseKey::new("embedder", "http_error");
        let second = CauseKey::new("embedder", "unreachable");

        assert!(matches!(
            h.observe(Some(first), false, t0),
            Some(Transition::Degraded(_))
        ));

        // The new cause's window starts at this first observation of it.
        let first_seen = t0 + Duration::from_secs(1);
        assert!(
            h.observe(Some(second), false, first_seen).is_none(),
            "a cause change inside a degraded subsystem serves the window"
        );

        // The same new cause again, once a second, all of it well inside the
        // window. Every one of these takes the keep-original-`since` arm.
        for i in 1..HEALTH_MIN_STABLE_SECS {
            assert!(
                h.observe(Some(second), false, first_seen + Duration::from_secs(i))
                    .is_none(),
                "still the same candidate: nothing to show yet"
            );
        }

        // At FIRST observation + window it must emit, and it must emit the NEW
        // cause: were the window restarted on each observation, `since` would be
        // the last of them and this tick would land a whole window early.
        let got = h.tick(first_seen + w);
        assert!(
            matches!(got, Some(Transition::Degraded(k)) if k.cause() == second.cause()),
            "the window runs from the first observation of the new cause, not \
             from the most recent one, and what comes out is that new cause: \
             {got:?}"
        );
    }

    #[test]
    fn a_degradation_after_a_shown_recovery_is_immediate_again() {
        // The immediacy rule of SC-L71 is not a once-per-process privilege: it
        // is a property of a subsystem that is currently healthy. Once a
        // recovery has actually been SHOWN the subsystem is healthy again, so
        // the next degradation is once more a first degradation.
        //
        // This also pins what the emission itself did to `shown`, which no
        // other test continues past a `tick` far enough to see.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let c = CauseKey::new("embedder", "http_error");

        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        let recovered_at = t0 + Duration::from_secs(1);
        assert!(h.observe(Some(c), true, recovered_at).is_none(), "pending");
        assert!(
            matches!(h.tick(recovered_at + w), Some(Transition::Restored(_))),
            "the recovery reaches the screen once its window elapses"
        );

        // Emitting it is what moved `shown` back to healthy, so a further
        // success is now the state already on screen and says nothing.
        let after = recovered_at + w + Duration::from_secs(1);
        assert!(
            h.observe(Some(c), true, after).is_none(),
            "after the recovery was shown, staying healthy is not news"
        );
        // And therefore the next degradation is a first degradation.
        assert!(
            matches!(
                h.observe(Some(c), false, after + Duration::from_secs(1)),
                Some(Transition::Degraded(_))
            ),
            "the immediacy rule re-arms: a healthy subsystem's degradation is \
             immediate however many times it has already been through one"
        );
    }

    #[test]
    fn a_flushed_recovery_leaves_the_subsystem_healthy() {
        // `flush`'s counterpart to the test above: it writes `shown` too, and
        // nothing else looks at the tracker after a flush has emitted.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let c = CauseKey::new("embedder", "http_error");

        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        assert!(h
            .observe(Some(c), true, t0 + Duration::from_secs(1))
            .is_none());
        assert!(matches!(h.flush().as_slice(), [Transition::Restored(_)]));

        // Flushed, so the subsystem is healthy: the next degradation is
        // immediate rather than windowed.
        assert!(matches!(
            h.observe(Some(c), false, t0 + Duration::from_secs(2)),
            Some(Transition::Degraded(_))
        ));
    }

    #[test]
    fn an_event_without_a_cause_key_is_ignored_rather_than_keyed_off_its_text() {
        let mut h = HealthTracker::new();
        assert!(h.observe(None, false, Instant::now()).is_none());
    }

    #[test]
    fn a_short_headless_run_flushes_a_pending_recovery() {
        // SC-L90 is about what is left PENDING at close, and the only
        // transition that can be pending in a short run is a RECOVERY: a
        // subsystem's first degradation is always immediate (SC-L71), so it
        // is never left waiting.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let c = CauseKey::new("embedder", "http_500");
        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        assert!(
            h.observe(Some(c), true, t0 + Duration::from_secs(1))
                .is_none(),
            "pending"
        );
        let flushed = h.flush();
        assert_eq!(
            flushed.len(),
            1,
            "closing must not swallow a pending transition"
        );
        assert!(matches!(flushed[0], Transition::Restored(_)));
    }

    #[test]
    fn flushing_with_nothing_pending_returns_an_empty_vec() {
        // The complement of the test above, and the one that shows why
        // SC-L90 does not apply to the first degradation: it was already
        // emitted, so there is nothing left to flush.
        let mut h = HealthTracker::new();
        let c = CauseKey::new("embedder", "http_500");
        assert!(matches!(
            h.observe(Some(c), false, Instant::now()),
            Some(Transition::Degraded(_))
        ));
        assert!(h.flush().is_empty());
    }

    #[test]
    fn flush_emits_one_pending_transition_per_subsystem() {
        // Why `flush` returns a `Vec`: a cascading failure leaves one
        // pending transition per subsystem, and an `Option` would show one
        // and silently drop the rest.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let emb = CauseKey::new("embedder", "unreachable");
        let prov = CauseKey::new("provider", "unreachable");
        h.observe(Some(emb), false, t0);
        h.observe(Some(prov), false, t0);
        assert!(h
            .observe(Some(emb), true, t0 + Duration::from_secs(1))
            .is_none());
        assert!(h
            .observe(Some(prov), true, t0 + Duration::from_secs(1))
            .is_none());
        assert_eq!(
            h.flush().len(),
            2,
            "one pending transition per subsystem, not just one"
        );
    }

    #[test]
    fn a_quiet_subsystem_is_untouched_while_another_ones_window_elapses() {
        // The mixed-state vector. Every other test here drives ONE subsystem at
        // a time or drives both the same way, so nothing pins that `tick` and
        // `flush` -- which both walk the whole `states` vector -- pick out the
        // subsystem that has something to say and step over the one that does
        // not. Getting that wrong invents a transition for a subsystem whose
        // state never changed.
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let quiet = CauseKey::new("provider", "unreachable");
        let moving = CauseKey::new("embedder", "http_error");
        let recovered_at = t0 + Duration::from_secs(1);

        // Shown and then silent, versus shown and then recovering: one entry
        // with no pending, one entry with a pending that goes on to elapse.
        let arrange = || {
            let mut h = HealthTracker::new();
            assert!(matches!(
                h.observe(Some(quiet), false, t0),
                Some(Transition::Degraded(_))
            ));
            assert!(matches!(
                h.observe(Some(moving), false, t0),
                Some(Transition::Degraded(_))
            ));
            assert!(
                h.observe(Some(moving), true, recovered_at).is_none(),
                "a recovery serves the window"
            );
            h
        };

        let mut ticked = arrange();
        let got = ticked.tick(recovered_at + w);
        assert!(
            matches!(got, Some(Transition::Restored(k)) if k.subsystem() == moving.subsystem()),
            "the elapsed pending is what comes out, and it belongs to the \
             subsystem that had one: {got:?}"
        );
        assert!(
            ticked.tick(recovered_at + w + w).is_none(),
            "and the quiet subsystem contributes nothing however late the tick"
        );

        let mut flushed = arrange();
        let out = flushed.flush();
        assert_eq!(
            out.len(),
            1,
            "flush must not invent a transition for a subsystem with nothing \
             pending: {out:?}"
        );
        assert!(
            matches!(out.first(), Some(Transition::Restored(k)) if k.subsystem() == moving.subsystem()),
            "and the one it does emit is the pending one: {out:?}"
        );
    }

    #[test]
    fn flush_order_is_first_observed_even_when_the_first_event_was_healthy() {
        // `states` is ordered by first OBSERVATION, not by first degradation,
        // and the two only disagree when some subsystem's first event is a
        // success -- which no other test arranges. Without this, an
        // implementation that pushed an entry on first FAILURE would satisfy
        // every other ordering assertion in the module.
        let mut h = HealthTracker::new();
        let t0 = Instant::now();
        let seen_first = CauseKey::new("provider", "unreachable");
        let broke_first = CauseKey::new("embedder", "http_error");

        // A healthy event creates the entry and says nothing.
        assert!(
            h.observe(Some(seen_first), true, t0).is_none(),
            "a healthy first event is not a transition"
        );
        // The OTHER subsystem is the first one that actually breaks.
        assert!(matches!(
            h.observe(Some(broke_first), false, t0 + Duration::from_secs(1)),
            Some(Transition::Degraded(_))
        ));
        assert!(matches!(
            h.observe(Some(seen_first), false, t0 + Duration::from_secs(2)),
            Some(Transition::Degraded(_))
        ));

        // Both leave a pending recovery, so both appear in the flush.
        assert!(h
            .observe(Some(broke_first), true, t0 + Duration::from_secs(3))
            .is_none());
        assert!(h
            .observe(Some(seen_first), true, t0 + Duration::from_secs(4))
            .is_none());

        let order: Vec<&str> = h
            .flush()
            .iter()
            .map(|t| match t {
                Transition::Degraded(k) | Transition::Restored(k) => k.subsystem(),
            })
            .collect();
        assert_eq!(
            order,
            vec![seen_first.subsystem(), broke_first.subsystem()],
            "first OBSERVED, not first degraded: the subsystem whose opening \
             event was a success still comes out ahead of the one that broke \
             before it did"
        );
    }

    /// The day's log file, as REQ-L23's third part reaches `render_transition`.
    const A_LOG_PATH: &str = ".magi/logs/magi-2026-08-14.log";
    /// The file name inside [`A_LOG_PATH`], which is what a message must name.
    const A_LOG_FILE: &str = "magi-2026-08-14.log";

    /// Every cause the message table declares, with the two texts it owes.
    ///
    /// Kept as one table so a row added to `render_transition` without its
    /// counterpart here is visible as an omission rather than as absence.
    fn declared_messages() -> Vec<(CauseKey, &'static str, &'static str)> {
        vec![
            (
                CauseKey::new("embedder", "unreachable"),
                "memory: retrieval unavailable — answers will not use past context",
                "✓ memory: retrieval restored",
            ),
            (
                CauseKey::new("embedder", "http_error"),
                "memory: retrieval failing — answers will not use past context",
                "✓ memory: retrieval restored",
            ),
            (
                CauseKey::new("provider", "unreachable"),
                "provider: unreachable — the turn cannot complete",
                "✓ provider: reachable again",
            ),
            (
                CauseKey::new("vault", "locked"),
                "vault: locked — stored credentials are unavailable",
                "✓ vault: unlocked",
            ),
        ]
    }

    #[test]
    fn each_declared_cause_names_what_broke_what_it_means_and_where_to_read_more() {
        // REQ-L23's three parts. The first two come from the table; the third
        // is always the day's log file, which is why this function takes a path
        // instead of composing one.
        let path = Path::new(A_LOG_PATH);
        for (key, degraded, restored) in declared_messages() {
            let d = render_transition(&Transition::Degraded(key), path);
            let r = render_transition(&Transition::Restored(key), path);
            assert!(
                d.starts_with(degraded),
                "the degradation text for {key:?} is not the declared one: {d}"
            );
            assert!(
                r.starts_with(restored),
                "the recovery text for {key:?} is not the declared one: {r}"
            );
            assert!(
                d.contains(A_LOG_FILE),
                "REQ-L23's third part -- where to read more -- is missing: {d}"
            );
            assert!(
                r.contains(A_LOG_FILE),
                "REQ-L23's third part -- where to read more -- is missing: {r}"
            );
        }
    }

    #[test]
    fn test_an_actionable_error_names_what_broke_what_it_means_and_where_to_read_more() {
        // REQ-L23: an actionable screen error has three parts, and this
        // asserts each on its own, specific content -- not on non-emptiness
        // and not on a `.log` substring, both of which a no-row branch or an
        // unrelated path would also satisfy. A fourth assertion at the end
        // protects part 3 from truncation; it is NOT the plan's width rule,
        // and the comment there says why.
        //
        // # SC-L19's producer is an accident, and it is worth writing down
        //
        // The scenario's *Given* is "an Ollama with no `llama-server`", which
        // names no subsystem this module knows. It has a producer today only
        // because the embedder happens to share Ollama's endpoint: the daemon
        // being down makes the embedder's call fail as `Network`, which is
        // `embedder/unreachable`. Point the embedder at a different endpoint
        // and SC-L19 loses its producer with nothing here going red, because
        // `provider` is declared in the message table and instrumented by
        // nobody (R24). What this test can be made independent of is WHICH
        // cause carries it: the budget below is checked over every declared
        // row rather than over the one this test names, so it holds for
        // whichever subsystem is the producer of the day.
        let path = Path::new(A_LOG_PATH);
        let line = render_transition(
            &Transition::Degraded(CauseKey::new("provider", "unreachable")),
            path,
        );

        // Part 1 -- what broke: the subsystem is named.
        assert!(
            line.contains("provider"),
            "REQ-L23 part 1 (what broke) is missing the subsystem name: {line}"
        );
        // Part 2 -- what it means for this session: the consequence is stated.
        assert!(
            line.contains("the turn cannot complete"),
            "REQ-L23 part 2 (what it means for this session) is missing: {line}"
        );
        // Part 3 -- where to read more: the exact path passed in, not merely
        // something ending in ".log".
        //
        // The needle is what `Path::display` MAKES of the path, never the
        // literal it was built from: `render_transition` renders through
        // `display`, and comparing against the literal asserts on top of that
        // rendering being an identity. It is one today for a path built from a
        // `/`-separated literal, so this assertion has been passing on Windows
        // by coincidence rather than by construction -- and the coincidence
        // ends the moment `A_LOG_PATH` is assembled with `join` instead.
        let rendered_path = path.display().to_string();
        assert!(
            line.contains(&rendered_path),
            "REQ-L23 part 3 (where to read more) does not carry the log path \
             actually passed in ({rendered_path}): {line}"
        );

        // Part 3 must survive the trip to the screen. `TUI_PAYLOAD_MAX_BYTES`
        // is what the layer applies to this exact string with
        // `truncate_for_display` before the screen sees it, and truncation
        // takes the TAIL -- so a line past that cap loses part 3 while parts 1
        // and 2 still assert green above. The number is read from the layer
        // rather than copied, or it would drift and keep passing while the real
        // line was being cut.
        //
        // **This is a 64 KiB PAYLOAD cap, not a terminal width, and calling it
        // one would be the more comfortable lie.** The plan's SC-L19 also asks
        // for "<= 100 characters" so the message fits a narrow terminal. No
        // constant in the tree expresses that, and inventing one to assert
        // against would fabricate a requirement the spec states only as prose
        // -- so the width rule is a DECLARED HOLE, recorded as such in the
        // certificate rather than counted as covered. What is known is a
        // measurement, not a guarantee: under R25 the eight shipped messages
        // were 65, 28, 61, 28, 48, 27, 50 and 17 characters. Nothing pins that,
        // so a future row may exceed 100 with nothing here going red.
        for (key, _, _) in declared_messages() {
            for t in [Transition::Degraded(key), Transition::Restored(key)] {
                let rendered = render_transition(&t, path);
                assert!(
                    rendered.len() <= crate::logging::magi_layer::TUI_PAYLOAD_MAX_BYTES,
                    "REQ-L23 part 3: {key:?} renders {} bytes, past the \
                     screen's {}-byte payload cap, so the layer truncates it \
                     from the tail and the log path is what the reader \
                     loses: {rendered}",
                    rendered.len(),
                    crate::logging::magi_layer::TUI_PAYLOAD_MAX_BYTES
                );
            }
        }
    }

    #[test]
    fn every_declared_cause_key_has_a_screen_message() {
        // The guard belongs to task 3.3 and not to the task that wrote the
        // table: above it, `CauseKey::ALL` is empty, the loop runs zero times
        // and the whole thing is a green tick over nothing.
        assert!(
            !CauseKey::ALL.is_empty(),
            "task 3.3 declares every cause it instruments here; with an empty \
             list the loop below guards nothing"
        );
        let path = Path::new(A_LOG_PATH);
        for key in CauseKey::ALL {
            let d = render_transition(&Transition::Degraded(*key), path);
            let r = render_transition(&Transition::Restored(*key), path);
            assert!(!d.is_empty(), "no degradation message for {key:?}");
            assert!(!r.is_empty(), "no recovery message for {key:?}");
            // The absence of the no-row branch is the real assertion: its text
            // is not empty either, so a check for emptiness alone would pass
            // for every undeclared cause in the list.
            assert!(
                !d.contains(NO_ROW_PREFIX) && !r.contains(NO_ROW_PREFIX),
                "{key:?} is declared but has no row in the message table, so \
                 the user would read a defect report instead of a message: {d}"
            );
            assert!(
                d.contains(A_LOG_FILE),
                "REQ-L23's third part -- where to read more -- is missing: {d}"
            );
        }
    }

    #[test]
    fn a_cause_with_no_declared_message_is_reported_as_a_programming_error() {
        // A `CauseKey` with no row is a bug in the task that introduced it, not
        // a runtime case to paper over with generic text: generic text is
        // exactly what REQ-L23 exists to forbid, and it would hide the omission
        // until the cause fires in production, which is when the message is
        // needed.
        let path = Path::new(A_LOG_PATH);
        let out = render_transition(
            &Transition::Degraded(CauseKey::new("nonesuch", "no_such_cause")),
            path,
        );
        assert!(
            out.contains("internal error"),
            "an undeclared cause must say it is a defect: {out}"
        );
        assert!(
            out.contains("nonesuch") && out.contains("no_such_cause"),
            "and it must name the cause, or nobody can find the missing row: {out}"
        );
    }

    /// Every line a closure emitted, through the real dispatcher.
    ///
    /// `testutil::capture` returns only the last line, which cannot tell "one
    /// notice" from "two" -- and "exactly one" is half of what the test below
    /// asserts.
    fn capture_all(emit: impl FnOnce()) -> Vec<String> {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt as _;

        let buf: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        struct CaptureLayer(Arc<Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if let Ok(mut g) = self.0.lock() {
                    g.push(crate::logging::render::render_event(event));
                }
            }
        }
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&buf)));
        tracing::subscriber::with_default(subscriber, emit);
        let lines = buf.lock().expect("never poisoned").clone();
        lines
    }

    #[test]
    fn test_a_warn_is_emitted_when_both_filters_exclude_info() {
        // The failure this announces is silent by construction: the operator
        // sees degradations and never sees one recover, months after setting
        // the filters. "Declared in the plan" does not reach that person.
        let warn_only = Filter::parse("warn").expect("a valid directive");
        let said = capture_all(|| {
            crate::logging::warn_if_recovery_detection_is_off(&warn_only, Some(Level::WARN))
        });
        assert_eq!(
            said.len(),
            1,
            "exactly one notice, naming the consequence: {said:?}"
        );
        assert!(
            said[0].contains("health recovery detection is off"),
            "the notice must name the consequence, not merely the setting: {said:?}"
        );

        // The other half, without which this passes against a notice that
        // always fires -- which is a notice nobody reads.
        let defaults = Filter::parse("info").expect("a valid directive");
        let quiet = capture_all(|| {
            crate::logging::warn_if_recovery_detection_is_off(&defaults, Some(Level::WARN))
        });
        assert!(
            quiet.is_empty(),
            "with the defaults the union admits info, so there is nothing to warn about: {quiet:?}"
        );
    }

    #[test]
    fn with_no_screen_branch_the_file_filter_alone_decides() {
        // The `None` arm, which nothing else exercises. It is not dead: MS1
        // ships without a screen branch, and every caller there passes `None`.
        // Read through the union, `None` has to mean "there is no second
        // source of INFO", and the failure mode of getting it wrong is silent
        // in both directions -- a `None` treated as INFO would suppress the
        // warning for the exact configuration that needs it.
        let warn_only = Filter::parse("warn").expect("a valid directive");
        assert!(
            recovery_detection_is_off(&warn_only, None),
            "with no screen branch there is nothing to rescue a file filter \
             that excludes info"
        );
        let admits_info = Filter::parse("info").expect("a valid directive");
        assert!(
            !recovery_detection_is_off(&admits_info, None),
            "and a file filter that admits info needs no rescuing: without \
             this half the function could answer `true` always"
        );
    }

    #[test]
    fn a_screen_level_that_admits_info_rescues_a_file_filter_that_does_not() {
        // The UNION is the contract, not the file filter: a screen branch that
        // admits INFO keeps recovery detection working however narrow the file
        // filter is. Production cannot reach this today -- `SCREEN_LEVEL` is
        // `WARN` -- which is exactly why it needs a test: the parameter is what
        // makes the union a union, and nothing else would notice it being
        // dropped for `file_filter.max_level()`.
        let warn_only = Filter::parse("warn").expect("a valid directive");
        assert!(
            recovery_detection_is_off(&warn_only, Some(Level::WARN)),
            "neither branch admits info, so recovery detection is off"
        );
        assert!(
            !recovery_detection_is_off(&warn_only, Some(Level::INFO)),
            "the same file filter, rescued by the screen branch: the answer is \
             the union of the two and not the file filter alone"
        );
    }

    #[test]
    fn the_collected_notice_carries_the_same_warning_as_the_emitted_one() {
        // `recovery_detection_notice` is the terminal surface's copy of the
        // warning above, and `main.rs` extends the TUI's startup list with it.
        // Its two properties are that it fires on the same condition and that
        // it says the same thing: a notice that fired always would be one
        // nobody reads, and one whose text drifted would send an operator
        // looking for a setting under a name the other half does not use.
        let warn_only = Filter::parse("warn").expect("a valid directive");
        let notice = crate::logging::recovery_detection_notice(&warn_only, Some(Level::WARN))
            .expect("the union excludes info, so there is a notice");
        assert_eq!(
            notice.level,
            Level::WARN,
            "it has to reach a screen, which INFO does not (REQ-L19)"
        );
        assert!(
            notice.text.contains("health recovery detection is off"),
            "the notice must name the consequence, not merely the setting: {notice:?}"
        );

        let defaults = Filter::parse("info").expect("a valid directive");
        assert!(
            crate::logging::recovery_detection_notice(&defaults, Some(Level::WARN)).is_none(),
            "with the defaults the union admits info, so the startup list gets \
             nothing to extend with"
        );
    }

    #[test]
    fn a_clock_jump_produces_no_false_transitions() {
        let mut h = HealthTracker::new();
        let c = CauseKey::new("embedder", "unreachable");
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let t1 = t0 + Duration::from_secs(60 * 60 * 24 * 60); // sixty days later

        // SC-L71: the subsystem's first degradation is IMMEDIATE, so `observe`
        // returns it and the subsequent `tick` has nothing pending. (These two
        // lines used to assert the pre-SC-L71 behaviour, where the window also
        // governed the first degradation -- the same call sequence as
        // `alternating_between_two_failing_causes_is_not_a_recovery` with the
        // opposite result, which no deterministic implementation could satisfy.)
        assert!(matches!(
            h.observe(Some(c), false, t0),
            Some(Transition::Degraded(_))
        ));
        assert!(
            h.tick(t0 + w).is_none(),
            "nothing pending right after an immediate emission"
        );
        assert!(
            h.observe(Some(c), false, t1).is_none(),
            "elapsed monotonic time is not a transition"
        );
        assert!(
            h.tick(t1 + w).is_none(),
            "and neither is the window expiring on an unchanged state"
        );
        // The POSITIVE case, without which this test would pass against a
        // tracker that stopped emitting altogether after the jump: a real
        // transition still comes out.
        assert!(
            h.observe(Some(c), true, t1 + w).is_none(),
            "recovery is windowed"
        );
        assert!(
            matches!(h.tick(t1 + w + w), Some(Transition::Restored(_))),
            "a real transition still fires after the jump"
        );
    }
}
