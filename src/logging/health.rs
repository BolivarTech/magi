// Author: Julian Bolivar
// Version: 0.18.1
// Date: 2026-08-31

//! Transition-only health tracking (D-L13, SC-L14...SC-L119).
//!
//! This module decides *when* a subsystem's health is worth putting on
//! screen. It is pure: no `tracing`, no filesystem, no TUI. `observe` takes
//! whatever the caller already knows about one event and returns a
//! [`Transition`] only when that transition is worth showing; everything
//! else is silence by design, because "everything else" is exactly the noise
//! this feature exists to remove.

use std::time::{Duration, Instant};

use crate::logging::auditor::CauseKey;

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
    /// One entry per subsystem observed so far, kept in first-observed
    /// order so [`Self::flush`] can report the earliest degradation first.
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
    /// `O(n)` in the number of distinct subsystems seen so far, which in
    /// practice is a handful (embedder, provider, vault, ...) -- amortised
    /// `O(1)` per observation against that fixed small `n`.
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
    /// `O(1)` amortised: one lookup into `states` (linear in the small,
    /// fixed number of subsystems seen so far) plus constant-time state
    /// comparison.
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
            // Already shown, not pending: nothing left to wait on.
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
    /// periodically (the TUI event loop's own `poll` timeout) and once more
    /// at shutdown; without it, a pending transition whose subsystem stops
    /// producing events -- exactly what happens when something goes fully
    /// dark -- would never be emitted.
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
    /// and an `Option` would show one and silently drop the rest. The order
    /// matches first-observed order, so the earliest subsystem to degrade
    /// is reported first.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_first_degradation_is_immediate_and_does_not_wait_for_the_window() {
        // SC-L71: making the first "something broke" notice wait 30 s turns
        // the anti-flapping defence into a delay on the signal that matters
        // most.
        let mut h = HealthTracker::new();
        let c = CauseKey::new("embedder_http_500", "embedder_http_500");
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
        let c = CauseKey::new("embedder_http_500", "embedder_http_500");
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
        let c = CauseKey::new("flapping_endpoint", "flapping_endpoint");
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
    fn a_clock_jump_produces_no_false_transitions() {
        let mut h = HealthTracker::new();
        let c = CauseKey::new("cause", "cause");
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
