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
//!
//! # Status
//!
//! Red phase stub: signatures are final, bodies are not. Every method below
//! returns an obviously-wrong value on purpose (G8) so the test suite fails
//! by assertion rather than by `unimplemented!()`, which this module's lint
//! policy denies outside `cfg(test)`.

use std::time::Instant;

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

/// Decides when a degradation or recovery is worth showing on screen.
#[derive(Debug, Clone, Default)]
pub struct HealthTracker;

impl HealthTracker {
    /// Builds an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Observes one event. Stub: always returns `None`.
    pub fn observe(
        &mut self,
        _cause: Option<CauseKey>,
        _ok: bool,
        _now: Instant,
    ) -> Option<Transition> {
        None
    }

    /// Expires the window. Stub: always returns `None`.
    pub fn tick(&mut self, _now: Instant) -> Option<Transition> {
        None
    }

    /// Emits pending transitions. Stub: always empty.
    pub fn flush(&mut self) -> Vec<Transition> {
        Vec::new()
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
    fn a_clock_jump_produces_no_false_transitions() {
        let mut h = HealthTracker::new();
        let c = CauseKey::new("cause", "cause");
        let t0 = Instant::now();
        let w = Duration::from_secs(HEALTH_MIN_STABLE_SECS);
        let t1 = t0 + Duration::from_secs(60 * 60 * 24 * 60); // sixty days later
        h.observe(Some(c), false, t0);
        assert!(matches!(h.tick(t0 + w), Some(Transition::Degraded(_))));
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
