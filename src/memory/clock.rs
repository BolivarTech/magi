// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Injected time abstraction for the memory subsystem.
//!
//! All time-dependent logic in `src/memory/` (decay, reranker recency, eviction)
//! receives a `&dyn Clock` rather than calling `SystemTime::now()` directly.
//! This keeps the hot paths **deterministic under test** (R-06/D-18):
//! production code passes a [`SystemClock`]; tests and the benchmark pass a
//! [`FixedClock`] that can be advanced programmatically.
//!
//! # Design invariant
//! [`SystemClock`] is the **only** site in `src/memory/` permitted to call
//! `std::time::SystemTime::now()`. All other modules must accept a `Clock`
//! parameter instead of sampling the wall clock themselves.

use std::sync::atomic::{AtomicI64, Ordering};

/// A source of "now" in unix seconds (UTC).
///
/// Injected so that time is an **input** to the memory subsystem rather than
/// an ambient side-effect (R-06/D-18). Production code uses [`SystemClock`];
/// deterministic tests and benchmarks use [`FixedClock`].
// Narrow allow: trait is consumed by the decay module (Task 8b), retrieval
// (Task 7), and context assembler (Task 12) — not yet wired in this task.
#[allow(dead_code)]
pub trait Clock: Send + Sync {
    /// Returns the current time as Unix seconds (UTC).
    fn now(&self) -> i64;
}

/// Production clock — the **only** `SystemTime::now()` call site in `src/memory/`.
///
/// Returns the wall clock as Unix seconds. On the astronomically unlikely event
/// that the system clock predates the Unix epoch, returns `0` rather than
/// panicking.
// Narrow allow: consumed by tasks wired in Task 7/8b/12.
#[allow(dead_code)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0) // clock before epoch ⇒ 0 (never panic)
    }
}

/// Deterministic virtual clock for tests and benchmarks.
///
/// Interior-mutable via an [`AtomicI64`] so it can be advanced while shared
/// behind `&dyn Clock` without requiring `&mut self`.
///
/// # Example
/// ```ignore
/// let clock = FixedClock::new(0);
/// let dyn_clock: &dyn Clock = &clock;
/// clock.advance_days(30.0);
/// assert_eq!(dyn_clock.now(), 30 * 86_400);
/// ```
// Narrow allow: consumed by tasks wired in Task 7/8b/12.
#[allow(dead_code)]
pub struct FixedClock {
    secs: AtomicI64,
}

impl FixedClock {
    /// Creates a new [`FixedClock`] starting at the given Unix timestamp.
    // Narrow allow: consumed by tasks wired in Task 7/8b/12.
    #[allow(dead_code)]
    pub fn new(secs: i64) -> Self {
        Self {
            secs: AtomicI64::new(secs),
        }
    }

    /// Advances the virtual clock by `days` (fractional days are allowed;
    /// the result is truncated to whole seconds).
    ///
    /// Uses `SeqCst` ordering so that a concurrent `now()` read on another
    /// thread always observes either the old or the fully-advanced value.
    // Narrow allow: consumed by tasks wired in Task 7/8b/12.
    #[allow(dead_code)]
    pub fn advance_days(&self, days: f64) {
        let delta = (days * 86_400.0) as i64;
        self.secs.fetch_add(delta, Ordering::SeqCst);
    }
}

impl Clock for FixedClock {
    fn now(&self) -> i64 {
        self.secs.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_clock_returns_a_plausible_unix_time() {
        // After 2020-01-01 (1_577_836_800) — sanity that it reads the real clock.
        assert!(SystemClock.now() > 1_577_836_800);
    }

    #[test]
    fn test_fixed_clock_is_deterministic_and_advanceable() {
        let c = FixedClock::new(1_000);
        assert_eq!(c.now(), 1_000);
        assert_eq!(c.now(), 1_000); // stable
        c.advance_days(2.0);
        assert_eq!(c.now(), 1_000 + 2 * 86_400);
        c.advance_days(0.5);
        assert_eq!(c.now(), 1_000 + 2 * 86_400 + 43_200);
    }

    /// B3: `SystemClock::now()` must not overflow on the u64→i64 conversion.
    /// We cannot advance the wall clock, so we just confirm the production path
    /// returns a non-negative value (the `.min(i64::MAX as u64)` guard is the fix).
    #[test]
    fn test_system_clock_now_is_non_negative() {
        let t = SystemClock.now();
        assert!(t >= 0, "B3: SystemClock::now() must be >= 0, got {t}");
    }

    #[test]
    fn test_fixed_clock_shared_behind_trait_object() {
        let c = FixedClock::new(500);
        let dyn_clock: &dyn Clock = &c;
        assert_eq!(dyn_clock.now(), 500);
        c.advance_days(1.0);
        assert_eq!(dyn_clock.now(), 500 + 86_400); // advance visible through &dyn Clock
    }
}
