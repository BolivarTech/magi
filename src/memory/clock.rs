// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

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

    #[test]
    fn test_fixed_clock_shared_behind_trait_object() {
        let c = FixedClock::new(500);
        let dyn_clock: &dyn Clock = &c;
        assert_eq!(dyn_clock.now(), 500);
        c.advance_days(1.0);
        assert_eq!(dyn_clock.now(), 500 + 86_400); // advance visible through &dyn Clock
    }
}
