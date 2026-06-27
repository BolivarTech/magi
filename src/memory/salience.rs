// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Deterministic salience assignment at write time (D-11).
//!
//! The single public entry point is [`assign_salience`], a **pure function**
//! with no side-effects, time-calls, or RNG. Safe to call from any context.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::Role;
    use crate::memory::config::MemoryConfig;
    use crate::memory::MemoryKind;

    #[test]
    fn test_salience_is_deterministic_and_preference_is_protected() {
        let cfg = MemoryConfig::default();
        let s1 = assign_salience(MemoryKind::Episodic, "note the budget", Role::User, &cfg);
        let s2 = assign_salience(MemoryKind::Episodic, "note the budget", Role::User, &cfg);
        assert_eq!(s1, s2); // R-06 determinism
        assert!((0.0..=1.0).contains(&s1));
        let p = assign_salience(MemoryKind::Preference, "always use rust", Role::User, &cfg);
        assert!(p >= cfg.protect_salience_threshold); // preference is protected
    }

    #[test]
    fn test_preference_marker_lifts_episodic_to_protected_tier() {
        let mut cfg = MemoryConfig::default();
        cfg.salience_markers = vec!["prefer".into(), "always".into()];
        let s = assign_salience(
            MemoryKind::Episodic,
            "I always prefer dark mode",
            Role::User,
            &cfg,
        );
        assert!(s >= cfg.protect_salience_threshold);
    }

    #[test]
    fn test_plain_episodic_uses_default_salience_floor() {
        let cfg = MemoryConfig::default();
        // No marker, assistant role → at least the base default (no spurious protection).
        let s = assign_salience(
            MemoryKind::Episodic,
            "the sky is blue",
            Role::Assistant,
            &cfg,
        );
        assert!(s >= cfg.default_salience && s < cfg.protect_salience_threshold);
    }
}
