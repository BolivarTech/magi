// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Deterministic salience assignment at write time (D-11).
//!
//! The single public entry point is [`assign_salience`], a **pure function**
//! with no side-effects, time-calls, or RNG. Safe to call from any context.

use crate::agent::messages::Role;
use crate::memory::config::MemoryConfig;
use crate::memory::MemoryKind;

// Narrow allow: wired into the write path in Task 12; consumed by tests now.
#[allow(dead_code)]
/// Assigns a deterministic salience in `[0, 1]` at write time (D-11).
///
/// Pure: a function of `kind`, `text`, `role`, and config only —
/// **NO LLM, NO time, NO RNG** (R-06).
///
/// # Rules
///
/// - [`MemoryKind::Preference`] ⇒ returns the protected floor
///   `cfg.preference_salience` clamped to `[0.0, 1.0]`.
///   Such memories are never evicted by decay (REQ-09, REQ-35).
///
/// - Otherwise: starts at `cfg.default_salience`, then applies two
///   deterministic lifts before clamping to `[0.0, 1.0]`:
///
///   1. **Marker lift** — if `text` contains any substring from
///      `cfg.salience_markers` (matching format: **case-insensitive
///      substring match** — each marker is lowercased and tested
///      against `text.to_lowercase()`, iteration in declaration
///      order, short-circuiting on first hit), the score is raised
///      to at least `cfg.protect_salience_threshold`.
///
///   2. **Structural nudge** — [`Role::User`] messages receive a
///      fixed bump of `+0.05` (user statements are likelier to
///      contain durable facts or preferences).
///
/// # Determinism
///
/// Identical inputs always produce identical output (R-06). Marker
/// iteration uses a `Vec` in declaration order with a short-circuit
/// `any`, so there is no map/set iteration non-determinism. All
/// arithmetic is plain IEEE-754 addition and `f64::max`/`clamp`.
pub fn assign_salience(kind: MemoryKind, text: &str, role: Role, cfg: &MemoryConfig) -> f64 {
    if kind == MemoryKind::Preference {
        return cfg.preference_salience.clamp(0.0, 1.0);
    }

    let mut s = cfg.default_salience;

    // Marker lift: raise to the protected tier when any configured marker
    // appears as a case-insensitive substring of the text. Markers are
    // iterated in declaration order; iteration short-circuits on first hit.
    let text_lower = text.to_lowercase();
    if cfg
        .salience_markers
        .iter()
        .any(|m| text_lower.contains(m.to_lowercase().as_str()))
    {
        s = s.max(cfg.protect_salience_threshold);
    }

    // Structural nudge: user messages carry a small salience premium.
    if role == Role::User {
        s += 0.05_f64;
    }

    s.clamp(0.0, 1.0)
}

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
        let cfg = MemoryConfig {
            salience_markers: vec!["prefer".into(), "always".into()],
            ..MemoryConfig::default()
        };
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
