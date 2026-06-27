// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Deterministic memory-strength / decay model (D-18/D-19, CP2-H).
//!
//! Provides [`strength`], a pure function that scores each [`Memory`] in
//! `[0, 1]` given an externally-supplied `now` timestamp (via the caller's
//! injected [`Clock`](crate::memory::clock::Clock)).

use crate::memory::config::MemoryConfig;
use crate::memory::store::Memory;

/// Computes the deterministic memory strength in `[0, 1]` (D-18/D-19, CP2-H).
///
/// A pure function of its inputs — `now` (Unix seconds) is supplied by the
/// caller's injected `Clock`, never from `SystemTime::now()`.
///
/// # Formula
/// ```text
/// strength = (w_rec·recency + reinforcement + w_sal·salience)
///            / (w_rec + 1 + w_sal)
/// ```
/// where:
/// - `recency  = 0.5^(age_days / decay_half_life_days)`,
///   `age_days = max(now − last_accessed_at, 0) / 86_400`   (CP2-AD: clamped ≥ 0)
/// - `reinforcement = min(access_count, access_saturation_cap) / access_saturation_cap`
///   (D-19: bounded to `[0, 1]`; cap = 0 ⇒ reinforcement = 0.0)
/// - `salience = m.salience` (already in `[0, 1]`)
/// - Weights: `w_rec = cfg.weight_recency`, `w_sal = cfg.weight_salience`;
///   the reinforcement weight is the implicit `1.0`.
///
/// Each weighted term is in `[0, 1]` and the divisor is the sum of weights,
/// so the result is always in `[0, 1]`.
///
/// # Parameters
/// - `m`   — the memory record to score.
/// - `now` — current time in Unix seconds (from the injected `Clock`).
/// - `cfg` — subsystem configuration with decay/weight fields.
// Narrow allow: consumed by the forgetting / eviction pass (Task 9) and the
// retrieval reranker (Task 7 integration); wired into the agent in Task 12.
#[allow(dead_code)]
pub fn strength(m: &Memory, now: i64, cfg: &MemoryConfig) -> f64 {
    // Recency term (D-18): decay over wall-clock time via caller-supplied `now`.
    // CP2-AD: clamp negative age (backward clock jump) to 0 so recency = 1.0.
    let age_secs = (now - m.last_accessed_at).max(0) as f64;
    let age_days = age_secs / 86_400.0;
    let recency = 0.5_f64.powf(age_days / cfg.decay_half_life_days);

    // Reinforcement term (D-19): bounded access contribution.
    // cap = 0 disables reinforcement entirely (treat as 0.0).
    let reinforcement = if cfg.access_saturation_cap == 0 {
        0.0
    } else {
        m.access_count.min(cfg.access_saturation_cap) as f64 / cfg.access_saturation_cap as f64
    };

    // Salience is already in [0, 1] (assigned at write time by the salience module).
    let salience = m.salience;

    // Weighted sum normalized by the total weight so the result is in [0, 1].
    let w_rec = cfg.weight_recency;
    let w_sal = cfg.weight_salience;
    let divisor = w_rec + 1.0 + w_sal;

    (w_rec * recency + reinforcement + w_sal * salience) / divisor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::MemoryConfig;
    use crate::memory::store::Memory;
    use crate::memory::MemoryKind;

    fn mem(last_accessed_at: i64, salience: f64, access_count: u64) -> Memory {
        Memory {
            id: "m".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: "t".into(),
            embedding: vec![],
            model_id: "".into(),
            dim: 0,
            created_at: last_accessed_at,
            salience,
            access_count,
            last_accessed_at,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        }
    }

    #[test]
    fn test_older_memory_decays_below_recent() {
        // SC-10/43
        let cfg = MemoryConfig::default();
        let recent = mem(1_000_000, 0.5, 0);
        let old = mem(1_000_000 - 60 * 86_400, 0.5, 0); // 60 days older
        assert!(strength(&recent, 1_000_000, &cfg) > strength(&old, 1_000_000, &cfg));
    }

    #[test]
    fn test_decay_is_deterministic_and_clock_driven() {
        // SC-43
        let cfg = MemoryConfig::default();
        let m = mem(0, 0.5, 0);
        assert_eq!(strength(&m, 100, &cfg), strength(&m, 100, &cfg)); // identical
        assert!(strength(&m, 0, &cfg) > strength(&m, 90 * 86_400, &cfg)); // 90d later, weaker
    }

    #[test]
    fn test_strength_is_normalized_to_unit_interval() {
        // CP2-H
        let cfg = MemoryConfig::default();
        for m in [mem(0, 0.0, 0), mem(0, 1.0, u64::MAX), mem(0, 0.5, 10)] {
            let s = strength(&m, 365 * 86_400, &cfg);
            assert!((0.0..=1.0).contains(&s), "strength out of [0,1]: {s}");
        }
    }

    #[test]
    fn test_access_count_overflow_safe_and_reinforcement_bounded() {
        // SC-44
        let cfg = MemoryConfig::default();
        let huge = mem(0, 0.3, u64::MAX);
        let capped = mem(0, 0.3, cfg.access_saturation_cap);
        let _ = strength(&huge, 1_000, &cfg); // must not panic
        assert!((strength(&huge, 1_000, &cfg) - strength(&capped, 1_000, &cfg)).abs() < 1e-9);
    }

    #[test]
    fn test_negative_age_is_clamped() {
        // CP2-AD: backward clock jump
        let cfg = MemoryConfig::default();
        let m = mem(1_000, 0.5, 0);
        // now < last_accessed_at (clock went backward) => age clamps to 0,
        // recency = 1.0, identical to now == last_accessed_at;
        // no negative age / spurious strength.
        assert_eq!(strength(&m, 500, &cfg), strength(&m, 1_000, &cfg));
    }
}
