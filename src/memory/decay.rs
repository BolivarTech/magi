// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Deterministic memory-strength / decay model and forgetting/eviction passes
//! (D-07, D-18, D-19, REQ-09, REQ-32, CP2-AD, CP2-Y).
//!
//! # Public surface
//! - [`strength`] — pure scoring function, `[0, 1]`.
//! - [`run_forgetting`] — evicts sub-threshold non-protected memories (SC-12, SC-13).
//! - [`enforce_size_cap`] — enforces `max_records` hard ceiling (SC-15, CP2-Y).
//! - [`purge_expired_archives`] — hard-deletes archives past retention window (SC-34).

use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::error::MemoryError;
use crate::memory::store::{Memory, VectorStore};
use crate::memory::MemoryKind;

// ─── strength ─────────────────────────────────────────────────────────────────

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
// Narrow allow: consumed by the forgetting/eviction pass (this module, Task 9)
// and the retrieval reranker (Task 7); wired into the agent in Task 12.
#[allow(dead_code)]
pub fn strength(m: &Memory, now: i64, cfg: &MemoryConfig) -> f64 {
    // Recency term (D-18): decay over wall-clock time via caller-supplied `now`.
    // CP2-AD: clamp negative age (backward clock jump) to 0 so recency = 1.0.
    let age_secs = (now - m.last_accessed_at).max(0) as f64;
    let age_days = age_secs / 86_400.0;
    // B1: guard against half_life=0 producing NaN (0.5^(x/0) = NaN).
    let half_life = cfg.decay_half_life_days.max(f64::MIN_POSITIVE);
    let recency = 0.5_f64.powf(age_days / half_life);

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

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Returns `true` if `m` must not be decay-evicted.
///
/// A memory is **protected** iff it is a durable preference (`kind ==
/// Preference`) OR its salience meets the protection threshold
/// (`salience >= cfg.protect_salience_threshold`).
///
/// Protected memories are excluded from [`run_forgetting`]'s eviction loop
/// and are only removed by [`enforce_size_cap`] as a last resort (CP2-Y).
fn is_protected(m: &Memory, cfg: &MemoryConfig) -> bool {
    m.kind == MemoryKind::Preference || m.salience >= cfg.protect_salience_threshold
}

/// Applies the D-07 eviction action for a single memory `id`.
///
/// - `evicted_retention_days == 0` → hard-delete immediately.
/// - `-1` or `N > 0` → archive via `set_evicted(id, Some(now))`.
///   For `N > 0`, [`purge_expired_archives`] performs the deferred hard-delete
///   once the retention window elapses.
///
/// # Errors
/// [`MemoryError::Storage`] on any SQLite failure.
async fn apply_eviction(
    store: &dyn VectorStore,
    id: &str,
    now: i64,
    cfg: &MemoryConfig,
) -> Result<(), MemoryError> {
    if cfg.evicted_retention_days == 0 {
        store.hard_delete(&[id.to_string()]).await
    } else {
        // -1 (archive forever) or N>0 (archive; purge_expired_archives cleans up later).
        store.set_evicted(id, Some(now)).await
    }
}

// ─── run_forgetting ───────────────────────────────────────────────────────────

/// Evicts active, NON-protected memories whose `strength < cfg.forget_strength_threshold`,
/// lowest-strength first, at most `cfg.max_evictions_per_pass` per call (CP2-AD clock-jump
/// guard). Eviction applies the D-07 `evicted_retention_days` action. Returns the count evicted.
///
/// # Scope
/// Only memories in `scope` are considered; typically `"root"`.
///
/// # Protection rule (REQ-09, REQ-35)
/// Memories where `kind == Preference` or `salience >= protect_salience_threshold`
/// are **never** evicted by this function. Use [`enforce_size_cap`] to handle
/// the hard-ceiling last-resort case (CP2-Y).
///
/// # Clock-jump guard (CP2-AD)
/// The per-pass cap (`max_evictions_per_pass`) prevents a large backward/forward
/// clock jump from mass-evicting healthy memories in a single run.
///
/// # Errors
/// [`MemoryError::Storage`] or [`MemoryError::Crypto`] on store failures.
// Narrow allow: wired into the agent session loop in Task 12.
#[allow(dead_code)]
pub async fn run_forgetting(
    store: &dyn VectorStore,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
    scope: &str,
) -> Result<usize, MemoryError> {
    let now = clock.now();
    let active = store.active(scope).await?;

    // Collect non-protected candidates below the threshold.
    let mut candidates: Vec<(String, f64)> = active
        .iter()
        .filter(|m| !is_protected(m, cfg))
        .filter_map(|m| {
            let s = strength(m, now, cfg);
            if s < cfg.forget_strength_threshold {
                Some((m.id.clone(), s))
            } else {
                None
            }
        })
        .collect();

    // Sort ascending by strength — weakest evicted first.
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // CP2-AD: cap per-pass evictions to guard against mass-eviction on a clock jump.
    let mut count = 0usize;
    for (id, _) in candidates.iter().take(cfg.max_evictions_per_pass) {
        apply_eviction(store, id, now, cfg).await?;
        count += 1;
    }

    Ok(count)
}

// ─── enforce_size_cap ─────────────────────────────────────────────────────────

/// Enforces the hard `max_records` ceiling (CP2-Y, 0 = no cap): while `active(scope).len() >
/// max_records`, evicts the lowest-strength UNPROTECTED memory; if only protected remain and
/// the count is still over the cap, evicts the lowest-strength PROTECTED one as a LAST RESORT
/// (one-time WARN via eprintln). Returns the count evicted.
///
/// A `max_records` value of `0` disables the ceiling entirely (operator opt-out).
///
/// # Errors
/// [`MemoryError::Storage`] or [`MemoryError::Crypto`] on store failures.
// Narrow allow: wired into the agent session loop in Task 12.
#[allow(dead_code)]
pub async fn enforce_size_cap(
    store: &dyn VectorStore,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
    scope: &str,
) -> Result<usize, MemoryError> {
    if cfg.max_records == 0 {
        return Ok(0); // operator opt-out
    }

    let now = clock.now();
    let active = store.active(scope).await?;
    let total = active.len();

    if total <= cfg.max_records {
        return Ok(0);
    }

    // Partition into (unprotected, protected), each sorted by strength ASC.
    let mut unprotected: Vec<(String, f64)> = active
        .iter()
        .filter(|m| !is_protected(m, cfg))
        .map(|m| (m.id.clone(), strength(m, now, cfg)))
        .collect();
    unprotected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut protected: Vec<(String, f64)> = active
        .iter()
        .filter(|m| is_protected(m, cfg))
        .map(|m| (m.id.clone(), strength(m, now, cfg)))
        .collect();
    protected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = total;
    let mut count = 0usize;

    // Phase 1: evict lowest-strength unprotected memories first.
    for (id, _) in &unprotected {
        if remaining <= cfg.max_records {
            break;
        }
        apply_eviction(store, id, now, cfg).await?;
        count += 1;
        remaining -= 1;
    }

    // Phase 2 (CP2-Y last resort): evict protected memories only if still over cap.
    if remaining > cfg.max_records {
        eprintln!(
            "WARN [magi-memory] enforce_size_cap: evicting protected memories as last resort \
             (active={remaining}, cap={}, CP2-Y)",
            cfg.max_records
        );
        for (id, _) in &protected {
            if remaining <= cfg.max_records {
                break;
            }
            apply_eviction(store, id, now, cfg).await?;
            count += 1;
            remaining -= 1;
        }
    }

    Ok(count)
}

// ─── purge_expired_archives ───────────────────────────────────────────────────

/// Hard-deletes archived memories whose retention window has elapsed: only when
/// `cfg.evicted_retention_days > 0` and `clock.now() - evicted_at > retention_days*86400`.
/// (`-1` never purges; `0` memories were already hard-deleted at eviction.) Returns count purged.
///
/// This is the deferred half of the D-07 `N > 0` eviction policy. Call it
/// periodically (e.g. at session close or on a background timer).
///
/// # Errors
/// [`MemoryError::Storage`] or [`MemoryError::Crypto`] on store failures.
// Narrow allow: wired into the agent session loop in Task 12.
#[allow(dead_code)]
pub async fn purge_expired_archives(
    store: &dyn VectorStore,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
) -> Result<usize, MemoryError> {
    if cfg.evicted_retention_days <= 0 {
        // -1 = archive forever (never purge); 0 = already hard-deleted at eviction time.
        return Ok(0);
    }

    let now = clock.now();
    let retention_secs = cfg.evicted_retention_days.saturating_mul(86_400);
    let archived = store.archived().await?;

    let expired_ids: Vec<String> = archived
        .into_iter()
        .filter_map(|m| {
            m.evicted_at.and_then(|evicted_at| {
                if now - evicted_at > retention_secs {
                    Some(m.id)
                } else {
                    None
                }
            })
        })
        .collect();

    let count = expired_ids.len();
    if !expired_ids.is_empty() {
        store.hard_delete(&expired_ids).await?;
    }
    Ok(count)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::clock::FixedClock;
    use crate::memory::config::MemoryConfig;
    use crate::memory::error::EmbeddingError;
    use crate::memory::retrieval::recall;
    use crate::memory::store::{Memory, SqliteVectorStore, VectorStore};
    use crate::memory::MemoryKind;
    use crate::system::database::EncryptedSqliteMemory;
    use async_trait::async_trait;

    // ── test-store helper (mirrors store.rs) ─────────────────────────────────

    fn make_test_store() -> (tempfile::NamedTempFile, SqliteVectorStore) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap();
        (tmp, store)
    }

    // ── memory constructor helpers ────────────────────────────────────────────

    /// A very old, low-salience, zero-access memory — strength ≈ 0 under a
    /// far-future `now` (e.g. `clock.now() = 1_000 * 86_400`).
    fn weak_mem(id: &str, kind: MemoryKind, salience: f64) -> Memory {
        Memory {
            id: id.into(),
            session_id: "s".into(),
            kind,
            text: "weak memory".into(),
            embedding: vec![],
            model_id: "".into(),
            dim: 0,
            created_at: 0,
            salience,
            access_count: 0,
            last_accessed_at: 0, // epoch start → very old under any future `now`
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        }
    }

    // ── existing strength tests (unchanged) ───────────────────────────────────

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

    // ── forgetting tests (Task 9) ─────────────────────────────────────────────

    /// SC-12: a low-strength non-protected memory is evicted and disappears from
    /// the active set; the store stays consistent.
    #[tokio::test]
    async fn test_sub_threshold_memory_is_evicted_and_excluded() {
        let (_tmp, store) = make_test_store();
        // "now" = 1000 days from epoch; the weak memory (last_accessed_at=0) has age=1000d.
        let now_secs = 1_000i64 * 86_400;
        let clock = FixedClock::new(now_secs);
        let cfg = MemoryConfig {
            forget_strength_threshold: 0.5, // high threshold → weak mem will be evicted
            evicted_retention_days: -1,     // archive mode
            ..MemoryConfig::default()
        };

        let weak = weak_mem("weak", MemoryKind::Episodic, 0.0);
        // Verify it is actually below threshold (sanity-check test setup).
        assert!(
            strength(&weak, now_secs, &cfg) < cfg.forget_strength_threshold,
            "test setup: weak memory must be below threshold"
        );

        store.insert(&weak).await.unwrap();
        assert_eq!(store.active("root").await.unwrap().len(), 1);

        let evicted = run_forgetting(&store, &clock, &cfg, "root").await.unwrap();
        assert_eq!(evicted, 1, "exactly 1 memory should be evicted (SC-12)");
        assert!(
            store.active("root").await.unwrap().is_empty(),
            "evicted memory must not appear in active set (SC-12)"
        );
    }

    /// SC-13: preference and high-salience memories are never evicted by the
    /// forgetting pass even if their strength is below the threshold.
    #[tokio::test]
    async fn test_preference_and_high_salience_never_evicted_by_decay() {
        let (_tmp, store) = make_test_store();
        let now_secs = 1_000i64 * 86_400;
        let clock = FixedClock::new(now_secs);
        let cfg = MemoryConfig {
            forget_strength_threshold: 0.99, // almost everything will be below threshold
            evicted_retention_days: -1,
            ..MemoryConfig::default()
        };

        // Non-protected low-strength memory (should be evicted).
        let evictable = weak_mem("evictable", MemoryKind::Episodic, 0.0);
        // Protected by kind=Preference.
        let pref = weak_mem("pref", MemoryKind::Preference, 0.0);
        // Protected by high salience (>= protect_salience_threshold = 0.9).
        let hi_sal = weak_mem("hi_sal", MemoryKind::Episodic, 0.95);

        store.insert(&evictable).await.unwrap();
        store.insert(&pref).await.unwrap();
        store.insert(&hi_sal).await.unwrap();

        run_forgetting(&store, &clock, &cfg, "root").await.unwrap();
        let active_ids: Vec<String> = store
            .active("root")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();

        assert!(
            !active_ids.contains(&"evictable".to_string()),
            "non-protected memory must be evicted (SC-13 control)"
        );
        assert!(
            active_ids.contains(&"pref".to_string()),
            "Preference memory must NOT be evicted (SC-13)"
        );
        assert!(
            active_ids.contains(&"hi_sal".to_string()),
            "high-salience memory must NOT be evicted (SC-13)"
        );
    }

    /// SC-34: tests all three D-07 retention policies:
    /// - `-1` (archive): row stays on disk, excluded from `active`, visible via `archived()`.
    /// - `0`  (immediate delete): row is hard-deleted, `get()` returns `None`.
    /// - `N>0` (delayed): row is archived now; `purge_expired_archives` deletes it
    ///   only after the window elapses.
    #[tokio::test]
    async fn test_retention_modes_archive_immediate_and_delayed() {
        // ── Case A: retention = -1 (archive forever) ─────────────────────────
        {
            let (_tmp, store) = make_test_store();
            let clock = FixedClock::new(1_000 * 86_400);
            let cfg = MemoryConfig {
                forget_strength_threshold: 0.99,
                evicted_retention_days: -1,
                ..MemoryConfig::default()
            };
            store
                .insert(&weak_mem("a", MemoryKind::Episodic, 0.0))
                .await
                .unwrap();

            run_forgetting(&store, &clock, &cfg, "root").await.unwrap();

            assert!(
                store.active("root").await.unwrap().is_empty(),
                "Case A: evicted memory must not be in active"
            );
            let ar = store.archived().await.unwrap();
            assert_eq!(
                ar.len(),
                1,
                "Case A: evicted memory must appear in archived()"
            );
            assert_eq!(ar[0].id, "a");
        }

        // ── Case B: retention = 0 (immediate hard-delete) ────────────────────
        {
            let (_tmp, store) = make_test_store();
            let clock = FixedClock::new(1_000 * 86_400);
            let cfg = MemoryConfig {
                forget_strength_threshold: 0.99,
                evicted_retention_days: 0,
                ..MemoryConfig::default()
            };
            store
                .insert(&weak_mem("b", MemoryKind::Episodic, 0.0))
                .await
                .unwrap();

            run_forgetting(&store, &clock, &cfg, "root").await.unwrap();

            assert!(
                store.active("root").await.unwrap().is_empty(),
                "Case B: evicted memory must not be in active"
            );
            assert!(
                store.get("b").await.unwrap().is_none(),
                "Case B: hard-deleted memory must not be retrievable via get()"
            );
        }

        // ── Case C: retention = 7 (archive; delete after 7 days) ─────────────
        {
            let (_tmp, store) = make_test_store();
            let start = 1_000i64 * 86_400;
            let clock = FixedClock::new(start);
            let retention_days = 7i64;
            let cfg = MemoryConfig {
                forget_strength_threshold: 0.99,
                evicted_retention_days: retention_days,
                ..MemoryConfig::default()
            };
            store
                .insert(&weak_mem("c", MemoryKind::Episodic, 0.0))
                .await
                .unwrap();

            // Evict (archives the row).
            run_forgetting(&store, &clock, &cfg, "root").await.unwrap();
            assert_eq!(
                store.archived().await.unwrap().len(),
                1,
                "Case C: archived immediately after eviction"
            );

            // Before the retention window: purge should NOT delete it.
            clock.advance_days(3.0); // 3 days later — under the 7-day window
            let purged = purge_expired_archives(&store, &clock, &cfg).await.unwrap();
            assert_eq!(purged, 0, "Case C: must not purge within retention window");
            assert_eq!(
                store.archived().await.unwrap().len(),
                1,
                "Case C: row still archived within window"
            );

            // After the retention window: purge MUST delete it.
            clock.advance_days(5.0); // total 8 days — past the 7-day window
            let purged = purge_expired_archives(&store, &clock, &cfg).await.unwrap();
            assert_eq!(
                purged, 1,
                "Case C: must purge after retention window (SC-34)"
            );
            assert!(
                store.get("c").await.unwrap().is_none(),
                "Case C: hard-deleted after retention window"
            );
        }
    }

    /// SC-15: the hard size ceiling evicts the weakest UNPROTECTED memories first.
    #[tokio::test]
    async fn test_size_cap_evicts_lowest_strength_unprotected_first() {
        let (_tmp, store) = make_test_store();
        let now_secs = 1_000i64 * 86_400;
        let clock = FixedClock::new(now_secs);
        // Allow 2 active records; we'll insert 5 unprotected memories of varying age.
        let cfg = MemoryConfig {
            max_records: 2,
            evicted_retention_days: -1,
            ..MemoryConfig::default()
        };

        // Insert 5 memories with different last_accessed_at (controls strength).
        // Older → weaker; newest two should survive.
        for i in 0..5usize {
            let last = (i as i64) * 10 * 86_400; // 0, 10, 20, 30, 40 days from epoch
            let m = Memory {
                id: format!("m{i}"),
                session_id: "s".into(),
                kind: MemoryKind::Episodic,
                text: "t".into(),
                embedding: vec![],
                model_id: "".into(),
                dim: 0,
                created_at: last,
                salience: 0.0,
                access_count: 0,
                last_accessed_at: last,
                superseded_by: None,
                evicted_at: None,
                scope: "root".into(),
                distilled_at: None,
            };
            store.insert(&m).await.unwrap();
        }

        let evicted = enforce_size_cap(&store, &clock, &cfg, "root")
            .await
            .unwrap();

        // 5 inserted, cap = 2 → must evict 3.
        assert_eq!(evicted, 3, "must evict 3 to reach cap of 2 (SC-15)");
        let remaining_ids: Vec<String> = store
            .active("root")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(remaining_ids.len(), 2, "exactly 2 should remain after cap");

        // The two strongest (most recent) should survive: m3 and m4.
        assert!(
            remaining_ids.contains(&"m3".to_string()),
            "m3 (second-newest) must survive (SC-15)"
        );
        assert!(
            remaining_ids.contains(&"m4".to_string()),
            "m4 (newest) must survive (SC-15)"
        );
    }

    /// CP2-Y: when the store contains ONLY protected memories and the cap is
    /// exceeded, `enforce_size_cap` still evicts as a last resort.
    #[tokio::test]
    async fn test_size_cap_is_hard_ceiling_even_against_protected_flood() {
        let (_tmp, store) = make_test_store();
        let clock = FixedClock::new(1_000 * 86_400);
        let cfg = MemoryConfig {
            max_records: 2,
            evicted_retention_days: -1,
            ..MemoryConfig::default()
        };

        // Insert 5 preference memories (all protected, none evictable by decay).
        for i in 0..5usize {
            store
                .insert(&weak_mem(&format!("p{i}"), MemoryKind::Preference, 0.0))
                .await
                .unwrap();
        }

        let evicted = enforce_size_cap(&store, &clock, &cfg, "root")
            .await
            .unwrap();

        // Must have evicted 3 as last resort (CP2-Y).
        assert_eq!(
            evicted, 3,
            "CP2-Y: must evict protected memories as last resort"
        );
        assert_eq!(
            store.active("root").await.unwrap().len(),
            2,
            "hard ceiling must be respected even against a protected flood"
        );
    }

    /// CP2-AD: when many memories are below threshold, a single pass must NOT
    /// mass-evict beyond `max_evictions_per_pass`.
    #[tokio::test]
    async fn test_clock_jump_does_not_mass_evict() {
        let (_tmp, store) = make_test_store();
        let clock = FixedClock::new(1_000 * 86_400);
        let cap = 5usize;
        let cfg = MemoryConfig {
            forget_strength_threshold: 0.99, // almost everything is eligible
            max_evictions_per_pass: cap,
            evicted_retention_days: -1,
            ..MemoryConfig::default()
        };

        // Insert 20 sub-threshold memories.
        for i in 0..20usize {
            store
                .insert(&weak_mem(&format!("m{i}"), MemoryKind::Episodic, 0.0))
                .await
                .unwrap();
        }

        let evicted = run_forgetting(&store, &clock, &cfg, "root").await.unwrap();

        // The cap must be respected.
        assert!(
            evicted <= cap,
            "must not evict more than max_evictions_per_pass (CP2-AD)"
        );
        // At least some must be evicted (proving forgetting ran).
        assert!(evicted > 0, "at least one memory should be evicted");
        assert_eq!(
            evicted, cap,
            "when all memories are sub-threshold, exactly max_evictions_per_pass should be evicted"
        );
    }

    // ── Soft supersession test (SC-14) ────────────────────────────────────────

    /// Bag-of-words embedding (L2-normalised): same as retrieval.rs test helper.
    fn bow(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0f32; dim];
        for w in text.to_lowercase().split_whitespace() {
            let h = w
                .bytes()
                .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
                % dim;
            v[h] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    struct FakeEmbedder {
        dim: usize,
        model: String,
    }

    #[async_trait]
    impl crate::memory::embedding::EmbeddingProvider for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|t| bow(t, self.dim)).collect())
        }
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    /// SC-14: soft supersession — after inserting F1 and then F2 about the same
    /// subject (identical embedding via FakeEmbedder), `recall` ranks F2 above F1
    /// *before* any distiller runs, solely via the recency term of the reranker.
    /// No new code is needed — this falls out of the existing Task-7 reranker.
    #[tokio::test]
    async fn test_soft_supersession_recency_ranks_newer_first() {
        let (_tmp, store) = make_test_store();
        let dim = 32usize;
        let emb = FakeEmbedder {
            dim,
            model: "fake".into(),
        };
        // F1 is older; F2 is newer (higher last_accessed_at).
        let now_secs = 1_000_000i64;
        let clock = FixedClock::new(now_secs);
        let cfg = MemoryConfig {
            top_k: 2,
            ..MemoryConfig::default()
        };

        let subject_text = "the api budget is 8000 tokens";
        let embedding = bow(subject_text, dim);

        let f1 = Memory {
            id: "F1".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: subject_text.into(),
            embedding: embedding.clone(),
            model_id: "fake".into(),
            dim,
            created_at: 100,
            salience: 0.5,
            access_count: 0,
            last_accessed_at: 100, // older
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };
        let f2 = Memory {
            id: "F2".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: subject_text.into(),
            embedding: embedding.clone(),
            model_id: "fake".into(),
            dim,
            created_at: 900_000,
            salience: 0.5,
            access_count: 0,
            last_accessed_at: 900_000, // much more recent
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };

        store.insert(&f1).await.unwrap();
        store.insert(&f2).await.unwrap();

        let results = recall(&store, &emb, &clock, &cfg, subject_text, 0, "root")
            .await
            .unwrap();

        assert_eq!(results.len(), 2, "both memories should be returned");
        assert_eq!(
            results[0].memory.id, "F2",
            "SC-14: the more recent fact (F2) must rank above the older (F1) via recency"
        );
    }

    // ── B1: NaN guard for zero half-life ─────────────────────────────────────

    /// B1: `strength()` with `decay_half_life_days = 0.0` must return a finite
    /// value, not NaN (`0.5.powf(x/0.0)` evaluates to NaN without the guard).
    #[test]
    fn test_strength_with_zero_half_life_returns_finite() {
        let cfg = MemoryConfig {
            decay_half_life_days: 0.0,
            ..MemoryConfig::default()
        };
        let m = mem(0, 0.5, 0);
        let s = strength(&m, 1000, &cfg);
        assert!(
            s.is_finite(),
            "B1: strength() with half_life=0.0 must be finite, got {s}"
        );
        assert!(
            !s.is_nan(),
            "B1: strength() must not be NaN with half_life=0.0"
        );
    }

    // ── B2: saturating_mul guard for i64 overflow in purge_expired_archives ───

    /// B2: `purge_expired_archives` with a near-`i64::MAX` retention value must
    /// not panic. With plain multiplication `i64::MAX * 86_400` overflows;
    /// `saturating_mul` returns `i64::MAX`.
    #[tokio::test]
    async fn test_purge_expired_archives_with_large_retention_does_not_panic() {
        let (_tmp, store) = make_test_store();
        let clock = FixedClock::new(1_000 * 86_400);
        let cfg = MemoryConfig {
            // i64::MAX / 86_400 + 1 would overflow with plain multiplication.
            evicted_retention_days: i64::MAX / 86_400 + 1,
            ..MemoryConfig::default()
        };
        // Must not panic; no archived rows → result is 0.
        let result = purge_expired_archives(&store, &clock, &cfg).await;
        assert!(
            result.is_ok(),
            "B2: purge must not panic on large retention: {result:?}"
        );
    }

    // ── Race / concurrency test (CP2-D) ───────────────────────────────────────

    /// CP2-D: concurrent `run_forgetting` + `active()` over an
    /// `Arc<SqliteVectorStore>` must complete without panic; the shared `Mutex`
    /// inside `SqliteVectorStore` serializes access.
    #[tokio::test]
    async fn test_recall_and_forgetting_are_serialized() {
        let (_tmp, store) = make_test_store();
        let store = std::sync::Arc::new(store);

        let now_secs = 1_000i64 * 86_400;
        let clock = std::sync::Arc::new(FixedClock::new(now_secs));
        let cfg = MemoryConfig {
            forget_strength_threshold: 0.99, // all inserted memories will be below threshold
            max_evictions_per_pass: 20,
            evicted_retention_days: -1,
            ..MemoryConfig::default()
        };

        // Insert 20 sub-threshold memories.
        for i in 0..20usize {
            store
                .insert(&weak_mem(&format!("m{i}"), MemoryKind::Episodic, 0.0))
                .await
                .unwrap();
        }

        let store1 = store.clone();
        let store2 = store.clone();
        let clock1 = clock.clone();
        let cfg1 = cfg.clone();

        // Spawn concurrent forgetting + active() — the Mutex serializes them.
        let h_forget = tokio::spawn(async move {
            run_forgetting(store1.as_ref(), clock1.as_ref(), &cfg1, "root").await
        });
        let h_active = tokio::spawn(async move { store2.active("root").await });

        let (r_forget, r_active) = tokio::join!(h_forget, h_active);
        let evicted = r_forget.unwrap().unwrap();
        let _ = r_active.unwrap().unwrap(); // must not panic

        // At least some evictions must have occurred (proves forgetting ran).
        assert!(
            evicted > 0,
            "at least one memory should have been evicted (CP2-D)"
        );
        // Store must be consistent: no active memories (all evicted) or at most
        // those active() observed before forgetting ran.
        let remaining = store.active("root").await.unwrap();
        assert!(
            remaining.len() + evicted <= 20,
            "active + evicted must not exceed the total (store consistency, CP2-D)"
        );
    }
}
