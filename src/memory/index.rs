// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! In-RAM vector index: brute-force exact (default) and optional HNSW ANN
//! behind the `ann` Cargo feature (CP2-A / D-05).
//!
//! # Design
//!
//! The default path ([`BruteForceIndex`]) is **exact** and **deterministic by
//! construction** — no seed is required and stable tie-breaking by `id_index`
//! makes it reproducible under any input ordering.
//!
//! The opt-in `ann` path (`InstantDistanceIndex`) uses an HNSW graph built
//! with a fixed seed for deterministic search results under the same input data
//! and same insertion order. Enable with `--features ann`; the feature is off
//! by default so the `instant-distance` crate is not compiled in release builds
//! unless explicitly requested.
//!
//! Both types implement [`VectorIndex`] and are constructed through its `build`
//! method, keeping the caller in [`retrieval`][crate::memory::retrieval] generic
//! over the index implementation.
//!
//! [`cosine`] is the shared similarity primitive; both paths use it for the
//! final per-result similarity value so the scores are directly comparable.

use std::cmp::Ordering;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Builds an in-RAM index over `(id_index, vector)` pairs and returns
/// cosine-ranked hits.
///
/// # Type parameter
/// The `id_index` values are opaque caller identifiers (positions in a
/// backing collection). The index preserves and returns them unchanged.
///
/// # Determinism (R-06)
/// Implementations must return the same results for the same `points`, `query`,
/// `k`, and `seed` on every call. Tie-breaking by ascending `id_index` is
/// **required** to ensure determinism when cosine scores are equal.
///
/// # Safety
/// Both `build` and `search` must never panic. Dimension mismatches between
/// the index vectors and the query are handled by returning 0.0 similarity
/// (or by the caller pre-filtering, which `recall` does via D-06).
pub trait VectorIndex {
    /// Builds the index from a slice of `(id_index, vector)` pairs.
    ///
    /// The `seed` is consumed by HNSW implementations for deterministic layer
    /// assignment; it is ignored by the brute-force implementation.
    fn build(points: &[(usize, Vec<f32>)], seed: u64) -> Self
    where
        Self: Sized;

    /// Returns the top-`k` hits as `(id_index, cosine)` pairs sorted by
    /// cosine similarity **descending**, tie-broken by ascending `id_index`.
    ///
    /// May return fewer than `k` results when fewer points are indexed.
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
}

// ─── Cosine primitive ─────────────────────────────────────────────────────────

/// Cosine similarity of two equal-length vectors; returns `0.0` when either
/// norm is zero or the lengths differ.
///
/// The result is clamped to `[-1.0, 1.0]` to absorb floating-point rounding.
///
/// # Examples
/// ```ignore
/// let sim = cosine(&[1.0, 0.0], &[1.0, 0.0]);
/// assert_eq!(sim, 1.0); // identical unit vectors
/// ```
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ─── BruteForceIndex ─────────────────────────────────────────────────────────

/// Exact brute-force cosine index — deterministic by construction.
///
/// The **default** retrieval path (D-05). Computes cosine similarity against
/// every indexed point for each query; correct for any dataset size and requires
/// no seed, no build-time overhead, and no extra dependencies.
///
/// For large corpora (> ~10K vectors) consider switching to
/// `InstantDistanceIndex` (requires `--features ann`).
pub struct BruteForceIndex {
    points: Vec<(usize, Vec<f32>)>,
}

impl VectorIndex for BruteForceIndex {
    /// Builds the index by cloning `points`. O(N) time and space.
    fn build(points: &[(usize, Vec<f32>)], _seed: u64) -> Self {
        Self {
            points: points.to_vec(),
        }
    }

    /// Exact cosine search. Skips points whose dimension differs from the query
    /// dimension (returns cosine = 0.0 for mismatches, which the caller has
    /// already filtered out via D-06 in [`recall`][crate::memory::retrieval::recall]).
    ///
    /// Results are sorted by cosine similarity **descending**; ties are broken
    /// by ascending `id_index` for deterministic ordering (R-06).
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .points
            .iter()
            .map(|(id_index, vec)| (*id_index, cosine(query, vec)))
            .collect();

        // Primary: score desc. Secondary: id_index asc (stable tie-break, R-06).
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        scored.truncate(k);
        scored
    }
}

// ─── InstantDistanceIndex (ANN, opt-in) ──────────────────────────────────────

/// HNSW approximate-nearest-neighbour index, built with [`instant-distance`].
///
/// Compiled **only** when the `ann` Cargo feature is enabled. The HNSW graph is
/// constructed with a fixed seed (`Builder::seed`) so that two builds from
/// identical inputs produce identical search results (CP2-A / R-06), provided
/// the dataset is small enough to avoid rayon-parallel layer construction
/// (for datasets ≤ `ceil(M / ml) ≈ 160` points, construction is single-threaded
/// and fully deterministic).
///
/// For correctness the `distance` function is the **cosine distance** (1 − cosine
/// similarity), keeping the HNSW metric consistent with the reranker.
/// Final results are re-scored with the exact [`cosine`] function so scores are
/// directly comparable with [`BruteForceIndex`] output.
#[cfg(feature = "ann")]
pub struct InstantDistanceIndex {
    hnsw: instant_distance::Hnsw<VecPoint>,
    /// Maps `PointId` (internal HNSW shuffle index) → original input position.
    ///
    /// HNSW shuffles points during construction; this table lets us recover the
    /// original `id_index` from the `PointId` returned by `search`.
    reverse_map: std::collections::HashMap<instant_distance::PointId, usize>,
    /// Original input position → `id_index` (the caller's identifier).
    id_mapping: Vec<usize>,
    /// Original vectors stored for exact cosine re-scoring after search.
    vectors: Vec<Vec<f32>>,
}

/// `Vec<f32>` wrapper that satisfies the `instant_distance::Point` trait by
/// measuring **cosine distance** (1 − cosine similarity). Values < 0 are clamped
/// to 0 so the HNSW metric remains non-negative.
#[cfg(feature = "ann")]
#[derive(Clone)]
struct VecPoint(Vec<f32>);

#[cfg(feature = "ann")]
impl instant_distance::Point for VecPoint {
    fn distance(&self, other: &Self) -> f32 {
        // cosine distance in [0, 2]; 0 = identical direction.
        (1.0 - cosine(&self.0, &other.0)).max(0.0)
    }
}

#[cfg(feature = "ann")]
impl VectorIndex for InstantDistanceIndex {
    /// Builds the HNSW index deterministically.
    ///
    /// The `seed` is forwarded to `Builder::seed`. Construction is
    /// single-threaded (and therefore deterministic) for datasets with at
    /// most `ceil(M / ml)` ≈ 160 points (M=32, ml≈0.2). Larger datasets
    /// use rayon for lower-layer insertion; results remain approximately
    /// correct but bitwise determinism is not guaranteed.
    fn build(points: &[(usize, Vec<f32>)], seed: u64) -> Self {
        use std::collections::HashMap;

        let id_mapping: Vec<usize> = points.iter().map(|(i, _)| *i).collect();
        let vectors: Vec<Vec<f32>> = points.iter().map(|(_, v)| v.clone()).collect();
        let vec_points: Vec<VecPoint> = vectors.iter().map(|v| VecPoint(v.clone())).collect();

        let (hnsw, ids) = instant_distance::Builder::default()
            .seed(seed)
            .build_hnsw(vec_points);

        // Build reverse map: PointId (internal shuffle index) → original input position.
        let mut reverse_map = HashMap::with_capacity(ids.len());
        for (original_idx, pid) in ids.iter().enumerate() {
            reverse_map.insert(*pid, original_idx);
        }

        Self {
            hnsw,
            reverse_map,
            id_mapping,
            vectors,
        }
    }

    /// Approximate search: queries the HNSW graph then re-scores hits with
    /// exact cosine for consistent ranking with [`BruteForceIndex`].
    ///
    /// Results are sorted by cosine similarity **descending**, tie-broken by
    /// ascending `id_index` (R-06).
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let qp = VecPoint(query.to_vec());
        let mut search = instant_distance::Search::default();

        let mut results: Vec<(usize, f32)> = self
            .hnsw
            .search(&qp, &mut search)
            .take(k)
            .filter_map(|item| {
                let &orig_idx = self.reverse_map.get(&item.pid)?;
                let id_index = self.id_mapping[orig_idx];
                let sim = cosine(query, &self.vectors[orig_idx]);
                Some((id_index, sim))
            })
            .collect();

        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        results
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bag-of-words helper (shared with retrieval tests) ──────────────────

    /// L2-normalised bag-of-words over a fixed-dim hash.  Texts sharing words
    /// get high cosine similarity.  The same helper is used in
    /// [`retrieval::tests`][super::super::retrieval::tests].
    pub(super) fn bow(text: &str, dim: usize) -> Vec<f32> {
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

    // ── cosine ────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_unit_vectors() {
        let v = vec![1.0f32, 0.0, 0.0];
        let sim = cosine(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical unit vectors → cosine = 1"
        );
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine(&a, &b)).abs() < 1e-6, "orthogonal → cosine = 0");
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let z = vec![0.0f32, 0.0, 0.0];
        let v = vec![1.0f32, 0.0, 0.0];
        assert_eq!(cosine(&z, &v), 0.0);
        assert_eq!(cosine(&v, &z), 0.0);
    }

    #[test]
    fn test_cosine_dim_mismatch_returns_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    // ── BruteForceIndex ───────────────────────────────────────────────────

    #[test]
    fn test_brute_force_nearest_is_correct() {
        // dim=64 gives enough bucket space to avoid hash collisions between the
        // "budget" word (shared by query and point 0) and unrelated words in
        // points 1 and 2 (which would falsely win at dim=16).
        let dim = 64usize;
        // Points: only point 0 shares the word "budget" with the query.
        let points: Vec<(usize, Vec<f32>)> = vec![
            (0, bow("budget allocation policy", dim)),
            (1, bow("network latency tuning", dim)),
            (2, bow("thread pool size", dim)),
        ];
        let query = bow("what is the budget", dim);
        let index = BruteForceIndex::build(&points, 42);
        let hits = index.search(&query, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].0, 0,
            "point 0 should be the nearest to the budget query"
        );
        assert!(hits[0].1 > 0.0, "cosine should be positive");
    }

    #[test]
    fn test_brute_force_tie_break_by_id_index_asc() {
        // Two identical vectors — tie-break must use ascending id_index.
        let v = vec![0.5f32, 0.5];
        let points: Vec<(usize, Vec<f32>)> = vec![(10, v.clone()), (5, v.clone())];
        let index = BruteForceIndex::build(&points, 42);
        let hits = index.search(&v, 2);
        assert_eq!(hits[0].0, 5, "lower id_index wins the tie");
        assert_eq!(hits[1].0, 10);
    }

    #[test]
    fn test_brute_force_returns_at_most_k() {
        let points: Vec<(usize, Vec<f32>)> =
            (0..10).map(|i| (i, bow(&format!("item {i}"), 8))).collect();
        let index = BruteForceIndex::build(&points, 42);
        let hits = index.search(&bow("item 3", 8), 3);
        assert!(hits.len() <= 3);
    }

    // ── InstantDistanceIndex (ANN) ─────────────────────────────────────────

    /// CP2-A: two [`InstantDistanceIndex`] builds from the same ≤64 deterministic
    /// points and seed return byte-identical top-5 results.
    ///
    /// Datasets ≤ ~160 points (M=32, ml≈0.2) produce a single HNSW layer whose
    /// construction is single-threaded and therefore reproducible across builds.
    #[cfg(feature = "ann")]
    #[test]
    fn test_ann_identical_rebuilds_are_byte_stable() {
        let dim = 32usize;
        let seed = 42u64;

        // 64 deterministic points → single-level HNSW → sequential construction
        let points: Vec<(usize, Vec<f32>)> = (0..64)
            .map(|i| {
                (
                    i,
                    bow(&format!("point number {i} with label {}", i % 8), dim),
                )
            })
            .collect();

        let query = bow("point label 3 feature", dim);

        let index1 = InstantDistanceIndex::build(&points, seed);
        let index2 = InstantDistanceIndex::build(&points, seed);

        let top5_1 = index1.search(&query, 5);
        let top5_2 = index2.search(&query, 5);

        assert!(
            !top5_1.is_empty(),
            "ANN should return at least one result for a non-empty index"
        );
        assert_eq!(
            top5_1, top5_2,
            "ANN builds with same seed and data must be byte-stable (CP2-A)"
        );
    }

    /// ANN top-1 must be within the brute-force top-3 (approximate oracle check).
    ///
    /// HNSW is an *approximate* algorithm — it is not guaranteed to return the
    /// exact nearest neighbour, but for a well-separated dataset the top-1
    /// result should be among the few highest-scoring brute-force hits. We use
    /// a dataset where the target point has a large similarity margin over the
    /// rest (unique query words shared only with point 0), so ANN returns it
    /// reliably. The assertion checks `top-3 membership` rather than exact top-1
    /// equality to account for the approximation guarantee.
    #[cfg(feature = "ann")]
    #[test]
    fn test_bruteforce_and_ann_agree_on_top1() {
        let dim = 64usize; // larger dim reduces hash collisions, widening the margin
        let seed = 42u64;

        // Point 0 is the clear nearest: it shares all query words exclusively.
        // Points 1..49 have entirely unrelated words and no query overlap.
        let mut pts: Vec<(usize, Vec<f32>)> = (1..50)
            .map(|i| {
                (
                    i,
                    bow(&format!("network latency server rack node {i}", i = i), dim),
                )
            })
            .collect();
        pts.insert(
            0,
            (0, bow("budget context token allocation policy limit", dim)),
        );

        let query = bow("budget context token allocation policy limit", dim);

        let bf = BruteForceIndex::build(&pts, seed);
        let ann = InstantDistanceIndex::build(&pts, seed);

        let bf_top3 = bf.search(&query, 3);
        let ann_top1 = ann.search(&query, 1);

        assert!(
            !bf_top3.is_empty(),
            "brute-force should find nearest points"
        );
        assert!(!ann_top1.is_empty(), "ANN should find a nearest point");

        let bf_top3_ids: Vec<usize> = bf_top3.iter().map(|(id, _)| *id).collect();
        assert!(
            bf_top3_ids.contains(&ann_top1[0].0),
            "ANN top-1 (id={}) must be within brute-force top-3 {:?} (oracle check)",
            ann_top1[0].0,
            bf_top3_ids
        );
    }
}
