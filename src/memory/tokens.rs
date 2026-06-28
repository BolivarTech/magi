// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Deterministic token-counting heuristic and budget reservation (D-02/D-16).
//!
//! These are **pure functions** — no I/O, no time, no RNG — consumed by the
//! context assembler (Task 11) to enforce the P3 context budget invariant.

// Narrow allow: both functions are the stable P3 seam consumed by the context assembler
// (Task 11). They are public API but have no caller yet in this task.
#[allow(dead_code)]
/// Conservative deterministic token estimate: `ceil(chars / chars_per_token)` (D-02/D-16).
///
/// `chars` is the Unicode scalar count (`text.chars().count()`), NOT bytes. A non-positive or
/// non-finite `chars_per_token` falls back to 1.0 (so it never divides by zero / returns NaN).
///
/// # Tuning guidance (D-16)
///
/// Choose `chars_per_token` conservatively for the script/content type to avoid
/// underestimating (which would cause the assembler to exceed the model context):
///
/// | Content | Recommended `chars_per_token` |
/// |---------|-------------------------------|
/// | Spanish / Latin script prose | ~3.5 (default) |
/// | English / source code | ~3.0 |
/// | CJK (Chinese / Japanese / Korean) | ~2.0 |
///
/// A smaller value produces a *larger* (more conservative) estimate, which keeps the
/// assembled context safely below the hard budget.
///
/// # Examples
///
/// ```
/// use magi_rs::memory::tokens::estimate_tokens;
///
/// assert_eq!(estimate_tokens("abcdefg", 3.5), 2); // ceil(7/3.5) = 2
/// assert_eq!(estimate_tokens("", 3.5), 0);
/// assert_eq!(estimate_tokens("abc", 0.0), 3);     // bad cpt → fallback 1.0
/// ```
/// Converts an `f64` to `usize` with explicit saturation semantics.
///
/// Rust 1.45+ already saturates out-of-range `f64 as usize` casts
/// (NaN→0, negative→0, huge→`usize::MAX`), but this helper makes the
/// contract self-documenting at every call site.
fn f64_to_usize_saturating(x: f64) -> usize {
    // `max(0.0)` maps NaN and negatives to 0.0 (IEEE 754 `max` returns the
    // non-NaN argument when one input is NaN). `min(...)` caps at `usize::MAX
    // as f64` (≈ 2^64 on 64-bit); values above that still saturate to
    // `usize::MAX` via the `as` cast.
    x.max(0.0).min(usize::MAX as f64) as usize
}

pub fn estimate_tokens(text: &str, chars_per_token: f64) -> usize {
    let count = text.chars().count();
    if count == 0 {
        return 0;
    }
    // `MemoryConfig::validate()` is the primary guard — it rejects
    // `chars_per_token <= 0.0` at config-load time so the assembler
    // never reaches this branch with a bad value.  The fallback below is
    // defense-in-depth for callers that construct `MemoryConfig` directly
    // (e.g. unit tests with hand-built configs) without calling `validate()`.
    // The fallback value of 1.0 (one char per token) is more conservative
    // than a smaller minimum like 0.1: it produces a larger estimate and
    // thereby keeps the assembled context safely below the budget cap.
    let cpt = if chars_per_token > 0.0 && chars_per_token.is_finite() {
        chars_per_token
    } else {
        1.0
    };
    // Use the saturating helper to make the f64→usize contract self-documenting
    // (Rust 1.45+: out-of-range casts saturate, but explicit > implicit).
    f64_to_usize_saturating((count as f64 / cpt).ceil())
}

// Narrow allow: consumed by the context assembler in Task 11.
#[allow(dead_code)]
/// Usable budget after reserving response headroom and a safety margin (D-16):
/// `budget - headroom - ceil(budget * margin_ratio)`, saturating at 0 (never underflows).
///
/// The caller should set `margin_ratio` to a small fraction (e.g. `0.1`) so that
/// even when `estimate_tokens` underestimates (due to mixed scripts or short tokens)
/// there is still a cushion that prevents the real token count from exceeding the model limit.
///
/// A non-finite or negative `margin_ratio` is treated as `0.0` (no margin reserved).
///
/// # Examples
///
/// ```
/// use magi_rs::memory::tokens::budget_after_margin;
///
/// assert_eq!(budget_after_margin(8000, 1024, 0.1), 6176); // 8000 - 1024 - 800
/// assert_eq!(budget_after_margin(100, 1024, 0.1), 0);     // saturates, never underflows
/// ```
pub fn budget_after_margin(budget: usize, headroom: usize, margin_ratio: f64) -> usize {
    let margin = if margin_ratio > 0.0 && margin_ratio.is_finite() {
        // Compute via f64 to avoid integer overflow, then cast with saturation
        // (Rust 1.45+: out-of-range f64→usize casts saturate rather than wrap).
        // Clamp to `budget` so a large ratio or extreme budget cannot produce a
        // margin that exceeds the budget itself (J2: robustness for usize::MAX).
        ((budget as f64 * margin_ratio).ceil() as usize).min(budget)
    } else {
        0
    };
    budget.saturating_sub(headroom).saturating_sub(margin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_estimate_is_conservative_and_margin_reserves_headroom() {
        assert_eq!(estimate_tokens("abcdefg", 3.5), 2); // ceil(7/3.5) = 2
        assert_eq!(estimate_tokens("", 3.5), 0);
        assert_eq!(budget_after_margin(8000, 1024, 0.1), 8000 - 1024 - 800);
    }

    #[test]
    fn test_budget_after_margin_saturates_to_zero() {
        assert_eq!(budget_after_margin(100, 1024, 0.1), 0); // never underflow
    }

    #[test]
    fn test_estimate_counts_unicode_scalars_and_handles_bad_cpt() {
        assert_eq!(estimate_tokens("héllo", 1.0), 5); // 5 chars, not bytes
        assert_eq!(estimate_tokens("abc", 0.0), 3); // bad cpt → fallback 1.0
    }

    #[test]
    fn test_budget_holds_with_cjk_chars_per_token() {
        // CP2-Q: with a CJK-tuned cpt (~2.0) the estimate is larger (more conservative),
        // so the assembler reserves more room — the budget never under-counts CJK.
        assert!(estimate_tokens("你好世界", 2.0) >= estimate_tokens("你好世界", 3.5));
    }

    #[test]
    fn test_f64_to_usize_saturating_handles_extreme_inputs() {
        // NaN → 0 (not a panic, not usize::MAX).
        assert_eq!(f64_to_usize_saturating(f64::NAN), 0, "NaN must map to 0");
        // +Infinity → usize::MAX (saturates).
        assert_eq!(
            f64_to_usize_saturating(f64::INFINITY),
            usize::MAX,
            "+Inf must saturate to usize::MAX"
        );
        // Huge finite (1e300 >> usize::MAX) → usize::MAX.
        assert_eq!(
            f64_to_usize_saturating(1e300),
            usize::MAX,
            "huge finite must saturate to usize::MAX"
        );
        // Negative → 0.
        assert_eq!(f64_to_usize_saturating(-42.0), 0, "negative must map to 0");
        // Normal value passes through unchanged.
        assert_eq!(
            f64_to_usize_saturating(7.0),
            7,
            "normal value must round-trip"
        );
    }

    #[test]
    fn test_budget_after_margin_does_not_panic_on_max_budget() {
        // J2: extreme budget (usize::MAX, headroom=0, ratio=0.5) must not panic
        // and must return a sane value strictly below usize::MAX.
        let result = budget_after_margin(usize::MAX, 0, 0.5);
        assert!(
            result < usize::MAX,
            "J2: result must be less than usize::MAX (margin was reserved)"
        );
        assert!(result > 0, "J2: result must be positive with a 50% margin");
    }
}
