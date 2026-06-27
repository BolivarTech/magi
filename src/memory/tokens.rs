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
pub fn estimate_tokens(text: &str, chars_per_token: f64) -> usize {
    let count = text.chars().count();
    if count == 0 {
        return 0;
    }
    let cpt = if chars_per_token > 0.0 && chars_per_token.is_finite() {
        chars_per_token
    } else {
        1.0
    };
    (count as f64 / cpt).ceil() as usize
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
        (budget as f64 * margin_ratio).ceil() as usize
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
}
