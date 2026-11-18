// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-18

//! The context window a review batch is sized against, and the token estimate it uses.

/// The divisor used to convert a percentage into a fraction.
const PERCENT: usize = 100;

/// Token estimator: the same heuristic as magi-core's `measure_input` (bytes / 4), which is
/// `pub(crate)` in magi-core and cannot be called from this crate, so it is duplicated here on
/// purpose; its error on real code is what the batch fraction leaves room for.
pub const BYTES_PER_TOKEN: usize = 4;

/// The window assumed for a model whose window was not measured; a declared starting point, not a
/// measurement.
///
/// REQ-RV04 names `src/defaults.rs` as its home, but that module belongs to the binary and this
/// one to the library, which cannot reach it; this is the only definition.
pub const DEFAULT_UNMEASURED_WINDOW_TOKENS: usize = 32_768;

/// The share of the pool window one batch (batch text plus retrieved context) may take, in
/// percent. The rest holds each reviewer's system prompt, the mode prompt and the reply.
///
/// The target value is 60, used only once the token estimator's error has been measured against
/// real backends: r = the largest ratio of a backend's prompt token count to `estimate_tokens`
/// on plain source text, P = the largest fixed cost of the mode prompt in tokens, R = the
/// largest observed reply in tokens, W = the pool window; 60 holds only if
/// `0.60 × W × max(r − 1, 0) + P + R ≤ 0.40 × W`.
///
/// That measurement has not been taken (no backend was available for it), so the value is the
/// provisional 50 until it is: 50 is the boundary for an estimator that underestimates about 2x
/// (a little less once P + R > 0), at the cost of roughly 20 % more batches than 60.
///
/// R is observed, not a bound: a reply longer than the measured one may not fit.
///
/// When the measurement is taken, it is recorded as a plain line comment just above this
/// constant whose text is `STEP-4B MEASURED: r=…, P=…, R=…, W=…, <date>`. A test reads this
/// source file and accepts 60 only with that record present.
pub const BATCH_WINDOW_FRACTION_PERCENT: usize = 50;

/// Estimate the tokens a plain text string occupies under the bytes-per-token heuristic.
///
/// # Arguments
///
/// * `text` - The source text to measure.
///
/// # Returns
///
/// The token count, rounded up.
///
/// # Examples
///
/// ```
/// use magi_rs::review::window::estimate_tokens;
///
/// assert_eq!(estimate_tokens(""), 0);
/// assert_eq!(estimate_tokens("ññ"), 1);
/// ```
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(BYTES_PER_TOKEN)
}

/// The binding context window for a pool of models and the list of models whose window was not
/// measured.
#[derive(Debug)]
pub struct PoolWindow {
    /// The binding window in tokens.
    pub tokens: usize,
    /// Names of the models whose window was not measured, in input order (the review report
    /// declares them).
    pub unmeasured: Vec<String>,
}

/// Compute the binding context window across a pool of models.
///
/// Every model with a `None` window contributes [`DEFAULT_UNMEASURED_WINDOW_TOKENS`]; the result
/// is the minimum over the whole pool, including those defaults. An empty pool yields
/// [`DEFAULT_UNMEASURED_WINDOW_TOKENS`] with no unmeasured names.
///
/// Complexity: O(m) in the number of models, one pass.
///
/// # Arguments
///
/// * `models` - A slice of `(model name, measured window in tokens or None)`.
///
/// # Returns
///
/// A [`PoolWindow`] containing the binding token limit and the unmeasured model names in input
/// order.
///
/// # Examples
///
/// ```
/// use magi_rs::review::window::pool_window;
///
/// let w = pool_window(&[
///     ("large".into(), Some(200_000)),
///     ("small".into(), Some(64_000)),
/// ]);
/// assert_eq!(w.tokens, 64_000);
/// assert!(w.unmeasured.is_empty());
/// ```
pub fn pool_window(models: &[(String, Option<usize>)]) -> PoolWindow {
    let mut tokens: Option<usize> = None;
    let mut unmeasured = Vec::new();
    for (name, window) in models {
        let candidate = window.unwrap_or_else(|| {
            unmeasured.push(name.clone());
            DEFAULT_UNMEASURED_WINDOW_TOKENS
        });
        tokens = Some(tokens.map_or(candidate, |t| t.min(candidate)));
    }
    PoolWindow {
        tokens: tokens.unwrap_or(DEFAULT_UNMEASURED_WINDOW_TOKENS),
        unmeasured,
    }
}

/// Compute the token budget a single batch may consume from a pool window.
///
/// The budget is the pool window multiplied by [`BATCH_WINDOW_FRACTION_PERCENT`] and divided by
/// 100 (multiplying first, saturating, so neither precision nor a huge window is lost).
///
/// # Arguments
///
/// * `window` - The pool window to budget against.
///
/// # Returns
///
/// The batch token limit.
///
/// # Examples
///
/// ```
/// use magi_rs::review::window::{batch_limit_tokens, pool_window};
///
/// let w = pool_window(&[("model".into(), Some(32_768))]);
/// assert_eq!(batch_limit_tokens(&w), 16_384);
/// ```
pub fn batch_limit_tokens(window: &PoolWindow) -> usize {
    window.tokens.saturating_mul(BATCH_WINDOW_FRACTION_PERCENT) / PERCENT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_model_counts_with_the_default_and_the_minimum_covers_the_whole_pool() {
        let w = pool_window(&[
            ("a".into(), Some(200_000)),
            ("b".into(), None),
            ("c".into(), Some(64_000)),
        ]);
        assert_eq!(w.tokens, 32_768);
        assert_eq!(w.unmeasured, vec!["b".to_string()]);
        // [CP2 MS2 S1b loop 13] 60 measured, 50 provisional (Step 4b); the literal for each, not the formula.
        assert_eq!(
            batch_limit_tokens(&w),
            if BATCH_WINDOW_FRACTION_PERCENT == 60 {
                19_660
            } else {
                16_384
            }
        );
    }

    #[test]
    fn the_batch_fraction_is_the_measured_sixty_or_the_provisional_fifty() {
        // [CP2 MS2 S1b loop 13; user, 2026-10-19] the only two values Step 4b may leave.
        // [CP2 MS2 S1b2 loop 16] and 60 only with the measurement recorded next to the constant: an unmeasured 60 is red.
        let src = include_str!("window.rs").replace('\r', "");
        let measured = src.contains(concat!("// STEP-4B", " MEASURED: "));
        assert_eq!(
            BATCH_WINDOW_FRACTION_PERCENT,
            if measured { 60 } else { 50 },
            "Step 4b record present: {measured}"
        );
    }

    #[test]
    fn with_every_model_measured_the_smallest_window_wins() {
        let w = pool_window(&[("a".into(), Some(200_000)), ("c".into(), Some(64_000))]);
        assert_eq!((w.tokens, w.unmeasured.len()), (64_000, 0));
        let empty = pool_window(&[]);
        assert_eq!((empty.tokens, empty.unmeasured.len()), (32_768, 0));
    }

    #[test]
    fn tokens_are_bytes_over_four_rounded_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("ññ"), 1); // 4 bytes
    }
}
