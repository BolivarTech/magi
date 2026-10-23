// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The context window a review batch is sized against, and the token estimate it uses.

/// Same heuristic as magi-core's `measure_input` (bytes / 4), which is `pub(crate)` there (E17).
pub const BYTES_PER_TOKEN: usize = 4;

/// The window assumed for a model whose window was not measured.
pub const DEFAULT_UNMEASURED_WINDOW_TOKENS: usize = 32_768;

/// The share of the pool window one batch may take, in percent.
pub const BATCH_WINDOW_FRACTION_PERCENT: usize = 60;

/// Estimated tokens of `text`.
pub fn estimate_tokens(text: &str) -> usize {
    let _ = text;
    todo!()
}

/// The window of a model pool.
#[derive(Debug)]
pub struct PoolWindow {
    /// The binding window in tokens.
    pub tokens: usize,
    /// Models without a measured window.
    pub unmeasured: Vec<String>,
}

/// The binding window of a pool.
pub fn pool_window(models: &[(String, Option<usize>)]) -> PoolWindow {
    let _ = models;
    todo!()
}

/// The per-batch token limit.
pub fn batch_limit_tokens(window: &PoolWindow) -> usize {
    let _ = window;
    todo!()
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
