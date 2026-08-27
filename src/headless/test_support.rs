// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27
// headless subsystem.
//!
//! Only helpers with no dependency on feature types live here (MAGI CP2 run 2 Melchior — not
//! all helpers in T0). Helpers with semantics of a concrete task (DB, formatting, logs) are
//! declared in the task that provides them. `with_var` is environment-generic ⇒ lives here.
//!
//! **Isolation:** `cargo nextest` runs one process per test, so the
//! mutation of environment variables by [`with_var`] is isolated between tests (no cross-test
//! race); even so the helper **restores** the previous value.

use std::env;

/// Runs `f` with the environment variable `key` set to `value` and restores its previous value
/// on completion.
///
/// `value = Some(v)` sets `key=v`; `value = None` **removes** it during `f`. On exit, `key`
/// recovers exactly its previous state (present-with-value or absent).
///
/// # Examples
/// ```ignore
/// let seen = with_var("MAGI_X", Some("1"), || std::env::var("MAGI_X").ok());
/// assert_eq!(seen.as_deref(), Some("1"));
/// ```
pub(crate) fn with_var<F, R>(key: &str, value: Option<&str>, f: F) -> R
where
    F: FnOnce() -> R,
{
    let previous = env::var(key).ok();
    match value {
        Some(v) => env::set_var(key, v),
        None => env::remove_var(key),
    }
    let result = f();
    match previous {
        Some(prev) => env::set_var(key, prev),
        None => env::remove_var(key),
    }
    result
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::with_var;

    #[test]
    fn test_with_var_sets_value_inside_and_restores_absent_after() {
        let key = "MAGI_TEST_WITH_VAR_ABSENT";
        env::remove_var(key);
        let inside = with_var(key, Some("hello"), || env::var(key).ok());
        assert_eq!(inside.as_deref(), Some("hello"));
        // Was absent before ⇒ absent again after.
        assert!(env::var(key).is_err());
    }

    #[test]
    fn test_with_var_none_removes_var_and_restores_prior_value() {
        let key = "MAGI_TEST_WITH_VAR_PRESENT";
        env::set_var(key, "original");
        let inside = with_var(key, None, || env::var(key).ok());
        assert_eq!(inside, None); // removed inside the closure
        assert_eq!(env::var(key).as_deref(), Ok("original")); // restored after
        env::remove_var(key);
    }
}
