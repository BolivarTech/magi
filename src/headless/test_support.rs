// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Helpers de test **genéricos** del subsistema headless.
//!
//! Solo helpers sin dependencia de tipos del feature viven acá (MAGI CP2 run 2
//! Melchior — no todos los helpers en T0). Los helpers con semántica de una
//! tarea concreta (DB, formateo, logs) se declaran en la tarea que los da.
//! `with_var` es genérico de entorno ⇒ vive acá.
//!
//! **Aislamiento:** `cargo nextest` corre un proceso por test, por lo que la
//! mutación de variables de entorno de [`with_var`] queda aislada entre tests
//! (no hay carrera cross-test); aun así el helper **restaura** el valor previo.

use std::env;

/// Ejecuta `f` con la variable de entorno `key` fijada a `value` y restaura su
/// valor previo al terminar.
///
/// `value = Some(v)` fija `key=v`; `value = None` la **elimina** durante `f`. Al
/// salir, `key` recupera exactamente su estado anterior (presente-con-valor o
/// ausente).
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
