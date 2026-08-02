// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Pasada de validación PREVIA al parseo, para reportar todas las incompatibilidades de
//! migración de un `magi.toml` de v0.11.0 juntas (REQ-A21b).
//!
//! # Stub (Task 1.1)
//!
//! Esta tarea nace el módulo y las FIRMAS que `MagiConfig::from_toml_str` consume —
//! `detect_migrations` y `render_migration_error` — porque esa función las llama en su
//! cuerpo y una fase no puede consumir un símbolo que otra posterior produce. La
//! DETECCIÓN real de los tres patrones de v0.11.0 (`provider = "openai"`,
//! `[openai].base_url`, `[headless].tool_result_cap_bytes`) es trabajo de Task 1.3.
//!
//! Hasta entonces, [`detect_migrations`] siempre devuelve una colección vacía —
//! comportamiento equivalente a "sin migraciones pendientes" — así que la rama de
//! migración de `from_toml_str` nunca dispara todavía, y [`render_migration_error`] es
//! inalcanzable en la práctica (se define solo porque el cuerpo de `from_toml_str` la
//! nombra).

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

/// Una incompatibilidad de migración detectada, con su corrección.
///
/// El shape es el contrato que comparten `from_toml_str` y `render_migration_error`;
/// Task 1.3 le da cuerpo real a quien lo produce ([`detect_migrations`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Clave afectada, tal como aparece en el archivo (p. ej. `"[openai].base_url"`).
    pub key: &'static str,
    /// Línea 1-indexada donde se encontró, para el mensaje. `0` si no se pudo ubicar.
    pub line: usize,
    /// Texto de la corrección, ya redactado si el valor original traía credenciales.
    pub correction: String,
}

/// Detecta los patrones de migración conocidos de v0.11.0 en un `magi.toml` crudo.
///
/// **Stub (Task 1.1): siempre devuelve vacío**, para que `from_toml_str` compile y se
/// comporte como "sin migraciones pendientes" mientras Task 1.3 no aterriza la detección
/// real de los tres patrones (`provider = "openai"`, `[openai].base_url`,
/// `[headless].tool_result_cap_bytes`).
#[must_use]
pub fn detect_migrations(_raw: &str) -> Vec<Migration> {
    Vec::new()
}

/// Renderiza el error de migración completo a partir de las incompatibilidades halladas.
///
/// **Stub (Task 1.1): inalcanzable en la práctica.** [`detect_migrations`] siempre
/// devuelve vacío en esta tarea, así que `from_toml_str` nunca entra a la rama que
/// llamaría a esta función. Se define ahora porque esa rama ya nombra la función en su
/// cuerpo; Task 1.3 implementa el mensaje real (con el consejo de backup, el TOML mínimo
/// válido para pegar, y la nota incondicional sobre el salto desde v0.10.x).
#[must_use]
pub fn render_migration_error(found: &[Migration]) -> String {
    format!(
        "magi.toml necesita migración ({} patrón(es) detectado(s))",
        found.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// El stub nunca reporta nada, para ningún input: `from_toml_str` no debe tomar la
    /// rama de migración todavía. Task 1.3 reemplaza esta cobertura por la real, contra
    /// fixtures de v0.11.0.
    #[test]
    fn the_stub_reports_no_migrations_for_any_input() {
        assert!(detect_migrations("").is_empty());
        assert!(
            detect_migrations("provider = \"openai\"\n[openai]\nbase_url = \"x\"\n").is_empty()
        );
    }

    /// El mensaje de fallback sigue siendo un texto no vacío incluso con la lista vacía,
    /// para que un llamador que lo invoque por error (no debería ocurrir todavía) no
    /// obtenga una cadena vacía silenciosa.
    #[test]
    fn the_stub_render_never_returns_an_empty_string() {
        assert!(!render_migration_error(&[]).is_empty());
    }
}
