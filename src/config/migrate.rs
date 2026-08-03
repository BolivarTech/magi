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

/// Tests unitarios de detección y renderizado de migraciones.
#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A21: las DOS incompatibilidades se reportan juntas.
    #[test]
    fn a_v0_11_0_config_reports_both_incompatibilities_at_once() {
        let toml = include_str!("../../tests/fixtures/v0.11.0/default.toml");
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 2, "esperaba provider + [openai].base_url");
        let rendered = render_migration_error(&found);
        assert!(rendered.contains("provider"));
        assert!(rendered.contains("base_url"));
        assert!(
            rendered.contains("ollama") && rendered.contains("openai-compat"),
            "debe decir CÓMO elegir entre los dos"
        );
    }

    /// SC-A21h: un archivo a medio migrar recibe SOLO lo que le falta.
    #[test]
    fn a_partially_migrated_file_reports_only_what_is_missing() {
        let toml = "provider = \"openai\"\nbase_url = \"http://x/v1\"\n[openai]\nmodel = \"m\"\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].key, "provider");
    }

    /// SC-A21e: con credenciales embebidas, REDACTAR gana sobre listo-para-pegar.
    #[test]
    fn embedded_credentials_are_redacted_in_the_migration_message() {
        let toml = "[openai]\nbase_url = \"https://user:s3cr3t@host/v1\"\n";
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            !rendered.contains("s3cr3t"),
            "la credencial NO puede aparecer"
        );
        assert!(
            rendered.contains("host"),
            "el host sí, es lo que hace accionable el mensaje"
        );
        assert!(
            rendered.contains("redactad"),
            "y el mensaje debe decir que está redactado"
        );
    }

    /// SC-A21g: un TOML sintácticamente roto NO recibe consejo de migración.
    #[test]
    fn a_syntactically_broken_toml_gets_a_syntax_error_not_migration_advice() {
        let toml = "provider = \"sin cerrar\n[magi]\n";
        assert!(
            detect_migrations(toml).is_empty(),
            "sin estructura no hay dónde buscar patrones; la pasada no rescata por grep"
        );
    }

    /// SC-A21f: un TOML vacío parsea y no dispara nada.
    #[test]
    fn an_empty_toml_is_valid_and_triggers_no_migration() {
        assert!(detect_migrations("").is_empty());
        assert!(detect_migrations("   \n\n  ").is_empty());
    }

    /// SC-A21i: el salto desde v0.10.x se declara no soportado, en TODO error de config.
    #[test]
    fn every_config_error_mentions_the_v0_10_x_path() {
        let rendered = render_migration_error(&detect_migrations(include_str!(
            "../../tests/fixtures/v0.11.0/default.toml"
        )));
        assert!(
            rendered.contains("v0.11.0"),
            "no hay detección de v0.10.x: hay una nota incondicional"
        );
    }

    /// `line_of` devuelve 0 cuando el patrón no aparece en el texto.
    #[test]
    fn line_of_returns_zero_when_needle_is_absent() {
        assert_eq!(line_of("foo\nbar\n", "baz"), 0);
    }

    /// `line_of` devuelve el número de línea en base 1.
    #[test]
    fn line_of_returns_one_indexed_line_number() {
        assert_eq!(line_of("foo\nbar\nbaz", "bar"), 2);
    }

    /// Detecta el tercer patrón: `[headless].tool_result_cap_bytes`.
    #[test]
    fn headless_tool_result_cap_bytes_is_detected() {
        let toml = "[headless]\ntool_result_cap_bytes = 4096\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].key, "[headless].tool_result_cap_bytes");
        assert!(found[0].correction.contains("tool_result_cap_bytes"));
    }

    /// Los tres patrones viejos en un mismo archivo se reportan todos juntos.
    #[test]
    fn all_three_patterns_together_are_reported() {
        let toml = "provider = \"openai\"\n[openai]\nbase_url = \"http://x/v1\"\n\
                    [headless]\ntool_result_cap_bytes = 2048\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 3);
    }
}
