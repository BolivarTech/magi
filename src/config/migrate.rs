// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Pasada de validación PREVIA al parseo, para reportar todas las incompatibilidades de
//! migración de un `magi.toml` de v0.11.0 juntas (REQ-A21b).
//!
//! `deny_unknown_fields` **aborta en la PRIMERA clave desconocida**, así que "todas juntas"
//! es imposible de lograr desde el error de serde. Esta pasada lee el TOML como documento
//! genérico **antes** de deserializar, junta los patrones conocidos y emite un solo mensaje.
//! Sin ella el usuario paga dos ciclos de editar-arrancar-fallar.
//!
//! # Deuda técnica con fecha
//!
//! **Se retira en v0.13.0 (MS3)**, cuando la migración deje de ser el caso común. Duplica un
//! poco de conocimiento del schema —los patrones son sobre la forma **vieja**, que ya no está
//! en el código— y esa duplicación se acepta a conciencia: es acotada a tres patrones y está
//! cubierta por tests contra archivos reales de v0.11.0.

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

use magi_rs::redact::redact_url;
use toml::Value;

/// Clave raíz del proveedor en `magi.toml`.
const PROVIDER_KEY: &str = "provider";

/// Valor de `provider` en v0.11.0 que dejó de existir en v0.12.0.
///
/// **No se auto-migra, y es deliberado.** `"openai"` era ambiguo —podía significar un Ollama
/// local sin credencial o un endpoint autenticado— y partir esa ambigüedad es la mitad del
/// punto del cambio. Elegir por el usuario sería adivinar exactamente lo que D-A01 prohíbe.
const PROVIDER_V0_11_0: &str = "openai";

/// Sección `[openai]` de v0.11.0; desde v0.12.0 `base_url` no vive ahí.
const OPENAI_SECTION: &str = "openai";

/// Sección `[headless]` de v0.11.0; desde v0.12.0 `tool_result_cap_bytes` sube a raíz.
const HEADLESS_SECTION: &str = "headless";

/// Clave `base_url`, que en v0.11.0 vivía dentro de `[openai]`.
const BASE_URL_KEY: &str = "base_url";

/// Clave `tool_result_cap_bytes`, que en v0.11.0 vivía dentro de `[headless]`.
const TOOL_RESULT_CAP_BYTES_KEY: &str = "tool_result_cap_bytes";

/// Etiqueta mostrada para `[openai].base_url`.
const OPENAI_BASE_URL_LABEL: &str = "[openai].base_url";

/// Etiqueta mostrada para `[headless].tool_result_cap_bytes`.
const HEADLESS_CAP_LABEL: &str = "[headless].tool_result_cap_bytes";

/// Versión de origen de la migración.
const VERSION_FROM: &str = "v0.11.0";

/// Versión destino de la migración.
const VERSION_TO: &str = "v0.12.0";

/// Corrección para `provider = "openai"`: nombra las dos opciones y el criterio para elegir.
const PROVIDER_CORRECTION: &str = "provider = \"ollama\"        # if it points to a local Ollama daemon, no credential\n           provider = \"openai-compat\" # for OpenAI, Groq, OpenRouter and other authenticated endpoints";

/// Prefijo de la corrección de `[openai].base_url`, antes del valor redactado.
const BASE_URL_CORRECTION_PREFIX: &str = "base_url = \"";

/// Sufijo de la corrección de `[openai].base_url`, tras el valor redactado.
///
/// Dice que el valor está **redactado**: con credenciales embebidas, redactar gana sobre
/// listo-para-pegar (SC-A21e). Un mensaje de migración que filtra una credencial al terminal,
/// al scrollback y a los logs de CI es peor problema que una línea que hay que completar.
const BASE_URL_CORRECTION_SUFFIX: &str =
    "\"   # at the root level, above every section. Value redacted: copy the real one from the old file.";

/// Prefijo de la corrección de `[headless].tool_result_cap_bytes`.
const CAP_CORRECTION_PREFIX: &str = "tool_result_cap_bytes = ";

/// Sufijo de la corrección de `[headless].tool_result_cap_bytes`.
const CAP_CORRECTION_SUFFIX: &str =
    "   # at the root level: now governs all THREE routes (TUI, magi query and headless consult).";

/// Nota incondicional sobre el salto desde v0.10.x.
///
/// **No hay detección de v0.10.x**: la pasada solo conoce los patrones de v0.11.0. Un archivo
/// de v0.10.x puede traer además incompatibilidades anteriores que nadie auditó, y recibiría
/// el error genérico justo cuando más ayuda necesita. Sostener dos generaciones duplicaría la
/// deuda por un salto que el usuario hace en dos pasos.
const V0_10_X_NOTE: &str =
    "If you're coming from v0.10.x, migrate to v0.11.0 first and then to v0.12.0: this pass only knows\nthe v0.11.0 patterns.";

/// Consejo de backup, en el cuerpo del error y no solo en el CHANGELOG.
///
/// Quien tropieza con este error llegó **arrancando el binario**, no leyendo notas de release.
/// Es el único momento en que todavía puede hacer la copia — o sea, antes de editar.
const BACKUP_ADVISORY: &str =
    "Save a copy of your magi.toml BEFORE editing it: this migration is one-way.";

/// Un `magi.toml` mínimo y válido de v0.12.0, listo para pegar.
///
/// **Va en el cuerpo del error y no en `docs/magi.toml.example`**: quien instaló con
/// `cargo install` o bajó un binario del release NO tiene el archivo de ejemplo, y sin
/// bandera de escape (REQ-A23) este mensaje es la única defensa. Son seis líneas.
const MINIMAL_VALID_CONFIG: &str = "provider = \"ollama\"\nbase_url = \"http://localhost:11434/v1\"\n\n[openai]\nmodel = \"kimi-k2.6:cloud\"\n";

/// Una incompatibilidad de migración detectada, con su corrección.
///
/// El shape es el contrato que comparten `from_toml_str` y [`render_migration_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Clave afectada, tal como aparece en el archivo (p. ej. `"[openai].base_url"`).
    pub key: &'static str,
    /// Línea 1-indexada donde se encontró, para el mensaje. `0` si no se pudo ubicar.
    pub line: usize,
    /// Texto de la corrección, ya redactado si el valor original traía credenciales.
    pub correction: String,
}

/// Detecta los tres patrones de migración de v0.11.0 en un `magi.toml` crudo.
///
/// Los patrones son `provider = "openai"` en la raíz, `[openai].base_url` y
/// `[headless].tool_result_cap_bytes`.
///
/// Devuelve vacío si el documento **no parsea como TOML**: sin estructura no hay dónde
/// buscar, y rescatarlo por búsqueda textual daría consejos sobre una forma que nadie sabe
/// cuál es (SC-A21g). Un archivo sintácticamente roto recibe su error de sintaxis, con línea
/// y columna, no consejo de migración.
///
/// Reporta **solo lo que ese archivo tiene mal**: un archivo a medio migrar recibe únicamente
/// la corrección que le falta (SC-A21h). Repetir una que el usuario ya aplicó lo haría dudar
/// de si la aplicó bien, que es el estado mental opuesto al que este mensaje busca.
#[must_use]
pub fn detect_migrations(raw: &str) -> Vec<Migration> {
    let Ok(doc) = raw.parse::<Value>() else {
        return Vec::new();
    };

    let mut found = Vec::new();

    if let Some(provider) = doc.get(PROVIDER_KEY).and_then(Value::as_str) {
        if provider.trim() == PROVIDER_V0_11_0 {
            found.push(Migration {
                key: PROVIDER_KEY,
                line: line_of(raw, PROVIDER_KEY),
                correction: PROVIDER_CORRECTION.to_owned(),
            });
        }
    }

    if let Some(url) = doc
        .get(OPENAI_SECTION)
        .and_then(Value::as_table)
        .and_then(|t| t.get(BASE_URL_KEY))
        .and_then(Value::as_str)
    {
        found.push(Migration {
            key: OPENAI_BASE_URL_LABEL,
            line: line_of(raw, BASE_URL_KEY),
            correction: format!(
                "{BASE_URL_CORRECTION_PREFIX}{}{BASE_URL_CORRECTION_SUFFIX}",
                redact_url(url)
            ),
        });
    }

    if let Some(cap) = doc
        .get(HEADLESS_SECTION)
        .and_then(Value::as_table)
        .and_then(|t| t.get(TOOL_RESULT_CAP_BYTES_KEY))
    {
        found.push(Migration {
            key: HEADLESS_CAP_LABEL,
            line: line_of(raw, TOOL_RESULT_CAP_BYTES_KEY),
            correction: format!("{CAP_CORRECTION_PREFIX}{cap}{CAP_CORRECTION_SUFFIX}"),
        });
    }

    found
}

/// Número de línea 1-indexado donde **empieza** una clave `needle`, o 0 si no aparece.
///
/// Compara contra el inicio de la línea ya recortada, no con `contains`: un `contains` haría
/// match con la clave mencionada dentro de un comentario, y el archivo por defecto de v0.11.0
/// está lleno de comentarios que nombran sus propias claves. Es solo para el mensaje — una
/// línea equivocada confunde, pero no cambia qué se detectó.
fn line_of(raw: &str, needle: &str) -> usize {
    raw.lines()
        .position(|line| line.trim_start().starts_with(needle))
        .map_or(0, |idx| idx + 1)
}

/// Renderiza el error de migración completo a partir de las incompatibilidades halladas.
///
/// El mensaje es **autocontenido** y no manda al usuario a ningún archivo del repo: incluye
/// cada corrección, el consejo de backup, un `magi.toml` mínimo válido para pegar, y la nota
/// incondicional sobre v0.10.x. Quien instaló por `cargo install` o bajó un binario **no
/// tiene** el árbol de fuentes, y mandarlo ahí lo deja igual de trabado que sin mensaje.
#[must_use]
pub fn render_migration_error(found: &[Migration]) -> String {
    let mut out = format!(
        "error: .magi/magi.toml is not compatible with magi-rs {VERSION_TO} (coming from {VERSION_FROM})\n\n"
    );

    for m in found {
        if m.line == 0 {
            out.push_str(&format!("  {}\n           {}\n\n", m.key, m.correction));
        } else {
            out.push_str(&format!(
                "  line {}  {}\n           {}\n\n",
                m.line, m.key, m.correction
            ));
        }
    }

    out.push_str(BACKUP_ADVISORY);
    out.push_str("\n\nA minimal, valid v0.12.0 magi.toml:\n\n");
    out.push_str(MINIMAL_VALID_CONFIG);
    out.push('\n');
    out.push_str(V0_10_X_NOTE);
    out.push('\n');
    out
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
            rendered.contains("redacted"),
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

    /// SC-A21d: el mensaje se valida contra los CUATRO archivos reales de v0.11.0.
    ///
    /// Un fixture escrito a mano prueba que el mensaje **se emite**; solo uno real prueba que
    /// **alcanza**. Los cuatro fueron generados o derivados del binario publicado de v0.11.0
    /// y verificados contra él — ver `tests/fixtures/v0.11.0/README.md`.
    ///
    /// Los cuatro comparten las mismas dos incompatibilidades porque las tres variantes se
    /// derivan del `default.toml` canónico sin agregar ni quitar claves migrables.
    #[test]
    fn every_real_v0_11_0_fixture_reports_its_own_incompatibilities() {
        for (name, toml) in [
            (
                "default",
                include_str!("../../tests/fixtures/v0.11.0/default.toml"),
            ),
            (
                "with-models",
                include_str!("../../tests/fixtures/v0.11.0/with-models.toml"),
            ),
            (
                "full",
                include_str!("../../tests/fixtures/v0.11.0/full.toml"),
            ),
            (
                "with-credentials",
                include_str!("../../tests/fixtures/v0.11.0/with-credentials.toml"),
            ),
        ] {
            let found = detect_migrations(toml);
            assert_eq!(
                found.len(),
                2,
                "{name}: esperaba provider + [openai].base_url"
            );
            let rendered = render_migration_error(&found);
            assert!(rendered.contains(PROVIDER_KEY), "{name}: nombra provider");
            assert!(rendered.contains(BASE_URL_KEY), "{name}: nombra base_url");
        }
    }

    /// SC-A21e sobre el archivo REAL: la credencial del fixture nunca llega al mensaje.
    ///
    /// El test de arriba usa un TOML inline; éste usa el archivo commiteado, que es el que
    /// un usuario tendría. Son distintos a propósito: el inline fija la regla, éste fija que
    /// la regla sobrevive al archivo de verdad.
    #[test]
    fn the_real_credentials_fixture_never_leaks_its_secret() {
        let toml = include_str!("../../tests/fixtures/v0.11.0/with-credentials.toml");
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            !rendered.contains("s3cr3t"),
            "la credencial del fixture no puede aparecer en el mensaje"
        );
        assert!(
            rendered.contains("host"),
            "el host sí, para que sea accionable"
        );
    }

    /// SC-A21d, segunda mitad: **lo que el mensaje propone parsea sin error en v0.12.0**.
    ///
    /// Es la parte que hace útil al mensaje y la que más fácil se pudre: el TOML mínimo es un
    /// literal, así que nada lo ata al schema salvo este test. Si una tarea posterior cambia
    /// una clave, el consejo que le damos al usuario trabado deja de funcionar — y sin esto,
    /// en silencio.
    #[test]
    fn the_minimal_config_the_error_hands_out_actually_parses_today() {
        super::super::MagiConfig::from_toml_str(MINIMAL_VALID_CONFIG)
            .expect("el magi.toml mínimo que el error propone debe parsear en v0.12.0");
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
