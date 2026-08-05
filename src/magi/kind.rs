// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Vocabulario de providers: los tres valores que nombran un backend concreto (REQ-A01b).
//!
//! # Por qué vive en el LIB y no en `config.rs`
//!
//! `probe_models` y `ProbeFactory` lo toman por parámetro y viven en el lib, así que dejarlo
//! en el bin los volvería incompilables. Y no es una concesión de empaquetado: es un enum
//! cerrado de tres variantes con su parser, o sea vocabulario **del dominio**, no de la forma
//! del TOML.

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

use std::fmt;

/// Los tres valores aceptados, en el texto que se muestra en un error.
///
/// Una `const` y no un literal repetido (B4): el mensaje de error y la documentación tienen
/// que nombrar el mismo conjunto.
pub const VALID_PROVIDER_KINDS: &str = "ollama, openai-compat, anthropic";

/// Provider concreto de magi-core (REQ-A01b).
///
/// **Vocabulario ÚNICO**: la clave `provider` de raíz y `[magi].kind` toman los mismos tres
/// valores, y el segundo **hereda** del primero cuando no se declara.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// Ollama: keyless, y el ÚNICO medible (`/api/show` + `/api/tags`).
    ///
    /// **No usa el tipo `OllamaProvider` de magi-core para las completions** (D-A07): su único
    /// constructor fija 300 s de timeout de cliente sin override, lo que hace imposible
    /// cumplir la escala de REQ-A04. Las completions van por el transporte OpenAI-compat
    /// keyless contra `…/v1`; `OllamaProvider` queda solo como sonda.
    Ollama,
    /// OpenAI, Groq, OpenRouter — cualquier Chat Completions. Con token, sin probe.
    OpenAiCompat,
    /// Anthropic Messages. Con token, sin probe.
    Anthropic,
}

/// Un valor de `provider` o `kind` presente que no nombra ningún backend.
///
/// **`ProviderKindParseError` y NO `ConfigError`**: este enum vive en el lib y `ConfigError` en
/// `config.rs`, que es del bin. Devolver el error del bin desde el lib invierte la dirección de
/// la dependencia y no compila; `config.rs` lo absorbe con un `From`.
#[derive(Debug, thiserror::Error)]
#[error("unknown provider: {got:?} (valid: {valid})")]
pub struct ProviderKindParseError {
    /// Lo que trajo el archivo.
    pub got: String,
    /// Los tres aceptados, para que el error sea accionable sin abrir la doc.
    pub valid: &'static str,
}

impl ProviderKind {
    /// Parsea un valor de configuración.
    ///
    /// # Errors
    ///
    /// [`ProviderKindParseError`] si el valor está **presente y no se reconoce**.
    ///
    /// Un valor **vacío o en blanco** devuelve `Ok(None)` — se trata como **ausente** (REQ-A12),
    /// porque una variable exportada y sin llenar en un script de CI es indistinguible de no
    /// haberla definido, y romper el arranque por eso castiga un accidente cotidiano. Un valor
    /// presente y no reconocido sí es un error: el usuario quiso decir algo y lo dijo mal.
    ///
    /// Recorta espacios **ASCII**, igual que `ModeExt::parse_config_value`, para que las dos
    /// claves de vocabulario del archivo se lean con la misma regla.
    pub fn parse(raw: &str) -> Result<Option<Self>, ProviderKindParseError> {
        match raw.trim_matches(|c: char| c.is_ascii_whitespace()) {
            "" => Ok(None),
            "ollama" => Ok(Some(Self::Ollama)),
            "openai-compat" => Ok(Some(Self::OpenAiCompat)),
            "anthropic" => Ok(Some(Self::Anthropic)),
            other => Err(ProviderKindParseError {
                got: other.to_string(),
                valid: VALID_PROVIDER_KINDS,
            }),
        }
    }

    /// Si este provider expone introspección de modelos (REQ-A24).
    ///
    /// Es una diferencia de **capacidad**, no de vocabulario: `ollama` y `openai-compat`
    /// comparten el protocolo de completions y se distinguen solo en que uno es medible.
    #[must_use]
    pub const fn is_probeable(self) -> bool {
        matches!(self, Self::Ollama)
    }
}

impl fmt::Display for ProviderKind {
    /// El inverso exacto de [`Self::parse`]: `ProviderKind::parse(&k.to_string()) ==
    /// Ok(Some(k))` para cualquier `k`. Task 4.1 lo necesita para renderizar de vuelta al
    /// vocabulario declarado (p. ej. al construir el `provider` por defecto que alimenta la
    /// resolución headless `env > TOML > default`), sin repetir los tres literales de
    /// `parse` en un segundo lugar (B3).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ollama => "ollama",
            Self::OpenAiCompat => "openai-compat",
            Self::Anthropic => "anthropic",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-A01b: los tres valores del vocabulario, y nada más.
    #[test]
    fn the_three_vocabulary_values_are_accepted_and_the_rest_are_not() {
        assert_eq!(
            ProviderKind::parse("ollama").unwrap(),
            Some(ProviderKind::Ollama)
        );
        assert_eq!(
            ProviderKind::parse("openai-compat").unwrap(),
            Some(ProviderKind::OpenAiCompat)
        );
        assert_eq!(
            ProviderKind::parse("anthropic").unwrap(),
            Some(ProviderKind::Anthropic)
        );
        assert!(ProviderKind::parse("banana").is_err());
    }

    /// El valor de v0.11.0 ya no es válido: es la mitad de la ruptura de REQ-A21.
    ///
    /// `"openai"` era ambiguo —podía ser Ollama o un endpoint autenticado— y esa ambigüedad es
    /// justo lo que el vocabulario nuevo parte. **No se auto-migra**: elegir por el usuario
    /// sería adivinar exactamente lo que D-A01 prohíbe.
    #[test]
    fn the_old_openai_value_is_rejected_rather_than_guessed() {
        let err = ProviderKind::parse("openai").unwrap_err();
        assert!(err.to_string().contains("openai"), "nombra lo recibido");
        assert!(
            err.to_string().contains("openai-compat"),
            "y los válidos, para que el arreglo sea obvio"
        );
    }

    /// SC-A12g / REQ-A12: vacío o en blanco es AUSENTE, nunca inválido.
    #[test]
    fn a_blank_value_is_absent_rather_than_invalid() {
        assert_eq!(ProviderKind::parse("").unwrap(), None);
        assert_eq!(ProviderKind::parse("   ").unwrap(), None);
        assert_eq!(ProviderKind::parse("\t\n").unwrap(), None);
        // Y el valor rodeado de espacios sigue siendo válido.
        assert_eq!(
            ProviderKind::parse("  ollama  ").unwrap(),
            Some(ProviderKind::Ollama)
        );
    }

    /// REQ-A24: solo Ollama es medible, y eso es capacidad, no vocabulario.
    #[test]
    fn only_ollama_exposes_model_introspection() {
        assert!(ProviderKind::Ollama.is_probeable());
        assert!(!ProviderKind::OpenAiCompat.is_probeable());
        assert!(!ProviderKind::Anthropic.is_probeable());
    }

    /// `Display` es el inverso exacto de `parse` para los tres valores — Task 4.1 depende
    /// de este roundtrip para renderizar el vocabulario de vuelta a texto.
    #[test]
    fn display_round_trips_through_parse_for_the_three_values() {
        for kind in [
            ProviderKind::Ollama,
            ProviderKind::OpenAiCompat,
            ProviderKind::Anthropic,
        ] {
            assert_eq!(ProviderKind::parse(&kind.to_string()).unwrap(), Some(kind));
        }
    }
}
