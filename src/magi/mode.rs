// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Vocabulario de modos: de dónde salió el modo efectivo y cómo se lee de un texto.
//!
//! Acá vive **solo lo puro** —entra un `&str`, sale un `Mode`—. La resolución en cuatro
//! niveles, la guarda de `untrusted_content` y el trait del clasificador los agrega Task 2.1
//! a este mismo archivo. La partición es por **madurez de dependencia**, no por tema: este
//! bloque no depende de nada y la Fase 1 ya lo consume (`config.rs` valida `default_mode`),
//! así que nacer en la Fase 2 dejaría la Fase 1 sin compilar.

use magi_core::schema::Mode;

/// Las tres etiquetas válidas, en el texto que se le muestra al usuario en un error.
///
/// Una `const` y no un literal repetido (B4): el mensaje de [`ModeParseError::Unknown`] y la
/// documentación tienen que nombrar el mismo conjunto, y escribirlo dos veces es cómo se
/// desincronizan.
const VALID_MODE_LABELS: &str = "code-review, design, analysis";

/// De qué nivel salió el modo efectivo (REQ-A08).
///
/// `Configured` es su propia variante y no un `Explicit`: comparte con él la semántica
/// —alguien lo eligió, así que saltea la inferencia— pero **no** de dónde vino, y esa
/// diferencia es la que hace auditable un veredicto raro. Ante *"¿por qué corrió en este
/// modo?"*, `Explicit` manda a revisar el comando y `Configured` manda al `magi.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeSource {
    /// `--mode` en la invocación, o el campo del envelope. Lo declaró un HUMANO.
    Explicit,
    /// `[magi].default_mode`.
    Configured,
    /// Lo eligió el AGENTE por el `mode` del `input_schema`. Cero llamadas extra.
    ///
    /// **Variante propia, y ese es el punto.** Mientras la elección del agente y la
    /// clasificación del contenido compartieron la etiqueta `Inferred`, ninguna guarda podía
    /// distinguirlas: la de `untrusted_content` terminaba bloqueando las dos y con eso mataba
    /// SC-A07d, que es requerimiento duro. Separarlas es lo que permite bloquear el nivel 4
    /// sin tocar el 3. Y **no es `Explicit`**, así que no satisface una guarda que exige
    /// declaración humana — que es lo que cierra el bypass sin sacarle el campo al schema.
    ///
    /// Va DEBAJO de `Configured`: un `default_mode` declarado fija la lente y el agente no la
    /// cambia. Esa es la perilla del operador.
    AgentChosen,
    /// Salió de una llamada de CLASIFICACIÓN sobre el contenido. La que `untrusted_content`
    /// bloquea, porque es la superficie de ataque dedicada.
    Inferred,
    /// `Analysis`, porque no hubo ninguno de los anteriores.
    Default,
}

/// Un valor de configuración presente que no nombra ningún modo.
#[derive(Debug, thiserror::Error)]
pub enum ModeParseError {
    /// El valor tiene contenido y no es una de las tres etiquetas.
    #[error("modo desconocido: {got:?} (válidos: {valid})")]
    Unknown {
        /// Lo que trajo el archivo.
        got: String,
        /// Los tres aceptados, para que el error sea accionable sin abrir la doc.
        valid: &'static str,
    },
}

/// Recorta espacios en blanco **ASCII** de los extremos.
///
/// `trim_matches` con predicado ASCII y **no** `trim()`: este último recorta espacios Unicode
/// —NBSP, anchos variables— y la spec dice ASCII. Abrir la normalización a Unicode agranda la
/// superficie que un contenido hostil controla, que es justo lo que la normalización cerrada
/// existe para evitar.
fn trim_ascii(raw: &str) -> &str {
    raw.trim_matches(|c: char| c.is_ascii_whitespace())
}

/// Normaliza y valida la respuesta del clasificador (REQ-A07c).
///
/// **Cerrada, tres pasos, en este orden:** recortar espacios ASCII → minúsculas ASCII →
/// comparar **literal** contra las tres etiquetas. Nada más: ni quitar comillas, ni
/// desenvolver JSON, ni tomar la primera palabra, ni buscar una etiqueta dentro de una
/// oración.
///
/// **El equilibrio es intencional en las dos direcciones.** Sin normalización, un
/// `"code-review\n"` —que es lo que devuelve buena parte de los modelos— fallaría y la
/// inferencia sería inútil en la práctica. Con normalización abierta, un `"el modo apropiado
/// sería code-review"` pasaría, y ahí la inyección deja de estar contenida: bastaría con que
/// el modelo *mencione* una etiqueta en su prosa.
///
/// Es el mismo esquema que el **sentinel de veredicto de magi-core**: la salida ES la
/// respuesta, o es un fallo. Ese crate borró su parser de búsqueda en 3.0.0, y esa lección es
/// la que se aplica acá un nivel más arriba.
///
/// # Examples
///
/// ```
/// use magi_core::schema::Mode;
/// use magi_rs::magi::mode::normalize_label;
///
/// assert_eq!(normalize_label(" Code-Review\n"), Some(Mode::CodeReview));
/// assert_eq!(normalize_label("creo que code-review"), None);
/// ```
#[must_use]
pub fn normalize_label(raw: &str) -> Option<Mode> {
    match trim_ascii(raw).to_ascii_lowercase().as_str() {
        "code-review" => Some(Mode::CodeReview),
        "design" => Some(Mode::Design),
        "analysis" => Some(Mode::Analysis),
        _ => None,
    }
}

/// Extensión de `Mode` con el parseo de un valor de **configuración**.
///
/// **Es un trait de extensión, no un `impl Mode`, y no es una preferencia de estilo:** `Mode`
/// es un tipo de magi-core y Rust no admite métodos inherentes sobre un tipo foráneo.
/// Verificado contra magi-core 3.1.0: `Mode` expone `Display` y `Deserialize` (kebab-case) y
/// **nada más** — no hay `parse_config_value`, no hay `FromStr`. Con el trait en alcance la
/// sintaxis de llamada es la misma, así que los call sites no cambian.
///
/// Se distingue de [`normalize_label`] en el eje que importa: aquella es para texto **de un
/// modelo**, donde ausente e inválido son lo mismo (`None`) y decide el llamador; esto es para
/// texto **de un humano en un archivo**, donde un `banana` mal tipeado tiene que doler y un
/// valor vacío no.
pub trait ModeExt: Sized {
    /// `Ok(Some(m))` si el valor nombra un modo; `Ok(None)` si está **ausente o en blanco**;
    /// `Err` si tiene contenido y no lo nombra.
    ///
    /// # Errors
    ///
    /// [`ModeParseError::Unknown`] con el valor recibido y los tres válidos.
    ///
    /// **`ModeParseError` y NO `ConfigError`**: este trait vive en el lib y `ConfigError` en
    /// `config.rs`, que es del binario. Devolver el error del bin desde el lib invierte la
    /// dirección de la dependencia y no compila. `config.rs` lo absorbe con un
    /// `From<ModeParseError> for ConfigError`.
    fn parse_config_value(raw: &str) -> Result<Option<Self>, ModeParseError>;
}

impl ModeExt for Mode {
    fn parse_config_value(raw: &str) -> Result<Option<Self>, ModeParseError> {
        // Blanco = ausente: una variable exportada vacía en un script de CI es un accidente
        // cotidiano y no debe romper el arranque (REQ-A12).
        if trim_ascii(raw).is_empty() {
            return Ok(None);
        }
        normalize_label(raw)
            .map(Some)
            .ok_or_else(|| ModeParseError::Unknown {
                got: raw.to_string(),
                valid: VALID_MODE_LABELS,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A07l: el exact-match absorbe FORMATO, no CONTENIDO.
    #[test]
    fn label_normalization_absorbs_format_but_not_prose() {
        for raw in [
            "code-review",
            "code-review\n",
            " Code-Review ",
            "CODE-REVIEW",
        ] {
            assert_eq!(
                normalize_label(raw),
                Some(Mode::CodeReview),
                "formato: {raw:?}"
            );
        }
        assert_eq!(
            normalize_label("el modo apropiado seria code-review"),
            None,
            "prosa que CONTIENE una etiqueta no es la etiqueta: ahí la inyección deja de \
             estar contenida"
        );
        assert_eq!(
            normalize_label("design analysis"),
            None,
            "dos etiquetas tampoco"
        );
    }

    /// SC-A07q: vacío es AUSENTE, presente-y-no-reconocido es ERROR.
    #[test]
    fn a_blank_config_value_is_absent_while_an_unknown_one_is_an_error() {
        assert_eq!(<Mode as ModeExt>::parse_config_value("").unwrap(), None);
        assert_eq!(<Mode as ModeExt>::parse_config_value("   ").unwrap(), None);
        assert_eq!(
            <Mode as ModeExt>::parse_config_value("design").unwrap(),
            Some(Mode::Design)
        );
        assert!(matches!(
            <Mode as ModeExt>::parse_config_value("banana"),
            Err(ModeParseError::Unknown { .. })
        ));
    }

    /// El error es del LIB y no arrastra al bin: `ConfigError` vive en `config.rs`, que es del
    /// binario, así que devolverlo desde acá haría incompilable el módulo.
    #[test]
    fn the_parse_error_belongs_to_the_library() {
        let e = <Mode as ModeExt>::parse_config_value("banana").unwrap_err();
        assert!(e.to_string().contains("banana"), "nombra el valor recibido");
        assert!(e.to_string().contains("code-review"), "y los tres válidos");
    }

    /// Los cinco niveles de [`ModeSource`] son distinguibles entre sí.
    ///
    /// No es ceremonia: `AgentChosen` existe **porque** una guarda tiene que poder bloquear la
    /// clasificación (nivel 4) sin bloquear la elección del agente (nivel 3), y mientras
    /// compartieron etiqueta eso era imposible. Colapsar dos variantes rompe SC-A07u y SC-A07v
    /// a la vez, así que la distinción se fija acá.
    #[test]
    fn every_mode_source_level_is_distinguishable() {
        let all = [
            ModeSource::Explicit,
            ModeSource::Configured,
            ModeSource::AgentChosen,
            ModeSource::Inferred,
            ModeSource::Default,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
            }
        }
    }
}
