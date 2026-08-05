// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Redacción de credenciales en URLs, **por posición y nunca por contenido** (REQ-A16).
//!
//! # Por qué vive en el LIB y no bajo `system/`
//!
//! Procesa entrada no confiable, así que es el candidato de §0.3 a `cargo fuzz`. `system/` es
//! del binario y no es alcanzable ni desde un fuzz target ni desde `tests/`.

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

use std::error::Error;
use std::fmt;

/// Lo que reemplaza a una credencial, y a una URL entera cuando no se pudo recorrer.
const FULLY_REDACTED: &str = "***";

/// Separador que abre la autoridad de una URL.
const SCHEME_SEPARATOR: &str = "://";

/// Redacta el `userinfo` de una URL **por POSICIÓN, nunca por contenido** (REQ-A16).
///
/// Regla exacta, en tres pasos:
/// 1. La **autoridad** empieza tras `://` y termina en el primer `/`, `?` o `#`.
/// 2. Dentro de esa ventana —y solo ahí— el `userinfo` es todo lo anterior al **último** `@`.
///    El último, no el primero: `user:p@ss@host` es una contraseña con `@`, legal en RFC 3986.
/// 3. Sin `@` dentro de la autoridad no hay `userinfo`, y no se toca nada.
///
/// **Por qué posicional y no por contenido:** «decodificar y después redactar» pierde contra el
/// doble percent-encoding — `%2570` decodifica una vez a `%70`, que sigue encodeado, y
/// decodificar en bucle invita a una bomba de decodificación. La posición del `userinfo` **no
/// depende de la codificación de su contenido**, así que la regla posicional vale para
/// cualquier codificación, presente o futura.
///
/// Los hosts IPv6 entre corchetes entran sin caso especial: el último `@` de la autoridad cae
/// antes del `[`, y la regla nunca busca `:`, así que los dos puntos de la dirección no se
/// confunden con el separador `usuario:clave`.
///
/// Una URL que no parsea se redacta **entera**: es justo donde un secreto puede estar en un
/// lugar inesperado, así que la dirección segura de fallo es esconder de más.
///
/// # Examples
///
/// ```
/// use magi_rs::redact::redact_url;
///
/// assert_eq!(redact_url("https://user:pass@host/v1"), "https://***@host/v1");
/// assert_eq!(redact_url("https://host/ruta@cosa"), "https://host/ruta@cosa");
/// ```
#[must_use]
pub fn redact_url(raw: &str) -> String {
    match locate_userinfo(raw) {
        UserinfoLocation::Unparseable => FULLY_REDACTED.to_string(),
        UserinfoLocation::Absent => raw.to_string(),
        UserinfoLocation::Found { start, end } => {
            let (Some(prefix), Some(tail)) = (raw.get(..start), raw.get(end..)) else {
                return FULLY_REDACTED.to_string();
            };
            let mut out = String::with_capacity(raw.len());
            out.push_str(prefix);
            out.push_str(FULLY_REDACTED);
            out.push_str(tail);
            out
        }
    }
}

/// Dónde cae el `userinfo` de una URL, si es que hay uno.
///
/// Se expone porque **dos** módulos necesitan la misma regla de autoridad: este redacta lo que
/// encuentra y `magi::endpoint` rechaza lo que encuentra si no son los placeholders. Escribir
/// el recorrido dos veces es cómo se desincronizan (B3), y acá desincronizarse significa que
/// uno de los dos deja de ver una credencial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserinfoLocation {
    /// La URL no se pudo recorrer. Quien la reciba debe **fallar hacia esconder**: es donde un
    /// secreto puede estar en un lugar inesperado.
    Unparseable,
    /// La autoridad no tiene `@`, así que no hay `userinfo` y no hay nada que ocultar.
    Absent,
    /// Rango `[start, end)` del `userinfo`, **sin** incluir el `@` que lo cierra.
    Found {
        /// Primer byte del `userinfo`, justo después de `://`.
        start: usize,
        /// Byte del `@` que cierra el `userinfo`.
        end: usize,
    },
}

/// Localiza el `userinfo` aplicando la regla de autoridad de RFC 3986.
///
/// La regla completa está documentada en [`redact_url`], que es su primer consumidor.
///
/// No usa `&raw[a..b]`: el bloque de atributos de este módulo incluye
/// `deny(clippy::string_slice, clippy::indexing_slicing)`, y `str::get` devuelve `Option` en
/// vez de panicar en una frontera de carácter — que es la garantía que se quiere en una
/// función que recorre entrada no confiable.
#[must_use]
pub fn locate_userinfo(raw: &str) -> UserinfoLocation {
    let Some(scheme_end) = raw.find(SCHEME_SEPARATOR) else {
        return UserinfoLocation::Unparseable;
    };
    let authority_start = scheme_end + SCHEME_SEPARATOR.len();
    let Some(rest) = raw.get(authority_start..) else {
        return UserinfoLocation::Unparseable;
    };
    // La autoridad termina en el primer `/`, `?` o `#`; sin ninguno, es todo el resto.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let Some(authority) = rest.get(..authority_end) else {
        return UserinfoLocation::Unparseable;
    };
    // ÚLTIMO `@` de la autoridad: `user:p@ss@host` es una contraseña con `@`, legal en RFC
    // 3986. Sin `@` no hay userinfo — un `@` en el path no es credencial.
    let Some(at) = authority.rfind('@') else {
        return UserinfoLocation::Absent;
    };
    UserinfoLocation::Found {
        start: authority_start,
        end: authority_start + at,
    }
}

/// Texto de error ya redactado. **Sus únicos constructores son [`redact_foreign_error`] y
/// [`redact_foreign_text`].**
///
/// Es lo que impide que un `String` sin redactar llegue a un error de dominio: sin el newtype
/// el camino sin redactar queda a un `.into()` de distancia, y la defensa vuelve a depender de
/// que alguien se acuerde en cada sitio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeErrorText(String);

impl SafeErrorText {
    /// El texto, ya seguro de mostrar.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SafeErrorText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Caracteres que pueden formar parte de un esquema de URL (RFC 3986: `ALPHA *( ALPHA / DIGIT
/// / "+" / "-" / "." )`).
fn is_scheme_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
}

/// Caracteres que terminan una URL embebida en prosa.
///
/// Deliberadamente generoso por el lado del final: incluir de más recorta la URL antes y a lo
/// sumo deja visible un tramo del host, mientras que incluir de menos podría dejar la
/// credencial fuera de la ventana redactada.
fn ends_embedded_url(c: char) -> bool {
    c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`' | ',' | ';' | ')')
}

/// Redacta las URLs **embebidas en cualquier prosa de un TERCERO**, conservando el resto —
/// el motor detrás de [`redact_foreign_error`], expuesto directamente para llamadores que no
/// tienen un `&dyn Error` que envolver.
///
/// # Por qué existe SEPARADA de [`redact_foreign_error`] (fix round 3, hallazgo CRÍTICO)
///
/// `redact_foreign_error` toma `&dyn Error` porque su primer consumidor (`explain_magi_error`)
/// tenía uno a mano. Pero la lógica entera opera sobre el `String` que produce `err.to_string()`
/// — nada de lo que sigue usa el `Error` en sí. Un segundo llamador con texto de un tercero que
/// **ya es `String`** (p. ej. `MagiReport::failed_agents`, cuyo valor es literalmente
/// `MagiError::Provider(e).to_string()` armado por magi-core, verificado contra
/// `orchestrator.rs::dispatch_one_agent`) no tiene un `Error` que envolver — envolverlo en un
/// `impl Error` de un solo uso para satisfacer una firma sería un workaround que deja intacto
/// el hueco real: el día que alguien tenga un `String` foráneo y no un `Error`, va a escribir
/// ESE `.to_string()` sin pasar por acá, exactamente el defecto que este fix cierra.
///
/// # Por qué hace falta redactar esto, y por qué una lista de sitios no alcanza
///
/// Los `format!` que escribimos nosotros se pueden enumerar y auditar. Este camino no: el
/// texto lo arma **otra crate** con la URL que le pasamos, así que ninguna revisión de
/// nuestros formateadores lo ve. Todo `String` que empaquete un error foráneo pasa por acá.
///
/// No es una segunda implementación de [`redact_url`]: barre las URLs del mensaje y le aplica
/// a cada una **la misma regla posicional**.
#[must_use]
pub fn redact_foreign_text(raw: &str) -> SafeErrorText {
    let mut out = String::with_capacity(raw.len());
    let mut cursor = 0usize;

    while let Some(found) = raw.get(cursor..).and_then(|r| r.find(SCHEME_SEPARATOR)) {
        let sep_at = cursor + found;
        // El esquema arranca donde terminan los caracteres válidos de esquema hacia atrás.
        let Some(before) = raw.get(cursor..sep_at) else {
            break;
        };
        let scheme_len = before
            .chars()
            .rev()
            .take_while(|c| is_scheme_char(*c))
            .map(char::len_utf8)
            .sum::<usize>();
        let url_start = sep_at - scheme_len;

        let after_sep = sep_at + SCHEME_SEPARATOR.len();
        let Some(tail) = raw.get(after_sep..) else {
            break;
        };
        let url_end = after_sep + tail.find(ends_embedded_url).unwrap_or(tail.len());

        let (Some(lead), Some(url)) = (raw.get(cursor..url_start), raw.get(url_start..url_end))
        else {
            break;
        };
        out.push_str(lead);
        out.push_str(&redact_url(url));
        cursor = url_end;
    }

    if let Some(rest) = raw.get(cursor..) {
        out.push_str(rest);
    }
    SafeErrorText(out)
}

/// Redacta las URLs **embebidas en la prosa de un error foráneo**, conservando el resto.
///
/// Envoltorio de [`redact_foreign_text`] sobre `err.to_string()` — ver su rustdoc para el
/// porqué de la separación. Se mantiene por compatibilidad con los llamadores que sí tienen un
/// `&dyn Error` (`explain_magi_error`).
#[must_use]
pub fn redact_foreign_error(err: &dyn Error) -> SafeErrorText {
    redact_foreign_text(&err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A13: userinfo simple.
    #[test]
    fn userinfo_is_redacted_and_the_host_survives() {
        assert_eq!(
            redact_url("https://user:pass@host/v1"),
            "https://***@host/v1"
        );
    }

    /// SC-A13c: el doble percent-encoding NO esquiva, porque la regla es POSICIONAL.
    #[test]
    fn double_percent_encoding_does_not_evade_redaction() {
        let doubly = "https://%2575%2573%2565%2572:%2570@host/v1";
        let out = redact_url(doubly);
        assert!(!out.contains("%2570"), "quedó credencial: {out}");
        assert!(out.contains("host"));
    }

    /// SC-A13d: un `@` en el PATH no dispara redacción.
    #[test]
    fn an_at_sign_in_the_path_is_not_userinfo() {
        assert_eq!(
            redact_url("https://host/ruta@cosa"),
            "https://host/ruta@cosa"
        );
    }

    /// SC-A13d: contraseña que contiene `@` — gana el ÚLTIMO `@` de la autoridad.
    #[test]
    fn the_last_at_within_the_authority_wins() {
        assert_eq!(
            redact_url("https://user:p@ss@host/v1"),
            "https://***@host/v1"
        );
    }

    /// IPv6 entre corchetes: entra en la regla sin caso especial.
    #[test]
    fn bracketed_ipv6_hosts_are_handled_without_a_special_case() {
        assert_eq!(redact_url("http://[::1]:11434/v1"), "http://[::1]:11434/v1");
        assert_eq!(
            redact_url("http://u:p@[::1]:11434/v1"),
            "http://***@[::1]:11434/v1"
        );
    }

    /// Dirección segura de fallo: lo que no parsea se redacta ENTERO.
    #[test]
    fn an_unparseable_url_is_redacted_whole() {
        assert_eq!(redact_url("no es una url"), "***");
    }

    /// Un error FORÁNEO trae la URL embebida en su prosa, y ahí también se redacta.
    ///
    /// Es el camino que una lista de `format!` propios no puede ver: el texto lo arma otra
    /// crate con la URL que le pasamos nosotros.
    #[test]
    fn a_foreign_errors_embedded_url_is_redacted_while_its_prose_survives() {
        let err = std::io::Error::other("connect to https://user:hunter2@host:8443/v1 failed");
        let safe = redact_foreign_error(&err);
        assert!(
            !safe.as_str().contains("hunter2"),
            "filtró: {}",
            safe.as_str()
        );
        assert!(
            safe.as_str().contains("host:8443"),
            "el host sigue siendo accionable"
        );
        assert!(safe.as_str().contains("failed"), "y la prosa se conserva");
    }

    /// Sin URL adentro, el texto pasa intacto: redactar de más lo volvería inservible.
    #[test]
    fn a_foreign_error_without_a_url_is_left_alone() {
        let err = std::io::Error::other("connection reset by peer");
        assert_eq!(
            redact_foreign_error(&err).as_str(),
            "connection reset by peer"
        );
    }

    /// Varias URLs en el mismo mensaje: se redactan TODAS, no la primera.
    #[test]
    fn every_embedded_url_is_redacted_not_just_the_first() {
        let err =
            std::io::Error::other("tried https://a:b@one/v1 then https://c:d@two/v1 and gave up");
        let safe = redact_foreign_error(&err);
        assert!(!safe.as_str().contains("a:b"), "primera: {}", safe.as_str());
        assert!(!safe.as_str().contains("c:d"), "segunda: {}", safe.as_str());
        assert!(safe.as_str().contains("one") && safe.as_str().contains("two"));
    }

    /// Fix round 3: [`redact_foreign_text`] es el motor `&str` que
    /// [`redact_foreign_error`] envuelve — se prueba DIRECTO, sin pasar por un
    /// `&dyn Error`, para el llamador que ya tiene un `String` foráneo (p. ej.
    /// `MagiReport::failed_agents`, que ES `MagiError::Provider(e).to_string()`).
    #[test]
    fn redact_foreign_text_redacts_a_string_with_no_error_to_wrap() {
        let raw = "network error: connect to https://user:hunter2@host:8443/v1 failed";
        let safe = redact_foreign_text(raw);
        assert!(
            !safe.as_str().contains("hunter2"),
            "filtró: {}",
            safe.as_str()
        );
        assert!(safe.as_str().contains("host:8443"), "sigue accionable");
    }

    /// Edge case (B13): sin URL adentro, pasa intacto — mismo contrato que
    /// [`redact_foreign_error`] sobre el mismo texto.
    #[test]
    fn redact_foreign_text_leaves_url_free_text_alone() {
        assert_eq!(
            redact_foreign_text("connection reset by peer").as_str(),
            "connection reset by peer"
        );
    }
}
