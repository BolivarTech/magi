// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Credenciales de `base_url` por **placeholder**, resueltas del vault en memoria (REQ-A16c).
//!
//! # Por qué placeholders y no redacción
//!
//! El diseño anterior aceptaba la credencial en el archivo y la redactaba antes de mostrarla.
//! Eso deja la seguridad dependiendo de que **cada** camino de salida se acuerde de redactar, y
//! ya se encontró uno que no lo hacía: el error de parseo de `toml` cita la línea ofensora, así
//! que un `magi.toml` malformado con `base_url = "https://u:p@host/v1"` escupía la credencial a
//! stderr y a los logs de CI. Cerrar ese camino no cierra la clase; cierra el camino.
//!
//! Con placeholders la propiedad es **estructural**: si el archivo no puede contener el
//! secreto, ningún camino de salida puede filtrarlo, incluidos los que nadie auditó. Es la
//! misma razón por la que las API keys nunca vivieron en `magi.toml` (REQ-A14) — `base_url` era
//! el hueco por el que una credencial sí entraba.

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

use crate::redact::{locate_userinfo, redact_url, UserinfoLocation};
use crate::vault::SecretStore;

/// Placeholder de usuario. Literal exacto, **no** un patrón: esto no es un motor de plantillas.
const USER_PLACEHOLDER: &str = "[user]";
/// Ver [`USER_PLACEHOLDER`].
const PASSWORD_PLACEHOLDER: &str = "[password]";

/// El `userinfo` que una plantilla con credenciales debe tener, exactamente.
const EXPECTED_USERINFO: &str = "[user]:[password]";

/// Qué `base_url` se está resolviendo. Determina el prefijo de las entradas de vault.
///
/// Cada `base_url` resuelve **sus** credenciales: dos endpoints distintos pueden tener usuarios
/// distintos, y compartir una entrada los acoplaría en silencio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// `base_url` de raíz ⇒ `BASE_URL_USER` / `BASE_URL_PASSWORD`.
    Root,
    /// `[magi].base_url` ⇒ `MAGI_BASE_URL_*`.
    Magi,
    /// `[embedding].base_url` ⇒ `EMBEDDING_BASE_URL_*`.
    Embedding,
}

impl Scope {
    /// Nombre de la entrada de vault con el usuario.
    #[must_use]
    pub fn user_entry(self) -> &'static str {
        match self {
            Self::Root => "BASE_URL_USER",
            Self::Magi => "MAGI_BASE_URL_USER",
            Self::Embedding => "EMBEDDING_BASE_URL_USER",
        }
    }

    /// Nombre de la entrada de vault con la contraseña.
    #[must_use]
    pub fn password_entry(self) -> &'static str {
        match self {
            Self::Root => "BASE_URL_PASSWORD",
            Self::Magi => "MAGI_BASE_URL_PASSWORD",
            Self::Embedding => "EMBEDDING_BASE_URL_PASSWORD",
        }
    }
}

/// Lo que puede salir mal al leer o resolver una `base_url`.
///
/// **Ningún mensaje repite el valor ofensor**: un error de seguridad que imprime el secreto que
/// está rechazando no sirve para nada.
#[derive(Debug, thiserror::Error)]
pub enum EndpointError {
    /// La `base_url` trae una credencial literal en vez de los placeholders.
    #[error(
        "`base_url` carries a literal credential. Replace it with \
         `{USER_PLACEHOLDER}:{PASSWORD_PLACEHOLDER}` and store the values in the vault: \
         `magi-rs vault set {user_entry}` and `magi-rs vault set {password_entry}`"
    )]
    LiteralCredential {
        /// Entrada de vault para el usuario, según el scope.
        user_entry: &'static str,
        /// Entrada de vault para la contraseña, según el scope.
        password_entry: &'static str,
    },

    /// El placeholder está declarado y la entrada de vault no existe.
    #[error(
        "`base_url` declares a placeholder but entry {entry} is missing from the vault. \
         Create it with `magi-rs vault set {entry}`"
    )]
    MissingVaultEntry {
        /// La entrada que hace falta.
        entry: &'static str,
    },

    /// Un placeholder que no es ninguno de los dos conocidos.
    #[error(
        "unknown placeholder in `base_url`: only `{USER_PLACEHOLDER}` and \
         `{PASSWORD_PLACEHOLDER}` are accepted, in the `userinfo` position"
    )]
    UnknownPlaceholder,

    /// La URL no se pudo recorrer, así que no se puede afirmar que no traiga un secreto.
    #[error("`base_url` does not have a recognizable form (`scheme://host/...`)")]
    Unparseable,
}

/// `base_url` **tal como está en el archivo**: con `[user]`/`[password]`, nunca el secreto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointTemplate(String);

impl EndpointTemplate {
    /// Lee una `base_url` del archivo y rechaza toda credencial literal (REQ-A16c).
    ///
    /// Reutiliza el localizador de autoridad de [`crate::redact`], que es el mismo que usa la
    /// redacción: la regla de dónde vive el `userinfo` se escribe **una vez**. Si se escribiera
    /// dos, desincronizarse significaría que uno de los dos deja de ver una credencial.
    ///
    /// # Errors
    ///
    /// [`EndpointError::LiteralCredential`] si el `userinfo` no son exactamente los dos
    /// placeholders; [`EndpointError::UnknownPlaceholder`] si trae otro placeholder;
    /// [`EndpointError::Unparseable`] si la URL no se pudo recorrer.
    pub fn parse(raw: &str) -> Result<Self, EndpointError> {
        match locate_userinfo(raw) {
            // Sin `userinfo` no hay credencial que validar — el caso común no paga nada.
            UserinfoLocation::Absent => Ok(Self(raw.to_string())),
            UserinfoLocation::Unparseable => Err(EndpointError::Unparseable),
            UserinfoLocation::Found { start, end } => {
                let Some(userinfo) = raw.get(start..end) else {
                    return Err(EndpointError::Unparseable);
                };
                if userinfo == EXPECTED_USERINFO {
                    return Ok(Self(raw.to_string()));
                }
                // Un placeholder mal escrito se nombra como tal; cualquier otra cosa es una
                // credencial literal. La distinción importa porque los arreglos son distintos.
                if userinfo.contains('[') || userinfo.contains(']') {
                    return Err(EndpointError::UnknownPlaceholder);
                }
                // El scope real lo pone el llamador al construir el mensaje; acá se nombra el
                // de raíz, que es el caso que un usuario ve primero.
                Err(EndpointError::LiteralCredential {
                    user_entry: Scope::Root.user_entry(),
                    password_entry: Scope::Root.password_entry(),
                })
            }
        }
    }

    /// El texto de la plantilla — **seguro por construcción, NO necesita redacción**.
    ///
    /// Lo que hay acá es `https://[user]:[password]@host/v1`: por REQ-A16c una credencial
    /// literal es error de configuración, así que la plantilla no puede contener un secreto.
    /// El que sí necesita redacción es [`ResolvedEndpoint`], que es el de después.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Sustituye los placeholders con los valores del vault, **en memoria**.
    ///
    /// Falla **cerrado**: una entrada ausente detiene el proceso nombrándola. Sustituir vacío
    /// daría un 401 en la primera consulta, sin relación aparente con la configuración — la
    /// misma clase de fallo tardío que D-A01 eliminó.
    ///
    /// # Errors
    ///
    /// [`EndpointError::MissingVaultEntry`] con la entrada que falta y el comando que la crea.
    pub fn resolve(
        &self,
        vault: &mut dyn SecretStore,
        scope: Scope,
    ) -> Result<ResolvedEndpoint, EndpointError> {
        // La sustitución se acota al `userinfo` de la AUTORIDAD, no a la cadena entera.
        //
        // Buscar el placeholder en todo el texto es lo que hacía la primera versión, y un
        // `https://host/v1/[user]` —donde `[user]` es un segmento literal del path— salía a
        // buscar una credencial al vault y fallaba cerrado por una entrada que nadie tenía por
        // qué crear. `parse` ya usa el mismo localizador; resolver con otra regla las
        // desincroniza.
        let UserinfoLocation::Found { start, end } = locate_userinfo(&self.0) else {
            // Sin `userinfo` no hay nada que sustituir, y el caso común —Ollama local,
            // keyless— no paga ni un lookup.
            return Ok(ResolvedEndpoint(self.0.clone()));
        };
        let (Some(prefix), Some(userinfo), Some(tail)) = (
            self.0.get(..start),
            self.0.get(start..end),
            self.0.get(end..),
        ) else {
            return Err(EndpointError::Unparseable);
        };
        if userinfo != EXPECTED_USERINFO {
            // `parse` ya garantizó que un `userinfo` distinto no llega hasta acá.
            return Ok(ResolvedEndpoint(self.0.clone()));
        }

        let user = vault
            .get(scope.user_entry())
            .map_err(|_| EndpointError::MissingVaultEntry {
                entry: scope.user_entry(),
            })?;
        let password =
            vault
                .get(scope.password_entry())
                .map_err(|_| EndpointError::MissingVaultEntry {
                    entry: scope.password_entry(),
                })?;

        let mut out = String::with_capacity(self.0.len());
        out.push_str(prefix);
        out.push_str(user.as_str());
        out.push(':');
        out.push_str(password.as_str());
        out.push_str(tail);
        Ok(ResolvedEndpoint(out))
    }
}

impl fmt::Display for EndpointTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// URL con los placeholders ya sustituidos.
///
/// **Solo [`EndpointTemplate::resolve`] la construye**, y resolver exige el vault — por eso un
/// `&str` sin resolver no puede llegar a un provider por accidente.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint(String);

impl ResolvedEndpoint {
    /// La URL efectiva. Quien la muestre debe pasarla por [`crate::redact::redact_url`].
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ResolvedEndpoint {
    /// A mano y redactado: un `derive(Debug)` acá es la forma más fácil de filtrar la
    /// credencial sin darse cuenta — basta un `{:?}` en un error o en una traza.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResolvedEndpoint({})", redact_url(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::VaultError;
    use crate::vault::{SecretEntry, SecretStore};
    use std::collections::BTreeMap;
    use zeroize::Zeroizing;

    /// Vault de prueba: un mapa en memoria, sin cripto ni SQLite.
    struct StubVault {
        /// Entradas disponibles.
        entries: BTreeMap<String, String>,
    }

    impl StubVault {
        fn with(pairs: &[(&str, &str)]) -> Self {
            Self {
                entries: pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            }
        }
        fn empty() -> Self {
            Self {
                entries: BTreeMap::new(),
            }
        }
    }

    impl SecretStore for StubVault {
        fn set(&mut self, name: &str, value: &str) -> Result<(), VaultError> {
            self.entries.insert(name.to_string(), value.to_string());
            Ok(())
        }
        fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
            self.entries
                .get(name)
                .map(|v| Zeroizing::new(v.clone()))
                .ok_or_else(|| VaultError::SecretNotFound(name.to_string()))
        }
        fn remove(&mut self, name: &str) -> Result<(), VaultError> {
            self.entries.remove(name);
            Ok(())
        }
        fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
            Ok(Vec::new())
        }
        fn contains(&mut self, name: &str) -> Result<bool, VaultError> {
            Ok(self.entries.contains_key(name))
        }
    }

    /// SC-A16d: credencial LITERAL es error, y el mensaje no la repite.
    #[test]
    fn a_literal_credential_is_a_config_error_that_does_not_echo_it() {
        let err = EndpointTemplate::parse("https://juan:s3cr3t@host/v1").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("[user]") && msg.contains("[password]"),
            "nombra el reemplazo: {msg}"
        );
        assert!(
            msg.contains("vault set"),
            "y el comando que lo guarda: {msg}"
        );
        assert!(
            !msg.contains("s3cr3t"),
            "un error de seguridad que repite el secreto no sirve: {msg}"
        );
        assert!(!msg.contains("juan"), "{msg}");
    }

    /// SC-A16e: el placeholder se resuelve del vault, y la plantilla no muestra nada.
    #[test]
    fn placeholders_resolve_from_the_vault_in_memory() {
        let mut vault =
            StubVault::with(&[("BASE_URL_USER", "juan"), ("BASE_URL_PASSWORD", "s3cr3t")]);
        let tpl = EndpointTemplate::parse("https://[user]:[password]@host/v1").unwrap();

        let resolved = tpl.resolve(&mut vault, Scope::Root).unwrap();
        assert_eq!(resolved.as_str(), "https://juan:s3cr3t@host/v1");
        // La plantilla es lo que se muestra: ya es segura, no hace falta redactarla.
        assert_eq!(tpl.as_str(), "https://[user]:[password]@host/v1");
    }

    /// El `Debug` de la URL resuelta redacta: un `derive` acá es la forma más fácil de filtrar.
    #[test]
    fn the_resolved_endpoints_debug_never_shows_the_credential() {
        let mut vault =
            StubVault::with(&[("BASE_URL_USER", "juan"), ("BASE_URL_PASSWORD", "s3cr3t")]);
        let resolved = EndpointTemplate::parse("https://[user]:[password]@host/v1")
            .unwrap()
            .resolve(&mut vault, Scope::Root)
            .unwrap();
        let shown = format!("{resolved:?}");
        assert!(!shown.contains("s3cr3t"), "filtró por Debug: {shown}");
        assert!(shown.contains("host"), "y el host sigue visible: {shown}");
    }

    /// SC-A16f: placeholder sin entrada falla CERRADO, no sustituye vacío.
    #[test]
    fn a_missing_vault_entry_fails_closed_naming_the_entry() {
        let mut vault = StubVault::with(&[("BASE_URL_USER", "juan")]); // falta la password
        let err = EndpointTemplate::parse("https://[user]:[password]@host/v1")
            .unwrap()
            .resolve(&mut vault, Scope::Root)
            .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("BASE_URL_PASSWORD"),
            "nombra la entrada: {msg}"
        );
        assert!(msg.contains("vault set"), "y cómo crearla: {msg}");
    }

    /// Cada `base_url` resuelve SUS credenciales: dos endpoints pueden tener usuarios distintos.
    #[test]
    fn each_scope_reads_its_own_vault_entries() {
        let mut vault = StubVault::with(&[
            ("BASE_URL_USER", "root-u"),
            ("BASE_URL_PASSWORD", "root-p"),
            ("MAGI_BASE_URL_USER", "trio-u"),
            ("MAGI_BASE_URL_PASSWORD", "trio-p"),
            ("EMBEDDING_BASE_URL_USER", "emb-u"),
            ("EMBEDDING_BASE_URL_PASSWORD", "emb-p"),
        ]);
        let tpl = EndpointTemplate::parse("https://[user]:[password]@host/v1").unwrap();
        assert!(tpl
            .resolve(&mut vault, Scope::Root)
            .unwrap()
            .as_str()
            .contains("root-u"));
        assert!(tpl
            .resolve(&mut vault, Scope::Magi)
            .unwrap()
            .as_str()
            .contains("trio-u"));
        assert!(tpl
            .resolve(&mut vault, Scope::Embedding)
            .unwrap()
            .as_str()
            .contains("emb-u"));
    }

    /// Solo esos dos placeholders, y solo en la autoridad. No es un motor de plantillas.
    #[test]
    fn only_the_two_known_placeholders_in_the_authority_are_recognized() {
        assert!(EndpointTemplate::parse("https://[banana]@host/v1").is_err());
        // Fuera de la autoridad es texto literal del path, no un placeholder.
        let tpl = EndpointTemplate::parse("https://host/v1/[user]").unwrap();
        assert_eq!(
            tpl.resolve(&mut StubVault::empty(), Scope::Root)
                .unwrap()
                .as_str(),
            "https://host/v1/[user]"
        );
    }

    /// Una URL sin credenciales pasa igual: el caso común no paga nada.
    #[test]
    fn a_plain_url_without_userinfo_resolves_to_itself() {
        let tpl = EndpointTemplate::parse("http://localhost:11434/v1").unwrap();
        assert_eq!(
            tpl.resolve(&mut StubVault::empty(), Scope::Root)
                .unwrap()
                .as_str(),
            "http://localhost:11434/v1"
        );
    }

    /// Sin `://` no hay autoridad que recorrer: `locate_userinfo` devuelve `Unparseable` y
    /// `parse` lo propaga como [`EndpointError::Unparseable`], en vez de asumir "sin credencial".
    ///
    /// Cubre el brazo `UserinfoLocation::Unparseable => Err(EndpointError::Unparseable)` de
    /// `EndpointTemplate::parse`, que no tenía ningún caso de prueba: verificado leyendo
    /// `locate_userinfo` (`src/redact.rs`) — el primer `let Some(scheme_end) = raw.find("://")
    /// else { return Unparseable }` es alcanzable con cualquier texto que no contenga `"://"`.
    #[test]
    fn a_url_without_a_scheme_separator_is_rejected_as_unparseable() {
        let err = EndpointTemplate::parse("localhost:11434/v1").unwrap_err();
        assert!(
            matches!(err, EndpointError::Unparseable),
            "esperaba Unparseable, salió {err:?}"
        );
    }

    /// El `Display` de la plantilla emite exactamente lo que guarda — es el mismo texto que
    /// `as_str()`, así que un consumidor que haga `format!("{tpl}")` ve la plantilla completa.
    #[test]
    fn display_renders_the_same_text_as_as_str() {
        let tpl = EndpointTemplate::parse("https://[user]:[password]@host/v1").unwrap();
        assert_eq!(format!("{tpl}"), tpl.as_str());
        assert_eq!(tpl.to_string(), "https://[user]:[password]@host/v1");
    }
}
