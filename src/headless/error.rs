// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Errores de dominio del subsistema Headless.
//!
//! Sigue el patrón de [`crate::vault::VaultError`]: los tipos foráneos se
//! **stringifican** en lugar de envolverse, salvo el error del vault
//! ([`VaultError`]) que se **envuelve** intacto para que el exit-mapper (T9)
//! distinga su clase.
//!
//! **Invariante de seguridad:** ningún mensaje de error contiene jamás un
//! secreto — solo su clase o la etapa que falló (hereda de [`VaultError`]).

use thiserror::Error;

use crate::vault::VaultError;

/// Errores de dominio del subsistema Headless.
///
/// Cada variante nombra una clase de fallo distinguible por el borde
/// (`main.rs`) para mapearla a un exit code accionable (REQ-H23).
#[derive(Debug, Error)]
pub enum HeadlessError {
    /// La entrada (`-i`/stdin) es sintácticamente inválida: no-UTF8, envelope
    /// sin `prompt` bajo `--input-format json`, campo desconocido, clave
    /// duplicada o anidamiento patológico. El mensaje **jamás** incluye el
    /// contenido crudo del prompt (podría ser sensible). Mapea a exit 2.
    #[error("invalid input: {0}")]
    InputInvalid(String),

    /// La entrada supera `MAX_INPUT_BYTES` (DoS bound, REQ-H29). Lleva el
    /// límite en bytes, nunca el contenido. Mapea a exit 2.
    #[error("input exceeds {0} bytes")]
    InputTooLarge(usize),

    /// Error de E/S en la lectura/escritura de la entrada o la salida. Lleva el
    /// mensaje del `io::Error`, sin material sensible. Mapea a exit 1.
    #[error("I/O error: {0}")]
    Io(String),

    /// Fallo a nivel de almacenamiento (SQLite) fuera de la clase de corrupción
    /// tipada del vault. Mapea a exit 1.
    #[error("storage error: {0}")]
    Storage(String),

    /// El operador (o el usuario interactivo) no confirmó una operación o la
    /// canceló. El borde sale con código distinto de cero; no es un fallo del
    /// sistema. Mapea a exit 1.
    #[error("operation cancelled")]
    Aborted,

    /// No hay TTY y no se proveyó la passphrase por `-p`/`MAGI_PASSPHRASE`
    /// (REQ-H25/REQ-V40): headless **jamás** cuelga esperando un prompt que no
    /// puede leer. Mapea a exit 1.
    #[error("no passphrase: use -p or MAGI_PASSPHRASE in non-interactive environments")]
    PassphraseUnavailable,

    /// Error propagado desde el subsistema Vault (passphrase incorrecta, meta
    /// corrupta, DB corrupta, etc.). Se **envuelve** intacto —su `Display` ya
    /// está sanitizado— para que el exit-mapper (T9) distinga la clase concreta.
    #[error(transparent)]
    Db(VaultError),
}

/// Traduce un [`VaultError`] a su [`HeadlessError`] correspondiente.
///
/// El `match` es **exhaustivo sin comodín `_`** (MAGI CP2 run 1/2
/// Melchior/Caspar): una variante nueva de [`VaultError`] **rompe el build**,
/// forzando una decisión de mapeo explícita en lugar de una degradación
/// silenciosa. Las variantes con equivalente directo en [`HeadlessError`]
/// (`PassphraseUnavailable`/`Aborted`/`Io`/`Storage`) se mapean a él; el resto
/// se **envuelve** en [`HeadlessError::Db`] para que T9 inspeccione su clase.
impl From<VaultError> for HeadlessError {
    fn from(err: VaultError) -> Self {
        match err {
            VaultError::PassphraseUnavailable => HeadlessError::PassphraseUnavailable,
            VaultError::Aborted => HeadlessError::Aborted,
            VaultError::Io(msg) => HeadlessError::Io(msg),
            VaultError::Storage(msg) => HeadlessError::Storage(msg),
            wrapped @ (VaultError::WrongPassphrase
            | VaultError::VaultMetaCorrupt
            | VaultError::DbCorrupt { .. }
            | VaultError::SecretNotFound(_)
            | VaultError::Crypto(_)
            | VaultError::WeakPassphrase(_)
            | VaultError::ValueTooLarge(_)) => HeadlessError::Db(wrapped),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::HeadlessError;
    use crate::vault::VaultError;

    #[test]
    fn test_input_invalid_display_is_user_facing() {
        let e = HeadlessError::InputInvalid("missing prompt".into());
        assert!(e.to_string().contains("missing prompt"));
    }

    #[test]
    fn test_input_too_large_includes_the_limit_only() {
        let e = HeadlessError::InputTooLarge(10 * 1024 * 1024);
        assert!(e.to_string().contains("10485760"));
    }

    #[test]
    fn test_from_vault_passphrase_unavailable_maps_to_dedicated_variant() {
        let e: HeadlessError = VaultError::PassphraseUnavailable.into();
        assert!(matches!(e, HeadlessError::PassphraseUnavailable));
    }

    #[test]
    fn test_from_vault_wrong_passphrase_is_wrapped_in_db() {
        let e: HeadlessError = VaultError::WrongPassphrase.into();
        assert!(matches!(e, HeadlessError::Db(VaultError::WrongPassphrase)));
    }

    #[test]
    fn test_from_vault_db_corrupt_is_wrapped_and_leaks_no_secret() {
        let e: HeadlessError = VaultError::DbCorrupt {
            db_path: PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".into(),
        }
        .into();
        // Transparent Display forwards the (already-sanitized) VaultError message.
        let msg = e.to_string();
        assert!(msg.contains("data present without envelope"));
        assert!(matches!(e, HeadlessError::Db(VaultError::DbCorrupt { .. })));
    }

    #[test]
    fn test_from_vault_storage_maps_to_dedicated_variant() {
        let e: HeadlessError = VaultError::Storage("disk full".into()).into();
        assert!(matches!(e, HeadlessError::Storage(_)));
    }
}
