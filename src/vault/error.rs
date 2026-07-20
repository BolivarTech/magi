// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
//! Errores de dominio del subsistema Vault.
//!
//! Sigue el patrón de [`crate::memory::error::MemoryError`]: los tipos foráneos
//! (`cryptovault`, `rusqlite`) se **stringifican** en lugar de envolverse, para
//! mantener los tipos externos fuera de la API pública del vault y conservar
//! `Send + Sync` sin acoplarse a sus versiones.
//!
//! **Invariante de seguridad:** ningún mensaje de error contiene jamás el
//! *valor* de un secreto — solo su *nombre* o la etapa que falló.

use std::path::PathBuf;

use thiserror::Error;

/// Errores de dominio del subsistema Vault.
///
/// Cada variante nombra una etapa de fallo distinguible por el llamador; el
/// AEAD subyacente impide falsificación sin importar cuál se exponga.
#[derive(Debug, Error)]
pub enum VaultError {
    /// El desenvolvimiento (`unwrap`) de la DEK falló el tag AEAD tras la
    /// corrección FEC — es decir, la clave maestra es **incorrecta**.
    ///
    /// Es **reintentable** y **nunca** dispara un borrado de datos: ver la
    /// política de nunca-borrar (REQ-V35).
    #[error("incorrect passphrase")]
    WrongPassphrase,

    /// `vault_meta` está presente pero es irrecuperable **incluso tras** la
    /// corrección FEC (corrupción más allá de la capacidad del códec).
    ///
    /// Requiere una acción **explícita** del usuario; el sistema jamás se
    /// auto-repara destruyendo datos.
    #[error("vault metadata is corrupt and unrecoverable")]
    VaultMetaCorrupt,

    /// La DB de `.magi/` está **corrupta**: hay datos cifrados presentes pero
    /// **no** hay envelope (DEK) para descifrarlos, o falta una tabla esperada
    /// del schema (§2.1 / D-H10). **Never-delete absoluto:** este estado
    /// **jamás** dispara un borrado ni un bootstrap encima — requiere acción
    /// explícita del usuario (restaurar un backup o quitar `.magi/` a mano).
    ///
    /// La variante es **estructurada** para que el borde construya un texto de
    /// recuperación accionable: `db_path` nombra qué DB y `detail` por qué. El
    /// `Display` expone **solo** el path y la clase — **nunca** un secreto.
    #[error("database corrupt at {}: {detail}", .db_path.display())]
    DbCorrupt {
        /// Ruta de la DB de `.magi/` afectada (no es material sensible).
        db_path: PathBuf,
        /// Clase de la corrupción (p.ej. "data present without envelope" o
        /// "missing table `<name>`") — **jamás** contiene un secreto.
        detail: String,
    },

    /// No existe un secreto con el nombre indicado. El nombre no es material
    /// sensible, por lo que puede figurar en el mensaje.
    #[error("secret not found: {0}")]
    SecretNotFound(String),

    /// Fallo criptográfico propagado desde `cryptovault` (mensaje ya
    /// sanitizado por el crate — sin oráculos de decode ni de timing).
    #[error("crypto error: {0}")]
    Crypto(String),

    /// Fallo a nivel de almacenamiento SQLite.
    #[error("storage error: {0}")]
    Storage(String),

    /// No hay TTY y no se proveyó la passphrase por `-p`/`MAGI_PASSPHRASE`
    /// (REQ-V40): la passphrase **jamás** se lee de un pipe. Reintentable.
    #[error("no passphrase: use -p or MAGI_PASSPHRASE in non-interactive environments")]
    PassphraseUnavailable,

    /// La passphrase no alcanza el piso duro de fortaleza (REQ-V18). El mensaje
    /// lleva los motivos + tips, **jamás** la passphrase.
    #[error("passphrase rejected: {0}")]
    WeakPassphrase(String),

    /// Error de E/S del terminal (prompt oculto / eco). Mensaje del `io::Error`,
    /// sin material sensible.
    #[error("I/O error: {0}")]
    Io(String),

    /// El usuario no confirmó una operación destructiva (REQ-V22). El CLI
    /// sale con código de salida distinto de cero para que los scripts lo
    /// detecten; no es un fallo del sistema.
    #[error("operation cancelled")]
    Aborted,

    /// El valor supera `cryptovault::MAX_PLAINTEXT_LEN` (MAGI run 4,
    /// Caspar). Lleva el límite, nunca el valor.
    #[error("value exceeds {0} bytes")]
    ValueTooLarge(usize),
}

#[cfg(test)]
mod tests {
    use super::VaultError;

    #[test]
    fn test_wrong_passphrase_display_is_user_facing_and_leaks_nothing() {
        let e = VaultError::WrongPassphrase;
        assert_eq!(e.to_string(), "incorrect passphrase");
    }

    #[test]
    fn test_vault_meta_corrupt_is_distinct_variant() {
        let e = VaultError::VaultMetaCorrupt;
        assert!(e.to_string().to_lowercase().contains("corrupt"));
    }

    #[test]
    fn test_secret_not_found_includes_name_only() {
        let e = VaultError::SecretNotFound("OPENAI_API_KEY".to_string());
        let msg = e.to_string();
        assert!(msg.contains("OPENAI_API_KEY"));
    }

    #[test]
    fn test_aborted_display_is_user_facing_and_stable() {
        assert_eq!(VaultError::Aborted.to_string(), "operation cancelled");
    }

    #[test]
    fn test_value_too_large_includes_the_limit_only() {
        let e = VaultError::ValueTooLarge(10 * 1024 * 1024);
        assert!(e.to_string().contains("10485760"));
    }

    #[test]
    fn test_db_corrupt_display_has_path_and_class_and_leaks_nothing() {
        let e = VaultError::DbCorrupt {
            db_path: std::path::PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains(".magi-rs-memory.db"));
        assert!(msg.contains("data present without envelope"));
    }
}
