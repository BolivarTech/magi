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
    #[error("passphrase incorrecta")]
    WrongPassphrase,

    /// `vault_meta` está presente pero es irrecuperable **incluso tras** la
    /// corrección FEC (corrupción más allá de la capacidad del códec).
    ///
    /// Requiere una acción **explícita** del usuario; el sistema jamás se
    /// auto-repara destruyendo datos.
    #[error("vault_meta corrupto e irrecuperable")]
    VaultMetaCorrupt,

    /// No existe un secreto con el nombre indicado. El nombre no es material
    /// sensible, por lo que puede figurar en el mensaje.
    #[error("secreto no encontrado: {0}")]
    SecretNotFound(String),

    /// Fallo criptográfico propagado desde `cryptovault` (mensaje ya
    /// sanitizado por el crate — sin oráculos de decode ni de timing).
    #[error("error de cripto: {0}")]
    Crypto(String),

    /// Fallo a nivel de almacenamiento SQLite.
    #[error("error de almacenamiento: {0}")]
    Storage(String),
}

#[cfg(test)]
mod tests {
    use super::VaultError;

    #[test]
    fn test_wrong_passphrase_display_is_user_facing_and_leaks_nothing() {
        let e = VaultError::WrongPassphrase;
        assert_eq!(e.to_string(), "passphrase incorrecta");
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
}
