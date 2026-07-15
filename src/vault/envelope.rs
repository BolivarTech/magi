// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
//! Primitivas de envelope DEK/KEK con `vault_meta` protegido por FEC.
//!
//! # Modelo
//!
//! Una **DEK** (Data Encryption Key) aleatoria de 32 B cifra todos los registros;
//! la clave maestra deriva una **KEK** (Key Encryption Key) que *envuelve* la DEK
//! (`wrap_key`, salt-as-AAD). La DEK envuelta y el salt viven en `vault_meta`.
//! Cambiar la clave maestra solo re-envuelve la DEK (O(1)); los datos no se
//! re-cifran.
//!
//! # Capas de FEC (por qué se FEC-encodea el blob del crate)
//!
//! `cryptovault::CryptoVault::wrap_key` ya aplica `AEAD → FEC → base64`, pero el
//! **base64 queda como capa externa**: un bit podrido en el texto base64
//! almacenado rompe el `base64-decode` **antes** de que el FEC interno del crate
//! pueda corregirlo. Por eso este módulo aplica **su propia** capa
//! [`ConcatenatedFec`] sobre la representación base64 en disco: corrige el
//! bit-rot de la *representación almacenada* de modo que el `base64-decode` del
//! crate nunca vea entrada corrupta.
//!
//! ## Alcance exacto de la corrección (precisión — code review Loop 1)
//!
//! La cobertura FEC **no es uniforme sobre todo el blob**: [`fec_encode`] antepone
//! un prefijo **crudo** de longitud (`u32` LE, [`LEN_PREFIX`] bytes) **fuera** de la
//! región FEC. Un bit podrido **dentro del prefijo** (4 bytes de ~600) **no** se
//! auto-corrige — pero **falla seguro**: produce siempre [`VaultError::VaultMetaCorrupt`]
//! (nunca panic, nunca una DEK incorrecta), porque un `pre_len` corrupto lo rechaza
//! el chequeo de bloques de la RS o falla downstream (largo de salt inválido / tag
//! AEAD). Verificado exhaustivamente (los 32 single-bit flips del prefijo).
//! *No se puede meter la longitud dentro de la región FEC:* `ConcatenatedFec::decode`
//! **exige** el `pre_len` como parámetro para decodificar, así que el prefijo debe ir
//! por fuera. Endurecerlo (p.ej. redundancia sobre el prefijo) es un follow-up de
//! robustez, no un defecto de correctitud (`dev-docs/PENDING_IMPLEMENTATION.md`).
//!
//! Nota: sobre este camino la FEC **interna del crate** rara vez corrige algo — si el
//! `fec_decode` externo tiene éxito, ya recuperó los bytes exactos, y [`open_envelope`]
//! solo entonces llama a `unwrap_key`; si falla, retorna `VaultMetaCorrupt` sin llegar
//! a la FEC interna. La FEC del crate sigue siendo la defensa del payload en tránsito
//! por otros canales; en disco, la capa externa es la que corrige.
//!
//! # Distinción corrupción vs. clave equivocada
//!
//! [`open_envelope`] evalúa la FEC **antes** que el AEAD, de modo que:
//! - un `fec_decode` fallido ⇒ [`VaultError::VaultMetaCorrupt`] (dato corrupto);
//! - un `unwrap_key` fallido tras un FEC exitoso (tag AEAD) ⇒
//!   [`VaultError::WrongPassphrase`] (clave equivocada).

use cryptovault::fec::{ConcatenatedFec, ErrorCorrection};
use cryptovault::{CryptoError, CryptoVault};
use zeroize::Zeroizing;

use crate::vault::VaultError;

/// Ancho del prefijo de longitud original (`u32` little-endian) que
/// [`fec_encode`] antepone al blob para que [`fec_decode`] sea auto-descriptivo.
const LEN_PREFIX: usize = 4;

/// Salida de [`bootstrap_envelope`]: `(salt_fec, wrapped_dek_fec, dek)`.
///
/// Las dos primeras entradas van a `vault_meta` (FEC-encoded); la tercera es la
/// DEK en claro para cachear en memoria.
type Bootstrapped = (Vec<u8>, Vec<u8>, Zeroizing<Vec<u8>>);

/// Traduce un [`cryptovault::CryptoError`] al dominio del vault.
///
/// El mapeo es inequívoco porque cada etapa produce una variante distinta:
/// `Cipher` solo lo genera el AEAD (fallo de tag = clave/AAD equivocada);
/// `ErrorCorrection`/`Encoding`/`InvalidInput` solo la capa FEC/framing.
fn map_crypto_err(e: CryptoError) -> VaultError {
    match e {
        CryptoError::Cipher(_) => VaultError::WrongPassphrase,
        CryptoError::ErrorCorrection(_)
        | CryptoError::Encoding(_)
        | CryptoError::InvalidInput(_) => VaultError::VaultMetaCorrupt,
        CryptoError::KeyDerivation(m) => VaultError::Crypto(m),
    }
}

/// Envuelve `bytes` en una capa [`ConcatenatedFec`] keyless, con la longitud
/// original prefijada (`u32` LE) para que el decode sea auto-descriptivo.
///
/// La FEC es un códec **sin clave**: corrige bit-rot de la representación en
/// disco sin necesitar el secreto.
fn fec_encode(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(LEN_PREFIX + bytes.len());
    // La longitud de `salt`/`wrapped_dek` cabe holgada en u32 (nunca > 10 MiB).
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&ConcatenatedFec::default().encode(bytes));
    out
}

/// Recupera los bytes originales de un blob producido por [`fec_encode`].
///
/// # Errors
///
/// [`VaultError::VaultMetaCorrupt`] si el prefijo de longitud falta o el
/// `ConcatenatedFec::decode` no logra corregir la corrupción.
fn fec_decode(blob: &[u8]) -> Result<Vec<u8>, VaultError> {
    let prefix = blob
        .get(0..LEN_PREFIX)
        .ok_or(VaultError::VaultMetaCorrupt)?;
    let len_arr: [u8; LEN_PREFIX] = prefix
        .try_into()
        .map_err(|_| VaultError::VaultMetaCorrupt)?;
    let pre_len = u32::from_le_bytes(len_arr) as usize;
    let payload = blob.get(LEN_PREFIX..).ok_or(VaultError::VaultMetaCorrupt)?;
    ConcatenatedFec::default()
        .decode(payload, pre_len)
        .map_err(map_crypto_err)
}

/// Bootstrapea un envelope nuevo para una clave maestra dada.
///
/// Genera una DEK y un salt aleatorios, deriva la KEK desde `master` + salt, y
/// envuelve la DEK. Devuelve `(salt_fec, wrapped_dek_fec, dek)`: las dos primeras
/// entradas van a `vault_meta` (FEC-encoded), la DEK se cachea en memoria.
///
/// `master` es la clave maestra vigente **como `&str`** (el keyring entrega
/// base64, UTF-8 válido); nunca bytes crudos no-UTF8.
///
/// # Errors
///
/// [`VaultError::Crypto`] si la generación de material aleatorio, la derivación
/// de la KEK o el envoltorio de la DEK fallan.
pub fn bootstrap_envelope(vault: &CryptoVault, master: &str) -> Result<Bootstrapped, VaultError> {
    let salt = cryptovault::generate_salt().map_err(map_crypto_err)?;
    let dek = cryptovault::generate_dek().map_err(map_crypto_err)?;
    let kek = vault.derive_key(master, &salt).map_err(map_crypto_err)?;
    let wrapped = vault.wrap_key(&kek, &salt, &dek).map_err(map_crypto_err)?;
    let salt_fec = fec_encode(&salt);
    let wrapped_fec = fec_encode(wrapped.as_bytes());
    Ok((salt_fec, wrapped_fec, dek))
}

/// Abre un envelope existente, recuperando la DEK.
///
/// Corrige el bit-rot de `salt_fec`/`wrapped_dek_fec` (FEC) **antes** de
/// desenvolver, y distingue corrupción de clave equivocada.
///
/// # Errors
///
/// - [`VaultError::VaultMetaCorrupt`] si la FEC no puede recuperar `salt` o el
///   blob envuelto (corrupción más allá de su capacidad), o si el blob no es
///   base64 válido.
/// - [`VaultError::WrongPassphrase`] si el `master` es incorrecto (el tag AEAD
///   del desenvoltorio falla tras un FEC exitoso). **Reintentable; nunca borra.**
/// - [`VaultError::Crypto`] ante un fallo de derivación de clave.
pub fn open_envelope(
    vault: &CryptoVault,
    master: &str,
    salt_fec: &[u8],
    wrapped_dek_fec: &[u8],
) -> Result<Zeroizing<Vec<u8>>, VaultError> {
    let salt = fec_decode(salt_fec)?;
    let wrapped_bytes = fec_decode(wrapped_dek_fec)?;
    let wrapped = String::from_utf8(wrapped_bytes).map_err(|_| VaultError::VaultMetaCorrupt)?;
    let kek = vault.derive_key(master, &salt).map_err(map_crypto_err)?;
    vault
        .unwrap_key(&kek, &salt, &wrapped)
        .map_err(map_crypto_err)
}

/// Entrypoint del fuzzer (`fuzz_vault_meta_decode`, Task 9).
///
/// Parte `data` en `(salt_fec, wrapped_fec)` de forma determinista y
/// bounds-safe — primer `u16` LE = `len(salt_fec)` — y llama [`open_envelope`]
/// con un `CryptoVault::default()` y un master fijo, **descartando** el `Result`.
/// El invariante que verifica el fuzzer: *nunca panic ni borrado*, sea cual sea
/// `data`.
#[doc(hidden)]
pub fn fuzz_open_entrypoint(data: &[u8]) {
    const SPLIT_PREFIX: usize = 2;
    let Some(prefix) = data.get(0..SPLIT_PREFIX) else {
        return;
    };
    let Ok(len_arr) = <[u8; SPLIT_PREFIX]>::try_from(prefix) else {
        return;
    };
    let salt_len = u16::from_le_bytes(len_arr) as usize;
    let rest = data.get(SPLIT_PREFIX..).unwrap_or(&[]);
    let split = salt_len.min(rest.len());
    let (salt_fec, wrapped_fec) = rest.split_at(split);
    let vault = CryptoVault::default();
    let _ = open_envelope(&vault, "fuzz-master-key-fixed", salt_fec, wrapped_fec);
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_envelope, open_envelope};
    use crate::vault::VaultError;

    // master de test = string base64-like (UTF-8 válido, como el del keyring).
    const M: &str = "bWFzdGVyLWtleS0zMi1ieXRlcy1iYXNlNjQtc3RyaW5n";

    #[test]
    fn test_envelope_bootstrap_then_open_recovers_same_dek() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("open");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_open_with_wrong_master_yields_wrong_passphrase_not_corrupt() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");
        let err = open_envelope(
            &vault,
            "d3JvbmctbWFzdGVyLWtleS1zdHJpbmc",
            &salt_fec,
            &wrapped_fec,
        )
        .expect_err("wrong master must fail");
        assert!(matches!(err, VaultError::WrongPassphrase));
    }

    #[test]
    fn test_open_with_fec_uncorrectable_wrapped_dek_yields_corrupt() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");
        for b in wrapped_fec.iter_mut() {
            *b ^= 0xFF; // daño masivo, más allá de la FEC
        }
        let err = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect_err("corrupt must fail");
        assert!(matches!(err, VaultError::VaultMetaCorrupt));
    }

    #[test]
    fn test_single_bit_flip_in_wrapped_dek_is_corrected_by_fec() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        // Voltear un bit en el PAYLOAD FEC (tras el prefijo de 4 bytes de longitud).
        wrapped_fec[super::LEN_PREFIX] ^= 0x01;
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("bit-flip corregible");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_single_bit_flip_in_salt_is_corrected_by_fec() {
        let vault = cryptovault::CryptoVault::default();
        let (mut salt_fec, wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        salt_fec[super::LEN_PREFIX] ^= 0x01;
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("salt bit-flip");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_fuzz_entrypoint_never_panics_on_arbitrary_input() {
        // Entradas degeneradas: vacía, corta, aleatoria — jamás panic.
        for data in [
            &b""[..],
            &b"\x00"[..],
            &b"\x05\x00abcdefghij"[..],
            &[0xFFu8; 300],
        ] {
            super::fuzz_open_entrypoint(data);
        }
    }
}
