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

use std::sync::{Arc, Mutex};

use cryptovault::fec::{ConcatenatedFec, ErrorCorrection};
use cryptovault::{CryptoError, CryptoVault};
use rusqlite::{Connection, OptionalExtension};
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
///
/// # Errors
///
/// [`VaultError::Crypto`] si `bytes.len()` no cabe en el prefijo `u32` (jamás
/// alcanzable para las entradas de `vault_meta`, ~48 B; se reporta como error
/// tipado en vez de truncar silenciosamente el prefijo — defensa en release).
fn fec_encode(bytes: &[u8]) -> Result<Vec<u8>, VaultError> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        VaultError::Crypto("vault_meta entry exceeds the u32 length prefix".to_string())
    })?;
    // Codificar primero: `ConcatenatedFec` expande ~2.3x, así que dimensionar por
    // `bytes.len()` forzaría un realloc. Con `encoded.len()` la capacidad es exacta.
    let encoded = ConcatenatedFec::default().encode(bytes);
    let mut out = Vec::with_capacity(LEN_PREFIX + encoded.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&encoded);
    Ok(out)
}

/// Recupera los bytes originales de un blob producido por [`fec_encode`].
///
/// # Errors
///
/// [`VaultError::VaultMetaCorrupt`] si el prefijo de longitud falta o el
/// `ConcatenatedFec::decode` no logra corregir la corrupción.
fn fec_decode(blob: &[u8]) -> Result<Vec<u8>, VaultError> {
    // `split_first_chunk` yields the 4-byte prefix as a `&[u8; LEN_PREFIX]` and the
    // remaining payload in one bounds-safe step — no fallible `try_into` (whose
    // error arm was unreachable once the length was already guaranteed).
    let (len_arr, payload) = blob
        .split_first_chunk::<LEN_PREFIX>()
        .ok_or(VaultError::VaultMetaCorrupt)?;
    let pre_len = u32::from_le_bytes(*len_arr) as usize;
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
/// `master` es la passphrase vigente del usuario **como `&str`** (UTF-8 válido
/// por construcción; nunca bytes crudos no-UTF8).
///
/// # Errors
///
/// [`VaultError::Crypto`] si la generación de material aleatorio, la derivación
/// de la KEK o el envoltorio de la DEK fallan.
pub fn bootstrap_envelope(vault: &CryptoVault, master: &str) -> Result<Bootstrapped, VaultError> {
    // Un fallo de RNG en el bootstrap NO es corrupción de `vault_meta` (aún no
    // existe metadata): se reporta como `Crypto`, no como `VaultMetaCorrupt`.
    let salt = cryptovault::generate_salt()
        .map_err(|e| VaultError::Crypto(format!("salt generation failed: {e}")))?;
    let dek = cryptovault::generate_dek()
        .map_err(|e| VaultError::Crypto(format!("DEK generation failed: {e}")))?;
    let kek = vault.derive_key(master, &salt).map_err(map_crypto_err)?;
    let wrapped = vault.wrap_key(&kek, &salt, &dek).map_err(map_crypto_err)?;
    let salt_fec = fec_encode(&salt)?;
    let wrapped_fec = fec_encode(wrapped.as_bytes())?;
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

/// Reads the FEC-encoded `salt` and `wrapped_dek` rows from `vault_meta`.
///
/// # Errors
/// [`VaultError::Storage`] on a SQL failure; [`VaultError::VaultMetaCorrupt`] if a
/// row is missing.
fn read_meta(guard: &Connection) -> Result<(Vec<u8>, Vec<u8>), VaultError> {
    let salt: Option<Vec<u8>> = guard
        .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
            r.get(0)
        })
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let wrapped: Option<Vec<u8>> = guard
        .query_row(
            "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    match (salt, wrapped) {
        (Some(s), Some(w)) => Ok((s, w)),
        _ => Err(VaultError::VaultMetaCorrupt),
    }
}

/// Re-keys the envelope: re-wraps the **same** DEK under a new passphrase (O(1) —
/// no record is re-encrypted). Implements `magi vault passwd` (REQ-V20).
///
/// Steps: (1) **verify `current`** by unwrapping the DEK — a wrong passphrase fails
/// the AEAD tag and returns [`VaultError::WrongPassphrase`], changing nothing (the
/// lock that stops `passwd` being a recovery path); (2) fresh salt → KEK_new →
/// re-wrap the SAME DEK; (3) write `{salt, wrapped_dek}` in ONE `BEGIN IMMEDIATE`
/// transaction, crash-safe (the DB stays openable with the old **or** new
/// passphrase, never bricked — SC-V35). Argon2 runs OUTSIDE the connection lock.
///
/// # Errors
/// [`VaultError::WrongPassphrase`] if `current` is wrong; [`VaultError::Storage`] on
/// a SQL failure or a detected concurrent re-wrap; [`VaultError::VaultMetaCorrupt`]
/// or [`VaultError::Crypto`] on corrupt metadata or a crypto failure.
pub fn rekey_envelope(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
) -> Result<(), VaultError> {
    rekey_envelope_inner(vault, conn, current, new, || {})
}

/// Shared body of [`rekey_envelope`] with an injectable hook that runs **between**
/// the initial `vault_meta` read and the write transaction — the exact TOCTOU
/// window the compare-and-abort guards. Production calls it with a no-op closure.
fn rekey_envelope_inner(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
    between_read_and_tx: impl FnOnce(),
) -> Result<(), VaultError> {
    // (1) Read current meta (short lock), then release before any Argon2.
    let (salt_fec, wrapped_fec) = {
        let guard = conn.lock().unwrap_or_else(|p| p.into_inner());
        read_meta(&guard)?
    };

    // (1b) Verify `current` and recover the DEK (Argon2 #1, off-lock). Wrong
    // passphrase ⇒ WrongPassphrase, nothing written.
    let dek = open_envelope(vault, current, &salt_fec, &wrapped_fec)?;

    between_read_and_tx();

    // (2) Re-wrap the SAME DEK under a fresh salt/KEK (Argon2 #2, off-lock).
    let new_salt = cryptovault::generate_salt().map_err(map_crypto_err)?;
    let kek_new = vault.derive_key(new, &new_salt).map_err(map_crypto_err)?;
    let wrapped_new = vault
        .wrap_key(&kek_new, &new_salt, &dek)
        .map_err(map_crypto_err)?;
    let new_salt_fec = fec_encode(&new_salt)?;
    let new_wrapped_fec = fec_encode(wrapped_new.as_bytes())?;

    // (3) Atomic write with a compare-and-abort against a concurrent re-wrap.
    let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
    let tx = guard
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let (_, current_wrapped) = read_meta(&tx)?;
    if current_wrapped != wrapped_fec {
        // Another process re-wrapped between our read and this transaction; abort
        // rather than clobber its envelope.
        return Err(VaultError::Storage(
            "concurrent rekey detected; aborted".to_string(),
        ));
    }
    tx.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('salt', ?1)",
        [&new_salt_fec],
    )
    .map_err(|e| VaultError::Storage(e.to_string()))?;
    tx.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
        [&new_wrapped_fec],
    )
    .map_err(|e| VaultError::Storage(e.to_string()))?;
    tx.commit()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    Ok(())
}

/// Test-only wrapper exposing the TOCTOU hook of [`rekey_envelope_inner`].
#[cfg(test)]
pub(crate) fn rekey_envelope_with_hook(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
    between_read_and_tx: impl FnOnce(),
) -> Result<(), VaultError> {
    rekey_envelope_inner(vault, conn, current, new, between_read_and_tx)
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
    use super::{bootstrap_envelope, open_envelope, rekey_envelope, rekey_envelope_with_hook};
    use crate::vault::VaultError;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    /// Builds an in-memory DB with a `vault_meta` table bootstrapped under `master`.
    /// Returns the shared connection.
    fn meta_conn(master: &str) -> Arc<Mutex<Connection>> {
        let conn = Connection::open_in_memory().expect("mem");
        conn.execute(
            "CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)",
            [],
        )
        .expect("ddl");
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) = bootstrap_envelope(&vault, master).expect("boot");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("s");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("w");
        Arc::new(Mutex::new(conn))
    }

    /// Reads back `(salt_fec, wrapped_fec)` from `vault_meta`.
    fn read_back(conn: &Arc<Mutex<Connection>>) -> (Vec<u8>, Vec<u8>) {
        let g = conn.lock().expect("lock");
        let s: Vec<u8> = g
            .query_row("SELECT value FROM vault_meta WHERE key='salt'", [], |r| {
                r.get(0)
            })
            .expect("s");
        let w: Vec<u8> = g
            .query_row(
                "SELECT value FROM vault_meta WHERE key='wrapped_dek'",
                [],
                |r| r.get(0),
            )
            .expect("w");
        (s, w)
    }

    #[test]
    fn test_rekey_opens_with_new_passphrase_and_same_dek_not_with_old() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        // Capture the DEK before, to prove it is unchanged.
        let (s0, w0) = read_back(&conn);
        let dek0 = open_envelope(&vault, "old-passphrase-long-enough", &s0, &w0).expect("open");

        rekey_envelope(
            &vault,
            &conn,
            "old-passphrase-long-enough",
            "new-passphrase-long-enough",
        )
        .expect("rekey");

        let (s1, w1) = read_back(&conn);
        let dek1 =
            open_envelope(&vault, "new-passphrase-long-enough", &s1, &w1).expect("new opens");
        assert_eq!(dek0.to_vec(), dek1.to_vec()); // SAME DEK (nothing re-encrypted)
        assert!(matches!(
            open_envelope(&vault, "old-passphrase-long-enough", &s1, &w1),
            Err(VaultError::WrongPassphrase)
        ));
    }

    #[test]
    fn test_rekey_with_wrong_current_changes_nothing() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        let before = read_back(&conn);
        let e =
            rekey_envelope(&vault, &conn, "WRONG", "new-passphrase-long-enough").expect_err("lock");
        assert!(matches!(e, VaultError::WrongPassphrase));
        assert_eq!(read_back(&conn), before); // untouched
    }

    #[test]
    fn test_rekey_detects_concurrent_rewrap_and_aborts_without_writing() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        let conn2 = conn.clone();
        // The hook runs between our read and our tx: a competitor re-keys first.
        let err = rekey_envelope_with_hook(
            &vault,
            &conn,
            "old-passphrase-long-enough",
            "our-new-passphrase-xyz",
            move || {
                let v = cryptovault::CryptoVault::default();
                rekey_envelope(
                    &v,
                    &conn2,
                    "old-passphrase-long-enough",
                    "winner-passphrase-xyz",
                )
                .expect("competitor rekey");
            },
        )
        .expect_err("must detect the concurrent rewrap");
        assert!(matches!(err, VaultError::Storage(_)));
        // The DB is openable with the WINNER's passphrase (never bricked — SC-V35).
        let (s, w) = read_back(&conn);
        open_envelope(&vault, "winner-passphrase-xyz", &s, &w).expect("winner opens");
    }

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
    fn test_bit_flip_in_length_prefix_fails_safe_as_corrupt() {
        // The 4-byte length prefix sits OUTSIDE the FEC-protected region
        // (see `fec_encode`), so corruption there is not self-corrected — but it
        // MUST fail safe: a typed `VaultMetaCorrupt`, never a panic and never a
        // wrong DEK (REQ-V35). Documents the intentionally-uncorrected window.
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");

        // Multi-bit flip: invert the whole 4-byte prefix so `pre_len` is enormous.
        let mut multi = wrapped_fec.clone();
        for b in multi.iter_mut().take(super::LEN_PREFIX) {
            *b ^= 0xFF;
        }
        let err = open_envelope(&vault, M, &salt_fec, &multi).expect_err("prefix corruption fails");
        assert!(matches!(err, VaultError::VaultMetaCorrupt));

        // Single-bit flip in the most-significant prefix byte (LE) — a large
        // `pre_len` the FEC block-count check rejects.
        let mut single = wrapped_fec.clone();
        single[super::LEN_PREFIX - 1] ^= 0x80;
        let err2 = open_envelope(&vault, M, &salt_fec, &single).expect_err("prefix bit-flip fails");
        assert!(matches!(err2, VaultError::VaultMetaCorrupt));
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
