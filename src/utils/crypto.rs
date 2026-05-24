// Author: Julian Bolivar
// Version: 1.3.0
// Date: 2026-05-24

//! Self-contained cryptographic module — key derivation, authenticated
//! encryption, and forward error correction.
//!
//! ### Data Flow:
//! 1. **Key Derivation (Argon2):** A master password (from OS Keyring) is hashed with a
//!    per-record random 16-byte salt to produce a **32-byte key only**. The nonce is NOT
//!    derived from Argon2.
//! 2. **Nonce Sampling (OsRng):** A 12-byte AES-256-GCM-SIV nonce is sampled independently
//!    from `OsRng` and stored in the blob. This guarantees nonce independence across
//!    encryptions of the same plaintext under the same key (C5 fix).
//! 3. **Authenticated Encryption (AES-256-GCM-SIV):** Plaintext is encrypted using the
//!    derived key and the independently sampled nonce. This cipher is nonce-misuse resistant,
//!    guaranteeing confidentiality and integrity (authentication tag).
//! 4. **Error Correction (Reed-Solomon):** The salt, nonce, and ciphertext are encoded with
//!    parity bytes to allow recovery from bit-rot or minor storage corruption.
//! 5. **Final Blob:** `[u32 LE original-len][RS-encoded(salt || nonce || ciphertext)]`,
//!    base64-encoded for storage.

use std::fmt;

use aes_gcm_siv::aead::generic_array::GenericArray;
use aes_gcm_siv::aead::{Aead, KeyInit};
use aes_gcm_siv::Aes256GcmSiv;
use argon2::Argon2;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use rand::RngCore;
use zeroize::Zeroizing;

// ── Public constants ────────────────────────────────────────────────

pub const SALT_LEN: usize = 16;
pub const KEY_LEN: usize = 32;
pub const RS_DEFAULT_PARITY_LEN: usize = 32;
pub const RS_DEFAULT_DATA_LEN: usize = 223;

#[allow(dead_code)]
const RS_MAX_BLOCK_SIZE: usize = 255;

/// Absolute upper bound on a single plaintext record (50 MiB). Caps the
/// `original_len` field of a blob so a malformed/hostile length prefix can
/// never drive an arbitrary allocation during decryption (audit finding C7),
/// and bounds legitimate encryption payloads.
pub const MAX_PLAINTEXT_LEN: usize = 50 * 1024 * 1024;

// ── Argon2 cost parameters (OWASP 2025) ─────────────────────────────
//
// Explicit, audited Argon2id work factors. OWASP's 2025 minimum for
// interactive use: 64 MiB memory, 3 iterations, parallelism 4. Pinning these
// avoids relying on the argon2 crate's implicit `Default`, which can drift
// between crate versions and silently weaken (or strengthen) key derivation.
/// Argon2 memory cost in KiB (64 MiB).
pub const ARGON2_M_COST_KIB: u32 = 65536;
/// Argon2 time cost (number of iterations).
pub const ARGON2_T_COST: u32 = 3;
/// Argon2 degree of parallelism.
pub const ARGON2_P_COST: u32 = 4;

// ── CryptoError ─────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CryptoError {
    KeyDerivation(String),
    Cipher(String),
    ErrorCorrection(String),
    Encoding(String),
    InvalidInput(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyDerivation(msg) => write!(f, "Key derivation error: {}", msg),
            Self::Cipher(msg) => write!(f, "Cipher error: {}", msg),
            Self::ErrorCorrection(msg) => write!(f, "Error correction error: {}", msg),
            Self::Encoding(msg) => write!(f, "Encoding error: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CryptoError {}

// ── Traits ──────────────────────────────────────────────────────────

pub trait KeyDerivation: Send + Sync {
    fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        output_len: usize,
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError>;
}

pub trait AuthenticatedCipher: Send + Sync {
    fn encrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn decrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn nonce_len(&self) -> usize;
}

pub trait ErrorCorrection: Send + Sync {
    fn encode(&self, data: &[u8]) -> Vec<u8>;
    fn decode(&self, encoded: &[u8], original_len: usize) -> Result<Vec<u8>, CryptoError>;
}

// ── Argon2Kdf ───────────────────────────────────────────────────────

pub struct Argon2Kdf;

impl Argon2Kdf {
    /// Returns the audited OWASP 2025 Argon2 cost parameters used for all key
    /// derivation in this module.
    ///
    /// # Panics
    /// Never in practice: the constants are valid (`m`/`t`/`p` within argon2's
    /// accepted ranges), so `Params::new` cannot fail here.
    pub fn owasp_params() -> argon2::Params {
        argon2::Params::new(ARGON2_M_COST_KIB, ARGON2_T_COST, ARGON2_P_COST, None)
            .expect("OWASP Argon2 parameters are statically valid")
    }
}

impl KeyDerivation for Argon2Kdf {
    fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        output_len: usize,
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        let mut key = Zeroizing::new(vec![0u8; output_len]);
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            Self::owasp_params(),
        );
        argon2
            .hash_password_into(password, salt, &mut key)
            .map_err(|e| CryptoError::KeyDerivation(format!("Argon2 failed: {}", e)))?;
        Ok(key)
    }
}

// ── Aes256GcmSivCipher ──────────────────────────────────────────────

pub struct Aes256GcmSivCipher;

const AES_GCM_SIV_NONCE_LEN: usize = 12;

impl AuthenticatedCipher for Aes256GcmSivCipher {
    fn encrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let cipher = Aes256GcmSiv::new_from_slice(key)
            .map_err(|e| CryptoError::Cipher(format!("Cipher init failed: {}", e)))?;
        let nonce = GenericArray::from_slice(nonce);
        cipher
            .encrypt(nonce, data)
            .map_err(|e| CryptoError::Cipher(format!("Encryption failed: {}", e)))
    }

    fn decrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let cipher = Aes256GcmSiv::new_from_slice(key)
            .map_err(|e| CryptoError::Cipher(format!("Cipher init failed: {}", e)))?;
        let nonce = GenericArray::from_slice(nonce);
        cipher
            .decrypt(nonce, data)
            .map_err(|e| CryptoError::Cipher(format!("Decryption failed: {}", e)))
    }

    fn nonce_len(&self) -> usize {
        AES_GCM_SIV_NONCE_LEN
    }
}

// ── ReedSolomonCodec ────────────────────────────────────────────────

#[derive(Debug)]
pub struct ReedSolomonCodec {
    parity_len: usize,
    data_len: usize,
}

impl Default for ReedSolomonCodec {
    fn default() -> Self {
        Self {
            parity_len: RS_DEFAULT_PARITY_LEN,
            data_len: RS_DEFAULT_DATA_LEN,
        }
    }
}

impl ReedSolomonCodec {
    #[allow(dead_code)]
    pub fn new(parity_len: usize, data_len: usize) -> Result<Self, CryptoError> {
        if parity_len == 0 || data_len == 0 {
            return Err(CryptoError::InvalidInput(
                "Parity and data length must be greater than zero".to_string(),
            ));
        }
        if parity_len + data_len > 255 {
            return Err(CryptoError::InvalidInput(format!(
                "parity_len ({}) + data_len ({}) exceeds GF(2^8) limit of 255",
                parity_len, data_len
            )));
        }
        Ok(Self {
            parity_len,
            data_len,
        })
    }
}

impl ErrorCorrection for ReedSolomonCodec {
    fn encode(&self, data: &[u8]) -> Vec<u8> {
        let enc = reed_solomon::Encoder::new(self.parity_len);
        let mut result = Vec::new();
        for chunk in data.chunks(self.data_len) {
            let encoded = enc.encode(chunk);
            result.extend_from_slice(&encoded);
        }
        result
    }

    fn decode(&self, encoded: &[u8], original_len: usize) -> Result<Vec<u8>, CryptoError> {
        let dec = reed_solomon::Decoder::new(self.parity_len);
        let block_size = self.data_len + self.parity_len;
        let mut result = Vec::new();

        for chunk in encoded.chunks(block_size) {
            if chunk.len() <= self.parity_len {
                return Err(CryptoError::ErrorCorrection(
                    "Encoded block too short for Reed-Solomon parity".to_string(),
                ));
            }
            let recovered = dec.correct(chunk, None).map_err(|_| {
                CryptoError::ErrorCorrection("Reed-Solomon error correction failed".to_string())
            })?;
            result.extend_from_slice(recovered.data());
        }

        result.truncate(original_len);
        Ok(result)
    }
}

// ── CryptoVault ─────────────────────────────────────────────────────

pub struct CryptoVault {
    kdf: Box<dyn KeyDerivation>,
    cipher: Box<dyn AuthenticatedCipher>,
    fec: Box<dyn ErrorCorrection>,
}

impl Default for CryptoVault {
    fn default() -> Self {
        Self {
            kdf: Box::new(Argon2Kdf),
            cipher: Box::new(Aes256GcmSivCipher),
            fec: Box::new(ReedSolomonCodec::default()),
        }
    }
}

impl CryptoVault {
    #[allow(dead_code)]
    pub fn new(
        kdf: Box<dyn KeyDerivation>,
        cipher: Box<dyn AuthenticatedCipher>,
        fec: Box<dyn ErrorCorrection>,
    ) -> Self {
        Self { kdf, cipher, fec }
    }

    #[allow(dead_code)]
    pub fn encrypt(&self, password: &str, plaintext: &str) -> Result<String, CryptoError> {
        if password.is_empty() {
            return Err(CryptoError::InvalidInput(
                "Password must not be empty".to_string(),
            ));
        }

        let nonce_len = self.cipher.nonce_len();

        let projected_original_len = SALT_LEN + nonce_len + plaintext.len() + 16; // +16 = GCM-SIV tag
        if projected_original_len > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::InvalidInput(format!(
                "Record length {} (salt+nonce+ciphertext) exceeds MAX_PLAINTEXT_LEN ({})",
                projected_original_len, MAX_PLAINTEXT_LEN
            )));
        }

        let mut salt = [0u8; SALT_LEN];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        let mut nonce = vec![0u8; nonce_len];
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        let key = self.kdf.derive_key(password.as_bytes(), &salt, KEY_LEN)?;

        let ciphertext = self.cipher.encrypt(&key, &nonce, plaintext.as_bytes())?;

        let mut plaindata = Vec::with_capacity(SALT_LEN + nonce_len + ciphertext.len());
        plaindata.extend_from_slice(&salt);
        plaindata.extend_from_slice(&nonce);
        plaindata.extend_from_slice(&ciphertext);

        let rs_encoded = self.fec.encode(&plaindata);

        let original_len_u32 = u32::try_from(plaindata.len())
            .map_err(|_| CryptoError::Encoding("Data too large for length header".to_string()))?;
        let mut blob = Vec::with_capacity(4 + rs_encoded.len());
        blob.extend_from_slice(&original_len_u32.to_le_bytes());
        blob.extend_from_slice(&rs_encoded);

        Ok(STANDARD.encode(&blob))
    }

    #[allow(dead_code)]
    pub fn decrypt(&self, password: &str, encrypted_base64: &str) -> Result<String, CryptoError> {
        if password.is_empty() {
            return Err(CryptoError::InvalidInput(
                "Password must not be empty".to_string(),
            ));
        }

        let nonce_len = self.cipher.nonce_len();
        let blob = STANDARD
            .decode(encrypted_base64)
            .map_err(|e| CryptoError::Encoding(format!("Invalid base64: {}", e)))?;

        if blob.len() < 4 {
            return Err(CryptoError::Encoding(
                "Encrypted blob too short".to_string(),
            ));
        }

        let len_bytes: [u8; 4] = blob[..4].try_into().unwrap();
        let original_len = u32::from_le_bytes(len_bytes) as usize;

        if original_len > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::InvalidInput(format!(
                "Length header {} exceeds MAX_PLAINTEXT_LEN ({}); refusing to allocate",
                original_len, MAX_PLAINTEXT_LEN
            )));
        }

        if original_len > (blob.len() - 4) {
            return Err(CryptoError::InvalidInput(
                "Length header exceeds encoded data size".to_string(),
            ));
        }

        // C7 hardening: the RS decode allocates proportional to the encoded
        // blob, not `original_len`, so a small declared length with a huge body
        // would still drive a large allocation. RS(223/32) expands by at most
        // ~1.144x; reject anything grossly beyond 2x + slack before decoding.
        if blob.len().saturating_sub(4) > original_len.saturating_mul(2).saturating_add(4096) {
            return Err(CryptoError::InvalidInput(format!(
                "Encoded blob length {} is inconsistent with declared plaintext length {}; refusing to allocate",
                blob.len() - 4,
                original_len
            )));
        }

        let plaindata = self.fec.decode(&blob[4..], original_len)?;
        if plaindata.len() < SALT_LEN + nonce_len {
            return Err(CryptoError::InvalidInput(
                "Decoded blob too short for salt and nonce".to_string(),
            ));
        }
        let salt = &plaindata[..SALT_LEN];
        let nonce = &plaindata[SALT_LEN..SALT_LEN + nonce_len];
        let ciphertext = &plaindata[SALT_LEN + nonce_len..];

        let key = self.kdf.derive_key(password.as_bytes(), salt, KEY_LEN)?;

        let plaintext = self.cipher.decrypt(&key, nonce, ciphertext)?;

        String::from_utf8(plaintext)
            .map_err(|e| CryptoError::Encoding(format!("Invalid UTF-8: {}", e)))
    }

    /// Derives a 32-byte key from `password` and `salt` using the module's
    /// audited Argon2id parameters. Exposed so a caller can derive the key
    /// **once** and reuse it across many records (key caching), avoiding a
    /// per-record KDF.
    ///
    /// # Errors
    /// [`CryptoError::InvalidInput`] if `password` is empty; [`CryptoError::KeyDerivation`]
    /// if Argon2 fails.
    pub fn derive_key(
        &self,
        password: &str,
        salt: &[u8],
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        if password.is_empty() {
            return Err(CryptoError::InvalidInput(
                "Password must not be empty".to_string(),
            ));
        }
        self.kdf.derive_key(password.as_bytes(), salt, KEY_LEN)
    }

    /// Encrypts `plaintext` under a pre-derived `key`. A fresh random nonce is
    /// sampled per call; the blob layout is `[u32 LE len][RS(nonce || ciphertext)]`
    /// (no salt — the salt lives once per database, not per record).
    ///
    /// # Errors
    /// [`CryptoError::InvalidInput`] if the record exceeds [`MAX_PLAINTEXT_LEN`];
    /// [`CryptoError::Cipher`] on encryption failure (e.g. an invalid key length).
    pub fn encrypt_with_key(&self, key: &[u8], plaintext: &str) -> Result<String, CryptoError> {
        let nonce_len = self.cipher.nonce_len();

        let projected_original_len = nonce_len + plaintext.len() + 16; // +16 = GCM-SIV tag
        if projected_original_len > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::InvalidInput(format!(
                "Record length {} (nonce+ciphertext) exceeds MAX_PLAINTEXT_LEN ({})",
                projected_original_len, MAX_PLAINTEXT_LEN
            )));
        }

        let mut nonce = vec![0u8; nonce_len];
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        let ciphertext = self.cipher.encrypt(key, &nonce, plaintext.as_bytes())?;

        let mut plaindata = Vec::with_capacity(nonce_len + ciphertext.len());
        plaindata.extend_from_slice(&nonce);
        plaindata.extend_from_slice(&ciphertext);

        let rs_encoded = self.fec.encode(&plaindata);

        let original_len_u32 = u32::try_from(plaindata.len())
            .map_err(|_| CryptoError::Encoding("Data too large for length header".to_string()))?;
        let mut blob = Vec::with_capacity(4 + rs_encoded.len());
        blob.extend_from_slice(&original_len_u32.to_le_bytes());
        blob.extend_from_slice(&rs_encoded);

        Ok(STANDARD.encode(&blob))
    }

    /// Decrypts a blob produced by [`Self::encrypt_with_key`] under the same `key`.
    /// Preserves the C7 allocation guards (length cap + blob-size bound).
    ///
    /// # Errors
    /// [`CryptoError::InvalidInput`] on a hostile/malformed length prefix;
    /// [`CryptoError::Cipher`] if the key is wrong or authentication fails.
    pub fn decrypt_with_key(
        &self,
        key: &[u8],
        encrypted_base64: &str,
    ) -> Result<String, CryptoError> {
        let nonce_len = self.cipher.nonce_len();
        let blob = STANDARD
            .decode(encrypted_base64)
            .map_err(|e| CryptoError::Encoding(format!("Invalid base64: {}", e)))?;

        if blob.len() < 4 {
            return Err(CryptoError::Encoding(
                "Encrypted blob too short".to_string(),
            ));
        }

        let len_bytes: [u8; 4] = blob[..4].try_into().unwrap();
        let original_len = u32::from_le_bytes(len_bytes) as usize;

        if original_len > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::InvalidInput(format!(
                "Length header {} exceeds MAX_PLAINTEXT_LEN ({}); refusing to allocate",
                original_len, MAX_PLAINTEXT_LEN
            )));
        }

        if original_len > (blob.len() - 4) {
            return Err(CryptoError::InvalidInput(
                "Length header exceeds encoded data size".to_string(),
            ));
        }

        if blob.len().saturating_sub(4) > original_len.saturating_mul(2).saturating_add(4096) {
            return Err(CryptoError::InvalidInput(format!(
                "Encoded blob length {} is inconsistent with declared plaintext length {}; refusing to allocate",
                blob.len() - 4,
                original_len
            )));
        }

        let plaindata = self.fec.decode(&blob[4..], original_len)?;
        if plaindata.len() < nonce_len {
            return Err(CryptoError::InvalidInput(
                "Decoded blob too short for nonce".to_string(),
            ));
        }
        let nonce = &plaindata[..nonce_len];
        let ciphertext = &plaindata[nonce_len..];

        let plaintext = self.cipher.decrypt(key, nonce, ciphertext)?;

        String::from_utf8(plaintext)
            .map_err(|e| CryptoError::Encoding(format!("Invalid UTF-8: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Derives a deterministic 32-byte test key.
    fn k(vault: &CryptoVault) -> Zeroizing<Vec<u8>> {
        vault.derive_key("pw", &[0u8; SALT_LEN]).unwrap()
    }

    #[test]
    fn test_encrypt_with_key_roundtrips() {
        // S-1
        let vault = CryptoVault::default();
        let key = k(&vault);
        let pt = "sk-ant-secret-payload";
        let blob = vault.encrypt_with_key(&key, pt).unwrap();
        assert_eq!(vault.decrypt_with_key(&key, &blob).unwrap(), pt);
    }

    #[test]
    fn test_blob_layout_carries_no_salt() {
        // S-2: plaindata == nonce_len + (L + 16 tag); no 16-byte salt prefix.
        let vault = CryptoVault::default();
        let key = k(&vault);
        let pt = "0123456789"; // L = 10
        let blob = vault.encrypt_with_key(&key, pt).unwrap();
        let raw = STANDARD.decode(&blob).unwrap();
        let original_len = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
        let codec = ReedSolomonCodec::default();
        let plaindata = codec.decode(&raw[4..], original_len).unwrap();
        assert_eq!(plaindata.len(), 12 + (pt.len() + 16));
    }

    #[test]
    fn test_encrypt_with_key_uses_independent_nonce() {
        // S-3
        let vault = CryptoVault::default();
        let key = k(&vault);
        let pt = "identical plaintext";
        let a = vault.encrypt_with_key(&key, pt).unwrap();
        let b = vault.encrypt_with_key(&key, pt).unwrap();
        assert_ne!(a, b, "independent nonces must yield different blobs");
    }

    #[test]
    fn test_decrypt_with_wrong_key_errors_without_panic() {
        // S-4
        let vault = CryptoVault::default();
        let key_a = vault.derive_key("pw-a", &[1u8; SALT_LEN]).unwrap();
        let key_b = vault.derive_key("pw-b", &[1u8; SALT_LEN]).unwrap();
        let blob = vault.encrypt_with_key(&key_a, "secret").unwrap();
        assert!(matches!(
            vault.decrypt_with_key(&key_b, &blob),
            Err(CryptoError::Cipher(_))
        ));
    }

    #[test]
    fn test_decrypt_with_key_preserves_c7_caps() {
        // S-5: oversized length-prefix and grossly-large body both rejected pre-alloc.
        let vault = CryptoVault::default();
        let key = k(&vault);

        let mut over = Vec::new();
        over.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        over.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            vault.decrypt_with_key(&key, &STANDARD.encode(&over)),
            Err(CryptoError::InvalidInput(_))
        ));

        let mut big = Vec::new();
        big.extend_from_slice(&100u32.to_le_bytes());
        big.extend_from_slice(&vec![0u8; 20_000]);
        assert!(matches!(
            vault.decrypt_with_key(&key, &STANDARD.encode(&big)),
            Err(CryptoError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_invalid_key_length_errors_without_panic() {
        // S-9: a 5-byte key is not a valid AES-256 key.
        let vault = CryptoVault::default();
        assert!(matches!(
            vault.encrypt_with_key(&[0u8; 5], "x"),
            Err(CryptoError::Cipher(_))
        ));
        let valid = vault.encrypt_with_key(&k(&vault), "x").unwrap();
        assert!(matches!(
            vault.decrypt_with_key(&[0u8; 5], &valid),
            Err(CryptoError::Cipher(_))
        ));
    }

    #[test]
    fn test_decrypt_rejects_blob_grossly_larger_than_declared_length() {
        // Declares 100 bytes of plaintext but carries a 20 KB body — far beyond
        // any legitimate Reed-Solomon expansion (~1.14x). Must be rejected BEFORE
        // the RS decode allocates against the oversized body.
        let mut blob = Vec::new();
        blob.extend_from_slice(&100u32.to_le_bytes());
        blob.extend_from_slice(&vec![0u8; 20_000]);
        let encoded = STANDARD.encode(&blob);

        let vault = CryptoVault::default();
        assert!(
            matches!(
                vault.decrypt("pw", &encoded),
                Err(CryptoError::InvalidInput(_))
            ),
            "a blob far larger than its declared plaintext length must be rejected as InvalidInput"
        );
    }

    #[test]
    fn test_decrypt_rejects_oversized_length_prefix_without_alloc() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let encoded = STANDARD.encode(&blob);

        let vault = CryptoVault::default();
        let err = vault.decrypt("pw", &encoded).unwrap_err();
        match err {
            CryptoError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput for oversized prefix, got {other:?}"),
        }
    }

    #[test]
    fn test_decrypt_rejects_prefix_at_exactly_cap_plus_one() {
        let oversized = (MAX_PLAINTEXT_LEN + 1) as u32;
        let mut blob = Vec::new();
        blob.extend_from_slice(&oversized.to_le_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let encoded = STANDARD.encode(&blob);

        let vault = CryptoVault::default();
        assert!(
            matches!(
                vault.decrypt("pw", &encoded),
                Err(CryptoError::InvalidInput(_))
            ),
            "a length prefix one byte over the cap must be rejected"
        );
    }

    #[test]
    fn test_encrypt_rejects_plaintext_over_cap() {
        let vault = CryptoVault::default();
        let huge = "a".repeat(MAX_PLAINTEXT_LEN + 1);
        assert!(
            matches!(
                vault.encrypt("pw", &huge),
                Err(CryptoError::InvalidInput(_))
            ),
            "encrypting beyond MAX_PLAINTEXT_LEN must be rejected"
        );
    }

    fn extract_nonce_for_test(blob_base64: &str) -> Vec<u8> {
        let blob = STANDARD.decode(blob_base64).unwrap();
        let original_len = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let codec = ReedSolomonCodec::default();
        let plaindata = codec.decode(&blob[4..], original_len).unwrap();
        plaindata[SALT_LEN..SALT_LEN + 12].to_vec()
    }

    /// Spy KDF: records the output_len it was called with and delegates to Argon2Kdf.
    struct SpyKdf {
        inner: Argon2Kdf,
        recorded_output_len: std::sync::Mutex<Option<usize>>,
    }

    impl SpyKdf {
        fn new() -> Self {
            Self {
                inner: Argon2Kdf,
                recorded_output_len: std::sync::Mutex::new(None),
            }
        }
        fn get_output_len(&self) -> Option<usize> {
            *self.recorded_output_len.lock().unwrap()
        }
    }

    impl KeyDerivation for SpyKdf {
        fn derive_key(
            &self,
            password: &[u8],
            salt: &[u8],
            output_len: usize,
        ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
            *self.recorded_output_len.lock().unwrap() = Some(output_len);
            self.inner.derive_key(password, salt, output_len)
        }
    }

    #[test]
    fn test_blob_stores_independent_nonce_in_layout() {
        // Verify that encrypt calls derive_key with KEY_LEN only (not KEY_LEN + nonce_len).
        // Under the OLD layout this fails: output_len == 44 (KEY_LEN + nonce_len), not 32.

        // ArcKdf wraps SpyKdf in Arc so we can inspect the recorded output_len after
        // the vault has consumed ownership.
        let spy_arc = std::sync::Arc::new(SpyKdf::new());

        struct ArcKdf(std::sync::Arc<SpyKdf>);
        impl KeyDerivation for ArcKdf {
            fn derive_key(
                &self,
                password: &[u8],
                salt: &[u8],
                output_len: usize,
            ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
                self.0.derive_key(password, salt, output_len)
            }
        }

        let vault = CryptoVault::new(
            Box::new(ArcKdf(spy_arc.clone())),
            Box::new(Aes256GcmSivCipher),
            Box::new(ReedSolomonCodec::default()),
        );

        let password = "my-secure-password";
        let plaintext = "identical plaintext";
        let blob = vault.encrypt(password, plaintext).unwrap();

        // Under the new layout, derive_key must be called with exactly KEY_LEN (32) bytes.
        let recorded = spy_arc
            .get_output_len()
            .expect("derive_key was never called");
        assert_eq!(
            recorded, KEY_LEN,
            "encrypt must call derive_key with KEY_LEN={} only; got {} (old layout derives nonce from KDF too)",
            KEY_LEN, recorded
        );

        // The blob plaindata must contain SALT_LEN + nonce_len + ciphertext.
        // extract_nonce_for_test reads bytes at [SALT_LEN..SALT_LEN+12] — must be the stored nonce.
        let nonce = extract_nonce_for_test(&blob);
        assert_eq!(
            nonce.len(),
            12,
            "an independent 12-byte nonce must be stored in the blob"
        );

        // Round-trip must succeed with the new layout.
        assert_eq!(vault.decrypt(password, &blob).unwrap(), plaintext);

        // Two encryptions of the same plaintext must have different stored nonces.
        let blob_b = vault.encrypt(password, plaintext).unwrap();
        assert_ne!(
            extract_nonce_for_test(&blob),
            extract_nonce_for_test(&blob_b),
            "independent nonces should differ across encryptions (corroborating)"
        );
    }

    #[test]
    fn test_new_layout_roundtrips_with_embedded_nonce() {
        let vault = CryptoVault::default();
        let secret = "sk-ant-api03-real-key-here";
        let password = "my-secure-password";
        let encrypted = vault.encrypt(password, secret).unwrap();
        let decrypted = vault.decrypt(password, &encrypted).unwrap();
        assert_eq!(decrypted, secret);
    }

    #[test]
    fn vault_decrypt_roundtrip() {
        let vault = CryptoVault::default();
        let secret = "sk-ant-api03-real-key-here";
        let password = "my-secure-password";
        let encrypted = vault.encrypt(password, secret).unwrap();
        let decrypted = vault.decrypt(password, &encrypted).unwrap();
        assert_eq!(decrypted, secret);
    }

    #[test]
    fn rs_corrects_corrupted_data() {
        let rs = ReedSolomonCodec::default();
        let data = b"FEC correction test payload for Reed-Solomon codec.";
        let mut encoded = rs.encode(data);
        for i in 0..10 {
            encoded[i * 7] ^= 0xAA;
        }
        let decoded = rs.decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_argon2_uses_owasp_2025_parameters() {
        let params = Argon2Kdf::owasp_params();
        assert_eq!(
            params.m_cost(),
            65536,
            "memory cost must be 64 MiB (OWASP 2025)"
        );
        assert_eq!(params.t_cost(), 3, "time cost (iterations) must be 3");
        assert_eq!(params.p_cost(), 4, "parallelism must be 4");
    }

    #[test]
    fn test_derive_key_still_roundtrips_under_owasp_params() {
        let vault = CryptoVault::default();
        let enc = vault.encrypt("pw", "payload under owasp params").unwrap();
        assert_eq!(
            vault.decrypt("pw", &enc).unwrap(),
            "payload under owasp params"
        );
    }
}
