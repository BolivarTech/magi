// Author: Julian Bolivar
// Version: 1.2.0
// Date: 2026-05-23

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

impl KeyDerivation for Argon2Kdf {
    fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        output_len: usize,
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        let mut key = Zeroizing::new(vec![0u8; output_len]);
        Argon2::default()
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

    pub fn encrypt(&self, password: &str, plaintext: &str) -> Result<String, CryptoError> {
        if password.is_empty() {
            return Err(CryptoError::InvalidInput(
                "Password must not be empty".to_string(),
            ));
        }

        let nonce_len = self.cipher.nonce_len();

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

        if original_len > (blob.len() - 4) {
            return Err(CryptoError::InvalidInput(
                "Length header exceeds encoded data size".to_string(),
            ));
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
