// Author: Julian Bolivar
// Version: 1.4.1
// Date: 2026-05-24

//! Self-contained cryptographic vault — key derivation, authenticated
//! encryption, and forward error correction, composed over three injected
//! strategy traits.
//!
//! ## Architecture
//!
//! [`CryptoVault`] is the public facade. It wires three orthogonal,
//! replaceable strategies:
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                         CryptoVault                              |
//! |  +--------------------+  +-------------------+  +-------------+ |
//! |  | kdf:               |  | cipher:           |  | fec:        | |
//! |  | KeyDerivation      |  | AuthenticatedCipher|  | Error-      | |
//! |  |                    |  |                   |  | Correction  | |
//! |  |  [Argon2Kdf]       |  | [Aes256GcmSiv-   |  | [ReedSolomon| |
//! |  |   Argon2id         |  |  Cipher]          |  |  Codec]     | |
//! |  |   OWASP 2025       |  |  nonce-misuse     |  |  RS(255,223)| |
//! |  |   m=64MiB, t=3,p=4 |  |  resistant        |  |  32 parity  | |
//! |  |   -> 32-byte key   |  |  AES-256, 12B     |  |  bytes/block| |
//! |  |                    |  |  nonce, 128b tag  |  |             | |
//! |  +--------------------+  +-------------------+  +-------------+ |
//! +------------------------------------------------------------------+
//! ```
//!
//! [`CryptoVault::default`] wires [`Argon2Kdf`], [`Aes256GcmSivCipher`],
//! and [`ReedSolomonCodec`]. Custom combinations can be injected via
//! [`CryptoVault::new`] for testing or future algorithm rotation.
//!
//! ---
//!
//! ## Blob layout
//!
//! [`CryptoVault::encrypt_with_key`] produces a **base64-encoded** string.
//! Decoded, the binary structure is:
//!
//! ```text
//! Offset  Size   Field
//! ------  -----  -------------------------------------------------------
//!   0      1 B   version  (BLOB_VERSION = 1; validated first on decrypt)
//!   1      4 B   original_len  (u32, little-endian; capped at 50 MiB)
//!   5     var.   RS-body: ReedSolomon( nonce(12 B) || ct || tag(16 B) )
//! ------  -----  -------------------------------------------------------
//!
//! RS-body (before encoding):
//!   +------------------+-------------------------------+----------+
//!   | nonce   12 bytes | AES-256-GCM-SIV ciphertext   | tag 16 B |
//!   | (OsRng, per      | (same length as plaintext)    |          |
//!   |  record)         |                               |          |
//!   +------------------+-------------------------------+----------+
//! ```
//!
//! The RS encoder splits the body into chunks of at most
//! [`RS_DEFAULT_DATA_LEN`] (223) bytes and appends [`RS_DEFAULT_PARITY_LEN`]
//! (32) parity bytes per chunk, yielding 255-byte RS blocks. `original_len`
//! records the pre-RS byte count so the decoder can strip chunk-padding.
//!
//! ### Field breakdown
//!
//! | Field | Offset | Size | Description |
//! |-------|--------|------|-------------|
//! | `version` | 0 | 1 B | `BLOB_VERSION = 1`. Checked first on decrypt; an unknown value is rejected before any other field is read. |
//! | `original_len` | 1 | 4 B, u32 LE | Byte length of the pre-RS body (`nonce \|\| ct \|\| tag`). Capped at [`MAX_PLAINTEXT_LEN`] (50 MiB) before any allocation (C7 guard). |
//! | RS body | 5 | variable | [`ReedSolomonCodec`] encoding of `nonce (12 B) \|\| ciphertext \|\| tag (16 B)`. Each 223-byte data chunk becomes a 255-byte RS block (+32 parity bytes). |
//!
//! **The per-database salt is NOT in this blob.** It lives once per database
//! in the `vault_meta` table. The store layer derives the session key exactly
//! once at startup and caches it in a `Zeroizing`-wrapped buffer for all
//! records in the session.
//!
//! ---
//!
//! ## Security properties
//!
//! | Property | Mechanism | Details |
//! |----------|-----------|---------|
//! | **Confidentiality** | AES-256-GCM-SIV | 256-bit key; IND-CCA2 secure |
//! | **Integrity / authenticity** | AES-GCM-SIV 128-bit auth tag | Forgery probability <= 2^-128 per record |
//! | **Nonce-misuse resistance** | GCM-SIV construction (RFC 8452) | A nonce collision leaks plaintext equality only; the key is not compromised |
//! | **Independent nonces** | 12-byte nonce from `OsRng` per record | Each record's nonce is sampled independently from the OS CSPRNG; never KDF-derived |
//! | **Brute-force resistance** | Argon2id, OWASP 2025 params | `m=64 MiB`, `t=3`, `p=4`; memory-hard; resists GPU/ASIC attacks on the master password |
//! | **One KDF call per session** | Per-DB salt -> key cached in `Zeroizing` | O(1) Argon2 cost regardless of history size (B-prime design) |
//! | **Error resilience** | RS(255, 223) | Corrects up to 16 corrupted bytes per 255-byte block; survives typical bit-rot |
//! | **Allocation-DoS guards (C7)** | Version check -> length cap -> body-size consistency | A hostile blob cannot drive unbounded allocation before the RS decoder runs |
//! | **Format portability** | Leading version byte `BLOB_VERSION` | Future layout changes produce a detectable version mismatch, not silent mis-parsing |
//!
//! ---
//!
//! ## Walkthrough
//!
//! ### Encryption (`encrypt_with_key`)
//!
//! 1. **Bounds check** — `nonce_len + plaintext.len() + 16` vs.
//!    [`MAX_PLAINTEXT_LEN`] (50 MiB). Exceeding the cap returns
//!    [`CryptoError::InvalidInput`] before any allocation.
//! 2. **Sample nonce** — a 12-byte nonce is drawn from `OsRng` on every call.
//!    Nonces are never derived from the KDF; each record's nonce is independent.
//! 3. **AES-256-GCM-SIV encrypt** — produces `ciphertext || tag (16 B)` from
//!    `key`, `nonce`, and the plaintext bytes.
//! 4. **Concatenate** — `body = nonce (12 B) || ciphertext || tag (16 B)`.
//! 5. **Reed-Solomon encode** — `body` is split into <=223-byte chunks; each
//!    chunk is extended to 255 bytes by appending 32 parity bytes.
//! 6. **Assemble blob** —
//!    `[BLOB_VERSION (1 B)] ++ [body_len as u32 LE (4 B)] ++ [RS-encoded body]`.
//! 7. **Base64 encode** — the binary blob is encoded with the standard alphabet.
//!
//! ### Decryption (`decrypt_with_key`)
//!
//! 1. **Base64 decode** — malformed base64 -> [`CryptoError::Encoding`].
//! 2. **Minimum length** — blob must be >= 5 bytes (header only); shorter -> error.
//! 3. **Version check (C7-a)** — `blob[0]` must equal `BLOB_VERSION = 1`.
//!    Rejected before reading any other field.
//! 4. **Length cap (C7-b)** — `original_len` (bytes 1-4, little-endian) must be
//!    <= [`MAX_PLAINTEXT_LEN`]; rejected before allocating the decoded buffer.
//! 5. **Body-size consistency (C7-c)** — the RS body length (bytes after offset 5)
//!    must be <= `original_len * 2 + 4096`. RS(255,223) expands by ~1.144x; a
//!    grossly larger body indicates tampering and is rejected before the RS
//!    decoder allocates working memory.
//! 6. **Reed-Solomon decode** — error-corrects up to 16 bytes per block; recovers
//!    `nonce || ciphertext || tag`.
//! 7. **AES-256-GCM-SIV decrypt** — verifies the authentication tag, then
//!    decrypts. Tampered data or wrong key -> [`CryptoError::Cipher`].
//! 8. **UTF-8** — plaintext bytes are decoded; invalid sequences ->
//!    [`CryptoError::Encoding`].
//!
//! ---
//!
//! ## Example
//!
//! ```ignore
//! // Derive the session key once (salt comes from vault_meta in production).
//! let salt = [0u8; SALT_LEN];
//! let vault = CryptoVault::default();
//! let key = vault.derive_key("master-password", &salt).expect("derive key");
//!
//! // Encrypt individual records under the cached key.
//! let blob = vault.encrypt_with_key(&key, "sk-ant-secret").expect("encrypt");
//!
//! // Decrypt when reading back from the store.
//! let plain = vault.decrypt_with_key(&key, &blob).expect("decrypt");
//! assert_eq!(plain, "sk-ant-secret");
//!
//! // Equal plaintexts produce different blobs (independent OsRng nonces).
//! let blob2 = vault.encrypt_with_key(&key, "sk-ant-secret").expect("encrypt");
//! assert_ne!(blob, blob2);
//! ```
//!
//! ## Data flow
//!
//! 1. **Key Derivation (Argon2):** A master password (from OS Keyring) is hashed with a
//!    per-database 16-byte salt (derived once and cached by the store) to produce a
//!    **32-byte key only**. The nonce is NOT derived from Argon2.
//! 2. **Nonce Sampling (OsRng):** A 12-byte AES-256-GCM-SIV nonce is sampled independently
//!    from `OsRng` and stored in the blob. This guarantees nonce independence across
//!    encryptions of the same plaintext under the same key (C5 fix).
//! 3. **Authenticated Encryption (AES-256-GCM-SIV):** Plaintext is encrypted using the
//!    derived key and the independently sampled nonce. This cipher is nonce-misuse resistant,
//!    guaranteeing confidentiality and integrity (authentication tag).
//! 4. **Error Correction (Reed-Solomon):** The nonce and ciphertext are encoded with
//!    parity bytes to allow recovery from bit-rot or minor storage corruption.
//! 5. **Final Blob:** `[u8 version][u32 LE original-len][RS-encoded(nonce || ciphertext)]`,
//!    base64-encoded for storage. The leading version byte makes the format
//!    self-describing: an unknown version is **detected and rejected**, not silently
//!    mis-parsed (version-dispatch/migration is not yet implemented). The
//!    salt is stored once per database by the store layer, not per record.

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

/// Length in bytes of the per-database Argon2 salt stored in `vault_meta`.
///
/// The salt is generated once when the database is created and persisted
/// by the store layer (`EncryptedSqliteMemory`). It is passed to
/// [`CryptoVault::derive_key`] exactly once per session; the derived key is
/// then cached for all subsequent record reads and writes.
pub const SALT_LEN: usize = 16;

/// Length in bytes of the AES-256 key produced by [`CryptoVault::derive_key`].
///
/// [`Argon2Kdf`] fills exactly `KEY_LEN` (32) bytes; this satisfies the
/// AES-256-GCM-SIV key-length requirement of [`Aes256GcmSivCipher`].
pub const KEY_LEN: usize = 32;

/// Number of Reed-Solomon parity bytes appended to each data chunk by the
/// default [`ReedSolomonCodec`].
///
/// Together with [`RS_DEFAULT_DATA_LEN`], this gives an RS(255, 223) code over
/// GF(2^8): 223 data bytes + 32 parity bytes = 255-byte codeword. The codec
/// can **correct up to 16 corrupted bytes** per codeword (`floor(32/2) = 16`).
pub const RS_DEFAULT_PARITY_LEN: usize = 32;

/// Number of data bytes per Reed-Solomon block in the default [`ReedSolomonCodec`].
///
/// The RS encoder splits input into chunks of at most `RS_DEFAULT_DATA_LEN`
/// bytes; each chunk is extended to a 255-byte codeword by appending
/// [`RS_DEFAULT_PARITY_LEN`] (32) parity bytes. This is the same (255, 223)
/// code used in deep-space communication (CCSDS standard).
pub const RS_DEFAULT_DATA_LEN: usize = 223;

#[allow(dead_code)]
const RS_MAX_BLOCK_SIZE: usize = 255;

/// Absolute upper bound on a single plaintext record (50 MiB). Caps the
/// `original_len` field of a blob so a malformed/hostile length prefix can
/// never drive an arbitrary allocation during decryption (audit finding C7),
/// and bounds legitimate encryption payloads.
pub const MAX_PLAINTEXT_LEN: usize = 50 * 1024 * 1024;

/// On-disk blob format version, prepended as the first byte of every blob so the
/// format is self-describing and a future layout change is distinguishable from
/// corruption (#13). Bump on any incompatible blob-layout change.
const BLOB_VERSION: u8 = 1;

// ── Argon2 cost parameters (OWASP 2025) ─────────────────────────────
//
// Explicit, audited Argon2id work factors. OWASP's 2025 minimum for
// interactive use: 64 MiB memory, 3 iterations, parallelism 4. Pinning these
// avoids relying on the argon2 crate's implicit `Default`, which can drift
// between crate versions and silently weaken (or strengthen) key derivation.

/// Argon2 memory cost in KiB (64 MiB).
///
/// This is the `m` parameter in the OWASP 2025 minimum recommendation for
/// interactive Argon2id: `m = 65536 KiB` (64 MiB), `t = 3`, `p = 4`.
/// See also [`ARGON2_T_COST`] and [`ARGON2_P_COST`].
pub const ARGON2_M_COST_KIB: u32 = 65536;

/// Argon2 time cost (number of iterations).
///
/// `t = 3` as per the OWASP 2025 minimum recommendation. Together with
/// [`ARGON2_M_COST_KIB`] and [`ARGON2_P_COST`] these are passed to
/// [`Argon2Kdf::owasp_params`] and used by every [`CryptoVault::derive_key`]
/// call.
pub const ARGON2_T_COST: u32 = 3;

/// Argon2 degree of parallelism.
///
/// `p = 4` as per the OWASP 2025 minimum recommendation. Together with
/// [`ARGON2_M_COST_KIB`] and [`ARGON2_T_COST`] these are passed to
/// [`Argon2Kdf::owasp_params`].
pub const ARGON2_P_COST: u32 = 4;

// ── CryptoError ─────────────────────────────────────────────────────

/// Typed error returned by all cryptographic operations in this module.
///
/// Every variant carries a human-readable message string suitable for logging.
/// Match on the variant for programmatic handling; use the
/// [`std::fmt::Display`] impl for user-visible messages.
///
/// # Error hierarchy
///
/// | Variant | Raised by |
/// |---------|-----------|
/// | [`CryptoError::KeyDerivation`] | [`KeyDerivation::derive_key`] / [`CryptoVault::derive_key`] |
/// | [`CryptoError::Cipher`] | [`AuthenticatedCipher::encrypt`] / [`AuthenticatedCipher::decrypt`] |
/// | [`CryptoError::ErrorCorrection`] | [`ErrorCorrection::decode`] |
/// | [`CryptoError::Encoding`] | base64 decode, UTF-8 decode |
/// | [`CryptoError::InvalidInput`] | precondition violations (length cap, version, empty password) |
#[derive(Debug)]
pub enum CryptoError {
    /// Key derivation failed.
    ///
    /// Possible causes: Argon2 rejected the parameters (e.g. salt too short)
    /// or the output length is zero. Inspect the message string for details.
    KeyDerivation(String),

    /// Authenticated cipher operation failed.
    ///
    /// On encrypt: the key length is not 32 bytes (AES-256 requirement).
    /// On decrypt: authentication tag verification failed — indicates a wrong
    /// key, a tampered ciphertext, or a mismatched nonce. The plaintext is
    /// **not** returned when authentication fails.
    Cipher(String),

    /// Reed-Solomon error correction failed.
    ///
    /// The encoded block contains more corrupted bytes than the codec can
    /// recover (more than `floor(parity_len / 2)` errors per block with the
    /// default RS(255, 223) codec: more than 16 corrupted bytes per
    /// 255-byte block).
    ErrorCorrection(String),

    /// Base64 decode or UTF-8 conversion failed.
    ///
    /// On decrypt input: the base64 string is malformed.
    /// On decrypt output: the recovered plaintext bytes are not valid UTF-8
    /// (should not occur for well-formed records written by
    /// [`CryptoVault::encrypt_with_key`], which accepts `&str`).
    Encoding(String),

    /// A precondition was violated before any cryptographic operation started.
    ///
    /// Covers: empty password passed to [`CryptoVault::derive_key`]; plaintext
    /// or projected ciphertext length exceeding [`MAX_PLAINTEXT_LEN`] (C7
    /// guard); blob too short; unsupported blob version byte; `original_len`
    /// field exceeding [`MAX_PLAINTEXT_LEN`]; RS-body size grossly inconsistent
    /// with the declared length (C7 guard). No allocation occurs for hostile
    /// blobs that trip these guards.
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

/// Strategy trait for password-based key derivation functions (KDFs).
///
/// An implementation derives a fixed-length cryptographic key from a password
/// (or passphrase) and a random salt. The only production implementation is
/// [`Argon2Kdf`]; alternate implementations can be injected via
/// [`CryptoVault::new`] for testing.
///
/// ## Contract
///
/// - The derived key must be deterministic: the same `password`, `salt`, and
///   `output_len` must always produce the same key bytes.
/// - The key must be computationally infeasible to reverse to the password
///   without the salt (one-wayness).
/// - Implementations must be `Send + Sync` so a [`CryptoVault`] can be shared
///   across threads.
pub trait KeyDerivation: Send + Sync {
    /// Derives a cryptographic key of `output_len` bytes from `password` and
    /// `salt`.
    ///
    /// # Parameters
    ///
    /// - `password` — the master passphrase bytes. For [`Argon2Kdf`] this
    ///   should be non-empty; the vault enforces the non-empty invariant at the
    ///   [`CryptoVault::derive_key`] boundary.
    /// - `salt` — a random, per-database salt. For [`Argon2Kdf`] this must be
    ///   exactly [`SALT_LEN`] (16) bytes.
    /// - `output_len` — the number of key bytes to produce. Pass [`KEY_LEN`]
    ///   (32) for AES-256.
    ///
    /// # Returns
    ///
    /// A `Zeroizing`-wrapped `Vec<u8>` of length `output_len`. The buffer is
    /// wiped from memory when the `Zeroizing` wrapper is dropped.
    ///
    /// # Errors
    ///
    /// [`CryptoError::KeyDerivation`] if the underlying KDF rejects the
    /// parameters (e.g. salt too short, `output_len` zero).
    fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        output_len: usize,
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError>;
}

/// Strategy trait for authenticated encryption schemes.
///
/// An `AuthenticatedCipher` provides both **confidentiality** (via encryption)
/// and **integrity** (via an authentication tag). The only production
/// implementation is [`Aes256GcmSivCipher`].
///
/// ## Contract
///
/// - `encrypt` followed by `decrypt` with the same `key` and `nonce` must
///   return the original `data`.
/// - `decrypt` must return an error (not silently produce wrong output) if
///   `data` was tampered with or if the wrong `key` or `nonce` is supplied.
/// - Implementations must be `Send + Sync`.
pub trait AuthenticatedCipher: Send + Sync {
    /// Encrypts `data` under `key` with the given `nonce`, returning
    /// `ciphertext || authentication_tag`.
    ///
    /// # Parameters
    ///
    /// - `key` — the symmetric key. For [`Aes256GcmSivCipher`] this must be
    ///   exactly 32 bytes (AES-256).
    /// - `nonce` — the nonce. For [`Aes256GcmSivCipher`] this must be exactly
    ///   12 bytes. Must be unique per `(key, plaintext)` pair in security-
    ///   critical use; GCM-SIV tolerates accidental reuse without key
    ///   compromise.
    /// - `data` — the plaintext bytes to encrypt.
    ///
    /// # Returns
    ///
    /// `ciphertext || authentication_tag` as a single `Vec<u8>`. For
    /// AES-256-GCM-SIV the tag is 16 bytes appended after the ciphertext.
    ///
    /// # Errors
    ///
    /// [`CryptoError::Cipher`] if the key or nonce length is wrong.
    fn encrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Decrypts and authenticates `data` (ciphertext || tag) under `key` and
    /// `nonce`, returning the original plaintext on success.
    ///
    /// # Parameters
    ///
    /// - `key` — the same key used during encryption.
    /// - `nonce` — the same nonce used during encryption (stored in the blob).
    /// - `data` — the `ciphertext || authentication_tag` bytes.
    ///
    /// # Returns
    ///
    /// The original plaintext bytes.
    ///
    /// # Errors
    ///
    /// [`CryptoError::Cipher`] if the authentication tag does not match (wrong
    /// key, tampered ciphertext, or mismatched nonce). The plaintext is never
    /// returned on authentication failure.
    fn decrypt(&self, key: &[u8], nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Returns the required nonce length in bytes for this cipher.
    ///
    /// For [`Aes256GcmSivCipher`] this is `12`. The vault samples exactly this
    /// many bytes from `OsRng` per record and prepends them to the RS body.
    fn nonce_len(&self) -> usize;
}

/// Strategy trait for forward error correction (FEC) codecs.
///
/// A FEC codec adds redundant parity bytes to a payload so that bounded
/// corruption can be detected and repaired on decode. The only production
/// implementation is [`ReedSolomonCodec`].
///
/// ## Contract
///
/// - `decode(encode(data), data.len())` must return `data` when the encoded
///   bytes have been corrupted within the codec's correction capacity.
/// - Implementations must be `Send + Sync`.
pub trait ErrorCorrection: Send + Sync {
    /// Encodes `data` into a byte sequence that includes parity information.
    ///
    /// The data is split into fixed-size chunks; each chunk is encoded
    /// independently. The returned bytes are suitable for persistent storage
    /// and can survive bounded per-block corruption.
    ///
    /// # Parameters
    ///
    /// - `data` — the raw bytes to protect (the pre-RS body: nonce || ct ||
    ///   tag in the vault's use case).
    ///
    /// # Returns
    ///
    /// The RS-encoded bytes. For [`ReedSolomonCodec`] each 223-byte data chunk
    /// becomes a 255-byte block; the total length is
    /// `ceil(data.len() / 223) * 255`.
    fn encode(&self, data: &[u8]) -> Vec<u8>;

    /// Decodes and error-corrects an encoded byte sequence, returning the
    /// original `original_len` bytes.
    ///
    /// # Parameters
    ///
    /// - `encoded` — the full encoded byte sequence as produced by
    ///   [`Self::encode`].
    /// - `original_len` — the exact byte length of the original (pre-encode)
    ///   data. Used to strip the zero-padding introduced by the last chunk.
    ///
    /// # Errors
    ///
    /// [`CryptoError::ErrorCorrection`] if any block contains more corrupted
    /// bytes than the codec can correct. For [`ReedSolomonCodec`] the limit is
    /// 16 corrupted bytes per 255-byte block.
    fn decode(&self, encoded: &[u8], original_len: usize) -> Result<Vec<u8>, CryptoError>;
}

// ── Argon2Kdf ───────────────────────────────────────────────────────

/// Argon2id key-derivation function (KDF) implementation.
///
/// Derives a 32-byte key from a password and a per-database salt using
/// Argon2id at the OWASP 2025 minimum parameters for interactive use:
///
/// | Parameter | Constant | Value |
/// |-----------|----------|-------|
/// | Memory (`m`) | [`ARGON2_M_COST_KIB`] | 65 536 KiB (64 MiB) |
/// | Iterations (`t`) | [`ARGON2_T_COST`] | 3 |
/// | Parallelism (`p`) | [`ARGON2_P_COST`] | 4 |
/// | Algorithm | — | Argon2id (hybrid of Argon2i + Argon2d) |
/// | Version | — | v1.3 (0x13) |
///
/// The parameters are pinned via the constants above (see
/// [`Argon2Kdf::owasp_params`]) rather than the `argon2` crate's
/// `Default`, which can drift between crate versions and silently change
/// key-derivation cost.
///
/// In production the store layer calls [`CryptoVault::derive_key`] exactly
/// once per session and caches the 32-byte key in a `Zeroizing`-wrapped
/// buffer. Individual record encrypt/decrypt calls use the cached key via
/// [`CryptoVault::encrypt_with_key`] / [`CryptoVault::decrypt_with_key`].
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

/// AES-256-GCM-SIV authenticated-encryption implementation.
///
/// AES-256-GCM-SIV is a **nonce-misuse-resistant** AEAD scheme standardized
/// in RFC 8452. It provides:
///
/// - **Confidentiality** — AES-256 in counter mode; 256-bit key.
/// - **Integrity / authenticity** — a 128-bit Galois/Counter MAC tag appended
///   to the ciphertext. [`AuthenticatedCipher::decrypt`] returns an error
///   rather than plaintext when authentication fails.
/// - **Nonce-misuse resistance** — if two records accidentally share the same
///   nonce under the same key, an attacker can learn whether their plaintexts
///   are equal, but the key and the individual plaintexts remain protected.
///   This is strictly stronger than standard AES-GCM, which leaks the
///   authentication key on nonce reuse. In practice the vault samples an
///   independent 12-byte nonce from `OsRng` per record, making collisions
///   negligible.
///
/// Key length must be exactly 32 bytes; nonce length must be exactly 12 bytes
/// (returned by [`AuthenticatedCipher::nonce_len`]).
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

/// Reed-Solomon forward error correction (FEC) codec over GF(2^8).
///
/// Encodes data as RS(*n*, *k*) codewords where by default:
///
/// - *k* = [`RS_DEFAULT_DATA_LEN`] (223) data bytes per block
/// - *t* = [`RS_DEFAULT_PARITY_LEN`] (32) parity bytes per block
/// - *n* = *k* + *t* = 255 (the maximum codeword length in GF(2^8))
///
/// This RS(255, 223) code can **correct up to 16 corrupted bytes per block**
/// (`floor(32 / 2) = 16`), regardless of where in the block the errors fall.
/// It is the same code used in deep-space communication (CCSDS standard).
///
/// ## Chunking
///
/// Input data longer than 223 bytes is split into 223-byte chunks. Each chunk
/// is encoded independently. On decode, each 255-byte block is corrected in
/// isolation, the data portions are concatenated, and the result is truncated
/// to `original_len` bytes to remove zero-padding in the final (possibly
/// short) chunk.
///
/// ## Custom sizes
///
/// Non-default parity / data lengths can be constructed via [`Self::new`],
/// subject to the constraint that `parity_len + data_len <= 255`. The default
/// constructor ([`Default::default`]) uses the RS(255, 223) parameters.
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
    /// Creates a `ReedSolomonCodec` with the specified parity and data block
    /// sizes.
    ///
    /// # Parameters
    ///
    /// - `parity_len` — number of parity bytes per block. Must be > 0. The
    ///   codec can correct up to `floor(parity_len / 2)` corrupted bytes per
    ///   block.
    /// - `data_len` — number of data bytes per block. Must be > 0.
    ///
    /// # Errors
    ///
    /// [`CryptoError::InvalidInput`] if either length is zero or if
    /// `parity_len + data_len > 255` (the GF(2^8) symbol limit).
    ///
    /// # Note
    ///
    /// Most callers should use [`Default::default`] to obtain the audited
    /// RS(255, 223) configuration. Only use `new` when a non-standard code
    /// rate is required.
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

/// Cryptographic vault: the public facade composing key derivation,
/// authenticated encryption, and forward error correction.
///
/// `CryptoVault` is the **sole entry point** for persistent record encryption
/// in magi-rs. The store layer (`EncryptedSqliteMemory`) holds one instance
/// per database session and follows this two-phase protocol:
///
/// **Phase 1 — key setup (once per session):**
/// ```ignore
/// let key = vault.derive_key(master_password, &salt)?;
/// // `key` is cached as a Zeroizing<Vec<u8>> for the session lifetime.
/// ```
///
/// **Phase 2 — per-record encrypt / decrypt:**
/// ```ignore
/// let blob  = vault.encrypt_with_key(&key, plaintext)?;
/// let plain = vault.decrypt_with_key(&key, &blob)?;
/// ```
///
/// This design eliminates the O(N) Argon2 cost that would arise from deriving
/// the key on every record access when loading a long history.
///
/// ## Composability
///
/// The three strategies are injected at construction (see [`CryptoVault::new`]):
///
/// - `kdf: Box<dyn KeyDerivation>` — defaults to [`Argon2Kdf`] (Argon2id,
///   OWASP 2025 params).
/// - `cipher: Box<dyn AuthenticatedCipher>` — defaults to
///   [`Aes256GcmSivCipher`] (AES-256-GCM-SIV, nonce-misuse resistant).
/// - `fec: Box<dyn ErrorCorrection>` — defaults to [`ReedSolomonCodec`]
///   (RS(255, 223), corrects up to 16 corrupted bytes per block).
///
/// Tests may substitute lightweight mock implementations to avoid the Argon2
/// cost and to inject deterministic nonces.
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
    /// Creates a `CryptoVault` with custom strategy implementations.
    ///
    /// Prefer [`CryptoVault::default`] for production use, which wires the
    /// audited [`Argon2Kdf`] + [`Aes256GcmSivCipher`] + [`ReedSolomonCodec`]
    /// defaults. Use `new` only when injecting test doubles or experimenting
    /// with alternative algorithms.
    ///
    /// # Parameters
    ///
    /// - `kdf` — key derivation strategy. Must produce a key of at least
    ///   [`KEY_LEN`] bytes for use with `cipher`.
    /// - `cipher` — authenticated encryption strategy.
    /// - `fec` — forward error correction strategy.
    #[allow(dead_code)]
    pub fn new(
        kdf: Box<dyn KeyDerivation>,
        cipher: Box<dyn AuthenticatedCipher>,
        fec: Box<dyn ErrorCorrection>,
    ) -> Self {
        Self { kdf, cipher, fec }
    }

    /// Derives a [`KEY_LEN`]-byte (32-byte) key from `password` and `salt`
    /// using the module's audited Argon2id parameters.
    ///
    /// This method is designed to be called **exactly once per session**. The
    /// returned key should be cached by the caller (e.g. in a
    /// `Zeroizing<Vec<u8>>` field) and passed to
    /// [`Self::encrypt_with_key`] / [`Self::decrypt_with_key`] for each record.
    /// Re-deriving the key per record incurs unnecessary Argon2 cost.
    ///
    /// # Parameters
    ///
    /// - `password` — the master passphrase (stored in the OS keyring under
    ///   `magi-rs-internal`). Must not be empty.
    /// - `salt` — the per-database random salt (stored in `vault_meta`; exactly
    ///   [`SALT_LEN`] = 16 bytes).
    ///
    /// # Returns
    ///
    /// A `Zeroizing`-wrapped 32-byte key. The buffer is wiped from memory
    /// when the wrapper is dropped, ensuring the key does not linger in the
    /// heap after the session ends.
    ///
    /// # Errors
    ///
    /// - [`CryptoError::InvalidInput`] if `password` is empty.
    /// - [`CryptoError::KeyDerivation`] if Argon2 fails (should not occur with
    ///   the default parameters and a valid 16-byte salt).
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

    /// Encrypts `plaintext` under a pre-derived `key`, returning a base64
    /// blob.
    ///
    /// A fresh random 12-byte nonce is sampled from `OsRng` on every call, so
    /// encrypting the same plaintext twice produces different blobs. The nonce
    /// is stored inside the blob and recovered transparently by
    /// [`Self::decrypt_with_key`].
    ///
    /// ## Blob layout
    ///
    /// ```text
    /// [BLOB_VERSION (1 B)] ++ [original_len as u32 LE (4 B)]
    ///     ++ [RS-encoded( nonce(12 B) || ciphertext || tag(16 B) )]
    /// ```
    ///
    /// The entire binary blob is then base64-encoded. The per-database salt is
    /// **not** included in the blob; it is managed by the store layer.
    ///
    /// # Parameters
    ///
    /// - `key` — the 32-byte session key from [`Self::derive_key`].
    /// - `plaintext` — the UTF-8 string to encrypt (e.g. a JSON-serialized
    ///   message or knowledge value).
    ///
    /// # Returns
    ///
    /// A base64-encoded blob string suitable for storage in a SQLite text
    /// column.
    ///
    /// # Errors
    ///
    /// - [`CryptoError::InvalidInput`] if `nonce_len + plaintext.len() + 16`
    ///   exceeds [`MAX_PLAINTEXT_LEN`] (50 MiB) — C7 allocation guard.
    /// - [`CryptoError::Cipher`] if `key` has the wrong length (must be 32
    ///   bytes for AES-256-GCM-SIV).
    pub fn encrypt_with_key(&self, key: &[u8], plaintext: &str) -> Result<String, CryptoError> {
        // Delegates to the shared byte-level core; a UTF-8 string is just bytes here.
        self.encrypt_bytes(key, plaintext.as_bytes())
    }

    /// Decrypts a base64 blob produced by [`Self::encrypt_with_key`] under the
    /// same `key`.
    ///
    /// Applies all C7 allocation-DoS guards before performing any large
    /// allocation:
    ///
    /// 1. **Version check** — the first byte must equal `BLOB_VERSION = 1`;
    ///    an unknown version is rejected immediately.
    /// 2. **Length cap** — the `original_len` field must be <=
    ///    [`MAX_PLAINTEXT_LEN`] (50 MiB).
    /// 3. **Body-size consistency** — the RS body length must be <=
    ///    `original_len * 2 + 4096`; a grossly over-sized body (indicating
    ///    a small declared length paired with a huge encoded body) is rejected
    ///    before the RS decoder allocates working memory.
    ///
    /// # Parameters
    ///
    /// - `key` — the 32-byte session key from [`Self::derive_key`].
    /// - `encrypted_base64` — the base64 string returned by
    ///   [`Self::encrypt_with_key`].
    ///
    /// # Returns
    ///
    /// The original UTF-8 plaintext string.
    ///
    /// # Errors
    ///
    /// - [`CryptoError::Encoding`] for malformed base64 or non-UTF-8 output.
    /// - [`CryptoError::InvalidInput`] for a hostile/malformed length prefix,
    ///   unsupported blob version, or body-size inconsistency (C7 guards).
    /// - [`CryptoError::ErrorCorrection`] if the RS decoder cannot recover the
    ///   data (more than 16 corrupted bytes per block).
    /// - [`CryptoError::Cipher`] if the key is wrong or the authentication tag
    ///   does not match (tampered ciphertext).
    pub fn decrypt_with_key(
        &self,
        key: &[u8],
        encrypted_base64: &str,
    ) -> Result<String, CryptoError> {
        // Delegates to the shared byte-level core (all C7 guards live there),
        // then re-imposes the UTF-8 contract on the recovered plaintext bytes.
        let plaintext = self.decrypt_bytes(key, encrypted_base64)?;
        String::from_utf8(plaintext.to_vec())
            .map_err(|e| CryptoError::Encoding(format!("Invalid UTF-8: {}", e)))
    }

    // ── Byte-level core + envelope key-wrapping helpers ─────────────────

    /// Byte-level encryption core shared by [`Self::encrypt_with_key`] (UTF-8
    /// strings) and [`Self::wrap_key`] (raw key material). Runs the full
    /// nonce → AES-256-GCM-SIV → Reed-Solomon → versioned-blob pipeline on
    /// arbitrary bytes, returning the base64 blob.
    ///
    /// # Errors
    /// [`CryptoError::InvalidInput`] if the projected record exceeds
    /// [`MAX_PLAINTEXT_LEN`]; [`CryptoError::Cipher`] on an invalid key length.
    fn encrypt_bytes(&self, key: &[u8], plaintext: &[u8]) -> Result<String, CryptoError> {
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

        let ciphertext = self.cipher.encrypt(key, &nonce, plaintext)?;

        let mut plaindata = Vec::with_capacity(nonce_len + ciphertext.len());
        plaindata.extend_from_slice(&nonce);
        plaindata.extend_from_slice(&ciphertext);

        let rs_encoded = self.fec.encode(&plaindata);

        let original_len_u32 = u32::try_from(plaindata.len())
            .map_err(|_| CryptoError::Encoding("Data too large for length header".to_string()))?;
        let mut blob = Vec::with_capacity(1 + 4 + rs_encoded.len());
        blob.push(BLOB_VERSION);
        blob.extend_from_slice(&original_len_u32.to_le_bytes());
        blob.extend_from_slice(&rs_encoded);

        Ok(STANDARD.encode(&blob))
    }

    /// Byte-level decryption core shared by [`Self::decrypt_with_key`] and
    /// [`Self::unwrap_key`]. Applies all C7 allocation-DoS guards (version →
    /// length cap → body-size consistency) before decoding, then returns the
    /// recovered plaintext bytes in a `Zeroizing` buffer (so key material is
    /// wiped on drop).
    ///
    /// # Errors
    /// [`CryptoError::Encoding`] for malformed base64; [`CryptoError::InvalidInput`]
    /// for a hostile/malformed length prefix, bad version, or body-size
    /// inconsistency; [`CryptoError::ErrorCorrection`] if RS cannot recover;
    /// [`CryptoError::Cipher`] if the key is wrong or the tag does not verify.
    fn decrypt_bytes(
        &self,
        key: &[u8],
        encrypted_base64: &str,
    ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        let nonce_len = self.cipher.nonce_len();
        let blob = STANDARD
            .decode(encrypted_base64)
            .map_err(|e| CryptoError::Encoding(format!("Invalid base64: {}", e)))?;

        if blob.len() < 5 {
            return Err(CryptoError::Encoding(
                "Encrypted blob too short".to_string(),
            ));
        }

        if blob[0] != BLOB_VERSION {
            return Err(CryptoError::InvalidInput(format!(
                "Unsupported blob version {} (expected {})",
                blob[0], BLOB_VERSION
            )));
        }

        let len_bytes: [u8; 4] = blob[1..5].try_into().unwrap();
        let original_len = u32::from_le_bytes(len_bytes) as usize;

        if original_len > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::InvalidInput(format!(
                "Length header {} exceeds MAX_PLAINTEXT_LEN ({}); refusing to allocate",
                original_len, MAX_PLAINTEXT_LEN
            )));
        }

        if original_len > (blob.len() - 5) {
            return Err(CryptoError::InvalidInput(
                "Length header exceeds encoded data size".to_string(),
            ));
        }

        if blob.len().saturating_sub(5) > original_len.saturating_mul(2).saturating_add(4096) {
            return Err(CryptoError::InvalidInput(format!(
                "Encoded blob length {} is inconsistent with declared plaintext length {}; refusing to allocate",
                blob.len() - 5,
                original_len
            )));
        }

        let plaindata = self.fec.decode(&blob[5..], original_len)?;
        if plaindata.len() < nonce_len {
            return Err(CryptoError::InvalidInput(
                "Decoded blob too short for nonce".to_string(),
            ));
        }
        let nonce = &plaindata[..nonce_len];
        let ciphertext = &plaindata[nonce_len..];

        Ok(Zeroizing::new(self.cipher.decrypt(key, nonce, ciphertext)?))
    }

    /// Wraps raw `key_material` (e.g. a 32-byte envelope DEK) under the
    /// key-encryption key `kek`, returning a base64 blob.
    ///
    /// Unlike [`Self::encrypt_with_key`] (which takes a UTF-8 `&str`), this
    /// accepts arbitrary bytes — the right primitive for envelope encryption,
    /// where a random Data Encryption Key is wrapped under a passphrase-derived
    /// KEK. The same authenticated pipeline applies, so unwrapping with the
    /// wrong `kek` fails the GCM-SIV tag (a clean wrong-passphrase signal).
    ///
    /// # Errors
    /// As [`Self::encrypt_bytes`].
    // Foundational primitive for the upcoming Vault store: wraps the random
    // envelope DEK under the passphrase-derived KEK (sbtdd/spec-behavior-Vault-base.md,
    // A-V10). No production caller yet — covered by the `wrap_unwrap` tests below.
    #[allow(dead_code)]
    pub fn wrap_key(&self, kek: &[u8], key_material: &[u8]) -> Result<String, CryptoError> {
        self.encrypt_bytes(kek, key_material)
    }

    /// Unwraps key material produced by [`Self::wrap_key`] under the same `kek`,
    /// returning the bytes in a `Zeroizing` buffer (wiped on drop).
    ///
    /// A wrong `kek` (e.g. derived from an incorrect passphrase) makes the
    /// authentication tag fail, returning [`CryptoError::Cipher`] **without**
    /// revealing any key material — the basis for safe wrong-passphrase
    /// detection in an envelope key model.
    ///
    /// # Errors
    /// As [`Self::decrypt_bytes`].
    // Foundational primitive for the upcoming Vault store: unwraps the envelope
    // DEK under the passphrase-derived KEK (sbtdd/spec-behavior-Vault-base.md,
    // A-V10). No production caller yet — covered by the `wrap_unwrap` tests below.
    #[allow(dead_code)]
    pub fn unwrap_key(&self, kek: &[u8], wrapped: &str) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        self.decrypt_bytes(kek, wrapped)
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
    fn test_wrap_unwrap_key_roundtrips_raw_bytes() {
        // V-1: wrap_key/unwrap_key round-trip arbitrary (non-UTF-8) key material —
        // the envelope DEK is raw 32-byte binary, not a UTF-8 string.
        let vault = CryptoVault::default();
        let kek = k(&vault);
        let dek: [u8; KEY_LEN] = [0xA5; KEY_LEN];
        let wrapped = vault.wrap_key(&kek, &dek).unwrap();
        let unwrapped = vault.unwrap_key(&kek, &wrapped).unwrap();
        assert_eq!(
            &*unwrapped, &dek,
            "unwrap must recover the exact key material"
        );
    }

    #[test]
    fn test_wrap_key_preserves_full_byte_range_non_utf8() {
        // V-2: every possible byte value round-trips, incl. 0x80..=0xFF that are
        // invalid as standalone UTF-8 — proving the bytes API has no UTF-8 gate.
        let vault = CryptoVault::default();
        let kek = k(&vault);
        let material: Vec<u8> = (0u8..=255).collect();
        let wrapped = vault.wrap_key(&kek, &material).unwrap();
        assert_eq!(&*vault.unwrap_key(&kek, &wrapped).unwrap(), &material[..]);
    }

    #[test]
    fn test_unwrap_with_wrong_kek_fails_with_cipher_error() {
        // V-3 (envelope wrong-passphrase signal): unwrapping under a KEK derived
        // from a different passphrase fails the GCM-SIV tag — a clean, typed
        // error, never a panic and never leaked key material.
        let vault = CryptoVault::default();
        let kek_a = vault.derive_key("pass-a", &[7u8; SALT_LEN]).unwrap();
        let kek_b = vault.derive_key("pass-b", &[7u8; SALT_LEN]).unwrap();
        let wrapped = vault.wrap_key(&kek_a, &[0x11; KEY_LEN]).unwrap();
        assert!(matches!(
            vault.unwrap_key(&kek_b, &wrapped),
            Err(CryptoError::Cipher(_))
        ));
    }

    #[test]
    fn test_wrap_key_handles_empty_material() {
        // V-4: empty key material is a valid (degenerate) input and round-trips.
        let vault = CryptoVault::default();
        let kek = k(&vault);
        let wrapped = vault.wrap_key(&kek, &[]).unwrap();
        assert_eq!(&*vault.unwrap_key(&kek, &wrapped).unwrap(), &[] as &[u8]);
    }

    #[test]
    fn test_unwrap_rejects_tampered_blob_beyond_rs_capacity() {
        // V-5: corruption beyond the RS(255,223) correction capacity (>16 bytes
        // per block) is rejected (RS or auth failure), never silently wrong.
        let vault = CryptoVault::default();
        let kek = k(&vault);
        let wrapped = vault.wrap_key(&kek, &[0x22; KEY_LEN]).unwrap();
        let mut raw = STANDARD.decode(&wrapped).unwrap();
        for b in raw.iter_mut().skip(5).take(40) {
            *b ^= 0xFF; // corrupt 40 bytes inside the first RS block
        }
        assert!(vault.unwrap_key(&kek, &STANDARD.encode(&raw)).is_err());
    }

    #[test]
    fn test_unwrap_preserves_c7_caps() {
        // V-6: the byte core enforces the same C7 allocation guards — an oversized
        // declared length is rejected pre-allocation (shared with decrypt_with_key).
        let vault = CryptoVault::default();
        let kek = k(&vault);
        let mut over = vec![1u8]; // valid version byte
        over.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        over.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            vault.unwrap_key(&kek, &STANDARD.encode(&over)),
            Err(CryptoError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_wrap_and_encrypt_with_key_share_one_format() {
        // V-7 (DRY/interop): a blob produced by encrypt_with_key (UTF-8 path) and
        // one produced by wrap_key share the same format — unwrap_key recovers the
        // exact UTF-8 bytes of an encrypt_with_key blob, proving a single pipeline.
        let vault = CryptoVault::default();
        let key = k(&vault);
        let s = "sk-ant-shared-format";
        let blob = vault.encrypt_with_key(&key, s).unwrap();
        assert_eq!(&*vault.unwrap_key(&key, &blob).unwrap(), s.as_bytes());
    }

    #[test]
    fn test_blob_carries_version_byte() {
        // A-S1 (#13): the blob starts with the format version byte (1).
        let vault = CryptoVault::default();
        let key = k(&vault);
        let raw = STANDARD
            .decode(vault.encrypt_with_key(&key, "payload").unwrap())
            .unwrap();
        assert_eq!(raw[0], 1, "blob must start with the format version byte");
    }

    #[test]
    fn test_decrypt_rejects_unsupported_blob_version() {
        // A-S2 (#13): a blob whose version byte is unsupported is rejected with a
        // version error, not misread as length/data.
        let vault = CryptoVault::default();
        let key = k(&vault);
        let mut raw = STANDARD
            .decode(vault.encrypt_with_key(&key, "x").unwrap())
            .unwrap();
        raw[0] = 2; // unsupported version (current is 1)
        let err = vault
            .decrypt_with_key(&key, &STANDARD.encode(&raw))
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("version"),
            "an unsupported blob version must be rejected with a version error: {err}"
        );
    }

    #[test]
    fn test_blob_layout_carries_no_salt() {
        // S-2: plaindata == nonce_len + (L + 16 tag); no 16-byte salt prefix.
        let vault = CryptoVault::default();
        let key = k(&vault);
        let pt = "0123456789"; // L = 10
        let blob = vault.encrypt_with_key(&key, pt).unwrap();
        let raw = STANDARD.decode(&blob).unwrap();
        let original_len = u32::from_le_bytes(raw[1..5].try_into().unwrap()) as usize;
        let codec = ReedSolomonCodec::default();
        let plaindata = codec.decode(&raw[5..], original_len).unwrap();
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

        let mut over = vec![1u8]; // valid version byte, so the cap (not version) is tested
        over.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        over.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            vault.decrypt_with_key(&key, &STANDARD.encode(&over)),
            Err(CryptoError::InvalidInput(_))
        ));

        let mut big = vec![1u8]; // valid version byte
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
    fn test_decrypt_rejects_prefix_at_exactly_cap_plus_one() {
        let oversized = (MAX_PLAINTEXT_LEN + 1) as u32;
        let mut blob = vec![1u8]; // valid version byte
        blob.extend_from_slice(&oversized.to_le_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let encoded = STANDARD.encode(&blob);

        let vault = CryptoVault::default();
        let key = k(&vault);
        assert!(
            matches!(
                vault.decrypt_with_key(&key, &encoded),
                Err(CryptoError::InvalidInput(_))
            ),
            "a length prefix one byte over the cap must be rejected"
        );
    }

    #[test]
    fn test_encrypt_rejects_plaintext_over_cap() {
        let vault = CryptoVault::default();
        let key = k(&vault);
        let huge = "a".repeat(MAX_PLAINTEXT_LEN + 1);
        assert!(
            matches!(
                vault.encrypt_with_key(&key, &huge),
                Err(CryptoError::InvalidInput(_))
            ),
            "encrypting beyond MAX_PLAINTEXT_LEN must be rejected"
        );
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
}
