// Author: Julian Bolivar
// Version: 0.24.2
// Date: 2026-10-14

//! Percent-encoding and lowercase hex, each in one place.
//!
//! [`lower_hex`] is here for the same reason as [`percent_encode`]: the memory
//! record ids (`pref:`, `turn:`, `mig:`) are SHA-256 digests written as hex and
//! persisted, so every site has to produce the same bytes. `sha2` 0.11 dropped
//! the `{:x}` formatting the sites used to rely on; one function keeps a future
//! change of that kind from altering only some of them.
//!
//! # Why this is a module and not two call sites
//!
//! `vault` needs it to compose the variants the auditor registers, and
//! `magi::endpoint` needs it to substitute a credential into a `base_url`. Those
//! must produce the **same bytes** or the auditor scans for a form that never
//! appears. Two independent calls to `urlencoding::encode` are not a function:
//! either one can move to `utf8_percent_encode` with its own `AsciiSet` and
//! nothing stops compiling, and the failure is silent — a password that is
//! encoded one way in the URL and hashed another way in the auditor.
//!
//! # Why a LEAF module at the crate root
//!
//! It imports nothing from the crate. `magi::endpoint` already does
//! `use crate::vault::SecretStore`, so `magi` depends on `vault`; putting the
//! encoder in either of them and importing it from the other closes a cycle.
//! At the root, both sides import downward.

/// A value that has been percent-encoded.
///
/// A newtype so an encoded value cannot be mistaken for a raw one at a call
/// site that takes `&str` — which is how a double-encoded credential gets built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PercentEncoded(String);

impl PercentEncoded {
    /// The encoded text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PercentEncoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Percent-encodes `raw` for use inside a URL's userinfo.
///
/// # Parameters
///
/// * `raw` — the value as the vault stores it.
///
/// # Returns
///
/// The encoded form, which is what actually appears in a `base_url`.
///
/// # Why the auditor needs this
///
/// It is REQ-L49's **main** case, not an extra. A password containing `@`, `/`,
/// `?` or `#` **never appears in its raw form inside a URL**, so an auditor that
/// only registered the raw value would be blind exactly where credentials live.
///
/// # Complexity
///
/// `O(n)`.
#[must_use]
pub fn percent_encode(raw: &str) -> PercentEncoded {
    PercentEncoded(urlencoding::encode(raw).into_owned())
}

/// Writes `bytes` as lowercase hexadecimal, two digits per byte.
///
/// # Parameters
///
/// * `bytes` — any byte slice, typically a digest (`Sha256::digest(..)` or
///   `hasher.finalize()`, both of which deref to `[u8]`).
///
/// # Returns
///
/// A string of `2 * bytes.len()` characters from `0-9a-f`; empty for empty
/// input. The output is identical to what `format!("{:x}", digest)` produced
/// under `sha2` 0.10, which is what keeps already persisted ids valid.
///
/// # Examples
///
/// ```
/// assert_eq!(magi_rs::encoding::lower_hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
/// ```
///
/// # Complexity
///
/// `O(n)` time and space.
#[must_use]
pub fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing into a `String` cannot fail.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_password_with_reserved_characters_does_not_survive_unencoded() {
        // The case REQ-L49 calls the main one: this value cannot appear raw
        // inside a URL, so registering only the raw form is blindness.
        let raw = "p4ss@word/with?reserved#chars";
        let encoded = percent_encode(raw);
        assert_ne!(encoded.as_str(), raw);
        for c in ['@', '/', '?', '#'] {
            assert!(
                !encoded.as_str().contains(c),
                "{c} survived encoding: {encoded}"
            );
        }
    }

    #[test]
    fn an_ordinary_password_encodes_to_itself() {
        // Without this the assertion above passes against an encoder that
        // mangles everything.
        let raw = "correcthorsebatterystaple";
        assert_eq!(percent_encode(raw).as_str(), raw);
    }

    #[test]
    fn encoding_is_not_idempotent_and_that_is_why_the_newtype_exists() {
        // A `%` becomes `%25`, so encoding twice produces a third string that
        // matches neither. The newtype is what keeps a call site from feeding an
        // already-encoded value back in.
        let once = percent_encode("a b");
        let twice = percent_encode(once.as_str());
        assert_ne!(once.as_str(), twice.as_str());
    }

    #[test]
    fn lower_hex_of_a_sha256_digest_matches_the_published_vector() {
        // Persisted memory ids are this hex. The vectors are the standard
        // SHA-256 of "" and of the profile test's sentence, computed outside
        // Rust; a formatting change that alters any id turns this red.
        use sha2::{Digest, Sha256};
        assert_eq!(
            lower_hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            lower_hex(&Sha256::digest(b"always use dark mode")),
            "2ddca4ab8b1560b5ff31988665a80c9e824b015f5bc4f9daab31e7aa54aab61c"
        );
    }

    #[test]
    fn lower_hex_pads_every_byte_to_two_digits_and_uses_lowercase() {
        assert_eq!(lower_hex(&[]), "");
        assert_eq!(lower_hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }
}
