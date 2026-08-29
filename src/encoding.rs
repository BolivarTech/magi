// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Percent-encoding, in one place.
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
}
