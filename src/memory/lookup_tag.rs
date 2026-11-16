// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-15

//! Keyed lookup tags for the encrypted code-review tables (RV19).
//!
//! A column that SQL must match or index, but that would identify the reviewed repository if
//! stored in the clear (a path, a content key), holds `HMAC-SHA-256(subkey, domain || value)`
//! in hex instead, where `subkey = HMAC-SHA-256(data key, purpose label)`. Equal inputs meet in
//! SQL; without the passphrase the tags reveal nothing but equality. Each table passes its own
//! purpose label, so the doc index and the history never share a subkey.
//!
//! # Examples
//!
//! ```ignore
//! let tags = lookup_tags(&mut dek, b"magi-rs/example/lookup-tag-key/v1", b"path\0", ["src/a.rs"]);
//! assert_eq!(tags.len(), 1);
//! ```

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use magi_rs::vault::MaskedDek;

/// The HMAC block size of SHA-256 (RFC 2104).
const HMAC_BLOCK_BYTES: usize = 64;

/// HMAC-SHA-256 (RFC 2104) of `data` under `key`: implemented over the `sha2` crate the project
/// already depends on, to avoid a new dependency for twenty lines; pinned by the RFC 4231 vector.
///
/// # Arguments
/// * `key` - the MAC key (any length).
/// * `data` - the message.
///
/// # Returns
/// The 32-byte tag, wiped on drop (with the data key as `key` it is the tag subkey). O(key + data).
///
/// Declared limit: the internal state of `sha2`'s hasher is not wiped (best effort, like
/// `MaskedDek`); the pads, the hashed key and both digests are.
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> Zeroizing<[u8; 32]> {
    // MAGI FIX (MS3 §6 loop19, R1): the key may be the data key; the pads (key XOR a constant,
    // so invertible) and the hashed long key are wiped on drop and never copied by value.
    let mut hashed = Zeroizing::new([0u8; 32]);
    let key = if key.len() > HMAC_BLOCK_BYTES {
        hashed.copy_from_slice(Sha256::digest(key).as_slice());
        hashed.as_slice()
    } else {
        key
    };
    let mut inner_pad = Zeroizing::new([0x36u8; HMAC_BLOCK_BYTES]);
    let mut outer_pad = Zeroizing::new([0x5cu8; HMAC_BLOCK_BYTES]);
    for ((i, o), k) in inner_pad.iter_mut().zip(outer_pad.iter_mut()).zip(key) {
        *i ^= k;
        *o ^= k;
    }
    let mut inner = Sha256::new()
        .chain_update(&inner_pad[..])
        .chain_update(data)
        .finalize();
    let mut outer = Sha256::new()
        .chain_update(&outer_pad[..])
        .chain_update(inner)
        .finalize();
    let mut mac = Zeroizing::new([0u8; 32]);
    mac.copy_from_slice(outer.as_slice());
    inner.as_mut_slice().zeroize();
    outer.as_mut_slice().zeroize();
    mac
}

/// Lowercase hex of `bytes`.
pub(crate) fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The lookup tags of `values`: `HMAC-SHA-256(subkey, domain || value)`, hex, where
/// `subkey = HMAC-SHA-256(data key, subkey_label)`.
///
/// The data key is reached only through `with_dek` (never copied out), the subkey lives for this
/// call and is zeroed, and one call serves a whole list so the mask is rotated once.
/// Deterministic, so equal inputs meet in SQL. Never call while holding the connection lock
/// (R-V08). O(total bytes).
///
/// # Arguments
/// * `dek` - the masked data key.
/// * `subkey_label` - the purpose label the subkey is derived under (one per table; versioned,
///   since changing it orphans every stored tag).
/// * `domain` - the domain prefix, so two columns with the same text never share a tag.
/// * `values` - the values to tag.
///
/// # Returns
/// One 64-character hex tag per value, in order.
pub(crate) fn lookup_tags<'v>(
    dek: &mut MaskedDek,
    subkey_label: &[u8],
    domain: &[u8],
    values: impl IntoIterator<Item = &'v str>,
) -> Vec<String> {
    dek.with_dek(|k| {
        let subkey = hmac_sha256(k, subkey_label);
        values
            .into_iter()
            .map(|v| {
                let mut msg = domain.to_vec();
                msg.extend_from_slice(v.as_bytes());
                to_hex(hmac_sha256(subkey.as_slice(), &msg).as_slice())
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_pads_are_zeroized() {
        // [MAGI §6 MS3 loop19, R1] a stack residue cannot be tested deterministically, so the
        // guard reads the source: the pads (key XOR constant, invertible to the data key) and the
        // hashed long key are `Zeroizing`, and no pad is passed by value to the hasher.
        let src = include_str!("lookup_tag.rs").replace('\r', "");
        let start = src.find("fn hmac_sha256(").expect("the function");
        let body = &src[start..];
        let body = &body[..body.find("/// Lowercase hex").expect("its end")];
        assert!(body.matches("Zeroizing").count() >= 3, "{body}");
        assert!(!body.contains("chain_update(inner_pad)"), "{body}");
        assert!(!body.contains("chain_update(outer_pad)"), "{body}");
    }

    #[test]
    fn hmac_result_and_digests_are_zeroized() {
        // [MAGI §6 MS3 loop22, R1] with the data key as key the MAC is the tag subkey, and the
        // inner/outer digests are derived from it: the MAC is returned `Zeroizing` and both
        // digests are wiped before return.
        let src = include_str!("lookup_tag.rs").replace('\r', "");
        let start = src.find("fn hmac_sha256(").expect("the function");
        let body = &src[start..];
        let body = &body[..body.find("/// Lowercase hex").expect("its end")];
        assert!(
            body.contains("-> Zeroizing<[u8; 32]>"),
            "the MAC is returned as Zeroizing: {body}"
        );
        assert!(body.contains("inner.as_mut_slice().zeroize()"), "{body}");
        assert!(body.contains("outer.as_mut_slice().zeroize()"), "{body}");
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231_test_case_2() {
        // RFC 4231, case 2: key "Jefe", data "what do ya want for nothing?".
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            to_hex(mac.as_slice()),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
