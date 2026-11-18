// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-18

//! Keyed lookup tags (RV19): a column SQL must match but that would name the repository holds
//! `HMAC-SHA-256(subkey, domain || value)` instead; each table derives its own subkey.
//!
//! ```ignore
//! let tags = lookup_tags(&mut dek, b"magi-rs/x/lookup-tag-key/v1", b"path\0", ["src/a.rs"]);
//! ```

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use magi_rs::encoding::lower_hex;
use magi_rs::vault::MaskedDek;

/// The HMAC block size of SHA-256 (RFC 2104).
const HMAC_BLOCK_BYTES: usize = 64;

/// HMAC-SHA-256 (RFC 2104) of `data` under `key` (any length), over `sha2` (no new dependency;
/// RFC 4231 vector). Returns the tag, wiped on drop. O(key + data). Declared: `sha2`'s internal
/// state is not wiped (best effort); the pads, hashed key and both digests are.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Zeroizing<[u8; 32]> {
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

/// One hex tag per value, in order: `HMAC(subkey, domain || value)`, `subkey = HMAC(dek,
/// subkey_label)` (one versioned label per table; `domain` separates columns). The key is reached
/// only through `with_dek`, the subkey is zeroed, the mask rotates once per call. Deterministic.
/// Never call under the connection lock (R-V08). O(total bytes).
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
                lower_hex(hmac_sha256(subkey.as_slice(), &msg).as_slice())
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
            lower_hex(mac.as_slice()),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
