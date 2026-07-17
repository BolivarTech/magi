// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! In-RAM protection of the data key (DEK) — REQ-V41 / REQ-V42.
//!
//! The DEK never sits in RAM in the clear between operations: it lives XOR-masked
//! with a random mask, and the mask is **rotated with fresh [`OsRng`] entropy on
//! every access** ([`MaskedDek::with_dek`]). The backing page is best-effort locked
//! against swap ([`region::lock`]); process-dump suppression is applied by
//! [`harden_process`]. All of these are **best-effort**: a failure warns and
//! continues, it never aborts (a headless/CI environment without `CAP_IPC_LOCK`
//! must keep working — REQ-V05 / SC-V48).
//!
//! # Honest claim
//!
//! These layers reduce **opportunistic/passive** capture of the DEK (swap,
//! pagefile, automatic crash dumps, hibernation). They do **not** defeat an active
//! attacker with debugger access to the process — the mask and the masked value
//! live in the same address space. See `sbtdd/spec-behavior.md` REQ-V42 and the
//! roadmap (`dev-docs/PENDING_IMPLEMENTATION.md` §13.4) for the hardware-backed
//! alternatives that are out of scope for v1.

use rand::rngs::OsRng;
use rand::RngCore;
use zeroize::Zeroize;

use crate::vault::VaultError;

/// 32 B = AES-256 key size (FIPS-197); the DEK is always this length.
const KEY_LEN: usize = 32;

/// Total backing buffer: `[0..KEY_LEN]` masked DEK, `[KEY_LEN..BUF_LEN]` mask.
const BUF_LEN: usize = KEY_LEN * 2;

/// Compile-time assertion (verified in CI, not only in tests — MAGI run 6): if a
/// future `region` release made `LockGuard` non-`Send`, this line breaks the build,
/// because `SecretStore: Send` requires `MaskedDek: Send`.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<MaskedDek>();
};

/// The data key held masked in RAM, with per-access mask rotation.
///
/// `buf` is a single contiguous heap buffer holding the masked DEK in its first
/// `KEY_LEN` bytes and the current mask in the rest; the plaintext DEK equals
/// `masked[i] ^ mask[i]`. Fixed size (`Box<[u8; BUF_LEN]>`) so it can never
/// reallocate (which would move the locked page). Zeroized on drop.
///
/// # Page locking
///
/// The backing page is locked with [`region::lock`] and the guard is **leaked**
/// ([`std::mem::forget`]) rather than stored: the DEK is meant to stay resident for
/// the whole process (REQ-V41), and page locks are **not reference-counted** —
/// two `MaskedDek`s whose small buffers share a page would otherwise double-unlock
/// on drop and panic (region crate, Windows). Locked pages are released by the OS at
/// process exit. Best-effort: a lock failure warns and continues (SC-V48).
pub struct MaskedDek {
    /// `[0..KEY_LEN]` = masked DEK, `[KEY_LEN..BUF_LEN]` = mask.
    buf: Box<[u8; BUF_LEN]>,
    /// Best-effort warnings (e.g. `mlock` denied) accumulated across the lifetime.
    warnings: Vec<String>,
}

impl Drop for MaskedDek {
    fn drop(&mut self) {
        self.buf.zeroize();
    }
}

/// XORs the masked half against the mask half of `buf` into `out` (recovers the DEK).
fn unmask_into(buf: &[u8; BUF_LEN], out: &mut [u8]) {
    let (masked, mask) = buf.split_at(KEY_LEN);
    for (o, (m, k)) in out.iter_mut().zip(masked.iter().zip(mask.iter())) {
        *o = m ^ k;
    }
}

impl MaskedDek {
    /// Takes ownership of the plaintext DEK, masks it immediately, and best-effort
    /// locks its page against swap.
    ///
    /// The 32-byte length is validated at **runtime** with an explicit
    /// `dek.len() != KEY_LEN` check — not a `debug_assert` (stripped in release)
    /// nor an `assert!` (forbidden by the crate lints). That is why this returns
    /// `Result`.
    ///
    /// The plaintext DEK is masked in place directly from `dek` (a `Zeroizing`
    /// buffer scrubbed on drop); no bare `[u8; KEY_LEN]` stack copy of the key
    /// is ever materialized, so no unlockable plaintext residue is left on the
    /// `new` stack frame (REQ-V42).
    ///
    /// # Errors
    ///
    /// [`VaultError::Crypto`] if `dek.len() != KEY_LEN` or random mask generation
    /// fails (a broken OS entropy source).
    pub fn new(dek: zeroize::Zeroizing<Vec<u8>>) -> Result<Self, VaultError> {
        if dek.len() != KEY_LEN {
            return Err(VaultError::Crypto("DEK length must be 32 bytes".into()));
        }

        let mut buf = Box::new([0u8; BUF_LEN]);
        {
            let (masked, mask) = buf.split_at_mut(KEY_LEN);
            OsRng
                .try_fill_bytes(mask)
                .map_err(|e| VaultError::Crypto(format!("OsRng failed: {e}")))?;
            // `dek` is the only holder of the plaintext key (Zeroizing, scrubbed
            // on drop); XOR straight from it — no plaintext stack copy is made.
            for (out, (d, k)) in masked.iter_mut().zip(dek.iter().zip(mask.iter())) {
                *out = d ^ k;
            }
        }

        let mut warnings = Vec::new();
        match region::lock(buf.as_ptr(), BUF_LEN) {
            // Leak the guard: keep the page locked for the process lifetime and never
            // unlock (page locks are not ref-counted; a shared-page double-unlock
            // would panic). The OS releases it at process exit.
            Ok(guard) => core::mem::forget(guard),
            Err(_) => warnings.push(
                "could not lock the key page in this environment; \
                 pages holding the key could reach swap"
                    .to_string(),
            ),
        }

        Ok(Self { buf, warnings })
    }

    /// Best-effort warnings accumulated so far (empty = the key page is locked).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Unmasks the DEK into a transient `Zeroizing` buffer, runs `f`, then re-masks
    /// with a **fresh** [`OsRng`] mask (SC-V50). Takes `&mut self` because the mask
    /// rotates on every access.
    ///
    /// The re-mask runs inside a drop-guard, so even if `f` panics the buffer is
    /// re-masked before unwinding — the DEK is never left in the clear at rest.
    pub fn with_dek<R>(&mut self, f: impl FnOnce(&[u8]) -> R) -> R {
        let mut plain = zeroize::Zeroizing::new(vec![0u8; KEY_LEN]);
        unmask_into(&self.buf, &mut plain);

        /// Re-masks `m` from `plain` on drop (panic-safe rotation).
        struct Remask<'a> {
            /// The masked DEK to re-mask in place.
            m: &'a mut MaskedDek,
            /// The plaintext DEK bytes to re-mask from.
            plain: &'a [u8],
        }
        impl Drop for Remask<'_> {
            fn drop(&mut self) {
                self.m.remask_from(self.plain);
            }
        }

        let guard = Remask {
            m: self,
            plain: &plain,
        };
        let out = f(&plain);
        drop(guard);
        out
    }

    /// A second copy with its own independent mask (for a sibling store).
    ///
    /// Takes `&mut self` because it unmasks to clone the DEK and, in doing so,
    /// **rotates `self`'s mask** like any other access (SC-V50).
    ///
    /// # Errors
    ///
    /// [`VaultError::Crypto`] if fresh mask generation fails for the copy (a broken
    /// OS entropy source). *(Plan refined `-> Self` to `-> Result` during MS2 impl:
    /// an `OsRng` failure is a real, if near-impossible, error, not a panic.)*
    pub fn duplicate(&mut self) -> Result<Self, VaultError> {
        self.with_dek(|dek| Self::new(zeroize::Zeroizing::new(dek.to_vec())))
    }

    /// Re-masks the buffer in place with a fresh `OsRng` mask; on the (essentially
    /// impossible) `OsRng` failure, re-masks with the existing mask so the buffer is
    /// never left in the clear, and records a warning. Never panics.
    fn remask_from(&mut self, plain: &[u8]) {
        let mut new_mask = [0u8; KEY_LEN];
        let ok = OsRng.try_fill_bytes(&mut new_mask).is_ok();
        let (masked, mask) = self.buf.split_at_mut(KEY_LEN);
        if ok {
            for (out, (p, nk)) in masked.iter_mut().zip(plain.iter().zip(new_mask.iter())) {
                *out = p ^ nk;
            }
            for (k, nk) in mask.iter_mut().zip(new_mask.iter()) {
                *k = *nk;
            }
        } else {
            for (out, (p, k)) in masked.iter_mut().zip(plain.iter().zip(mask.iter())) {
                *out = p ^ k;
            }
            self.warnings
                .push("OsRng failed during mask rotation; mask not rotated".to_string());
        }
        new_mask.zeroize();
    }

    /// Test-only snapshot of the masked half at rest (never the mask nor the
    /// plaintext). Lets tests observe rotation without unmasking.
    #[cfg(test)]
    pub(crate) fn debug_masked_snapshot(&self) -> Vec<u8> {
        let (masked, _) = self.buf.split_at(KEY_LEN);
        masked.to_vec()
    }
}

/// Applies REQ-V42 layer 2 (best-effort dump suppression) and returns any warnings.
///
/// POSIX: `RLIMIT_CORE = 0`. Linux additionally: `PR_SET_DUMPABLE = 0` (also blocks a
/// same-user peer `ptrace`). Windows has no in-process dump suppression (documented
/// asymmetry) and returns an empty vector. Idempotent; never panics or aborts.
pub fn harden_process() -> Vec<String> {
    let warnings = Vec::new();
    #[cfg(unix)]
    let mut warnings = warnings;
    #[cfg(unix)]
    {
        use rlimit::Resource;
        if let Err(e) = rlimit::setrlimit(Resource::CORE, 0, 0) {
            warnings.push(format!("could not set RLIMIT_CORE=0: {e}"));
        }
    }
    #[cfg(target_os = "linux")]
    {
        if prctl::set_dumpable(false).is_err() {
            warnings.push("could not set PR_SET_DUMPABLE=0".to_string());
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::{harden_process, MaskedDek, KEY_LEN};
    use zeroize::Zeroizing;

    fn dek() -> Zeroizing<Vec<u8>> {
        Zeroizing::new((0u8..32).collect())
    }

    #[test]
    fn test_with_dek_exposes_the_original_value_every_time() {
        let mut m = MaskedDek::new(dek()).expect("32B");
        let a = m.with_dek(|d| d.to_vec());
        let b = m.with_dek(|d| d.to_vec());
        assert_eq!(a, (0u8..32).collect::<Vec<u8>>());
        assert_eq!(a, b);
    }

    #[test]
    fn test_masked_representation_rotates_between_accesses() {
        let mut m = MaskedDek::new(dek()).expect("32B");
        let snap1 = m.debug_masked_snapshot();
        m.with_dek(|_| ());
        let snap2 = m.debug_masked_snapshot();
        assert_ne!(snap1, snap2);
    }

    #[test]
    fn test_masked_representation_never_equals_plain_dek() {
        let mut m = MaskedDek::new(dek()).expect("32B");
        let plain = (0u8..32).collect::<Vec<u8>>();
        assert_ne!(m.debug_masked_snapshot(), plain);
        m.with_dek(|_| ());
        assert_ne!(m.debug_masked_snapshot(), plain);
    }

    #[test]
    fn test_duplicate_preserves_value_with_independent_mask() {
        let mut m = MaskedDek::new(dek()).expect("32B");
        let mut d = m.duplicate().expect("dup");
        assert_eq!(m.with_dek(|x| x.to_vec()), d.with_dek(|x| x.to_vec()));
        assert_ne!(m.debug_masked_snapshot(), d.debug_masked_snapshot());
    }

    #[test]
    fn test_new_and_harden_never_fail_in_restricted_environments() {
        let m = MaskedDek::new(dek()).expect("32B");
        let _ = m.warnings();
        let _warns: Vec<String> = harden_process();
    }

    #[test]
    fn test_new_rejects_wrong_length_with_typed_error() {
        assert!(MaskedDek::new(Zeroizing::new(vec![0u8; 16])).is_err());
        assert!(MaskedDek::new(Zeroizing::new(vec![0u8; 33])).is_err());
        assert!(MaskedDek::new(Zeroizing::new(vec![0u8; KEY_LEN])).is_ok());
    }

    #[test]
    fn test_with_dek_remasks_even_if_the_operation_panics() {
        let mut m = MaskedDek::new(dek()).expect("32B");
        let before = m.debug_masked_snapshot();
        let plain = (0u8..32).collect::<Vec<u8>>();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.with_dek(|_k| panic!("boom inside the op"));
        }));
        assert!(r.is_err());
        let after = m.debug_masked_snapshot();
        assert_ne!(after, plain);
        assert_ne!(after, before);
    }
}
