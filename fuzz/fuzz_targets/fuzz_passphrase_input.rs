// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! Fuzz target: passphrase input with arbitrary bytes (REQ-V39).
//!
//! Feeds arbitrary bytes as a passphrase (empty, non-UTF8 via lossy, huge) through
//! `check_strength` + a KEK derivation. Invariant: never panic — a weak passphrase
//! returns `WeakPassphrase`. NOTE: input crosses as `&str`, so this covers raw
//! bytes via `String::from_utf8_lossy`, not UTF-8 rejection (no such path exists).
//! Entrypoint lives in `magi_rs::vault::fuzz_passphrase_entrypoint`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    magi_rs::vault::fuzz_passphrase_entrypoint(data);
});
