// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! Fuzz target: secret VALUE round-trip with arbitrary bytes (REQ-V39).
//!
//! Feeds arbitrary bytes as a vault secret value (empty, non-UTF8, huge) through
//! set/get against a throwaway in-memory store. Invariant: never panic — every
//! failure is a typed `VaultError`, and a successful round-trip returns the exact
//! value. Entrypoint lives in `magi_rs::vault::fuzz_value_roundtrip_entrypoint`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    magi_rs::vault::fuzz_value_roundtrip_entrypoint(data);
});
