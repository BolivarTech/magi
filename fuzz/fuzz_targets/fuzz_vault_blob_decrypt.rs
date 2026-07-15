// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-15
//! Fuzz target: decrypt de un `value_blob` arbitrario (REQ-V39).
//!
//! Alimenta bytes arbitrarios como blob cifrado a `decrypt_with_key`. Como el
//! blob es un `&str` base64 pero el fuzzer entrega `&[u8]` arbitrario, se
//! convierte con `from_utf8_lossy` (maneja entrada no-UTF8, fix Checkpoint 2).
//! Invariante: **error tipado, jamás panic ni un valor incorrecto en silencio**.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let blob = String::from_utf8_lossy(data);
    let vault = cryptovault::CryptoVault::default();
    let key = [0u8; 32];
    let _ = vault.decrypt_with_key(&key, &blob);
});
