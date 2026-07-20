// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Fuzz target: sanitizador de mensajes de error + redactor de secretos con
//! bytes arbitrarios (REQ-H35 / REQ-H15c).
//!
//! Alimenta bytes arbitrarios (como `&str` vía `from_utf8_lossy`) a
//! `sanitize_error_message` y `redact_secret_patterns`. Invariante: **jamás
//! panic, jamás UB**, y la redacción es **idempotente** — un patrón tipo-clave
//! superviviente rompería la idempotencia en un segundo pase, el proxy de "nunca
//! se deja pasar un patrón tipo-clave sin redactar". El entrypoint vive en
//! `magi_rs::headless::fuzz_sanitize_error_entrypoint`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    magi_rs::headless::fuzz_sanitize_error_entrypoint(data);
});
