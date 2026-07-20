// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-19
//! Fuzz target: matriz de autorización de tools por tier headless (REQ-H35 /
//! MS2 Task 10).
//!
//! Interpreta bytes arbitrarios como `(tier_byte, nombre_de_tool)` y ejercita
//! toda la superficie pública de `Policy` (`approves`/`silences_soft_guards`/
//! `warnings` + accesores). Invariantes: **jamás panic** sobre ninguna entrada
//! y **fail-closed** — una aprobación implica un nombre de tool conocido, en
//! cualquier tier (un nombre desconocido nunca devuelve `true`). El entrypoint
//! vive en `magi_rs::headless::fuzz_policy_entrypoint`, espejando la convención
//! de los `fuzz_*_entrypoint` del vault.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    magi_rs::headless::fuzz_policy_entrypoint(data);
});
