// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-15
//! Fuzz target: decode de `vault_meta` con bytes arbitrarios (REQ-V39).
//!
//! Alimenta datos arbitrarios (como si fueran `{salt_fec, wrapped_dek}` leídos de
//! disco, posiblemente corruptos/truncados/forjados) al camino de apertura del
//! envelope. Invariante: **jamás panic, jamás un borrado de datos** — siempre un
//! `Result` tipado (`WrongPassphrase` / `VaultMetaCorrupt`). El split determinista
//! (primer `u16` LE = `len(salt_fec)`) vive en `magi_rs::vault::fuzz_open_entrypoint`
//! y es bounds-safe (el lint `clippy::indexing_slicing` del módulo lo obliga).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    magi_rs::vault::fuzz_open_entrypoint(data);
});
