// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Fuzz target: lectura acotada + parser de envelope headless con bytes
//! arbitrarios (REQ-H35).
//!
//! Alimenta bytes arbitrarios a `read_input_bounded` (lectura acotada de
//! stdin/`-i`) y a `parse_input` en los tres modos de `InputFormat`
//! (auto-detect, JSON forzado, texto forzado). Invariante: **jamás panic, jamás
//! UB** — toda entrada produce un `Result` tipado, sin OOM (la lectura está
//! acotada a `MAX_INPUT_BYTES`) ni stack overflow (la profundidad JSON está
//! acotada a `MAX_JSON_DEPTH`). Espeja la estructura de los targets del vault.
#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use magi_rs::headless::input::{parse_input, read_input_bounded, InputFormat};
use magi_rs::headless::limits::MAX_INPUT_BYTES;

fuzz_target!(|data: &[u8]| {
    // Lectura acotada de una fuente arbitraria: nunca OOM ni panic. Se pasa el
    // límite constante por default — el target ejercita el parser puro, no la
    // resolución de `[headless] max_input_bytes` de `magi.toml` (eso vive en
    // `main.rs`, bin-only, fuera del alcance de este target de fuzz de la lib).
    let _ = read_input_bounded(Cursor::new(data), MAX_INPUT_BYTES);
    // Auto-detect y ambos formatos forzados: nunca panic / stack-overflow.
    let _ = parse_input(data, None);
    let _ = parse_input(data, Some(InputFormat::Json));
    let _ = parse_input(data, Some(InputFormat::Text));
});
