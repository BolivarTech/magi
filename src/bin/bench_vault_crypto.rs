// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-15
//! Línea base de performance de `ConcatenatedFec` (REQ-V36).
//!
//! Mide el **costo de decrypt** y la **expansión en disco** del pipeline
//! `AES-256-GCM-SIV + ConcatenatedFec` del crate `cryptovault`, sobre un corpus
//! representativo de mensajes. Es **historial de desempeño**, NO un gate: la
//! elección de FEC está fijada por diseño (D-V02); este número existe para
//! comparar en releases futuros, no para reabrir la decisión.
//!
//! Uso: `cargo run --release --bin bench_vault_crypto`

use std::time::Instant;

/// Cantidad de récords del corpus sintético.
const CORPUS_SIZE: usize = 1_000;
/// Longitud representativa de un mensaje de conversación (bytes).
const MESSAGE_LEN: usize = 220;

fn main() {
    let vault = cryptovault::CryptoVault::default();
    let salt = cryptovault::generate_salt().expect("salt");
    let key = vault
        .derive_key("bench-master-key-string", &salt)
        .expect("key");

    let plaintext: String = "x".repeat(MESSAGE_LEN);

    // Cifrar el corpus y medir la expansión en disco.
    let mut blobs = Vec::with_capacity(CORPUS_SIZE);
    let mut encoded_bytes = 0usize;
    for _ in 0..CORPUS_SIZE {
        let blob = vault.encrypt_with_key(&key, &plaintext).expect("encrypt");
        encoded_bytes += blob.len();
        blobs.push(blob);
    }
    let plain_bytes = CORPUS_SIZE * MESSAGE_LEN;
    let expansion = encoded_bytes as f64 / plain_bytes as f64;

    // Medir la latencia de decrypt (el patrón de `recall()`: N por turno).
    let start = Instant::now();
    for blob in &blobs {
        let recovered = vault.decrypt_with_key(&key, blob).expect("decrypt");
        std::hint::black_box(recovered.as_str());
    }
    let elapsed = start.elapsed();
    let per_decrypt_us = elapsed.as_micros() as f64 / CORPUS_SIZE as f64;

    println!("=== ConcatenatedFec performance baseline (REQ-V36) ===");
    println!("corpus:            {CORPUS_SIZE} records x {MESSAGE_LEN} B plaintext");
    println!("plaintext total:   {plain_bytes} B");
    println!("encoded total:     {encoded_bytes} B");
    println!("disk expansion:    {expansion:.2}x");
    println!(
        "decrypt total:     {:.1} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("decrypt per-record: {per_decrypt_us:.1} us");
    println!("(recall() with N memories/turn pays N x the per-record cost above)");
}
