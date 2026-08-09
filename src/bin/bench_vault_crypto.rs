// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-15 Performance baseline of
// `ConcatenatedFec` (REQ-V36).
//!
//! Measures the **decrypt cost** and **disk expansion** of the `AES-256-GCM-SIV +
//! ConcatenatedFec` pipeline in the `cryptovault` crate, over a representative corpus of
//! messages. It is **performance history**, NOT a gate: the FEC choice is fixed by design
//! (D-V02); this number exists to compare in future releases, not to reopen the decision.
//!
//! Usage: `cargo run --release --bin bench_vault_crypto`

use std::time::Instant;

/// Number of records in the synthetic corpus.
const CORPUS_SIZE: usize = 1_000;
/// Representative length of a conversation message (bytes).
const MESSAGE_LEN: usize = 220;

fn main() {
    let vault = cryptovault::CryptoVault::default();
    let salt = cryptovault::generate_salt().expect("salt");
    let key = vault
        .derive_key("bench-master-key-string", &salt)
        .expect("key");

    let plaintext: String = "x".repeat(MESSAGE_LEN);

    // Encrypt the corpus and measure the disk expansion.
    let mut blobs = Vec::with_capacity(CORPUS_SIZE);
    let mut encoded_bytes = 0usize;
    for _ in 0..CORPUS_SIZE {
        let blob = vault.encrypt_with_key(&key, &plaintext).expect("encrypt");
        encoded_bytes += blob.len();
        blobs.push(blob);
    }
    let plain_bytes = CORPUS_SIZE * MESSAGE_LEN;
    let expansion = encoded_bytes as f64 / plain_bytes as f64;

    // Measure decrypt latency (the `recall()` pattern: N per turn).
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
