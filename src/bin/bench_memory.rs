// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Two-arm benchmark binary for the tiered memory subsystem (T14, D-08, REQ-24').
//!
//! Runs both `load_all` and `selective` arms against the synthetic evaluation
//! dataset, computes recall accuracy, staleness rate, and mean context tokens,
//! and writes a machine-readable `report.json` to `target/bench_memory_report.json`.
//!
//! This binary is **self-contained**: it re-implements the minimal primitives
//! needed (DeterministicEmbedder, brute-force cosine retrieval) without the
//! encrypted SQLite store so that it can run without the full crate module tree.
//! The authoritative algorithmic tests live in `cargo nextest run memory::bench::tests`.
//!
//! # Usage
//! ```sh
//! cargo run --bin bench_memory
//! ```

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;

// ─── Dataset ──────────────────────────────────────────────────────────────────

/// A planted fact in the benchmark dataset (REQ-23').
struct Fact {
    id: &'static str,
    text: &'static str,
    /// ID of the fact that this one supersedes, if any (REQ-10).
    supersedes: Option<&'static str>,
}

/// A probe question for one benchmark query.
struct Probe {
    id: &'static str,
    query: &'static str,
    /// Fact IDs that must appear in the context for a recall hit.
    expected: &'static [&'static str],
    /// Fact IDs that are stale (superseded). Their presence is a staleness hit.
    stale: &'static [&'static str],
}

/// Synthetic benchmark dataset — mirrors `tests/data/memory_eval/dataset.json`
/// and `make_small_dataset()` in `bench.rs` (REQ-23', R-08, R-06).
static FACTS: &[Fact] = &[
    Fact {
        id: "pref_rust",
        text: "I prefer Rust over Python for systems programming",
        supersedes: None,
    },
    Fact {
        id: "pref_dark",
        text: "I use dark mode in all editors and terminals",
        supersedes: None,
    },
    Fact {
        id: "editor_old",
        text: "I use vim as my primary editor",
        supersedes: None,
    },
    Fact {
        id: "editor_new",
        text: "I switched from vim to neovim",
        supersedes: Some("editor_old"),
    },
    Fact {
        id: "distractor_1",
        text: "The capital of France is Paris",
        supersedes: None,
    },
    Fact {
        id: "distractor_2",
        text: "Water boils at one hundred degrees Celsius",
        supersedes: None,
    },
];

static PROBES: &[Probe] = &[
    Probe {
        id: "q_rust",
        query: "Rust Python systems programming",
        expected: &["pref_rust"],
        stale: &[],
    },
    Probe {
        id: "q_editor",
        query: "editor vim neovim",
        expected: &["editor_new"],
        stale: &["editor_old"],
    },
    Probe {
        id: "q_dark",
        query: "dark mode editors terminals",
        expected: &["pref_dark"],
        stale: &[],
    },
];

// ─── Deterministic embedder ───────────────────────────────────────────────────

const DIM: usize = 128;

/// Produces a normalised word-bag vector (SHA-256) without any network call
/// (mirrors `DeterministicEmbedder::word_bag` in `bench.rs`, CP2-O, R-06).
fn word_bag(text: &str) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    for word in text.split_whitespace() {
        let hash = Sha256::digest(word.to_lowercase().as_bytes());
        let idx =
            u64::from_le_bytes(hash[..8].try_into().expect("sha256 ≥ 8 bytes")) as usize % DIM;
        v[idx] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Cosine similarity between two equal-length slices (`[-1, 1]`).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Token accounting ─────────────────────────────────────────────────────────

/// Heuristic token count (`ceil(chars / chars_per_token)`, D-02/D-16).
fn estimate_tokens(text: &str, chars_per_token: f64) -> usize {
    ((text.len() as f64) / chars_per_token).ceil() as usize
}

// ─── Result types ─────────────────────────────────────────────────────────────

struct ProbeResult {
    probe_id: &'static str,
    /// IDs of facts whose text appears in the assembled context.
    retrieved_ids: Vec<&'static str>,
    context_tokens: usize,
}

#[derive(serde::Serialize)]
struct ArmReport {
    recall_accuracy: f64,
    staleness_rate: f64,
    mean_context_tokens: f64,
}

#[derive(serde::Serialize)]
struct BenchmarkReport {
    version: &'static str,
    seed: u64,
    load_all: ArmReport,
    selective: ArmReport,
}

// ─── Metric helpers ───────────────────────────────────────────────────────────

fn recall_accuracy(probes: &[Probe], results: &[ProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let hits = results
        .iter()
        .filter(|r| {
            let Some(p) = probes.iter().find(|p| p.id == r.probe_id) else {
                return false;
            };
            p.expected.iter().all(|e| r.retrieved_ids.contains(e))
        })
        .count();
    hits as f64 / results.len() as f64
}

fn staleness_rate(probes: &[Probe], results: &[ProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let stale = results
        .iter()
        .filter(|r| {
            let Some(p) = probes.iter().find(|p| p.id == r.probe_id) else {
                return false;
            };
            p.stale.iter().any(|s| r.retrieved_ids.contains(s))
        })
        .count();
    stale as f64 / results.len() as f64
}

fn mean_tokens(results: &[ProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let total: usize = results.iter().map(|r| r.context_tokens).sum();
    total as f64 / results.len() as f64
}

// ─── Benchmark arms ───────────────────────────────────────────────────────────

/// `load_all` arm: all facts injected into context per probe (v0.6.0 behaviour,
/// SC-19/REQ-15). The superseded `editor_old` is included — demonstrating
/// staleness (SC-29).
fn run_load_all(chars_per_token: f64) -> Vec<ProbeResult> {
    let all_text: String = FACTS.iter().map(|f| f.text).collect::<Vec<_>>().join(" ");
    let total_tokens = estimate_tokens(&all_text, chars_per_token);
    PROBES
        .iter()
        .map(|p| ProbeResult {
            probe_id: p.id,
            retrieved_ids: FACTS.iter().map(|f| f.id).collect(),
            context_tokens: total_tokens,
        })
        .collect()
}

/// `selective` arm: top-`k` active (non-superseded) memories recalled per probe
/// via brute-force cosine (D-05 deterministic, R-06). Only active facts are
/// candidates — superseded `editor_old` is excluded (SC-13 / SC-29).
fn run_selective(top_k: usize, chars_per_token: f64) -> Vec<ProbeResult> {
    // Build a set of superseded fact IDs.
    let superseded: std::collections::HashSet<&str> =
        FACTS.iter().filter_map(|f| f.supersedes).collect();

    // Embed only active facts.
    let active: Vec<(&Fact, [f32; DIM])> = FACTS
        .iter()
        .filter(|f| !superseded.contains(f.id))
        .map(|f| (f, word_bag(f.text)))
        .collect();

    PROBES
        .iter()
        .map(|p| {
            let q_vec = word_bag(p.query);

            // Score each active fact by cosine similarity.
            let mut scored: Vec<(f32, &Fact)> = active
                .iter()
                .map(|(f, v)| (cosine(&q_vec, v), *f))
                .collect();
            // Descending by cosine score (ties broken by stable sort order → deterministic).
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let top: Vec<&Fact> = scored.into_iter().take(top_k).map(|(_, f)| f).collect();
            let context_text: String = top.iter().map(|f| f.text).collect::<Vec<_>>().join(" ");
            let context_tokens = estimate_tokens(&context_text, chars_per_token);
            let retrieved_ids = top.iter().map(|f| f.id).collect();

            ProbeResult {
                probe_id: p.id,
                retrieved_ids,
                context_tokens,
            }
        })
        .collect()
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    const TOP_K: usize = 2;
    const CHARS_PER_TOKEN: f64 = 4.0;
    const SEED: u64 = 42;

    println!("bench_memory: tiered memory benchmark harness (T14, D-08)");
    println!(
        "Dataset   : {} facts, {} probes (seed={})",
        FACTS.len(),
        PROBES.len(),
        SEED
    );
    println!("Config    : top_k={TOP_K}, chars_per_token={CHARS_PER_TOKEN}");
    println!();

    // ── Run both arms ────────────────────────────────────────────────────────
    let la_results = run_load_all(CHARS_PER_TOKEN);
    let sel_results = run_selective(TOP_K, CHARS_PER_TOKEN);

    let la = ArmReport {
        recall_accuracy: recall_accuracy(PROBES, &la_results),
        staleness_rate: staleness_rate(PROBES, &la_results),
        mean_context_tokens: mean_tokens(&la_results),
    };
    let sel = ArmReport {
        recall_accuracy: recall_accuracy(PROBES, &sel_results),
        staleness_rate: staleness_rate(PROBES, &sel_results),
        mean_context_tokens: mean_tokens(&sel_results),
    };

    // ── SC-29 headline ────────────────────────────────────────────────────────
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             SC-29 HEADLINE — bench_memory results            ║");
    println!("╠══════════════════════════╦════════════╦══════════════════════╣");
    println!("║ Metric                   ║ load_all   ║ selective            ║");
    println!("╠══════════════════════════╬════════════╬══════════════════════╣");
    println!(
        "║ recall_accuracy          ║ {:.3}      ║ {:.3}               ║",
        la.recall_accuracy, sel.recall_accuracy
    );
    println!(
        "║ staleness_rate           ║ {:.3}      ║ {:.3}               ║",
        la.staleness_rate, sel.staleness_rate
    );
    println!(
        "║ mean_context_tokens      ║ {:<10.1} ║ {:<20.1} ║",
        la.mean_context_tokens, sel.mean_context_tokens
    );
    println!("╠══════════════════════════╩════════════╩══════════════════════╣");

    let token_fraction = if la.mean_context_tokens > 0.0 {
        sel.mean_context_tokens / la.mean_context_tokens
    } else {
        1.0
    };
    let selective_ge = sel.recall_accuracy >= la.recall_accuracy;
    let bounded_window = sel.mean_context_tokens < la.mean_context_tokens;
    let lower_staleness = sel.staleness_rate <= la.staleness_rate;

    println!(
        "║ selective recall ≥ load_all: {} (SC-29)                       ║",
        if selective_ge { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "║ token fraction: {:.1}% of load_all ({})                 ║",
        token_fraction * 100.0,
        if bounded_window {
            "bounded ✓"
        } else {
            "not bounded ✗"
        }
    );
    println!(
        "║ staleness delta: {:+.3} ({})                         ║",
        sel.staleness_rate - la.staleness_rate,
        if lower_staleness {
            "lower ✓"
        } else {
            "not lower ✗"
        }
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Write report.json ────────────────────────────────────────────────────
    let report = BenchmarkReport {
        version: "1.0",
        seed: SEED,
        load_all: la,
        selective: sel,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialization must not fail");

    let report_path = Path::new("target/bench_memory_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent).expect("cannot create target/ dir");
    }
    let mut f = fs::File::create(report_path).expect("cannot create report file");
    f.write_all(json.as_bytes()).expect("cannot write report");

    println!("report.json written to: {}", report_path.display());

    // ── Exit code ────────────────────────────────────────────────────────────
    // Non-zero exit if SC-29 headline fails (CI-friendly).
    if !selective_ge || !bounded_window || !lower_staleness {
        eprintln!("ERROR: SC-29 headline assertions failed — check the report.");
        std::process::exit(1);
    }
}
