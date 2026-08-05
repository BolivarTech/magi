// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-03

//! Order and cap of startup notices (Task 1.5).
//!
//! # Why it lives in the LIB and not under `system/`/`tui/`
//!
//! It is pure: no I/O, no network, no state. `main.rs` consumes it to assemble the list of
//! notices the TUI displays at startup.
//!
//! # Which SURFACE this applies to, and which it does NOT
//!
//! The tiering in this module is for notices **rendered to a human** in a startup list — today,
//! only the TUI. The headless path (`magi query`/`consult`) has its own output contract (the
//! JSON envelope and the run log, REQ-H23) and does not consume [`Notice`]: there is no startup
//! list there for a human to read, so assigning it a tier would be a representation that
//! nothing on that path needs. This is the correct boundary of the module, not a scope cut — if
//! headless ever gains a human-readable startup list, THAT is the moment to decide whether it
//! consumes this type.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::collections::HashSet;

/// Priority of a startup notice.
///
/// **The enum's declaration order IS the print order**: the `derive(Ord)` does not
/// is decorative — [`render_notices`] sorts with `sort_by_key(|n| n.tier)` and relies on
/// `Blocking < Resolution < Info` in that exact sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NoticeTier {
    /// Something the user asked for is NOT available. Action required.
    Blocking,
    /// The config resolved differently from what the file seems to say — or a low-level
    /// diagnostic that does not rise to blocking but is not noise either: hardening/vault
    /// (mlock, dump suppression), a failure to open or derive the vault key, or loss of
    /// persistence. None of these cases demand immediate action like `Blocking`, but all are
    /// surprising enough to always survive the [`NOTICE_MAX_INFO`] cap.
    Resolution,
    /// Diagnostic. Useful, never urgent.
    Info,
}

/// A startup notice, with the priority that decides its place in the final list.
///
/// **Every source pushes `Notice`, not `String`** — before this task, several sources in
/// `main.rs` pushed plain `String`s into a shared list while the tier design lived only in the
/// spec, so the order could not be applied to anything real.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    /// Priority — governs the print order and whether the [`render_notices`] cap can reach it.
    pub tier: NoticeTier,
    /// Text to display, already formatted by whoever built it.
    pub text: String,
}

impl Notice {
    /// Builds a `Blocking` notice: something the user asked for is not available.
    pub fn blocking(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Blocking,
            text: text.into(),
        }
    }

    /// Builds a `Resolution` notice: the config resolved differently from what was written.
    pub fn resolution(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Resolution,
            text: text.into(),
        }
    }

    /// Builds an `Info` notice: diagnostic, never urgent.
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Info,
            text: text.into(),
        }
    }
}

/// How many `Info`s survive the [`render_notices`] cap.
///
/// 5: with ten possible sources, half a screen is what someone actually reads at startup. It is
/// not a measurement — it is the same kind of hand-picked number as the complexity-gate
/// thresholds (REQ-A20), and it is stated so as not to pretend otherwise.
pub const NOTICE_MAX_INFO: usize = 5;

/// Sorts by tier (`Blocking` first), deduplicates by exact text, and trims only the `Info`s
/// that exceed [`NOTICE_MAX_INFO`].
///
/// # Contrato
/// - **Order**: `Blocking` → `Resolution` → `Info`. The `sort_by_key` is stable, so two notices of the same tier keep the order in which they were passed.
/// - **Dedup**: two notices with the same `text` collapse into one — the trio can emit the same `base_url` normalization warning three times (once per seat), and the user does not need to read it three times. It is applied AFTER sorting, so the first appearance in tier order survives.
/// - **Cap**: `Blocking` and `Resolution` are NEVER trimmed — the cap exists for the diagnostic noise, not for actionable or surprising items. When it trims, the last line of the result says how many `Info`s were omitted.
///
/// Complexity: `O(n log n)` for the sort plus `O(n)` for the dedup (a `HashSet` of already-seen
/// texts) — acceptable because `n` is the number of notices from ONE startup (a handful of
/// sources, never thousands).
pub fn render_notices(notices: Vec<Notice>) -> Vec<String> {
    let mut sorted = notices;
    sorted.sort_by_key(|n| n.tier);

    let mut seen_text = HashSet::with_capacity(sorted.len());
    let deduped = sorted
        .into_iter()
        .filter(|n| seen_text.insert(n.text.clone()));

    let mut info_seen = 0usize;
    let mut dropped = 0usize;
    let mut out = Vec::new();
    for n in deduped {
        if n.tier == NoticeTier::Info {
            info_seen += 1;
            if info_seen > NOTICE_MAX_INFO {
                dropped += 1;
                continue;
            }
        }
        out.push(n.text);
    }
    if dropped > 0 {
        out.push(format!("… {dropped} more diagnostic notice(s) omitted"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actionable items first, regardless of the order in which they were discovered.
    #[test]
    fn notices_are_ordered_by_tier_not_by_discovery() {
        let out = render_notices(vec![
            Notice::info("ventana medida: 128k"),
            Notice::blocking("el trío no es construible: falta OPENAI_API_KEY"),
            Notice::resolution("`[embedding].base_url` heredó la raíz"),
        ]);
        assert!(
            out[0].contains("no es construible"),
            "primero lo que exige acción"
        );
        assert!(out[1].contains("heredó"));
        assert!(out[2].contains("ventana medida"));
    }

    /// The cap trims NOISE, never signals.
    #[test]
    fn the_cap_truncates_info_only_and_says_how_many_it_dropped() {
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO + 3)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        v.push(Notice::blocking("b1"));
        v.push(Notice::resolution("r1"));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains("b1")),
            "Blocking NUNCA se recorta"
        );
        assert!(out.iter().any(|n| n.contains("r1")), "Resolution tampoco");
        assert_eq!(
            out.iter().filter(|n| n.starts_with('d')).count(),
            NOTICE_MAX_INFO
        );
        assert!(out.last().unwrap().contains('3'), "dice cuántos omitió");
    }

    /// Two sources can produce the SAME warning: the three seats with the same `base_url` emit
    /// the `/v1` normalization.
    #[test]
    fn identical_notices_are_emitted_once() {
        let n = "notice: `base_url` de Ollama sin sufijo `/v1`";
        let out = render_notices(vec![
            Notice::resolution(n),
            Notice::resolution(n),
            Notice::resolution(n),
        ]);
        assert_eq!(out.len(), 1, "tres asientos, un aviso");
    }

    /// Empty edge case (B13): nothing to sort, deduplicate, or trim — never panics, and with no
    /// `Info` to trim there is no "omitted" line.
    #[test]
    fn empty_input_renders_to_an_empty_list() {
        let out = render_notices(vec![]);
        assert!(out.is_empty());
    }

    /// Exact cap boundary: `info_seen > NOTICE_MAX_INFO` is strict, so exactly
    /// `NOTICE_MAX_INFO` `Info` notices do not trigger ANY trimming. Only the above-the-cap
    /// case was covered before this test; the off-by-one at the boundary is the classic defect
    /// of this kind of guard.
    #[test]
    fn exactly_the_cap_worth_of_info_drops_nothing() {
        let v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        let out = render_notices(v);
        assert_eq!(
            out.len(),
            NOTICE_MAX_INFO,
            "ninguno se recorta en la frontera exacta"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "sin recorte no hay línea de omitidos: {out:?}"
        );
    }

    /// The signal-vs-noise property the module exists to guarantee: same text, different tiers
    /// — the more severe one (`Blocking`) survives, not the `Info`.
    ///
    /// With IDENTICAL text, which one survived cannot be read directly from the output `String`
    /// (they are the same string). It is tested by its EFFECT on the cap: exactly
    /// `NOTICE_MAX_INFO` distinct filler `Info`s are added, which on their own do not trigger
    /// any trimming (see the exact boundary test above). If the duplicate survived as `Info`
    /// instead of `Blocking`, it would add one more `Info` and WOULD trigger the trim. That it
    /// does not trigger it, and that the duplicate text is still present, is the proof that the
    /// `Blocking` survived — which never counts against the cap.
    #[test]
    fn cross_tier_duplicate_text_keeps_the_more_severe_tier() {
        let dup_text = "el trío no es construible: falta OPENAI_API_KEY";
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("filler{i}")))
            .collect();
        v.push(Notice::info(dup_text));
        v.push(Notice::blocking(dup_text));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains(dup_text)),
            "el duplicado debe sobrevivir (bajo el tier Blocking): {out:?}"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "si sobreviviera el Info, se pasaría del tope y algo se recortaría: {out:?}"
        );
    }
}
