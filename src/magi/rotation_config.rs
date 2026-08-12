// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-11

//! Shape of the rotation configuration: the fallback candidates declared in `magi.toml`.
//!
//! The pool is **shared by the three seats**, not per-seat. Per-seat lists would triple what has to
//! be declared for the same guarantee and would add a failure mode the shared pool does not have —
//! one list drying up while the other two still hold candidates.

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

use magi_core::schema::AgentName;
use serde::Deserialize;

use crate::magi::lineage::{seats_without_coverage, Lineage};
use crate::notices::Notice;

/// One fallback rotation candidate, deserialized from a `[[magi.fallback]]` entry.
///
/// `deny_unknown_fields` is not decoration: a typo in a pool entry must be a parse error naming the
/// key, never a silently ignored one. An entry that looks configured and is not is exactly the kind
/// of safety net an operator believes they have and does not.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackEntry {
    /// Model tag to rotate to.
    pub model: String,

    /// Independent failure domain this candidate belongs to, chosen by the user.
    #[serde(deserialize_with = "deserialize_lineage")]
    pub lineage: Lineage,
}

/// Deserializes a [`Lineage`] by routing through [`Lineage::parse`].
///
/// **Why not derive `Deserialize` on `Lineage`:** the blank-is-absent rule lives inside `parse`, and
/// a derived implementation would bypass it — letting `lineage = "  "` through as a valid empty
/// lineage. Routing through `parse` turns a blank value into a serde error that names the field.
///
/// # Errors
///
/// Returns a serde error when the value is blank or whitespace-only.
fn deserialize_lineage<'de, D>(deserializer: D) -> Result<Lineage, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Lineage::parse(&raw).map_err(serde::de::Error::custom)
}

/// Formats a slice of seat names as a lowercase, comma-separated list for user-facing messages.
fn seat_names(seats: &[AgentName]) -> String {
    seats
        .iter()
        .map(|name| name.display_name().to_lowercase())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Diversity problem detected while validating seat lineages against the fallback pool.
#[derive(Debug, thiserror::Error)]
pub enum DiversityError {
    /// Two or more seats share a lineage, defeating the ensemble even when no rotation occurs.
    #[error(
        "the seats {} share lineage '{}'; every mage must run a different lineage. \
         Declare finer lineage labels by model family, or set enforce_diversity to false.",
        seat_names(seats),
        lineage.as_str()
    )]
    NonDistinctSeats {
        /// Seat names that share the duplicated lineage.
        seats: Vec<AgentName>,
        /// The lineage shared by those seats.
        lineage: Lineage,
    },

    /// One or more seats have no fallback candidate from a different lineage in the pool.
    #[error(
        "the seats {} have no fallback coverage; every seat must be rotatable to a different lineage. \
         Add fallback entries covering these seats, declare finer lineage labels by model family, \
         or set enforce_diversity to false.",
        seat_names(seats)
    )]
    UncoveredSeats {
        /// Seat names without fallback coverage.
        seats: Vec<AgentName>,
    },
}

/// Validates that the configured seats are lineage-diverse and covered by the fallback pool.
///
/// Distinctness of the seat lineages is required unconditionally. Coverage is evaluated only when
/// the pool is non-empty, because an empty pool explicitly means "no rotation" and must not
/// produce warnings. When `enforce` is false, coverage gaps become notices instead of errors.
///
/// # Errors
///
/// Returns [`DiversityError`] when `enforce` is true and a rule is violated.
pub fn validate_diversity(
    seats: &[(AgentName, Lineage)],
    pool: &[FallbackEntry],
    enforce: bool,
) -> Result<Vec<Notice>, DiversityError> {
    let mut notices = Vec::new();

    // Distinctness is required regardless of whether a pool exists or rotation is enabled:
    // three mages in the same lineage provide no ensemble benefit.
    let mut by_lineage: Vec<(Lineage, Vec<AgentName>)> = Vec::new();
    for (name, lineage) in seats {
        match by_lineage.iter_mut().find(|(l, _)| l == lineage) {
            Some((_, names)) => names.push(*name),
            None => by_lineage.push((lineage.clone(), vec![*name])),
        }
    }
    for (lineage, names) in by_lineage {
        // Not enforced ⇒ a shared lineage is not even reported. Coverage notices are the only soft
        // signal, so the operator who deliberately relaxed this setting is not nagged about it.
        if names.len() > 1 && enforce {
            return Err(DiversityError::NonDistinctSeats {
                seats: names,
                lineage,
            });
        }
    }

    // Coverage only matters when there is a pool to rotate into.
    if !pool.is_empty() {
        let uncovered = seats_without_coverage(seats, pool);
        if !uncovered.is_empty() {
            if enforce {
                return Err(DiversityError::UncoveredSeats { seats: uncovered });
            }
            notices.push(Notice::resolution(format!(
                "the seats {} have no fallback coverage; every seat must be rotatable to a \
                 different lineage. Add fallback entries covering these seats, declare finer \
                 lineage labels by model family, or set enforce_diversity to false.",
                seat_names(&uncovered)
            )));
        }
    }

    Ok(notices)
}

/// Corroborates the declared lineages against the **cached** weights digests (REQ-R29, SC-R45).
///
/// # Why this warns and the declarative half errors
///
/// The dividing line is the **nature of the evidence**, not the setting. Three distinct seat
/// lineages and pool coverage are *declarative*: free to check, and **true right now**, so they are
/// required and a violation is a load error. A shared digest is *empirical* corroboration, and the
/// evidence can have aged — the cache only changes when the configuration does, so a stored digest
/// may be weeks old. Denying a start over a collision that might no longer exist is a worse failure
/// than the one it prevents.
///
/// # It never probes
///
/// Everything here is read from what a measurement already stored. `ProviderProbe::digest` is
/// per-instance rather than per-model, so re-verifying would cost one request **per model on every
/// start** — and buy a check that is secondary and advisory. The digest is measured on the first
/// trip and never re-read (SC-R42), at either value of `enforce_diversity`.
///
/// # Unresolved digests are reported, not passed over in silence
///
/// A `None` digest cannot prove or disprove anything, and saying nothing would read as
/// *"corroborated and fine"* — a different claim from *"there was nothing to corroborate with"*.
/// The same rule the rest of this milestone follows: a resolution that does not come from what the
/// operator wrote gets announced.
///
/// # Complexity
///
/// `O(n²)` over the declared models, with `n` bounded by the three seats plus the pool — a handful
/// of entries read once at startup. A map keyed by digest would be `O(n)` and less legible for a
/// gain that does not exist at this size.
#[must_use]
pub fn corroborate_by_digest(entries: &[(String, Lineage, Option<String>)]) -> Vec<Notice> {
    let mut notices = Vec::new();

    for (i, (model_a, lineage_a, digest_a)) in entries.iter().enumerate() {
        let Some(digest_a) = digest_a else { continue };
        for (model_b, lineage_b, digest_b) in entries.iter().skip(i + 1) {
            let Some(digest_b) = digest_b else { continue };
            // Same lineage sharing a digest is the declaration being ACCURATE: nothing to report.
            if digest_a == digest_b && lineage_a != lineage_b {
                notices.push(Notice::info(format!(
                    "notice: `{model_a}` (lineage `{lineage_a}`) and `{model_b}` (lineage                      `{lineage_b}`) are declared as different failure domains but share the same                      cached weights digest, so rotating between them may buy no diversity. The                      digest is the one recorded when they were measured and is not re-checked, so                      this is a hint rather than a finding."
                )));
            }
        }
    }

    if entries.iter().any(|(_, _, digest)| digest.is_none()) {
        notices.push(Notice::info(
            "notice: lineage diversity was not corroborated against weights digests for every              declared model, because some digests are unresolved. The declarative checks still              applied."
                .to_owned(),
        ));
    }

    notices
}

/// Unit tests for `FallbackEntry` deserialization and diversity validation.
#[cfg(test)]
mod tests {
    use super::*;

    /// A complete fallback entry parses successfully.
    #[test]
    fn a_complete_entry_parses() {
        let e: FallbackEntry = toml::from_str("model = \"glm-5.2:cloud\"\nlineage = \"zhipu\"\n")
            .expect("a complete entry must parse");
        assert_eq!(e.model, "glm-5.2:cloud");
        assert_eq!(e.lineage.as_str(), "zhipu");
    }

    /// A fallback without its lineage is a parse error: the lineage is never inferred from the
    /// model name, because inferring it would fabricate the label that decides all rotation
    /// eligibility.
    #[test]
    fn an_entry_without_lineage_is_a_parse_error() {
        let err = toml::from_str::<FallbackEntry>("model = \"glm-5.2:cloud\"\n")
            .expect_err("a fallback without lineage must not parse");
        assert!(
            err.to_string().contains("lineage"),
            "the error must name the field: {err}"
        );
    }

    /// Blank must not slip through as an empty lineage — the rule lives in `Lineage::parse` and the
    /// field is routed through it precisely so a derive cannot bypass it.
    #[test]
    fn a_blank_lineage_is_a_parse_error_not_an_empty_lineage() {
        let err = toml::from_str::<FallbackEntry>("model = \"glm-5.2:cloud\"\nlineage = \"   \"\n")
            .expect_err("blank must not produce an empty Lineage");
        assert!(
            err.to_string().contains("lineage"),
            "the error must name the field: {err}"
        );
    }

    /// A typo in a pool entry is a parse error, not a silently ignored key.
    #[test]
    fn an_unknown_key_is_rejected_by_name() {
        let err = toml::from_str::<FallbackEntry>(
            "model = \"glm-5.2:cloud\"\nlineage = \"zhipu\"\nweight = 3\n",
        )
        .expect_err("an unknown key must be rejected");
        assert!(
            err.to_string().contains("weight"),
            "the error must name the key: {err}"
        );
    }

    /// Builds a seat triple from string identifiers for tests.
    fn trio(entries: &[(&str, &str)]) -> Vec<(AgentName, Lineage)> {
        entries
            .iter()
            .map(|(name, lineage)| {
                let seat = match *name {
                    "melchior" => AgentName::Melchior,
                    "balthasar" => AgentName::Balthasar,
                    _ => AgentName::Caspar,
                };
                (seat, Lineage::parse(lineage).expect("valid lineage"))
            })
            .collect()
    }

    /// Builds a fallback pool from string identifiers for tests.
    fn pool_of(entries: &[(&str, &str)]) -> Vec<FallbackEntry> {
        entries
            .iter()
            .map(|(model, lineage)| FallbackEntry {
                model: (*model).to_owned(),
                lineage: Lineage::parse(lineage).expect("valid lineage"),
            })
            .collect()
    }

    /// Three seats sharing a lineage are rejected even without a pool, because the ensemble is
    /// degraded regardless of whether rotation ever happens.
    #[test]
    fn three_seats_of_the_same_lineage_are_a_load_error_even_without_a_pool() {
        let seats = trio(&[
            ("melchior", "anthropic"),
            ("balthasar", "anthropic"),
            ("caspar", "anthropic"),
        ]);
        let err =
            validate_diversity(&seats, &[], true).expect_err("same lineage must fail under true");
        let msg = err.to_string();
        assert!(
            msg.contains("melchior") && msg.contains("balthasar") && msg.contains("caspar"),
            "the error must name the affected seats: {msg}"
        );
        assert!(
            msg.contains("enforce_diversity"),
            "the error must carry the ONE-LINE way out: {msg}"
        );
    }

    /// Three seats with distinct lineages and no pool start cleanly: no pool means no coverage
    /// question.
    #[test]
    fn a_distinct_lineage_trio_without_a_pool_starts_with_no_notice() {
        let seats = trio(&[
            ("melchior", "opus"),
            ("balthasar", "sonnet"),
            ("caspar", "haiku"),
        ]);
        let notices = validate_diversity(&seats, &[], true).expect("distinct lineages must pass");
        assert!(notices.is_empty(), "no pool means no coverage question");
    }

    /// A coverage gap is an error under enforcement and a single notice under lenient mode.
    #[test]
    fn a_coverage_gap_is_an_error_under_true_and_a_notice_under_false() {
        let seats = trio(&[
            ("melchior", "opus"),
            ("balthasar", "sonnet"),
            ("caspar", "haiku"),
        ]);
        let pool = pool_of(&[("m2", "opus")]);
        assert!(validate_diversity(&seats, &pool, true).is_err());
        let notices = validate_diversity(&seats, &pool, false).expect("false must not error");
        assert_eq!(notices.len(), 1, "all uncovered seats in ONE message");
        assert!(
            notices[0].text.contains("balthasar") && notices[0].text.contains("caspar"),
            "notice must name the uncovered seats: {}",
            notices[0].text
        );
    }

    /// With enforcement disabled, no configuration produces an error.
    #[test]
    fn enforce_false_never_errors_even_on_a_single_lineage_trio() {
        let seats = trio(&[("melchior", "x"), ("balthasar", "x"), ("caspar", "x")]);
        assert!(validate_diversity(&seats, &pool_of(&[("m", "x")]), false).is_ok());
    }

    // ---------------------------------------------------------------------------------------
    // Digest corroboration — REQ-R29 (second half) / SC-R45 (Task 6.7).

    /// A digest of the right shape: 64 lowercase hex.
    const DIGEST: &str = "d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2";

    /// Builds `(model, lineage, digest)` rows the way the cache hands them over.
    fn rows(entries: &[(&str, &str, Option<&str>)]) -> Vec<(String, Lineage, Option<String>)> {
        entries
            .iter()
            .map(|(model, lineage, digest)| {
                (
                    (*model).to_owned(),
                    Lineage::parse(lineage).expect("valid lineage"),
                    digest.map(str::to_owned),
                )
            })
            .collect()
    }

    /// SC-R45: two models of DECLARED DISTINCT lineages sharing a cached digest produce a WARNING
    /// naming both, and the process STARTS.
    ///
    /// Never an error, and the reason is the nature of the evidence: the cached digest can be
    /// weeks old — the cache only changes when the configuration does — so refusing to start over
    /// a collision that may no longer exist is a worse failure than the one it prevents.
    #[test]
    fn two_distinct_lineages_with_the_same_digest_warn_and_start() {
        let notices = corroborate_by_digest(&rows(&[
            ("a", "zhipu", Some(DIGEST)),
            ("b", "minimax", Some(DIGEST)),
        ]));
        assert_eq!(notices.len(), 1, "one message naming the pair");
        let t = &notices[0].text;
        assert!(t.contains('a') && t.contains('b'), "both models named: {t}");
        assert!(
            t.contains("zhipu") && t.contains("minimax"),
            "and both declared lineages, or the operator cannot act: {t}"
        );
    }

    /// Two models of the SAME declared lineage sharing a digest is not a contradiction — it is the
    /// declaration being accurate. Nothing to report.
    #[test]
    fn the_same_lineage_sharing_a_digest_is_not_reported() {
        let notices = corroborate_by_digest(&rows(&[
            ("a", "zhipu", Some(DIGEST)),
            ("b", "zhipu", Some(DIGEST)),
        ]));
        assert!(
            notices.is_empty(),
            "the declaration agreed with the evidence"
        );
    }

    /// With UNRESOLVED digests the corroboration cannot happen, and a notice says so.
    ///
    /// MS2's rule: any resolution that does not come from what the user wrote is announced.
    /// Silence here would read as "corroborated and fine", which is a different claim from
    /// "there was nothing to corroborate with".
    #[test]
    fn unresolved_digests_skip_corroboration_and_say_so() {
        let notices = corroborate_by_digest(&rows(&[("a", "zhipu", None), ("b", "minimax", None)]));
        assert!(
            notices.iter().any(|n| n.text.contains("not corroborated")),
            "the absence of evidence must be stated, not implied: {notices:?}"
        );
    }

    /// A single unresolved digest among resolved ones does not silence the check for the rest.
    #[test]
    fn one_unresolved_digest_does_not_silence_the_others() {
        let notices = corroborate_by_digest(&rows(&[
            ("a", "zhipu", Some(DIGEST)),
            ("b", "minimax", Some(DIGEST)),
            ("c", "google", None),
        ]));
        assert!(
            notices
                .iter()
                .any(|n| n.text.contains('a') && n.text.contains('b')),
            "the resolved pair is still corroborated: {notices:?}"
        );
    }

    /// Distinct digests are exactly what the declaration promised: silence.
    #[test]
    fn distinct_digests_are_silent() {
        let notices = corroborate_by_digest(&rows(&[
            ("a", "zhipu", Some(DIGEST)),
            ("b", "minimax", Some(&DIGEST.replace('d', "e"))),
        ]));
        assert!(notices.is_empty());
    }
}
