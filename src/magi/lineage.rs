// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-11

//! Lineage: a user-chosen independent failure domain for an LLM model.
//!
//! A lineage is **not** "the vendor". The vendor is a good proxy when you have several of them, and
//! the model family is one when you only have a single vendor. `magi-core` treats the label as an
//! opaque, trimmed string and never validates it against a registry — and neither does this module.

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

use std::collections::HashMap;
use std::fmt;

use magi_core::schema::AgentName;
use thiserror::Error;

use crate::magi::rotation_config::FallbackEntry;

/// Sentinel key used by [`Lineage::parse`], which cannot know which configuration key produced a
/// blank value: it is a value parser, not a config reader. Callers that know the key remap the
/// error with the real name.
const UNKNOWN_MISSING_KEY: &str = "";

/// Separator placed between the base message and the configuration key, when the key is known.
const MISSING_KEY_PREFIX: &str = " for key: ";

/// A user-chosen independent failure domain for an LLM model.
///
/// Deliberately opaque: this type does not validate the label against any vendor registry or model
/// family list. It stores a trimmed, non-blank string and nothing else.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lineage(String);

impl Lineage {
    /// Parses a raw lineage label.
    ///
    /// Surrounding whitespace is trimmed. A blank or whitespace-only value is treated as **absent**
    /// and reported as [`LineageError::Missing`] — never as an invalid value. There is no
    /// allow-list: any non-blank label is accepted, because the lineage is a semantic choice of the
    /// user and validating it against a vendor registry would invent a decision that is not ours.
    ///
    /// # Errors
    ///
    /// Returns [`LineageError::Missing`] when `raw` holds no non-whitespace characters. The error
    /// carries the unknown-key sentinel: a value parser does not know which key was blank, so a
    /// caller that does should remap it with the real key name.
    pub fn parse(raw: &str) -> Result<Self, LineageError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            Err(LineageError::Missing {
                key: UNKNOWN_MISSING_KEY,
            })
        } else {
            Ok(Self(trimmed.to_owned()))
        }
    }

    /// Returns the trimmed lineage label.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<Lineage> for magi_core::rotation::Lineage {
    /// Hands the validated label to magi-core, which owns the type the builder registers.
    ///
    /// The two types exist for different reasons and neither replaces the other: magi-core's is
    /// **infallible** — `Lineage::new("")` is a valid *value*, rejected later at
    /// `MagiBuilder::build()` — while this crate's rejects blank at construction, so a missing
    /// declaration is caught while the configuration key that produced it is still known. The
    /// conversion goes one way on purpose: a label that reached magi-core has already passed here.
    fn from(lineage: Lineage) -> Self {
        magi_core::rotation::Lineage::new(lineage.0)
    }
}

impl fmt::Display for Lineage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Errors produced when parsing a [`Lineage`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LineageError {
    /// The lineage value was absent — blank or whitespace-only.
    #[error("missing lineage value{}", key_suffix(.key))]
    Missing {
        /// Configuration key that produced the missing value, when the caller knows it.
        ///
        /// [`Lineage::parse`] uses the unknown-key sentinel because a value parser does not know
        /// which key was blank; a caller that does should construct or remap this variant with the
        /// real key name.
        key: &'static str,
    },
}

/// Renders the key suffix of [`LineageError::Missing`], or nothing when the key is unknown.
///
/// Keeps the message sensible for the parser, which cannot name a key: `"missing lineage value"`
/// reads correctly on its own, and `"missing lineage value for key: melchior_lineage"` reads
/// correctly once a caller supplies one.
fn key_suffix(key: &str) -> String {
    if key.is_empty() {
        String::new()
    } else {
        format!("{MISSING_KEY_PREFIX}{key}")
    }
}

/// Seats that no pool entry can serve.
///
/// A seat `S` is covered **iff** the pool holds an entry whose lineage is `lineage(S)` **and no
/// other seat holds that same lineage**, or an entry with a lineage **no seat** has.
///
/// # Why the uniqueness clause is not optional
///
/// A rotating seat releases its own lineage, but the *other two* keep theirs in play. So if two
/// seats share a lineage and one of them rotates, a candidate of that lineage is still blocked —
/// by the seat that did not rotate. Dropping the clause makes `seats_without_coverage` claim
/// coverage that rotation will refuse at the moment it matters.
pub fn seats_without_coverage(
    seats: &[(AgentName, Lineage)],
    pool: &[FallbackEntry],
) -> Vec<AgentName> {
    // An empty pool is "no rotation", a legitimate choice — not a coverage gap.
    if pool.is_empty() {
        return Vec::new();
    }

    let mut seats_per_lineage: HashMap<&str, usize> = HashMap::new();
    for (_, lineage) in seats {
        *seats_per_lineage.entry(lineage.as_str()).or_insert(0) += 1;
    }

    seats
        .iter()
        .filter(|(_, lineage)| {
            let covered = pool.iter().any(|entry| {
                let label = entry.lineage.as_str();
                let foreign = !seats_per_lineage.contains_key(label);
                let own_and_unique =
                    label == lineage.as_str() && seats_per_lineage.get(label) == Some(&1);
                foreign || own_and_unique
            });
            !covered
        })
        .map(|(seat, _)| *seat)
        .collect()
}

/// Unit tests for the [`Lineage`] value type.
#[cfg(test)]
mod tests {
    use super::*;

    /// Blank strings must be reported as missing, never as invalid.
    ///
    /// An exported-but-unfilled environment variable in a CI script is an everyday accident;
    /// breaking startup over it punishes the accident instead of falling through.
    #[test]
    fn a_blank_lineage_is_treated_as_absent_not_as_invalid() {
        for blank in ["", "   ", "\t", "\n  "] {
            let err = Lineage::parse(blank).expect_err("blank must error as missing");
            assert!(
                matches!(err, LineageError::Missing { .. }),
                "blank must be MISSING, not invalid: {err:?}"
            );
        }
    }

    /// Any non-blank label is accepted. There is no allow-list, by design: the lineage is a
    /// semantic choice of the user about what counts as an independent failure domain, and
    /// validating it against a vendor registry would invent a decision that is not ours.
    #[test]
    fn any_non_blank_label_is_a_valid_lineage() {
        for label in ["alibaba", "opus", "tier-1", "mi-linaje-raro", "z"] {
            let l =
                Lineage::parse(label).unwrap_or_else(|e| panic!("{label} must parse, got {e:?}"));
            assert_eq!(l.as_str(), label);
        }
    }

    /// `parse` strips surrounding whitespace from the label.
    #[test]
    fn parse_trims_surrounding_whitespace() {
        assert_eq!(Lineage::parse("  alibaba  ").unwrap().as_str(), "alibaba");
    }

    /// Lineages order and compare by their trimmed string value.
    #[test]
    fn lineages_compare_and_order_by_their_string_value() {
        let a = Lineage::parse("alibaba").unwrap();
        let b = Lineage::parse("zhipu").unwrap();
        assert!(a < b);
        assert_eq!(a, Lineage::parse("alibaba").unwrap());
    }

    /// `Display` prints the trimmed inner label.
    #[test]
    fn display_prints_the_inner_value() {
        assert_eq!(Lineage::parse("openai").unwrap().to_string(), "openai");
    }

    fn trio(v: &[(&str, &str)]) -> Vec<(AgentName, Lineage)> {
        v.iter()
            .map(|(n, l)| {
                let seat = match *n {
                    "melchior" => AgentName::Melchior,
                    "balthasar" => AgentName::Balthasar,
                    _ => AgentName::Caspar,
                };
                (seat, Lineage::parse(l).unwrap())
            })
            .collect()
    }

    fn pool_of(v: &[(&str, &str)]) -> Vec<FallbackEntry> {
        v.iter()
            .map(|(m, l)| FallbackEntry {
                model: (*m).to_string(),
                lineage: Lineage::parse(l).unwrap(),
            })
            .collect()
    }

    /// An entry sharing a seat's lineage covers ONLY that seat: the rotating seat releases its own
    /// lineage, so the entry is blocked for the other two.
    #[test]
    fn a_pool_entry_sharing_a_seats_lineage_covers_exactly_that_seat() {
        let seats = trio(&[
            ("melchior", "opus"),
            ("balthasar", "sonnet"),
            ("caspar", "haiku"),
        ]);
        let pool = pool_of(&[("m2", "opus")]);
        assert_eq!(
            seats_without_coverage(&seats, &pool),
            vec![AgentName::Balthasar, AgentName::Caspar],
            "only Melchior is covered by an `opus` entry"
        );
    }

    /// An entry with a label no seat has covers all three.
    #[test]
    fn a_pool_entry_with_a_foreign_lineage_covers_all_three_seats() {
        let seats = trio(&[
            ("melchior", "opus"),
            ("balthasar", "sonnet"),
            ("caspar", "haiku"),
        ]);
        assert!(seats_without_coverage(&seats, &pool_of(&[("m2", "zhipu")])).is_empty());
    }

    /// An empty pool is silence, not a gap: that is "no rotation", a choice.
    #[test]
    fn an_empty_pool_reports_no_uncovered_seats() {
        let seats = trio(&[
            ("melchior", "opus"),
            ("balthasar", "sonnet"),
            ("caspar", "haiku"),
        ]);
        assert!(seats_without_coverage(&seats, &[]).is_empty());
    }

    /// THE CASE THE UNIQUENESS CLAUSE EXISTS FOR. With all three seats on one lineage, a candidate
    /// of that same lineage covers NOBODY — whichever seat rotates, the other two still hold the
    /// lineage in play. Only a foreign label covers them.
    #[test]
    fn a_single_lineage_trio_is_covered_only_by_a_foreign_label() {
        let seats = trio(&[
            ("melchior", "anthropic"),
            ("balthasar", "anthropic"),
            ("caspar", "anthropic"),
        ]);
        assert_eq!(
            seats_without_coverage(&seats, &pool_of(&[("m2", "anthropic")])).len(),
            3
        );
        assert!(seats_without_coverage(&seats, &pool_of(&[("m2", "zhipu")])).is_empty());
    }
}
