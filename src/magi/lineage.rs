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

use std::fmt;

use thiserror::Error;

/// A user-chosen independent failure domain for an LLM model.
///
/// Deliberately opaque: this type does not validate the label against any vendor registry or model
/// family list. It stores a trimmed, non-blank string and nothing else.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lineage(String);

impl Lineage {
    /// Parses a raw lineage label.
    ///
    /// # Errors
    ///
    /// Returns [`LineageError::Missing`] when `raw` holds no non-whitespace characters.
    pub fn parse(_raw: &str) -> Result<Self, LineageError> {
        // STUB (Red phase): no real logic yet. The tests below must fail on their assertions,
        // not on a compile error.
        Ok(Self(String::new()))
    }

    /// Returns the trimmed lineage label.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
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
    #[error("missing lineage value")]
    Missing {
        /// Configuration key that produced the missing value, when the caller knows it.
        key: &'static str,
    },
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
}
