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
    /// carries [`UNKNOWN_MISSING_KEY`]: a value parser does not know which key was blank, so a
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
        /// [`Lineage::parse`] uses [`UNKNOWN_MISSING_KEY`] because a value parser does not know
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
