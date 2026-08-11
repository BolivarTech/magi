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

use serde::Deserialize;

use crate::magi::lineage::Lineage;

/// One fallback rotation candidate, deserialized from a `[[magi.fallback]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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

/// Unit tests for `FallbackEntry` deserialization.
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
}
