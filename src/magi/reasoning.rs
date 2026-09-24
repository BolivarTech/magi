// Author: Julian Bolivar
// Version: 0.19.1
// Date: 2026-09-24

//! Vocabulary for `[magi].reasoning` and `[magi].reasoning_spelling` (REQ-EE-3, REQ-V42-4).
//!
//! # Why it lives in its own module
//!
//! Both keys name a value from a type magi-core owns ([`ReasoningControl`], `#[non_exhaustive]`)
//! or defines but does not serialize ([`ReasoningSpelling`], no `serde` impl at all). Parsing
//! either from a `magi.toml` string is domain vocabulary — the same class of thing
//! [`crate::magi::kind::ProviderKind`] already is — not the shape of the TOML struct that carries
//! it, which is why it does not live in `config.rs` (bin).

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

use magi_core::provider::ReasoningControl;
use magi_core::providers::openai_compat::ReasoningSpelling;

/// Accepted `[magi].reasoning_spelling` tags, in declaration order, for error messages and the
/// scaffold.
///
/// A `const` and not a repeated literal, for the same reason
/// [`crate::magi::kind::VALID_PROVIDER_KINDS`] is: the error message, the scaffold and any future
/// documentation must name the same set, and a fourth hand-copied list is a fourth place to
/// forget when [`ReasoningSpelling`] gains a variant.
pub const VALID_REASONING_SPELLINGS: &str = "effort-none, effort-minimal, reasoning-enabled-object";

/// A `[magi]` reasoning key carried a value outside its vocabulary (S-5).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("[magi].{key} = {got:?} is not accepted: {reason}")]
pub struct ReasoningVocabularyError {
    /// The `[magi]` key, without the section (`"reasoning"` or `"reasoning_spelling"`).
    pub key: &'static str,
    /// The value the file carried, verbatim (blank included).
    pub got: String,
    /// Why, naming every accepted value.
    pub reason: String,
}

/// Parses `[magi].reasoning` through magi-core's own serde (REQ-EE-3, OQ-10).
///
/// **It never trims and never guesses.** `raw` is compared exactly against the three kebab-case
/// tags [`ReasoningControl`] serializes as (`"default"`, `"disabled"`, `"enabled"`) — padding,
/// case and blankness are all rejected the same way an unknown word is, because `[magi]` keys
/// have no "blank is absent" rule the way env vars do (a `[magi]` table only exists when the
/// operator wrote it).
///
/// The deserialization goes through `serde_json` rather than a hand-written `match`, so the set
/// of accepted values — and the exact wording naming them in [`ReasoningVocabularyError::reason`]
/// — comes from magi-core's own `Deserialize` impl and can never drift from it one release
/// behind.
///
/// # Arguments
/// * `raw` - the value exactly as written in `magi.toml`.
///
/// # Errors
/// [`ReasoningVocabularyError`] naming the `"reasoning"` key, the value received, and every
/// accepted tag, when `raw` is not one of magi-core's three exact tags.
pub fn parse_reasoning_control(raw: &str) -> Result<ReasoningControl, ReasoningVocabularyError> {
    serde_json::from_value(serde_json::Value::String(raw.to_string())).map_err(|e| {
        ReasoningVocabularyError {
            key: "reasoning",
            got: raw.to_string(),
            reason: e.to_string(),
        }
    })
}

/// Parses `[magi].reasoning_spelling` against [`VALID_REASONING_SPELLINGS`] (REQ-V42-4).
///
/// **magi-rs owns this vocabulary — [`ReasoningSpelling`] has no `serde` impl of its own**, so
/// unlike [`parse_reasoning_control`] this is a plain exact match, never a `.trim()`: padding,
/// case and blankness are all unknown vocabulary.
///
/// # Arguments
/// * `raw` - the value exactly as written in `magi.toml`.
///
/// # Errors
/// [`ReasoningVocabularyError`] naming the `"reasoning_spelling"` key, the value received, and
/// the three accepted tags, when `raw` matches none of them.
pub fn parse_reasoning_spelling(raw: &str) -> Result<ReasoningSpelling, ReasoningVocabularyError> {
    match raw {
        "effort-none" => Ok(ReasoningSpelling::EffortNone),
        "effort-minimal" => Ok(ReasoningSpelling::EffortMinimal),
        "reasoning-enabled-object" => Ok(ReasoningSpelling::ReasoningEnabledObject),
        other => Err(ReasoningVocabularyError {
            key: "reasoning_spelling",
            got: other.to_string(),
            reason: format!("valid values: {VALID_REASONING_SPELLINGS}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::provider::ReasoningControl;
    use magi_core::providers::openai_compat::ReasoningSpelling;

    /// OQ-10: the key mirrors magi-core EXACTLY — every tag magi-core serializes parses back to
    /// the same variant, and the round trip goes through magi-core's own serde, never a table of
    /// ours.
    #[test]
    fn every_tag_magi_core_serializes_parses_to_the_same_control() {
        for control in [
            ReasoningControl::Default,
            ReasoningControl::Disabled,
            ReasoningControl::Enabled,
        ] {
            let tag = serde_json::to_value(control)
                .expect("magi-core serializes its own control")
                .as_str()
                .expect("a unit variant serializes as a string")
                .to_string();
            assert_eq!(
                parse_reasoning_control(&tag),
                Ok(control),
                "`{tag}` is what magi-core writes for {control:?}; it must parse back"
            );
        }
    }

    /// S-5: an unknown, blank, differently-cased or padded tag is an error naming the key and
    /// every accepted value — blank is unknown vocabulary for `[magi]` keys, never "absent".
    #[test]
    fn an_unknown_reasoning_tag_is_an_error_naming_the_key_and_the_accepted_values() {
        for bad in ["low", "", "  ", "Disabled", "disabled ", "off"] {
            let err = parse_reasoning_control(bad).expect_err("not in magi-core's vocabulary");
            assert_eq!(err.key, "reasoning");
            assert_eq!(err.got, bad);
            let text = err.to_string();
            for accepted in ["default", "disabled", "enabled"] {
                assert!(
                    text.contains(accepted),
                    "the message must name `{accepted}`: {text}"
                );
            }
            assert!(text.contains("[magi].reasoning"), "names the key: {text}");
        }
    }

    /// REQ-V42-4: the three spellings magi-rs declares map one-to-one onto magi-core's variants.
    #[test]
    fn each_spelling_tag_maps_to_its_magi_core_variant() {
        let table = [
            ("effort-none", ReasoningSpelling::EffortNone),
            ("effort-minimal", ReasoningSpelling::EffortMinimal),
            (
                "reasoning-enabled-object",
                ReasoningSpelling::ReasoningEnabledObject,
            ),
        ];
        for (tag, variant) in table {
            assert_eq!(parse_reasoning_spelling(tag), Ok(variant), "tag `{tag}`");
        }
        // The published list and the parser cannot drift: every listed tag parses.
        for tag in VALID_REASONING_SPELLINGS.split(", ") {
            assert!(
                parse_reasoning_spelling(tag).is_ok(),
                "listed but rejected: `{tag}`"
            );
        }
    }

    /// S-5 for the spelling: unknown, blank, snake_case, upper-case and magi-core's wire values
    /// (`none`, `minimal`) are all rejected, naming the key and the three accepted tags.
    #[test]
    fn an_unknown_spelling_is_an_error_naming_the_key_and_the_accepted_values() {
        for bad in [
            "none",
            "minimal",
            "",
            " ",
            "effort_none",
            "EFFORT-NONE",
            "effort-none ",
        ] {
            let err = parse_reasoning_spelling(bad).expect_err("not in the vocabulary");
            assert_eq!(err.key, "reasoning_spelling");
            assert_eq!(err.got, bad);
            let text = err.to_string();
            assert!(
                text.contains("[magi].reasoning_spelling"),
                "names the key: {text}"
            );
            for accepted in ["effort-none", "effort-minimal", "reasoning-enabled-object"] {
                assert!(text.contains(accepted), "names `{accepted}`: {text}");
            }
        }
    }
}
