// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Provider vocabulary: the three values that name a specific backend (REQ-A01b).
//!
//! # Why it lives in the LIB and not in `config.rs`
//!
//! `probe_models` and `ProbeFactory` take it as a parameter and live in the lib, so leaving it
//! in the bin would make them fail to compile. And it is not a packaging concession: it is an enum
//! closed to three variants with its parser, that is, **domain** vocabulary, not the shape
//! of the TOML.

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

/// The three accepted values, in the text shown in an error.
///
/// A `const` and not a repeated literal (B4): the error message and the documentation must
/// name the same set.
pub const VALID_PROVIDER_KINDS: &str = "ollama, openai-compat, anthropic";

/// Concrete magi-core provider (REQ-A01b).
///
/// **SINGLE vocabulary**: the root `provider` key and `[magi].kind` accept the same three
/// values, and the second **inherits** from the first when not declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// Ollama: keyless, and the ONLY measurable one (`/api/show` + `/api/tags`).
    ///
    /// **Completes through magi-core's `OllamaProvider`** (REQ-R30, which reverts D-A07). Both
    /// `base_url` spellings are accepted — with `/v1` and without — and identity in errors and
    /// reports is `"ollama"`, where it used to be `"openai-compat"`.
    ///
    /// D-A07 held that this kind must NOT use the type, on the reason that *"its only constructor
    /// sets a 300 s client timeout with no override, so it cannot meet the scale of REQ-A04"*.
    /// That was an **impossibility**, and magi-core 3.2.0 removed it by adding `with_timeout`,
    /// which bounds both HTTP clients the type builds. With the impossibility gone the question
    /// was decided on its merits — neither option gains capability, since `OllamaProvider::complete`
    /// delegates to an internal `OpenAiCompatibleProvider` — and legibility won: this variant
    /// wired to that type is what anyone reading the two names together expects.
    ///
    /// **What survives the reversal is narrower and sharper: never build it with `new`.** It
    /// delegates with the 300 s default, which breaks `operation_budget + client_timeout <=
    /// ceiling`, and getting it wrong compiles, runs, and breaks the derived scale in silence.
    Ollama,
    /// OpenAI, Groq, OpenRouter — any Chat Completions. With token, no probe.
    OpenAiCompat,
    /// Anthropic Messages. With token, no probe.
    Anthropic,
}

/// A present `provider` or `kind` value that does not name any backend.
///
/// **`ProviderKindParseError` and NOT `ConfigError`**: this enum lives in the lib and `ConfigError`
/// in `config.rs`, which belongs to the bin. Returning the bin's error from the lib reverses the
/// direction of the dependency and does not compile; `config.rs` absorbs it with a `From`.
#[derive(Debug, thiserror::Error)]
#[error("unknown provider: {got:?} (valid: {valid})")]
pub struct ProviderKindParseError {
    /// What the file brought in.
    pub got: String,
    /// The three accepted ones, so the error is actionable without opening the docs.
    pub valid: &'static str,
}

impl ProviderKind {
    /// Parses a configuration value.
    ///
    /// # Errors
    ///
    /// [`ProviderKindParseError`] if the value is **present and not recognized**.
    ///
    /// An **empty or blank** value returns `Ok(None)` — it is treated as **absent** (REQ-A12),
    /// because an exported variable left empty in a CI script is indistinguishable from never
    /// having defined it, and breaking startup for that would punish an everyday accident. A
    /// present and unrecognized value is an error: the user meant to say something and said it wrong.
    ///
    /// Trims **ASCII** whitespace, just like `ModeExt::parse_config_value`, so the two
    /// vocabulary keys in the file are read with the same rule.
    pub fn parse(raw: &str) -> Result<Option<Self>, ProviderKindParseError> {
        match raw.trim_matches(|c: char| c.is_ascii_whitespace()) {
            "" => Ok(None),
            "ollama" => Ok(Some(Self::Ollama)),
            "openai-compat" => Ok(Some(Self::OpenAiCompat)),
            "anthropic" => Ok(Some(Self::Anthropic)),
            other => Err(ProviderKindParseError {
                got: other.to_string(),
                valid: VALID_PROVIDER_KINDS,
            }),
        }
    }

    /// Whether this provider exposes model introspection (REQ-A24).
    ///
    /// It is a difference of **capability**, not vocabulary: `ollama` and `openai-compat`
    /// share the completions protocol and differ only in that one is measurable.
    #[must_use]
    pub const fn is_probeable(self) -> bool {
        matches!(self, Self::Ollama)
    }
}

impl fmt::Display for ProviderKind {
    /// The exact inverse of [`Self::parse`]: `ProviderKind::parse(&k.to_string()) ==
    /// Ok(Some(k))` for any `k`. Task 4.1 needs it to render back to the
    /// declared vocabulary (e.g., when building the default `provider` that feeds the
    /// headless resolution `env > TOML > default`), without repeating the three literals from
    /// `parse` in a second place (B3).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ollama => "ollama",
            Self::OpenAiCompat => "openai-compat",
            Self::Anthropic => "anthropic",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-A01b: the three vocabulary values, and nothing else.
    #[test]
    fn the_three_vocabulary_values_are_accepted_and_the_rest_are_not() {
        assert_eq!(
            ProviderKind::parse("ollama").unwrap(),
            Some(ProviderKind::Ollama)
        );
        assert_eq!(
            ProviderKind::parse("openai-compat").unwrap(),
            Some(ProviderKind::OpenAiCompat)
        );
        assert_eq!(
            ProviderKind::parse("anthropic").unwrap(),
            Some(ProviderKind::Anthropic)
        );
        assert!(ProviderKind::parse("banana").is_err());
    }

    /// The v0.11.0 value is no longer valid: it is half of the REQ-A21 break.
    ///
    /// `"openai"` was ambiguous — it could be Ollama or an authenticated endpoint — and that ambiguity is
    /// exactly what the new vocabulary removes. **It is not auto-migrated**: choosing for the user
    /// would be guessing exactly what D-A01 forbids.
    #[test]
    fn the_old_openai_value_is_rejected_rather_than_guessed() {
        let err = ProviderKind::parse("openai").unwrap_err();
        assert!(
            err.to_string().contains("openai"),
            "names what was received"
        );
        assert!(
            err.to_string().contains("openai-compat"),
            "and the valid ones, so the fix is obvious"
        );
    }

    /// SC-A12g / REQ-A12: empty or blank is ABSENT, never invalid.
    #[test]
    fn a_blank_value_is_absent_rather_than_invalid() {
        assert_eq!(ProviderKind::parse("").unwrap(), None);
        assert_eq!(ProviderKind::parse("   ").unwrap(), None);
        assert_eq!(ProviderKind::parse("\t\n").unwrap(), None);
        // And the value surrounded by spaces remains valid.
        assert_eq!(
            ProviderKind::parse("  ollama  ").unwrap(),
            Some(ProviderKind::Ollama)
        );
    }

    /// REQ-A24: only Ollama is measurable, and that is capability, not vocabulary.
    #[test]
    fn only_ollama_exposes_model_introspection() {
        assert!(ProviderKind::Ollama.is_probeable());
        assert!(!ProviderKind::OpenAiCompat.is_probeable());
        assert!(!ProviderKind::Anthropic.is_probeable());
    }

    /// `Display` is the exact inverse of `parse` for the three values — Task 4.1 depends
    /// on this roundtrip to render the vocabulary back to text.
    #[test]
    fn display_round_trips_through_parse_for_the_three_values() {
        for kind in [
            ProviderKind::Ollama,
            ProviderKind::OpenAiCompat,
            ProviderKind::Anthropic,
        ] {
            assert_eq!(ProviderKind::parse(&kind.to_string()).unwrap(), Some(kind));
        }
    }
}
