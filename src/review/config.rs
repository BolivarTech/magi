// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-15

//! Configuration types of the `[code_review]` section: the endpoint and the model chain of the
//! false-positive gate.
//!
//! The section follows the rules of every endpoint section: `deny_unknown_fields` (an `api_key`
//! is a parse error), a `base_url` of its own or inherited from the root, credentials only as
//! vault placeholders. Its thresholds are its own and are never shared with `[decision]`: they
//! were calibrated with the gate's question.

use crate::decision::ModelSpec;

/// HTTP timeout of each gate call, in seconds, when `[code_review].timeout_secs` is absent:
/// sized like `[decision]` to cover a local cold model load. The single literal: the bin's
/// default and [`crate::review::gate::Gate`] both read it from here.
pub const DEFAULT_CALL_TIMEOUT_SECS: u64 = 120;

/// The `[code_review]` section as written in `magi.toml`.
///
/// # Examples
/// ```
/// use magi_rs::review::config::CodeReviewSection;
///
/// let section: CodeReviewSection = toml::from_str(
///     r#"
///     model = "clef:cloud"
///     min_confidence = 0.40
///     [[fallback]]
///     model = "clef-flash"
///     min_confidence = 0.32
///     "#,
/// )
/// .unwrap();
/// section.validate().unwrap();
/// let models: Vec<_> = section.chain().into_iter().map(|m| m.model).collect();
/// assert_eq!(models, ["clef:cloud", "clef-flash"]);
/// ```
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeReviewSection {
    /// Endpoint of the gate's models; absent means the root `base_url`.
    pub base_url: Option<String>,
    /// Primary gate model.
    pub model: String,
    /// HTTP timeout of each gate call, in seconds.
    pub timeout_secs: Option<u64>,
    /// Largest serialized gate `state`, in UTF-8 bytes; a larger one is never truncated.
    pub max_state_bytes: Option<usize>,
    /// Approval policy of the agent's `code-review` tool.
    pub auto_approve: Option<bool>,
    /// Confidence threshold of the primary model, required ([`CodeReviewSection::validate`]
    /// rejects a section without it): the gate's thresholds were calibrated per model.
    pub min_confidence: Option<f64>,
    /// Ordered fallbacks, on the same endpoint.
    #[serde(default)]
    pub fallback: Vec<CodeReviewFallback>,
}

/// One `[[code_review.fallback]]` entry.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeReviewFallback {
    /// Fallback model.
    pub model: String,
    /// Confidence threshold of this model, required like the primary's.
    pub min_confidence: Option<f64>,
}

/// Why a `[code_review]` section is invalid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CodeReviewConfigError {
    /// A model name is empty or blank.
    #[error("{position} must name a model")]
    BlankModel {
        /// Where the blank model is.
        position: String,
    },
    /// A model appears twice in the chain.
    #[error("model '{model}' appears more than once in [code_review] and its fallbacks")]
    DuplicateModel {
        /// The repeated model.
        model: String,
    },
    /// A model of the chain declares no threshold.
    #[error(
        "model '{model}' in [code_review] needs a min_confidence (a number between 0 and 1); \
         every gate model carries its own threshold"
    )]
    MissingMinConfidence {
        /// The model without a threshold.
        model: String,
    },
    /// A threshold is not a finite number in [0, 1].
    #[error("min_confidence of model '{model}' in [code_review] must be a number between 0 and 1")]
    MinConfidenceOutOfRange {
        /// The model whose threshold is invalid.
        model: String,
    },
    /// `timeout_secs` is zero.
    #[error("[code_review].timeout_secs must be at least 1")]
    TimeoutBelowOne,
    /// `max_state_bytes` is zero.
    #[error("[code_review].max_state_bytes must be at least 1")]
    MaxStateBytesBelowOne,
}

impl CodeReviewSection {
    /// Checks the section's values.
    ///
    /// Every model, the primary and each fallback, needs a finite `min_confidence` in [0, 1]
    /// (REQ-RV14). Models are compared after `trim`, as the server sees them: a repeated model would make a
    /// fallback retry what just failed. O(m) for m models.
    ///
    /// # Returns
    /// `Ok(())` when every rule holds.
    ///
    /// # Errors
    /// The first violated rule as a [`CodeReviewConfigError`].
    pub fn validate(&self) -> Result<(), CodeReviewConfigError> {
        const PRIMARY_POSITION: &str = "[code_review].model";
        const FALLBACK_POSITION_PREFIX: &str = "[[code_review.fallback]] #";
        const CONFIDENCE_RANGE: std::ops::RangeInclusive<f64> = 0.0..=1.0;

        if self.model.trim().is_empty() {
            return Err(CodeReviewConfigError::BlankModel {
                position: PRIMARY_POSITION.into(),
            });
        }
        for (idx, fb) in self.fallback.iter().enumerate() {
            if fb.model.trim().is_empty() {
                return Err(CodeReviewConfigError::BlankModel {
                    position: format!("{FALLBACK_POSITION_PREFIX}{}", idx + 1),
                });
            }
        }
        if self.timeout_secs == Some(0) {
            return Err(CodeReviewConfigError::TimeoutBelowOne);
        }
        if self.max_state_bytes == Some(0) {
            return Err(CodeReviewConfigError::MaxStateBytesBelowOne);
        }
        let mut seen = std::collections::HashSet::new();
        for spec in self.chain() {
            // `nan` parses from TOML and compares false with everything: finiteness is explicit.
            match spec.min_confidence {
                None => {
                    return Err(CodeReviewConfigError::MissingMinConfidence { model: spec.model })
                }
                Some(v) if !v.is_finite() || !CONFIDENCE_RANGE.contains(&v) => {
                    return Err(CodeReviewConfigError::MinConfidenceOutOfRange {
                        model: spec.model,
                    })
                }
                Some(_) => {}
            }
            if !seen.insert(spec.model.clone()) {
                return Err(CodeReviewConfigError::DuplicateModel { model: spec.model });
            }
        }
        Ok(())
    }

    /// The gate's chain: the primary model first, then the fallbacks in order, names trimmed.
    ///
    /// Each entry keeps its configured `min_confidence` unchanged: the gate
    /// ([`crate::review::gate::Gate`]) applies these thresholds itself. Only the decision client
    /// the bin builds for the gate receives the model names with the thresholds cleared, so the
    /// client returns every answer and its lean instead of applying a threshold twice.
    ///
    /// # Returns
    /// One [`ModelSpec`] per model, in declaration order, thresholds as configured.
    pub fn chain(&self) -> Vec<ModelSpec> {
        std::iter::once((&self.model, self.min_confidence))
            .chain(
                self.fallback
                    .iter()
                    .map(|fb| (&fb.model, fb.min_confidence)),
            )
            .map(|(model, min_confidence)| ModelSpec {
                model: model.trim().to_owned(),
                min_confidence,
            })
            .collect()
    }
}
