// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! Configuration types of the `[code_review]` section: the endpoint and the model chain of the
//! false-positive gate.
//!
//! The section follows the rules of every endpoint section: `deny_unknown_fields` (an `api_key`
//! is a parse error), a `base_url` of its own or inherited from the root, credentials only as
//! vault placeholders. Its thresholds are its own and are never shared with `[decision]`: they
//! were calibrated with the gate's question.

use crate::decision::ModelSpec;

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
    /// Confidence threshold of the primary model; absent means no threshold.
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
    /// Confidence threshold of this model; absent means no threshold.
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
    /// # Returns
    /// `Ok(())` when every rule holds.
    ///
    /// # Errors
    /// The first violated rule as a [`CodeReviewConfigError`].
    pub fn validate(&self) -> Result<(), CodeReviewConfigError> {
        todo!()
    }

    /// The gate's chain: the primary model first, then the fallbacks in order.
    ///
    /// # Returns
    /// One [`ModelSpec`] per model, in declaration order.
    pub fn chain(&self) -> Vec<ModelSpec> {
        todo!()
    }
}
