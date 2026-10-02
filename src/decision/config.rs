//! Configuration types of the `[decision]` section (REQ-D01, REQ-D21).

/// Value of `[decision].keep_alive`: an Ollama duration (`"10m"`) or a number of
/// seconds (`-1` keeps the model loaded). Sent verbatim when declared (REQ-D01).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum KeepAlive {
    /// A duration string such as `"10m"` or `"2h"`.
    Text(String),
    /// A number of seconds; `-1` means indefinitely.
    Seconds(i64),
}

/// The `[decision]` section as written in `magi.toml` (REQ-D01, REQ-D21).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSection {
    /// Endpoint of the System One models; absent means the root `base_url`.
    pub base_url: Option<String>,
    /// Primary decision model.
    pub model: String,
    /// Per-attempt timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Ollama `keep_alive`, sent only when declared.
    pub keep_alive: Option<KeepAlive>,
    /// Approval policy of the agent's `decide` tool.
    pub auto_approve: Option<bool>,
    /// Confidence threshold of the primary model.
    pub min_confidence: Option<f64>,
    /// Ordered fallbacks, on the same endpoint.
    #[serde(default)]
    pub fallback: Vec<DecisionFallback>,
}

/// One `[[decision.fallback]]` entry.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFallback {
    /// Fallback model.
    pub model: String,
    /// Confidence threshold of this model.
    pub min_confidence: Option<f64>,
}

/// A model of the chain with its threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Model name.
    pub model: String,
    /// Confidence threshold, if declared.
    pub min_confidence: Option<f64>,
}

/// Why a `[decision]` section is invalid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DecisionConfigError {
    /// A model name is empty or blank.
    #[error("{position} must name a model")]
    BlankModel {
        /// Where the blank model is.
        position: String,
    },
    /// A model appears twice in the chain.
    #[error("model '{model}' appears more than once in [decision] and its fallbacks")]
    DuplicateModel {
        /// The repeated model.
        model: String,
    },
    /// A threshold is not a finite number in [0, 1].
    #[error("min_confidence of model '{model}' must be a number between 0 and 1")]
    MinConfidenceOutOfRange {
        /// The model whose threshold is invalid.
        model: String,
    },
    /// `timeout_secs` is zero.
    #[error("[decision].timeout_secs must be at least 1")]
    TimeoutBelowOne,
    /// `keep_alive` is a blank string.
    #[error("[decision].keep_alive must not be blank")]
    BlankKeepAlive,
}

impl DecisionSection {
    /// Checks the section's values.
    ///
    /// # Errors
    ///
    /// Returns the first violated rule as a [`DecisionConfigError`].
    pub fn validate(&self) -> Result<(), DecisionConfigError> {
        todo!()
    }

    /// The chain: the primary model first, then the fallbacks in order.
    pub fn chain(&self) -> Vec<ModelSpec> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(toml_text: &str) -> Result<DecisionSection, toml::de::Error> {
        toml::from_str(toml_text)
    }

    #[test]
    fn a_minimal_section_needs_only_the_model_and_has_no_fallbacks() {
        let s = section("model = \"nimble\"").expect("parses");
        assert_eq!(s.model, "nimble");
        assert!(s.fallback.is_empty());
        assert_eq!(
            s.chain(),
            vec![ModelSpec {
                model: "nimble".into(),
                min_confidence: None
            }]
        );
        assert_eq!(s.validate(), Ok(()));
    }

    #[test]
    fn the_full_reference_section_parses_with_both_keep_alive_forms() {
        let s = section(
            "base_url = \"http://localhost:11434\"\nmodel = \"nimble\"\ntimeout_secs = 120\n\
             keep_alive = \"10m\"\nauto_approve = true\nmin_confidence = 0.6\n\
             [[fallback]]\nmodel = \"tev1\"\nmin_confidence = 0.5\n[[fallback]]\nmodel = \"tev1:0.8b\"\n",
        )
        .expect("parses");
        assert_eq!(s.keep_alive, Some(KeepAlive::Text("10m".into())));
        assert_eq!(
            s.chain(),
            vec![
                ModelSpec {
                    model: "nimble".into(),
                    min_confidence: Some(0.6)
                },
                ModelSpec {
                    model: "tev1".into(),
                    min_confidence: Some(0.5)
                },
                ModelSpec {
                    model: "tev1:0.8b".into(),
                    min_confidence: None
                },
            ]
        );
        let s = section("model = \"m\"\nkeep_alive = -1\n").expect("parses");
        assert_eq!(s.keep_alive, Some(KeepAlive::Seconds(-1)));
    }

    #[test]
    fn unknown_keys_an_api_key_and_endpoint_fields_in_a_fallback_are_rejected() {
        assert!(section("model = \"m\"\nmodle = \"x\"\n").is_err(), "typo");
        assert!(
            section("model = \"m\"\napi_key = \"sk-x\"\n").is_err(),
            "api_key never lives in the toml"
        );
        assert!(
            section("model = \"m\"\n[[fallback]]\nmodel = \"f\"\nbase_url = \"http://x\"\n")
                .is_err()
        );
        assert!(
            section("model = \"m\"\n[[fallback]]\nmodel = \"f\"\nkeep_alive = \"5m\"\n").is_err()
        );
        assert!(section("model = \"m\"\n[[fallback]]\nmodel = \"f\"\napi_key = \"k\"\n").is_err());
        assert!(
            section("model = \"m\"\n[[fallback]]\nmodel = \"f\"\nmin_confidence = 0.4\n").is_ok()
        );
        assert!(section("timeout_secs = 5\n").is_err(), "model is required");
    }

    #[test]
    fn validation_rejects_each_invalid_value() {
        let cases = [
            ("model = \"  \"\n", "blank model"),
            (
                "model = \"m\"\n[[fallback]]\nmodel = \"\"\n",
                "blank fallback model",
            ),
            (
                "model = \"m\"\n[[fallback]]\nmodel = \" m \"\n",
                "fallback repeats the primary",
            ),
            (
                "model = \"m\"\n[[fallback]]\nmodel = \"f\"\n[[fallback]]\nmodel = \"f\"\n",
                "repeated fallback",
            ),
            ("model = \"m\"\nmin_confidence = 1.5\n", "above 1"),
            ("model = \"m\"\nmin_confidence = -0.1\n", "below 0"),
            ("model = \"m\"\nmin_confidence = nan\n", "not finite"),
            (
                "model = \"m\"\n[[fallback]]\nmodel = \"f\"\nmin_confidence = 2.0\n",
                "fallback above 1",
            ),
            ("model = \"m\"\ntimeout_secs = 0\n", "timeout below 1"),
            ("model = \"m\"\nkeep_alive = \" \"\n", "blank keep_alive"),
        ];
        for (text, what) in cases {
            let s = section(text).unwrap_or_else(|e| panic!("{what}: must parse, got {e}"));
            assert!(s.validate().is_err(), "{what}");
        }
    }

    #[test]
    fn a_repeated_model_error_names_the_model() {
        let s = section("model = \"nimble\"\n[[fallback]]\nmodel = \"nimble\"\n").expect("parses");
        assert_eq!(
            s.validate(),
            Err(DecisionConfigError::DuplicateModel {
                model: "nimble".into()
            })
        );
    }
}
