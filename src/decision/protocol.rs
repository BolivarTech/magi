//! System One protocol types and input validation (REQ-D07, REQ-D22, REQ-D23).

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use crate::decision::config::KeepAlive;
use crate::decision::error::DecisionError;

/// Questions of a request, keyed by name; the map order is the wire order.
pub type Questions = BTreeMap<String, Question>;

/// One question, discriminated by `type` (System One protocol).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    /// A yes/no question answered with a probability.
    Noul {
        /// Free-text instructions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        /// Optional descriptions of the two outcomes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// A closed choice among named options.
    Choice {
        /// Free-text instructions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        /// Option name to description; required, at least two.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, String>>,
    },
    /// A score over ordered levels.
    Score {
        /// Free-text instructions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        /// Level descriptions; required, at least two.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<Vec<String>>,
    },
}

/// Descriptions of the two outcomes of a `noul` question.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    /// Meaning of "true".
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    /// Meaning of "false".
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

/// A validated request: state, questions and per-invocation overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    _state: Value,
    _questions: Questions,
    _model: Option<String>,
    _timeout: Option<Duration>,
}

impl DecisionRequest {
    /// Validates the state shape and the questions (REQ-D07, REQ-D23).
    ///
    /// Complexity: O(questions x options).
    ///
    /// # Arguments
    /// * `state` - a string, object or array, forwarded verbatim.
    /// * `questions` - at least one question, each with its required criteria.
    ///
    /// # Returns
    /// The validated request, without model or timeout override.
    ///
    /// # Errors
    /// [`DecisionError::InvalidInput`] naming the problem.
    pub fn new(_state: Value, _questions: Questions) -> Result<Self, DecisionError> {
        todo!()
    }

    /// Replaces the titular model for this request only.
    pub fn with_model(self, _model: impl Into<String>) -> Self {
        todo!()
    }

    /// Sets the per-attempt timeout for this request only.
    pub fn with_timeout(self, _timeout: Duration) -> Self {
        todo!()
    }

    /// The state, as given.
    pub fn state(&self) -> &Value {
        todo!()
    }

    /// The questions.
    pub fn questions(&self) -> &Questions {
        todo!()
    }

    /// The explicit model, if any.
    pub fn model(&self) -> Option<&str> {
        todo!()
    }

    /// The per-attempt timeout override, if any.
    pub fn timeout(&self) -> Option<Duration> {
        todo!()
    }

    /// The JSON body sent to the provider for `model`; `keep_alive` only when declared.
    pub fn wire_body(&self, _model: &str, _keep_alive: Option<&KeepAlive>) -> Value {
        todo!()
    }
}

/// Parses and validates the input of `magi decide` (REQ-D22, REQ-D23).
///
/// # Arguments
/// * `input` - contents of `-i` or stdin.
/// * `questions_file` - contents of `-q`, when given.
///
/// # Returns
/// The validated request.
///
/// # Errors
/// [`DecisionError::InvalidInput`] naming the problem.
pub fn parse_decide_input(
    _input: &[u8],
    _questions_file: Option<&[u8]>,
) -> Result<DecisionRequest, DecisionError> {
    todo!()
}

/// One answer, as Magi reports it (REQ-D10); `confidence` is always Magi's.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Answer to a `noul` question.
    Noul {
        /// Probability of "true".
        noul: f64,
        /// Magi confidence.
        confidence: f64,
    },
    /// Answer to a `choice` question.
    Choice {
        /// The chosen option.
        choice: String,
        /// Probability per option.
        probabilities: BTreeMap<String, f64>,
        /// Magi confidence.
        confidence: f64,
        /// Confidence the provider reported; informative only.
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_confidence: Option<f64>,
    },
    /// Answer to a `score` question.
    Score {
        /// Expected level, in level units.
        score: f64,
        /// Level index to level name.
        legend: BTreeMap<String, String>,
        /// Probability per level index.
        probabilities: BTreeMap<String, f64>,
        /// Magi confidence.
        confidence: f64,
        /// Confidence the provider reported; informative only.
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_confidence: Option<f64>,
    },
}

impl Answer {
    /// Magi's confidence for this answer.
    pub fn confidence(&self) -> f64 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::config::KeepAlive;
    use crate::decision::error::DecisionError;
    use serde_json::json;
    use std::time::Duration;

    fn parse(input: serde_json::Value) -> Result<DecisionRequest, DecisionError> {
        parse_decide_input(input.to_string().as_bytes(), None)
    }

    fn reason(r: Result<DecisionRequest, DecisionError>) -> String {
        match r {
            Err(DecisionError::InvalidInput { reason }) => reason,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    fn noul() -> serde_json::Value {
        json!({"triage": {"type": "noul", "instructions": "Is it urgent?"}})
    }

    /// CP2 MS1 r3 (Caspar): a field of the wrong type is invalid_input that names the field (REQ-D23).
    #[test]
    fn a_field_of_the_wrong_type_is_invalid_input_naming_it() {
        for (input, field) in [
            (
                json!({"state": "x", "questions": noul(), "model": 7}),
                "model",
            ),
            (json!({"state": null, "questions": noul()}), "state"),
            (json!({"state": true, "questions": noul()}), "state"),
            (json!({"state": 3, "questions": noul()}), "state"),
            (json!({"state": "x", "questions": [noul()]}), "questions"),
        ] {
            let r = reason(parse(input.clone()));
            assert!(r.contains(field), "{input}: {r}");
        }
        for file in [&b"not json"[..], &br#"["a list"]"#[..]] {
            let r = reason(parse_decide_input(br#"{"state":"x"}"#, Some(file)));
            assert!(r.contains("questions"), "{r}");
        }
    }

    #[test]
    fn a_full_input_with_one_question_of_each_type_parses() {
        let req = parse(json!({
            "state": {"ticket": "I was charged twice this month."},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "Payments", "technical": "Bugs", "other": "Anything else"}},
                "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
                "urgency": {"type": "score", "instructions": "How urgent?",
                            "criteria": ["Routine", "Soon", "Urgent"]}
            }
        }))
        .expect("the REQ-D23 example 1 is valid");
        assert_eq!(req.questions().len(), 3);
        assert!(matches!(
            req.questions().get("refund"),
            Some(Question::Noul { .. })
        ));
        assert!(matches!(
            req.questions().get("team"),
            Some(Question::Choice { .. })
        ));
        assert!(matches!(
            req.questions().get("urgency"),
            Some(Question::Score { .. })
        ));
        assert_eq!(req.model(), None);
        assert_eq!(req.timeout(), None);
    }

    #[test]
    fn the_state_is_forwarded_verbatim_in_its_three_shapes() {
        for state in [
            json!("text"),
            json!({"a": {"b": [1, 2]}}),
            json!([1, "two", null]),
        ] {
            let req = parse(json!({"state": state.clone(), "questions": noul()})).expect("valid");
            assert_eq!(req.state(), &state);
        }
    }

    #[test]
    fn a_state_that_is_not_a_string_object_or_array_is_invalid_input() {
        for state in [json!(null), json!(3), json!(true)] {
            let r = reason(parse(json!({"state": state, "questions": noul()})));
            assert!(r.contains("state"), "{r}");
        }
    }

    #[test]
    fn a_missing_state_is_invalid_input() {
        let r = reason(parse(json!({"questions": noul()})));
        assert!(r.contains("state"), "{r}");
    }

    #[test]
    fn an_explicit_model_is_carried_and_a_blank_one_is_invalid_input() {
        let req =
            parse(json!({"model": "tev1:0.8b", "state": "s", "questions": noul()})).expect("valid");
        assert_eq!(req.model(), Some("tev1:0.8b"));
        let r = reason(parse(
            json!({"model": "  ", "state": "s", "questions": noul()}),
        ));
        assert!(r.contains("model"), "{r}");
    }

    /// The table of invalid inputs of REQ-D23, row by row.
    #[test]
    fn every_invalid_input_of_req_d23_is_rejected_naming_the_problem() {
        let cases: [(serde_json::Value, &str); 6] = [
            (json!({"state": "x"}), "questions"),
            (json!({"state": "x", "questons": {}}), "questons"),
            (
                json!({"state": "x", "questions": {"triage": {"type": "choice", "instructions": "..."}}}),
                "question \"triage\"",
            ),
            (
                json!({"state": "x", "questions": {}}),
                "at least one question",
            ),
            (
                json!({"state": "x", "questions": {"triage": {"type": "choice", "criteria": {"bug": "Broken"}}}}),
                "question \"triage\"",
            ),
            (
                json!({"state": "x", "questions": {"triage": {"type": "score", "criteria": ["Urgent"]}}}),
                "question \"triage\"",
            ),
        ];
        for (input, needle) in cases {
            let r = reason(parse(input.clone()));
            assert!(r.contains(needle), "{input} -> {r}");
        }
    }

    #[test]
    fn text_that_is_not_json_is_invalid_input() {
        let r = reason(parse_decide_input(b"not json at all", None));
        assert!(r.contains("not valid JSON"), "{r}");
    }

    #[test]
    fn unknown_fields_inside_a_question_are_rejected_by_name() {
        let r = reason(parse(json!({"state": "x",
            "questions": {"triage": {"type": "noul", "instrucions": "?"}}})));
        assert!(r.contains("instrucions"), "{r}");
    }

    #[test]
    fn an_unknown_question_type_is_rejected() {
        let r = reason(parse(
            json!({"state": "x", "questions": {"triage": {"type": "bogus"}}}),
        ));
        assert!(r.contains("bogus"), "{r}");
    }

    #[test]
    fn keys_inside_the_state_and_option_names_are_user_data_not_unknown_fields() {
        parse(json!({
            "state": {"anything": {"nested": true}, "questons": 1},
            "questions": {"triage": {"type": "choice", "criteria": {"weird key": "a", "another one": "b"}}}
        }))
        .expect("state content and option names are data (REQ-D23)");
    }

    #[test]
    fn noul_criteria_accept_only_true_and_false_keys() {
        parse(
            json!({"state": "x", "questions": {"obstacle": {"type": "noul",
            "criteria": {"true": "closer than the safety distance", "false": "path clear"}}}}),
        )
        .expect("valid");
        let r = reason(parse(json!({"state": "x",
            "questions": {"obstacle": {"type": "noul", "criteria": {"maybe": "?"}}}})));
        assert!(r.contains("maybe"), "{r}");
    }

    #[test]
    fn with_a_questions_file_the_input_carries_only_state_and_model() {
        let questions =
            json!({"next_action": {"type": "choice", "criteria": {"forward": "f", "stop": "s"}}});
        let req = parse_decide_input(
            json!({"state": {"front_distance_m": 0.35}, "model": "tev1"})
                .to_string()
                .as_bytes(),
            Some(questions.to_string().as_bytes()),
        )
        .expect("valid");
        assert!(req.questions().contains_key("next_action"));
        assert_eq!(req.model(), Some("tev1"));
    }

    #[test]
    fn questions_in_both_the_input_and_the_questions_file_is_invalid_input() {
        let r = reason(parse_decide_input(
            json!({"state": "x", "questions": noul()})
                .to_string()
                .as_bytes(),
            Some(noul().to_string().as_bytes()),
        ));
        assert!(r.contains("questions"), "{r}");
    }

    #[test]
    fn a_questions_file_with_an_invalid_question_is_invalid_input_naming_it() {
        let r = reason(parse_decide_input(
            br#"{"state":"x"}"#,
            Some(br#"{"triage":{"type":"choice"}}"#),
        ));
        assert!(r.contains("question \"triage\""), "{r}");
    }

    #[test]
    fn new_applies_the_same_validation_as_the_parser() {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        assert!(DecisionRequest::new(json!("s"), questions.clone()).is_ok());
        assert!(matches!(
            DecisionRequest::new(json!(null), questions),
            Err(DecisionError::InvalidInput { .. })
        ));
        assert!(matches!(
            DecisionRequest::new(json!("s"), Questions::new()),
            Err(DecisionError::InvalidInput { .. })
        ));
    }

    #[test]
    fn with_model_and_with_timeout_override_per_invocation() {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        let req = DecisionRequest::new(json!("s"), questions)
            .expect("valid")
            .with_model("tev1")
            .with_timeout(Duration::from_secs(5));
        assert_eq!(req.model(), Some("tev1"));
        assert_eq!(req.timeout(), Some(Duration::from_secs(5)));
    }

    #[test]
    fn the_wire_body_carries_model_state_questions_and_keep_alive_only_when_declared() {
        let req = parse(json!({"state": "s", "questions": noul()})).expect("valid");
        let body = req.wire_body("nimble", None);
        assert_eq!(body["model"], "nimble");
        assert_eq!(body["state"], "s");
        assert_eq!(body["questions"]["triage"]["type"], "noul");
        assert_eq!(body["questions"]["triage"]["instructions"], "Is it urgent?");
        assert!(
            body["questions"]["triage"].get("criteria").is_none(),
            "None fields are omitted"
        );
        assert!(
            body.get("keep_alive").is_none(),
            "absent keep_alive is not sent (REQ-D01)"
        );
        let body = req.wire_body("nimble", Some(&KeepAlive::Text("10m".into())));
        assert_eq!(body["keep_alive"], "10m");
        let body = req.wire_body("nimble", Some(&KeepAlive::Seconds(-1)));
        assert_eq!(body["keep_alive"], -1);
    }

    /// Review Focus 2.
    #[test]
    fn an_empty_question_name_is_invalid_input() {
        let r = reason(parse(
            json!({"state": "x", "questions": {"": {"type": "noul"}}}),
        ));
        assert!(r.contains("question \"\""), "{r}");
    }

    /// Review Focus 2: names are user data and travel as they are.
    #[test]
    fn question_and_option_names_with_quotes_newlines_and_unicode_are_data() {
        let req = parse(json!({"state": "x", "questions": {"a\"b
ñ": {"type": "choice",
            "criteria": {"sí": "yes", "no
": "no"}}}}))
        .expect("valid");
        assert!(req.questions().contains_key(
            "a\"b
ñ"
        ));
    }

    /// Review Focus 5.
    #[test]
    fn an_explicit_model_is_trimmed() {
        let req =
            parse(json!({"model": " tev1 ", "state": "s", "questions": noul()})).expect("valid");
        assert_eq!(req.model(), Some("tev1"));
    }
}
