// Author: Julian Bolivar
// Version: 0.26.0
// Date: 2026-11-26

//! System One protocol types and input validation (REQ-D07, REQ-D22, REQ-D23).

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use crate::decision::config::KeepAlive;
use crate::decision::error::{DecisionError, Usage};
use crate::review::secrets::{
    any_string, joined_strings_hold_key_block, leaks_secret, masked_strings,
};

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

/// Field of the input holding the state.
const STATE_FIELD: &str = "state";
/// Field of the input holding the questions.
const QUESTIONS_FIELD: &str = "questions";
/// Field of the input holding the explicit model.
const MODEL_FIELD: &str = "model";
/// Field of the wire body holding the keep-alive duration.
const KEEP_ALIVE_FIELD: &str = "keep_alive";
/// Minimum options of a `choice` and levels of a `score` (REQ-D07): one
/// possibility is not a decision.
const MIN_CHOICES: usize = 2;
/// Fields the input of `magi decide` may carry (REQ-D23).
const INPUT_FIELDS: [&str; 3] = [STATE_FIELD, QUESTIONS_FIELD, MODEL_FIELD];
/// Problem reported for a question without usable `instructions`.
const INSTRUCTIONS_REQUIRED: &str =
    "`instructions` is required (a non-empty description of what to decide)";
/// Problem reported for a state holding a private-key block; quotes nothing of the state.
const KEY_BLOCK_REFUSED: &str = "the state holds a private-key block and is never sent";
/// Problem reported for a state that still holds a secret after masking; quotes nothing of it.
const RESIDUAL_SECRET_REFUSED: &str =
    "the state still holds a secret after masking and is never sent";
/// Problem reported for questions holding a secret or a private-key block; quotes nothing of
/// them. Questions are never masked: a masked name or option would break the answer mapping.
const QUESTIONS_SECRET_REFUSED: &str =
    "the questions hold text that matches a secret pattern and are never sent; rename or remove it";
/// The fixed reason for an input whose top-level field name holds a secret: the `unknown field`
/// message would quote it (REQ-F12b).
const INPUT_FIELD_SECRET_REFUSED: &str =
    "an input field name matches a secret pattern and is never sent; rename or remove it";
/// The fixed reason for an explicit `model` that holds a secret: it would be sent, and
/// `ModelUnavailable` quotes it (REQ-F12b).
const MODEL_SECRET_REFUSED: &str =
    "the model name matches a secret pattern and is never sent; rename or remove it";
/// Problem reported when the questions cannot be serialized.
const QUESTIONS_UNSERIALIZABLE: &str = "the questions could not be serialized";
/// The fixed reason for a model of an attempt that holds a secret, whatever set it
/// (configuration, `with_model`, an explicit input): it would be sent and logged.
const WIRE_MODEL_SECRET_REFUSED: &str =
    "a decision model name matches a secret pattern and is never sent";
/// The fixed reason for a `keep_alive` text that holds a secret: it would be sent verbatim.
const KEEP_ALIVE_SECRET_REFUSED: &str =
    "the keep_alive value matches a secret pattern and is never sent";

/// Builds the error every validation failure of this module returns.
fn invalid(reason: impl Into<String>) -> DecisionError {
    DecisionError::InvalidInput {
        reason: reason.into(),
    }
}

/// The problem of one question, if any; never quotes a value of the question.
///
/// The type's own rules come first, then `instructions`, which every type requires
/// (user decision 2026-10-10: Ollama rejects a question without them with an opaque 400).
fn question_problem(question: &Question) -> Option<&'static str> {
    type_problem(question).or_else(|| instructions_problem(question))
}

/// The problem with the `instructions` of one question, if any: missing, empty or blank.
fn instructions_problem(question: &Question) -> Option<&'static str> {
    let (Question::Noul { instructions, .. }
    | Question::Choice { instructions, .. }
    | Question::Score { instructions, .. }) = question;
    match instructions.as_deref().map(str::trim) {
        Some(text) if !text.is_empty() => None,
        _ => Some(INSTRUCTIONS_REQUIRED),
    }
}

/// The problem with the type-specific fields of one question, if any.
fn type_problem(question: &Question) -> Option<&'static str> {
    match question {
        Question::Noul { .. } => None,
        Question::Choice { criteria, .. } => match criteria {
            None => Some("`criteria` is required"),
            Some(c) if c.len() < MIN_CHOICES => Some("needs at least 2 options"),
            Some(_) => None,
        },
        Question::Score { criteria, .. } => match criteria {
            None => Some("`criteria` is required"),
            Some(c) if c.len() < MIN_CHOICES => Some("needs at least 2 levels"),
            Some(_) => None,
        },
    }
}

/// Validates the state shape and every question (REQ-D07, REQ-D23).
///
/// Complexity: O(questions).
fn validate(state: &Value, questions: &Questions) -> Result<(), DecisionError> {
    if !matches!(state, Value::String(_) | Value::Object(_) | Value::Array(_)) {
        return Err(invalid("`state` must be a string, object or array"));
    }
    if questions.is_empty() {
        return Err(invalid("at least one question is required"));
    }
    for (name, question) in questions {
        if name.is_empty() {
            return Err(invalid("question \"\": the name must not be blank"));
        }
        if let Some(problem) = question_problem(question) {
            return Err(invalid(format!("question \"{name}\": {problem}")));
        }
    }
    Ok(())
}

/// Reads the `questions` map, naming the question that fails to parse.
///
/// Complexity: O(size of the map).
fn parse_questions(value: Value) -> Result<Questions, DecisionError> {
    let Value::Object(map) = value else {
        return Err(invalid("`questions` must be a JSON object"));
    };
    map.into_iter()
        .map(
            |(name, raw)| match serde_json::from_value::<Question>(raw) {
                Ok(question) => Ok((name, question)),
                Err(e) => Err(invalid(format!("question \"{name}\": {e}"))),
            },
        )
        .collect()
}

/// A validated request: state, questions and per-invocation overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    state: Value,
    questions: Questions,
    model: Option<String>,
    timeout: Option<Duration>,
}

impl DecisionRequest {
    /// Scans the questions and the state for what never leaves the process, validates the state
    /// shape and the questions (REQ-D07, REQ-D23), then masks the secrets of the state (REQ-F12).
    /// The scans run before validation because a validation reason quotes a question name.
    ///
    /// Every path that sends a decision (the CLI and TUI `decide`, the `decide` tool and the
    /// internal adapters) builds its request here, so no state leaves the process with a
    /// secret-shaped token in a string value; see [`masked_strings`]. It fails closed like the
    /// review gate: a state whose strings (values and keys, read joined by
    /// [`joined_strings_hold_key_block`], so a header and its body in two fields count) hold a
    /// private-key block is refused before masking, since a key body matches no pattern, and a
    /// state nested past the depth bound is refused the same way. A state that still holds
    /// a secret after masking (keys are never masked) is refused as well. The questions travel with
    /// the state and are never masked, so any of their strings or keys that holds a secret or a
    /// private-key block refuses the request. Both residual checks ask [`leaks_secret`], the
    /// review's one egress predicate. A size limit applied to the state must measure
    /// [`DecisionRequest::state`], the masked state that is sent.
    ///
    /// Complexity: O(questions) plus O(size of the state) for the checks and the masking.
    ///
    /// # Arguments
    /// * `state` - a string, object or array; its string values are masked, the rest is kept.
    /// * `questions` - at least one question, each with its required criteria.
    ///
    /// # Returns
    /// The validated request, without model or timeout override.
    ///
    /// # Errors
    /// [`DecisionError::InvalidInput`] naming the problem; for refused state or questions, a fixed
    /// reason that quotes nothing of them.
    pub fn new(state: Value, questions: Questions) -> Result<Self, DecisionError> {
        // The egress scans come first: a validation reason quotes a question name, and no
        // reason may be built from a string the scans would flag.
        let wire_questions =
            serde_json::to_value(&questions).map_err(|_| invalid(QUESTIONS_UNSERIALIZABLE))?;
        if any_string(&wire_questions, &leaks_secret) {
            return Err(invalid(QUESTIONS_SECRET_REFUSED));
        }
        if joined_strings_hold_key_block(&state) {
            return Err(invalid(KEY_BLOCK_REFUSED));
        }
        validate(&state, &questions)?;
        let state = masked_strings(state);
        if any_string(&state, &leaks_secret) {
            return Err(invalid(RESIDUAL_SECRET_REFUSED));
        }
        Ok(Self {
            state,
            questions,
            model: None,
            timeout: None,
        })
    }

    /// Replaces the titular model for this request only.
    ///
    /// The name is sent to the decision endpoint as it is, and this method does not screen it
    /// for secrets: a caller passes a model name from configuration, or one already screened.
    /// [`parse_decide_input`] screens the explicit `model` of a user's input with
    /// [`leaks_secret`] before it gets here.
    ///
    /// # Arguments
    /// * `model` - model name; surrounding whitespace is trimmed and a blank
    ///   name leaves the request without an explicit model.
    ///
    /// # Returns
    /// The request with the explicit model set, or unchanged-in-effect
    /// (no explicit model) when the name is blank.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model: String = model.into();
        let trimmed = model.trim();
        self.model = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        self
    }

    /// Sets the per-attempt timeout for this request only.
    ///
    /// # Arguments
    /// * `timeout` - timeout of each attempt of the chain.
    ///
    /// # Returns
    /// The request with the timeout override set.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The state, with its secrets masked.
    ///
    /// # Returns
    /// A borrow of the masked state, the one forwarded to the provider.
    pub fn state(&self) -> &Value {
        &self.state
    }

    /// The questions.
    ///
    /// # Returns
    /// A borrow of the validated question map.
    pub fn questions(&self) -> &Questions {
        &self.questions
    }

    /// The explicit model, if any.
    ///
    /// # Returns
    /// The model that replaces the titular one, or `None`.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The per-attempt timeout override, if any.
    ///
    /// # Returns
    /// The override, or `None` to use the configured timeout.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// The JSON body sent to the provider for `model`; `keep_alive` only when declared.
    ///
    /// Serializing the validated questions and the keep-alive value cannot fail in practice
    /// (plain maps, strings and numbers), so no unit test can trigger the error arm; it exists
    /// so a future type change surfaces as an error instead of an empty body on the wire.
    ///
    /// # Arguments
    /// * `model` - the model of this attempt.
    /// * `keep_alive` - sent verbatim when `Some`, omitted otherwise (REQ-D01).
    ///
    /// # Returns
    /// The request body (System One protocol).
    ///
    /// # Errors
    /// [`DecisionError::InvalidInput`] if a part of the request cannot be serialized.
    pub(crate) fn wire_body(
        &self,
        model: &str,
        keep_alive: Option<&KeepAlive>,
    ) -> Result<Value, DecisionError> {
        let mut body = serde_json::Map::new();
        body.insert(MODEL_FIELD.into(), Value::String(model.to_owned()));
        body.insert(STATE_FIELD.into(), self.state.clone());
        body.insert(
            QUESTIONS_FIELD.into(),
            serde_json::to_value(&self.questions).map_err(|_| invalid(QUESTIONS_UNSERIALIZABLE))?,
        );
        if let Some(keep_alive) = keep_alive {
            body.insert(
                KEEP_ALIVE_FIELD.into(),
                serde_json::to_value(keep_alive).map_err(|_| DecisionError::InvalidInput {
                    reason: "keep_alive could not be serialized".into(),
                })?,
            );
        }
        Ok(Value::Object(body))
    }
}

/// Refuses the parts of a wire body that are not the request itself, before any attempt is
/// sent or logged: each model of the attempts and a `keep_alive` text (REQ-F12b). Whatever set
/// them (configuration, [`DecisionRequest::with_model`], an explicit input), a part that
/// [`leaks_secret`] flags would leave verbatim in the body and in the run log. O(bytes of the
/// names and the text).
///
/// # Arguments
/// * `models` - The model of every attempt the call may make.
/// * `keep_alive` - The keep-alive value sent with each attempt, if any.
///
/// # Returns
/// `Ok(())` when nothing is flagged.
///
/// # Errors
/// [`DecisionError::InvalidInput`] with a fixed reason that quotes nothing.
pub(crate) fn screen_wire_parts<'a>(
    mut models: impl Iterator<Item = &'a str>,
    keep_alive: Option<&KeepAlive>,
) -> Result<(), DecisionError> {
    if let Some(KeepAlive::Text(text)) = keep_alive {
        if leaks_secret(text) {
            return Err(invalid(KEEP_ALIVE_SECRET_REFUSED));
        }
    }
    if models.any(leaks_secret) {
        return Err(invalid(WIRE_MODEL_SECRET_REFUSED));
    }
    Ok(())
}

/// Refuses an input object whose field names hold a secret, before any message that quotes a
/// field name (`unknown field`) is built (REQ-F12b). Every surface that parses a decision
/// request from untrusted JSON calls it first. O(bytes of the names).
///
/// # Arguments
/// * `fields` - The top-level fields of the input object.
///
/// # Returns
/// `Ok(())` when nothing is flagged.
///
/// # Errors
/// [`DecisionError::InvalidInput`] with a fixed reason that quotes nothing.
pub fn screen_field_names(fields: &serde_json::Map<String, Value>) -> Result<(), DecisionError> {
    if fields.keys().any(|k| leaks_secret(k)) {
        return Err(invalid(INPUT_FIELD_SECRET_REFUSED));
    }
    Ok(())
}

/// Refuses raw `questions` holding a secret in any string or key, before parsing: a parse error
/// quotes the question name and serde's text, which can quote a value (REQ-F12b).
/// O(bytes of the questions).
///
/// # Arguments
/// * `questions` - The unparsed `questions` value.
///
/// # Returns
/// `Ok(())` when nothing is flagged.
///
/// # Errors
/// [`DecisionError::InvalidInput`] with a fixed reason that quotes nothing.
pub fn screen_raw_questions(questions: &Value) -> Result<(), DecisionError> {
    if any_string(questions, &leaks_secret) {
        return Err(invalid(QUESTIONS_SECRET_REFUSED));
    }
    Ok(())
}

/// Parses and validates the input of `magi decide` (REQ-D22, REQ-D23).
///
/// Complexity: O(input size).
///
/// # Arguments
/// * `input` - contents of `-i` or stdin.
/// * `questions_file` - contents of `-q`, when given; then the input must not
///   carry `questions`.
///
/// # Returns
/// The validated request, with the explicit `model` (trimmed) if the input has one.
///
/// # Errors
/// [`DecisionError::InvalidInput`] naming the problem; no value of the state
/// or of an option is quoted.
///
/// # Examples
/// ```
/// use magi_rs::decision::protocol::parse_decide_input;
///
/// let input = br#"{"state": "a ticket", "questions": {"q": {"type": "noul", "instructions": "Is it urgent?"}}}"#;
/// let request = parse_decide_input(input, None).unwrap();
/// assert!(request.model().is_none());
/// ```
pub fn parse_decide_input(
    input: &[u8],
    questions_file: Option<&[u8]>,
) -> Result<DecisionRequest, DecisionError> {
    let Value::Object(mut fields) =
        serde_json::from_slice(input).map_err(|_| invalid("the input is not valid JSON"))?
    else {
        return Err(invalid("the input must be a JSON object"));
    };
    screen_field_names(&fields)?;
    if let Some(unknown) = fields.keys().find(|k| !INPUT_FIELDS.contains(&k.as_str())) {
        return Err(invalid(format!("unknown field `{unknown}`")));
    }
    let model = match fields.remove(MODEL_FIELD) {
        None => None,
        Some(Value::String(m)) if leaks_secret(&m) => return Err(invalid(MODEL_SECRET_REFUSED)),
        Some(Value::String(m)) if !m.trim().is_empty() => Some(m.trim().to_owned()),
        Some(_) => return Err(invalid("`model` must be a non-blank string")),
    };
    let state = fields
        .remove(STATE_FIELD)
        .ok_or_else(|| invalid("missing field `state`"))?;
    let questions = match (fields.remove(QUESTIONS_FIELD), questions_file) {
        (Some(_), Some(_)) => {
            return Err(invalid(
                "`questions` is given in both the input and the questions file",
            ))
        }
        (Some(value), None) => value,
        (None, Some(bytes)) => serde_json::from_slice(bytes)
            .map_err(|_| invalid("the questions file is not valid JSON"))?,
        (None, None) => return Err(invalid("missing field `questions`")),
    };
    screen_raw_questions(&questions)?;
    let mut request = DecisionRequest::new(state, parse_questions(questions)?)?;
    request.model = model;
    Ok(request)
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
    ///
    /// # Returns
    /// The confidence in [0, 1] computed by [`magi_confidence`], never the provider's.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Noul { confidence, .. }
            | Self::Choice { confidence, .. }
            | Self::Score { confidence, .. } => *confidence,
        }
    }
}

/// Largest deviation from 1 the probabilities of one question may have
/// (REQ-D07); measured against the providers' rounding (E9).
pub const PROBABILITY_SUM_TOLERANCE: f64 = 0.01;

/// Magi confidence of a distribution: `clamp((n * p_max - 1) / (n - 1), 0, 1)`.
///
/// A distribution of fewer than two entries carries no doubt to measure and
/// returns 0. The clamp absorbs a `p_max` just under `1 / n`, which the sum
/// tolerance allows.
///
/// Complexity: O(n).
///
/// # Arguments
/// * `probabilities` - the probabilities of the options or levels of one question.
///
/// # Returns
/// The confidence in [0, 1].
///
/// # Examples
/// ```
/// use magi_rs::decision::protocol::magi_confidence;
///
/// assert_eq!(magi_confidence(&[1.0, 0.0]), 1.0);
/// assert_eq!(magi_confidence(&[0.5, 0.5]), 0.0);
/// ```
pub fn magi_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < MIN_CHOICES {
        return 0.0;
    }
    let n = probabilities.len() as f64;
    let p_max = probabilities.iter().copied().fold(0.0_f64, f64::max);
    ((n * p_max - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// A validated System One response.
#[derive(Debug)]
pub(crate) struct ParsedResponse {
    /// One answer per question sent.
    pub answers: BTreeMap<String, Answer>,
    /// Tokens (and cost) the provider reported.
    pub usage: Usage,
}

/// A finite number in `[0, 1]`; `None` for anything else.
fn unit_number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| (0.0..=1.0).contains(v))
}

/// Reads a probability map whose keys must be exactly `keys`.
///
/// Complexity: O(keys log keys).
fn read_probabilities(
    raw: Option<&Value>,
    keys: &[&str],
) -> Result<BTreeMap<String, f64>, &'static str> {
    let Some(Value::Object(map)) = raw else {
        return Err("`probabilities` is missing or not an object");
    };
    if map.len() != keys.len() {
        return Err("`probabilities` keys do not match the question");
    }
    let mut out = BTreeMap::new();
    for key in keys {
        let value = map
            .get(*key)
            .ok_or("`probabilities` keys do not match the question")?;
        let p = unit_number(value).ok_or("a probability is not a number in [0, 1]")?;
        out.insert((*key).to_owned(), p);
    }
    if (out.values().sum::<f64>() - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err("`probabilities` do not sum to 1");
    }
    Ok(out)
}

/// The confidence the provider reported, if it sent a number.
fn reported_confidence(raw: &Value) -> Option<f64> {
    raw.get("confidence").and_then(Value::as_f64)
}

/// Validates the answer to one question against the question sent.
///
/// Complexity: O(options log options).
fn read_answer(question: &Question, raw: &Value) -> Result<Answer, &'static str> {
    const TYPE_MISMATCH: &str = "the answer type does not match the question";
    let kind = raw.get("type").and_then(Value::as_str);
    match question {
        Question::Noul { .. } => {
            if kind != Some("noul") {
                return Err(TYPE_MISMATCH);
            }
            let noul = raw
                .get("noul")
                .and_then(unit_number)
                .ok_or("`noul` is not a number in [0, 1]")?;
            Ok(Answer::Noul {
                noul,
                confidence: magi_confidence(&[noul, 1.0 - noul]),
            })
        }
        Question::Choice { criteria, .. } => {
            if kind != Some("choice") {
                return Err(TYPE_MISMATCH);
            }
            let options: Vec<&str> = criteria
                .iter()
                .flat_map(|c| c.keys().map(String::as_str))
                .collect();
            let probabilities = read_probabilities(raw.get("probabilities"), &options)?;
            let choice = raw
                .get("choice")
                .and_then(Value::as_str)
                .filter(|c| options.contains(c))
                .ok_or("`choice` is not one of the options sent")?;
            let values: Vec<f64> = probabilities.values().copied().collect();
            Ok(Answer::Choice {
                choice: choice.to_owned(),
                confidence: magi_confidence(&values),
                provider_confidence: reported_confidence(raw),
                probabilities,
            })
        }
        Question::Score { criteria, .. } => {
            if kind != Some("score") {
                return Err(TYPE_MISMATCH);
            }
            let levels = criteria.as_deref().unwrap_or_default();
            let indices: Vec<String> = (0..levels.len()).map(|i| i.to_string()).collect();
            let keys: Vec<&str> = indices.iter().map(String::as_str).collect();
            let probabilities = read_probabilities(raw.get("probabilities"), &keys)?;
            let top = levels.len().saturating_sub(1) as f64;
            let score = raw
                .get("score")
                .and_then(Value::as_f64)
                .filter(|s| {
                    *s >= -PROBABILITY_SUM_TOLERANCE * top
                        && *s <= top * (1.0 + PROBABILITY_SUM_TOLERANCE)
                })
                .ok_or("`score` is outside the range of the levels")?
                .clamp(0.0, top);
            let values: Vec<f64> = probabilities.values().copied().collect();
            let legend = indices
                .iter()
                .cloned()
                .zip(levels.iter().cloned())
                .collect();
            Ok(Answer::Score {
                score,
                legend,
                confidence: magi_confidence(&values),
                provider_confidence: reported_confidence(raw),
                probabilities,
            })
        }
    }
}

/// Reads `usage`; absent counts as zero, a malformed one is invalid.
fn read_usage(raw: Option<&Value>) -> Result<Usage, &'static str> {
    const MALFORMED: &str = "`usage` is malformed";
    let Some(raw) = raw else {
        return Ok(Usage::default());
    };
    let tokens = |field: &str| match raw.get(field) {
        None => Ok(0),
        Some(v) => v.as_u64().ok_or(MALFORMED),
    };
    let cost = match raw.get("cost") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().ok_or(MALFORMED)?),
    };
    Ok(Usage {
        input_tokens: tokens("input_tokens")?,
        output_tokens: tokens("output_tokens")?,
        cost,
    })
}

/// Tolerant read plus strict validation of the fields Magi uses (REQ-D07).
///
/// Fields Magi does not use (`id`, `provider`, invented ones, the provider's
/// `legend`) are ignored. Every question sent needs exactly one answer of its
/// type; `confidence` is Magi's, the provider's goes to `provider_confidence`.
///
/// Complexity: O(questions x options log options).
///
/// # Arguments
/// * `body` - raw response body.
/// * `questions` - the questions that were sent.
///
/// # Errors
/// A reason naming the question and the class of problem; it never quotes a
/// value returned by the provider.
pub(crate) fn parse_response(body: &[u8], questions: &Questions) -> Result<ParsedResponse, String> {
    let root: Value = serde_json::from_slice(body)
        .map_err(|_| "the response body is not valid JSON".to_owned())?;
    let Some(Value::Object(answers)) = root.get("answers") else {
        return Err("the response has no `answers` object".to_owned());
    };
    if answers.keys().any(|name| !questions.contains_key(name)) {
        return Err("the response answers a question that was not sent".to_owned());
    }
    let mut parsed = BTreeMap::new();
    for (name, question) in questions {
        let raw = answers
            .get(name)
            .ok_or_else(|| format!("question \"{name}\": no answer"))?;
        let answer = read_answer(question, raw)
            .map_err(|problem| format!("question \"{name}\": {problem}"))?;
        parsed.insert(name.clone(), answer);
    }
    let usage = read_usage(root.get("usage")).map_err(str::to_owned)?;
    Ok(ParsedResponse {
        answers: parsed,
        usage,
    })
}

/// Consumption reported by a response that failed validation, so a spent attempt is still
/// counted (REQ-D10 `usage`). A body that is not JSON or has no usable `usage` counts as zero.
///
/// Complexity: O(body).
pub(crate) fn salvage_usage(body: &[u8]) -> Usage {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|root| read_usage(root.get("usage")).ok())
        .unwrap_or_default()
}

/// Lowest Magi confidence across the answers (1 when there are none).
///
/// Complexity: O(answers).
///
/// # Arguments
/// * `answers` - the answers of one attempt, by question name.
///
/// # Returns
/// The smallest [`Answer::confidence`], or 1.0 for an empty map.
pub fn lowest_confidence(answers: &BTreeMap<String, Answer>) -> f64 {
    answers.values().map(Answer::confidence).fold(1.0, f64::min)
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
            "questions": {"triage": {"type": "choice", "instructions": "Which one?",
                "criteria": {"weird key": "a", "another one": "b"}}}
        }))
        .expect("state content and option names are data (REQ-D23)");
    }

    #[test]
    fn noul_criteria_accept_only_true_and_false_keys() {
        parse(
            json!({"state": "x", "questions": {"obstacle": {"type": "noul",
            "instructions": "Is there an obstacle ahead?",
            "criteria": {"true": "closer than the safety distance", "false": "path clear"}}}}),
        )
        .expect("valid");
        let r = reason(parse(json!({"state": "x",
            "questions": {"obstacle": {"type": "noul", "criteria": {"maybe": "?"}}}})));
        assert!(r.contains("maybe"), "{r}");
    }

    #[test]
    fn with_a_questions_file_the_input_carries_only_state_and_model() {
        let questions = json!({"next_action": {"type": "choice", "instructions": "What next?",
                "criteria": {"forward": "f", "stop": "s"}}});
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
    fn with_model_trims_and_treats_a_blank_model_as_absent() {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        let base = DecisionRequest::new(json!("s"), questions).expect("valid");
        assert_eq!(base.clone().with_model("   ").model(), None);
        assert_eq!(base.with_model(" tev1 ").model(), Some("tev1"));
    }

    #[test]
    fn the_wire_body_carries_model_state_questions_and_keep_alive_only_when_declared() {
        let req = parse(json!({"state": "s", "questions": noul()})).expect("valid");
        let body = req.wire_body("nimble", None).unwrap();
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
        let body = req
            .wire_body("nimble", Some(&KeepAlive::Text("10m".into())))
            .unwrap();
        assert_eq!(body["keep_alive"], "10m");
        let body = req
            .wire_body("nimble", Some(&KeepAlive::Seconds(-1)))
            .unwrap();
        assert_eq!(body["keep_alive"], -1);
    }

    /// A GitHub-shaped token `mask_secrets` recognizes.
    const SECRET_TOKEN: &str = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan

    /// REQ-F12 / F-S18: a secret-shaped token inside a string of an object inside an array never
    /// reaches the wire body; the mask does.
    #[test]
    fn a_secret_nested_in_the_state_is_masked_before_it_reaches_the_wire() {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        let state = json!({"items": [{"note": format!("token {SECRET_TOKEN} here")}, 7]});
        let req = DecisionRequest::new(state, questions).expect("valid");
        let wire = req.wire_body("nimble", None).expect("body").to_string();
        assert!(!wire.contains(SECRET_TOKEN), "{wire}");
        assert!(wire.contains("<redacted:"), "{wire}");
        let note = req.state()["items"][0]["note"].as_str().expect("note");
        assert!(
            note.starts_with("token ") && note.ends_with(" here"),
            "{note}"
        );
        assert_eq!(req.state()["items"][1], 7, "numbers are untouched");
    }

    /// REQ-F12: a string state and strings directly in an array are masked too.
    #[test]
    fn a_secret_in_a_string_state_or_an_array_of_strings_is_masked() {
        for state in [json!(SECRET_TOKEN), json!(["ok", SECRET_TOKEN])] {
            let questions: Questions = serde_json::from_value(noul()).expect("map");
            let req = DecisionRequest::new(state, questions).expect("valid");
            let wire = req.wire_body("nimble", None).expect("body").to_string();
            assert!(!wire.contains(SECRET_TOKEN), "{wire}");
            assert!(wire.contains("<redacted:"), "{wire}");
        }
    }

    /// REQ-F12: object keys are left alone (masking them could collapse two entries), and plain
    /// strings, numbers, booleans and null travel as they are. A key holding a secret is refused
    /// instead (`a_state_with_a_secret_only_in_a_key_is_refused_without_quoting_it`).
    #[test]
    fn object_keys_and_values_without_secrets_are_left_unchanged() {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        let state = json!({"plain key": "plain text", "n": 3, "b": true, "z": null});
        let req = DecisionRequest::new(state.clone(), questions).expect("valid");
        assert_eq!(req.state(), &state);
    }

    /// A private-key block: its header, one body line and its footer.
    fn key_block() -> String {
        let d = "-----";
        format!("{d}BEGIN RSA PRIVATE KEY{d}\nMIIEowIBAAKCAQEAbodyline\n{d}END RSA PRIVATE KEY{d}")
    }

    /// The reason of an `InvalidInput` from [`DecisionRequest::new`].
    fn new_reason(state: serde_json::Value) -> String {
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        match DecisionRequest::new(state, questions) {
            Err(DecisionError::InvalidInput { reason }) => reason,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// REQ-F12 (fail closed, mutation-verified): a key block nested in a value is refused, and
    /// the reason quotes none of it; a key body matches no pattern, so masking cannot be trusted.
    #[test]
    fn a_state_with_a_private_key_block_in_a_value_is_refused_without_quoting_it() {
        let reason = new_reason(json!({"items": [{"note": format!("see {}", key_block())}]}));
        assert!(!reason.contains("BEGIN"), "{reason}");
        assert!(!reason.contains("bodyline"), "{reason}");
        assert!(reason.contains("private-key block"), "{reason}");
    }

    /// REQ-F12b (mutation-verified): a key block in an object key is refused by the key-block
    /// check itself, before masking (keys are never masked), with its own fixed reason.
    #[test]
    fn a_state_with_a_private_key_block_in_a_key_is_refused() {
        let mut fields = serde_json::Map::new();
        fields.insert(key_block(), json!("plain"));
        let reason = new_reason(serde_json::Value::Object(fields));
        assert!(!reason.contains("bodyline"), "{reason}");
        assert_eq!(reason, KEY_BLOCK_REFUSED);
    }

    /// Mutation-verified: a key block whose header and body sit in two fields of the state is
    /// refused (the header alone already opens a block); two clean fields are accepted as they
    /// are. Passes before the joined check by design: it pins that the joined check keeps it.
    #[test]
    fn a_state_with_a_key_block_split_across_two_fields_is_refused() {
        let d = "-----";
        let reason = new_reason(json!({
            "title": format!("{d}BEGIN RSA PRIVATE KEY{d}"),
            "detail": "MIIEowIBAAKCAQEAbodyline",
        }));
        assert_eq!(reason, KEY_BLOCK_REFUSED);
        let questions: Questions = serde_json::from_value(noul()).expect("map");
        let clean = json!({"title": "plain", "detail": "also plain"});
        let req = DecisionRequest::new(clean.clone(), questions).expect("valid");
        assert_eq!(req.state(), &clean);
    }

    /// REQ-F12 (fail closed, mutation-verified): a secret masking cannot remove, here one in an
    /// object key, is refused and never echoed.
    #[test]
    fn a_state_with_a_secret_only_in_a_key_is_refused_without_quoting_it() {
        let reason = new_reason(json!({SECRET_TOKEN: "plain text"}));
        assert!(!reason.contains(SECRET_TOKEN), "{reason}");
        assert!(reason.contains("secret"), "{reason}");
    }

    /// Mutation-verified: a state nested past `MAX_SCAN_DEPTH` (built in process, so no parser
    /// limit applies) is refused, never walked to its end or sent.
    #[test]
    fn a_state_nested_past_the_depth_bound_is_refused() {
        let mut state = json!("plain");
        for _ in 0..200 {
            state = serde_json::Value::Array(vec![state]);
        }
        let reason = new_reason(state);
        assert!(reason.contains("never sent"), "{reason}");
    }

    /// The reason of an `InvalidInput` from [`DecisionRequest::new`] for a plain state and the
    /// questions `questions`.
    fn questions_reason(questions: serde_json::Value) -> String {
        let questions: Questions = serde_json::from_value(questions).expect("map");
        match DecisionRequest::new(json!({"ticket": "plain"}), questions) {
            Err(DecisionError::InvalidInput { reason }) => reason,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// REQ-F12 (fail closed, mutation-verified): the questions travel with the state but are
    /// never masked (a masked name would break the answer mapping), so a secret in an
    /// instruction refuses the request and is never echoed.
    #[test]
    fn questions_with_a_secret_in_an_instruction_are_refused_without_quoting_it() {
        let reason = questions_reason(json!({"triage": {"type": "noul",
            "instructions": format!("Is {SECRET_TOKEN} urgent?")}}));
        assert!(!reason.contains(SECRET_TOKEN), "{reason}");
        assert!(reason.contains("questions"), "{reason}");
    }

    /// REQ-F12 (mutation-verified): the egress scan runs before validation, so a malformed
    /// question whose name holds a secret is refused with the fixed reason; the validation
    /// reason, which quotes the name, is never built from it.
    #[test]
    fn a_malformed_question_with_a_secret_name_is_refused_before_validation_quotes_it() {
        let questions: Questions = serde_json::from_value(json!({SECRET_TOKEN: {
            "type": "choice", "instructions": "Which kind?"}}))
        .expect("map");
        let err = DecisionRequest::new(json!({"ticket": "plain"}), questions)
            .expect_err("a malformed question is refused");
        assert!(!err.to_string().contains(SECRET_TOKEN), "{err}");
        match err {
            DecisionError::InvalidInput { reason } => {
                assert_eq!(reason, QUESTIONS_SECRET_REFUSED);
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// REQ-F12b: a question that fails parsing (an unknown `type`) and whose name holds a secret
    /// is refused before the parse error, which quotes the name, is built.
    #[test]
    fn an_unparsable_question_with_a_secret_name_is_refused_before_the_parse_error() {
        let input = json!({"state": {"ticket": "plain"},
            "questions": {SECRET_TOKEN: {"type": "nonsense", "instructions": "Which?"}}});
        let err = parse_decide_input(input.to_string().as_bytes(), None).expect_err("refused");
        assert!(!err.to_string().contains(SECRET_TOKEN), "{err}");
        assert!(
            matches!(&err, DecisionError::InvalidInput { reason } if reason == QUESTIONS_SECRET_REFUSED),
            "{err:?}"
        );
    }

    /// REQ-F12b: an explicit `model` that holds a secret is refused before it is sent or quoted
    /// (`ModelUnavailable` names the model).
    #[test]
    fn an_explicit_model_holding_a_secret_is_refused_without_quoting_it() {
        let input = json!({"state": {"ticket": "plain"}, "model": SECRET_TOKEN,
            "questions": {"q": {"type": "noul", "instructions": "Is it?"}}});
        let err = parse_decide_input(input.to_string().as_bytes(), None).expect_err("refused");
        assert!(!err.to_string().contains(SECRET_TOKEN), "{err}");
        assert!(
            matches!(&err, DecisionError::InvalidInput { reason }
                if reason == MODEL_SECRET_REFUSED),
            "{err:?}"
        );
    }

    /// REQ-F12b: an unknown top-level field whose name holds a secret is refused without the
    /// `unknown field` message, which quotes the name.
    #[test]
    fn an_unknown_input_field_with_a_secret_name_is_refused_without_quoting_it() {
        let input = json!({"state": {"ticket": "plain"}, SECRET_TOKEN: 1,
            "questions": {"q": {"type": "noul", "instructions": "Is it?"}}});
        let err = parse_decide_input(input.to_string().as_bytes(), None).expect_err("refused");
        assert!(!err.to_string().contains(SECRET_TOKEN), "{err}");
        assert!(
            matches!(&err, DecisionError::InvalidInput { reason } if reason == INPUT_FIELD_SECRET_REFUSED),
            "{err:?}"
        );
    }

    /// REQ-F12: a private-key block in a criteria value refuses the request too.
    #[test]
    fn questions_with_a_private_key_block_in_a_criteria_value_are_refused() {
        let reason = questions_reason(json!({"kind": {"type": "choice",
            "instructions": "Which kind?",
            "criteria": {"bug": key_block(), "feature": "New"}}}));
        assert!(!reason.contains("bodyline"), "{reason}");
        assert!(reason.contains("questions"), "{reason}");
    }

    /// REQ-F12: the measured internal questions (the fixture the adapters ship) pass the check.
    #[test]
    fn the_internal_questions_of_the_fixture_are_accepted() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/decide/internal-questions.json"
        ))
        .expect("fixture");
        for name in ["mode", "contradicts"] {
            let questions: Questions =
                serde_json::from_value(fixture[name]["questions"].clone()).expect("map");
            let state = fixture[name]["state"].clone();
            assert!(DecisionRequest::new(state, questions).is_ok(), "{name}");
        }
    }

    /// A GitHub-shaped token split by a zero-width space: no pattern matches it as it is, so
    /// masking leaves it, and only the egress predicate (`leaks_secret`) sees it.
    fn split_token() -> String {
        format!("see ghp_AAAA\u{200b}{} here", "A".repeat(40)) // allow-secret-scan
    }

    /// REQ-F12 (mutation-verified): a token split by a hidden character survives masking, so the
    /// state is refused by the residual check.
    #[test]
    fn a_state_with_a_token_split_by_a_hidden_character_is_refused() {
        let reason = new_reason(json!({"note": split_token()}));
        assert!(!reason.contains(&"A".repeat(40)), "{reason}");
    }

    /// REQ-F12 (mutation-verified): the same split token in an instruction refuses the questions.
    #[test]
    fn questions_with_a_token_split_by_a_hidden_character_are_refused() {
        let reason = questions_reason(json!({"triage": {"type": "noul",
            "instructions": split_token()}}));
        assert!(reason.contains("questions"), "{reason}");
    }

    /// REQ-F12: no text the review's egress predicate withholds leaves through `decide`, in a
    /// state value or in a question; the sibling of
    /// `secrets::tests::no_egress_predicate_is_weaker_than_leaks_secret` for this egress.
    #[test]
    fn no_text_of_the_withheld_corpus_is_sent_by_decide() {
        for text in crate::review::secrets::withheld_corpus() {
            let questions: Questions = serde_json::from_value(noul()).expect("map");
            assert!(
                DecisionRequest::new(json!({"note": text}), questions).is_err(),
                "state: {text:?}"
            );
            let questions: Questions = serde_json::from_value(json!({"triage": {
                "type": "noul", "instructions": text}}))
            .expect("map");
            assert!(
                DecisionRequest::new(json!("plain"), questions).is_err(),
                "question: {text:?}"
            );
        }
    }

    /// Review Focus 2.
    #[test]
    fn an_empty_question_name_is_invalid_input() {
        let r = reason(parse(
            json!({"state": "x", "questions": {"": {"type": "noul"}}}),
        ));
        assert!(r.contains("question \"\""), "{r}");
    }

    /// One question of each type, complete except for `instructions`.
    fn question_without_instructions(kind: &str) -> serde_json::Value {
        match kind {
            "noul" => json!({"type": "noul"}),
            "choice" => json!({"type": "choice", "criteria": {"bug": "Broken", "feature": "New"}}),
            _ => json!({"type": "score", "criteria": ["Routine", "Urgent"]}),
        }
    }

    /// User decision 2026-10-10: every question type needs `instructions`, rejected locally
    /// with a message naming the question and the field.
    #[test]
    fn a_question_of_any_type_without_instructions_is_invalid_input_naming_it() {
        for kind in ["noul", "choice", "score"] {
            let r = reason(parse(json!({"state": "x",
                "questions": {"team": question_without_instructions(kind)}})));
            assert_eq!(
                r,
                "question \"team\": `instructions` is required \
                 (a non-empty description of what to decide)",
                "{kind}"
            );
        }
    }

    #[test]
    fn empty_or_blank_instructions_are_invalid_input() {
        for kind in ["noul", "choice", "score"] {
            for text in ["", "   ", "\t\n "] {
                let mut question = question_without_instructions(kind);
                question["instructions"] = json!(text);
                let r = reason(parse(
                    json!({"state": "x", "questions": {"team": question}}),
                ));
                assert!(
                    r.contains("question \"team\"") && r.contains("instructions"),
                    "{kind} {text:?}: {r}"
                );
            }
        }
    }

    #[test]
    fn new_rejects_a_question_without_instructions_like_the_parser() {
        let questions: Questions =
            serde_json::from_value(json!({"team": question_without_instructions("noul")}))
                .expect("map");
        let r = reason(DecisionRequest::new(json!("s"), questions));
        assert!(r.contains("question \"team\""), "{r}");
    }

    /// Review Focus 2: names are user data and travel as they are.
    #[test]
    fn question_and_option_names_with_quotes_newlines_and_unicode_are_data() {
        let req = parse(json!({"state": "x", "questions": {"a\"b
ñ": {"type": "choice", "instructions": "Which one?",
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

    fn three_questions() -> Questions {
        serde_json::from_value(json!({
            "team": {"type": "choice", "criteria": {"billing": "b", "technical": "t", "other": "o"}},
            "today": {"type": "noul", "instructions": "Is it today?"},
            "urgency": {"type": "score", "criteria": ["Routine", "Soon", "Urgent"]}
        }))
        .expect("questions")
    }

    /// Values measured with `nimble` on 2026-10-01 (spec REQ-D10 example).
    const NIMBLE_BODY: &str = r#"{"model":"nimble","answers":{
        "team":{"type":"choice","choice":"billing","confidence":0.948,
                "probabilities":{"billing":0.990,"technical":0.001,"other":0.009}},
        "today":{"type":"noul","noul":0.824},
        "urgency":{"type":"score","score":1.011,"confidence":0.009,
                   "legend":{"0":"Routine","1":"Soon","2":"Urgent"},
                   "probabilities":{"0":0.361,"1":0.268,"2":0.372}}},
        "usage":{"input_tokens":526,"output_tokens":3}}"#;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn body_with(answers: serde_json::Value) -> Vec<u8> {
        json!({"model": "m", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})
            .to_string()
            .into_bytes()
    }

    #[test]
    fn a_valid_response_yields_one_answer_per_question_with_magi_confidence() {
        let parsed = parse_response(NIMBLE_BODY.as_bytes(), &three_questions()).expect("valid");
        match parsed.answers.get("team") {
            Some(Answer::Choice {
                choice,
                confidence,
                provider_confidence,
                probabilities,
            }) => {
                assert_eq!(choice, "billing");
                assert!(close(*confidence, (3.0 * 0.990 - 1.0) / 2.0));
                assert_eq!(*provider_confidence, Some(0.948));
                assert_eq!(probabilities.len(), 3);
            }
            other => panic!("team: {other:?}"),
        }
        match parsed.answers.get("today") {
            Some(Answer::Noul { noul, confidence }) => {
                assert!(close(*noul, 0.824));
                assert!(close(*confidence, (2.0f64 * 0.824 - 1.0).abs()));
            }
            other => panic!("today: {other:?}"),
        }
        match parsed.answers.get("urgency") {
            Some(Answer::Score {
                score,
                legend,
                confidence,
                provider_confidence,
                ..
            }) => {
                assert!(close(*score, 1.011));
                assert_eq!(legend.get("2").map(String::as_str), Some("Urgent"));
                assert!(close(*confidence, (3.0 * 0.372 - 1.0) / 2.0));
                assert_eq!(*provider_confidence, Some(0.009));
            }
            other => panic!("urgency: {other:?}"),
        }
        assert_eq!(
            parsed.usage,
            Usage {
                input_tokens: 526,
                output_tokens: 3,
                cost: None
            }
        );
    }

    /// PM-C3: an OpenRouter response plus an invented field still parses; cost is read.
    #[test]
    fn unknown_fields_in_the_response_are_ignored_and_cost_is_read() {
        let body = json!({"id": "gen-1", "provider": "TypeSafe", "model": "typesafe/jev-1.13",
            "invented": {"x": 1},
            "answers": {"today": {"type": "noul", "noul": 0.9, "extra": true}},
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.00002}});
        let qs: Questions = serde_json::from_value(json!({"today": {"type": "noul"}})).expect("q");
        let parsed = parse_response(body.to_string().as_bytes(), &qs).expect("tolerant read");
        assert_eq!(parsed.usage.cost, Some(0.00002));
    }

    #[test]
    fn an_absent_usage_counts_as_zero() {
        let qs: Questions = serde_json::from_value(json!({"today": {"type": "noul"}})).expect("q");
        let body = json!({"model": "m", "answers": {"today": {"type": "noul", "noul": 0.5}}});
        let parsed = parse_response(body.to_string().as_bytes(), &qs).expect("valid");
        assert_eq!(parsed.usage, Usage::default());
    }

    #[test]
    fn a_choice_outside_the_sent_options_is_invalid_and_the_reason_does_not_quote_it() {
        let planted = "sk-planted-option-0123456789abcdef";
        let answers = json!({"team": {"type": "choice", "choice": planted,
            "probabilities": {"billing": 0.5, "technical": 0.3, "other": 0.2}}});
        let qs: Questions = serde_json::from_value(json!({"team": {"type": "choice",
            "criteria": {"billing": "b", "technical": "t", "other": "o"}}}))
        .expect("q");
        let r = parse_response(&body_with(answers), &qs).expect_err("invalid");
        assert!(r.contains("team"), "{r}");
        assert!(
            !r.contains(planted),
            "a reason never quotes a provider value: {r}"
        );
    }

    #[test]
    fn probability_keys_must_match_the_options_or_level_indices() {
        let qs = three_questions();
        let bad_choice = json!({
            "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "nope": 0.1}},
            "today": {"type": "noul", "noul": 0.5},
            "urgency": {"type": "score", "score": 1.0, "legend": {}, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}}});
        assert!(parse_response(&body_with(bad_choice), &qs).is_err());
        let bad_score = json!({
            "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.1, "other": 0.1}},
            "today": {"type": "noul", "noul": 0.5},
            "urgency": {"type": "score", "score": 1.0, "legend": {},
                        "probabilities": {"Routine": 0.2, "Soon": 0.6, "Urgent": 0.2}}});
        assert!(parse_response(&body_with(bad_score), &qs).is_err());
    }

    #[test]
    fn numeric_violations_are_invalid_responses() {
        let qs: Questions = serde_json::from_value(json!({"today": {"type": "noul"},
            "pick": {"type": "choice", "criteria": {"a": "a", "b": "b"}}}))
        .expect("q");
        let mk = |noul: serde_json::Value, pa: serde_json::Value, pb: serde_json::Value| {
            json!({"today": {"type": "noul", "noul": noul},
                   "pick": {"type": "choice", "choice": "a", "probabilities": {"a": pa, "b": pb}}})
        };
        assert!(
            parse_response(&body_with(mk(json!(1.2), json!(0.5), json!(0.5))), &qs).is_err(),
            "noul > 1"
        );
        assert!(
            parse_response(&body_with(mk(json!(-0.1), json!(0.5), json!(0.5))), &qs).is_err(),
            "noul < 0"
        );
        assert!(
            parse_response(&body_with(mk(json!(0.5), json!(-0.2), json!(1.2))), &qs).is_err(),
            "p < 0"
        );
        assert!(
            parse_response(&body_with(mk(json!(0.5), json!(0.5), json!(0.3))), &qs).is_err(),
            "sum 0.8"
        );
        assert!(
            parse_response(&body_with(mk(json!(0.5), json!(0.6), json!(0.405))), &qs).is_ok(),
            "sum 1.005"
        );
        let nan = br#"{"model":"m","answers":{"today":{"type":"noul","noul":NaN}}}"#;
        assert!(
            parse_response(nan, &qs).is_err(),
            "NaN is not JSON and never reaches a decision"
        );
    }

    /// CP2 MS1 r3 (Caspar): structural violations and the score's own probability sum are invalid too.
    #[test]
    fn structural_violations_are_invalid_responses() {
        let qs = three_questions();
        assert!(parse_response(b"", &qs).is_err(), "empty body");
        assert!(
            parse_response(b"[]", &qs).is_err(),
            "top level is not an object"
        );
        assert!(
            parse_response(br#"{"model":"m","answers":[]}"#, &qs).is_err(),
            "answers as an array"
        );
        let team = json!({"type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.8, "technical": 0.1, "other": 0.1}});
        let legend = json!({"0": "Routine", "1": "Soon", "2": "Urgent"});
        let missing = json!({"team": team, "today": {"type": "noul", "noul": 0.5}});
        assert!(
            parse_response(&body_with(missing), &qs).is_err(),
            "a sent question without an answer"
        );
        let score_sum = json!({"team": team, "today": {"type": "noul", "noul": 0.5},
            "urgency": {"type": "score", "score": 1.0, "legend": legend,
                        "probabilities": {"0": 0.5, "1": 0.5, "2": 0.5}}});
        assert!(
            parse_response(&body_with(score_sum), &qs).is_err(),
            "score probabilities sum 1.5"
        );
    }

    /// Carry item 6: the upper bound is the binding one; the lower bound is tested too.
    #[test]
    fn a_score_is_checked_against_its_tolerant_range_then_clamped() {
        let qs: Questions = serde_json::from_value(json!({"urgency": {"type": "score",
            "criteria": ["Routine", "Soon", "Urgent"]}}))
        .expect("q");
        let mk = |score: f64| {
            json!({"urgency": {"type": "score", "score": score, "legend": {},
            "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}}})
        };
        let parsed = parse_response(&body_with(mk(2.015)), &qs).expect("within 2 * 1.01");
        assert!(
            matches!(parsed.answers.get("urgency"), Some(Answer::Score { score, .. }) if close(*score, 2.0))
        );
        assert!(
            parse_response(&body_with(mk(2.05)), &qs).is_err(),
            "above (n-1)(1+tol)"
        );
        let parsed = parse_response(&body_with(mk(-0.01)), &qs).expect("within -tol(n-1)");
        assert!(
            matches!(parsed.answers.get("urgency"), Some(Answer::Score { score, .. }) if close(*score, 0.0))
        );
        assert!(
            parse_response(&body_with(mk(-0.03)), &qs).is_err(),
            "below -tol(n-1)"
        );
    }

    #[test]
    fn the_legend_comes_from_the_sent_levels_not_from_the_provider() {
        let qs: Questions = serde_json::from_value(json!({"urgency": {"type": "score",
            "criteria": ["Routine", "Urgent"]}}))
        .expect("q");
        let answers = json!({"urgency": {"type": "score", "score": 1.0,
            "legend": {"0": "sk-planted-legend-0123456789abcdef", "1": "x"},
            "probabilities": {"0": 0.1, "1": 0.9}}});
        let parsed = parse_response(&body_with(answers), &qs).expect("valid");
        match parsed.answers.get("urgency") {
            Some(Answer::Score { legend, .. }) => {
                assert_eq!(legend.get("0").map(String::as_str), Some("Routine"));
                assert_eq!(legend.get("1").map(String::as_str), Some("Urgent"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_missing_answer_an_unasked_one_or_a_wrong_type_is_invalid() {
        let qs: Questions = serde_json::from_value(json!({"today": {"type": "noul"}})).expect("q");
        assert!(
            parse_response(&body_with(json!({})), &qs).is_err(),
            "missing"
        );
        assert!(
            parse_response(
                &body_with(json!({"today": {"type": "noul", "noul": 0.5},
            "other": {"type": "noul", "noul": 0.5}})),
                &qs
            )
            .is_err(),
            "unasked"
        );
        assert!(
            parse_response(
                &body_with(json!({"today": {"type": "choice", "choice": "a",
            "probabilities": {"a": 1.0}}})),
                &qs
            )
            .is_err(),
            "wrong type"
        );
    }

    #[test]
    fn a_choice_without_provider_confidence_omits_it_in_the_output() {
        let qs: Questions = serde_json::from_value(json!({"pick": {"type": "choice",
            "criteria": {"a": "a", "b": "b"}}}))
        .expect("q");
        let parsed = parse_response(
            &body_with(json!({"pick": {"type": "choice", "choice": "a",
            "probabilities": {"a": 0.8, "b": 0.2}}})),
            &qs,
        )
        .expect("valid");
        let out = serde_json::to_value(parsed.answers.get("pick").expect("pick")).expect("ser");
        assert!(out.get("provider_confidence").is_none(), "{out}");
        assert_eq!(out["type"], "choice");
    }

    #[test]
    fn a_body_that_is_not_json_is_invalid() {
        assert!(parse_response(b"<html>502</html>", &three_questions()).is_err());
    }

    #[test]
    fn magi_confidence_follows_the_formula_and_is_clamped_to_zero_one() {
        assert!(close(magi_confidence(&[0.5, 0.5]), 0.0));
        assert!(close(magi_confidence(&[1.0, 0.0]), 1.0));
        assert!(close(magi_confidence(&[0.7, 0.2, 0.1]), 0.55));
        assert!(close(magi_confidence(&[0.824, 0.176]), 0.648));
        // p_max just under 1/n (allowed by the sum tolerance) never goes negative.
        assert!(close(magi_confidence(&[0.332, 0.331, 0.332]), 0.0));
    }

    /// Carry item 7 lives in MS2 (`noul >= 0.5` counts as yes); here: exactly 0.5 has zero confidence.
    #[test]
    fn a_noul_of_exactly_one_half_has_zero_confidence() {
        let qs: Questions = serde_json::from_value(json!({"today": {"type": "noul"}})).expect("q");
        let parsed = parse_response(
            &body_with(json!({"today": {"type": "noul", "noul": 0.5}})),
            &qs,
        )
        .expect("valid");
        assert!(close(
            parsed
                .answers
                .get("today")
                .map(Answer::confidence)
                .unwrap_or(1.0),
            0.0
        ));
    }

    #[test]
    fn lowest_confidence_is_the_minimum_across_answers() {
        let parsed = parse_response(NIMBLE_BODY.as_bytes(), &three_questions()).expect("valid");
        assert!(close(
            lowest_confidence(&parsed.answers),
            (3.0 * 0.372 - 1.0) / 2.0
        ));
    }

    #[test]
    fn usage_add_sums_tokens_and_keeps_cost_none_only_if_neither_has_one() {
        let mut a = Usage {
            input_tokens: 1,
            output_tokens: 2,
            cost: None,
        };
        a.add(&Usage {
            input_tokens: 3,
            output_tokens: 4,
            cost: None,
        });
        assert_eq!(
            a,
            Usage {
                input_tokens: 4,
                output_tokens: 6,
                cost: None
            }
        );
        a.add(&Usage {
            input_tokens: 0,
            output_tokens: 0,
            cost: Some(0.5),
        });
        a.add(&Usage {
            input_tokens: 0,
            output_tokens: 0,
            cost: Some(0.25),
        });
        assert_eq!(a.cost, Some(0.75));
    }
}
