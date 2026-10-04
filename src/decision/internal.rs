// Author: Julian Bolivar
// Version: 0.23.0
// Date: 2026-10-04

//! Internal uses of `decide`: the mode and contradiction questions, the closed
//! set of fallback causes and the run-log record of an internal fallback
//! (REQ-D12, REQ-D13, REQ-D19).

use magi_core::schema::Mode;
use serde_json::json;

use crate::decision::error::{AttemptOutcome, Cause, DecisionError};
use crate::decision::protocol::{DecisionRequest, NoulCriteria, Question, Questions};

/// Log target of the internal-fallback entries (REQ-D12).
pub const FALLBACK_LOG_TARGET: &str = "magi_rs::decision::fallback";
/// Name of the agent tool that asks for a decision.
pub const DECIDE_TOOL_NAME: &str = "decide";
/// Marker that replaces the `state` in the recorded tool input (REQ-D19).
pub const STATE_OMITTED: &str = "<omitted>";
/// D-PM-1: initial value (Ollama's 64 KiB body cap minus the rest of the request); T3 measures it and
/// may lower it (PM-S-2). The final value and its measurement live in the ledger.
pub const MODE_STATE_LIMIT_BYTES: usize = 19_000;
/// REQ-D13: a `noul` at or above this counts as "contradicts" (exactly 0.5 is yes).
pub const NOUL_YES_THRESHOLD: f64 = 0.5;

/// Name of the single mode question.
const MODE_QUESTION_NAME: &str = "mode";
/// Instructions of the mode question (Ollama rejects a question without them).
const MODE_INSTRUCTIONS: &str = "Which review lens fits this request?";
/// Option label of the code-review mode.
const MODE_LABEL_CODE_REVIEW: &str = "code-review";
/// Option label of the design mode.
const MODE_LABEL_DESIGN: &str = "design";
/// Option label of the analysis mode.
const MODE_LABEL_ANALYSIS: &str = "analysis";
/// Mode options with the descriptions the model sees.
const MODE_LABELS: [(&str, &str); 3] = [
    (
        MODE_LABEL_CODE_REVIEW,
        "Find defects in concrete source code or a diff",
    ),
    (
        MODE_LABEL_DESIGN,
        "Evaluate an architecture, API or design proposal",
    ),
    (
        MODE_LABEL_ANALYSIS,
        "A general question or analysis that is neither",
    ),
];
/// Name of the single contradiction question.
const CONTRADICTS_QUESTION_NAME: &str = "contradicts";
/// Instructions of the contradiction question.
const CONTRADICTS_INSTRUCTIONS: &str = concat!(
    "Does statement B contradict statement A, or replace it with a different value, ",
    "about the same subject?"
);
/// Meaning of "true" for the contradiction question.
const CONTRADICTS_YES: &str = "B contradicts A or replaces it with a different value";
/// Meaning of "false" for the contradiction question.
const CONTRADICTS_NO: &str = "B agrees with A, adds to it, or is about something else";
/// Field of a `decide` tool input holding the state.
const STATE_FIELD: &str = "state";

/// Who asked for an internal decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consumer {
    /// Review-mode inference.
    Mode,
    /// Memory contradiction judgment.
    Contradicts,
}

impl Consumer {
    /// The log token of this consumer.
    ///
    /// # Returns
    /// `"mode"` or `"contradicts"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Consumer::Mode => "mode",
            Consumer::Contradicts => "contradicts",
        }
    }
}

/// Why an internal decision fell back to the LLM (closed set, REQ-D12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackCause {
    /// An attempt of the chain failed with this cause.
    Attempt(Cause),
    /// The chain ended below its confidence thresholds.
    LowConfidence,
    /// The content exceeded the mode state limit.
    TooLarge,
    /// The client could not be built.
    ClientUnavailable,
    /// An earlier pair of the same pass failed on transport.
    SkippedAfterFailure,
}

impl FallbackCause {
    /// The log token of this cause.
    ///
    /// # Returns
    /// A token of the closed set.
    pub fn as_str(self) -> &'static str {
        match self {
            FallbackCause::Attempt(cause) => cause.as_str(),
            FallbackCause::LowConfidence => "low_confidence",
            FallbackCause::TooLarge => "too_large",
            FallbackCause::ClientUnavailable => "client_unavailable",
            FallbackCause::SkippedAfterFailure => "skipped_after_failure",
        }
    }

    /// Derives the cause from the error of a decision.
    ///
    /// # Arguments
    ///
    /// * `error` - The error the client returned.
    ///
    /// # Returns
    /// The cause that explains the fallback.
    pub fn from_error(error: &DecisionError) -> Self {
        match error {
            DecisionError::InvalidInput { .. } | DecisionError::Rejected { .. } => {
                FallbackCause::Attempt(Cause::Rejected)
            }
            DecisionError::Unauthorized { .. } | DecisionError::MissingVaultEntry { .. } => {
                FallbackCause::Attempt(Cause::Unauthorized)
            }
            DecisionError::ClientUnavailable { .. } => FallbackCause::ClientUnavailable,
            DecisionError::ModelUnavailable { trace, .. }
            | DecisionError::ChainExhausted { trace } => {
                match trace.attempts.last().map(|a| &a.outcome) {
                    Some(AttemptOutcome::LowConfidence { .. }) => FallbackCause::LowConfidence,
                    Some(AttemptOutcome::Failed { cause, .. }) => FallbackCause::Attempt(*cause),
                    Some(AttemptOutcome::Answered) | None => FallbackCause::ClientUnavailable,
                }
            }
        }
    }
}

/// Writes one INFO entry for an internal fallback (never the `state`).
///
/// # Arguments
///
/// * `consumer` - Who asked.
/// * `cause` - Why it fell back.
/// * `endpoint` - Redacted endpoint.
/// * `detail` - Detail generated by Magi, if any.
pub fn log_fallback(
    consumer: Consumer,
    cause: FallbackCause,
    endpoint: &str,
    detail: Option<&str>,
) {
    tracing::event!(
        target: FALLBACK_LOG_TARGET,
        tracing::Level::INFO,
        consumer = consumer.as_str(),
        cause = cause.as_str(),
        endpoint = endpoint,
        detail = detail.unwrap_or(""),
        "internal decision fell back to the LLM"
    );
}

/// The mode question (REQ-D12).
///
/// # Returns
/// One `choice` question over the three review modes.
pub fn mode_questions() -> Questions {
    let criteria = MODE_LABELS
        .iter()
        .map(|(label, description)| ((*label).to_owned(), (*description).to_owned()))
        .collect();
    Questions::from([(
        MODE_QUESTION_NAME.to_owned(),
        Question::Choice {
            instructions: Some(MODE_INSTRUCTIONS.to_owned()),
            criteria: Some(criteria),
        },
    )])
}

/// Maps the chosen label to a mode.
///
/// # Arguments
///
/// * `label` - The option the model chose.
///
/// # Returns
/// The mode, or `None` for any other label.
pub fn mode_from_label(label: &str) -> Option<magi_core::schema::Mode> {
    match label {
        MODE_LABEL_CODE_REVIEW => Some(Mode::CodeReview),
        MODE_LABEL_DESIGN => Some(Mode::Design),
        MODE_LABEL_ANALYSIS => Some(Mode::Analysis),
        _ => None,
    }
}

/// The contradiction request (REQ-D13).
///
/// # Arguments
///
/// * `older` - The older memory.
/// * `newer` - The newer memory.
///
/// # Errors
/// Never in practice; validation errors are propagated.
pub fn contradiction_request(older: &str, newer: &str) -> Result<DecisionRequest, DecisionError> {
    let questions = Questions::from([(
        CONTRADICTS_QUESTION_NAME.to_owned(),
        Question::Noul {
            instructions: Some(CONTRADICTS_INSTRUCTIONS.to_owned()),
            criteria: Some(NoulCriteria {
                yes: Some(CONTRADICTS_YES.to_owned()),
                no: Some(CONTRADICTS_NO.to_owned()),
            }),
        },
    )]);
    DecisionRequest::new(json!({ "A": older, "B": newer }), questions)
}

/// The tool input as stored in `tool_calls[]` (REQ-D19).
///
/// # Arguments
///
/// * `tool_name` - Tool invoked.
/// * `input` - Its raw input.
///
/// # Returns
/// The input, with the `state` of a `decide` call replaced by a marker.
pub fn tool_call_record_input(tool_name: &str, input: &serde_json::Value) -> serde_json::Value {
    if tool_name != DECIDE_TOOL_NAME {
        return input.clone();
    }
    let mut record = input.clone();
    if let Some(state) = record.get_mut(STATE_FIELD) {
        *state = serde_json::Value::String(STATE_OMITTED.to_owned());
    }
    record
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::error::{Attempt, AttemptOutcome, Cause, DecisionError, Trace, Usage};
    use crate::decision::protocol::Question;
    use magi_core::schema::Mode;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/decide/internal-questions.json"
        ))
        .expect("fixture")
    }

    /// What T3 measured is exactly what ships.
    #[test]
    fn the_mode_question_is_the_measured_one() {
        assert_eq!(
            serde_json::to_value(mode_questions()).expect("ser"),
            fixture()["mode"]["questions"]
        );
    }

    #[test]
    fn the_contradiction_request_is_the_measured_one_with_a_older_and_b_newer() {
        let r = contradiction_request("older text", "newer text").expect("valid");
        assert_eq!(
            serde_json::to_value(r.questions()).expect("ser"),
            fixture()["contradicts"]["questions"]
        );
        assert_eq!(r.state(), &json!({"A": "older text", "B": "newer text"}));
    }

    /// Ollama 0.35 rejects a question without instructions (captured 2026-10-01).
    #[test]
    fn every_internal_question_carries_instructions() {
        let contradiction = contradiction_request("a", "b").expect("valid");
        for q in mode_questions()
            .values()
            .chain(contradiction.questions().values())
        {
            let has = match q {
                Question::Noul { instructions, .. }
                | Question::Choice { instructions, .. }
                | Question::Score { instructions, .. } => instructions
                    .as_deref()
                    .is_some_and(|s| !s.trim().is_empty()),
            };
            assert!(has, "{q:?}");
        }
    }

    #[test]
    fn only_the_three_labels_map_to_a_mode() {
        assert_eq!(mode_from_label("code-review"), Some(Mode::CodeReview));
        assert_eq!(mode_from_label("design"), Some(Mode::Design));
        assert_eq!(mode_from_label("analysis"), Some(Mode::Analysis));
        assert_eq!(mode_from_label("other"), None);
        assert_eq!(mode_from_label(""), None);
    }

    /// REQ-D19 (mutation-verified in T7).
    #[test]
    fn the_state_of_a_decide_call_is_omitted_in_the_record_and_the_questions_are_kept() {
        let input =
            json!({"state": {"secret": "STATE-MARKER-91e2"}, "questions": {"q": {"type": "noul"}}});
        let rec = tool_call_record_input(DECIDE_TOOL_NAME, &input);
        assert_eq!(rec["state"], STATE_OMITTED);
        assert_eq!(rec["questions"], input["questions"]);
        assert!(!rec.to_string().contains("STATE-MARKER-91e2"));
        assert_eq!(
            tool_call_record_input("view", &input),
            input,
            "other tools are recorded as they are"
        );
        let no_state = json!({"questions": {}});
        assert_eq!(
            tool_call_record_input(DECIDE_TOOL_NAME, &no_state),
            no_state
        );
    }

    fn exhausted(last: AttemptOutcome) -> DecisionError {
        DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![Attempt {
                    model: "m".into(),
                    outcome: last,
                    usage: Usage::default(),
                }],
                usage: Usage::default(),
            },
        }
    }

    #[test]
    fn every_error_maps_to_its_fallback_cause() {
        let cases = [
            (
                DecisionError::InvalidInput { reason: "r".into() },
                "rejected",
            ),
            (
                DecisionError::Unauthorized {
                    status: 401,
                    trace: Trace::default(),
                },
                "unauthorized",
            ),
            (
                DecisionError::MissingVaultEntry {
                    entry: "DECISION_BASE_URL_USER",
                },
                "unauthorized",
            ),
            (
                DecisionError::ClientUnavailable { reason: "r".into() },
                "client_unavailable",
            ),
            (
                exhausted(AttemptOutcome::Failed {
                    cause: Cause::Timeout,
                    detail: None,
                }),
                "timeout",
            ),
            (
                exhausted(AttemptOutcome::LowConfidence {
                    confidence: 0.1,
                    min_confidence: 0.5,
                    answers: Default::default(),
                }),
                "low_confidence",
            ),
            (
                DecisionError::ChainExhausted {
                    trace: Trace::default(),
                },
                "client_unavailable",
            ),
        ];
        for (e, token) in cases {
            assert_eq!(FallbackCause::from_error(&e).as_str(), token, "{e:?}");
        }
        assert_eq!(FallbackCause::TooLarge.as_str(), "too_large");
        assert_eq!(
            FallbackCause::SkippedAfterFailure.as_str(),
            "skipped_after_failure"
        );
        assert_eq!(Consumer::Mode.as_str(), "mode");
        assert_eq!(Consumer::Contradicts.as_str(), "contradicts");
    }

    /// REQ-D12: one INFO entry naming consumer, cause, endpoint and detail; never above INFO (no screen).
    #[test]
    fn a_fallback_is_one_info_event_with_its_fields() {
        struct Capture(Arc<Mutex<Vec<(tracing::Level, String)>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if let Ok(mut v) = self.0.lock() {
                    v.push((
                        *event.metadata().level(),
                        crate::logging::render::render_event(event),
                    ));
                }
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(
            tracing_subscriber::registry().with(Capture(Arc::clone(&seen))),
            || {
                log_fallback(
                    Consumer::Mode,
                    FallbackCause::Attempt(Cause::NotFound),
                    "http://h:1",
                    Some("the endpoint returned 404 for http://h:1/v1/systemone; check base_url"),
                );
            },
        );
        let seen = seen.lock().expect("lines");
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, tracing::Level::INFO);
        for needle in ["mode", "not_found", "http://h:1", "check base_url"] {
            assert!(seen[0].1.contains(needle), "{needle}: {}", seen[0].1);
        }
    }
}
