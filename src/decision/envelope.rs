//! The `magi decide` JSON envelope (REQ-D10) and its redaction per audience (REQ-D11).
//!
//! One renderer serves the direct user (`magi decide`, `/decide`) and the
//! agent's `decide` tool: the audience decides how much provider text the
//! output may carry.

use serde::Serialize;
use serde_json::Value;

use crate::decision::client::KEEP_ALIVE_HINT;
use crate::decision::endpoint::NOT_FOUND_HINT_MARKER;
use crate::decision::error::{AttemptOutcome, Decision, DecisionError, Trace, Usage};
use crate::decision::protocol::Answer;
use crate::headless::output::redact_secret_patterns;
use crate::redact::redact_foreign_text;

/// Version of the envelope layout (REQ-D10); first key of every envelope.
pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

/// Cap, in characters, for `error.message` and every `detail` (REQ-D10/D11).
pub const MESSAGE_CAP_CHARS: usize = 300;

/// Who reads the envelope; decides which provider text it may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Audience {
    /// The person who asked: redacted, capped provider text is allowed.
    User,
    /// The agent's tool result: only text Magi generated, no rejected answers.
    Agent,
}

/// Key of the optional list of rejected answers inside a chain entry.
const ANSWERS_KEY: &str = "answers";

/// Key of the free text Magi generates for an attempt.
const DETAIL_KEY: &str = "detail";

/// Physical layout of the envelope (REQ-D10): field order is key order.
#[derive(Serialize)]
struct Envelope<'a> {
    schema_version: u32,
    run_id: &'a str,
    model: Option<String>,
    endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    answers: Option<std::collections::BTreeMap<String, Answer>>,
    chain: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorBody>,
    usage: Usage,
    timings: Timings,
}

/// `error` object of a failed envelope.
#[derive(Serialize)]
struct ErrorBody {
    kind: crate::decision::error::ErrorKind,
    message: String,
}

/// `timings` object of the envelope.
#[derive(Serialize)]
struct Timings {
    total_ms: u64,
}

/// Renders the envelope as JSON text with `schema_version` as the first key.
///
/// Provider text reaches the output only through `error.message` of a
/// rejection shown to [`Audience::User`], and only redacted and capped
/// (REQ-D11). Complexity: O(attempts x answers) in the size of the trace.
///
/// # Arguments
/// * `result` - Outcome of the decision.
/// * `endpoint_display` - Redacted `base_url` of `[decision]`.
/// * `total_ms` - Wall time of the whole invocation.
/// * `audience` - Who reads the output.
///
/// # Returns
/// The envelope as one JSON object.
///
/// Test-only: production output must go through [`render_envelope_masked`] so the process
/// auditor's exact pass always runs; this identity-mask variant exists for the unit tests.
#[cfg(test)]
pub fn render_envelope(
    result: &Result<Decision, DecisionError>,
    endpoint_display: &str,
    total_ms: u64,
    audience: Audience,
) -> String {
    render_envelope_masked(
        result,
        endpoint_display,
        total_ms,
        audience,
        &str::to_string,
    )
}

/// Renders the envelope with every text field passed through `mask` first.
///
/// The mask runs on each string of the envelope (model, endpoint, `error.message`, every string
/// and key inside `chain` and `answers`), never on the serialized text: masking JSON as a whole
/// can orphan a backslash and stop it parsing. `schema_version` stays the first key.
///
/// Complexity: O(attempts x answers) in the size of the trace, times the mask's own cost.
///
/// # Arguments
/// * `result` - Outcome of the decision.
/// * `endpoint_display` - Redacted `base_url` of `[decision]`.
/// * `total_ms` - Wall time of the whole invocation.
/// * `audience` - Who reads the output.
/// * `mask` - Maps a text field to its masked form (the process auditor in production).
///
/// # Returns
/// The envelope as one JSON object.
pub fn render_envelope_masked(
    result: &Result<Decision, DecisionError>,
    endpoint_display: &str,
    total_ms: u64,
    audience: Audience,
    mask: &dyn Fn(&str) -> String,
) -> String {
    let trace = match result {
        Ok(decision) => Some(&decision.trace),
        Err(error) => error.trace(),
    };
    let envelope = Envelope {
        schema_version: ENVELOPE_SCHEMA_VERSION,
        run_id: crate::logging::run_id(),
        model: result.as_ref().ok().map(|d| mask(&d.model)),
        endpoint: mask(endpoint_display),
        answers: result.as_ref().ok().map(|d| mask_answers(&d.answers, mask)),
        chain: trace
            .map(|t| chain_values(t, audience, mask))
            .unwrap_or_default(),
        error: result.as_ref().err().map(|e| ErrorBody {
            kind: e.kind(),
            message: cap_chars(&mask(&error_message(e, audience)), MESSAGE_CAP_CHARS),
        }),
        usage: trace.map(|t| t.usage).unwrap_or_default(),
        timings: Timings { total_ms },
    };
    serde_json::to_string(&envelope)
        .unwrap_or_else(|_| format!("{{\"schema_version\":{ENVELOPE_SCHEMA_VERSION}}}"))
}

/// Masks every string and object key of a JSON tree, in place of the serialized text.
fn mask_value(value: Value, mask: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::String(text) => Value::String(mask(&text)),
        Value::Array(items) => {
            Value::Array(items.into_iter().map(|v| mask_value(v, mask)).collect())
        }
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, v)| (mask(&key), mask_value(v, mask)))
                .collect(),
        ),
        other => other,
    }
}

/// Masks the provider-supplied text of the answers: question names, chosen options, option
/// names and legend entries.
fn mask_answers(
    answers: &std::collections::BTreeMap<String, Answer>,
    mask: &dyn Fn(&str) -> String,
) -> std::collections::BTreeMap<String, Answer> {
    let masked_map = |map: &std::collections::BTreeMap<String, f64>| {
        map.iter().map(|(k, v)| (mask(k), *v)).collect()
    };
    answers
        .iter()
        .map(|(name, answer)| {
            let masked = match answer {
                Answer::Noul { .. } => answer.clone(),
                Answer::Choice {
                    choice,
                    probabilities,
                    confidence,
                    provider_confidence,
                } => Answer::Choice {
                    choice: mask(choice),
                    probabilities: masked_map(probabilities),
                    confidence: *confidence,
                    provider_confidence: *provider_confidence,
                },
                Answer::Score {
                    score,
                    legend,
                    probabilities,
                    confidence,
                    provider_confidence,
                } => Answer::Score {
                    score: *score,
                    legend: legend.iter().map(|(k, v)| (mask(k), mask(v))).collect(),
                    probabilities: masked_map(probabilities),
                    confidence: *confidence,
                    provider_confidence: *provider_confidence,
                },
            };
            (mask(name), masked)
        })
        .collect()
}

/// Serializes the attempts; the agent never sees rejected answers (REQ-D16)
/// and every `detail` is capped (REQ-D10).
fn chain_values(trace: &Trace, audience: Audience, mask: &dyn Fn(&str) -> String) -> Vec<Value> {
    let mut values = serde_json::to_value(&trace.attempts)
        .ok()
        .and_then(|v| match v {
            Value::Array(items) => Some(items),
            _ => None,
        })
        .unwrap_or_default();
    for entry in &mut values {
        // Mask first, cap after: a secret straddling the cap must not leave a prefix behind.
        *entry = mask_value(entry.take(), mask);
        let Some(object) = entry.as_object_mut() else {
            continue;
        };
        if audience == Audience::Agent {
            object.remove(ANSWERS_KEY);
        }
        if let Some(Value::String(detail)) = object.get_mut(DETAIL_KEY) {
            *detail = cap_chars(detail, MESSAGE_CAP_CHARS);
        }
    }
    values
}

/// Last attempt of the trace, when any model was called.
fn last_attempt(error: &DecisionError) -> Option<&crate::decision::error::Attempt> {
    error.trace().and_then(|t| t.attempts.last())
}

/// `detail` of the last attempt when it failed with one.
fn last_detail(error: &DecisionError) -> Option<&str> {
    match last_attempt(error).map(|a| &a.outcome) {
        Some(AttemptOutcome::Failed { detail, .. }) => detail.as_deref(),
        _ => None,
    }
}

/// Builds `error.message` per audience (REQ-D10/D11), before the cap.
fn error_message(error: &DecisionError, audience: Audience) -> String {
    if let DecisionError::Rejected {
        provider_message, ..
    } = error
    {
        let detail = last_detail(error);
        let magi_text = || detail.map_or_else(|| error.to_string(), str::to_string);
        return match audience {
            Audience::User => {
                let redacted =
                    redact_secret_patterns(redact_foreign_text(provider_message).as_str());
                if redacted.trim().is_empty() {
                    // An empty provider body says nothing: fall back to what Magi knows.
                    magi_text()
                } else if detail.is_some_and(|d| d.contains(KEEP_ALIVE_HINT)) {
                    format!("{redacted}{KEEP_ALIVE_HINT}")
                } else {
                    redacted
                }
            }
            Audience::Agent => magi_text(),
        };
    }
    let base = error.to_string();
    // The prefix hint goes first so the message cap never cuts the tried URL or the hint.
    match last_detail(error).and_then(|d| d.split_once(NOT_FOUND_HINT_MARKER)) {
        Some((_, rest)) => format!("{NOT_FOUND_HINT_MARKER}{rest}; {base}"),
        None => base,
    }
}

/// Keeps at most `cap` characters of `text`, never splitting a character.
///
/// Complexity: O(cap).
///
/// # Arguments
/// * `text` - Text to cap.
/// * `cap` - Maximum number of characters.
///
/// # Returns
/// The capped text.
pub fn cap_chars(text: &str, cap: usize) -> String {
    text.chars().take(cap).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::error::{
        Attempt, AttemptOutcome, Cause, Decision, DecisionError, Trace, Usage,
    };
    use crate::decision::protocol::Answer;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    fn answers() -> BTreeMap<String, Answer> {
        let mut m = BTreeMap::new();
        m.insert(
            "today".to_string(),
            Answer::Noul {
                noul: 0.824,
                confidence: 0.648,
            },
        );
        m
    }

    fn attempt(model: &str, outcome: AttemptOutcome) -> Attempt {
        Attempt {
            model: model.to_string(),
            outcome,
            usage: Usage::default(),
        }
    }

    fn ok() -> Result<Decision, DecisionError> {
        Ok(Decision {
            model: "nimble".into(),
            answers: answers(),
            trace: Trace {
                attempts: vec![attempt("nimble", AttemptOutcome::Answered)],
                usage: Usage {
                    input_tokens: 526,
                    output_tokens: 3,
                    cost: None,
                },
            },
        })
    }

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).expect("the envelope is one JSON object")
    }

    #[test]
    fn the_mask_reaches_every_text_field_and_leaves_valid_json_in_order() {
        let failed = Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::NotFound,
                        detail: Some("saw SECRET at \"x\"".into()),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let mask = |s: &str| s.replace("SECRET", "***");
        let text =
            render_envelope_masked(&failed, "http://SECRET.example", 1, Audience::User, &mask);
        assert!(!text.contains("SECRET"), "{text}");
        assert!(text.starts_with("{\"schema_version\":1,"), "{text}");
        assert_eq!(parse(&text)["chain"][0]["detail"], "saw *** at \"x\"");
        assert_eq!(parse(&text)["endpoint"], "http://***.example");
    }

    fn rejection(message: &str) -> Result<Decision, DecisionError> {
        Err(DecisionError::Rejected {
            status: 422,
            provider_message: message.to_string(),
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::Rejected,
                        detail: None,
                    },
                )],
                usage: Usage::default(),
            },
        })
    }

    #[test]
    fn a_secret_straddling_the_cap_is_masked_before_it_is_cut() {
        let secret = "toysecret-9f3a1c7e5b2d4a60";
        let padding = "a ".repeat((MESSAGE_CAP_CHARS - 10) / 2);
        let result = rejection(&format!("{padding}{secret} tail"));
        let mask = |s: &str| s.replace(secret, "***");
        let text = render_envelope_masked(&result, "http://h", 1, Audience::User, &mask);
        assert!(
            !text.contains("toysecre"),
            "a prefix of the secret survived the cut: {text}"
        );
    }

    #[test]
    fn a_secret_straddling_the_cap_of_a_detail_is_masked_before_it_is_cut() {
        let secret = "toysecret-9f3a1c7e5b2d4a60";
        let padding = "a ".repeat((MESSAGE_CAP_CHARS - 10) / 2);
        let result = Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::NotFound,
                        detail: Some(format!("{padding}{secret} tail")),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let mask = |s: &str| s.replace(secret, "***");
        let text = render_envelope_masked(&result, "http://h", 1, Audience::User, &mask);
        assert!(
            !text.contains("toysecre"),
            "a prefix of the secret survived the cut: {text}"
        );
    }

    /// Guards the pattern redaction of `error_message` on its own: the identity mask means no
    /// exact pass can shadow it.
    #[test]
    fn the_redaction_layer_alone_hides_a_pattern_shaped_key_in_a_rejection() {
        let key = "sk-ant-api03-toy0123456789abcdefghij"; // allow-secret-scan
        let result = rejection(&format!("rejected, saw {key} in the request"));
        let text = render_envelope(&result, "http://h", 1, Audience::User);
        assert!(!text.contains(key), "{text}");
        assert!(text.contains("rejected"), "{text}");
    }

    #[test]
    fn schema_version_is_the_first_physical_key_followed_by_run_id() {
        let text = render_envelope(&ok(), "http://localhost:11434", 412, Audience::User);
        assert!(
            text.starts_with("{\"schema_version\":1,\"run_id\":\""),
            "{text}"
        );
    }

    #[test]
    fn a_success_has_the_req_d10_keys_in_order_and_no_error() {
        let text = render_envelope(&ok(), "http://localhost:11434", 412, Audience::User);
        let pos = |k: &str| text.find(&format!("\"{k}\":")).unwrap_or(usize::MAX);
        let order = [
            "schema_version",
            "run_id",
            "model",
            "endpoint",
            "answers",
            "chain",
            "usage",
            "timings",
        ];
        assert!(order.windows(2).all(|w| pos(w[0]) < pos(w[1])), "{text}");
        let v = parse(&text);
        assert!(v.get("error").is_none());
        assert_eq!(v["model"], "nimble");
        assert_eq!(
            v["chain"][0],
            json!({"model": "nimble", "outcome": "answered"})
        );
        assert_eq!(
            v["usage"],
            json!({"input_tokens": 526, "output_tokens": 3, "cost": null})
        );
        assert_eq!(v["timings"]["total_ms"], 412);
        assert_eq!(v["answers"]["today"]["type"], "noul");
    }

    #[test]
    fn an_exhausted_chain_has_a_null_model_no_answers_and_its_kind() {
        let trace = Trace {
            attempts: vec![
                attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::ModelNotFound,
                        detail: Some("model 'nimble' is not available on this endpoint".into()),
                    },
                ),
                attempt(
                    "tev1",
                    AttemptOutcome::Failed {
                        cause: Cause::Timeout,
                        detail: None,
                    },
                ),
            ],
            usage: Usage::default(),
        };
        let v = parse(&render_envelope(
            &Err(DecisionError::ChainExhausted { trace }),
            "e",
            9,
            Audience::User,
        ));
        assert_eq!(v["model"], Value::Null);
        assert!(
            v.get("answers").is_none(),
            "no top-level answers without a decision (S13)"
        );
        assert_eq!(v["error"]["kind"], "decision_unavailable");
        assert_eq!(
            v["error"]["message"],
            "no decision model answered: chain exhausted (2 attempts)"
        );
        assert_eq!(
            v["chain"][1],
            json!({"model": "tev1", "outcome": "failed", "cause": "timeout"})
        );
    }

    #[test]
    fn low_confidence_answers_stay_in_the_chain_for_the_user_and_are_dropped_for_the_agent() {
        let trace = Trace {
            attempts: vec![attempt(
                "tev1:0.8b",
                AttemptOutcome::LowConfidence {
                    confidence: 0.725,
                    min_confidence: 0.8,
                    answers: answers(),
                },
            )],
            usage: Usage::default(),
        };
        let r: Result<Decision, DecisionError> = Err(DecisionError::ChainExhausted { trace });
        let user = parse(&render_envelope(&r, "e", 1, Audience::User));
        assert_eq!(user["chain"][0]["outcome"], "low_confidence");
        assert!(user["chain"][0].get("answers").is_some());
        assert!(
            user.get("answers").is_none(),
            "rejected answers never reach the top level"
        );
        let agent = parse(&render_envelope(&r, "e", 1, Audience::Agent));
        assert_eq!(agent["chain"][0]["confidence"], 0.725);
        assert!(
            agent["chain"][0].get("answers").is_none(),
            "REQ-D16: the agent never sees rejected answers"
        );
    }

    /// CP2 MS1 r3 (Caspar): REQ-D10's error table for 401 and model_unavailable, and the keep_alive note the
    /// agent sees on a rejection.
    #[test]
    fn unauthorized_model_unavailable_and_the_keep_alive_note_follow_the_contract() {
        let unauthorized: Result<Decision, DecisionError> = Err(DecisionError::Unauthorized {
            status: 401,
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::Unauthorized,
                        detail: None,
                    },
                )],
                usage: Usage::default(),
            },
        });
        let v = parse(&render_envelope(&unauthorized, "e", 1, Audience::User));
        assert_eq!(v["error"]["kind"], "missing_key");
        assert_eq!(v["chain"][0]["cause"], "unauthorized");
        assert_eq!(v["usage"]["input_tokens"], 0);

        let unavailable: Result<Decision, DecisionError> = Err(DecisionError::ModelUnavailable {
            model: "tev1".into(),
            trace: Trace {
                attempts: vec![attempt(
                    "tev1",
                    AttemptOutcome::Failed {
                        cause: Cause::Timeout,
                        detail: None,
                    },
                )],
                usage: Usage {
                    input_tokens: 7,
                    output_tokens: 1,
                    cost: None,
                },
            },
        });
        let v = parse(&render_envelope(&unavailable, "e", 1, Audience::User));
        assert_eq!(v["error"]["kind"], "model_unavailable");
        assert_eq!(v["chain"].as_array().map(Vec::len), Some(1));
        assert_eq!(v["usage"]["input_tokens"], 7, "what the chain consumed");

        let note = "provider rejected the request as invalid (HTTP 422); keep_alive was sent and providers other than Ollama may reject it";
        let rejected: Result<Decision, DecisionError> = Err(DecisionError::Rejected {
            status: 422,
            provider_message: "unprocessable".into(),
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::Rejected,
                        detail: Some(note.into()),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let msg = parse(&render_envelope(&rejected, "e", 1, Audience::Agent))["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(
            msg.contains("keep_alive") && !msg.contains("unprocessable"),
            "{msg}"
        );
    }

    /// CP2 MS1 r2 (Caspar): the two helpers this envelope relies on still mask the planted shapes, on their own.
    #[test]
    fn the_redaction_helpers_mask_the_planted_shapes() {
        let planted = "sk-ant-api03-planted0123456789abcdefghij"; // allow-secret-scan
        let raw = format!("Bearer {planted} at https://alice:s3cr3t@host/v1");
        let masked = crate::headless::output::redact_secret_patterns(
            crate::redact::redact_foreign_text(&raw).as_str(),
        );
        assert!(
            !masked.contains(planted) && !masked.contains("s3cr3t"),
            "{masked}"
        );
    }

    /// REQ-D11 guardian (mutation-verified).
    #[test]
    fn a_rejection_shows_the_redacted_capped_provider_text_to_the_user_and_only_magis_text_to_the_agent(
    ) {
        let planted = "sk-ant-api03-planted0123456789abcdefghij"; // allow-secret-scan
        let provider = format!(
            "bad request: Authorization: Bearer {planted} from https://alice:s3cr3t@host/v1 {}",
            "x".repeat(600)
        );
        let r: Result<Decision, DecisionError> = Err(DecisionError::Rejected {
            status: 413,
            provider_message: provider,
            trace: Trace {
                attempts: vec![attempt(
                    "nimble",
                    AttemptOutcome::Failed {
                        cause: Cause::Rejected,
                        detail: Some("provider rejected the request as invalid (HTTP 413)".into()),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let user = parse(&render_envelope(&r, "e", 1, Audience::User));
        let msg = user["error"]["message"].as_str().expect("message");
        assert!(msg.contains("bad request"), "{msg}");
        assert!(!msg.contains("s3cr3t") && !msg.contains(planted), "{msg}");
        assert!(
            msg.chars().count() <= MESSAGE_CAP_CHARS,
            "{}",
            msg.chars().count()
        );
        assert_eq!(user["error"]["kind"], "invalid_input");
        let agent = parse(&render_envelope(&r, "e", 1, Audience::Agent));
        let msg = agent["error"]["message"].as_str().expect("message");
        assert!(
            msg.contains("HTTP 413") && !msg.contains("bad request"),
            "{msg}"
        );
    }

    /// S20: the not-found hint reaches `error.message` for both audiences.
    #[test]
    fn the_not_found_hint_reaches_the_error_message() {
        let hint = crate::decision::endpoint::not_found_hint("https://openrouter.ai/v1/systemone");
        let trace = Trace {
            attempts: vec![attempt(
                "m",
                AttemptOutcome::Failed {
                    cause: Cause::NotFound,
                    detail: Some(hint),
                },
            )],
            usage: Usage::default(),
        };
        let r: Result<Decision, DecisionError> = Err(DecisionError::ChainExhausted { trace });
        for audience in [Audience::User, Audience::Agent] {
            let v = parse(&render_envelope(&r, "https://openrouter.ai", 1, audience));
            let msg = v["error"]["message"].as_str().expect("message");
            assert!(
                msg.contains("https://openrouter.ai/v1/systemone") && msg.contains("base_url"),
                "{msg}"
            );
        }
    }

    /// S20: an `invalid_response` whose detail carries the hint (non-JSON 200) shows it to the user.
    #[test]
    fn the_hint_of_an_invalid_response_reaches_the_error_message() {
        let url = "https://openrouter.ai/v1/systemone";
        let detail = format!(
            "invalid response: body is not JSON; {}",
            crate::decision::endpoint::not_found_hint(url)
        );
        let r: Result<Decision, DecisionError> = Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![attempt(
                    "m",
                    AttemptOutcome::Failed {
                        cause: Cause::InvalidResponse,
                        detail: Some(detail),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let v = parse(&render_envelope(&r, "e", 1, Audience::User));
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(msg.contains(url) && msg.contains("API prefix"), "{msg}");
    }

    /// The cap never cuts the URL or the hint, even with a long base URL.
    #[test]
    fn the_hint_survives_the_message_cap_with_a_long_url() {
        let url = format!("https://{}.example.com/v1/systemone", "h".repeat(60));
        let r: Result<Decision, DecisionError> = Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![attempt(
                    "m",
                    AttemptOutcome::Failed {
                        cause: Cause::NotFound,
                        detail: Some(crate::decision::endpoint::not_found_hint(&url)),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let v = parse(&render_envelope(&r, "e", 1, Audience::User));
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(msg.contains(&url), "{msg}");
        assert!(msg.contains("(Ollama 0.35 or later)"), "{msg}");
        assert!(msg.chars().count() <= MESSAGE_CAP_CHARS);
    }

    /// A rejection whose provider body is blank still tells the user what happened.
    #[test]
    fn a_rejection_with_an_empty_body_falls_back_to_the_magi_detail() {
        let r: Result<Decision, DecisionError> = Err(DecisionError::Rejected {
            status: 422,
            provider_message: "   ".into(),
            trace: Trace {
                attempts: vec![attempt(
                    "m",
                    AttemptOutcome::Failed {
                        cause: Cause::Rejected,
                        detail: Some("provider rejected the request as invalid (HTTP 422)".into()),
                    },
                )],
                usage: Usage::default(),
            },
        });
        let v = parse(&render_envelope(&r, "e", 1, Audience::User));
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(msg.contains("HTTP 422"), "{msg}");
    }

    #[test]
    fn errors_without_a_model_call_have_an_empty_chain_and_zero_usage() {
        for e in [
            DecisionError::InvalidInput {
                reason: "question \"q\": choice needs criteria".into(),
            },
            DecisionError::MissingVaultEntry {
                entry: "DECISION_BASE_URL_USER",
            },
            DecisionError::ClientUnavailable {
                reason: "tls".into(),
            },
        ] {
            let v = parse(&render_envelope(&Err(e.clone()), "e", 0, Audience::User));
            assert_eq!(v["chain"], json!([]), "{e:?}");
            assert_eq!(
                v["usage"],
                json!({"input_tokens": 0, "output_tokens": 0, "cost": null}),
                "{e:?}"
            );
        }
        let v = parse(&render_envelope(
            &Err(DecisionError::MissingVaultEntry {
                entry: "DECISION_BASE_URL_USER",
            }),
            "e",
            0,
            Audience::User,
        ));
        assert_eq!(v["error"]["kind"], "missing_key");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("DECISION_BASE_URL_USER"));
    }

    #[test]
    fn cap_chars_never_splits_a_character() {
        assert_eq!(cap_chars("ñandú", 3), "ñan");
        assert_eq!(cap_chars("short", 300), "short");
        assert_eq!(
            cap_chars(&"é".repeat(400), MESSAGE_CAP_CHARS)
                .chars()
                .count(),
            MESSAGE_CAP_CHARS
        );
    }
}
