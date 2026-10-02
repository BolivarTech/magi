//! The `magi decide` JSON envelope (REQ-D10) and its redaction per audience (REQ-D11).
//!
//! One renderer serves the direct user (`magi decide`, `/decide`) and the
//! agent's `decide` tool: the audience decides how much provider text the
//! output may carry.

use crate::decision::error::{Decision, DecisionError};

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

/// Renders the envelope as JSON text with `schema_version` as the first key.
///
/// # Arguments
/// * `result` - Outcome of the decision.
/// * `endpoint_display` - Redacted `base_url` of `[decision]`.
/// * `total_ms` - Wall time of the whole invocation.
/// * `audience` - Who reads the output.
///
/// # Returns
/// The envelope as one JSON object.
pub fn render_envelope(
    _result: &Result<Decision, DecisionError>,
    _endpoint_display: &str,
    _total_ms: u64,
    _audience: Audience,
) -> String {
    todo!()
}

/// Keeps at most `cap` characters of `text`, never splitting a character.
///
/// # Arguments
/// * `text` - Text to cap.
/// * `cap` - Maximum number of characters.
///
/// # Returns
/// The capped text.
pub fn cap_chars(_text: &str, _cap: usize) -> String {
    todo!()
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
