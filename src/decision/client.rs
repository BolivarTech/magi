//! One System One attempt over HTTP and the client that walks the chain (REQ-D00, REQ-D03,
//! REQ-D04, REQ-D07, REQ-D10).

use crate::decision::config::{KeepAlive, ModelSpec};
use crate::decision::error::{Decision, DecisionError};
use crate::decision::protocol::DecisionRequest;
use std::time::Duration;

/// Tracing target of every decision log entry.
pub const DECISION_LOG_TARGET: &str = "magi_rs::decision";
/// Most bytes of an error body that are read to pick a `cause` token or to relay a rejection.
pub const NOT_FOUND_BODY_READ_CAP: usize = 64 * 1024;
/// A System One answer for a few questions is a few KiB; anything past this is not read and is an
/// invalid response.
pub const RESPONSE_BODY_CAP: usize = 1024 * 1024;
/// Ollama 0.35.0, captured 2026-10-01: `{"error":"model \"<m>\" not found, try pulling it first"}`.
pub const MODEL_NOT_FOUND_MARKERS: &[&str] = &["not found, try pulling it first"];
/// OpenRouter, E4: "0 endpoints out of 1 requested are available matching your guardrail
/// restrictions and data policy...".
pub const ACCOUNT_POLICY_MARKERS: &[&str] = &["guardrail restrictions and data policy"];

/// Asks System One models for typed decisions.
#[cfg_attr(test, mockall::automock)]
#[async_trait::async_trait]
pub trait DecisionClient: Send + Sync {
    /// One System One request through the fallback chain (REQ-D00).
    ///
    /// # Arguments
    /// * `request` - the validated request.
    ///
    /// # Returns
    /// The decision with the chain walked to get it.
    ///
    /// # Errors
    /// A [`DecisionError`] carrying the trace of every attempt.
    async fn decide(&self, request: &DecisionRequest) -> Result<Decision, DecisionError>;
}

/// Everything the client needs, already resolved.
#[derive(Clone)]
pub struct DecisionSettings {
    /// Resolved `base_url` (placeholders substituted).
    pub base_url: String,
    /// Key for the endpoint, if any.
    pub api_key: Option<zeroize::Zeroizing<String>>,
    /// Models in order.
    pub chain: Vec<ModelSpec>,
    /// Per-attempt timeout.
    pub timeout: Duration,
    /// Sent verbatim when declared.
    pub keep_alive: Option<KeepAlive>,
}

impl std::fmt::Debug for DecisionSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionSettings")
            .field("base_url", &crate::redact::redact_url(&self.base_url))
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .field("chain", &self.chain)
            .field("timeout", &self.timeout)
            .field("keep_alive", &self.keep_alive)
            .finish()
    }
}

/// HTTP implementation of [`DecisionClient`].
pub struct SystemOneClient {
    _settings: DecisionSettings,
}

impl SystemOneClient {
    /// Builds the client.
    ///
    /// # Errors
    /// `ClientUnavailable` when the HTTP client cannot be built.
    pub fn new(_settings: DecisionSettings) -> Result<Self, DecisionError> {
        todo!()
    }

    /// The `base_url`, redacted.
    pub fn endpoint_display(&self) -> String {
        todo!()
    }
}

#[async_trait::async_trait]
impl DecisionClient for SystemOneClient {
    async fn decide(&self, _request: &DecisionRequest) -> Result<Decision, DecisionError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::config::{KeepAlive, ModelSpec};
    use crate::decision::error::{AttemptOutcome, Cause, DecisionError};
    use crate::decision::protocol::{DecisionRequest, Questions};
    use mockito::Matcher;
    use serde_json::json;
    use std::time::Duration;

    /// Failure deadline for every network test: generous, never a measured duration.
    const DEADLINE: Duration = Duration::from_secs(30);

    const OK_BODY: &str = r#"{"model":"nimble","answers":{"today":{"type":"noul","noul":0.9}},"usage":{"input_tokens":5,"output_tokens":1}}"#;
    /// Ollama 0.35.0, captured 2026-10-01.
    const OLLAMA_MISSING_MODEL: &str =
        r#"{"error":"model \"nimble\" not found, try pulling it first"}"#;
    /// OpenRouter, E4 (`or-probe-results.json`).
    const OPENROUTER_POLICY: &str = r#"{"error":{"message":"0 endpoints out of 1 requested are available matching your guardrail restrictions and data policy. We removed them for the following reasons:\nZDR violation (account settings): 1 endpoint excluded","code":404}}"#;

    fn settings(base: &str, chain: &[&str]) -> DecisionSettings {
        DecisionSettings {
            base_url: base.to_string(),
            api_key: None,
            chain: chain
                .iter()
                .map(|m| ModelSpec {
                    model: (*m).to_string(),
                    min_confidence: None,
                })
                .collect(),
            timeout: Duration::from_secs(10),
            keep_alive: None,
        }
    }

    fn request() -> DecisionRequest {
        let questions: Questions = serde_json::from_value(
            json!({"today": {"type": "noul", "instructions": "Is it today?"}}),
        )
        .expect("q");
        DecisionRequest::new(json!({"ticket": "x"}), questions).expect("valid")
    }

    async fn decide(s: DecisionSettings, r: &DecisionRequest) -> Result<Decision, DecisionError> {
        let client = SystemOneClient::new(s).expect("client");
        tokio::time::timeout(DEADLINE, client.decide(r))
            .await
            .expect("bounded by the failure deadline")
    }

    fn first_failure(r: &Result<Decision, DecisionError>) -> Option<(Cause, Option<String>)> {
        let attempt = r.as_ref().err()?.trace()?.attempts.first()?.clone();
        match attempt.outcome {
            AttemptOutcome::Failed { cause, detail } => Some((cause, detail)),
            _ => None,
        }
    }

    #[tokio::test]
    async fn posts_model_state_and_questions_to_the_derived_url() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/systemone")
            .match_body(Matcher::PartialJson(
                json!({"model": "nimble", "state": {"ticket": "x"},
                "questions": {"today": {"type": "noul", "instructions": "Is it today?"}}}),
            ))
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        let d = decide(settings(&server.url(), &["nimble"]), &request())
            .await
            .expect("answered");
        assert_eq!(d.model, "nimble");
        assert!(d.answers.contains_key("today"));
        m.assert_async().await;
    }

    #[tokio::test]
    async fn a_base_url_with_an_api_prefix_keeps_it() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/api/v1/systemone")
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        decide(
            settings(&format!("{}/api/v1", server.url()), &["nimble"]),
            &request(),
        )
        .await
        .expect("answered");
        m.assert_async().await;
    }

    #[tokio::test]
    async fn the_bearer_header_is_sent_only_when_a_key_is_configured() {
        let mut server = mockito::Server::new_async().await;
        let with = server
            .mock("POST", "/v1/systemone")
            .match_header("authorization", "Bearer dec-key")
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        let mut s = settings(&server.url(), &["nimble"]);
        s.api_key = Some(zeroize::Zeroizing::new("dec-key".to_string()));
        decide(s, &request()).await.expect("answered");
        with.assert_async().await;

        let mut server = mockito::Server::new_async().await;
        let without = server
            .mock("POST", "/v1/systemone")
            .match_header("authorization", Matcher::Missing)
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        decide(settings(&server.url(), &["nimble"]), &request())
            .await
            .expect("answered");
        without.assert_async().await;
    }

    #[tokio::test]
    async fn keep_alive_is_sent_only_when_declared() {
        let mut server = mockito::Server::new_async().await;
        let absent = server
            .mock("POST", "/v1/systemone")
            .match_request(|r| {
                r.utf8_lossy_body()
                    .map(|b| !b.contains("keep_alive"))
                    .unwrap_or(false)
            })
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        decide(settings(&server.url(), &["nimble"]), &request())
            .await
            .expect("answered");
        absent.assert_async().await;

        let mut server = mockito::Server::new_async().await;
        let present = server
            .mock("POST", "/v1/systemone")
            .match_body(Matcher::PartialJson(json!({"keep_alive": "10m"})))
            .with_status(200)
            .with_body(OK_BODY)
            .create_async()
            .await;
        let mut s = settings(&server.url(), &["nimble"]);
        s.keep_alive = Some(KeepAlive::Text("10m".into()));
        decide(s, &request()).await.expect("answered");
        present.assert_async().await;
    }

    #[tokio::test]
    async fn each_failing_status_maps_to_its_cause() {
        let long_then_marker = format!(
            "{}{}",
            "x".repeat(NOT_FOUND_BODY_READ_CAP + 10),
            OLLAMA_MISSING_MODEL
        );
        let cases: Vec<(usize, Vec<u8>, Cause)> = vec![
            (
                404,
                OLLAMA_MISSING_MODEL.as_bytes().to_vec(),
                Cause::ModelNotFound,
            ),
            (
                404,
                OPENROUTER_POLICY.as_bytes().to_vec(),
                Cause::AccountPolicy,
            ),
            (404, b"404 page not found".to_vec(), Cause::NotFound),
            (404, Vec::new(), Cause::NotFound),
            (
                404,
                b"<html><body>Not Found</body></html>".to_vec(),
                Cause::NotFound,
            ),
            (404, vec![0xff, 0xfe, 0xfd], Cause::NotFound),
            (404, long_then_marker.into_bytes(), Cause::NotFound),
            (429, Vec::new(), Cause::RateLimited),
            (500, Vec::new(), Cause::Http5xx),
            (503, b"busy".to_vec(), Cause::Http5xx),
        ];
        for (status, body, expected) in cases {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/v1/systemone")
                .with_status(status)
                .with_body(body)
                .create_async()
                .await;
            let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
            assert_eq!(
                first_failure(&r).map(|(c, _)| c),
                Some(expected),
                "HTTP {status}"
            );
        }
    }

    #[tokio::test]
    async fn credential_and_caller_errors_end_the_call_with_their_own_error() {
        for status in [401usize, 403] {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/v1/systemone")
                .with_status(status)
                .create_async()
                .await;
            let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
            assert!(
                matches!(r, Err(DecisionError::Unauthorized { .. })),
                "HTTP {status}: {r:?}"
            );
        }
        for status in [400usize, 413, 422] {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/v1/systemone")
                .with_status(status)
                .with_body(r#"{"error":"question \"q\": instructions must be a nonempty string"}"#)
                .create_async()
                .await;
            match decide(settings(&server.url(), &["nimble"]), &request()).await {
                Err(DecisionError::Rejected {
                    status: s,
                    provider_message,
                    trace,
                }) => {
                    assert_eq!(usize::from(s), status);
                    assert!(
                        provider_message.contains("instructions must be"),
                        "{provider_message}"
                    );
                    let detail = match &trace.attempts[0].outcome {
                        AttemptOutcome::Failed { detail, .. } => detail.clone().unwrap_or_default(),
                        other => panic!("{other:?}"),
                    };
                    assert!(
                        !detail.contains("instructions must be"),
                        "a detail is Magi's own text: {detail}"
                    );
                    assert!(detail.contains(&format!("HTTP {status}")), "{detail}");
                }
                other => panic!("HTTP {status}: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn a_redirect_is_never_followed() {
        let mut target = mockito::Server::new_async().await;
        let never = target
            .mock("POST", Matcher::Any)
            .expect(0)
            .create_async()
            .await;
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/systemone")
            .with_status(302)
            .with_header("location", &format!("{}/v1/systemone", target.url()))
            .create_async()
            .await;
        let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
        let (cause, detail) = first_failure(&r).expect("failed attempt");
        assert_eq!(cause, Cause::Redirected);
        assert!(detail.unwrap_or_default().contains("not followed"));
        never.assert_async().await;
    }

    /// S23(b) at unit level: the request's timeout overrides the settings' one, per attempt.
    #[tokio::test]
    async fn a_silent_server_is_a_timeout_and_the_request_timeout_wins() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let hold = std::thread::spawn(move || {
            let accepted = listener.accept();
            std::thread::sleep(Duration::from_secs(5));
            drop(accepted);
        });
        let mut s = settings(&format!("http://{addr}"), &["nimble"]);
        s.timeout = Duration::from_secs(120);
        let r = decide(s, &request().with_timeout(Duration::from_millis(300))).await;
        assert_eq!(first_failure(&r).map(|(c, _)| c), Some(Cause::Timeout));
        drop(hold);
    }

    #[tokio::test]
    async fn a_closed_port_is_a_transport_failure() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind")
            .local_addr()
            .expect("addr")
            .port();
        let r = decide(
            settings(&format!("http://127.0.0.1:{port}"), &["nimble"]),
            &request(),
        )
        .await;
        assert_eq!(first_failure(&r).map(|(c, _)| c), Some(Cause::Transport));
    }

    /// S20 at unit level: the hint names the URL that was tried.
    #[tokio::test]
    async fn a_plain_404_carries_the_prefix_hint_with_the_tried_url() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/systemone")
            .with_status(404)
            .with_body("404 page not found")
            .create_async()
            .await;
        let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
        let (_, detail) = first_failure(&r).expect("failed");
        let detail = detail.unwrap_or_default();
        assert!(
            detail.contains(&format!("{}/v1/systemone", server.url())),
            "{detail}"
        );
        assert!(detail.contains("base_url"), "{detail}");
    }

    #[tokio::test]
    async fn an_invalid_200_is_an_invalid_response_that_quotes_nothing_from_the_provider() {
        let planted = "sk-ant-planted0123456789abcdefghij";
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_body(
                json!({"model": "m", "answers": {"today": {"type": "choice", "choice": planted,
                "probabilities": {planted: 1.0}}}})
                .to_string(),
            )
            .create_async()
            .await;
        let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
        let (cause, detail) = first_failure(&r).expect("failed");
        assert_eq!(cause, Cause::InvalidResponse);
        assert!(!detail.unwrap_or_default().contains(planted));
    }

    #[test]
    fn neither_the_settings_debug_nor_the_endpoint_display_leak_a_secret() {
        let mut s = settings("https://alice:s3cr3t@host/api/v1", &["nimble"]);
        s.api_key = Some(zeroize::Zeroizing::new(
            "sk-planted-key-0123456789".to_string(),
        ));
        let debug = format!("{s:?}");
        assert!(
            !debug.contains("s3cr3t") && !debug.contains("sk-planted-key"),
            "{debug}"
        );
        let client = SystemOneClient::new(s).expect("client");
        let shown = client.endpoint_display();
        assert!(
            !shown.contains("s3cr3t") && shown.contains("host"),
            "{shown}"
        );
    }

    /// REQ-D11 / REQ-D03 (CP2 MS1 r1, Caspar): the 404 hint names the tried URL redacted, never the
    /// credentials the resolved base_url carries.
    #[tokio::test]
    async fn the_not_found_hint_never_carries_base_url_credentials() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/systemone")
            .with_status(404)
            .with_body("404 page not found")
            .create_async()
            .await;
        let base = server.url().replace("http://", "http://alice:s3cr3t@");
        let r = decide(settings(&base, &["nimble"]), &request()).await;
        let (cause, detail) = first_failure(&r).expect("a failed attempt");
        assert_eq!(cause, Cause::NotFound);
        let detail = detail.unwrap_or_default();
        assert!(
            detail.contains("/v1/systemone") && detail.contains("base_url"),
            "{detail}"
        );
        assert!(
            !detail.contains("s3cr3t") && !detail.contains("alice"),
            "{detail}"
        );
    }

    /// CP2 MS1 r1 (Caspar): the cap rejects what is over it and reads whole what is under it.
    #[tokio::test]
    async fn a_valid_200_just_under_the_cap_is_read_whole() {
        let mut server = mockito::Server::new_async().await;
        let json = r#"{"model":"nimble","answers":{"today":{"type":"noul","noul":0.9}},"usage":{"input_tokens":5,"output_tokens":1}}"#;
        let body = format!("{json}{}", " ".repeat(RESPONSE_BODY_CAP - json.len() - 1));
        server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;
        let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
        assert!(r.is_ok(), "{r:?}");
    }

    /// Review Focus 3: a 200 bigger than `RESPONSE_BODY_CAP` is not read whole and does not decide.
    #[tokio::test]
    async fn a_huge_non_json_200_is_an_invalid_response() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_header("content-type", "text/html")
            .with_body("<html>".repeat(2 * RESPONSE_BODY_CAP / 6 + 1))
            .create_async()
            .await;
        let r = decide(settings(&server.url(), &["nimble"]), &request()).await;
        assert_eq!(
            first_failure(&r).map(|(c, _)| c),
            Some(Cause::InvalidResponse)
        );
    }

    /// REQ-D01: a 400/422 after sending keep_alive names it as the probable cause.
    #[tokio::test]
    async fn a_rejection_names_keep_alive_when_it_was_sent() {
        for (keep_alive, expected) in [(Some(KeepAlive::Text("10m".into())), true), (None, false)] {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/v1/systemone")
                .with_status(422)
                .with_body("unprocessable")
                .create_async()
                .await;
            let mut s = settings(&server.url(), &["nimble"]);
            s.keep_alive = keep_alive;
            match decide(s, &request()).await {
                Err(DecisionError::Rejected { trace, .. }) => {
                    let detail = match &trace.attempts[0].outcome {
                        AttemptOutcome::Failed { detail, .. } => detail.clone().unwrap_or_default(),
                        other => panic!("{other:?}"),
                    };
                    assert_eq!(detail.contains("keep_alive"), expected, "{detail}");
                }
                other => panic!("{other:?}"),
            }
        }
    }
}
