//! The agent tool `decide` (REQ-D16, REQ-D19): a typed decision served by a System One model.
//! The agent chooses it for closed choices, yes/no questions and rubric scores; the result is
//! the envelope for the agent audience, redacted by the process auditor's pass.

use crate::tools::{Tool, ToolError};
use async_trait::async_trait;
use magi_rs::decision::DecisionClient;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// The `decide` tool.
pub struct DecideTool {
    /// The shared decision client.
    _client: Arc<dyn DecisionClient>,
    /// Whether an invocation needs no approval.
    _auto_approve: bool,
    /// The redacted decision endpoint shown in the envelope.
    _endpoint: String,
}

impl DecideTool {
    /// Builds the tool.
    ///
    /// # Arguments
    /// * `client` - The shared decision client.
    /// * `auto_approve` - `[decision].auto_approve`.
    /// * `endpoint` - The redacted decision endpoint.
    ///
    /// # Returns
    /// The tool.
    pub fn new(client: Arc<dyn DecisionClient>, auto_approve: bool, endpoint: String) -> Self {
        Self {
            _client: client,
            _auto_approve: auto_approve,
            _endpoint: endpoint,
        }
    }
}

#[async_trait]
impl Tool for DecideTool {
    /// The tool name.
    fn name(&self) -> &str {
        todo!()
    }

    /// What the agent reads to choose the tool.
    fn description(&self) -> &str {
        todo!()
    }

    /// The input schema.
    fn input_schema(&self) -> Value {
        todo!()
    }

    /// Runs one decision.
    async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> Result<Value, ToolError> {
        todo!()
    }

    /// Whether an invocation needs approval.
    fn requires_approval(&self) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use magi_rs::decision::error::{
        Attempt, AttemptOutcome, Cause, Decision, DecisionError, Trace, Usage,
    };
    use magi_rs::decision::{Answer, DecisionClient, DecisionRequest};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    struct Scripted {
        reply: Result<Decision, DecisionError>,
        calls: AtomicUsize,
    }
    #[async_trait]
    impl DecisionClient for Scripted {
        async fn decide(&self, _r: &DecisionRequest) -> Result<Decision, DecisionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.reply.clone()
        }
    }

    fn tool(
        reply: Result<Decision, DecisionError>,
        auto_approve: bool,
    ) -> (DecideTool, Arc<Scripted>) {
        let client = Arc::new(Scripted {
            reply,
            calls: AtomicUsize::new(0),
        });
        (
            DecideTool::new(
                Arc::clone(&client) as Arc<dyn DecisionClient>,
                auto_approve,
                "http://localhost:11434".into(),
            ),
            client,
        )
    }

    fn args() -> serde_json::Value {
        json!({"state": "Is this issue a bug, a feature or a question?",
               "questions": {"kind": {"type": "choice", "instructions": "Classify the issue",
                 "criteria": {"bug": "a defect", "feature": "a request", "question": "a question"}}}})
    }

    fn answered() -> Result<Decision, DecisionError> {
        let mut answers = BTreeMap::new();
        answers.insert(
            "kind".to_string(),
            Answer::Choice {
                choice: "bug".into(),
                probabilities: BTreeMap::from([
                    ("bug".to_string(), 0.8),
                    ("feature".to_string(), 0.1),
                    ("question".to_string(), 0.1),
                ]),
                confidence: 0.7,
                provider_confidence: Some(0.6),
            },
        );
        Ok(Decision {
            model: "nimble".into(),
            answers,
            trace: Trace {
                attempts: vec![Attempt {
                    model: "nimble".into(),
                    outcome: AttemptOutcome::Answered,
                    usage: Usage::default(),
                }],
                usage: Usage::default(),
            },
        })
    }

    #[test]
    fn the_tool_is_named_decide_and_requires_state_and_questions() {
        let (t, _) = tool(answered(), true);
        assert_eq!(t.name(), "decide");
        let schema = t.input_schema();
        assert_eq!(schema["required"], json!(["state", "questions"]));
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    /// REQ-D19: the description says when, when not, how to cite and what to do on failure.
    #[test]
    fn the_description_tells_the_agent_when_and_how_to_use_it() {
        let (t, _) = tool(answered(), true);
        let d = t.description();
        for needle in ["choice", "noul", "score", "consult", "confidence", "fails"] {
            assert!(d.contains(needle), "{needle}: {d}");
        }
    }

    /// S24: the default (`auto_approve = true`) needs no approval; `false` does.
    #[test]
    fn approval_follows_auto_approve() {
        assert!(!tool(answered(), true).0.requires_approval());
        assert!(tool(answered(), false).0.requires_approval());
    }

    #[tokio::test]
    async fn a_decision_returns_the_agent_envelope() {
        let (t, _) = tool(answered(), true);
        let v = t
            .execute(args(), &CancellationToken::new())
            .await
            .expect("ok");
        assert_eq!(v["model"], "nimble");
        assert_eq!(v["answers"]["kind"]["choice"], "bug");
        assert_eq!(v["answers"]["kind"]["confidence"], 0.7);
        assert_eq!(v["chain"][0]["outcome"], "answered");
    }

    #[tokio::test]
    async fn invalid_arguments_never_reach_the_client() {
        let (t, client) = tool(answered(), true);
        let r = t
            .execute(json!({"state": "x"}), &CancellationToken::new())
            .await;
        assert!(matches!(r, Err(ToolError::InvalidArguments(_))), "{r:?}");
        assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    }

    /// S21/REQ-D11 (mutation-verified): the agent gets Magi's text, never the provider's, nor rejected answers.
    #[tokio::test]
    async fn a_failure_reaches_the_agent_without_provider_text_or_rejected_answers() {
        let mut low = BTreeMap::new();
        low.insert(
            "kind".to_string(),
            Answer::Noul {
                noul: 0.6,
                confidence: 0.2,
            },
        );
        let rejected: Result<Decision, DecisionError> = Err(DecisionError::Rejected {
            status: 413,
            provider_message: "body too large: STATE-MARKER-5d1c".into(),
            trace: Trace {
                attempts: vec![
                    Attempt {
                        model: "nimble".into(),
                        outcome: AttemptOutcome::LowConfidence {
                            confidence: 0.2,
                            min_confidence: 0.9,
                            answers: low,
                        },
                        usage: Usage::default(),
                    },
                    Attempt {
                        model: "tev1".into(),
                        outcome: AttemptOutcome::Failed {
                            cause: Cause::Rejected,
                            detail: Some(
                                "provider rejected the request as invalid (HTTP 413)".into(),
                            ),
                        },
                        usage: Usage::default(),
                    },
                ],
                usage: Usage::default(),
            },
        });
        let (t, _) = tool(rejected, true);
        match t.execute(args(), &CancellationToken::new()).await {
            Err(ToolError::ExecutionError(msg)) => {
                assert!(msg.contains("HTTP 413"), "{msg}");
                assert!(
                    !msg.contains("STATE-MARKER-5d1c"),
                    "provider text leaked: {msg}"
                );
                let v: serde_json::Value = serde_json::from_str(&msg).expect("the agent envelope");
                assert!(
                    v["chain"][0].get("answers").is_none(),
                    "rejected answers reached the agent: {msg}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn cancellation_ends_the_call_and_drops_the_request() {
        struct Hanging {
            dropped: Arc<AtomicBool>,
        }
        struct OnDrop(Arc<AtomicBool>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        #[async_trait]
        impl DecisionClient for Hanging {
            async fn decide(&self, _r: &DecisionRequest) -> Result<Decision, DecisionError> {
                let _g = OnDrop(Arc::clone(&self.dropped));
                std::future::pending::<()>().await;
                unreachable!("pending never resolves")
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let t = DecideTool::new(
            Arc::new(Hanging {
                dropped: Arc::clone(&dropped),
            }),
            true,
            "e".into(),
        );
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let r = tokio::time::timeout(Duration::from_secs(30), t.execute(args(), &cancel))
            .await
            .expect("bounded");
        assert!(
            matches!(r, Err(ToolError::ExecutionError(ref m)) if m.contains("cancelled")),
            "{r:?}"
        );
        assert!(dropped.load(Ordering::SeqCst));
    }

    /// The tool result goes through the process auditor's exact pass, never an identity mask: a
    /// registered secret echoed by the provider must not reach the agent (mutation-verified).
    #[tokio::test]
    async fn a_registered_secret_echoed_by_the_provider_is_masked_in_the_tool_result() {
        const SECRET: &str = "zq9-plainword-registered-7731";
        assert!(
            magi_rs::logging::process_auditor().register_secret(
                magi_rs::logging::auditor::SecretName::new("DECIDE_TOOL_TEST_SECRET"),
                &[SECRET],
            ),
            "test setup: the secret must enter the exact pass"
        );
        let Ok(mut decision) = answered() else {
            unreachable!("the fixture answers")
        };
        decision.model = SECRET.to_string();
        let (t, _) = tool(Ok(decision), true);
        let v = t
            .execute(args(), &CancellationToken::new())
            .await
            .expect("ok");
        assert!(!v.to_string().contains(SECRET), "secret leaked: {v}");
    }
}
