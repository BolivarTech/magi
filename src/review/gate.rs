// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! The false-positive gate: one `decide` question per consolidated finding.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use crate::decision::{DecisionClient, ModelSpec, Questions};

use super::consolidate::ConsolidatedFinding;

/// Name of the gate's only question.
pub const GATE_QUESTION_NAME: &str = "gate";
/// The question's instructions, verbatim from the calibration: changing them invalidates it.
pub const GATE_INSTRUCTIONS: &str = "Does this finding point to a real defect in the code shown?";
/// Description of the `real_defect` option, verbatim from the calibration.
pub const GATE_REAL_DEFECT: &str = "The finding describes a defect that exists in the code shown";
/// Description of the `false_positive` option, verbatim from the calibration.
pub const GATE_FALSE_POSITIVE: &str =
    "The finding is wrong about the code shown, or what it describes is not a defect";
/// Lines of code shown on each side of the finding's line.
pub const GATE_CONTEXT_LINES: u32 = 40;
/// Run-log target of the gate's WARN entries.
pub const GATE_LOG_TARGET: &str = "magi_rs::review::gate";

/// What the gate made of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateState {
    /// A model judged it a real defect.
    Kept,
    /// A model judged it a false positive with enough confidence; listed apart, never deleted.
    Filtered,
    /// No model reached its threshold, or a critical finding was judged a false positive.
    LowConfidence,
    /// The gate could not judge it; it passes with its cause.
    Unfiltered,
}

/// The gate's verdict on one finding.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GateVerdict {
    /// The state.
    pub state: GateState,
    /// Why, for `unfiltered` and for a critical kept as `low_confidence`.
    pub cause: Option<&'static str>,
    /// The model whose answer decided the state.
    pub model: Option<String>,
    /// The option the deciding (or last) answer leaned to.
    pub lean: Option<String>,
    /// Magi's confidence of that answer.
    pub confidence: Option<f64>,
    /// Models skipped because they answered HTTP 400 in this review, in chain order.
    pub skipped: Vec<String>,
}

/// The `state` sent to `decide` for a finding's first constituent.
///
/// # Arguments
/// * `finding` - the consolidated finding.
/// * `code` - the whole file the finding points into.
///
/// # Returns
/// `{"finding": {file, line, severity, title, detail}, "code": "<numbered lines>"}`.
pub fn gate_state(finding: &ConsolidatedFinding, code: &str) -> serde_json::Value {
    gate_state_of(finding, 0, code)
}

/// The `state` sent to `decide` for the `i`-th distinct (title, detail) pair of a finding.
///
/// # Arguments
/// * `finding` - the consolidated finding.
/// * `i` - index among its distinct (title, detail) pairs, in order of appearance.
/// * `code` - the whole file the finding points into.
///
/// # Returns
/// The state, in the calibration's single-finding shape.
pub fn gate_state_of(finding: &ConsolidatedFinding, i: usize, code: &str) -> serde_json::Value {
    let _ = (finding, i, code);
    todo!()
}

/// The gate's question set.
///
/// # Returns
/// The single `choice` question of the calibration.
pub fn gate_questions() -> Questions {
    todo!()
}

/// The gate of one review.
pub struct Gate<'a> {
    client: &'a dyn DecisionClient,
    chain: Vec<ModelSpec>,
    max_state_bytes: usize,
    call_timeout: Option<Duration>,
    rejected: BTreeSet<String>,
    stopped: Option<&'static str>,
    per_model: BTreeMap<String, usize>,
}

impl<'a> Gate<'a> {
    /// A gate over `chain`.
    ///
    /// # Arguments
    /// * `client` - the decision client.
    /// * `chain` - the models, each with its threshold.
    /// * `max_state_bytes` - the largest serialized state sent.
    ///
    /// # Returns
    /// A gate with no model rejected yet.
    pub fn new(
        client: &'a dyn DecisionClient,
        chain: Vec<ModelSpec>,
        max_state_bytes: usize,
    ) -> Self {
        Self {
            client,
            chain,
            max_state_bytes,
            call_timeout: None,
            rejected: BTreeSet::new(),
            stopped: None,
            per_model: BTreeMap::new(),
        }
    }

    /// Sets the per-call HTTP timeout.
    ///
    /// # Arguments
    /// * `timeout` - timeout of each call.
    ///
    /// # Returns
    /// The gate.
    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = Some(timeout);
        self
    }

    /// Judges one finding.
    ///
    /// # Arguments
    /// * `finding` - the consolidated finding.
    /// * `code` - the file it points into, if it was read.
    /// * `remaining` - what is left of the review clock, if there is one.
    ///
    /// # Returns
    /// The verdict.
    pub async fn judge(
        &mut self,
        finding: &ConsolidatedFinding,
        code: Option<&str>,
        remaining: Option<Duration>,
    ) -> GateVerdict {
        let _ = (
            finding,
            code,
            remaining,
            &self.client,
            &self.chain,
            self.max_state_bytes,
        );
        let _ = (&self.rejected, &mut self.per_model, self.call_timeout);
        todo!()
    }

    /// Answers counted per deciding model.
    ///
    /// # Returns
    /// Model name to number of findings it decided.
    pub fn per_model(&self) -> &BTreeMap<String, usize> {
        &self.per_model
    }

    /// The cause that stopped the gate for the rest of the review, if any.
    ///
    /// # Returns
    /// `unauthorized` or `timeout`, or `None`.
    pub fn stopped(&self) -> Option<&'static str> {
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::client::MockDecisionClient;
    use crate::decision::{Answer, Attempt, AttemptOutcome, Cause, Decision, DecisionError, Trace};
    use magi_core::schema::{AgentName, Category, Severity};

    fn finding(sev: Severity, line: Option<u32>) -> ConsolidatedFinding {
        ConsolidatedFinding {
            file: Some("a.rs".into()),
            line,
            severity: sev,
            reported_severity: sev,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        }
    }
    fn chain() -> Vec<ModelSpec> {
        vec![
            ModelSpec {
                model: "clef:cloud".into(),
                min_confidence: Some(0.40),
            },
            ModelSpec {
                model: "clef-flash".into(),
                min_confidence: Some(0.32),
            },
            ModelSpec {
                model: "nimble".into(),
                min_confidence: Some(0.80),
            },
        ]
    }
    fn answer(model: &str, choice: &str, confidence: f64) -> Result<Decision, DecisionError> {
        let other = if choice == "real_defect" {
            "false_positive"
        } else {
            "real_defect"
        };
        let p = (1.0 + confidence) / 2.0; // Magi's confidence for two options: 2·p − 1
        let answers = [(
            GATE_QUESTION_NAME.to_string(),
            Answer::Choice {
                choice: choice.into(),
                probabilities: [(choice.to_string(), p), (other.to_string(), 1.0 - p)]
                    .into_iter()
                    .collect(),
                confidence,
                provider_confidence: None,
            },
        )]
        .into_iter()
        .collect();
        Ok(Decision {
            model: model.into(),
            answers,
            trace: Trace::default(),
        })
    }
    /// What T2.2 returns for a single explicit model that answered 400.
    fn bad_request(model: &str) -> Result<Decision, DecisionError> {
        Err(DecisionError::Rejected {
            status: 400,
            provider_message: format!("{model}: requires a local model"),
            trace: Trace::default(),
        })
    }

    #[tokio::test]
    async fn a_merged_finding_is_filtered_only_when_every_constituent_is() {
        // [CP2 MS2 cierre (sin MAGI)] RV-S31 meets the gate: each constituent goes in the calibration's single-finding shape.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["noise".into(), "real".into()];
        f.details = vec!["dn".into(), "dr".into()];
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(2).returning(|r| {
            let fp = r.state()["finding"]["title"].as_str() == Some("noise");
            answer(
                r.model().unwrap_or(""),
                if fp { "false_positive" } else { "real_defect" },
                0.9,
            )
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        assert_eq!(
            g.judge(&f, Some("x\n"), None).await.state,
            GateState::Kept,
            "one real defect keeps it"
        );
        let mut c2 = MockDecisionClient::new();
        c2.expect_decide()
            .times(2)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g2 = Gate::new(&c2, chain(), 61_000);
        assert_eq!(
            g2.judge(&f, Some("x\n"), None).await.state,
            GateState::Filtered
        );
    }

    #[test]
    fn the_question_is_verbatim() {
        let q = gate_questions();
        let v = serde_json::to_value(&q).expect("serializes");
        assert_eq!(v["gate"]["type"], "choice");
        assert_eq!(v["gate"]["instructions"], GATE_INSTRUCTIONS);
        assert_eq!(v["gate"]["criteria"]["real_defect"], GATE_REAL_DEFECT);
        assert_eq!(v["gate"]["criteria"]["false_positive"], GATE_FALSE_POSITIVE);
    }

    #[test]
    fn the_state_carries_the_finding_and_forty_numbered_lines_each_side() {
        let code: String = (1..=200).map(|i| format!("l{i}\n")).collect();
        let s = gate_state(&finding(Severity::Warning, Some(100)), &code);
        let text = s["code"].as_str().expect("code");
        assert!(text.starts_with("60: l60\n"), "{}", &text[..20]);
        assert!(text.trim_end().ends_with("140: l140"));
        assert_eq!(s["finding"]["line"], 100);
        assert_eq!(s["finding"]["severity"], "warning");
        // [CP2 MS2 S2a1 loop 10] the rest of RV10's state, and the title a merged finding sends.
        assert_eq!(s["finding"]["file"], "a.rs");
        assert_eq!(
            (
                s["finding"]["title"].as_str(),
                s["finding"]["detail"].as_str()
            ),
            (Some("t"), Some("d"))
        );
        let mut merged = finding(Severity::Warning, Some(100));
        merged.titles.push("t2".into());
        merged.details.push("d2".into());
        assert_eq!(gate_state(&merged, &code)["finding"]["title"], "t");
    }

    #[tokio::test]
    async fn a_confident_false_positive_is_filtered_and_a_real_defect_is_kept() {
        let mut c = MockDecisionClient::new();
        let mut seq = mockall::Sequence::new();
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| answer("clef:cloud", "false_positive", 0.9));
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| answer("clef:cloud", "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        assert_eq!(
            g.judge(&finding(Severity::Warning, Some(3)), Some("x\n"), None)
                .await
                .state,
            GateState::Filtered
        );
        assert_eq!(
            g.judge(&finding(Severity::Warning, Some(3)), Some("x\n"), None)
                .await
                .state,
            GateState::Kept
        );
    }

    #[tokio::test]
    async fn a_model_without_min_confidence_has_no_threshold_like_in_decision() {
        // [CP2 MS2 S2a loop 8] a user-written chain entry that omits min_confidence.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|_| answer("own", "false_positive", 0.05));
        let mut g = Gate::new(
            &c,
            vec![ModelSpec {
                model: "own".into(),
                min_confidence: None,
            }],
            61_000,
        );
        assert_eq!(
            g.judge(&finding(Severity::Warning, Some(3)), Some("x\n"), None)
                .await
                .state,
            GateState::Filtered
        );
    }

    #[tokio::test]
    async fn a_critical_is_never_filtered() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(&finding(Severity::Critical, Some(3)), Some("x\n"), None)
            .await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("critical never filtered"))
        );
    }

    #[tokio::test]
    async fn a_finding_without_a_line_is_unfiltered_without_a_call() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&finding(Severity::Warning, None), None, None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::Unfiltered, Some("no location"))
        );
    }

    #[tokio::test]
    async fn a_state_over_the_limit_is_unfiltered_not_truncated() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 10);
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
            .await;
        assert_eq!(v.cause, Some("state too large"));
    }

    #[tokio::test]
    async fn a_state_carrying_a_secret_is_never_sent() {
        // RV22: every byte that leaves the process passes the secret detector, the gate's state too.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let code = "let k = \"sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some(code), None)
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("secret")));
        let mut f = finding(Severity::Warning, Some(1));
        f.details = vec!["token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaks".into()]; // allow-secret-scan
        let v = g.judge(&f, Some("x\n"), None).await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("secret")));
    }

    #[tokio::test]
    async fn a_bad_request_head_is_skipped_for_the_rest_of_the_review() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .withf(|r| r.model() == Some("clef:cloud"))
            .times(1)
            .returning(|_| bad_request("clef:cloud"));
        c.expect_decide()
            .withf(|r| r.model() == Some("clef-flash"))
            .times(2)
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            let v = g
                .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
                .await;
            assert_eq!(v.model.as_deref(), Some("clef-flash"));
            assert_eq!(v.skipped, vec!["clef:cloud".to_string()]); // [CP2 MS2 S2 loop 2] RV-S15: the skip is visible
        }
        assert_eq!(g.per_model().get("clef-flash"), Some(&2));
    }

    #[tokio::test]
    async fn a_new_review_tries_the_rejected_head_again() {
        // [CP2 MS2 S2a loop 7] U23 / RV-S20: the 400 memory lives in the Gate, and run_review builds one Gate per review
        // (so one per tool call, T2.10). Moving it to the client or a static would skip the head in the next review.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .withf(|r| r.model() == Some("clef:cloud"))
            .times(2)
            .returning(|_| bad_request("clef:cloud"));
        c.expect_decide()
            .withf(|r| r.model() == Some("clef-flash"))
            .times(2)
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        for _ in 0..2 {
            let mut g = Gate::new(&c, chain(), 61_000);
            let v = g
                .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
                .await;
            assert_eq!(v.model.as_deref(), Some("clef-flash")); // `times(2)` on the head is the check: asked once per review
        }
    }

    #[tokio::test]
    async fn a_provider_413_or_422_is_chain_exhausted_and_never_remembered_as_a_400() {
        // [CP2 MS2 S1 loop 6] the memory keys on the typed status, never on the shared `rejected` label.
        // [CP2 MS2 S2 loop 1] 422 too: the spec has no better unfiltered cause for an unprocessable request.
        for status in [413, 422] {
            let mut c = MockDecisionClient::new();
            c.expect_decide()
                .withf(|r| r.model() == Some("clef:cloud"))
                .times(2)
                .returning(move |_| {
                    Err(DecisionError::Rejected {
                        status,
                        provider_message: "too large".into(),
                        trace: Trace::default(),
                    })
                });
            c.expect_decide()
                .withf(|r| r.model() == Some("clef-flash"))
                .times(0);
            let mut g = Gate::new(&c, chain(), 61_000);
            for _ in 0..2 {
                assert_eq!(
                    g.judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
                        .await
                        .cause,
                    Some("chain exhausted"),
                    "{status}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_chain_where_every_model_answers_400_is_exhausted_and_never_called_again() {
        // [CP2 MS2 S2a loop 6] every model lands in `rejected`: chain exhausted, all skipped, later findings cost nothing.
        let mut c = MockDecisionClient::new();
        for m in ["clef:cloud", "clef-flash", "nimble"] {
            c.expect_decide()
                .withf(move |r| r.model() == Some(m))
                .times(1)
                .returning(move |_| bad_request(m));
        }
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            let v = g
                .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
                .await;
            assert_eq!(
                (v.state, v.cause),
                (GateState::Unfiltered, Some("chain exhausted"))
            );
            assert_eq!(
                v.skipped,
                vec![
                    "clef:cloud".to_string(),
                    "clef-flash".into(),
                    "nimble".into()
                ]
            );
        }
    }

    #[tokio::test]
    async fn a_policy_rejection_is_classified_by_its_http_status_never_by_its_text() {
        // [CP2 MS2 S2a loop 7] RV-S34 (user, 2026-10-19): `account policy` is unreachable; a ZDR message on a 400 is a 400.
        let zdr = |m: &str| -> Result<Decision, DecisionError> {
            Err(DecisionError::Rejected {
                status: 400,
                provider_message: format!(
                    "{m}: blocked by your account's zero data retention policy"
                ),
                trace: Trace::default(),
            })
        };
        let mut c = MockDecisionClient::new();
        for m in ["clef:cloud", "clef-flash", "nimble"] {
            c.expect_decide()
                .withf(move |r| r.model() == Some(m))
                .times(1)
                .returning(move |_| zdr(m));
        }
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
            .await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::Unfiltered, Some("chain exhausted"))
        );
        assert_eq!(v.skipped.len(), 3, "each 400 rotated like any other 400");
    }

    /// A single-model chain that ran out of time.
    fn timed_out(model: &str) -> Result<Decision, DecisionError> {
        Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![Attempt {
                    model: model.into(),
                    outcome: AttemptOutcome::Failed {
                        cause: Cause::Timeout,
                        detail: None,
                    },
                    usage: Default::default(),
                }],
                usage: Default::default(),
            },
        })
    }

    #[tokio::test]
    async fn each_call_is_capped_by_what_is_left_and_running_out_mid_gate_is_timeout() {
        // [CP2 MS2 S2 loop 1] N-10: min(call timeout, remaining); a call capped by `remaining` that times out is RV-S16's `timeout`.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            // [CP2 MS2 cierre (sin MAGI)] a range: the cap is what is left of the deadline, a little under 5 s by the time it is sent.
            .withf(|r| {
                r.timeout().is_some_and(|t| {
                    t > std::time::Duration::from_secs(4) && t <= std::time::Duration::from_secs(5)
                })
            })
            .returning(|r| timed_out(r.model().unwrap_or("")));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\n"),
                Some(std::time::Duration::from_secs(5)),
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("timeout")));
        // the review clock is spent: the next finding is not sent at all (times(1) above)
        let v = g
            .judge(
                &finding(Severity::Warning, Some(2)),
                Some("x\n"),
                Some(std::time::Duration::from_secs(5)),
            )
            .await;
        assert_eq!(v.cause, Some("timeout"));
        assert_eq!(g.stopped(), Some("timeout")); // [CP2 MS2 S2 loop 4]
    }

    #[tokio::test]
    async fn the_review_clock_is_re_read_before_each_call_of_the_chain() {
        // [CP2 MS2 S2a loop 9] `remaining` is a deadline taken at entry: the second model gets what is left after the
        // first one answered, never the whole `remaining` again.
        let mut c = MockDecisionClient::new();
        let mut seq = mockall::Sequence::new();
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| {
                std::thread::sleep(std::time::Duration::from_millis(1_100));
                answer("clef:cloud", "false_positive", 0.10) // under clef:cloud's 0.40: next model
            });
        // [CP2 MS2 S2a1 loop 10] 30 s of review left: reusing the entry `remaining` passes 30 s and fails; a stall has
        // to reach ~29 s before the honest implementation flakes (it was 0.9 s with a 2 s budget).
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .withf(|r| {
                r.timeout()
                    .is_some_and(|t| t < std::time::Duration::from_secs(29))
            })
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\n"),
                Some(std::time::Duration::from_secs(30)),
            )
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[tokio::test]
    async fn a_spent_review_clock_never_reaches_the_http_client() {
        // [CP2 MS2 S2a2 loop 10] a cap of zero is never sent as a 0 s HTTP timeout: the call is skipped as `timeout`.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\n"),
                Some(std::time::Duration::ZERO),
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("timeout")));
    }

    #[tokio::test]
    async fn the_first_stop_cause_wins_and_a_spent_clock_does_not_replace_a_401() {
        // [CP2 MS2 S2 loop 4] precedence: stopped is write-once.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        g.judge(
            &finding(Severity::Warning, Some(1)),
            Some("x\n"),
            Some(std::time::Duration::from_secs(60)),
        )
        .await;
        let v = g
            .judge(
                &finding(Severity::Warning, Some(2)),
                Some("x\n"),
                Some(std::time::Duration::ZERO),
            )
            .await;
        assert_eq!(
            (v.cause, g.stopped()),
            (Some("unauthorized"), Some("unauthorized"))
        );
    }

    #[tokio::test]
    async fn every_model_timing_out_on_its_own_call_timeout_is_transport() {
        // [CP2 MS2 S2 loop 4] per-call timeouts are transport-class; `timeout` is reserved for the review clock (RV-S16).
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(3)
            .returning(|r| timed_out(r.model().unwrap_or("")));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
            .await;
        assert_eq!(
            (v.state, v.cause, g.stopped()),
            (GateState::Unfiltered, Some("transport"), None)
        );
    }

    #[tokio::test]
    async fn with_time_to_spare_each_call_uses_the_configured_timeout() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .withf(|r| r.timeout() == Some(std::time::Duration::from_secs(120)))
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\n"),
                Some(std::time::Duration::from_secs(5_000)),
            )
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[tokio::test]
    async fn every_unfiltered_finding_logs_one_warn_never_deduplicated() {
        // [CP2 MS2 S2 loop 1] RV16. A counting layer on GATE_LOG_TARGET at WARN.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Count(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Count {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if e.metadata().target() == GATE_LOG_TARGET
                    && *e.metadata().level() == tracing::Level::WARN
                {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let count = Count::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(count.clone()));
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            g.judge(&finding(Severity::Warning, None), None, None).await; // the same cause twice
        }
        assert_eq!(count.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unauthorized_stops_the_gate_for_the_whole_review() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..3 {
            assert_eq!(
                g.judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
                    .await
                    .cause,
                Some("unauthorized")
            );
        }
        assert_eq!(g.stopped(), Some("unauthorized")); // [CP2 MS2 S2 loop 3] what the report declares once
    }

    #[tokio::test]
    async fn a_low_confidence_chain_passes_with_its_lean() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.10));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some("x\n"), None)
            .await;
        assert_eq!(
            (v.state, v.lean.as_deref()),
            (GateState::LowConfidence, Some("false_positive"))
        );
    }
}
