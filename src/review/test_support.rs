// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! Test doubles shared by the `review` module tests.

use std::collections::BTreeMap;

use super::error::ReviewError;
use super::source::{ContentSource, GitCommand};

/// Answers git by the exact joined argument list; anything else is an error naming it.
pub(crate) struct FakeGit(pub(crate) BTreeMap<String, Result<String, String>>);

impl FakeGit {
    pub(crate) fn from(pairs: &[(&str, Result<&str, &str>)]) -> Self {
        Self(
            pairs
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        (*v).map(str::to_string).map_err(str::to_string),
                    )
                })
                .collect(),
        )
    }
}

impl GitCommand for FakeGit {
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        self.0
            .get(&args.join(" "))
            .cloned()
            .unwrap_or_else(|| Err(format!("unexpected: {}", args.join(" "))))
            .map(String::into_bytes)
            .map_err(ReviewError::Git)
    }
}

/// An in-memory content source: `new` side and `old` side by path.
pub(crate) struct MapSource {
    new: BTreeMap<String, Vec<u8>>,
    old: BTreeMap<String, Vec<u8>>,
}

impl MapSource {
    pub(crate) fn new(new: &[(&str, &str)], old: &[(&str, &str)]) -> Self {
        let m = |xs: &[(&str, &str)]| {
            xs.iter()
                .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
                .collect()
        };
        Self {
            new: m(new),
            old: m(old),
        }
    }

    pub(crate) fn with_bytes(new: &[(&str, &[u8])]) -> Self {
        Self {
            new: new
                .iter()
                .map(|(p, b)| (p.to_string(), b.to_vec()))
                .collect(),
            old: BTreeMap::new(),
        }
    }
}

impl ContentSource for MapSource {
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        Ok(self.new.get(path).cloned())
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        Ok(self.old.get(path).cloned())
    }
}

/// Builds an `AgentOutput` through `Deserialize`, the only constructor magi-core offers.
pub(crate) fn out(
    agent: &str,
    verdict: &str,
    confidence: f64,
    findings: serde_json::Value,
) -> magi_core::schema::AgentOutput {
    serde_json::from_value(serde_json::json!({
        "agent": agent,
        "verdict": verdict,
        "confidence": confidence,
        "summary": "s",
        "reasoning": "r",
        "findings": findings,
        "recommendation": "x"
    }))
    .expect("a valid agent output")
}

/// One mage's verdict block, as a real seat emits it.
pub(crate) fn verdict(
    agent: &str,
    verdict: &str,
    confidence: f64,
    findings: serde_json::Value,
) -> String {
    let body = serde_json::json!({
        "agent": agent,
        "verdict": verdict,
        "confidence": confidence,
        "summary": format!("{agent} s"),
        "reasoning": "r",
        "recommendation": "go",
        "findings": findings
    });
    format!(
        "{}\n{body}\n{}",
        magi_core::verdict_markers::VERDICT_OPEN,
        magi_core::verdict_markers::VERDICT_CLOSE
    )
}

/// A REAL `MagiReport`, produced by magi-core's orchestrator over its no-network provider.
pub(crate) async fn real_report(
    findings: [serde_json::Value; 3],
) -> magi_core::reporting::MagiReport {
    use magi_core::schema::AgentName;
    let [m, b, c] = findings;
    let provider = magi_core::test_support::RoutingMockProvider::new()
        .with_agent_responses(
            AgentName::Melchior,
            vec![Ok(verdict("melchior", "approve", 0.9, m))],
        )
        .with_agent_responses(
            AgentName::Balthasar,
            vec![Ok(verdict("balthasar", "approve", 0.9, b))],
        )
        .with_agent_responses(
            AgentName::Caspar,
            vec![Ok(verdict("caspar", "approve", 0.9, c))],
        );
    magi_core::orchestrator::Magi::new(std::sync::Arc::new(provider))
        .analyze(&magi_core::schema::Mode::CodeReview, "review")
        .await
        .expect("a canned report")
}
