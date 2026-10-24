// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Consolidation of the batch reports of one review into a single report (REQ-RV12), and the
//! agreement weight applied after the gate (REQ-RV11).

use std::collections::{BTreeMap, BTreeSet};

use magi_core::consensus::ConsensusResult;
use magi_core::schema::{AgentName, AgentOutput, Category, Severity};

use super::error::ReviewError;

/// What one batch's `consult` returned: the mages that answered and their outputs.
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    /// Zero-based batch index.
    pub batch: usize,
    /// The outputs of the mages that answered, as the batch report carried them.
    pub agents: Vec<AgentOutput>,
    /// The mages that answered in this batch.
    pub answered: BTreeSet<AgentName>,
}

/// One finding of the consolidated report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidatedFinding {
    /// The reported file, if any.
    pub file: Option<String>,
    /// The reported 1-based line, if any.
    pub line: Option<u32>,
    /// The severity after the agreement weight.
    pub severity: Severity,
    /// The highest severity the trio reported.
    pub reported_severity: Severity,
    /// Every title merged into this finding, in order of appearance.
    pub titles: Vec<String>,
    /// Every detail merged into this finding, in order of appearance.
    pub details: Vec<String>,
    /// Every original category merged into this finding, in order of appearance.
    pub categories: Vec<Category>,
    /// The mages that reported it.
    pub sources: Vec<AgentName>,
    /// Marks set by consolidation and by the agreement weight.
    pub marks: Vec<FindingMark>,
    /// The batches where it appeared, in order.
    pub batches: Vec<usize>,
}

/// A mark on a consolidated finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingMark {
    /// Only one mage reported it while two or more answered; one level lower.
    SingleMage,
    /// Only one mage answered where it appeared; nothing to weigh.
    PartialTrio,
    /// Grouped without asking the consensus engine, past [`MAX_UNLOCATED_PROBES`].
    Ungrouped,
}

/// The consolidated review.
#[derive(Debug, Clone)]
pub struct Consolidated {
    /// One synthetic output per mage.
    pub agents: Vec<AgentOutput>,
    /// The consensus over those outputs.
    pub consensus: ConsensusResult,
    /// The consolidated findings.
    pub findings: Vec<ConsolidatedFinding>,
    /// Which mages answered in each batch.
    pub answered: BTreeMap<usize, BTreeSet<AgentName>>,
    /// Unlocated findings that formed their own group because the probe budget was spent.
    pub ungrouped_unlocated: usize,
}

/// Probes one review may spend grouping unlocated findings.
pub const MAX_UNLOCATED_PROBES: usize = 100_000;

/// Merges the batch reports of one review into one report.
///
/// # Errors
///
/// [`ReviewError::Consolidation`] when the consensus cannot be computed.
pub fn consolidate(batches: &[BatchOutcome]) -> Result<Consolidated, ReviewError> {
    let _ = batches;
    todo!("T2.7 Green")
}

/// [`consolidate`] with an explicit probe budget.
#[cfg(test)]
pub(crate) fn consolidate_with_probe_cap(
    batches: &[BatchOutcome],
    cap: usize,
) -> Result<Consolidated, ReviewError> {
    let _ = (batches, cap);
    todo!("T2.7 Green")
}

/// Applies the agreement weight (REQ-RV11).
pub fn apply_agreement(
    findings: &mut [ConsolidatedFinding],
    answered: &BTreeMap<usize, BTreeSet<AgentName>>,
) {
    let _ = (findings, answered);
    todo!("T2.7 Green")
}

/// The group an unlocated title belongs to.
#[allow(dead_code)] // Red stub
fn group_of(title: &str, reps: &[String], same: &dyn Fn(&str, &str) -> bool) -> Option<usize> {
    let _ = (title, reps, same);
    todo!("T2.7 Green")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::{out, real_report};
    use magi_core::schema::Verdict;

    fn test_finding() -> ConsolidatedFinding {
        ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Info,
            reported_severity: Severity::Info,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior],
            marks: vec![],
            batches: vec![0],
        }
    }
    fn batch(i: usize, agents: Vec<AgentOutput>) -> BatchOutcome {
        let answered = agents.iter().map(|a| a.agent).collect();
        BatchOutcome {
            batch: i,
            agents,
            answered,
        }
    }

    #[test]
    fn a_mage_takes_its_worst_verdict_across_batches() {
        let c = consolidate(&[
            batch(
                0,
                vec![
                    out("melchior", "approve", 0.9, serde_json::json!([])),
                    out("balthasar", "approve", 0.9, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
            batch(
                1,
                vec![
                    out("melchior", "reject", 0.7, serde_json::json!([])),
                    out("balthasar", "approve", 0.8, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
        ])
        .expect("consolidates");
        let m = c
            .agents
            .iter()
            .find(|a| a.agent == AgentName::Melchior)
            .expect("melchior");
        assert_eq!(m.verdict, Verdict::Reject);
        assert!((m.confidence - 0.7).abs() < 1e-9);
        assert_eq!(
            c.consensus.votes.get(&AgentName::Melchior),
            Some(&Verdict::Reject)
        );
    }

    #[test]
    fn the_same_finding_in_two_batches_appears_once_at_the_higher_severity() {
        let f = |sev: &str| serde_json::json!([{"severity": sev, "title":"x","detail":"d","file":"a.rs","line":3}]);
        // [RED-DRYRUN 2026-10-17, D14] A second mage per batch: alone, Melchior is one agent and
        // ConsensusEngine::default() (min_agents = 2) returns InsufficientAgents, which T2.9 relies on as a
        // consolidation failure.
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let c = consolidate(&[
            batch(
                0,
                vec![out("melchior", "approve", 0.9, f("warning")), quiet()],
            ),
            batch(
                1,
                vec![out("melchior", "approve", 0.9, f("critical")), quiet()],
            ),
        ])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].reported_severity, Severity::Critical);
    }

    #[test]
    fn different_categories_on_the_same_line_are_one_finding_with_both_sources() {
        let c = consolidate(&[batch(0, vec![
            out("melchior", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"a","detail":"da","file":"src/a.rs","line":10,"category":"logic-error"}])),
            out("caspar", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"b","detail":"db","file":"src/a.rs","line":10,"category":"error-handling"}])),
        ])]).expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        let f = &c.findings[0];
        assert_eq!(f.sources.len(), 2);
        assert_eq!(f.titles, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(f.details, vec!["da".to_string(), "db".to_string()]);
        assert_eq!(
            f.categories,
            vec![Category::LogicError, Category::ErrorHandling]
        );
    }

    #[test]
    fn a_single_mage_finding_drops_one_level_and_a_shared_one_keeps_it() {
        let mut fs = vec![
            ConsolidatedFinding {
                severity: Severity::Critical,
                reported_severity: Severity::Critical,
                sources: vec![AgentName::Melchior],
                ..test_finding()
            },
            ConsolidatedFinding {
                severity: Severity::Warning,
                reported_severity: Severity::Warning,
                sources: vec![AgentName::Melchior, AgentName::Caspar],
                ..test_finding()
            },
            ConsolidatedFinding {
                severity: Severity::Info,
                reported_severity: Severity::Info,
                sources: vec![AgentName::Caspar],
                ..test_finding()
            },
        ];
        let all: BTreeSet<AgentName> =
            [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar]
                .into_iter()
                .collect();
        apply_agreement(&mut fs, &[(0, all)].into_iter().collect());
        assert_eq!(
            (fs[0].severity, fs[0].marks.clone()),
            (Severity::Warning, vec![FindingMark::SingleMage])
        );
        assert_eq!(
            (fs[1].severity, fs[1].marks.clone()),
            (Severity::Warning, vec![])
        );
        assert_eq!(
            (fs[2].severity, fs[2].marks.clone()),
            (Severity::Info, vec![FindingMark::SingleMage])
        );
    }

    #[test]
    fn when_only_one_mage_answered_there_is_no_disagreement_to_weigh() {
        let mut fs = vec![ConsolidatedFinding {
            severity: Severity::Critical,
            reported_severity: Severity::Critical,
            sources: vec![AgentName::Melchior],
            ..test_finding()
        }];
        apply_agreement(
            &mut fs,
            &[(0, [AgentName::Melchior].into_iter().collect())]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            (fs[0].severity, fs[0].marks.clone()),
            (Severity::Critical, vec![FindingMark::PartialTrio])
        );
    }

    #[tokio::test]
    async fn consolidating_a_real_report_matches_its_own_consensus() {
        // PM-S-1: the AgentOutputs come from magi-core's orchestrator, not from a literal.
        let f = serde_json::json!([{"severity":"warning","title":"x","detail":"d","file":"a.rs","line":3}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        assert_eq!(
            c.consensus.consensus_verdict,
            report.consensus.consensus_verdict
        );
        assert_eq!(c.consensus.votes, report.consensus.votes);
        assert_eq!(c.findings.len(), report.consensus.findings.len());
        assert_eq!(
            c.findings[0].sources.len(),
            report.consensus.findings[0].sources.len()
        );
    }

    #[tokio::test]
    async fn unlocated_findings_group_like_magi_core() {
        // [CP2 MS2 S2 loop 2] PM-S-1 for unlocated findings: same title, different categories → magi-core's key is the title alone.
        let m = serde_json::json!([{"severity":"warning","title":"Leaks  A Handle","detail":"d","category":"logic-error"}]);
        let c2 = serde_json::json!([{"severity":"info","title":"leaks  a handle","detail":"d2","category":"error-handling"}]);
        let report = real_report([m, serde_json::json!([]), c2]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        assert_eq!(c.findings.len(), report.consensus.findings.len());
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].sources.len(), 2);
    }

    #[test]
    fn a_finding_with_a_file_and_no_line_groups_by_title_like_magi_core() {
        // [CP2 MS2 S2a loop 7] load-bearing for the sondas: magi-core keys `file` + no line by title (consensus.rs:210).
        // Two mages, same title, different files, no line → one group, both sources. Goes on the magi-core bump checklist.
        let a = serde_json::json!([{"severity":"warning","title":"leaks a handle","detail":"da","file":"a.rs"}]);
        let b = serde_json::json!([{"severity":"warning","title":"leaks a handle","detail":"db","file":"b.rs"}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, a),
                out("caspar", "approve", 0.9, b),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(
            c.findings[0].sources,
            vec![AgentName::Melchior, AgentName::Caspar]
        );
    }

    #[test]
    fn past_the_probe_cap_unlocated_findings_stay_apart_and_are_counted() {
        // [CP2 MS2 S2a loop 8] the O(u·g) probe is bounded: past the cap no probe runs and nothing is dropped.
        let m = serde_json::json!([{"severity":"warning","title":"Leaks  A Handle","detail":"d"}]);
        let c2 =
            serde_json::json!([{"severity":"warning","title":"leaks  a handle","detail":"d2"}]);
        let b = [batch(
            0,
            vec![
                out("melchior", "approve", 0.9, m),
                out("caspar", "approve", 0.9, c2),
            ],
        )];
        let open = consolidate(&b).expect("consolidates");
        assert_eq!(
            (open.findings.len(), open.ungrouped_unlocated),
            (1, 0),
            "magi-core merges them"
        );
        let capped = consolidate_with_probe_cap(&b, 0).expect("consolidates");
        assert_eq!(
            capped.findings.len(),
            2,
            "no probe ran: the second title is its own group"
        );
        assert_eq!(capped.ungrouped_unlocated, 1);
        // [CP2 MS2 cierre (sin MAGI)] a fallback group's sources come from its own input findings, never from a group it left.
        assert_eq!(
            capped
                .findings
                .iter()
                .map(|f| f.sources.clone())
                .collect::<Vec<_>>(),
            vec![vec![AgentName::Melchior], vec![AgentName::Caspar]]
        );
        // [CP2 MS2 S2a loop 9] past the cap the agreement was never measured: marked, and no single_mage downgrade.
        assert!(
            capped
                .findings
                .iter()
                .all(|f| f.marks == vec![FindingMark::Ungrouped]),
            "{:?}",
            capped.findings
        );
        let mut fs = capped.findings.clone();
        apply_agreement(&mut fs, &capped.answered);
        assert!(fs.iter().all(
            |f| f.severity == Severity::Warning && !f.marks.contains(&FindingMark::SingleMage)
        ));
    }

    #[test]
    fn membership_is_decided_by_magi_core_not_by_a_local_key() {
        // [CP2 MS2 S2 loop 4] "\u{fb01}" (the "fi" ligature) folds to "fi" under NFKC: magi-core groups both titles; a naive
        // lowercase key would not. Both texts survive the merge (RV12, no loss).
        let a = serde_json::json!([{"severity":"warning","title":"Leaks a \u{fb01}le handle","detail":"da"}]);
        let b =
            serde_json::json!([{"severity":"warning","title":"leaks a file handle","detail":"db"}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, a),
                out("caspar", "approve", 0.9, b),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].titles.len(), 2);
        assert_eq!(
            c.findings[0].sources,
            vec![AgentName::Melchior, AgentName::Caspar]
        );
    }

    #[test]
    fn a_mage_repeating_a_finding_across_batches_is_one_source() {
        // [CP2 MS2 S2 loop 4] sources come from magi-core (E18): same mage, same key, two batches → one source → single_mage.
        let f = serde_json::json!([{"severity":"critical","title":"x","detail":"d","file":"a.rs","line":3}]);
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let mut c = consolidate(&[
            batch(0, vec![out("melchior", "approve", 0.9, f.clone()), quiet()]),
            batch(1, vec![out("melchior", "approve", 0.9, f), quiet()]),
        ])
        .expect("consolidates");
        assert_eq!(c.findings[0].sources, vec![AgentName::Melchior]);
        assert_eq!(c.findings[0].batches, vec![0, 1]);
        let answered = c.answered.clone();
        apply_agreement(&mut c.findings, &answered);
        assert_eq!(
            (c.findings[0].severity, c.findings[0].marks.clone()),
            (Severity::Warning, vec![FindingMark::SingleMage])
        );
    }

    #[test]
    fn a_finding_no_group_claims_is_never_dropped() {
        // [CP2 MS2 S2 loop 5] the fallback: group_of says None, so consolidate gives the finding its own group (RV12, no loss).
        assert_eq!(group_of("x", &["y".to_string()], &|_, _| false), None);
        assert_eq!(
            group_of("y", &["y".to_string()], &|_, _| panic!(
                "identical titles never probe"
            )),
            Some(0)
        );
    }

    #[test]
    fn magi_core_still_refuses_a_one_mage_consensus() {
        // [CP2 MS2 S2 loop 4] RV-S27 is declared unreachable while this holds: partial_trio needs a one-mage report.
        // If magi-core changes its default, this goes red and RV-S27 applies again.
        assert_eq!(
            magi_core::consensus::ConsensusEngine::default().min_agents(),
            2
        );
    }

    #[test]
    fn consolidated_findings_hold_only_redacted_model_text() {
        // [CP2 MS2 S2 loop 3] redaction at ingestion: no serializable struct keeps a credential a mage echoed.
        let f = serde_json::json!([{"severity":"warning","title":"t https://u:hunter2@example.invalid",
                                    "detail":"see https://user:hunter2@example.invalid/x","file":"a.rs","line":1}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        let all = serde_json::to_string(&c.findings).expect("serializes");
        assert!(!all.contains("hunter2"), "{all}");
    }

    #[test]
    fn one_mage_or_no_batch_is_a_consolidation_error() {
        // [CP2 MS2 S2a1 loop 10] the M-9 trigger T2.9 relies on (D14): magi-core refuses a one-agent consensus.
        let one = consolidate(&[batch(
            0,
            vec![out("melchior", "approve", 0.9, serde_json::json!([]))],
        )]);
        assert!(matches!(one, Err(ReviewError::Consolidation(_))));
        assert!(matches!(
            consolidate(&[]),
            Err(ReviewError::Consolidation(_))
        ));
    }

    #[test]
    fn a_per_mage_union_past_magi_cores_max_findings_is_consolidated_whole() {
        // [CP2 MS2 S2a1 loop 10] magi-core caps a mage at 100 findings when it validates a response (validate.rs:212),
        // never in `determine` (consensus.rs:253): three batches of 50 give a 150-finding union per mage, all kept.
        let many = |b: usize| {
            serde_json::Value::Array((0..50).map(|i| serde_json::json!(
            {"severity":"info","title":format!("t{b}-{i}"),"detail":"d","file":"a.rs","line":b * 100 + i + 1})).collect())
        };
        let batches: Vec<_> = (0..3)
            .map(|b| {
                batch(
                    b,
                    vec![
                        out("melchior", "approve", 0.9, many(b)),
                        out("caspar", "approve", 0.9, many(b)),
                    ],
                )
            })
            .collect();
        let c = consolidate(&batches).expect("consolidates");
        assert_eq!(c.findings.len(), 150);
        assert!(c.findings.iter().all(|f| f.sources.len() == 2));
    }

    #[test]
    fn a_zero_line_or_an_empty_file_groups_by_title_like_magi_core() {
        // [CP2 MS2 S2a1 loop 10] magi-core id-keys only a non-empty file with line > 0 (consensus.rs:203-216); the split
        // here is the same, so these group by title with both sources instead of falling to the WARN branch.
        let f = serde_json::json!([{"severity":"warning","title":"zero","detail":"d","file":"a.rs","line":0},
                                   {"severity":"warning","title":"empty","detail":"d","file":"","line":3}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 2);
        assert!(c.findings.iter().all(|x| x.sources.len() == 2));
    }

    #[test]
    fn redacting_model_text_twice_changes_nothing() {
        // [CP2 MS2 S2a1 loop 10] ingestion and write time both redact: the second pass must be a no-op, so the report and
        // the gate state show the same text.
        for raw in [
            "see https://user:hunter2@example.invalid/x",
            "two: https://a:b@h.invalid and postgres://u:p@db.invalid/x",
            "plain text",
        ] {
            let once = crate::redact::redact_foreign_text(raw).as_str().to_string();
            assert_eq!(
                crate::redact::redact_foreign_text(&once).as_str(),
                once,
                "{raw}"
            );
        }
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"see https://user:hunter2@example.invalid/x",
                                    "file":"a.rs","line":1}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        let d = &c.findings[0].details[0];
        assert_eq!(crate::redact::redact_foreign_text(d).as_str(), d.as_str());
    }

    #[test]
    fn agreement_counts_who_answered_in_the_batches_where_the_finding_appeared() {
        // [CP2 MS2 S2 loop 1] N-7 with real provenance: the finding appears only in batch 1, where Melchior alone answered.
        let crit = serde_json::json!([{"severity":"critical","title":"c","detail":"d","file":"b.rs","line":2}]);
        let mut c = consolidate(&[
            batch(
                0,
                vec![
                    out("melchior", "approve", 0.9, serde_json::json!([])),
                    out("balthasar", "approve", 0.9, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
            batch(1, vec![out("melchior", "reject", 0.8, crit)]),
        ])
        .expect("three mages across the review");
        assert_eq!(c.findings[0].batches, vec![1]);
        assert_eq!(c.answered[&1].len(), 1);
        let answered = c.answered.clone();
        apply_agreement(&mut c.findings, &answered);
        assert_eq!(
            (c.findings[0].severity, c.findings[0].marks.clone()),
            (Severity::Critical, vec![FindingMark::PartialTrio])
        );
    }
}
