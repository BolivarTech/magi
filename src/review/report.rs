// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! What a review reports (REQ-RV13): the summary of every stage, its JSON form and its text form.

use std::collections::BTreeMap;

use magi_core::reporting::MagiReport;

use super::consolidate::{Consolidated, ConsolidatedFinding};
use super::filter::Skipped;
use super::gate::GateVerdict;
use super::validate::Discarded;

/// What every stage of a review declared.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ReviewSummary {
    /// Findings per gate state (`kept`, `filtered`, `low_confidence`, `unfiltered`).
    pub gate_states: BTreeMap<String, usize>,
    /// Findings per gate cause.
    pub gate_causes: BTreeMap<String, usize>,
    /// Findings each model of the gate chain decided.
    pub per_model: BTreeMap<String, usize>,
    /// Fraction of the findings that reached the gate and got an answer.
    pub judged_fraction: f64,
    /// Findings removed by the deterministic validation.
    pub discarded: Vec<Discarded>,
    /// Files left out of the review.
    pub skipped: Vec<Skipped>,
    /// Reviewed files with no language extractor.
    pub no_extractor: Vec<String>,
    /// Pool models whose window was not measured.
    pub unmeasured_models: Vec<String>,
    /// Milliseconds the preparation took (never charged to `--timeout`).
    pub prep_ms: u64,
    /// Per-batch metrics, for the batches that completed.
    pub batches: Vec<BatchMetrics>,
    /// Batches that completed.
    pub batches_completed: usize,
    /// Batches the review was split into.
    pub batches_planned: usize,
    /// The findings of the main list, with their gate verdict.
    pub findings: Vec<(ConsolidatedFinding, GateVerdict)>,
    /// The findings the gate filtered, listed apart.
    pub filtered: Vec<(ConsolidatedFinding, GateVerdict)>,
    /// Whether the ceiling derived from `--timeout` is below `[magi].agent_timeout_secs`.
    pub ceiling_below_configured: bool,
    /// Batches that did not complete, with why.
    pub failed_batches: Vec<(usize, String)>,
    /// The cause that stopped the gate, declared once.
    pub gate_stopped: Option<String>,
    /// Whether the symbol check was off because the lexical index hit its cap.
    pub symbol_check_skipped: bool,
    /// Critical findings whose only failed check was the symbol one.
    pub symbol_unverified: Vec<Discarded>,
    /// Unlocated findings grouped without the consensus engine.
    pub ungrouped_unlocated: usize,
}

/// The metrics of one completed batch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchMetrics {
    /// Zero-based batch index.
    pub index: usize,
    /// Prompt tokens of every completion of the batch.
    pub input_tokens: u64,
    /// Completion tokens of every completion of the batch.
    pub output_tokens: u64,
    /// Wall clock of the batch's `consult`, in milliseconds.
    pub ms: u64,
}

/// The text report of a consolidated review.
pub fn render_text(c: &Consolidated, s: &ReviewSummary) -> String {
    let _ = (c, s);
    todo!()
}

/// The JSON form of a summary.
pub fn summary_json(s: &ReviewSummary) -> serde_json::Value {
    let _ = s;
    todo!()
}

/// The paid batch reports, when consolidation failed.
pub fn render_batch_reports(reports: &[MagiReport]) -> String {
    let _ = reports;
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::consolidate::{consolidate, BatchOutcome};
    use crate::review::gate::GateState;
    use crate::review::test_support::real_report;
    use crate::review::validate::DiscardCause;
    use magi_core::schema::{AgentName, Category, Severity};

    #[test]
    fn the_summary_json_counts_states_and_models_and_never_carries_raw_model_text_unredacted() {
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["see https://user:hunter2@example.invalid/x".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::Filtered,
            cause: None,
            model: Some("clef:cloud".into()),
            lean: Some("false_positive".into()),
            confidence: Some(0.9),
            skipped: vec!["nimble".into()],
        };
        let mut s = ReviewSummary::default();
        s.gate_states.insert("filtered".into(), 1);
        s.per_model.insert("clef:cloud".into(), 1);
        s.filtered.push((finding, verdict));
        // [CP2 MS2 S2a2 loop 10] the failed-batch messages get the same backstop as model text.
        s.failed_batches.push((
            1,
            "consult failed: https://user:hunter2@example.invalid/x".into(),
        ));
        // [CP2 MS2 cierre (sin MAGI)] the discarded and symbol_unverified lists carry model text too.
        let leak = magi_core::schema::Finding::new(
            Severity::Critical,
            "t https://user:hunter2@example.invalid/x",
            "see https://user:hunter2@example.invalid/y",
        );
        s.discarded.push(Discarded {
            agent: AgentName::Melchior,
            batch: 0,
            finding: leak.clone(),
            cause: DiscardCause::LineOutOfRange,
            symbol: None,
        });
        s.symbol_unverified.push(Discarded {
            agent: AgentName::Caspar,
            batch: 0,
            finding: leak,
            cause: DiscardCause::UnknownSymbol,
            symbol: Some("ghost".into()),
        });
        s.batches = vec![
            BatchMetrics {
                index: 0,
                input_tokens: 10,
                output_tokens: 1,
                ms: 3,
            },
            BatchMetrics {
                index: 1,
                input_tokens: 20,
                output_tokens: 2,
                ms: 4,
            },
        ];
        let v = summary_json(&s);
        assert_eq!(v["gate_states"]["filtered"], 1);
        assert_eq!(v["per_model"]["clef:cloud"], 1);
        assert!(!v.to_string().contains("hunter2"), "{v}");
        // [CP2 MS2 S2 loop 2] the shape S47 reads, the RV13 totals and the RV-S15 skips
        assert_eq!(v["filtered"][0]["file"], "a.rs");
        assert_eq!(v["filtered"][0]["gate"]["state"], "filtered");
        assert_eq!(v["rejected_skips"]["nimble"], 1);
        assert_eq!(
            (
                v["batch_totals"]["input_tokens"].as_u64(),
                v["batch_totals"]["ms"].as_u64()
            ),
            (Some(30), Some(7))
        );
        // [CP2 MS2 S2a2 loop 10] every documented key is present, so a dropped field goes red.
        for k in [
            "gate_states",
            "gate_causes",
            "per_model",
            "judged_fraction",
            "discarded",
            "skipped",
            "no_extractor",
            "unmeasured_models",
            "prep_ms",
            "batches",
            "batches_completed",
            "batches_planned",
            "findings",
            "filtered",
            "ceiling_below_configured",
            "failed_batches",
            "gate_stopped",
            "symbol_check_skipped",
            "symbol_unverified",
            "ungrouped_unlocated",
            "batch_totals",
            "rejected_skips",
        ] {
            assert!(v.get(k).is_some(), "missing {k}: {v}");
        }
    }

    #[tokio::test]
    async fn the_text_report_redacts_model_text_too() {
        // [CP2 MS2 S2 loop 2] the subcommand's stdout is render_text: a credential echoed by a mage never reaches it.
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"see https://user:hunter2@example.invalid/x","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let mut c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        assert!(!render_text(&c, &ReviewSummary::default()).contains("hunter2"));
        // [CP2 MS2 S2 loop 5] render_text's own redaction, bypassing consolidate's: a raw detail injected after consolidation.
        c.findings[0].details = vec!["see https://user:hunter2@example.invalid/x".into()];
        assert!(!render_text(&c, &ReviewSummary::default()).contains("hunter2"));
        // [CP2 MS2 S2a2 loop 10] the failed-batch messages get the same backstop.
        let mut s = ReviewSummary::default();
        s.failed_batches.push((
            1,
            "consult failed: https://user:hunter2@example.invalid/x".into(),
        ));
        assert!(!render_text(&c, &s).contains("hunter2"));
        // [CP2 MS2 cierre (sin MAGI)] and the discarded list it renders.
        s.discarded.push(Discarded {
            agent: AgentName::Melchior,
            batch: 0,
            finding: magi_core::schema::Finding::new(
                Severity::Critical,
                "t https://user:hunter2@example.invalid/x",
                "d",
            ),
            cause: DiscardCause::LineOutOfRange,
            symbol: None,
        });
        assert!(!render_text(&c, &s).contains("hunter2"));
    }
}
