// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! What a review reports (REQ-RV13): the summary of every stage, its JSON form and its text form.
//!
//! Both forms pass every string through [`redact_foreign_text`] as they are written: the
//! summary holds model text (titles, details, failure messages) and this is the last point
//! before it leaves the process, so the redaction here is a backstop over the one consolidation
//! already applied.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use magi_core::consensus::DedupFinding;
use magi_core::reporting::{MagiReport, ReportFormatter};
use serde_json::{json, Map, Value};

use super::consolidate::{Consolidated, ConsolidatedFinding};
use super::filter::Skipped;
use super::gate::GateVerdict;
use super::validate::Discarded;
use crate::redact::redact_foreign_text;

/// The key under which a finding of the JSON form carries its gate verdict.
const GATE_KEY: &str = "gate";
/// How the titles of a merged finding are joined on one line.
const TITLE_JOINER: &str = " / ";

/// What every stage of a review declared.
///
/// Batch indices are 0-based here; the text and JSON forms show them 1-based, the
/// same base as the `[batch k]` labels of the consolidated summaries.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ReviewSummary {
    /// Findings per gate state (`kept`, `filtered`, `low_confidence`, `unfiltered`).
    pub gate_states: BTreeMap<String, usize>,
    /// Findings per gate cause.
    pub gate_causes: BTreeMap<String, usize>,
    /// Findings whose gate state each model of the chain decided (one per finding).
    pub per_model: BTreeMap<String, usize>,
    /// Questions the gate asked: one per (title, detail) pair put to the chain, at most
    /// `MAX_GATE_QUESTIONS_PER_FINDING` per finding.
    pub gate_questions: usize,
    /// Fraction of the findings that reached the gate and got an answer; 1.0 with none.
    pub judged_fraction: f64,
    /// Findings removed by the deterministic validation, with their cause.
    pub discarded: Vec<Discarded>,
    /// Files left out of the review, with their cause.
    pub skipped: Vec<Skipped>,
    /// Reviewed files with no language extractor.
    pub no_extractor: Vec<String>,
    /// Pool models whose window was not measured.
    pub unmeasured_models: Vec<String>,
    /// Milliseconds the preparation took (never charged to `--timeout`).
    pub prep_ms: u64,
    /// Per-batch metrics, for every batch that ran: a failed batch has its time and no tokens
    /// (its report never arrived); a batch cut before it started has none.
    pub batches: Vec<BatchMetrics>,
    /// Batches that completed.
    pub batches_completed: usize,
    /// Batches the review was split into.
    pub batches_planned: usize,
    /// The findings of the main list, with their gate verdict.
    pub findings: Vec<(ConsolidatedFinding, GateVerdict)>,
    /// The findings the gate filtered, listed apart and never deleted.
    pub filtered: Vec<(ConsolidatedFinding, GateVerdict)>,
    /// Whether the ceiling of the initial split of `--timeout` is below
    /// `[magi].agent_timeout_secs` (a declaration of that split, not of each batch).
    pub ceiling_below_configured: bool,
    /// Batches that did not complete, with why.
    pub failed_batches: Vec<(usize, String)>,
    /// The cause that stopped the gate, declared once.
    pub gate_stopped: Option<String>,
    /// Whether the symbol check was off because the lexical index hit its cap.
    pub symbol_check_skipped: bool,
    /// Critical findings whose only failed check was the symbol one; they reached the gate.
    pub symbol_unverified: Vec<Discarded>,
    /// Unlocated findings grouped without the consensus engine.
    pub ungrouped_unlocated: usize,
}

/// The metrics of one batch that ran (completed or not).
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchMetrics {
    /// Zero-based batch index.
    pub index: usize,
    /// Prompt tokens of every completion of the batch (0 where the backend did not count).
    pub input_tokens: u64,
    /// Completion tokens of every completion of the batch (0 where the backend did not count).
    pub output_tokens: u64,
    /// Wall clock of the batch's `consult`, in milliseconds.
    pub ms: u64,
}

/// The text report of a consolidated review.
///
/// magi-core's report over the consolidated outputs, its `Key Findings` limited to the findings
/// the gate did not filter and at their severity after the agreement weight; then the review's
/// own sections: batches and their failures, the gate's counts (and, once, what stopped it),
/// the models skipped after an HTTP 400, the findings of the main list with their gate verdict
/// (every cause, the deciding model, its lean and confidence, the models it skipped), the filtered ones listed apart, the
/// discarded and symbol-unverified findings, the skipped files and the other declarations of
/// the summary. The whole text is redacted as it is returned.
///
/// # Arguments
///
/// * `c` - The consolidated review.
/// * `s` - Its summary.
///
/// # Returns
///
/// The report. O(findings + summary).
pub fn render_text(c: &Consolidated, s: &ReviewSummary) -> String {
    let mut consensus = c.consensus.clone();
    consensus.findings = s
        .findings
        .iter()
        .filter_map(|(f, _)| key_finding(f))
        .collect();
    let unshown = s.findings.len().saturating_sub(consensus.findings.len());
    let mut out = ReportFormatter::new().format_report(&c.agents, &consensus);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    // Writing into a `String` cannot fail.
    let _ = write_review_section(&mut out, s, unshown);
    redacted(&out)
}

/// The finding as magi-core's `Key Findings` lists it: the review's severity (after the
/// agreement weight), its titles, its first detail and its sources.
///
/// Built through `Deserialize`, the only constructor magi-core offers; `None` when that shape
/// stops deserializing, which [`render_text`] declares instead of dropping the finding silently.
fn key_finding(f: &ConsolidatedFinding) -> Option<DedupFinding> {
    serde_json::from_value(json!({
        "severity": f.severity,
        "title": f.titles.join(TITLE_JOINER),
        "detail": f.details.first().cloned().unwrap_or_default(),
        "sources": f.sources,
        "file": f.file,
        "line": f.line,
        "category": f.categories.first().cloned().unwrap_or_default(),
    }))
    .ok()
}

/// Writes the review's own sections of [`render_text`].
fn write_review_section(out: &mut String, s: &ReviewSummary, unshown: usize) -> std::fmt::Result {
    writeln!(out, "\n## Review")?;
    writeln!(
        out,
        "batches: {} of {} completed",
        s.batches_completed, s.batches_planned
    )?;
    for (k, why) in &s.failed_batches {
        writeln!(out, "batch {} did not complete: {why}", shown(*k))?;
    }
    if let Some(cause) = &s.gate_stopped {
        writeln!(out, "gate stopped: {cause}")?;
    }
    writeln!(out, "gate states: {}", counts(&s.gate_states))?;
    if !s.gate_causes.is_empty() {
        writeln!(out, "gate causes: {}", counts(&s.gate_causes))?;
    }
    if !s.per_model.is_empty() {
        writeln!(out, "decided by: {}", counts(&s.per_model))?;
    }
    let skips = rejected_skips(s);
    if !skips.is_empty() {
        writeln!(out, "rejected skips: {}", counts(&skips))?;
    }
    writeln!(out, "gate questions: {}", s.gate_questions)?;
    writeln!(out, "judged: {:.0} %", s.judged_fraction * 100.0)?;
    let (input, output, ms) = totals(s);
    writeln!(
        out,
        "tokens: {input} in, {output} out; batches {ms} ms; preparation {} ms",
        s.prep_ms
    )?;
    if !s.unmeasured_models.is_empty() {
        writeln!(
            out,
            "models without a measured window: {}",
            s.unmeasured_models.join(", ")
        )?;
    }
    if s.ceiling_below_configured {
        writeln!(
            out,
            "the --timeout leaves each batch a ceiling below [magi].agent_timeout_secs"
        )?;
    }
    if s.symbol_check_skipped {
        writeln!(
            out,
            "the symbol check was off: the lexical index hit its cap"
        )?;
    }
    if s.ungrouped_unlocated > 0 {
        writeln!(
            out,
            "{} unlocated findings were grouped without the consensus engine",
            s.ungrouped_unlocated
        )?;
    }
    if unshown > 0 {
        writeln!(
            out,
            "{unshown} findings could not be listed under Key Findings; they are listed below"
        )?;
    }
    writeln!(out, "\n## Findings")?;
    for (f, v) in &s.findings {
        write_finding(out, f, v)?;
    }
    if !s.filtered.is_empty() {
        writeln!(out, "\n## Filtered by the gate")?;
        for (f, v) in &s.filtered {
            write_finding(out, f, v)?;
        }
    }
    write_discarded(out, "Discarded", &s.discarded)?;
    write_discarded(
        out,
        "Critical with an unverified symbol",
        &s.symbol_unverified,
    )?;
    if !s.skipped.is_empty() {
        writeln!(out, "\n## Skipped")?;
        for skip in &s.skipped {
            writeln!(out, "- {} ({})", skip.path, name(&skip.cause))?;
        }
    }
    if !s.no_extractor.is_empty() {
        writeln!(out, "\n## Reviewed without language context")?;
        for path in &s.no_extractor {
            writeln!(out, "- {path}")?;
        }
    }
    Ok(())
}

/// Writes one finding with its gate verdict: severity, location, titles, marks, then its details.
fn write_finding(out: &mut String, f: &ConsolidatedFinding, v: &GateVerdict) -> std::fmt::Result {
    write!(
        out,
        "- [{}] {} {}",
        name(&f.severity),
        location(f.file.as_deref(), f.line),
        f.titles.join(TITLE_JOINER)
    )?;
    if !f.marks.is_empty() {
        let marks: Vec<String> = f.marks.iter().map(name).collect();
        write!(out, " ({})", marks.join(", "))?;
    }
    write!(out, "; gate: {}", name(&v.state))?;
    // MAGI FIX (S6 loop 3): the text says of each verdict what its JSON says: every cause, the
    // answer that set it (model, lean, confidence) and the models skipped after an HTTP 400.
    if !v.causes.is_empty() {
        write!(out, " ({})", v.causes.join("; "))?;
    }
    if let Some(model) = &v.model {
        write!(out, " by {model}")?;
    }
    if let Some(lean) = &v.lean {
        write!(out, ", lean {lean}")?;
    }
    if let Some(confidence) = v.confidence {
        write!(out, ", confidence {confidence:.2}")?;
    }
    if !v.skipped.is_empty() {
        write!(out, "; skipped after HTTP 400: {}", v.skipped.join(", "))?;
    }
    writeln!(out)?;
    for detail in &f.details {
        writeln!(out, "  {detail}")?;
    }
    Ok(())
}

/// Writes one list of discarded findings under `title`, nothing when it is empty.
fn write_discarded(out: &mut String, title: &str, list: &[Discarded]) -> std::fmt::Result {
    if list.is_empty() {
        return Ok(());
    }
    writeln!(out, "\n## {title}")?;
    for d in list {
        writeln!(
            out,
            "- {} [{}] {} (batch {}, {})",
            location(d.finding.file.as_deref(), d.finding.line),
            name(&d.cause),
            d.finding.title,
            shown(d.batch),
            name(&d.agent)
        )?;
    }
    Ok(())
}

/// The JSON form of a summary.
///
/// Every field of [`ReviewSummary`], with `findings` and `filtered` as
/// `[{<fields of the finding>, "gate": <verdict>}]`, plus `batch_totals`
/// (`{input_tokens, output_tokens, ms}` summed over `batches`) and `rejected_skips`
/// (`{model: n}`, the models skipped after an HTTP 400). Every string is redacted.
///
/// # Arguments
///
/// * `s` - The summary.
///
/// # Returns
///
/// The JSON object. O(size of the summary).
pub fn summary_json(s: &ReviewSummary) -> Value {
    // Serializing these types cannot fail: every map key is a string.
    let mut v = serde_json::to_value(s).unwrap_or_default();
    if let Value::Object(fields) = &mut v {
        fields.insert("findings".into(), with_gate(&s.findings));
        fields.insert("filtered".into(), with_gate(&s.filtered));
        let (input, output, ms) = totals(s);
        fields.insert(
            "batch_totals".into(),
            json!({"input_tokens": input, "output_tokens": output, "ms": ms}),
        );
        fields.insert("rejected_skips".into(), json!(rejected_skips(s)));
        number_batches_from_one(fields);
    }
    redact_strings(&mut v);
    v
}

/// The paid batch reports, when consolidation failed (M-9).
///
/// Each report as its `consult` produced it, magi-core's report under `## [batch k]`, with `k`
/// its 1-based batch number. The whole text is redacted, since the reports are held raw.
///
/// # Arguments
///
/// * `reports` - The reports of the completed batches, each with its 0-based batch index, in
///   batch order.
///
/// # Returns
///
/// The text.
pub fn render_batch_reports(reports: &[(usize, MagiReport)]) -> String {
    let formatter = ReportFormatter::new();
    let mut out = String::new();
    for (batch, report) in reports {
        // Writing into a `String` cannot fail.
        let _ = writeln!(out, "## [batch {}]", shown(*batch));
        out.push_str(&formatter.format_report(&report.agents, &report.consensus));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    redacted(&out)
}

/// The 1-based number a reader sees for a 0-based batch index: the one base every output uses,
/// the same as the `[batch k]` labels of the consolidated summaries.
fn shown(batch: usize) -> usize {
    batch.saturating_add(1)
}

/// Rewrites every batch index of the JSON form 1-based ([`shown`]): `failed_batches[][0]`,
/// `batches[].index`, `discarded[].batch`, `symbol_unverified[].batch` and the `batches` of
/// each finding of `findings` and `filtered`.
fn number_batches_from_one(fields: &mut Map<String, Value>) {
    let renumber = |value: &mut Value| {
        if let Some(n) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
            *value = json!(shown(n));
        }
    };
    for (list, key) in [
        ("batches", "index"),
        ("discarded", "batch"),
        ("symbol_unverified", "batch"),
    ] {
        if let Some(Value::Array(items)) = fields.get_mut(list) {
            items
                .iter_mut()
                .filter_map(|item| item.get_mut(key))
                .for_each(renumber);
        }
    }
    if let Some(Value::Array(items)) = fields.get_mut("failed_batches") {
        items
            .iter_mut()
            .filter_map(|pair| pair.get_mut(0))
            .for_each(renumber);
    }
    for list in ["findings", "filtered"] {
        if let Some(Value::Array(items)) = fields.get_mut(list) {
            for batches in items.iter_mut().filter_map(|f| f.get_mut("batches")) {
                if let Value::Array(numbers) = batches {
                    numbers.iter_mut().for_each(renumber);
                }
            }
        }
    }
}

/// The findings with their verdict under [`GATE_KEY`].
fn with_gate(list: &[(ConsolidatedFinding, GateVerdict)]) -> Value {
    Value::Array(
        list.iter()
            .map(|(finding, verdict)| {
                let mut entry = match serde_json::to_value(finding) {
                    Ok(Value::Object(fields)) => fields,
                    _ => Map::new(),
                };
                entry.insert(
                    GATE_KEY.into(),
                    serde_json::to_value(verdict).unwrap_or_default(),
                );
                Value::Object(entry)
            })
            .collect(),
    )
}

/// The models skipped after an HTTP 400, each with the number of findings it was skipped for,
/// over the main list and the filtered one. O(findings x models).
fn rejected_skips(s: &ReviewSummary) -> BTreeMap<String, usize> {
    let mut skips = BTreeMap::new();
    for (_, verdict) in s.findings.iter().chain(&s.filtered) {
        for model in &verdict.skipped {
            *skips.entry(model.clone()).or_insert(0) += 1;
        }
    }
    skips
}

/// Input tokens, output tokens and milliseconds summed over the batches, saturating.
fn totals(s: &ReviewSummary) -> (u64, u64, u64) {
    s.batches.iter().fold((0, 0, 0), |(i, o, ms), b| {
        (
            i.saturating_add(b.input_tokens),
            o.saturating_add(b.output_tokens),
            ms.saturating_add(b.ms),
        )
    })
}

/// `k1 n1, k2 n2`, in key order.
fn counts(map: &BTreeMap<String, usize>) -> String {
    let parts: Vec<String> = map.iter().map(|(k, n)| format!("{k} {n}")).collect();
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

/// `file:line`, `file`, or `(no location)`.
fn location(file: Option<&str>, line: Option<u32>) -> String {
    match (file, line) {
        (Some(f), Some(l)) => format!("{f}:{l}"),
        (Some(f), None) => f.to_string(),
        (None, _) => "(no location)".to_string(),
    }
}

/// The serialized name of a unit enum value (`warning`, `kept`, `file not reviewed`, ...), or its
/// `Debug` form if it does not serialize to a string: the one enum-to-text helper of the review.
pub(crate) fn name<T: serde::Serialize + std::fmt::Debug>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(Value::String(text)) => text,
        _ => format!("{value:?}"),
    }
}

/// `text` with every credential in an embedded URL redacted.
fn redacted(text: &str) -> String {
    redact_foreign_text(text).as_str().to_string()
}

/// Redacts every string of `value` in place (keys are field names, model names, states and causes). O(bytes).
fn redact_strings(value: &mut Value) {
    match value {
        Value::String(text) => *text = redacted(text),
        Value::Array(items) => items.iter_mut().for_each(redact_strings),
        Value::Object(fields) => fields.values_mut().for_each(redact_strings),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
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
            causes: vec![],
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
            "gate_questions",
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

    #[test]
    fn every_batch_index_of_the_json_form_is_one_based() {
        // [MAGI S6 loop 4, C2] the renumbering patches the serialized JSON by field name: a renamed
        // field would leave its indices 0-based while the text form stays 1-based.
        let finding = |batches: Vec<usize>| ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches,
        };
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
        };
        let discarded = |cause| Discarded {
            agent: AgentName::Melchior,
            batch: 0,
            finding: magi_core::schema::Finding::new(Severity::Critical, "t", "d"),
            cause,
            symbol: None,
        };
        let mut s = ReviewSummary::default();
        s.findings.push((finding(vec![0]), verdict.clone()));
        s.filtered.push((finding(vec![1]), verdict));
        s.failed_batches.push((0, "why".into()));
        s.batches.push(BatchMetrics {
            index: 0,
            input_tokens: 0,
            output_tokens: 0,
            ms: 0,
        });
        s.discarded.push(discarded(DiscardCause::LineOutOfRange));
        s.symbol_unverified
            .push(discarded(DiscardCause::UnknownSymbol));
        let v = summary_json(&s);
        assert_eq!(v["findings"][0]["batches"][0], 1, "{v}");
        assert_eq!(v["filtered"][0]["batches"][0], 2, "{v}");
        assert_eq!(v["failed_batches"][0][0], 1, "{v}");
        assert_eq!(v["batches"][0]["index"], 1, "{v}");
        assert_eq!(v["discarded"][0]["batch"], 1, "{v}");
        assert_eq!(v["symbol_unverified"][0]["batch"], 1, "{v}");
    }

    #[tokio::test]
    async fn the_paid_batch_reports_redact_a_credential_in_a_finding_title() {
        // magi-core's report prints titles, never details: a credential in a detail cannot show
        // that this backstop redacts; one in a title can.
        let f = serde_json::json!([{"severity":"warning","title":"see https://user:hunter2@example.invalid/x","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let text = render_batch_reports(&[(0, report)]);
        assert!(text.contains("## [batch 1]"), "{text}");
        assert!(
            text.contains("example.invalid"),
            "the title is rendered: {text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
    }

    #[tokio::test]
    async fn the_text_report_shows_what_the_json_says_of_each_gate_verdict() {
        // [MAGI §6 MS2 C loop 3] parity with the JSON: the deciding model, its lean and
        // confidence, every cause, the models skipped after a 400 and their count.
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        let verdict = GateVerdict {
            state: GateState::LowConfidence,
            cause: Some("critical never filtered"),
            causes: vec!["critical never filtered", "not every claim judged"],
            model: Some("clef-flash".into()),
            lean: Some("false_positive".into()),
            confidence: Some(0.625),
            skipped: vec!["clef:cloud".into()],
        };
        let mut s = ReviewSummary::default();
        s.findings.push((c.findings[0].clone(), verdict));
        let text = render_text(&c, &s);
        for needle in [
            "gate: low_confidence (critical never filtered; not every claim judged)",
            "by clef-flash, lean false_positive, confidence 0.62",
            "skipped after HTTP 400: clef:cloud",
            "rejected skips: clef:cloud 1",
        ] {
            assert!(text.contains(needle), "missing `{needle}` in:\n{text}");
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
