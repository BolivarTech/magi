// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! What a review reports (REQ-RV13): the summary of every stage, its JSON form and its text form.
//!
//! Both forms pass every string through [`redact_foreign_text`] and then the RV22 detector
//! ([`mask_secrets`]) as they are written: the summary holds model text (files, titles, details,
//! failure messages) and this is the last point before it leaves the process, so the redaction
//! here is a backstop over the one consolidation already applied.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use magi_core::consensus::{ConsensusResult, DedupFinding};
use magi_core::reporting::{MagiReport, ReportFormatter};
use magi_core::schema::AgentOutput;
use serde_json::{json, Map, Value};

use super::consolidate::{Consolidated, ConsolidatedFinding};
use super::error::MAX_OMISSIONS_SHOWN;
use super::filter::{Masked, Skipped};
use super::gate::GateVerdict;
use super::rag::ContextKind;
use super::secrets::{escape_controls, mask_secrets};
use super::validate::Discarded;
use crate::redact::redact_foreign_text;

/// The key under which a finding of the JSON form carries its gate verdict.
const GATE_KEY: &str = "gate";
/// The JSON key of the skipped files (the [`ReviewSummary::skipped`] field).
const SKIPPED_KEY: &str = "skipped";
/// The JSON key that declares how many skipped files are not listed.
const SKIPPED_UNLISTED_KEY: &str = "skipped_unlisted";
/// The JSON key of the files reviewed with their secrets masked ([`ReviewSummary::masked`]).
const MASKED_KEY: &str = "masked";
/// The JSON key that declares how many of those files are not listed.
const MASKED_UNLISTED_KEY: &str = "masked_unlisted";
/// The JSON key of the context left out of a batch ([`ReviewSummary::trimmed`]).
const TRIMMED_KEY: &str = "trimmed";
/// The JSON key that declares how many of those entries are not listed.
const TRIMMED_UNLISTED_KEY: &str = "trimmed_unlisted";
/// The JSON key of the files reviewed without an extractor ([`ReviewSummary::no_extractor`]).
const NO_EXTRACTOR_KEY: &str = "no_extractor";
/// The JSON key that declares how many of those files are not listed.
const NO_EXTRACTOR_UNLISTED_KEY: &str = "no_extractor_unlisted";
/// Every character a terminal or a log viewer may take as the start of a new line.
const LINE_BREAKS: [char; 7] = [
    '\n', '\r', '\u{0B}', '\u{0C}', '\u{85}', '\u{2028}', '\u{2029}',
];
/// What starts each continuation line of multi-line model prose in the text form.
const CONTINUATION_INDENT: &str = "\n  ";
/// The character that opens a section heading of the text report (`## Skipped`).
const SECTION_MARK: char = '#';
/// How the titles of a merged finding are joined on one line.
const TITLE_JOINER: &str = " / ";
/// The key under which a value of the JSON form that could not be serialized declares why.
const SERIALIZATION_ERROR_KEY: &str = "serialization_error";
/// The run-log target of the report.
const LOG_TARGET: &str = "magi_rs::review";

/// What every stage of a review declared.
///
/// Batch indices are 0-based here; the text and JSON forms show them 1-based, the
/// same base as the `[batch k]` labels of the consolidated summaries. In the JSON form the
/// renumbering is part of each index field's type-level `Serialize` (`serialize_batch` and its
/// siblings, declared on the field), so no pass over the serialized layout names fields.
///
/// Flat on purpose: each field is a top-level key of the `code_review` JSON object the README
/// documents and the smoke harness reads (`without_memory`, `context_withheld`, `trimmed`…), so
/// grouping fields into sub-structs would move published keys; the exhaustive `Default` keeps a
/// new field from being forgotten. A regrouping is a schema change, decided as one.
#[derive(Debug, Clone, serde::Serialize)]
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
    /// Both forms write it clamped to `[0, 1]`, and a non-finite value as 0.
    #[serde(serialize_with = "serialize_fraction")]
    pub judged_fraction: f64,
    /// Findings removed by the deterministic validation, with their cause.
    pub discarded: Vec<Discarded>,
    /// Files left out of the review, with their cause (both forms list at most
    /// [`MAX_OMISSIONS_SHOWN`] and declare the rest).
    pub skipped: Vec<Skipped>,
    /// Files reviewed with their secrets masked (REQ-RV22), listed like `skipped`. The two lists
    /// declare different things, so a path can be in both: the file reviewed masked (here) and a
    /// context item of the same path the pipeline withheld (`skipped`, cause `secret`; a unit
    /// holding a key block, a documentation source). Neither entry contradicts the other.
    pub masked: Vec<Masked>,
    /// Reviewed files with no language extractor (listed like `skipped`).
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
    /// Batches that did not complete, with why; each index serialized 1-based.
    #[serde(serialize_with = "serialize_failed_batches")]
    pub failed_batches: Vec<(usize, String)>,
    /// The cause that stopped the gate, declared once.
    pub gate_stopped: Option<String>,
    /// Whether the symbol check was off because the lexical index hit its cap.
    pub symbol_check_skipped: bool,
    /// Critical findings whose only failed check was the symbol one; they reached the gate.
    pub symbol_unverified: Vec<Discarded>,
    /// Unlocated findings grouped without the consensus engine.
    pub ungrouped_unlocated: usize,
    /// Retrieved context left out of a batch, in priority order, each label as
    /// [`crate::review::secrets::display_path`] shows it (REQ-RV07, REQ-RV13).
    pub trimmed: Vec<TrimmedItem>,
    /// Whether documentation and history were left out because no embedder answered. Only that
    /// cause sets it: an index that could not be built for another reason (git failing to list
    /// the revision, an unreadable source) leaves it `false` and adds `index unavailable` to
    /// [`Self::trimmed`] once (RV21/U29;
    /// `a_git_failure_is_declared_once_as_an_unavailable_index_never_as_no_embedder`).
    pub without_embedder: bool,
    /// Whether documentation and history were left out because the run opened no database
    /// (`--no-memory`).
    pub without_memory: bool,
    /// Context items left out because the secret detector matched them (REQ-RV22), as the
    /// pipeline counts them (`prepare_contexts`): an item holding a private-key block (RV22's
    /// exception), one whose path quotes a secret (a path cannot be masked and still name its
    /// file), one in which masking left a match, plus those the context source withheld itself.
    /// An item holding a one-line secret is not counted here, even when its mask is longer than
    /// what the budget charged: it is sent masked, and its file (the item's path, as
    /// [`display_path`](crate::review::secrets::display_path) shows it) is declared in
    /// [`Self::masked`] (the pipeline's `declare_masked` adds a documentation or history source
    /// there as it does a reviewed file, so no masked egress goes undeclared); the batch
    /// pays for the growth with trimmed optional context, listed in [`Self::trimmed`] (RV22
    /// amended; pipeline tests `a_context_item_quoting_a_one_line_secret_reaches_the_prompt_masked`
    /// and `the_growth_of_a_mask_is_paid_by_trimming_optional_context_never_by_omitting_the_item`).
    /// This module only writes the count it is given.
    pub context_withheld: usize,
    /// Containing units (RV07(1), mandatory) the context source promised at planning and did not
    /// bring when it assembled the batch's context, by label as
    /// [`display_path`](crate::review::secrets::display_path) shows it. A unit the mage was meant
    /// to see and did not is a different fact from optional context trimmed to fit the budget
    /// ([`Self::trimmed`]), so it is declared on its own.
    pub undelivered_units: Vec<String>,
    /// `(file, line)` of the findings whose line names no line of the reviewed revision: one
    /// whose number can only be a removed line (read on the old side only). A number that is
    /// both a removed and a live line (an in-place edit, or a removed line whose old number
    /// falls in a new-side range) is not here: it is anchored on the new side, as the gate reads
    /// it. The review history does not record these (RV08). Internal: never serialized. Each key
    /// is the consolidated finding's own `file`, inserted from that finding by the pipeline, so
    /// the spellings agree by construction; the match is exact (`off_revision_matches_exact_file`).
    /// A spelling that ever differed would keep the finding among the candidates (it is never
    /// dropped): the git check of RV08 then judges it, the safe direction (declared).
    #[serde(skip)]
    pub off_revision: BTreeMap<String, BTreeSet<u32>>,
}

impl ReviewSummary {
    /// The main list's findings the review history may record (RV08): every one except those
    /// in [`Self::off_revision`], whose line is a removed line. History anchors an
    /// entry to the reviewed revision's line numbers, so such a finding would be checked against
    /// an unrelated line. Whether a finding is recorded at all is still decided by
    /// `history::entries_from_review` (its gate state, its location).
    ///
    /// # Returns
    ///
    /// A copy of those findings, in order: owned on purpose, since its one caller hands them to
    /// the post-report work, which runs on a blocking thread after this summary was rendered and
    /// dropped, so indices or borrows would not outlive it. O(findings · log off_revision): the lookup borrows the path (a
    /// map of path to lines), so no key is built per finding.
    // MAGI FIX (Loop1 r1, M2): removed-line findings were anchored with their old-file number.
    pub fn anchored_findings(&self) -> Vec<(ConsolidatedFinding, GateVerdict)> {
        self.findings
            .iter()
            .filter(|(f, _)| match (&f.file, f.line) {
                (Some(file), Some(line)) => !self
                    .off_revision
                    .get(file)
                    .is_some_and(|lines| lines.contains(&line)),
                _ => true,
            })
            .cloned()
            .collect()
    }
}

/// One piece of retrieved context a batch left out, and what it was.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TrimmedItem {
    /// What the item was.
    pub kind: ContextKind,
    /// Its label, as it may leave the process.
    pub label: String,
}

// MAGI FIX (S6 loop 8, C2-c5): written out so `judged_fraction` starts at its documented 1.0
// (nothing reached the gate, so nothing was left unjudged); a derived `Default` said 0.0.
// A new field cannot be forgotten here: the literal below has no `..` rest, so the compiler
// refuses it until the field is listed.
impl Default for ReviewSummary {
    /// An empty summary: no finding, no batch, and `judged_fraction` 1.0.
    fn default() -> Self {
        Self {
            gate_states: BTreeMap::new(),
            gate_causes: BTreeMap::new(),
            per_model: BTreeMap::new(),
            gate_questions: 0,
            judged_fraction: 1.0,
            discarded: Vec::new(),
            skipped: Vec::new(),
            masked: Vec::new(),
            no_extractor: Vec::new(),
            unmeasured_models: Vec::new(),
            prep_ms: 0,
            batches: Vec::new(),
            batches_completed: 0,
            batches_planned: 0,
            findings: Vec::new(),
            filtered: Vec::new(),
            ceiling_below_configured: false,
            failed_batches: Vec::new(),
            gate_stopped: None,
            symbol_check_skipped: false,
            symbol_unverified: Vec::new(),
            ungrouped_unlocated: 0,
            trimmed: Vec::new(),
            without_embedder: false,
            without_memory: false,
            context_withheld: 0,
            undelivered_units: Vec::new(),
            off_revision: BTreeMap::new(),
        }
    }
}

/// The metrics of one batch that ran (completed or not).
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchMetrics {
    /// Zero-based batch index, serialized 1-based.
    #[serde(serialize_with = "serialize_batch")]
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
/// (every cause and the HTTP status of a 413/422 that ended its chain, the deciding model, its
/// lean and confidence, the models it skipped), the filtered ones listed apart, the
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
        .filter_map(|(f, _)| key_finding(c.consensus.findings.first(), f))
        .collect();
    let unshown = s.findings.len().saturating_sub(consensus.findings.len());
    let (agents, consensus) = line_safe(&c.agents, &consensus);
    let mut out = ReportFormatter::new().format_report(&agents, &consensus);
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
/// magi-core offers no constructor for a `#[non_exhaustive]` [`DedupFinding`] (E4), so the
/// value is a clone of `template`, a finding magi-core itself produced for this review, with
/// every field assigned by name. A field magi-core renames or removes is a compile error here,
/// never a runtime drop; a field it adds keeps the template's value, which magi-core made. A
/// compile-time constructor proper (or a `trybuild` test of it) would need an API magi-core does
/// not have, or a new dependency for an assertion this assignment already makes at compile time.
/// `None` without a template, which [`render_text`] declares instead of dropping the finding.
// MAGI FIX (S6 loop 5): field assignment on magi-core's own value, not a JSON round trip.
// MAGI FIX (S6 loop 8, C2-c1): a field magi-core adds is no longer silent: the test
// `a_key_finding_overwrites_every_field_of_its_template` pins the serialized field set.
fn key_finding(template: Option<&DedupFinding>, f: &ConsolidatedFinding) -> Option<DedupFinding> {
    let mut key = template?.clone();
    key.severity = f.severity;
    // MAGI FIX (S6 loop 10, C2-m2): raw here; `line_safe` escapes every Key Finding title once,
    // so the output no longer depends on `escape_controls` being idempotent.
    key.title = f.titles.join(TITLE_JOINER);
    key.detail = f.details.first().cloned().unwrap_or_default();
    key.sources.clone_from(&f.sources);
    key.file.clone_from(&f.file);
    key.line = f.line;
    key.category = f.categories.first().cloned().unwrap_or_default();
    key.id = None;
    Some(key)
}

/// Writes the review's own sections of [`render_text`].
fn write_review_section(out: &mut String, s: &ReviewSummary, unshown: usize) -> std::fmt::Result {
    writeln!(out, "\n## Review")?;
    writeln!(
        out,
        "batches: {} of {} completed",
        s.batches_completed, s.batches_planned
    )?;
    out.push_str(&failed_batch_text(s));
    if let Some(cause) = &s.gate_stopped {
        writeln!(out, "gate stopped: {}", escape_controls(cause))?;
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
    // MAGI FIX (S6 loop 12, C2-m4 / C2-b2): truncated, never rounded: `{:.0}` printed 299 of
    // 300 as `100 %`.
    let tenths = (shown_fraction(s.judged_fraction) * PERCENT_TENTHS).floor() / TENTHS_PER_UNIT;
    writeln!(out, "judged: {tenths:.1} %")?;
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
            joined(&s.unmeasured_models)
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
    if s.without_memory {
        writeln!(
            out,
            "documentation and history were left out: the run opened no database"
        )?;
    } else if s.without_embedder {
        writeln!(
            out,
            "documentation and history were left out: no embedder answered"
        )?;
    }
    if s.context_withheld > 0 {
        writeln!(
            out,
            "{} context items were withheld for a secret",
            s.context_withheld
        )?;
    }
    if !s.undelivered_units.is_empty() {
        let shown: Vec<String> = s
            .undelivered_units
            .iter()
            .take(MAX_OMISSIONS_SHOWN)
            .map(|u| listed(u))
            .collect();
        writeln!(
            out,
            "{} containing units were not delivered: {}",
            s.undelivered_units.len(),
            shown.join(", ")
        )?;
    }
    if !s.trimmed.is_empty() {
        // MAGI FIX (MS3 §6 loop2): capped like the other omission lists; it grows with the batches.
        let mut trimmed: Vec<String> = s
            .trimmed
            .iter()
            .take(MAX_OMISSIONS_SHOWN)
            .map(|t| format!("{} {}", name(&t.kind), listed(&t.label)))
            .collect();
        match unlisted(s.trimmed.len()) {
            0 => {}
            rest => trimmed.push(format!("and {rest} more")),
        }
        writeln!(out, "context left out: {}", trimmed.join("; "))?;
    }
    if unshown > 0 {
        writeln!(
            out,
            "{unshown} findings could not be listed under Key Findings; they are listed below"
        )?;
    }
    // `## Findings` is written even when the main list is empty, on purpose: it is the stable
    // anchor of the main list, the one the `unshown` line above points at ("listed below"),
    // found whatever the gate decided; the gate-states line already says the list is empty.
    // Every other section is written only when it has entries. Pinned by
    // `the_findings_heading_is_written_even_when_the_main_list_is_empty`.
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
        for skip in s.skipped.iter().take(MAX_OMISSIONS_SHOWN) {
            writeln!(out, "- {} ({})", listed(&skip.path), name(&skip.cause))?;
        }
        write_unlisted(out, &s.skipped)?;
    }
    if !s.masked.is_empty() {
        writeln!(out, "\n## Reviewed with secrets masked")?;
        for m in s.masked.iter().take(MAX_OMISSIONS_SHOWN) {
            writeln!(
                out,
                "- {} ({}: {})",
                listed(&m.path),
                name(&m.cause),
                // MAGI FIX (MS3 §6 loop2): line-safe like the path beside it.
                // MAGI FIX (MS3 §6 loop5): and through the same backstop, as the JSON form does.
                listed(m.pattern)
            )?;
        }
        write_unlisted(out, &s.masked)?;
    }
    if !s.no_extractor.is_empty() {
        writeln!(out, "\n## Reviewed without language context")?;
        for path in s.no_extractor.iter().take(MAX_OMISSIONS_SHOWN) {
            writeln!(out, "- {}", listed(path))?;
        }
        write_unlisted(out, &s.no_extractor)?;
    }
    Ok(())
}

/// How many entries of a list of `len` lie past [`MAX_OMISSIONS_SHOWN`] and are not listed.
/// O(1).
fn unlisted(len: usize) -> usize {
    len.saturating_sub(MAX_OMISSIONS_SHOWN)
}

/// Writes `- ... and N more` for the [`unlisted`] entries of `list`, nothing when there are none.
fn write_unlisted<T>(out: &mut String, list: &[T]) -> std::fmt::Result {
    match unlisted(list.len()) {
        0 => Ok(()),
        rest => writeln!(out, "- ... and {rest} more"),
    }
}

/// Writes one finding with its gate verdict: severity, location, titles, marks, categories, then
/// its details.
///
/// Model text never starts a line of its own: the location and the titles go through
/// [`escape_controls`], so a line break (`\n`, `\r`, U+2028, ...) is written as an escape, and
/// each line of a detail is indented with every control left in it escaped the same way.
// MAGI FIX (S6 loop 9, C2-m1 / C2-c1): titles and locations get the line safety details had.
fn write_finding(out: &mut String, f: &ConsolidatedFinding, v: &GateVerdict) -> std::fmt::Result {
    write!(
        out,
        "- [{}] {} {}",
        name(&f.severity),
        location(f.file.as_deref(), f.line),
        escape_controls(&f.titles.join(TITLE_JOINER))
    )?;
    if !f.marks.is_empty() {
        let marks: Vec<String> = f.marks.iter().map(name).collect();
        write!(out, " ({})", marks.join(", "))?;
    }
    // MAGI FIX (S6 loop 4): every category a merged finding conserves (RV-S18), as the JSON has.
    // MAGI FIX (S6 loop 12, C2-m1 / C2-b1 / C2-c1): `Category` is a closed enum today, so its
    // names are safe literals; escaped anyway, since a foreign `#[non_exhaustive]` enum may gain
    // a variant that carries model text (`FinishReason::Other(String)` did).
    let categories: Vec<String> = f.categories.iter().map(name).collect();
    write!(out, " [{}]", joined(&categories))?;
    write!(out, "; gate: {}", name(&v.state))?;
    // MAGI FIX (S6 loop 3): the text says of each verdict what its JSON says: every cause, the
    // answer that set it (model, lean, confidence) and the models skipped after an HTTP 400.
    // MAGI FIX (S6 loop 5): a 413/422 that ended the chain names its HTTP status after the causes.
    // The causes are written without `escape_controls`, unlike model text: each is a
    // `&'static str` of the gate's closed vocabulary (`CAUSE_*` in `gate.rs`), one line of
    // printable ASCII, so none can carry provider or transport text. Pinned by
    // `every_gate_cause_is_a_one_line_literal_written_as_is`: a cause that ever carries foreign
    // text has to change the field's type, which turns that test red.
    let mut causes: Vec<String> = v.causes.iter().map(|c| (*c).to_string()).collect();
    causes.extend(v.status.map(|status| format!("HTTP {status}")));
    if !causes.is_empty() {
        write!(out, " ({})", causes.join("; "))?;
    }
    // MAGI FIX (S6 loop 10, C2-c1): model names and the lean are line-safe like model text.
    if let Some(model) = &v.model {
        write!(out, " by {}", escape_controls(model))?;
    }
    if let Some(lean) = &v.lean {
        write!(out, ", lean {}", escape_controls(lean))?;
    }
    if let Some(confidence) = v.confidence {
        write!(out, ", confidence {confidence:.2}")?;
    }
    if !v.skipped.is_empty() {
        write!(out, "; skipped after HTTP 400: {}", joined(&v.skipped))?;
    }
    writeln!(out)?;
    // MAGI FIX (S6 loop 4): every line of a detail is indented, so none reads as a report line.
    for line in f.details.iter().flat_map(|detail| detail.lines()) {
        writeln!(out, "  {}", escape_controls(line))?;
    }
    Ok(())
}

/// Writes one list of discarded findings under `title`, nothing when it is empty.
///
/// The file and the title failed validation, so they are the model's raw claims: both go through
/// [`escape_controls`] and each entry stays on one line.
// MAGI FIX (S6 loop 9, C2-m1): the discarded list was written raw.
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
            escape_controls(&d.finding.title),
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
/// (`{model: n}`, the models skipped after an HTTP 400). `skipped`, `masked`, `trimmed` and
/// `no_extractor` list at most [`MAX_OMISSIONS_SHOWN`] entries each, and `skipped_unlisted`,
/// `masked_unlisted`, `trimmed_unlisted` / `no_extractor_unlisted` declare how many more there
/// are (0 when none), as the text form's `and N more` does: the lists grow with the reviewed
/// tree and its batches, not with the review's result. Every string and every
/// object key is redacted, so a model name used as a key passes the RV22 detector too.
///
/// # Arguments
///
/// * `s` - The summary.
///
/// # Returns
///
/// The JSON object. O(size of the summary).
pub fn summary_json(s: &ReviewSummary) -> Value {
    let mut v = to_json(s, "summary");
    if let Value::Object(fields) = &mut v {
        fields.insert("findings".into(), with_gate(&s.findings));
        fields.insert("filtered".into(), with_gate(&s.filtered));
        let (input, output, ms) = totals(s);
        fields.insert(
            "batch_totals".into(),
            json!({"input_tokens": input, "output_tokens": output, "ms": ms}),
        );
        fields.insert("rejected_skips".into(), json!(rejected_skips(s)));
        // MAGI FIX (S6 loop 13, C2-c4 / C2-m3): the omission lists grew with the reviewed tree;
        // capped like the error path's, the rest declared.
        for (list, rest, len) in [
            (SKIPPED_KEY, SKIPPED_UNLISTED_KEY, s.skipped.len()),
            (MASKED_KEY, MASKED_UNLISTED_KEY, s.masked.len()),
            (TRIMMED_KEY, TRIMMED_UNLISTED_KEY, s.trimmed.len()),
            (
                NO_EXTRACTOR_KEY,
                NO_EXTRACTOR_UNLISTED_KEY,
                s.no_extractor.len(),
            ),
        ] {
            if let Some(Value::Array(items)) = fields.get_mut(list) {
                items.truncate(MAX_OMISSIONS_SHOWN);
            }
            fields.insert(rest.into(), json!(unlisted(len)));
        }
    }
    redact_strings(&mut v);
    v
}

/// One line per batch that did not complete, `batch k did not complete: <cause>` with `k` its
/// 1-based number, in the order they failed; empty when every batch completed. The text is
/// redacted, since the causes carry provider and transport text.
///
/// # Arguments
///
/// * `s` - The review's summary.
///
/// # Returns
///
/// The lines, each ending in a newline.
// MAGI FIX (S6 loop 7, D-b2): one renderer of the batch causes for the report and the failure.
pub fn failed_batch_lines(s: &ReviewSummary) -> String {
    redacted(&failed_batch_text(s))
}

/// The lines of [`failed_batch_lines`] before redaction, for [`render_text`], which redacts the
/// whole report once: one redaction at the boundary, so the text form no longer depends on the
/// redaction helpers being idempotent. O(causes).
// MAGI FIX (S6 loop 13, C2-b4 / C2-c6): the batch causes were redacted twice in the text form.
fn failed_batch_text(s: &ReviewSummary) -> String {
    let mut out = String::new();
    for (k, why) in &s.failed_batches {
        // Writing into a `String` cannot fail.
        let _ = writeln!(
            out,
            "batch {} did not complete: {}",
            shown(*k),
            escape_controls(why)
        );
    }
    out
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
        // MAGI FIX (S6 loop 9, C2-m1): no model text starts a line of its own here either.
        let (agents, consensus) = line_safe(&report.agents, &report.consensus);
        out.push_str(&formatter.format_report(&agents, &consensus));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    redacted(&out)
}

/// Copies of `agents` and `consensus` whose model text cannot start a line of the text form.
///
/// magi-core's formatter writes the Key Findings' titles, the dissent's summaries, the
/// conditions and the recommendations raw. Titles become one line ([`escape_controls`]); the
/// prose fields keep their line breaks as indented continuation lines ([`indented_prose`]), so
/// none starts at column 0 where it would read as a report line. The agents' summary, reasoning,
/// recommendation and findings, each dissent's reasoning and each Key Finding's detail, file and
/// id are not rendered by magi-core 4.2.0; they get the same treatment so a formatter that starts
/// rendering them inherits the safety. The
/// deprecated `majority_summary` is neither rendered nor touched (assigning it is a deprecation
/// warning, and the build denies warnings). Completeness is checked by construction, not by this
/// list: `every_free_text_field_of_magi_cores_report_types_is_line_safe` forges every leaf that
/// holds free text, so a field magi-core adds is red until it is handled here, and it renders
/// the result to prove `majority_summary` stays unrendered.
///
/// # Arguments
///
/// * `agents` - The agents' outputs.
/// * `consensus` - The consensus magi-core renders.
///
/// # Returns
///
/// The line-safe copies. O(bytes of the text fields). The copies are deliberate: the egress
/// boundary never edits the consolidated review in place, so a large review holds its text
/// twice while the report is rendered (bounded by the batches' replies, never by the input).
// MAGI FIX (S6 loop 9, C2 commit review): every model string the formatter may render.
fn line_safe(
    agents: &[AgentOutput],
    consensus: &ConsensusResult,
) -> (Vec<AgentOutput>, ConsensusResult) {
    let mut agents = agents.to_vec();
    for agent in &mut agents {
        agent.summary = indented_prose(&agent.summary);
        agent.reasoning = indented_prose(&agent.reasoning);
        agent.recommendation = indented_prose(&agent.recommendation);
        // MAGI FIX (S6 loop 13, C2-m1): each agent's own findings, unrendered by magi-core
        // 4.2.0, get the Key Findings' treatment.
        for finding in &mut agent.findings {
            finding.title = escape_controls(&finding.title);
            finding.detail = indented_prose(&finding.detail);
            finding.file = finding.file.as_deref().map(escape_controls);
        }
    }
    let mut consensus = consensus.clone();
    // MAGI FIX (S6 loop 13, C2-c2): the label and each Key Finding's id are magi-core's own text
    // (identity here); escaped so every free-text field the formatter receives is line-safe, as
    // `every_free_text_field_of_magi_cores_report_types_is_line_safe` checks by construction.
    consensus.consensus = escape_controls(&consensus.consensus);
    // MAGI FIX (S6 loop 10, C2-m1): the Key Finding detail, unrendered by magi-core 4.2.0, gets
    // the indented treatment of the agents' unrendered prose.
    for finding in &mut consensus.findings {
        finding.title = escape_controls(&finding.title);
        finding.detail = indented_prose(&finding.detail);
        // MAGI FIX (S6 loop 12, C2-m1 / C2-c1): the file is repository text (a Unix path may hold
        // a line break), unrendered by magi-core 4.2.0; one line, like the title.
        finding.file = finding.file.as_deref().map(escape_controls);
        finding.id = finding.id.as_deref().map(escape_controls);
    }
    for dissent in &mut consensus.dissent {
        dissent.summary = indented_prose(&dissent.summary);
        dissent.reasoning = indented_prose(&dissent.reasoning);
    }
    for condition in &mut consensus.conditions {
        condition.condition = indented_prose(&condition.condition);
    }
    for recommendation in consensus.recommendations.values_mut() {
        *recommendation = indented_prose(recommendation);
    }
    (agents, consensus)
}

/// `text` split on every [`LINE_BREAKS`] character (`\r\n` counts once), each line passed
/// through [`escape_controls`] and every line after the first indented
/// ([`CONTINUATION_INDENT`]). The first line is indented too when it opens with
/// [`SECTION_MARK`], so it cannot read as a section heading even where a formatter starts the
/// field at column 0. O(n).
// MAGI FIX (S6 loop 11, C2-m1 / C2-b2): a first line `## …` relied on magi-core writing a label
// before the field, an assumption about a third-party formatter that is no longer needed.
fn indented_prose(text: &str) -> String {
    let prose = text
        .replace("\r\n", "\n")
        .split(|c| LINE_BREAKS.contains(&c))
        .map(escape_controls)
        .collect::<Vec<_>>()
        .join(CONTINUATION_INDENT);
    if prose.trim_start().starts_with(SECTION_MARK) {
        format!("{}{prose}", CONTINUATION_INDENT.trim_start_matches('\n'))
    } else {
        prose
    }
}

/// The 1-based number a reader sees for a 0-based batch index: the one base every output uses,
/// the same as the `[batch k]` labels of the consolidated summaries.
fn shown(batch: usize) -> usize {
    batch.saturating_add(1)
}

/// Serializes a 0-based batch index 1-based ([`shown`]), for `#[serde(serialize_with)]` on every
/// batch-index field of the review's types.
///
/// # Errors
/// The serializer's.
// MAGI FIX (S6 loop 5): the renumbering lives on the fields, not in a pass over JSON paths.
pub(crate) fn serialize_batch<S: serde::Serializer>(
    batch: &usize,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_u64(u64::try_from(shown(*batch)).unwrap_or(u64::MAX))
}

/// Serializes a list of 0-based batch indices 1-based, as [`serialize_batch`] does each.
///
/// # Errors
/// The serializer's.
pub(crate) fn serialize_batches<S: serde::Serializer>(
    batches: &[usize],
    s: S,
) -> Result<S::Ok, S::Error> {
    s.collect_seq(batches.iter().map(|&b| shown(b)))
}

/// Tenths of a percent in a whole fraction, for the text form's truncated percentage.
const PERCENT_TENTHS: f64 = 1000.0;

/// Tenths in a unit, turning [`PERCENT_TENTHS`] back into a percentage with one decimal.
const TENTHS_PER_UNIT: f64 = 10.0;

/// `fraction` as the report writes it: clamped to `[0, 1]`, and 0 when it is not finite, so
/// coverage the review cannot vouch for is never claimed. O(1).
///
/// # Arguments
///
/// * `fraction` - A fraction computed elsewhere (the pipeline guards its own division).
///
/// # Returns
///
/// A finite value in `[0, 1]`.
// MAGI FIX (S6 loop 12, C2-m2 / C2-b4 / C2-c6): `serde_json` writes NaN and infinity as `null`,
// so a non-finite fraction reached a consumer as `null` and the text form as `NaN %`.
fn shown_fraction(fraction: f64) -> f64 {
    if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Serializes a fraction through [`shown_fraction`], for `#[serde(serialize_with)]`.
///
/// # Errors
/// The serializer's.
fn serialize_fraction<S: serde::Serializer>(fraction: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(shown_fraction(*fraction))
}

/// Serializes `(batch, why)` pairs with each batch 1-based, as [`serialize_batch`] does.
///
/// # Errors
/// The serializer's.
fn serialize_failed_batches<S: serde::Serializer>(
    failed: &[(usize, String)],
    s: S,
) -> Result<S::Ok, S::Error> {
    s.collect_seq(failed.iter().map(|(b, why)| (shown(*b), why)))
}

/// The findings with their verdict under [`GATE_KEY`].
fn with_gate(list: &[(ConsolidatedFinding, GateVerdict)]) -> Value {
    Value::Array(
        list.iter()
            .map(|(finding, verdict)| {
                // An object by construction (a struct with named fields); pinned by the tests
                // that read `filtered[0].file` and `findings[0].batches`.
                let mut entry = match to_json(finding, "finding") {
                    Value::Object(fields) => fields,
                    _ => Map::new(),
                };
                entry.insert(GATE_KEY.into(), to_json(verdict, "gate verdict"));
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

/// `k1 n1, k2 n2`, in key order, each key passed through [`escape_controls`]: a key is a model
/// name (configuration or pool, never validated against control characters) or a label.
// MAGI FIX (S6 loop 10, C2-c1): a model name with a line break forged a report line.
fn counts(map: &BTreeMap<String, usize>) -> String {
    let parts: Vec<String> = map
        .iter()
        .map(|(k, n)| format!("{} {n}", escape_controls(k)))
        .collect();
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

/// `names` joined with `, `, each passed through [`escape_controls`]. O(bytes).
fn joined(names: &[String]) -> String {
    names
        .iter()
        .map(|n| escape_controls(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `file:line`, `file`, or `(no location)`, the file passed through [`escape_controls`] (a
/// reported file is model text until validation says otherwise).
fn location(file: Option<&str>, line: Option<u32>) -> String {
    match (file, line) {
        (Some(f), Some(l)) => format!("{}:{l}", escape_controls(f)),
        (Some(f), None) => escape_controls(f),
        (None, _) => "(no location)".to_string(),
    }
}

/// The serialized name of a unit enum value (`warning`, `kept`, `file not reviewed`, ...), or its
/// `Debug` form if it does not serialize to a string: the one enum-to-text helper of the review.
///
/// The text is passed through [`escape_controls`] here, so no caller can write a label raw: the
/// review's labels are closed vocabularies today (identity for them), but a foreign
/// `#[non_exhaustive]` enum may gain a variant whose serialized text or `Debug` form carries
/// model text. O(length of the text).
///
/// # Arguments
///
/// * `value` - The value to name.
///
/// # Returns
///
/// Its line-safe name.
// MAGI FIX (S6 loop 13, C2-c1): four callers (`Discarded.agent` and `.cause`, a finding's
// severity, a verdict's state) wrote the name raw; the escape now lives in the one helper.
pub(crate) fn name<T: serde::Serialize + std::fmt::Debug>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(Value::String(text)) => escape_controls(&text),
        _ => escape_controls(&format!("{value:?}")),
    }
}

/// `value` as JSON, `what` naming it in a failure.
///
/// The review's types cannot fail to serialize (every map key is a string, no field has a
/// fallible `Serialize`). `serde_json` does not fail on a non-finite float either: it writes
/// `null`, which is why the review's one computed float goes through [`shown_fraction`]. Were a
/// value to fail, the failure is never silent: the value becomes
/// `{"serialization_error": "<what>: <error>"}` and one ERROR entry goes to the run log.
fn to_json<T: serde::Serialize>(value: &T, what: &str) -> Value {
    // MAGI FIX (S6 loop 5): never a silent `Null` that drops the whole envelope.
    serde_json::to_value(value).unwrap_or_else(|e| {
        let error = redacted(&format!("{what}: {e}"));
        tracing::event!(
            target: LOG_TARGET,
            tracing::Level::ERROR,
            error = error.as_str(),
            "a value of the review report could not be serialized"
        );
        json!({ SERIALIZATION_ERROR_KEY: error })
    })
}

/// `text` as the report may write it: every credential in an embedded URL redacted, then every
/// secret the RV22 detector names masked. A private-key block in model text is masked whole,
/// from its header through its END line or, with none, to the end of the text, so none of its
/// body lines (which match no pattern of their own) is written in either form
/// (`a_private_key_block_in_model_text_leaves_no_body_line_in_either_form`). A body line quoted
/// without its header cannot be recognised by any pattern: that limit is RV22's, stated in the
/// README.
// MAGI FIX (S6 loop 6): the report is an egress point of its own (REQ-RV22), so it runs the
// product's detector instead of relying on the caller's `egress_safe`.
fn redacted(text: &str) -> String {
    mask_secrets(redact_foreign_text(text).as_str())
}

/// A path or label of an omission list as the text form writes it: [`redacted`], then
/// [`escape_controls`], so the list is line-safe and holds no secret whatever its builder did
/// (`the_text_form_masks_a_raw_secret_in_its_paths_and_labels_as_it_writes_them`).
// MAGI FIX (MS3 §6 loop4): these lists relied on their builders' `display_path` alone.
fn listed(text: &str) -> String {
    escape_controls(&redacted(text))
}

/// `base`, or `base (2)`, `base (3)`, ... : the first spelling `taken` does not hold yet. Two keys
/// that redact alike (two model names holding the same secret pattern) stay two fields.
/// O(collisions · log size of `taken`). Public, not `pub(crate)`, because its consumer is in
/// another crate: the agent tool's own JSON masking (`tools::code_review`, in the binary) keeps
/// colliding keys the same way, and a `pub(crate)` item of this library would not compile there.
///
/// MAGI FIX (MS3 §6 loop14): there is no overflow to report. The counter climbs one at a time and
/// stops at the first free spelling, so it reaches at most `taken.len() + 2`; `taken` is held in
/// memory, so that is far below `usize::MAX` and the loop ends before a saturating add could
/// repeat a value. The counter is still `checked_add`: running out is a typed
/// `KeySuffixesExhausted` from `distinct_key_from` (both crate-private), which this function maps
/// to a fixed key.
///
/// # Arguments
/// * `taken` - The object the key goes into.
/// * `base` - The key as it may leave the process.
///
/// # Returns
/// A key `taken` does not hold.
///
/// # Examples
///
/// ```
/// use magi_rs::review::report::distinct_key;
///
/// let mut taken = serde_json::Map::new();
/// taken.insert("k".into(), serde_json::json!(1));
/// assert_eq!(distinct_key(&taken, "k".into()), "k (2)");
/// ```
pub fn distinct_key(taken: &serde_json::Map<String, Value>, base: String) -> String {
    // MAGI FIX (MS3 §6 loop4): `base` was cloned on every call; a free key (the usual case) is
    // now returned as it came, with no allocation.
    if !taken.contains_key(&base) {
        return base;
    }
    // MAGI FIX (MS3 §6 loop15): the counter is checked, not saturating; running out of
    // suffixes (unreachable in memory) is a typed failure that lands on a fixed key.
    distinct_key_from(taken, &base, 2)
        .unwrap_or_else(|KeySuffixesExhausted| KEY_SUFFIXES_EXHAUSTED_KEY.to_string())
}

/// The key a `base` whose suffixes ran out is written under (see [`KeySuffixesExhausted`]).
const KEY_SUFFIXES_EXHAUSTED_KEY: &str = "(too many colliding keys)";

/// Every ` (n)` suffix from the first one tried to `usize::MAX` was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeySuffixesExhausted;

/// The first free `base (n)` key with `n >= first`; O(taken) for a sane `first`.
///
/// # Arguments
/// * `taken` - The object the key goes into.
/// * `base` - The key without a suffix.
/// * `first` - The first suffix to try.
///
/// # Returns
/// The free key, or [`KeySuffixesExhausted`] when the counter would pass `usize::MAX`.
pub(crate) fn distinct_key_from(
    taken: &serde_json::Map<String, Value>,
    base: &str,
    first: usize,
) -> Result<String, KeySuffixesExhausted> {
    let mut n = first;
    loop {
        let key = format!("{base} ({n})");
        if !taken.contains_key(&key) {
            return Ok(key);
        }
        n = n.checked_add(1).ok_or(KeySuffixesExhausted)?;
    }
}

/// Redacts every string of `value` in place, object keys included: besides field names, the
/// keys are model names, states and causes, and a model name is operator text. Two keys that
/// redact to the same text are both kept, the later one suffixed ` (n)` with the first free `n`
/// from 2 ([`distinct_key`]), so no count is lost. O(bytes).
// MAGI FIX (S6 loop 11, C2-c1): keys were written as is, so a model name shaped like a key left
// the process unmasked as a key of `per_model` or `rejected_skips` (RV22).
fn redact_strings(value: &mut Value) {
    match value {
        Value::String(text) => *text = redacted(text),
        Value::Array(items) => items.iter_mut().for_each(redact_strings),
        Value::Object(fields) => {
            let mut safe = serde_json::Map::with_capacity(fields.len());
            for (key, mut item) in std::mem::take(fields) {
                redact_strings(&mut item);
                let key = distinct_key(&safe, redacted(&key));
                safe.insert(key, item);
            }
            *fields = safe;
        }
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
            status: None,
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
            "masked",
            "masked_unlisted",
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
            // [MAGI S6 MS3 loop2, R6b Melchior/Balthasar] the RAG declarations (RV07, RV13) the
            // text form writes, pinned in the JSON form too.
            "trimmed",
            "without_embedder",
            "without_memory",
            "context_withheld",
            // [MAGI S6 MS3 loop3, R6b Melchior] the count past the trimmed list's cap.
            "trimmed_unlisted",
        ] {
            assert!(v.get(k).is_some(), "missing {k}: {v}");
        }
    }

    #[test]
    fn the_text_names_each_undelivered_unit() {
        // [MAGI S6 MS3 loop16, R4] A mandatory unit the source promised and did not bring is
        // declared on its own line, apart from the optional context trimmed to fit.
        let s = ReviewSummary {
            undelivered_units: vec!["unit_a".into(), "unit_b".into()],
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(
            out.contains("containing units were not delivered: unit_a, unit_b"),
            "{out}"
        );
    }

    #[test]
    fn without_a_database_the_text_names_that_cause_not_a_missing_embedder() {
        // [MAGI S6 MS3 loop3, R6b Melchior] Both flags can be set for one review (no store, so
        // no embedder either): the text names the database, the cause the user can act on.
        let s = ReviewSummary {
            without_memory: true,
            without_embedder: true,
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(out.contains("the run opened no database"), "{out}");
        assert!(!out.contains("no embedder answered"), "{out}");
    }

    #[test]
    fn the_text_form_masks_a_raw_secret_in_its_paths_and_labels_as_it_writes_them() {
        // [MAGI S6 MS3 loop4, R6b Melchior] The module promises both forms redact as they are
        // written; the omission lists relied on their builders (`display_path`) alone. A raw
        // secret handed in any of them must not reach the text.
        let key = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let s = ReviewSummary {
            trimmed: vec![TrimmedItem {
                kind: ContextKind::Documentation,
                label: format!("section {key}"),
            }],
            skipped: vec![Skipped {
                path: format!("skipped/{key}.txt"),
                cause: crate::review::filter::SkipCause::Secret,
            }],
            masked: vec![Masked {
                path: format!("masked/{key}.rs"),
                cause: crate::review::filter::SkipCause::Secret,
                pattern: "anthropic-key",
            }],
            no_extractor: vec![format!("plain/{key}.txt")],
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(!out.contains("sk-ant-"), "{out}");
    }

    #[test]
    fn a_masked_pattern_name_is_written_line_safe() {
        // [MAGI S6 MS3 loop2, R6b Caspar] the path was escaped and the pattern written raw; the
        // names are fixed today, and the text form keeps them line-safe whatever they become.
        let s = ReviewSummary {
            masked: vec![Masked {
                path: "a.rs".into(),
                cause: crate::review::filter::SkipCause::Secret,
                pattern: "bell\u{7}\nname",
            }],
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(
            out.contains("- a.rs (secret: bell\\u{7}\\nname)\n"),
            "{out}"
        );
    }

    #[test]
    fn the_text_form_writes_the_context_only_cause_with_the_spec_text() {
        // [MAGI S6 MS3 loop5, R6b Melchior] The JSON form pins `context only`; the text form's
        // discarded list writes the same serde name, pinned here.
        let s = ReviewSummary {
            discarded: vec![Discarded {
                agent: AgentName::Melchior,
                batch: 0,
                finding: magi_core::schema::Finding::new(Severity::Warning, "t", "d"),
                cause: DiscardCause::ContextOnly,
                symbol: None,
            }],
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(out.contains(" [context only] t (batch 1, "), "{out}");
    }

    #[test]
    fn a_masked_pattern_name_shaped_like_a_secret_is_masked_in_the_text_form() {
        // [MAGI S6 MS3 loop5, R6b Caspar] The pattern names are the detector's own constants
        // today, but the text form wrote this field with `escape_controls` alone while every
        // other omission field went through the backstop; the JSON form masks it already.
        const PATTERN: &str = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let s = ReviewSummary {
            masked: vec![Masked {
                path: "a.rs".into(),
                cause: crate::review::filter::SkipCause::Secret,
                pattern: PATTERN,
            }],
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(!out.contains("sk-ant-"), "{out}");
        assert!(!summary_json(&s).to_string().contains("sk-ant-"));
    }

    #[test]
    fn the_context_left_out_is_listed_up_to_the_cap_and_the_rest_counted() {
        // [MAGI S6 MS3 loop2, R6b Melchior] `trimmed` grows with the batches; it was written whole,
        // unlike the other omission lists of the section, in the text and in the JSON.
        let extra = 3;
        let s = ReviewSummary {
            trimmed: (0..MAX_OMISSIONS_SHOWN + extra)
                .map(|i| TrimmedItem {
                    kind: ContextKind::Caller,
                    label: format!("f{i}"),
                })
                .collect(),
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        let line = out
            .lines()
            .find(|l| l.starts_with("context left out: "))
            .expect("the line");
        assert_eq!(
            line.matches("caller f").count(),
            MAX_OMISSIONS_SHOWN,
            "{line}"
        );
        assert!(line.ends_with(&format!("; and {extra} more")), "{line}");
        let v = summary_json(&s);
        assert_eq!(
            v["trimmed"].as_array().map(Vec::len),
            Some(MAX_OMISSIONS_SHOWN)
        );
        assert_eq!(v["trimmed_unlisted"], extra);
    }

    #[test]
    fn only_findings_on_a_line_of_the_reviewed_revision_are_anchored() {
        // [MAGI S6 MS3 loop2, R6b Caspar/Balthasar/Melchior] RV08: a removed-line finding never
        // reaches the history; an unlocated one and one on the same number of another file do.
        let at = |file: Option<&str>, line: Option<u32>| {
            (
                ConsolidatedFinding {
                    file: file.map(str::to_string),
                    line,
                    severity: Severity::Warning,
                    reported_severity: Severity::Warning,
                    titles: vec!["t".into()],
                    details: vec![],
                    categories: vec![Category::Other],
                    sources: vec![AgentName::Melchior],
                    marks: vec![],
                    batches: vec![0],
                },
                GateVerdict {
                    state: GateState::Kept,
                    cause: None,
                    causes: vec![],
                    model: None,
                    lean: None,
                    confidence: None,
                    skipped: vec![],
                    status: None,
                },
            )
        };
        let mut s = ReviewSummary {
            findings: vec![
                at(Some("a.rs"), Some(9)),
                at(Some("a.rs"), Some(1)),
                at(Some("b.rs"), Some(9)),
                at(None, None),
                at(Some("a.rs"), None),
            ],
            ..ReviewSummary::default()
        };
        s.off_revision
            .entry("a.rs".to_string())
            .or_default()
            .insert(9);
        let kept: Vec<(Option<String>, Option<u32>)> = s
            .anchored_findings()
            .into_iter()
            .map(|(f, _)| (f.file, f.line))
            .collect();
        assert_eq!(
            kept,
            vec![
                (Some("a.rs".to_string()), Some(1)),
                (Some("b.rs".to_string()), Some(9)),
                (None, None),
                (Some("a.rs".to_string()), None),
            ]
        );
    }

    #[test]
    fn off_revision_matches_exact_file() {
        // [MAGI S6 MS3 loop8, R6b Melchior] The key is the finding's own `file` string, inserted
        // by the pipeline from that very finding; the match is exact, so another spelling of the
        // path (`./a.rs`) is a different key and is never dropped by accident.
        let f = |file: &str| {
            let finding = ConsolidatedFinding {
                file: Some(file.into()),
                line: Some(9),
                severity: Severity::Warning,
                reported_severity: Severity::Warning,
                titles: vec!["t".into()],
                details: vec![],
                categories: vec![Category::Other],
                sources: vec![AgentName::Melchior],
                marks: vec![],
                batches: vec![0],
            };
            let verdict = GateVerdict {
                state: GateState::Kept,
                cause: None,
                causes: vec![],
                model: None,
                lean: None,
                confidence: None,
                skipped: vec![],
                status: None,
            };
            (finding, verdict)
        };
        let s = |off: &str| ReviewSummary {
            findings: vec![f("a.rs"), f("./a.rs")],
            off_revision: BTreeMap::from([(off.to_string(), BTreeSet::from([9]))]),
            ..ReviewSummary::default()
        };
        let files = |s: ReviewSummary| -> Vec<Option<String>> {
            s.anchored_findings()
                .into_iter()
                .map(|(f, _)| f.file)
                .collect()
        };
        assert_eq!(files(s("a.rs")), vec![Some("./a.rs".to_string())]);
        assert_eq!(files(s("./a.rs")), vec![Some("a.rs".to_string())]);
    }

    #[test]
    fn a_masked_file_is_declared_in_both_forms_and_never_as_skipped() {
        // [T3.7] RV22 amended: a file reviewed with its secret masked is declared with the cause
        // `secret` and its pattern, apart from the files left out.
        let mut s = ReviewSummary::default();
        s.masked.push(Masked {
            path: "a.rs".into(),
            cause: crate::review::filter::SkipCause::Secret,
            pattern: "url-credential",
        });
        let v = summary_json(&s);
        assert_eq!(
            v["masked"],
            json!([{"path": "a.rs", "cause": "secret", "pattern": "url-credential"}])
        );
        assert_eq!(v["masked_unlisted"], 0);
        assert_eq!(v["skipped"], json!([]));
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(
            out.contains("\n## Reviewed with secrets masked\n- a.rs (secret: url-credential)\n"),
            "{out}"
        );
        assert!(!out.contains("## Skipped"), "{out}");
    }

    #[test]
    fn a_secret_in_a_model_name_is_masked_in_the_keys_of_the_summary_json() {
        // [MAGI S6 loop 11, C2-c1, security] `redact_strings` visited values only, so a model
        // name shaped like a key reached the JSON unmasked as a key of `per_model` or
        // `rejected_skips` (RV22: every byte that leaves the process passes the detector).
        let secret = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: Some("clef:cloud".into()),
            lean: None,
            confidence: Some(0.9),
            skipped: vec![secret.into()],
            status: None,
        };
        let mut s = ReviewSummary::default();
        s.per_model.insert(secret.into(), 1);
        s.per_model.insert("clef:cloud".into(), 2);
        s.findings.push((finding, verdict));
        let v = summary_json(&s);
        assert!(!v.to_string().contains(secret), "{v}");
        assert_eq!(v["per_model"]["clef:cloud"], 2, "{v}");
        assert_eq!(v["per_model"].as_object().map(|m| m.len()), Some(2), "{v}");
        assert_eq!(
            v["rejected_skips"].as_object().map(|m| m.len()),
            Some(1),
            "{v}"
        );
    }

    #[test]
    fn a_distinct_key_climbs_past_every_taken_spelling() {
        // [MAGI S6 MS3 loop15] `distinct_key` cannot overflow: by pigeonhole `taken.len()` keys
        // block at most that many spellings, so with `k`, `k (2)` .. `k (1000)` taken it ends at
        // `k (1001)`, `taken.len() + 1`, with no failure to report and no value repeated.
        let mut taken = serde_json::Map::new();
        taken.insert("k".into(), serde_json::json!(0));
        for n in 2..=1000 {
            taken.insert(format!("k ({n})"), serde_json::json!(n));
        }
        assert_eq!(distinct_key(&taken, "k".into()), "k (1001)");
    }

    #[test]
    fn running_out_of_key_suffixes_is_a_typed_failure_not_a_repeat() {
        // [MAGI S6 MS3 loop15] the counter used to saturate, repeating `usize::MAX` forever.
        let mut taken = serde_json::Map::new();
        taken.insert(format!("k ({})", usize::MAX), serde_json::json!(0));
        assert_eq!(
            distinct_key_from(&taken, "k", usize::MAX),
            Err(KeySuffixesExhausted)
        );
        assert_eq!(
            distinct_key_from(&taken, "k", usize::MAX - 1),
            Ok(format!("k ({})", usize::MAX - 1))
        );
    }

    #[test]
    fn two_model_names_masked_alike_keep_both_counts_in_the_summary_json() {
        // [MAGI S6 loop 11, C2-c1] masking keys must not merge two of them into one entry.
        let a = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let b = concat!("ghp_", "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB");
        let mut s = ReviewSummary::default();
        s.per_model.insert(a.into(), 1);
        s.per_model.insert(b.into(), 2);
        let v = summary_json(&s);
        let models = v["per_model"].as_object().cloned().unwrap_or_default();
        assert_eq!(models.len(), 2, "{v}");
        assert_eq!(models.values().filter_map(Value::as_u64).sum::<u64>(), 3);
        assert!(
            !v.to_string().contains(a) && !v.to_string().contains(b),
            "{v}"
        );
    }

    #[test]
    fn every_batch_index_of_the_json_form_is_one_based() {
        // [MAGI S6 loop 4, C2] a renamed or new batch field must not stay 0-based while the text
        // form is 1-based. Since S6 loop 5 the renumbering is each field's `serialize_with`.
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
            status: None,
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
        // [MAGI S6 loop 5, C2-b2] and by shape, not by path: every batch index above is 0 in
        // memory, so a 0 under a key that names batch indices (`batch`, `index`, or a list
        // ending in `batches`; the `batches_*` keys are counts) is one some field forgot.
        fn zeros_under_batch_keys(v: &Value, under_batch_key: bool, out: &mut usize) {
            match v {
                Value::Number(n) if under_batch_key && n.as_u64() == Some(0) => *out += 1,
                Value::Array(items) => items
                    .iter()
                    .for_each(|i| zeros_under_batch_keys(i, under_batch_key, out)),
                Value::Object(fields) => fields.iter().for_each(|(k, i)| {
                    let names_indices = k == "batch" || k == "index" || k.ends_with("batches");
                    zeros_under_batch_keys(i, names_indices, out)
                }),
                _ => {}
            }
        }
        let mut zeros = 0;
        zeros_under_batch_keys(&v, false, &mut zeros);
        assert_eq!(zeros, 0, "{v}");
    }

    #[tokio::test]
    async fn every_consolidated_finding_still_builds_magi_cores_key_finding() {
        // [MAGI S6 loop 4, C2] `key_finding` goes through magi-core's `Deserialize`; a magi-core
        // bump that changes that shape must turn this red, not silently move every finding out
        // of Key Findings.
        let f = serde_json::json!([{"severity":"warning","title":"unique-title","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        assert!(key_finding(c.consensus.findings.first(), &c.findings[0]).is_some());
        let mut s = ReviewSummary::default();
        s.findings.push((c.findings[0].clone(), verdict));
        let text = render_text(&c, &s);
        assert!(!text.contains("could not be listed"), "{text}");
    }

    #[test]
    fn a_refused_request_shows_its_http_status_in_the_text_and_the_json() {
        // [MAGI S6 loop 5, C1-c3] a 413/422 `chain exhausted` names its status in both forms.
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::Unfiltered,
            cause: Some("chain exhausted"),
            causes: vec!["chain exhausted"],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: Some(413),
        };
        let mut s = ReviewSummary::default();
        s.findings.push((finding, verdict));
        let v = summary_json(&s);
        assert_eq!(v["findings"][0]["gate"]["status"], 413, "{v}");
        let mut text = String::new();
        write_review_section(&mut text, &s, 0).expect("writes");
        assert!(text.contains("chain exhausted; HTTP 413"), "{text}");
    }

    #[test]
    fn every_gate_cause_is_a_one_line_literal_written_as_is() {
        // [MAGI S6 loop 14, C2-1] the causes are the one gate field written without
        // `escape_controls`; that is safe only while each is a printable literal of the gate's
        // closed vocabulary. This pins the field's type and every literal.
        // Static (`GateVerdict::causes` holds `&'static str`); no needle holds a line break and
        // `lines()` drops a `\r`, so the check reads LF and CRLF trees alike.
        let gate: &'static str = include_str!("gate.rs");
        assert!(
            gate.contains("pub causes: Vec<&'static str>,"),
            "the causes are no longer static literals"
        );
        let declared = gate.matches("\nconst CAUSE_").count();
        let causes: Vec<&str> = gate
            .lines()
            .filter_map(|l| l.strip_prefix("const CAUSE_"))
            .filter_map(|rest| rest.split_once(": &str = \""))
            .filter_map(|(_, value)| value.strip_suffix("\";"))
            .collect();
        assert_eq!(causes.len(), declared, "a cause is not a one-line literal");
        assert!(causes.len() >= 10, "{causes:?}");
        for cause in causes {
            assert!(
                // No escape either: the source text of `"a\nb"` is printable, its value is not.
                !cause.is_empty()
                    && cause
                        .bytes()
                        .all(|b| (b.is_ascii_graphic() && b != b'\\') || b == b' '),
                "{cause:?}"
            );
            assert_eq!(escape_controls(cause), cause);
            let finding = ConsolidatedFinding {
                file: Some("a.rs".into()),
                line: Some(1),
                severity: Severity::Warning,
                reported_severity: Severity::Warning,
                titles: vec!["t".into()],
                details: vec![],
                categories: vec![Category::Other],
                sources: vec![AgentName::Melchior, AgentName::Caspar],
                marks: vec![],
                batches: vec![0],
            };
            let verdict = GateVerdict {
                state: GateState::Unfiltered,
                cause: Some(cause),
                causes: vec![cause],
                model: None,
                lean: None,
                confidence: None,
                skipped: vec![],
                status: None,
            };
            let mut out = String::new();
            write_finding(&mut out, &finding, &verdict).expect("writes");
            assert_eq!(out.lines().count(), 1, "{out}");
            assert!(out.contains(&format!("({cause})")), "{out}");
        }
    }

    #[test]
    fn the_findings_heading_is_written_even_when_the_main_list_is_empty() {
        // [MAGI S6 loop 14, C2-2] the heading of the main list is a stable anchor, written with
        // no finding under it; the optional sections are not.
        let mut out = String::new();
        write_review_section(&mut out, &ReviewSummary::default(), 0).expect("writes");
        assert!(out.ends_with("\n## Findings\n"), "{out}");
        for optional in [
            "## Filtered by the gate",
            "## Skipped",
            "## Reviewed without language context",
        ] {
            assert!(!out.contains(optional), "{out}");
        }
    }

    #[test]
    fn a_value_that_fails_to_serialize_is_declared_never_null() {
        // [MAGI S6 loop 5, C2-m1/b4/c1] the impossible failure is loud, not an empty envelope.
        struct Failing;
        impl serde::Serialize for Failing {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("boom"))
            }
        }
        let v = to_json(&Failing, "summary");
        assert_eq!(
            v[SERIALIZATION_ERROR_KEY].as_str(),
            Some("summary: boom"),
            "{v}"
        );
    }

    #[test]
    fn the_text_form_lists_every_category_of_a_merged_finding() {
        // [MAGI S6 loop 4, C2] RV-S18 conserves both categories; the JSON showed them and the text
        // showed none.
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(10),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::LogicError, Category::ErrorHandling],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        let mut out = String::new();
        write_finding(&mut out, &finding, &verdict).expect("writes");
        assert!(out.contains("[logic-error, error-handling]"), "{out}");
    }

    #[test]
    fn every_line_of_a_multiline_detail_stays_indented_under_its_finding() {
        // [MAGI S6 loop 4, C2] a detail with a newline broke out of its finding's indentation, so
        // its second line read as a line of the report.
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["first\nsecond\r\nthird".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        let mut out = String::new();
        write_finding(&mut out, &finding, &verdict).expect("writes");
        let details: Vec<&str> = out.lines().skip(1).collect();
        assert_eq!(details, ["  first", "  second", "  third"], "{out}");
    }

    #[test]
    fn a_non_finite_judged_fraction_is_written_as_nothing_judged_in_both_forms() {
        // [MAGI S6 loop 12, C2-m2 / C2-b4 / C2-c6] `serde_json` writes NaN and infinity as
        // `null`, so a non-finite fraction left a consumer a `null` where it reads a number, and
        // the text form printed `NaN %`. Unknown coverage is never claimed: it reads as 0.
        for fraction in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let s = ReviewSummary {
                judged_fraction: fraction,
                ..ReviewSummary::default()
            };
            assert_eq!(summary_json(&s)["judged_fraction"], 0.0, "{fraction}");
            let mut out = String::new();
            write_review_section(&mut out, &s, 0).expect("writes");
            assert!(out.contains("judged: 0.0 %"), "{fraction}: {out}");
        }
    }

    #[test]
    fn an_incomplete_gate_never_reads_as_fully_judged_in_the_text_form() {
        // [MAGI S6 loop 12, C2-m4 / C2-b2] `{:.0}` rounded 299 of 300 up to `100 %`, hiding the
        // unjudged finding C-7 made visible; the JSON kept the exact value.
        let s = ReviewSummary {
            judged_fraction: 299.0 / 300.0,
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(out.contains("judged: 99.6 %"), "{out}");
        assert!(!out.contains("judged: 100"), "{out}");
        let full = ReviewSummary::default();
        let mut out = String::new();
        write_review_section(&mut out, &full, 0).expect("writes");
        assert!(out.contains("judged: 100.0 %"), "{out}");
    }

    #[test]
    fn a_capped_lexical_index_is_declared_so_zero_symbol_discards_is_not_misread() {
        // [MAGI S6 loop 4, C1] with the index over its cap no symbol is checked: the report must
        // say so, or "no unknown symbol" would read as "none cited".
        let s = ReviewSummary {
            symbol_check_skipped: true,
            ..ReviewSummary::default()
        };
        let mut out = String::new();
        write_review_section(&mut out, &s, 0).expect("writes");
        assert!(
            out.contains("the symbol check was off: the lexical index hit its cap"),
            "{out}"
        );
        assert_eq!(summary_json(&s)["symbol_check_skipped"], true);
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
            status: None,
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
    async fn a_private_key_block_in_model_text_leaves_no_body_line_in_either_form() {
        // [MAGI S6 MS3 loop5, R6b Balthasar] A key body matches no pattern of its own; the
        // report's backstop (`redacted`, `mask_secrets`) masks a block from its header through
        // its END line, and an open block to the end of the text, so no body line is written,
        // whether the block sits in a detail, spans lines of a title or has no END line.
        let begin = concat!("-----", "BEGIN ", "OPENSSH PRIVATE KEY-----");
        let end = concat!("-----", "END ", "OPENSSH PRIVATE KEY-----");
        let body = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB";
        let block = format!("{begin}\n{body}\n{body}\n{end}");
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let mut c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered: report.agents.iter().map(|a| a.agent).collect(),
        }])
        .expect("consolidates");
        c.findings[0].details = vec![format!("the key:\n{block}\nends here")];
        c.findings[0].titles = vec![format!("open {begin}\n{body}")];
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        let mut s = ReviewSummary::default();
        s.findings.push((c.findings[0].clone(), verdict.clone()));
        s.filtered.push((c.findings[0].clone(), verdict));
        s.discarded.push(Discarded {
            agent: AgentName::Melchior,
            batch: 0,
            finding: magi_core::schema::Finding::new(Severity::Critical, block.as_str(), "d"),
            cause: DiscardCause::LineOutOfRange,
            symbol: None,
        });
        let text = render_text(&c, &s);
        assert!(!text.contains(body), "{text}");
        assert!(text.contains("<redacted: "), "{text}");
        let json = summary_json(&s).to_string();
        assert!(!json.contains(body), "{json}");
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

    #[tokio::test]
    async fn both_report_forms_mask_a_named_secret_in_a_finding_file_title_and_detail() {
        // [MAGI S6 loop 6, C2-c1 / C1-c2] RV22: the report is an egress point, so its own
        // redaction runs the product's detector, not only the URL-credential one.
        let github = format!("ghp_{}", "A".repeat(36));
        let openai = format!("sk-{}", "b".repeat(30));
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let mut c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        c.findings[0].file = Some(format!("src/{github}.rs"));
        c.findings[0].titles = vec![format!("key {openai}")];
        c.findings[0].details = vec![format!("token {github}")];
        let verdict = GateVerdict {
            state: GateState::Kept,
            cause: None,
            causes: vec![],
            model: Some("m".into()),
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        let mut s = ReviewSummary::default();
        s.findings.push((c.findings[0].clone(), verdict));
        let text = render_text(&c, &s);
        let json = summary_json(&s).to_string();
        for form in [text, json] {
            assert!(!form.contains(&github), "{form}");
            assert!(!form.contains(&openai), "{form}");
            assert!(form.contains("<redacted: "), "{form}");
        }
    }

    #[test]
    fn a_default_summary_declares_everything_judged() {
        // [MAGI S6 loop 8, C2-c5] the field's contract is 1.0 with no finding; a derived
        // `Default` said 0.0, so a summary rendered before the gate ran declared nothing judged.
        let s = ReviewSummary::default();
        assert_eq!(s.judged_fraction, 1.0);
        assert_eq!(summary_json(&s)["judged_fraction"], 1.0);
    }

    #[test]
    fn a_key_finding_overwrites_every_field_of_its_template() {
        // [MAGI S6 loop 8, C2-c1] `key_finding` clones one magi-core finding and assigns its
        // fields by name, because `DedupFinding` is `#[non_exhaustive]`. A field magi-core adds
        // would keep the template's value for every Key Finding; this guard turns that into a
        // red test: the serialized key set must be exactly the fields assigned, and each of them
        // must carry the consolidated finding's value, not the template's.
        const ASSIGNED: [&str; 8] = [
            "category", "detail", "file", "id", "line", "severity", "sources", "title",
        ];
        let template: DedupFinding = serde_json::from_value(serde_json::json!({
            "severity": "critical", "title": "template", "detail": "template detail",
            "sources": ["balthasar"], "file": "template.rs", "line": 99,
            "category": "logic-error", "id": "template-id"
        }))
        .expect("a DedupFinding");
        let f = ConsolidatedFinding {
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
        };
        let before = serde_json::to_value(&template).expect("serializes");
        let keys: Vec<&str> = before
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        assert_eq!(
            keys, ASSIGNED,
            "magi-core's DedupFinding changed its fields"
        );
        let key = key_finding(Some(&template), &f).expect("a template");
        let after = serde_json::to_value(&key).expect("serializes");
        for field in ASSIGNED {
            assert_ne!(
                after.get(field),
                before.get(field),
                "{field} kept the template's value"
            );
        }
    }

    /// Every character a terminal or a log viewer may take as the start of a new line.
    const BREAKS: [char; 7] = [
        '\n', '\r', '\u{0B}', '\u{0C}', '\u{85}', '\u{2028}', '\u{2029}',
    ];

    /// Whether some line of `text`, split on every [`BREAKS`] character, starts with `FORGED`.
    fn has_forged_line(text: &str) -> bool {
        text.split(|c| BREAKS.contains(&c))
            .any(|line| line.starts_with("FORGED"))
    }

    #[tokio::test]
    async fn no_line_break_in_model_text_can_forge_a_line_of_the_text_report() {
        // [MAGI S6 loop 9, C2-m1 / C2-c1] titles, the discarded list's file and title, a failed
        // batch's cause and the gate's stop cause were written raw: a model-emitted line break
        // put the text after it at column 0, where it read as a line of the report.
        for b in BREAKS {
            let forged = format!("x{b}FORGED");
            let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
            let report = real_report([f.clone(), f, serde_json::json!([])]).await;
            let answered = report.agents.iter().map(|a| a.agent).collect();
            let mut c = consolidate(&[BatchOutcome {
                batch: 0,
                agents: report.agents.clone(),
                answered,
            }])
            .expect("consolidates");
            c.findings[0].file = Some(format!("a{forged}.rs"));
            c.findings[0].titles = vec![forged.clone()];
            c.findings[0].details = vec![forged.clone()];
            let verdict = GateVerdict {
                state: GateState::Kept,
                cause: None,
                causes: vec![],
                model: None,
                lean: None,
                confidence: None,
                skipped: vec![],
                status: None,
            };
            let discarded = Discarded {
                agent: AgentName::Caspar,
                batch: 0,
                finding: serde_json::from_value(serde_json::json!({
                    "severity": "critical", "title": forged, "detail": "d",
                    "file": format!("b{forged}.rs"), "line": 7
                }))
                .expect("a Finding"),
                cause: DiscardCause::FileNotReviewed,
                symbol: None,
            };
            let mut s = ReviewSummary::default();
            s.findings.push((c.findings[0].clone(), verdict));
            s.discarded.push(discarded.clone());
            s.symbol_unverified.push(discarded);
            s.failed_batches.push((1, forged.clone()));
            s.gate_stopped = Some(forged.clone());
            let text = render_text(&c, &s);
            assert!(!has_forged_line(&text), "break {b:?}: {text}");
            let paid = render_batch_reports(&[(0, report)]);
            let mut forged_report = real_report([
                serde_json::json!([{"severity":"warning","title":forged,"detail":"d","file":"a.rs","line":1}]),
                serde_json::json!([]),
                serde_json::json!([]),
            ])
            .await;
            forged_report.consensus.findings.truncate(1);
            let paid_forged = render_batch_reports(&[(0, forged_report)]);
            assert!(!has_forged_line(&paid), "{paid}");
            assert!(!has_forged_line(&paid_forged), "break {b:?}: {paid_forged}");
        }
    }

    #[tokio::test]
    async fn no_line_break_in_a_model_name_or_a_count_key_can_forge_a_report_line() {
        // [MAGI S6 loop 10, C2-c1] model names come from the configuration and the pool, never
        // validated against control characters: the counts' keys, the models without a measured
        // window and each verdict's model, lean and skipped models were written raw.
        for b in BREAKS {
            let forged = format!("x{b}FORGED");
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
                state: GateState::Kept,
                cause: None,
                causes: vec![],
                model: Some(forged.clone()),
                lean: Some(forged.clone()),
                confidence: Some(0.9),
                skipped: vec![forged.clone()],
                status: None,
            };
            let mut s = ReviewSummary::default();
            s.findings.push((c.findings[0].clone(), verdict.clone()));
            s.filtered.push((c.findings[0].clone(), verdict));
            s.per_model.insert(forged.clone(), 1);
            s.gate_states.insert(forged.clone(), 1);
            s.gate_causes.insert(forged.clone(), 1);
            s.unmeasured_models.push(forged.clone());
            let text = render_text(&c, &s);
            assert!(text.matches("FORGED").count() >= 9, "{text}");
            assert!(!has_forged_line(&text), "break {b:?}: {text}");
        }
    }

    #[tokio::test]
    async fn a_key_finding_detail_is_line_safe_before_the_formatter_sees_it() {
        // [MAGI S6 loop 10, C2-m1] magi-core 4.2.0 does not render a Key Finding's detail, but a
        // formatter that starts to must inherit the line safety the other fields have.
        for b in BREAKS {
            let forged = format!("x{b}FORGED");
            let f = serde_json::json!([{"severity":"warning","title":"t","detail":forged,"file":"a.rs","line":1}]);
            let report = real_report([f.clone(), f, serde_json::json!([])]).await;
            let (_, consensus) = line_safe(&report.agents, &report.consensus);
            let detail = &consensus.findings[0].detail;
            assert!(detail.contains("FORGED"), "{detail}");
            assert!(!has_forged_line(detail), "break {b:?}: {detail:?}");
        }
    }

    #[tokio::test]
    async fn a_key_finding_file_is_line_safe_before_the_formatter_sees_it() {
        // [MAGI S6 loop 12, C2-m1 / C2-c1, security] `line_safe` scrubbed only a Key Finding's
        // title and detail; the file is repository text (a Unix path may hold a line break), and
        // a formatter that starts rendering it must inherit the line safety of the other fields.
        for b in BREAKS {
            let forged = format!("a{b}FORGED.rs");
            let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":forged,"line":1}]);
            let report = real_report([f.clone(), f, serde_json::json!([])]).await;
            let (_, consensus) = line_safe(&report.agents, &report.consensus);
            let file = consensus.findings[0].file.clone().unwrap_or_default();
            assert!(file.contains("FORGED"), "{file}");
            assert!(!has_forged_line(&file), "break {b:?}: {file:?}");
        }
    }

    #[tokio::test]
    async fn a_model_category_is_a_closed_vocabulary_and_cannot_forge_a_report_line() {
        // [MAGI S6 loop 12, C2-m1 / C2-b1 / C2-c1] reviewed as a false positive: `category` is
        // magi-core's closed `Category` enum (`#[serde(other)]`), so model text with a line break
        // deserializes to `other` and never reaches the report as text. Pinned so a magi-core bump
        // that gives `Category` a data-carrying variant turns this red.
        for b in BREAKS {
            let forged = format!("logic{b}FORGED");
            let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1,"category":forged}]);
            let report = real_report([f.clone(), f, serde_json::json!([])]).await;
            let answered = report.agents.iter().map(|a| a.agent).collect();
            let c = consolidate(&[BatchOutcome {
                batch: 0,
                agents: report.agents.clone(),
                answered,
            }])
            .expect("consolidates");
            assert!(c.findings[0].categories.iter().all(|k| name(k) == "other"));
            let verdict = GateVerdict {
                state: GateState::Kept,
                cause: None,
                causes: vec![],
                model: None,
                lean: None,
                confidence: None,
                skipped: vec![],
                status: None,
            };
            let mut s = ReviewSummary::default();
            s.findings.push((c.findings[0].clone(), verdict));
            let text = render_text(&c, &s);
            assert!(!has_forged_line(&text), "break {b:?}: {text}");
            let paid = render_batch_reports(&[(0, report)]);
            assert!(!has_forged_line(&paid), "break {b:?}: {paid}");
        }
    }
    #[tokio::test]
    async fn a_prose_field_opening_with_a_heading_cannot_open_a_report_section() {
        // [MAGI S6 loop 11, C2-m1 / C2-b2, security] only the continuation lines were indented,
        // so a prose field whose first line was `## …` relied on magi-core's formatter writing a
        // label before it; a formatter that started the field at column 0 would print a forged
        // section heading.
        let forged = "## FORGED\nmore";
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":forged,"file":"a.rs","line":1}]);
        let mut report = real_report([f.clone(), f, serde_json::json!([])]).await;
        forge_report_text(&mut report.agents, &mut report.consensus, forged);
        let (agents, consensus) = line_safe(&report.agents, &report.consensus);
        let mut prose: Vec<&str> = agents
            .iter()
            .flat_map(|a| [&a.summary, &a.reasoning, &a.recommendation])
            .map(String::as_str)
            .collect();
        prose.extend(consensus.findings.iter().map(|f| f.detail.as_str()));
        prose.extend(
            consensus
                .dissent
                .iter()
                .flat_map(|d| [d.summary.as_str(), d.reasoning.as_str()]),
        );
        prose.extend(consensus.conditions.iter().map(|c| c.condition.as_str()));
        prose.extend(consensus.recommendations.values().map(String::as_str));
        assert!(prose.len() > 3, "{prose:?}");
        for field in prose {
            assert!(field.contains("## FORGED"), "{field:?}");
            assert!(
                !field
                    .split(|c| BREAKS.contains(&c))
                    .any(|l| l.starts_with('#')),
                "{field:?}"
            );
        }
        let mut c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered: report.agents.iter().map(|a| a.agent).collect(),
        }])
        .expect("consolidates");
        forge_report_text(&mut c.agents, &mut c.consensus, forged);
        let text = render_text(&c, &ReviewSummary::default());
        assert!(text.contains("## FORGED"), "{text}");
        assert!(!text.lines().any(|l| l.starts_with("## FORGED")), "{text}");
    }

    /// Puts `forged` in every model-supplied string magi-core's formatter may render, and in the
    /// agents' own prose: the dissent, the conditions, the recommendations, the Key Findings'
    /// titles and each agent's summary, reasoning and recommendation.
    fn forge_report_text(
        agents: &mut [magi_core::schema::AgentOutput],
        consensus: &mut magi_core::consensus::ConsensusResult,
        forged: &str,
    ) {
        for agent in agents.iter_mut() {
            agent.summary = forged.to_string();
            agent.reasoning = forged.to_string();
            agent.recommendation = forged.to_string();
        }
        for finding in &mut consensus.findings {
            finding.title = forged.to_string();
        }
        consensus.dissent = vec![serde_json::from_value(serde_json::json!({
            "agent": "caspar", "summary": forged, "reasoning": forged
        }))
        .expect("a Dissent")];
        consensus.conditions = vec![serde_json::from_value(serde_json::json!({
            "agent": "balthasar", "condition": forged
        }))
        .expect("a Condition")];
        for rec in consensus.recommendations.values_mut() {
            *rec = forged.to_string();
        }
    }

    #[tokio::test]
    async fn no_line_break_in_magi_cores_sections_or_listed_paths_can_forge_a_report_line() {
        // [MAGI S6 loop 9, C2 commit review] magi-core's formatter writes the dissent, the
        // conditions and the recommendations raw, and the review wrote the skipped and the
        // no-extractor paths raw: a line break there put model or path text at column 0.
        for b in BREAKS {
            let forged = format!("x{b}FORGED");
            let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
            let mut report = real_report([f.clone(), f, serde_json::json!([])]).await;
            let answered = report.agents.iter().map(|a| a.agent).collect();
            let mut c = consolidate(&[BatchOutcome {
                batch: 0,
                agents: report.agents.clone(),
                answered,
            }])
            .expect("consolidates");
            forge_report_text(&mut c.agents, &mut c.consensus, &forged);
            let mut s = ReviewSummary::default();
            s.skipped.push(Skipped {
                path: forged.clone(),
                cause: crate::review::filter::SkipCause::Binary,
            });
            s.no_extractor.push(forged.clone());
            let text = render_text(&c, &s);
            assert!(text.matches("FORGED").count() >= 5, "{text}");
            assert!(!has_forged_line(&text), "break {b:?}: {text}");
            forge_report_text(&mut report.agents, &mut report.consensus, &forged);
            let paid = render_batch_reports(&[(0, report)]);
            assert!(paid.matches("FORGED").count() >= 4, "{paid}");
            assert!(!has_forged_line(&paid), "break {b:?}: {paid}");
        }
    }

    #[test]
    fn no_label_name_writes_a_line_break_raw_whatever_its_type() {
        // [MAGI S6 loop 13, C2-c1, security] `name` returned the serialized text, or the `Debug`
        // form, as it was; every caller then had to remember `escape_controls`, and four did not
        // (`Discarded.agent`, `Discarded.cause`, a finding's severity, a verdict's state). Their
        // types are closed today; a type whose serialized text or `Debug` form carries model text
        // (a foreign `#[non_exhaustive]` enum with a `String` payload) wrote it straight in.
        struct Prose;
        impl serde::Serialize for Prose {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_u8(0)
            }
        }
        impl std::fmt::Debug for Prose {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("x\nFORGED")
            }
        }
        for b in BREAKS {
            let text = name(&format!("x{b}FORGED"));
            assert!(text.contains("FORGED"), "{text:?}");
            assert!(!has_forged_line(&text), "break {b:?}: {text:?}");
        }
        let debug = name(&Prose);
        assert!(
            debug.contains("FORGED") && !has_forged_line(&debug),
            "{debug:?}"
        );
    }

    /// Every JSON pointer of `value`'s string and null leaves, depth first.
    fn leaf_pointers(value: &Value, at: &str, out: &mut Vec<String>) {
        match value {
            Value::String(_) | Value::Null => out.push(at.to_string()),
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    leaf_pointers(item, &format!("{at}/{i}"), out);
                }
            }
            Value::Object(fields) => {
                for (k, item) in fields {
                    leaf_pointers(item, &format!("{at}/{k}"), out);
                }
            }
            Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// `value` with `probe` in every leaf that holds free text, and those leaves' pointers.
    ///
    /// A leaf holds free text when, set to `probe`, the value still deserializes as `T` and
    /// serializes `probe` back at that pointer; an enum or a fail-soft field (a `category` that
    /// reads unknown text as `other`) does not. The set is found by construction, so a text
    /// field magi-core adds is probed without anyone listing it.
    fn forge_free_text<T>(value: &T, probe: &str) -> (T, Vec<String>)
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let mut current = serde_json::to_value(value).expect("serializes");
        let mut pointers = Vec::new();
        leaf_pointers(&current, "", &mut pointers);
        let mut free = Vec::new();
        for p in pointers {
            let mut candidate = current.clone();
            if let Some(leaf) = candidate.pointer_mut(&p) {
                *leaf = Value::String(probe.to_string());
            }
            let back = serde_json::from_value::<T>(candidate.clone())
                .ok()
                .and_then(|t| serde_json::to_value(t).ok());
            if back.as_ref().and_then(|b| b.pointer(&p)) == Some(&Value::String(probe.into())) {
                current = candidate;
                free.push(p);
            }
        }
        (serde_json::from_value(current).expect("deserializes"), free)
    }

    /// The free-text fields `line_safe` leaves raw on purpose: the deprecated
    /// `majority_summary` (assigning it is a deprecation warning), which magi-core 4.2.0 does
    /// not render; the test below turns red the day it does. Top-level keys of the consensus,
    /// each written as a JSON key literal (the deprecated-surface guard allows exactly that).
    const LEFT_RAW: [&str; 1] = ["majority_summary"];

    #[tokio::test]
    async fn every_free_text_field_of_magi_cores_report_types_is_line_safe() {
        // [MAGI S6 loop 13, C2-c2 / C2-m1 / C2-b3, security] `line_safe` assigned a hand-written
        // list of fields: the agents' own findings were left out, and a field magi-core adds or
        // starts rendering would have been written raw with no test turning red. The fields are
        // now found by construction: every leaf that holds free text is forged.
        let forged = "x\nFORGED";
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1,"category":"logic-error"}]);
        let mut report = real_report([f.clone(), f, serde_json::json!([])]).await;
        forge_report_text(&mut report.agents, &mut report.consensus, "seed");
        let (consensus, consensus_fields) = forge_free_text(&report.consensus, forged);
        let mut agents = Vec::new();
        let mut agent_fields = Vec::new();
        for agent in &report.agents {
            let (a, fields) = forge_free_text(agent, forged);
            agents.push(a);
            agent_fields.push(fields);
        }
        for needed in [
            "/findings/0/title",
            "/findings/0/detail",
            "/findings/0/file",
        ] {
            assert!(
                agent_fields[0].iter().any(|p| p == needed),
                "{agent_fields:?}"
            );
        }
        for needed in [
            "/dissent/0/summary",
            "/conditions/0/condition",
            "/findings/0/id",
        ] {
            assert!(
                consensus_fields.iter().any(|p| p == needed),
                "{consensus_fields:?}"
            );
        }
        let (safe_agents, safe_consensus) = line_safe(&agents, &consensus);
        let mut checks: Vec<(Value, &Vec<String>)> = safe_agents
            .iter()
            .map(|a| serde_json::to_value(a).expect("serializes"))
            .zip(&agent_fields)
            .collect();
        checks.push((
            serde_json::to_value(&safe_consensus).expect("serializes"),
            &consensus_fields,
        ));
        for (value, fields) in &checks {
            let left_raw = |p: &str| p.strip_prefix('/').is_some_and(|k| LEFT_RAW.contains(&k));
            for p in fields.iter().filter(|p| !left_raw(p)) {
                let text = value.pointer(p).and_then(Value::as_str).unwrap_or_default();
                assert!(text.contains("FORGED"), "{p}: {text:?}");
                assert!(!has_forged_line(text), "{p} is written raw: {text:?}");
            }
        }
        let rendered = ReportFormatter::new().format_report(&safe_agents, &safe_consensus);
        assert!(
            !has_forged_line(&rendered),
            "a raw field is rendered: {rendered}"
        );
    }

    #[tokio::test]
    async fn magi_cores_report_types_keep_the_fields_the_report_was_written_for() {
        // [MAGI S6 loop 13, C2-c2] the serialized field set of each magi-core type the text
        // report hands to the formatter, pinned like `DedupFinding`'s: a field added, renamed or
        // removed upstream turns this red, so `line_safe` is reviewed with the bump.
        fn keys(v: &Value) -> Vec<&str> {
            v.as_object()
                .map(|o| o.keys().map(String::as_str).collect())
                .unwrap_or_default()
        }
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
        let mut report = real_report([f.clone(), f, serde_json::json!([])]).await;
        forge_report_text(&mut report.agents, &mut report.consensus, "seed");
        let agent = serde_json::to_value(&report.agents[0]).expect("serializes");
        assert_eq!(
            keys(&agent),
            [
                "agent",
                "confidence",
                "findings",
                "reasoning",
                "recommendation",
                "summary",
                "verdict"
            ]
        );
        assert_eq!(
            keys(&agent["findings"][0]),
            ["category", "detail", "file", "line", "severity", "title"]
        );
        let consensus = serde_json::to_value(&report.consensus).expect("serializes");
        assert_eq!(
            keys(&consensus),
            [
                "agent_count",
                "conditions",
                "confidence",
                "consensus",
                "consensus_verdict",
                "dissent",
                "findings",
                "majority_summary",
                "recommendations",
                "score",
                "votes"
            ]
        );
        assert_eq!(
            keys(&consensus["dissent"][0]),
            ["agent", "reasoning", "summary"]
        );
        assert_eq!(keys(&consensus["conditions"][0]), ["agent", "condition"]);
    }

    #[tokio::test]
    async fn both_report_forms_cap_the_omission_lists_and_declare_the_rest() {
        // [MAGI S6 loop 13, C2-c4 / C2-m3, performance] `skipped` and `no_extractor` grew with the
        // reviewed tree, one line or entry per file, in the text and the JSON alike; the error
        // path already capped the same kind of list. Both forms now list the first
        // `MAX_OMISSIONS_SHOWN` and declare how many more there are.
        use crate::review::error::MAX_OMISSIONS_SHOWN;
        let extra = 3;
        let mut s = ReviewSummary::default();
        for i in 0..MAX_OMISSIONS_SHOWN + extra {
            s.skipped.push(Skipped {
                path: format!("bin{i}.dat"),
                cause: crate::review::filter::SkipCause::Binary,
            });
            s.no_extractor.push(format!("doc{i}.txt"));
        }
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"a.rs","line":1}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered: report.agents.iter().map(|a| a.agent).collect(),
        }])
        .expect("consolidates");
        let text = render_text(&c, &s);
        let last = MAX_OMISSIONS_SHOWN - 1;
        for shown in [format!("bin{last}.dat"), format!("doc{last}.txt")] {
            assert!(text.contains(&shown), "{text}");
        }
        for hidden in [
            format!("bin{MAX_OMISSIONS_SHOWN}.dat"),
            format!("doc{MAX_OMISSIONS_SHOWN}.txt"),
        ] {
            assert!(!text.contains(&hidden), "{text}");
        }
        assert_eq!(
            text.matches(&format!("- ... and {extra} more")).count(),
            2,
            "{text}"
        );
        let v = summary_json(&s);
        for (list, rest) in [
            ("skipped", "skipped_unlisted"),
            ("no_extractor", "no_extractor_unlisted"),
        ] {
            assert_eq!(
                v[list].as_array().map(Vec::len),
                Some(MAX_OMISSIONS_SHOWN),
                "{v}"
            );
            assert_eq!(v[rest], extra, "{v}");
        }
        let none = summary_json(&ReviewSummary::default());
        assert_eq!(
            (&none["skipped_unlisted"], &none["no_extractor_unlisted"]),
            (&json!(0), &json!(0))
        );
    }

    #[test]
    fn a_finding_has_no_own_field_the_gate_verdict_could_overwrite() {
        // [MAGI S6 loop 13, C2-c5] `with_gate` inserts the verdict under `gate` into the
        // finding's own object; a field of that name would vanish with valid JSON and no error,
        // the trap `AppliedCaps` and `BudgetTelemetry` taught (serde has no disjointness check).
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior],
            marks: vec![],
            batches: vec![0],
        };
        let own = serde_json::to_value(&finding).expect("serializes");
        let keys: Vec<&str> = own
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        assert!(!keys.is_empty(), "{own}");
        assert!(!keys.contains(&GATE_KEY), "{keys:?}");
    }
}
