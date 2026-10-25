// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! The review pipeline (REQ-RV05, REQ-RV06, §5.0): one `consult code-review` per batch, in
//! series, then one report.
//!
//! [`run_review`] is the only entry. It resolves the input, filters it, plans the batches, checks
//! an explicit `--timeout` before any trio call, runs the batches, validates each against what
//! the whole review showed, consolidates, runs the false-positive gate and applies the agreement
//! weight, in that order.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use magi_core::reporting::MagiReport;

use super::batching::{file_header, plan_batches, BatchItem, ReviewBatch};
use super::clock::{batch_wall_clock_secs, pre_budget};
use super::consolidate::{
    apply_agreement, consolidate, BatchOutcome, Consolidated, ConsolidatedFinding,
};
use super::error::ReviewError;
use super::filter::{filter_files, Filtered, SkipCause, Skipped};
use super::gate::{Gate, GateState, GateVerdict};
use super::index::{LexicalIndex, Reviewed};
use super::input::{resolve_input, ReviewFile, ReviewInput, SourceKind};
use super::languages::LanguageRegistry;
use super::report::{name, BatchMetrics, ReviewSummary};
use super::secrets::display_path;
use super::source::{CachedSource, ContentSource, GitCommand, RevisionSource, WorktreeSource};
use super::validate::validate_batch;
use super::window::{batch_limit_tokens, pool_window, PoolWindow};
use crate::redact::redact_foreign_text;

/// Run-log target of the pipeline's WARN entries.
const LOG_TARGET: &str = "magi_rs::review";
/// Why a batch did not run, or stopped, when its wall clock ran out.
const CLOCK_EXPIRED: &str = "clock expired";
/// Windows' verbatim path prefix, which a reported location never carries.
const VERBATIM_PREFIX: &str = r"\\?\";
/// What follows [`VERBATIM_PREFIX`] in a verbatim UNC path (`\\?\UNC\srv\share`).
const VERBATIM_UNC: &str = r"UNC\";
/// The prefix of a UNC path as an editor writes it (`\\srv\share`).
const UNC_PREFIX: &str = r"\\";

/// Why one batch's `consult` gave no report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsultFailure {
    /// The batch's wall clock ran out: the batches still pending are not run.
    ClockExpired,
    /// Any other failure, already redacted.
    Other(String),
}

/// One `consult code-review` over the trio.
///
/// The production implementation, in the binary, calls the trio with the code-review mode as
/// an argument (never written into the prompt) and maps its error with
/// [`run_with_wall_clock`].
#[async_trait::async_trait]
pub trait BatchConsult: Send + Sync {
    /// Runs the trio over `prompt`, under `wall_clock` when there is one.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The batch: its items, each file preceded by its header.
    /// * `wall_clock` - The batch's wall clock; `None` runs it on the normal `consult` clock.
    ///
    /// # Returns
    ///
    /// The batch report.
    ///
    /// # Errors
    ///
    /// [`ConsultFailure::ClockExpired`] when the wall clock ran out, otherwise
    /// [`ConsultFailure::Other`] with a redacted message.
    async fn consult(
        &self,
        prompt: &str,
        wall_clock: Option<Duration>,
    ) -> Result<MagiReport, ConsultFailure>;
}

/// Seconds elapsed since some fixed instant. Production: [`SystemReviewClock`].
pub trait ReviewClock: Send + Sync {
    /// Seconds elapsed.
    fn elapsed_secs(&self) -> u64;
}

/// The production [`ReviewClock`], over an [`Instant`].
pub struct SystemReviewClock {
    start: Instant,
}

impl SystemReviewClock {
    /// A clock that starts now.
    ///
    /// # Returns
    ///
    /// The clock.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl ReviewClock for SystemReviewClock {
    fn elapsed_secs(&self) -> u64 {
        self.start.elapsed().as_secs()
    }
}

/// Runs one batch under its wall clock.
///
/// When the wall clock elapses, `fut` is dropped, and with it every request it owns: nothing is
/// left running on this side (a backend may still finish a generation nobody reads).
///
/// # Arguments
///
/// * `wall` - The batch's wall clock; `None` waits for `fut` however long it takes.
/// * `fut` - The analysis, whose error is already redacted.
///
/// # Returns
///
/// The batch report.
///
/// # Errors
///
/// [`ConsultFailure::ClockExpired`] when the wall clock elapsed, [`ConsultFailure::Other`]
/// with the future's error otherwise (a timeout inside the analysis included).
pub async fn run_with_wall_clock<F>(
    wall: Option<Duration>,
    fut: F,
) -> Result<MagiReport, ConsultFailure>
where
    F: std::future::Future<Output = Result<MagiReport, String>> + Send,
{
    let result = match wall {
        Some(wall) => tokio::time::timeout(wall, fut)
            .await
            .map_err(|_| ConsultFailure::ClockExpired)?,
        None => fut.await,
    };
    result.map_err(ConsultFailure::Other)
}

/// What to review and under which clock.
pub struct ReviewRequest<'a> {
    /// What the caller asked to review.
    pub input: ReviewInput,
    /// The workspace root.
    pub workspace: &'a Path,
    /// The explicit `--timeout`, the total of the review, counted from the first trio call.
    pub timeout_secs: Option<u64>,
    /// `[magi].max_rotations`, resolved.
    pub max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    pub retry_disabled: bool,
    /// `[magi].agent_timeout_secs`, resolved: what the derived ceiling is compared with.
    pub agent_timeout_secs: u64,
    /// The batch count the caller approved, if any: a review that now splits otherwise fails
    /// with [`ReviewError::PlanChanged`] before any trio call.
    pub approved_batches: Option<usize>,
}

/// What a review runs on.
pub struct ReviewDeps<'a> {
    /// The review's git.
    pub git: &'a dyn GitCommand,
    /// The sandbox check of `File`/`Directory` paths.
    pub guard: &'a (dyn Fn(&Path) -> Result<PathBuf, ReviewError> + Sync),
    /// Whether `.gitignore` excludes a path.
    pub is_ignored: &'a (dyn Fn(&str) -> bool + Sync),
    /// The language extractors.
    pub registry: &'a LanguageRegistry,
    /// The pool, as `(model, measured window)`.
    pub pool: &'a [(String, Option<usize>)],
    /// The trio.
    pub consult: &'a dyn BatchConsult,
    /// The false-positive gate of this review.
    pub gate: Gate<'a>,
    /// The review clock.
    pub clock: &'a dyn ReviewClock,
}

/// What a review produced.
pub struct ReviewOutcome {
    /// The consolidated report; `None` when no batch completed or consolidation failed.
    pub consolidated: Option<Consolidated>,
    /// The paid batch reports, raw, each with its 0-based batch index, when consolidation failed
    /// (empty otherwise).
    pub batch_reports: Vec<(usize, MagiReport)>,
    /// What every stage declared.
    pub summary: ReviewSummary,
    /// Why there is no consolidated report, redacted.
    pub consolidation_error: Option<String>,
}

/// Where a file split by units stands: whether every batch holding one of its units completed,
/// and the line ranges of the units that did.
struct UnitProgress {
    /// Every batch holding one of its units completed.
    all_done: bool,
    /// The ranges of its finished units.
    finished: Vec<(u32, u32)>,
}

/// One batch that the trio answered.
struct Finished {
    /// Zero-based batch index.
    batch: usize,
    /// Its report, raw.
    report: MagiReport,
}

/// The faces of a file that the completed batches showed, to pick the code the gate sees.
#[derive(Default)]
struct Faces {
    /// The file was shown whole or by units: its new side covers every line.
    whole: bool,
    /// New-side ranges of its hunks.
    new: Vec<(u32, u32)>,
    /// Old-side ranges of its hunks.
    old: Vec<(u32, u32)>,
}

/// Runs a review, in the order of the spec's §5.0.
///
/// 1. Resolve the input and its single content source, wrapped in [`CachedSource`] so every
///    stage reads the same bytes, once.
/// 2. Filter what never leaves the process and what is not code.
/// 3. Plan the batches against the pool window.
/// 4. With an explicit `--timeout`, check the budget ([`pre_budget`]) before any trio call; then
///    build the lexical index. Steps 1-4 are the preparation, measured apart and never charged
///    to the `--timeout`.
/// 5. Run one `consult` per batch, in series. Without `--timeout` each batch runs on the normal
///    `consult` clock; with it, each gets what is left split over the pending batches, counted
///    from the first trio call. A batch whose wall clock ran out stops the review's batches
///    (the cut ones are declared); any other failure is declared and the next batch runs.
/// 6. Validate each completed batch against what all completed batches showed.
/// 7. Consolidate.
/// 8. Run the gate once per consolidated finding, within what is left of the `--timeout`.
/// 9. Apply the agreement weight.
///
/// # Arguments
///
/// * `req` - What to review and under which clock.
/// * `deps` - Git, sandbox, extractors, pool, trio, gate and clock.
///
/// # Returns
///
/// The outcome. When no batch completed, or consolidation failed, it carries no consolidated
/// report, the reason, and (for a failed consolidation) the paid batch reports.
///
/// # Errors
///
/// Any [`ReviewError`] of steps 1-4, all before the first trio call: those of the input, the
/// filter and the batching, [`ReviewError::InputTooLarge`] from the cache and
/// [`ReviewError::Budget`].
///
/// O(bytes reviewed) for the preparation, plus the consolidation's own bound.
pub async fn run_review(
    req: ReviewRequest<'_>,
    mut deps: ReviewDeps<'_>,
) -> Result<ReviewOutcome, ReviewError> {
    let prep = Instant::now();
    let mut summary = ReviewSummary::default();
    let resolved = resolve_input(&req.input, req.workspace, deps.git, deps.guard)?;
    let inner = content_source(resolved.source_kind, req.workspace, deps.git);
    let source = CachedSource::new(inner.as_ref());
    let explicit = matches!(req.input, ReviewInput::File(_));
    let (filtered, window, batches) = filter_and_plan(
        resolved.files,
        &source,
        explicit,
        deps.registry,
        deps.is_ignored,
        deps.pool,
    )?;
    summary.skipped = filtered.skipped;
    summary
        .skipped
        .extend(resolved.not_utf8.iter().map(|path| Skipped {
            path: display_path(path),
            cause: SkipCause::NotUtf8,
        }));
    summary.no_extractor = filtered.no_extractor;
    summary.unmeasured_models = window.unmeasured;
    let planned = batches.len();
    summary.batches_planned = planned;
    if let Some(approved) = req.approved_batches.filter(|a| *a != planned) {
        return Err(ReviewError::PlanChanged {
            approved,
            now: planned,
        });
    }
    if let Some(total) = req.timeout_secs {
        let ceiling = pre_budget(total, planned, req.max_rotations, req.retry_disabled)?;
        summary.ceiling_below_configured = ceiling < req.agent_timeout_secs;
    }
    let index = build_index(&filtered.kept, &source, deps.registry)?;
    summary.symbol_check_skipped = !index.is_complete();
    if summary.symbol_check_skipped {
        tracing::event!(
            target: LOG_TARGET,
            tracing::Level::WARN,
            "the lexical index hit its cap: the symbol check is off for this review"
        );
    }
    summary.prep_ms = millis(prep.elapsed());

    let clock = ReviewTimer {
        clock: deps.clock,
        t0: deps.clock.elapsed_secs(),
        total: req.timeout_secs,
    };
    let finished = run_batches(&batches, deps.consult, &clock, &mut summary).await;

    if finished.is_empty() {
        let message = format!(
            "no batch completed: {} of {planned} failed",
            summary.failed_batches.len()
        );
        return Ok(ReviewOutcome {
            consolidated: None,
            batch_reports: Vec::new(),
            summary,
            consolidation_error: Some(message),
        });
    }

    let completed: BTreeSet<usize> = finished.iter().map(|f| f.batch).collect();
    let reviewed = reviewed_of(&batches, &completed, &source)?;
    let workspace = without_verbatim_prefix(req.workspace);
    let mut outcomes = Vec::with_capacity(finished.len());
    for f in &finished {
        let mut agents = f.report.agents.clone();
        let answered = agents.iter().map(|a| a.agent).collect();
        let validation = validate_batch(&mut agents, f.batch, &reviewed, &index, &workspace);
        summary.discarded.extend(validation.discarded);
        summary
            .symbol_unverified
            .extend(validation.symbol_unverified);
        outcomes.push(BatchOutcome {
            batch: f.batch,
            agents,
            answered,
        });
    }

    let mut consolidated = match consolidate(&outcomes) {
        Ok(c) => c,
        Err(e) => {
            let message = redact_foreign_text(&e.to_string()).as_str().to_string();
            return Ok(ReviewOutcome {
                consolidated: None,
                batch_reports: finished.into_iter().map(|f| (f.batch, f.report)).collect(),
                summary,
                consolidation_error: Some(message),
            });
        }
    };
    summary.ungrouped_unlocated = consolidated.ungrouped_unlocated;
    if summary.ungrouped_unlocated > 0 {
        tracing::event!(
            target: LOG_TARGET,
            tracing::Level::WARN,
            ungrouped = summary.ungrouped_unlocated,
            "unlocated findings were grouped without the consensus engine"
        );
    }

    let faces = faces_of(&batches, &completed);
    let mut verdicts = Vec::with_capacity(consolidated.findings.len());
    for finding in &consolidated.findings {
        let code = gate_code(finding.file.as_deref(), finding.line, &faces, &source);
        verdicts.push(
            deps.gate
                .judge(finding, code.as_deref(), clock.remaining())
                .await,
        );
    }
    summary.per_model = deps.gate.per_model().clone();
    summary.gate_stopped = deps.gate.stopped().map(str::to_string);

    apply_agreement(&mut consolidated.findings, &consolidated.answered);
    record_verdicts(&mut summary, &consolidated.findings, verdicts);

    Ok(ReviewOutcome {
        consolidated: Some(consolidated),
        batch_reports: Vec::new(),
        summary,
        consolidation_error: None,
    })
}

/// Counts the batches a review would run, without calling any model (§5.0 steps 1-3).
///
/// The same resolution, filter and batching [`run_review`] applies, so the count shown before a
/// review (the TUI confirmation, the tool's approval prompt) is the count it runs with.
///
/// # Arguments
///
/// * `input` - What to review.
/// * `workspace` - The workspace root.
/// * `git` - The review's git.
/// * `guard` - The sandbox check of `File`/`Directory` paths.
/// * `is_ignored` - Whether `.gitignore` excludes a path.
/// * `registry` - The language extractors.
/// * `pool` - The pool, as `(model, measured window)`.
///
/// # Returns
///
/// The number of batches.
///
/// # Errors
///
/// The errors of [`run_review`]'s preparation before the budget check: the input, the filter,
/// the batching and [`ReviewError::InputTooLarge`].
///
/// O(bytes reviewed).
pub fn plan_review(
    input: &ReviewInput,
    workspace: &Path,
    git: &dyn GitCommand,
    guard: &(dyn Fn(&Path) -> Result<PathBuf, ReviewError> + Sync),
    is_ignored: &(dyn Fn(&str) -> bool + Sync),
    registry: &LanguageRegistry,
    pool: &[(String, Option<usize>)],
) -> Result<usize, ReviewError> {
    let resolved = resolve_input(input, workspace, git, guard)?;
    let inner = content_source(resolved.source_kind, workspace, git);
    let source = CachedSource::new(inner.as_ref());
    let explicit = matches!(input, ReviewInput::File(_));
    let (_, _, batches) = filter_and_plan(
        resolved.files,
        &source,
        explicit,
        registry,
        is_ignored,
        pool,
    )?;
    Ok(batches.len())
}

/// Steps 2-3 of §5.0 over the resolved files, shared by [`run_review`] and [`plan_review`]: the
/// filter, the pool window and the batches. O(bytes reviewed).
fn filter_and_plan(
    files: Vec<ReviewFile>,
    source: &dyn ContentSource,
    explicit: bool,
    registry: &LanguageRegistry,
    is_ignored: &dyn Fn(&str) -> bool,
    pool: &[(String, Option<usize>)],
) -> Result<(Filtered, PoolWindow, Vec<ReviewBatch>), ReviewError> {
    let filtered = filter_files(files, source, registry, explicit, is_ignored)?;
    let window = pool_window(pool);
    let batches = plan_batches(
        &filtered.kept,
        source,
        registry,
        batch_limit_tokens(&window),
    )?;
    Ok((filtered, window, batches))
}

/// The single content source an input resolved to: the worktree (with its old side, if any) or
/// a git revision.
fn content_source<'g>(
    kind: SourceKind,
    workspace: &Path,
    git: &'g dyn GitCommand,
) -> Box<dyn ContentSource + 'g> {
    match kind {
        SourceKind::Worktree { old_rev: None } => {
            Box::new(WorktreeSource::new(workspace.to_path_buf()))
        }
        SourceKind::Worktree {
            old_rev: Some(old_rev),
        } => Box::new(WorktreeSource::with_old(
            workspace.to_path_buf(),
            git,
            old_rev,
        )),
        SourceKind::Revision { new_rev, old_rev } => {
            Box::new(RevisionSource::new(git, new_rev, old_rev))
        }
    }
}

/// The review clock from the first trio call: every elapsed time is relative to `t0`, so the
/// preparation is never charged to the `--timeout`, and every subtraction saturates.
struct ReviewTimer<'c> {
    /// The review clock.
    clock: &'c dyn ReviewClock,
    /// Its reading just before the first batch.
    t0: u64,
    /// The explicit `--timeout`, if any.
    total: Option<u64>,
}

impl ReviewTimer<'_> {
    /// What is left of the `--timeout` in seconds, or `None` without one.
    fn left_secs(&self) -> Option<u64> {
        let elapsed = self.clock.elapsed_secs().saturating_sub(self.t0);
        self.total.map(|total| total.saturating_sub(elapsed))
    }

    /// What is left of the `--timeout`, or `None` without one.
    fn remaining(&self) -> Option<Duration> {
        self.left_secs().map(Duration::from_secs)
    }
}

/// Runs every batch in series (§5.0 step 5), recording metrics and failures in `summary`.
///
/// Each batch's wall clock is recomputed from what is left, split over the pending batches; a
/// batch whose wall clock ran out cuts it and every pending one, each declared as
/// `clock expired`.
async fn run_batches(
    batches: &[ReviewBatch],
    consult: &dyn BatchConsult,
    clock: &ReviewTimer<'_>,
    summary: &mut ReviewSummary,
) -> Vec<Finished> {
    let planned = batches.len();
    let mut finished = Vec::new();
    for batch in batches {
        let wall = clock.left_secs().map(|left| {
            let pending = planned.saturating_sub(batch.index);
            Duration::from_secs(batch_wall_clock_secs(left, pending))
        });
        let started = Instant::now();
        match consult.consult(&prompt_of(batch), wall).await {
            Ok(report) => {
                summary
                    .batches
                    .push(metrics(batch.index, &report, started.elapsed()));
                finished.push(Finished {
                    batch: batch.index,
                    report,
                });
            }
            Err(ConsultFailure::ClockExpired) => {
                summary
                    .failed_batches
                    .extend((batch.index..planned).map(|k| (k, CLOCK_EXPIRED.to_string())));
                break;
            }
            Err(ConsultFailure::Other(message)) => {
                summary.failed_batches.push((batch.index, message));
            }
        }
    }
    summary.batches_completed = finished.len();
    finished
}

/// Records the gate's verdicts in `summary`: the counts per state and cause, the fraction
/// judged (every `unfiltered` counts as not judged; 1.0 with no finding), and each finding in
/// the main list or, when filtered, apart.
fn record_verdicts(
    summary: &mut ReviewSummary,
    findings: &[ConsolidatedFinding],
    verdicts: Vec<GateVerdict>,
) {
    let judged = verdicts
        .iter()
        .filter(|v| v.state != GateState::Unfiltered)
        .count();
    summary.judged_fraction = if verdicts.is_empty() {
        1.0
    } else {
        ratio(judged, verdicts.len())
    };
    for (finding, verdict) in findings.iter().zip(verdicts) {
        *summary.gate_states.entry(name(&verdict.state)).or_insert(0) += 1;
        if let Some(cause) = verdict.cause {
            *summary.gate_causes.entry(cause.to_string()).or_insert(0) += 1;
        }
        let list = if verdict.state == GateState::Filtered {
            &mut summary.filtered
        } else {
            &mut summary.findings
        };
        list.push((finding.clone(), verdict));
    }
}

/// The prompt of a batch: its items in order, each file's first item preceded by its header.
/// The mode is an argument of the trio call, never part of this text.
fn prompt_of(batch: &ReviewBatch) -> String {
    let mut prompt = String::new();
    let mut previous: Option<&str> = None;
    for item in &batch.items {
        let (file, text) = match item {
            BatchItem::Hunk { file, text, .. }
            | BatchItem::Unit { file, text, .. }
            | BatchItem::WholeFile { file, text, .. } => (file.as_str(), text.as_str()),
        };
        if previous != Some(file) {
            prompt.push_str(&file_header(file));
            previous = Some(file);
        }
        prompt.push_str(text);
        if !text.ends_with('\n') {
            prompt.push('\n');
        }
    }
    prompt
}

/// The metrics of a completed batch: every completion record's tokens, summed over the seats.
fn metrics(index: usize, report: &MagiReport, took: Duration) -> BatchMetrics {
    let (input_tokens, output_tokens) =
        report
            .completions
            .values()
            .flatten()
            .fold((0u64, 0u64), |(i, o), r| {
                (
                    i.saturating_add(u64::from(r.prompt_tokens.unwrap_or(0))),
                    o.saturating_add(u64::from(r.completion_tokens.unwrap_or(0))),
                )
            });
    BatchMetrics {
        index,
        input_tokens,
        output_tokens,
        ms: millis(took),
    }
}

/// A duration in whole milliseconds, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// `part / whole` as a fraction; `whole > 0`.
fn ratio(part: usize, whole: usize) -> f64 {
    let as_f64 = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    as_f64(part) / as_f64(whole)
}

/// The lexical index over both sides of every reviewed file. O(bytes reviewed).
fn build_index(
    files: &[ReviewFile],
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
) -> Result<LexicalIndex, ReviewError> {
    let mut texts = Vec::new();
    for file in files {
        let mut sides = vec![source.read(&file.path)?];
        if file.diff.is_some() {
            sides.push(source.read_old(&file.path)?);
        }
        for bytes in sides.into_iter().flatten() {
            texts.push((
                file.path.clone(),
                String::from_utf8_lossy(&bytes).into_owned(),
            ));
        }
    }
    Ok(LexicalIndex::build(&texts, registry))
}

/// What the completed batches showed (U24): both ranges of each hunk, each whole file, and each
/// file split by units whole when every batch holding one completed, else only its finished
/// units. O(items · log r).
fn reviewed_of(
    batches: &[ReviewBatch],
    completed: &BTreeSet<usize>,
    source: &dyn ContentSource,
) -> Result<Reviewed, ReviewError> {
    let mut reviewed = Reviewed::default();
    let mut units: BTreeMap<&str, UnitProgress> = BTreeMap::new();
    for batch in batches {
        let done = completed.contains(&batch.index);
        for item in &batch.items {
            match item {
                BatchItem::Hunk {
                    file,
                    first_line,
                    last_line,
                    old_first_line,
                    old_last_line,
                    ..
                } if done => {
                    reviewed.add_lines(file, *first_line, *last_line);
                    reviewed.add_lines(file, *old_first_line, *old_last_line);
                }
                BatchItem::WholeFile { file, lines, .. } if done => {
                    reviewed.add_whole_file(file, *lines);
                }
                BatchItem::Unit {
                    file,
                    first_line,
                    last_line,
                    ..
                } => {
                    let progress = units.entry(file.as_str()).or_insert(UnitProgress {
                        all_done: true,
                        finished: Vec::new(),
                    });
                    progress.all_done &= done;
                    if done {
                        progress.finished.push((*first_line, *last_line));
                    }
                }
                BatchItem::Hunk { .. } | BatchItem::WholeFile { .. } => {}
            }
        }
    }
    for (file, UnitProgress { all_done, finished }) in units {
        if all_done {
            let lines = source
                .read(file)?
                .map(|bytes| String::from_utf8_lossy(&bytes).lines().count())
                .unwrap_or(0);
            reviewed.add_whole_file(file, u32::try_from(lines).unwrap_or(u32::MAX));
        } else {
            for (first, last) in finished {
                reviewed.add_lines(file, first, last);
            }
        }
    }
    Ok(reviewed)
}

/// The faces each file showed in the completed batches. O(items).
fn faces_of<'b>(
    batches: &'b [ReviewBatch],
    completed: &BTreeSet<usize>,
) -> HashMap<&'b str, Faces> {
    let mut faces: HashMap<&str, Faces> = HashMap::new();
    for batch in batches.iter().filter(|b| completed.contains(&b.index)) {
        for item in &batch.items {
            match item {
                BatchItem::Hunk {
                    file,
                    first_line,
                    last_line,
                    old_first_line,
                    old_last_line,
                    ..
                } => {
                    let f = faces.entry(file.as_str()).or_default();
                    f.new.push((*first_line, *last_line));
                    f.old.push((*old_first_line, *old_last_line));
                }
                BatchItem::Unit { file, .. } | BatchItem::WholeFile { file, .. } => {
                    faces.entry(file.as_str()).or_default().whole = true;
                }
            }
        }
    }
    faces
}

/// The code the gate sees for a finding (RV10 + RV21): the new side when the line lies in a new
/// range or the file was shown whole, else the old side when it lies in an old range, numbered
/// as that side. `None` (the gate's `no location`) without a line, without a face that holds
/// it, or when the read fails, with a WARN; with the cache that read is unreachable, because
/// every face of a reviewed file was read during the preparation.
fn gate_code(
    file: Option<&str>,
    line: Option<u32>,
    faces: &HashMap<&str, Faces>,
    source: &dyn ContentSource,
) -> Option<String> {
    let (file, line) = (file?, line?);
    let face = faces.get(file)?;
    let holds = |ranges: &[(u32, u32)]| ranges.iter().any(|&(a, b)| a <= line && line <= b);
    let read = if face.whole || holds(&face.new) {
        source.read(file)
    } else if holds(&face.old) {
        source.read_old(file)
    } else {
        return None;
    };
    match read {
        Ok(bytes) => bytes.map(|b| String::from_utf8_lossy(&b).into_owned()),
        Err(e) => {
            let error = redact_foreign_text(&e.to_string());
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = error.as_str(),
                "the gate could not read the code of a finding"
            );
            None
        }
    }
}

/// `path` without Windows' verbatim prefix, so a reported absolute location compares with the
/// workspace as an editor writes it: `\\?\C:\w` becomes `C:\w` and `\\?\UNC\srv\share\w` becomes
/// `\\srv\share\w`. Any other path is returned as it is.
fn without_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(VERBATIM_PREFIX) {
        Some(rest) => match rest.strip_prefix(VERBATIM_UNC) {
            Some(share) => PathBuf::from(format!("{UNC_PREFIX}{share}")),
            None => PathBuf::from(rest),
        },
        None => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Duration;

    use magi_core::reporting::MagiReport;
    use magi_core::schema::Severity;

    use super::*;
    // [RED-DRYRUN 2026-10-17, D20] Inside the lib the crate is `crate`, and MockDecisionClient lives in
    // `decision::client` (not re-exported); likewise `crate::magi::headless_consult_timeout_secs` below.
    use crate::decision::client::MockDecisionClient;
    use crate::decision::{Answer, Decision, DecisionError, ModelSpec, Trace};
    use crate::review::consolidate::FindingMark;
    use crate::review::gate::GateState;
    use crate::review::report::render_batch_reports;
    use crate::review::report::render_text;
    use crate::review::test_support::{real_report, FakeGit};
    use crate::review::validate::DiscardCause;
    use crate::review::window::BATCH_WINDOW_FRACTION_PERCENT;

    /// The pool window whose batch limit is 120 tokens: one `workspace` file (104 tokens with its
    /// header) per batch. Derived from the fraction so the fixture follows it (200 at 60 %).
    const POOL_WINDOW: usize = 120 * 100 / BATCH_WINDOW_FRACTION_PERCENT;

    /// Scripted trio: one reply per batch, recording the wall clock it was given; may advance a FakeClock.
    struct FakeConsult {
        replies: Mutex<VecDeque<Result<MagiReport, ConsultFailure>>>,
        clocks: Mutex<Vec<Option<Duration>>>,
        prompts: Mutex<Vec<String>>,
        advance_to: Option<(std::sync::Arc<FakeClock>, u64)>,
    }
    impl FakeConsult {
        fn new(replies: Vec<Result<MagiReport, ConsultFailure>>) -> Self {
            Self {
                replies: Mutex::new(replies.into()),
                clocks: Mutex::new(Vec::new()),
                prompts: Mutex::new(Vec::new()),
                advance_to: None,
            }
        }
        fn calls(&self) -> usize {
            self.clocks.lock().expect("lock").len()
        }
    }
    #[async_trait::async_trait]
    impl BatchConsult for FakeConsult {
        async fn consult(
            &self,
            p: &str,
            w: Option<Duration>,
        ) -> Result<MagiReport, ConsultFailure> {
            self.clocks.lock().expect("lock").push(w);
            self.prompts.lock().expect("lock").push(p.to_string());
            if let Some((clock, secs)) = &self.advance_to {
                clock.set(*secs);
            }
            self.replies
                .lock()
                .expect("lock")
                .pop_front()
                .unwrap_or_else(|| Err(ConsultFailure::Other("no scripted reply".into())))
        }
    }

    #[derive(Default)]
    struct FakeClock(std::sync::atomic::AtomicU64);
    impl FakeClock {
        fn set(&self, s: u64) {
            self.0.store(s, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl ReviewClock for FakeClock {
        fn elapsed_secs(&self) -> u64 {
            self.0.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    fn chain() -> Vec<ModelSpec> {
        vec![ModelSpec {
            model: "clef:cloud".into(),
            min_confidence: Some(0.40),
        }]
    }

    fn answer(model: &str, choice: &str, confidence: f64) -> Result<Decision, DecisionError> {
        let other = if choice == "real_defect" {
            "false_positive"
        } else {
            "real_defect"
        };
        let p = (1.0 + confidence) / 2.0;
        let answers = [(
            "gate".to_string(),
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

    fn gate_always(choice: &'static str) -> MockDecisionClient {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(move |r| answer(r.model().unwrap_or(""), choice, 0.95));
        c
    }

    /// A directory with `n` files of ~100 tokens each; the pool window (limit 120) puts one file per batch.
    fn workspace(n: usize) -> tempfile::TempDir {
        let d = tempfile::tempdir().expect("tempdir");
        for i in 0..n {
            let text = format!("fn f{i}() {{\n{}}}\n", "    let _x = 1;\n".repeat(24));
            std::fs::write(d.path().join(format!("f{i}.rs")), text).expect("write");
        }
        d
    }

    async fn review(
        dir: &Path,
        timeout: Option<u64>,
        consult: &FakeConsult,
        client: &MockDecisionClient,
        clock: &dyn ReviewClock,
    ) -> Result<ReviewOutcome, ReviewError> {
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        run_review(
            ReviewRequest {
                input: ReviewInput::Directory(dir.to_path_buf()),
                workspace: dir,
                timeout_secs: timeout,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
            },
            ReviewDeps {
                git: &git,
                guard: &guard,
                is_ignored: &|_| false,
                registry: &registry,
                pool: &pool,
                consult,
                gate: Gate::new(client, chain(), 61_000),
                clock,
            },
        )
        .await
    }

    fn empty() -> [serde_json::Value; 3] {
        [
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!([]),
        ]
    }

    #[tokio::test]
    async fn run_review_fills_the_batch_metrics_from_each_report() {
        // [CP2 MS2 S2a loop 9] RV13's per-batch tokens: the report's completion records, summed over seats.
        let dir = workspace(2);
        let mut v = serde_json::to_value(real_report(empty()).await).expect("serializes");
        for recs in v["completions"]
            .as_object_mut()
            .expect("completions")
            .values_mut()
        {
            for rec in recs.as_array_mut().expect("records") {
                rec["prompt_tokens"] = serde_json::json!(7);
                rec["completion_tokens"] = serde_json::json!(2);
            }
        }
        let report: MagiReport = serde_json::from_value(v).expect("still a MagiReport");
        let seats = report.completions.values().map(Vec::len).sum::<usize>() as u64;
        assert!(seats > 0, "the fixture must carry completion records");
        let consult = FakeConsult::new(vec![Ok(report.clone()), Ok(report)]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        let m = &out.summary.batches;
        assert_eq!(m.iter().map(|b| b.index).collect::<Vec<_>>(), vec![0, 1]);
        assert!(
            m.iter()
                .all(|b| (b.input_tokens, b.output_tokens) == (7 * seats, 2 * seats)),
            "{m:?}"
        );
    }

    #[tokio::test]
    async fn a_finding_on_a_file_only_in_a_failed_batch_is_not_reviewed() {
        // [CP2 MS2 S1 loop 2] batch 0 (f0.rs) answers and talks about f1.rs; batch 1 (f1.rs) fails, so nobody saw f1.rs.
        let dir = workspace(2);
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"f1.rs","line":3}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([f.clone(), f, serde_json::json!([])]).await),
            Err(ConsultFailure::Other("seat failed".into())),
        ]);
        let client = gate_always("real_defect");
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("reports");
        assert_eq!(out.summary.failed_batches.len(), 1);
        assert!(out
            .summary
            .discarded
            .iter()
            .all(|d| d.cause == DiscardCause::FileNotReviewed));
        assert_eq!(out.summary.discarded.len(), 2); // one per mage that reported it
        assert!(out.summary.findings.is_empty());
    }

    #[tokio::test]
    async fn a_blank_gap_between_the_units_of_a_whole_file_counts_as_reviewed() {
        // A `// gap` comment becomes a unit of its own; blank lines between two units belong to
        // none, so only registering the finished file whole covers them (U24).
        let d = tempfile::tempdir().expect("tempdir");
        let body = "    let _x = 1;\n".repeat(20);
        std::fs::write(
            d.path().join("big.rs"),
            format!("fn a() {{\n{body}}}\n\n\nfn b() {{\n{body}}}\n"),
        )
        .expect("write");
        let gap = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"big.rs","line":23}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([gap.clone(), gap, serde_json::json!([])]).await),
            Ok(real_report(empty()).await),
        ]);
        let out = review(
            d.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("reports");
        assert_eq!(out.summary.batches_planned, 2);
        assert!(
            out.summary.discarded.is_empty(),
            "{:?}",
            out.summary.discarded
        );
        assert_eq!(out.summary.findings.len(), 1);
    }

    #[tokio::test]
    async fn a_whole_file_split_by_units_is_reviewed_entire() {
        // [CP2 MS2 S1 loop 8] U24: «en archivo o directorio, el archivo entero», the lines between units included.
        let d = tempfile::tempdir().expect("tempdir");
        let body = "    let _x = 1;\n".repeat(20);
        std::fs::write(
            d.path().join("big.rs"),
            format!("fn a() {{\n{body}}}\n// gap\n\nfn b() {{\n{body}}}\n"),
        )
        .expect("write");
        let gap = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"big.rs","line":23}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([gap.clone(), gap, serde_json::json!([])]).await),
            Ok(real_report(empty()).await),
        ]);
        let out = review(
            d.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("reports");
        assert_eq!(out.summary.batches_planned, 2); // one unit per batch
        assert!(
            out.summary.discarded.is_empty(),
            "{:?}",
            out.summary.discarded
        );
        assert_eq!(out.summary.findings.len(), 1);
    }

    #[tokio::test]
    async fn an_interrupted_whole_file_registers_only_its_finished_units() {
        // [CP2 MS2 S1b loop 9] the batch holding `fn b` failed: nobody saw it, so a finding inside it is out of range.
        let d = tempfile::tempdir().expect("tempdir");
        let body = "    let _x = 1;\n".repeat(20);
        std::fs::write(
            d.path().join("big.rs"),
            format!("fn a() {{\n{body}}}\n// gap\n\nfn b() {{\n{body}}}\n"),
        )
        .expect("write");
        let in_b = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"big.rs","line":30}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([in_b.clone(), in_b, serde_json::json!([])]).await),
            Err(ConsultFailure::Other("seat failed".into())),
        ]);
        let out = review(
            d.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("reports");
        assert_eq!(out.summary.batches_planned, 2);
        assert!(out.summary.findings.is_empty());
        assert_eq!(
            out.summary
                .discarded
                .iter()
                .map(|x| x.cause.clone())
                .collect::<Vec<_>>(),
            vec![DiscardCause::LineOutOfRange; 2]
        );
    }

    #[tokio::test]
    async fn without_a_timeout_every_batch_runs_on_the_normal_consult_clock() {
        let dir = workspace(4);
        let mut replies = Vec::new();
        for _ in 0..4 {
            replies.push(Ok(real_report(empty()).await));
        }
        let consult = FakeConsult::new(replies);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("review runs");
        assert_eq!(
            *consult.clocks.lock().expect("lock"),
            vec![None, None, None, None]
        );
        assert_eq!(out.summary.batches_planned, 4);
        assert_eq!(out.summary.batches_completed, 4);
        assert!(out.consolidated.is_some());
        assert_eq!(out.summary.judged_fraction, 1.0); // [CP2 MS2 S2 loop 3] no findings: nothing was left unjudged
    }

    #[tokio::test]
    async fn a_finding_on_a_deleted_file_reaches_the_gate_with_old_side_code() {
        // [CP2 MS2 S2 loop 3] RV10 + RV21: a deleted file has no new side; the gate sees the old side, numbered as the old side.
        let dir = tempfile::tempdir().expect("tempdir");
        let old: String = (1..=10).map(|i| format!("old{i}\n")).collect();
        let removed: String = (1..=10).map(|i| format!("-old{i}\n")).collect();
        let diff = format!("diff --git a/gone.rs b/gone.rs\ndeleted file mode 100644\n--- a/gone.rs\n+++ /dev/null\n@@ -1,10 +0,0 @@\n{removed}");
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(diff.as_str()),
            ),
            ("ls-files --others --exclude-standard -z", Ok("")),
            ("ls-tree -r --name-only -z abc", Ok("gone.rs\0")),
            (
                "show --no-ext-diff --no-textconv abc:./gone.rs",
                Ok(old.as_str()),
            ),
        ]);
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"gone.rs","line":5}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let mut client = MockDecisionClient::new();
        client
            .expect_decide()
            .times(1)
            .withf(|r| {
                r.state()["code"]
                    .as_str()
                    .is_some_and(|c| c.contains("5: old5"))
            })
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.95));
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Uncommitted,
                workspace: dir.path(),
                timeout_secs: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
            },
            ReviewDeps {
                git: &git,
                guard: &guard,
                is_ignored: &|_| false,
                registry: &registry,
                pool: &pool,
                consult: &consult,
                gate: Gate::new(&client, chain(), 61_000),
                clock: &FakeClock::default(),
            },
        )
        .await
        .expect("runs");
        let (_, v) = out
            .summary
            .findings
            .iter()
            .find(|(f, _)| f.file.as_deref() == Some("gone.rs"))
            .expect("reached the gate");
        assert_eq!(v.state, GateState::Kept);
    }

    #[tokio::test]
    async fn each_side_of_each_file_is_read_from_git_once_per_review() {
        // [CP2 MS2 S2a loop 6] run_review wraps its source in CachedSource: the filter, the index and the gate all read the
        // old side of gone.rs, and git answers it once.
        struct Counting<'a>(
            &'a FakeGit,
            Mutex<std::collections::BTreeMap<String, usize>>,
        );
        impl GitCommand for Counting<'_> {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                *self
                    .1
                    .lock()
                    .expect("lock")
                    .entry(args.join(" "))
                    .or_insert(0) += 1;
                self.0.run_bytes(args)
            }
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let old: String = (1..=10).map(|i| format!("old{i}\n")).collect();
        let removed: String = (1..=10).map(|i| format!("-old{i}\n")).collect();
        let diff = format!("diff --git a/gone.rs b/gone.rs\ndeleted file mode 100644\n--- a/gone.rs\n+++ /dev/null\n@@ -1,10 +0,0 @@\n{removed}");
        let fake = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(diff.as_str()),
            ),
            ("ls-files --others --exclude-standard -z", Ok("")),
            ("ls-tree -r --name-only -z abc", Ok("gone.rs\0")),
            (
                "show --no-ext-diff --no-textconv abc:./gone.rs",
                Ok(old.as_str()),
            ),
        ]);
        let git = Counting(&fake, Mutex::new(std::collections::BTreeMap::new()));
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"gone.rs","line":5}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let client = gate_always("real_defect");
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        run_review(
            ReviewRequest {
                input: ReviewInput::Uncommitted,
                workspace: dir.path(),
                timeout_secs: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
            },
            ReviewDeps {
                git: &git,
                guard: &guard,
                is_ignored: &|_| false,
                registry: &registry,
                pool: &pool,
                consult: &consult,
                gate: Gate::new(&client, chain(), 61_000),
                clock: &FakeClock::default(),
            },
        )
        .await
        .expect("runs");
        let calls = git.1.lock().expect("lock");
        assert_eq!(
            calls.get("show --no-ext-diff --no-textconv abc:./gone.rs"),
            Some(&1),
            "{calls:?}"
        );
    }

    #[tokio::test]
    async fn when_no_batch_completes_the_message_says_so() {
        // [CP2 MS2 S2 loop 3] every consult failed: not a consolidation failure, and no consolidate call.
        let dir = workspace(2);
        let consult = FakeConsult::new(vec![
            Err(ConsultFailure::Other("down".into())),
            Err(ConsultFailure::Other("down".into())),
        ]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("reports");
        assert!(out.consolidated.is_none() && out.batch_reports.is_empty());
        assert_eq!(
            out.consolidation_error.as_deref(),
            Some("no batch completed: 2 of 2 failed")
        );
    }

    #[tokio::test]
    async fn the_summary_carries_the_omissions_and_the_gate_causes() {
        // [CP2 MS2 S2a2 loop 11] by content: a default-empty list or map would pass a key-presence check.
        let dir = workspace(1);
        std::fs::write(dir.path().join("logo.png"), b"\0\0PNG").expect("write");
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"f0.rs","line":2}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let mut client = MockDecisionClient::new();
        client.expect_decide().returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        assert_eq!(
            serde_json::to_value(&out.summary.skipped).expect("json"),
            serde_json::json!([{"path": "logo.png", "cause": "binary"}])
        );
        assert_eq!(
            out.summary.gate_causes.get("unauthorized"),
            Some(&1),
            "{:?}",
            out.summary.gate_causes
        );
    }

    #[tokio::test]
    async fn the_summary_lists_the_unmeasured_models_and_the_files_without_an_extractor() {
        // [CP2 MS2 cierre (sin MAGI)] by content: an empty default would pass a key-presence check.
        let dir = workspace(1);
        std::fs::write(dir.path().join("notes.txt"), "x\n").expect("write");
        let consult = FakeConsult::new(vec![
            Ok(real_report(empty()).await),
            Ok(real_report(empty()).await),
        ]);
        let client = MockDecisionClient::new();
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![
            ("m".to_string(), Some(POOL_WINDOW)),
            ("u".to_string(), None),
        ];
        let clock = FakeClock::default();
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Directory(dir.path().to_path_buf()),
                workspace: dir.path(),
                timeout_secs: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
            },
            ReviewDeps {
                git: &git,
                guard: &guard,
                is_ignored: &|_| false,
                registry: &registry,
                pool: &pool,
                consult: &consult,
                gate: Gate::new(&client, chain(), 61_000),
                clock: &clock,
            },
        )
        .await
        .expect("runs");
        assert_eq!(out.summary.no_extractor, vec!["notes.txt".to_string()]);
        assert_eq!(out.summary.unmeasured_models, vec!["u".to_string()]);
    }

    #[tokio::test]
    async fn a_stopped_gate_is_declared_once_in_the_report() {
        // [CP2 MS2 S2 loop 3] C-6 / RV-S15: a 401 stops the gate; the report says so in one line.
        let dir = workspace(1);
        let f = |l: u32| serde_json::json!({"severity":"warning","title":format!("t{l}"),"detail":"d","file":"f0.rs","line":l});
        let both = serde_json::json!([f(2), f(4)]);
        let consult = FakeConsult::new(vec![Ok(real_report([
            both.clone(),
            both,
            serde_json::json!([]),
        ])
        .await)]);
        let mut client = MockDecisionClient::new();
        client.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        assert_eq!(out.summary.gate_stopped.as_deref(), Some("unauthorized"));
        let text = render_text(
            out.consolidated.as_ref().expect("consolidated"),
            &out.summary,
        );
        assert_eq!(text.matches("gate stopped: ").count(), 1, "{text}");
    }

    #[tokio::test]
    async fn the_text_report_lists_filtered_findings_apart_and_prints_the_lowered_severity() {
        // RV10: a filtered finding is listed apart, never among the findings; RV11: a single-mage
        // critical is a warning everywhere the text names its severity.
        let dir = workspace(1);
        let fp = serde_json::json!({"severity":"warning","title":"fp-title","detail":"d","file":"f0.rs","line":3});
        let crit = serde_json::json!({"severity":"critical","title":"crit-title","detail":"d","file":"f0.rs","line":5});
        let consult = FakeConsult::new(vec![Ok(real_report([
            serde_json::json!([fp.clone()]),
            serde_json::json!([fp]),
            serde_json::json!([crit]),
        ])
        .await)]);
        let mut client = MockDecisionClient::new();
        client.expect_decide().returning(|r| {
            let line = r.state()["finding"]["line"].as_u64().unwrap_or(0);
            let choice = if line == 3 {
                "false_positive"
            } else {
                "real_defect"
            };
            answer(r.model().unwrap_or(""), choice, 0.95)
        });
        let mut out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        // render_text's own redaction: a raw credential injected after the pipeline.
        out.summary.findings[0].0.details =
            vec!["see https://user:hunter2@example.invalid/x".into()];
        let text = render_text(
            out.consolidated.as_ref().expect("consolidated"),
            &out.summary,
        );
        // One question per finding here: each has one (title, detail) pair.
        assert_eq!(out.summary.gate_questions, 2);
        assert!(text.contains("gate questions: 2"), "{text}");
        let (main, apart) = text
            .split_once("## Filtered by the gate")
            .expect("a filtered section");
        assert!(!main.contains("fp-title"), "{text}");
        assert!(apart.contains("fp-title"), "{text}");
        let key = main
            .split_once("## Key Findings")
            .expect("key findings")
            .1
            .split_once("\n## ")
            .expect("next section")
            .0;
        assert!(
            key.contains("[WARNING]") && key.contains("crit-title"),
            "{key}"
        );
        let findings = main.split_once("## Findings").expect("findings").1;
        assert!(
            findings.contains("[warning] f0.rs:5 crit-title (single_mage)"),
            "{findings}"
        );
        assert!(
            !text.contains("CRITICAL") && !text.contains("[critical]"),
            "{text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
    }

    #[tokio::test]
    async fn the_pipeline_runs_validation_then_consolidation_then_gate_then_agreement() {
        // RV-S33. f0.rs has 26 lines: line 999 is out of range; line 3 is seen by two mages and is a false
        // positive for the gate; line 5 is a critical seen by Caspar alone and is real.
        let dir = workspace(1);
        let a = serde_json::json!({"severity":"warning","title":"a","detail":"d","file":"f0.rs","line":999});
        let b = serde_json::json!({"severity":"warning","title":"b","detail":"d","file":"f0.rs","line":3});
        let c = serde_json::json!({"severity":"critical","title":"c","detail":"d","file":"f0.rs","line":5});
        let consult = FakeConsult::new(vec![Ok(real_report([
            serde_json::json!([a, b.clone()]),
            serde_json::json!([b]),
            serde_json::json!([c]),
        ])
        .await)]);
        let mut client = MockDecisionClient::new();
        client.expect_decide().times(2).returning(|r| {
            let line = r.state()["finding"]["line"].as_u64().unwrap_or(0);
            answer(
                r.model().unwrap_or(""),
                if line == 3 {
                    "false_positive"
                } else {
                    "real_defect"
                },
                0.95,
            )
        });
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        let s = &out.summary;
        assert_eq!(
            s.discarded
                .iter()
                .map(|d| d.cause.clone())
                .collect::<Vec<_>>(),
            vec![DiscardCause::LineOutOfRange]
        );
        assert_eq!(s.filtered.len(), 1);
        assert_eq!(s.filtered[0].0.line, Some(3));
        let kept: Vec<_> = s
            .findings
            .iter()
            .filter(|(f, _)| f.line == Some(5))
            .collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0.reported_severity, Severity::Critical);
        assert_eq!(
            kept[0].0.severity,
            Severity::Warning,
            "lowered AFTER the gate judged it as critical"
        );
        assert_eq!(kept[0].0.marks, vec![FindingMark::SingleMage]);
        // [CP2 MS2 S2a2 loop 10] run_review fills the count maps itself (the shape test builds them by hand).
        assert_eq!(
            s.gate_states,
            [("filtered".to_string(), 1), ("kept".to_string(), 1)]
                .into_iter()
                .collect()
        );
        assert_eq!(s.per_model.get("clef:cloud"), Some(&2));
    }

    #[tokio::test]
    async fn the_gate_is_charged_from_the_first_trio_call_not_from_when_the_clock_was_created() {
        // [CP2 MS2 S2a2 loop 10] N-6: the clock already reads past --timeout when batch 0 starts (a long preparation);
        // t0-relative, the gate still has the whole budget and judges the finding. Absolute, it would be timeout.
        let dir = workspace(1);
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"f0.rs","line":3}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let clock = FakeClock::default();
        clock.set(20_000);
        let out = review(
            dir.path(),
            Some(10_000),
            &consult,
            &gate_always("real_defect"),
            &clock,
        )
        .await
        .expect("runs");
        assert_eq!(out.summary.findings.len(), 1);
        assert_eq!(out.summary.findings[0].1.state, GateState::Kept);
    }

    #[tokio::test]
    async fn a_critical_whose_only_failure_is_an_unknown_symbol_reaches_the_gate_marked() {
        // [CP2 MS2 S1b loop 13, user 2026-10-19] RV-S35 end to end: RV09 never removes a critical on the symbol alone.
        let dir = workspace(1);
        let c = serde_json::json!({"severity":"critical","title":"`ghost_symbol` leaks","detail":"d","file":"f0.rs","line":5});
        let w = serde_json::json!({"severity":"warning","title":"`ghost_symbol` again","detail":"d","file":"f0.rs","line":6});
        let consult = FakeConsult::new(vec![Ok(real_report([
            serde_json::json!([c.clone(), w]),
            serde_json::json!([c]),
            serde_json::json!([]),
        ])
        .await)]);
        let mut client = MockDecisionClient::new();
        client
            .expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.95));
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        let s = &out.summary;
        assert_eq!(
            s.discarded
                .iter()
                .map(|d| d.finding.title.as_str())
                .collect::<Vec<_>>(),
            vec!["`ghost_symbol` again"]
        );
        assert_eq!(
            s.symbol_unverified.len(),
            2,
            "one per mage that reported it"
        );
        let crit: Vec<_> = s
            .findings
            .iter()
            .filter(|(f, _)| f.line == Some(5))
            .collect();
        assert_eq!(crit.len(), 1);
        assert_eq!(crit[0].1.cause, Some("critical never filtered"));
    }

    #[tokio::test]
    async fn a_clock_that_runs_out_mid_review_consolidates_the_finished_batches() {
        let dir = workspace(4);
        let consult = FakeConsult::new(vec![
            Ok(real_report(empty()).await),
            Ok(real_report(empty()).await),
            Err(ConsultFailure::ClockExpired),
            Ok(real_report(empty()).await),
        ]);
        let out = review(
            dir.path(),
            Some(100_000),
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("a partial review is still a review");
        assert_eq!(
            (out.summary.batches_planned, out.summary.batches_completed),
            (4, 2)
        );
        assert_eq!(consult.calls(), 3, "no batch runs after the clock expired");
        assert!(out.consolidated.is_some());
        // [CP2 MS2 S2 loop 5] RV-S22: the cut batches are declared, not implied.
        assert_eq!(
            out.summary.failed_batches,
            vec![
                (2, "clock expired".to_string()),
                (3, "clock expired".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn preparation_time_is_not_charged_to_the_timeout() {
        // [CP2 MS2 S2 loop 5] N-6: the clock already reads 90 000 s when the first batch starts (a slow preparation).
        let dir = workspace(1);
        let clock = FakeClock::default();
        clock.set(90_000);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        review(
            dir.path(),
            Some(100_000),
            &consult,
            &gate_always("real_defect"),
            &clock,
        )
        .await
        .expect("runs");
        assert_eq!(
            *consult.clocks.lock().expect("lock"),
            vec![Some(Duration::from_secs(100_000))]
        );
    }

    #[tokio::test]
    async fn a_non_canonical_reported_path_dedups_and_reaches_the_gate() {
        // [CP2 MS2 S2 loop 5] RV09 → RV12/gate: `./f0.rs` and `f0.rs` are one finding, and the gate gets its code.
        let dir = workspace(1);
        let a = serde_json::json!([{"severity":"warning","title":"a","detail":"d","file":"./f0.rs","line":3}]);
        let b = serde_json::json!([{"severity":"warning","title":"b","detail":"d","file":"f0.rs","line":3}]);
        let consult = FakeConsult::new(vec![Ok(real_report([a, b, serde_json::json!([])]).await)]);
        let mut client = MockDecisionClient::new();
        client
            .expect_decide()
            .times(1)
            .withf(|r| {
                r.state()["code"]
                    .as_str()
                    .is_some_and(|c| c.contains("3: "))
            })
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.95));
        let out = review(dir.path(), None, &consult, &client, &FakeClock::default())
            .await
            .expect("runs");
        assert_eq!(out.summary.findings.len(), 1);
        let (f, v) = &out.summary.findings[0];
        assert_eq!(
            (f.file.as_deref(), f.sources.len(), v.state),
            (Some("f0.rs"), 2, GateState::Kept)
        );
    }

    #[tokio::test(start_paused = true)] // [CP2 MS2 S2a loop 6] virtual time: the cutoff is under test, not a real wait
    async fn an_elapsed_wall_clock_is_clock_expired_and_an_analyze_error_is_other() {
        // [CP2 MS2 S2 loop 5] the bin's mapping, in the lib: only the batch wall clock cuts the review.
        let slow =
            run_with_wall_clock(Some(Duration::from_millis(10)), std::future::pending()).await;
        assert_eq!(slow.err(), Some(ConsultFailure::ClockExpired));
        let failed = run_with_wall_clock(Some(Duration::from_secs(60)), async {
            Err("seat timed out".to_string())
        })
        .await;
        assert_eq!(
            failed.err(),
            Some(ConsultFailure::Other("seat timed out".into()))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_elapsed_wall_clock_drops_the_batch_future() {
        // [CP2 MS2 S2a loop 6] cancellation semantics: the batch future (and the requests it owns) is dropped, not left running.
        struct Dropped(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = Dropped(std::sync::Arc::clone(&flag));
        let fut = async move {
            let _g = guard;
            std::future::pending::<Result<MagiReport, String>>().await
        };
        assert_eq!(
            run_with_wall_clock(Some(Duration::from_secs(5)), fut)
                .await
                .err(),
            Some(ConsultFailure::ClockExpired)
        );
        assert!(
            flag.load(std::sync::atomic::Ordering::SeqCst),
            "the batch future was not dropped"
        );
    }

    #[tokio::test]
    async fn a_budget_too_small_for_the_batches_fails_before_any_trio_call() {
        // [CP2 MS2 S2a loop 6] RV-S7 at the layer that owns §5.0 step 4: four batches in 60 s, and no consult runs.
        let dir = workspace(4);
        let consult = FakeConsult::new(Vec::new());
        let r = review(
            dir.path(),
            Some(60),
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await;
        assert!(
            matches!(r, Err(ReviewError::Budget { batches: 4, .. })),
            "a Budget error was expected"
        );
        assert_eq!(consult.calls(), 0);
    }

    #[tokio::test]
    async fn judged_fraction_counts_an_unjudged_finding_against_the_judged_ones() {
        // [CP2 MS2 S2a loop 6] one kept, one without a line (unfiltered/no location, never judged): 1 of 2.
        let dir = workspace(1);
        let f = serde_json::json!([{"severity":"warning","title":"a","detail":"d","file":"f0.rs","line":3},
                                   {"severity":"warning","title":"b","detail":"d","file":"f0.rs"}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert_eq!(out.summary.findings.len(), 2);
        assert_eq!(out.summary.judged_fraction, 0.5);
    }

    #[tokio::test]
    async fn findings_the_gate_could_not_reach_in_time_pass_as_timeout() {
        let dir = workspace(1);
        let f = serde_json::json!([{"severity":"warning","title":"x","detail":"d","file":"f0.rs","line":3}]);
        let clock = std::sync::Arc::new(FakeClock::default());
        let mut consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        consult.advance_to = Some((std::sync::Arc::clone(&clock), 100_000)); // the trio spends the whole total
        let mut client = MockDecisionClient::new();
        client.expect_decide().times(0);
        let out = review(dir.path(), Some(100_000), &consult, &client, clock.as_ref())
            .await
            .expect("runs");
        assert!(!out.summary.findings.is_empty());
        assert!(out
            .summary
            .findings
            .iter()
            .all(|(_, v)| v.state == GateState::Unfiltered && v.cause == Some("timeout")));
    }

    #[tokio::test]
    async fn batch_numbers_are_one_based_and_name_the_real_batch_everywhere() {
        // Batch 1 fails, batch 2 is paid and cannot reach consensus alone: the paid report is
        // "batch 2", the failure is "batch 1", in the text and in the JSON alike.
        let dir = workspace(2);
        let mut lonely = serde_json::to_value(real_report(empty()).await).expect("serializes");
        lonely["agents"].as_array_mut().expect("agents").truncate(1);
        let lonely: MagiReport = serde_json::from_value(lonely).expect("still a MagiReport");
        let consult = FakeConsult::new(vec![Err(ConsultFailure::Other("down".into())), Ok(lonely)]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("the paid batch comes back");
        let text = render_batch_reports(&out.batch_reports);
        assert!(
            text.contains("## [batch 2]") && !text.contains("## [batch 1]"),
            "{text}"
        );
        let v = crate::review::report::summary_json(&out.summary);
        assert_eq!(v["failed_batches"][0][0], 1, "{v}");
        assert_eq!(v["batches"][0]["index"], 2, "{v}");
    }

    #[tokio::test]
    async fn a_discarded_finding_names_its_batch_one_based_in_the_json() {
        let dir = workspace(2);
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"f1.rs","line":3}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([f.clone(), f, serde_json::json!([])]).await),
            Err(ConsultFailure::Other("seat failed".into())),
        ]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("reports");
        let v = crate::review::report::summary_json(&out.summary);
        assert_eq!(v["discarded"][0]["batch"], 1, "{v}");
        assert_eq!(v["failed_batches"][0][0], 2, "{v}");
        let text = render_text(
            out.consolidated.as_ref().expect("consolidated"),
            &out.summary,
        );
        assert!(
            text.contains("batch 2 did not complete: seat failed"),
            "{text}"
        );
        assert!(text.contains("(batch 1, "), "{text}");
    }

    #[tokio::test]
    async fn a_failed_consolidation_returns_the_paid_batch_reports() {
        // A report that kept a single agent cannot reach consensus (min_agents = 2): consolidation fails.
        // [CP2 MS2 S2 loop 4] the paid report carries a credential a mage echoed: the fallback path must redact it too.
        let dir = workspace(2);
        let cred = serde_json::json!([{"severity":"warning","title":"t","detail":"see https://user:hunter2@example.invalid/x","file":"f0.rs","line":1}]);
        let mut lonely = serde_json::to_value(
            real_report([cred, serde_json::json!([]), serde_json::json!([])]).await,
        )
        .expect("serializes");
        let agents = lonely["agents"].as_array_mut().expect("agents");
        agents.truncate(1);
        let lonely: MagiReport = serde_json::from_value(lonely).expect("still a MagiReport");
        let consult = FakeConsult::new(vec![Ok(lonely.clone()), Ok(lonely)]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("the paid batches come back");
        assert!(out.consolidated.is_none());
        assert_eq!(out.batch_reports.len(), 2);
        // [CP2 MS2 S2 loop 1] RV-S21's user-visible half: the typed error survives and the paid reports render.
        assert!(
            out.consolidation_error
                .as_deref()
                .is_some_and(|e| e.contains("consolidation failed")),
            "{:?}",
            out.consolidation_error
        );
        let text = render_batch_reports(&out.batch_reports);
        assert!(
            text.contains("## [batch 1]") && text.contains("## [batch 2]"),
            "{text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
    }

    #[tokio::test]
    async fn a_finding_on_a_later_batchs_file_is_kept_because_validation_sees_the_union() {
        // [CP2 MS2 S2 loop 1] U24 at the pipeline: batch 0 (f0.rs) talks about f1.rs, which batch 1 reviewed and finished.
        let dir = workspace(2);
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":"f1.rs","line":3}]);
        let consult = FakeConsult::new(vec![
            Ok(real_report([f.clone(), f, serde_json::json!([])]).await),
            Ok(real_report(empty()).await),
        ]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert!(
            out.summary.discarded.is_empty(),
            "{:?}",
            out.summary.discarded
        );
        assert_eq!(out.summary.findings.len(), 1);
        let prompts = consult.prompts.lock().expect("lock");
        assert!(
            prompts[0].contains("### f0.rs (rust)\n") && prompts[1].contains("### f1.rs (rust)\n"),
            "RV03 header"
        );
    }

    #[tokio::test]
    async fn with_a_timeout_each_batch_gets_what_is_left_split_over_the_pending_ones() {
        // [CP2 MS2 S2 loop 1] RV06 at the pipeline: 100 000 s, 2 batches → 50 000 s; the trio spends 60 000 → 40 000 s for the last.
        let dir = workspace(2);
        let clock = std::sync::Arc::new(FakeClock::default());
        let mut consult = FakeConsult::new(vec![
            Ok(real_report(empty()).await),
            Ok(real_report(empty()).await),
        ]);
        consult.advance_to = Some((std::sync::Arc::clone(&clock), 60_000));
        review(
            dir.path(),
            Some(100_000),
            &consult,
            &gate_always("real_defect"),
            clock.as_ref(),
        )
        .await
        .expect("runs");
        assert_eq!(
            *consult.clocks.lock().expect("lock"),
            vec![
                Some(Duration::from_secs(50_000)),
                Some(Duration::from_secs(40_000))
            ]
        );
    }

    #[tokio::test]
    async fn an_overshot_clock_never_underflows_and_the_gate_reports_timeout() {
        // [CP2 MS2 S2 loop 1] elapsed > total (the per-batch floor can overshoot): saturating, never a panic or a huge budget.
        let dir = workspace(1);
        let f = serde_json::json!([{"severity":"warning","title":"x","detail":"d","file":"f0.rs","line":3}]);
        let clock = std::sync::Arc::new(FakeClock::default());
        let mut consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        consult.advance_to = Some((std::sync::Arc::clone(&clock), 150_000)); // past the 100 000 s total
        let mut client = MockDecisionClient::new();
        client.expect_decide().times(0);
        let out = review(dir.path(), Some(100_000), &consult, &client, clock.as_ref())
            .await
            .expect("runs");
        assert!(out
            .summary
            .findings
            .iter()
            .all(|(_, v)| v.cause == Some("timeout")));
        assert_eq!(out.summary.judged_fraction, 0.0);
    }

    #[tokio::test]
    async fn a_finding_from_a_batch_where_one_mage_answered_keeps_its_severity_as_partial_trio() {
        // [CP2 MS2 S2 loop 2] Fault injection: magi-core never returns a one-agent report today (min_agents = 2, REQ-A15; such a
        // batch is a failed batch). The test pins RV11's rule end to end in case that ever changes; RV-S27 is unreachable today.
        // [CP2 MS2 S2 loop 1] RV-S27 end to end: batch 1 kept only Melchior (the others failed), who reported a critical.
        let dir = workspace(2);
        let crit = serde_json::json!([{"severity":"critical","title":"c","detail":"d","file":"f1.rs","line":3}]);
        let mut partial = serde_json::to_value(
            real_report([crit, serde_json::json!([]), serde_json::json!([])]).await,
        )
        .expect("serializes");
        partial["agents"]
            .as_array_mut()
            .expect("agents")
            .truncate(1); // Melchior is first
        let partial: MagiReport = serde_json::from_value(partial).expect("still a MagiReport");
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await), Ok(partial)]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        let (f, _) = out
            .summary
            .findings
            .iter()
            .find(|(f, _)| f.line == Some(3))
            .expect("kept");
        assert_eq!(
            (f.severity, f.marks.clone()),
            (Severity::Critical, vec![FindingMark::PartialTrio])
        );
        assert_eq!(out.summary.judged_fraction, 1.0);
    }

    #[test]
    fn a_verbatim_prefix_is_dropped_from_a_drive_and_from_a_unc_path() {
        // Windows canonicalizes to `\\?\C:\..` and `\\?\UNC\srv\share\..`; a mage reports the
        // forms an editor shows, `C:\..` and `\\srv\share\..`.
        let plain = |p: &str| {
            without_verbatim_prefix(Path::new(p))
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(plain(r"\\?\C:\w"), r"C:\w");
        assert_eq!(plain(r"\\?\UNC\srv\share"), r"\\srv\share");
        assert_eq!(plain("/w"), "/w");
    }

    #[test]
    fn a_unc_location_under_a_verbatim_unc_workspace_is_reviewed() {
        // The workspace the validator gets and the location a mage writes name the same file.
        let workspace = without_verbatim_prefix(Path::new(r"\\?\UNC\srv\share\w"));
        let mut reviewed = crate::review::index::Reviewed::default();
        reviewed.add_whole_file("f0.rs", 10);
        let index = crate::review::index::LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![crate::review::test_support::out(
            "melchior",
            "approve",
            0.9,
            serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":r"\\srv\share\w\f0.rs","line":3}]),
        )];
        let v = validate_batch(&mut agents, 0, &reviewed, &index, &workspace);
        assert!(v.discarded.is_empty(), "{:?}", v.discarded);
        assert_eq!(agents[0].findings[0].file.as_deref(), Some("f0.rs"));
    }

    #[tokio::test]
    async fn an_absolute_location_under_a_verbatim_workspace_is_kept_and_made_relative() {
        // The workspace may arrive canonical (a verbatim path on Windows) while the mage writes the
        // path an editor shows: the location must still fall in what was reviewed.
        let dir = workspace(1);
        let canonical = std::fs::canonicalize(dir.path()).expect("canonical");
        // Built by hand, not with the helper under test: a mutated helper must not move both sides.
        let text = canonical.to_string_lossy().into_owned();
        let reported = format!("{}/f0.rs", text.strip_prefix(r"\\?\").unwrap_or(&text));
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"d","file":reported,"line":3}]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let out = review(
            &canonical,
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert!(
            out.summary.discarded.is_empty(),
            "{:?}",
            out.summary.discarded
        );
        assert_eq!(out.summary.findings.len(), 1);
        assert_eq!(out.summary.findings[0].0.file.as_deref(), Some("f0.rs"));
    }

    #[tokio::test]
    async fn a_ceiling_below_the_configured_agent_timeout_is_declared() {
        // [CP2 MS2 S2 loop 2] RV06: 10 000 s over one batch derives a ceiling (≈1 388 s) below agent_timeout_secs (2 335).
        let dir = workspace(1);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        let out = review(
            dir.path(),
            Some(10_000),
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert!(out.summary.ceiling_below_configured);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        let out = review(
            dir.path(),
            None,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert!(
            !out.summary.ceiling_below_configured,
            "no --timeout, no derived ceiling"
        );
    }
}
