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
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf, Prefix, MAIN_SEPARATOR_STR};
use std::time::{Duration, Instant};

use magi_core::reporting::MagiReport;
use sha2::{Digest, Sha256};

use super::batching::{file_header, plan_batches, BatchItem, ReviewBatch};
use super::clock::{batch_wall_clock_secs, pre_budget};
use super::consolidate::{
    apply_agreement, consolidate, BatchOutcome, Consolidated, ConsolidatedFinding,
};
use super::error::{BudgetLimit, ReviewError};
use super::filter::{filter_files, Filtered, SkipCause, Skipped};
use super::gate::{Gate, GateState, GateVerdict};
use super::index::{LexicalIndex, Reviewed};
use super::input::{renamed_from, resolve_input, ReviewFile, ReviewInput, SourceKind};
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
    /// The calling run's end, if it has one: the total is then at most what is left of it when
    /// the preparation ends, so the review delivers what it finished before the run is cut.
    pub deadline: Option<Instant>,
    /// `[magi].max_rotations`, resolved.
    pub max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    pub retry_disabled: bool,
    /// `[magi].agent_timeout_secs`, resolved: what the derived ceiling is compared with.
    pub agent_timeout_secs: u64,
    /// The batch count the caller approved, if any: a review that now splits otherwise fails
    /// with [`ReviewError::PlanChanged`] before any trio call.
    pub approved_batches: Option<usize>,
    /// The content digest the caller approved ([`PlanFingerprint::digest`]), if any: a review
    /// whose content moved since fails with [`ReviewError::ContentChanged`] before any trio call,
    /// even when its batch count did not change.
    pub approved_digest: Option<[u8; 32]>,
    /// When the caller started preparing this review, if before this call: the preparation it
    /// did first (its own git reads, the gate) is part of `prep_ms`. `None` starts it here.
    pub prep_started: Option<Instant>,
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
/// 4. Build the lexical index; then, with an explicit `--timeout` or a caller's deadline, check
///    the budget ([`pre_budget`]) before any trio call. The total is the `--timeout`, at most what
///    is left of the deadline when the preparation ends. Steps 1-4 are the preparation, measured
///    apart and never charged to the `--timeout`.
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
    // MAGI FIX (S6 loop 3): `prep_ms` counts from the caller's start, when it gave one.
    let prep = req.prep_started.unwrap_or_else(Instant::now);
    let mut summary = ReviewSummary::default();
    let resolved = resolve_input(&req.input, req.workspace, deps.git, deps.guard)?;
    let inner = content_source(resolved.source_kind, req.workspace, deps.git);
    let source = CachedSource::new(inner.as_ref());
    let explicit = matches!(req.input, ReviewInput::File(_));
    let (filtered, window, batches) = with_not_utf8(
        filter_and_plan(
            resolved.files,
            &source,
            explicit,
            deps.registry,
            deps.is_ignored,
            deps.pool,
        ),
        &resolved.not_utf8,
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
    if let Some(approved) = req.approved_digest {
        if content_digest(&filtered.kept, &source)? != approved {
            return Err(ReviewError::ContentChanged);
        }
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
    let bound = review_total(req.timeout_secs, req.deadline);
    let total = bound.map(|(secs, _)| secs);
    if let Some((total, limit)) = bound {
        let ceiling = pre_budget(total, planned, req.max_rotations, req.retry_disabled)
            .map_err(|e| with_budget_limit(e, limit))?;
        summary.ceiling_below_configured = ceiling < req.agent_timeout_secs;
    }
    summary.prep_ms = millis(prep.elapsed());

    let clock = ReviewTimer {
        clock: deps.clock,
        t0: deps.clock.elapsed_secs(),
        total,
    };
    let finished = run_batches(&batches, deps.consult, &clock, &mut summary).await;

    // MAGI FIX (S6 loop 4): `planned` is never 0 here: a plan without a batch already failed in
    // `filter_and_plan` with `NothingToReview` naming the files left out (RV20), before any spend.
    // So an empty `finished` always means every planned batch failed or was cut.
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
    let reviewed = reviewed_of(&batches, &completed, &source);
    let workspace = without_verbatim_prefix(req.workspace);
    let old_paths = renamed_old_paths(&filtered.kept);
    let renames: HashMap<String, String> = old_paths
        .iter()
        .map(|(new_path, old)| (old.clone(), new_path.clone()))
        .collect();
    let mut outcomes = Vec::with_capacity(finished.len());
    for f in &finished {
        let mut agents = f.report.agents.clone();
        let answered = agents.iter().map(|a| a.agent).collect();
        let validation = validate_batch(
            &mut agents,
            f.batch,
            &reviewed,
            &renames,
            &index,
            &workspace,
        );
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
        let code = gate_code(
            finding.file.as_deref(),
            finding.line,
            &faces,
            &old_paths,
            &source,
        );
        verdicts.push(
            deps.gate
                .judge(finding, code.as_deref(), clock.remaining())
                .await,
        );
    }
    summary.per_model = deps.gate.per_model().clone();
    summary.gate_questions = deps.gate.questions();
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
/// The batch count and the digest of what the review would send ([`PlanFingerprint`]).
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
) -> Result<PlanFingerprint, ReviewError> {
    let resolved = resolve_input(input, workspace, git, guard)?;
    let inner = content_source(resolved.source_kind, workspace, git);
    let source = CachedSource::new(inner.as_ref());
    let explicit = matches!(input, ReviewInput::File(_));
    let (filtered, _, batches) = with_not_utf8(
        filter_and_plan(
            resolved.files,
            &source,
            explicit,
            registry,
            is_ignored,
            pool,
        ),
        &resolved.not_utf8,
    )?;
    Ok(PlanFingerprint {
        batches: batches.len(),
        digest: content_digest(&filtered.kept, &source)?,
    })
}

/// SHA-256 over every kept file's path, diff and content, in input order, each part
/// length-prefixed so no two different reviews hash alike by moving bytes between parts.
///
/// The content comes from `source`, the same cached reads the batches are built from, so the
/// digest covers exactly what the review would send. O(bytes reviewed).
///
/// # Errors
///
/// The read errors of `source`.
// MAGI FIX (S6 loop 6): the approval pins the content, not only the batch count (D-c1).
fn content_digest(
    kept: &[ReviewFile],
    source: &dyn ContentSource,
) -> Result<[u8; 32], ReviewError> {
    let mut hash = Sha256::new();
    let mut part = |bytes: &[u8]| {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    };
    for file in kept {
        part(file.path.as_bytes());
        part(file.diff.as_deref().unwrap_or_default().as_bytes());
        part(&source.read(&file.path)?.unwrap_or_default());
    }
    Ok(hash.finalize().into())
}

/// What an approval saw, computed without any model call by [`plan_review`]: the batch count it
/// showed and a SHA-256 digest of what the review would send. [`run_review`] recomputes both
/// before the first trio call and refuses a review whose content moved in between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanFingerprint {
    /// The batches.
    pub batches: usize,
    /// SHA-256 over every kept file's path, diff and content, in input order.
    pub digest: [u8; 32],
}

/// `planned`, with every non-UTF-8 name of the input added to a `NothingToReview` error's
/// omissions; any other result unchanged.
fn with_not_utf8<T>(
    planned: Result<T, ReviewError>,
    not_utf8: &[String],
) -> Result<T, ReviewError> {
    // MAGI FIX (S6 loop 6): binaries plus non-UTF-8 names named only the binaries (RV20).
    planned.map_err(|e| match e {
        ReviewError::NothingToReview { mut skipped } => {
            skipped.extend(not_utf8.iter().map(|path| Skipped {
                path: display_path(path),
                cause: SkipCause::NotUtf8,
            }));
            ReviewError::NothingToReview { skipped }
        }
        other => other,
    })
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
    // MAGI FIX (S6 loop 4): when the kept files give no batch (all empty, or diffs without a
    // hunk), the error names what the filter left out, as the filter's own error does.
    let batches = plan_batches(
        &filtered.kept,
        source,
        registry,
        batch_limit_tokens(&window),
    )
    .map_err(|e| match e {
        ReviewError::NothingToReview { .. } => ReviewError::NothingToReview {
            skipped: filtered.skipped.clone(),
        },
        other => other,
    })?;
    Ok((filtered, window, batches))
}

/// The review's total in seconds, read when the preparation ends, with the bound that set it:
/// the explicit `--timeout`, at most what is left of the caller's deadline; `None` when there is
/// neither.
///
/// The deadline is an absolute [`Instant`] of the calling run, so what is left of it is read on
/// the real clock; the injected [`ReviewClock`] only measures the batches' progress from this
/// total on. A test double of that clock therefore never moves the deadline, only the batches.
fn review_total(
    timeout_secs: Option<u64>,
    deadline: Option<Instant>,
) -> Option<(u64, BudgetLimit)> {
    let left = deadline.map(|d| d.saturating_duration_since(Instant::now()).as_secs());
    match (timeout_secs, left) {
        (Some(t), Some(l)) if l < t => Some((l, BudgetLimit::RunDeadline)),
        (Some(t), _) => Some((t, BudgetLimit::Timeout)),
        (None, l) => l.map(|l| (l, BudgetLimit::RunDeadline)),
    }
}

/// `e` with its budget bound set to `limit` (any other error unchanged), so the message names the
/// bound the user can move.
// MAGI FIX (S6 loop 4): `pre_budget` knows only a total; the pipeline knows what set it.
fn with_budget_limit(mut e: ReviewError, limit: BudgetLimit) -> ReviewError {
    if let ReviewError::Budget { limit: bound, .. } = &mut e {
        *bound = limit;
    }
    e
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
/// `clock expired`. Every batch that ran gets its metrics, failed or not (a failed one counts no
/// tokens: its report never arrived); a batch cut before it started spent nothing and has none.
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
                summary.batches.push(spent(batch.index, started.elapsed()));
                summary
                    .failed_batches
                    .extend((batch.index..planned).map(|k| (k, CLOCK_EXPIRED.to_string())));
                break;
            }
            Err(ConsultFailure::Other(message)) => {
                summary.batches.push(spent(batch.index, started.elapsed()));
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
        // MAGI FIX (S6 loop 3): a verdict counts under every cause it carries.
        for cause in &verdict.causes {
            *summary.gate_causes.entry((*cause).to_string()).or_insert(0) += 1;
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

/// The metrics of a batch that ran and gave no report: its time, and no tokens.
fn spent(index: usize, took: Duration) -> BatchMetrics {
    BatchMetrics {
        index,
        input_tokens: 0,
        output_tokens: 0,
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

/// The path of `file`'s old side: its `rename from` path when its diff is a rename, else its
/// own path (RV21: the deleted lines of a rename live under the old path).
fn old_path(file: &ReviewFile) -> String {
    file.diff
        .as_deref()
        .and_then(renamed_from)
        .unwrap_or_else(|| file.path.clone())
}

/// The old path of every renamed file, by its new path. O(files).
fn renamed_old_paths(files: &[ReviewFile]) -> HashMap<String, String> {
    files
        .iter()
        .filter_map(|f| {
            let old = old_path(f);
            (old != f.path).then(|| (f.path.clone(), old))
        })
        .collect()
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
            sides.push(source.read_old(&old_path(file))?);
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
/// units. Never fails, because the batches are already paid: a unit-split file that cannot be
/// read again (unreachable with the cache, which already holds it) counts only its finished
/// units, with a WARN. O(items · log r).
fn reviewed_of(
    batches: &[ReviewBatch],
    completed: &BTreeSet<usize>,
    source: &dyn ContentSource,
) -> Reviewed {
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
        let whole = if all_done {
            match source.read(file) {
                Ok(bytes) => Some(bytes.map_or(0, |b| String::from_utf8_lossy(&b).lines().count())),
                Err(e) => {
                    let error = redact_foreign_text(&e.to_string());
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        file = file,
                        error = error.as_str(),
                        "a reviewed file could not be read again: only its units count as reviewed"
                    );
                    None
                }
            }
        } else {
            None
        };
        match whole {
            Some(lines) => reviewed.add_whole_file(file, u32::try_from(lines).unwrap_or(u32::MAX)),
            None => {
                for (first, last) in finished {
                    reviewed.add_lines(file, first, last);
                }
            }
        }
    }
    reviewed
}

/// The faces each file showed in the completed batches: both ranges of each hunk, each whole
/// file, and each unit-split file whole only when every batch holding one of its units
/// completed (as [`reviewed_of`] decides), else only its finished units, as new-side ranges.
/// O(items).
///
/// Within a face the gate's state is RV10's ±40 lines around the finding's line, read from the
/// revision (RV21): as for a hunk, that window may reach lines no batch reviewed. They are lines
/// of a file that passed RV22 whole (a file with a secret is never sent) and the state itself is
/// scanned for a secret before any call.
// MAGI FIX (S6 loop 5): a unit-split file with an unfinished unit is not whole for the gate.
fn faces_of<'b>(
    batches: &'b [ReviewBatch],
    completed: &BTreeSet<usize>,
) -> HashMap<&'b str, Faces> {
    let mut faces: HashMap<&str, Faces> = HashMap::new();
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
                    let f = faces.entry(file.as_str()).or_default();
                    f.new.push((*first_line, *last_line));
                    f.old.push((*old_first_line, *old_last_line));
                }
                BatchItem::WholeFile { file, .. } if done => {
                    faces.entry(file.as_str()).or_default().whole = true;
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
        if finished.is_empty() {
            continue;
        }
        let f = faces.entry(file).or_default();
        if all_done {
            f.whole = true;
        } else {
            f.new.extend(finished);
        }
    }
    faces
}

/// The code the gate sees for a finding (RV10 + RV21): the new side when the line lies in a new
/// range or the file was shown whole, else the old side when it lies in an old range, numbered
/// as that side; the old side of a renamed file is read under its old path (`old_paths`).
/// `None` (the gate's `no location`) without a line, without a face that holds
/// it, or when the read fails, with a WARN; with the cache that read is unreachable, because
/// every face of a reviewed file was read during the preparation.
fn gate_code(
    file: Option<&str>,
    line: Option<u32>,
    faces: &HashMap<&str, Faces>,
    old_paths: &HashMap<String, String>,
    source: &dyn ContentSource,
) -> Option<String> {
    let (file, line) = (file?, line?);
    let face = faces.get(file)?;
    let holds = |ranges: &[(u32, u32)]| ranges.iter().any(|&(a, b)| a <= line && line <= b);
    let read = if face.whole || holds(&face.new) {
        source.read(file)
    } else if holds(&face.old) {
        source.read_old(old_paths.get(file).map_or(file, String::as_str))
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
                file = file,
                line = line,
                error = error.as_str(),
                "the gate could not read the code of a finding"
            );
            None
        }
    }
}

/// `path` without Windows' verbatim prefix, so a reported absolute location compares with the
/// workspace as an editor writes it: `\\?\C:\w` becomes `C:\w` and `\\?\UNC\srv\share\w` becomes
/// `\\srv\share\w`. Any other path (a volume GUID path, a device path, any path off Windows) is
/// returned as it is. It works on the path's components, so a component that is not Unicode
/// keeps every unit it had. O(components).
fn without_verbatim_prefix(path: &Path) -> PathBuf {
    let mut components = path.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return path.to_path_buf();
    };
    let mut head = OsString::new();
    match prefix.kind() {
        Prefix::VerbatimDisk(letter) => head.push(format!("{}:", char::from(letter))),
        Prefix::VerbatimUNC(server, share) => {
            head.push(UNC_PREFIX);
            head.push(server);
            head.push(MAIN_SEPARATOR_STR);
            head.push(share);
        }
        _ => return path.to_path_buf(),
    }
    let mut out = PathBuf::from(head);
    out.extend(components);
    out
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

    #[test]
    fn nothing_left_after_batching_names_what_the_filter_left_out() {
        // [MAGI S6 loop 4, B1] the filter kept an empty file and left out a binary one; the
        // batching then found nothing and its error named no file at all.
        use crate::review::filter::SkipCause;
        use crate::review::test_support::MapSource;
        let files = vec![
            ReviewFile {
                path: "a.bin".into(),
                diff: None,
            },
            ReviewFile {
                path: "empty.rs".into(),
                diff: None,
            },
        ];
        let source = MapSource::new(&[("a.bin", "x\0y"), ("empty.rs", "")], &[]);
        let pool = [("m".to_string(), Some(POOL_WINDOW))];
        match filter_and_plan(
            files,
            &source,
            false,
            &LanguageRegistry::new(),
            &|_| false,
            &pool,
        ) {
            Err(ReviewError::NothingToReview { skipped }) => {
                assert_eq!(skipped.len(), 1, "{skipped:?}");
                assert_eq!(
                    (skipped[0].path.as_str(), skipped[0].cause),
                    ("a.bin", SkipCause::Binary)
                );
            }
            other => panic!("expected NothingToReview, got {other:?}"),
        }
    }

    #[test]
    fn nothing_to_review_names_the_non_utf8_names_beside_the_binaries() {
        // [MAGI S6 loop 6, A] mixed case: a binary file and a non-UTF-8 name, nothing reviewable.
        // The filter's error named the binary only; the non-UTF-8 name was dropped.
        use crate::review::filter::SkipCause;
        use crate::review::test_support::MapSource;
        let files = vec![ReviewFile {
            path: "a.bin".into(),
            diff: None,
        }];
        let source = MapSource::new(&[("a.bin", "x\0y")], &[]);
        let pool = [("m".to_string(), Some(POOL_WINDOW))];
        let planned = filter_and_plan(
            files,
            &source,
            false,
            &LanguageRegistry::new(),
            &|_| false,
            &pool,
        );
        match with_not_utf8(planned, &["bad\u{fffd}.rs".to_string()]) {
            Err(ReviewError::NothingToReview { skipped }) => {
                let got: Vec<_> = skipped.iter().map(|s| (s.path.as_str(), s.cause)).collect();
                assert_eq!(
                    got,
                    vec![
                        ("a.bin", SkipCause::Binary),
                        ("bad\u{fffd}.rs", SkipCause::NotUtf8)
                    ]
                );
            }
            other => panic!("expected NothingToReview, got {other:?}"),
        }
    }

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
        review_until(dir, timeout, None, Duration::ZERO, consult, client, clock).await
    }

    /// [`review`] with a deadline, and a guard that takes `prep` per path (a slow preparation).
    async fn review_until(
        dir: &Path,
        timeout: Option<u64>,
        deadline: Option<Instant>,
        prep: Duration,
        consult: &FakeConsult,
        client: &MockDecisionClient,
        clock: &dyn ReviewClock,
    ) -> Result<ReviewOutcome, ReviewError> {
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> {
            std::thread::sleep(prep);
            Ok(p.to_path_buf())
        };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        run_review(
            ReviewRequest {
                input: ReviewInput::Directory(dir.to_path_buf()),
                workspace: dir,
                timeout_secs: timeout,
                deadline,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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

    /// Runs [`plan_review`] then [`run_review`] on `dir`, calling `between` in between, as the
    /// approval prompt and the run it approved would.
    async fn approve_then_run(
        dir: &Path,
        between: impl FnOnce(),
        consult: &FakeConsult,
        client: &MockDecisionClient,
    ) -> Result<ReviewOutcome, ReviewError> {
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        let input = ReviewInput::Directory(dir.to_path_buf());
        let plan =
            plan_review(&input, dir, &git, &guard, &|_| false, &registry, &pool).expect("plans");
        between();
        run_review(
            ReviewRequest {
                input,
                workspace: dir,
                timeout_secs: None,
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: Some(plan.batches),
                approved_digest: Some(plan.digest),
                prep_started: None,
            },
            ReviewDeps {
                git: &git,
                guard: &guard,
                is_ignored: &|_| false,
                registry: &registry,
                pool: &pool,
                consult,
                gate: Gate::new(client, chain(), 61_000),
                clock: &FakeClock::default(),
            },
        )
        .await
    }

    #[tokio::test]
    async fn content_that_moved_after_the_approval_is_refused_before_any_trio_call() {
        // [MAGI S6 loop 6, D-c1] same batch count, other bytes: what was approved is not what
        // would be sent, so nothing is sent.
        let dir = workspace(1);
        let consult = FakeConsult::new(vec![]);
        let client = MockDecisionClient::new();
        let path = dir.path().join("f0.rs");
        let r = approve_then_run(
            dir.path(),
            || {
                let text = std::fs::read_to_string(&path).expect("read");
                std::fs::write(&path, text.replace("_x = 1", "_y = 2")).expect("write");
            },
            &consult,
            &client,
        )
        .await;
        assert!(
            matches!(r, Err(ReviewError::ContentChanged)),
            "{:?}",
            r.err()
        );
        assert!(consult.prompts.lock().expect("lock").is_empty());
    }

    #[tokio::test]
    async fn content_unchanged_since_the_approval_runs() {
        let dir = workspace(1);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        let client = MockDecisionClient::new();
        let r = approve_then_run(dir.path(), || {}, &consult, &client).await;
        assert!(r.is_ok(), "{:?}", r.err());
    }

    fn empty() -> [serde_json::Value; 3] {
        [
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!([]),
        ]
    }

    #[tokio::test]
    async fn the_preparation_a_caller_did_first_counts_in_prep_ms() {
        // [MAGI §6 MS2 D loop 3] the runner's own preparation (its git reads, the gate) happens
        // before `run_review`; the start it passes is where `prep_ms` counts from.
        const EARLIER: Duration = Duration::from_secs(2);
        let dir = workspace(1);
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        let client = MockDecisionClient::new();
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Directory(dir.path().to_path_buf()),
                workspace: dir.path(),
                timeout_secs: None,
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: Instant::now().checked_sub(EARLIER),
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
        assert!(
            u128::from(out.summary.prep_ms) >= EARLIER.as_millis(),
            "{} ms",
            out.summary.prep_ms
        );
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
            (
                "ls-tree -r -z abc",
                Ok("100644 blob 0000000000000000000000000000000000000000\tgone.rs\0"),
            ),
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
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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
            (
                "ls-tree -r -z abc",
                Ok("100644 blob 0000000000000000000000000000000000000000\tgone.rs\0"),
            ),
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
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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

    #[test]
    fn every_cause_of_a_verdict_is_counted() {
        // [MAGI §6 MS2 C loop 3] a verdict with two causes counts under both.
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: magi_core::schema::Severity::Critical,
            reported_severity: magi_core::schema::Severity::Critical,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![magi_core::schema::Category::Other],
            sources: vec![magi_core::schema::AgentName::Melchior],
            marks: vec![],
            batches: vec![0],
        };
        let verdict = GateVerdict {
            state: GateState::LowConfidence,
            cause: Some("critical never filtered"),
            causes: vec!["critical never filtered", "not every claim judged"],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        };
        let mut summary = ReviewSummary::default();
        record_verdicts(&mut summary, &[finding], vec![verdict]);
        assert_eq!(
            (
                summary.gate_causes.get("critical never filtered"),
                summary.gate_causes.get("not every claim judged")
            ),
            (Some(&1), Some(&1)),
            "{:?}",
            summary.gate_causes
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
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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
    async fn a_batch_that_ran_and_failed_is_timed_and_counted_in_the_totals() {
        // [MAGI §6 MS2 D loop 2] RV13: a batch that spent its clock and gave no report still
        // spent it. Batch 0 fails after 30 ms, batch 1 runs out of its wall clock after 30 ms,
        // batches 2-3 never start (no time spent, no metrics).
        struct Slow(Mutex<VecDeque<ConsultFailure>>);
        #[async_trait::async_trait]
        impl BatchConsult for Slow {
            async fn consult(
                &self,
                _: &str,
                _: Option<Duration>,
            ) -> Result<MagiReport, ConsultFailure> {
                tokio::time::sleep(Duration::from_millis(30)).await;
                Err(self
                    .0
                    .lock()
                    .expect("lock")
                    .pop_front()
                    .unwrap_or(ConsultFailure::Other("no scripted reply".into())))
            }
        }
        let dir = workspace(4);
        let consult = Slow(Mutex::new(
            vec![
                ConsultFailure::Other("seat failed".into()),
                ConsultFailure::ClockExpired,
            ]
            .into(),
        ));
        let git = FakeGit::from(&[(
            "ls-files --others --ignored --exclude-standard --directory -z -- .",
            Ok(""),
        )]);
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW))];
        let client = gate_always("real_defect");
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Directory(dir.path().to_path_buf()),
                workspace: dir.path(),
                timeout_secs: Some(100_000),
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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
        let m = &out.summary.batches;
        assert_eq!(m.iter().map(|b| b.index).collect::<Vec<_>>(), vec![0, 1]);
        assert!(m.iter().all(|b| b.ms >= 30 && b.input_tokens == 0), "{m:?}");
        let v = crate::review::report::summary_json(&out.summary);
        assert!(v["batch_totals"]["ms"].as_u64().unwrap_or(0) >= 60, "{v}");
        assert_eq!(out.summary.failed_batches.len(), 4);
    }

    #[test]
    fn the_gate_read_failure_log_names_the_file_and_the_line() {
        // [MAGI §6 MS2 D loop 2] the WARN must say which finding lost its code.
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;
        struct Failing;
        impl ContentSource for Failing {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io(format!("{}: gone", path.len())))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        struct Capture(Arc<Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if let Ok(mut lines) = self.0.lock() {
                    lines.push(crate::logging::render::render_event(event));
                }
            }
        }
        let lines = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Capture(Arc::clone(&lines)));
        let faces = HashMap::from([(
            "src/a.rs",
            Faces {
                whole: true,
                ..Faces::default()
            },
        )]);
        let code = tracing::subscriber::with_default(subscriber, || {
            gate_code(Some("src/a.rs"), Some(7), &faces, &HashMap::new(), &Failing)
        });
        assert!(code.is_none());
        let lines = lines.lock().expect("lock");
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("file=src/a.rs") && lines[0].contains("line=7"),
            "{lines:?}"
        );
    }

    #[test]
    fn the_gate_sees_no_code_for_a_line_of_a_unit_whose_batch_did_not_complete() {
        // [MAGI S6 loop 5, D-c1] a unit-split file is whole for the gate only when every batch
        // holding one of its units completed, as `reviewed_of` decides; otherwise only its
        // finished units are faces, so a line of an unfinished unit gets no code.
        struct Text;
        impl ContentSource for Text {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(
                    b"x
"
                    .repeat(20),
                ))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let unit = |first_line, last_line| BatchItem::Unit {
            file: "a.rs".into(),
            name: "f".into(),
            text: String::new(),
            first_line,
            last_line,
        };
        let batch = |index, item| ReviewBatch {
            index,
            items: vec![item],
            tokens: 0,
            companion_tokens: 0,
        };
        let batches = vec![batch(0, unit(1, 10)), batch(1, unit(11, 20))];
        let old_paths = HashMap::new();
        let partial = faces_of(&batches, &BTreeSet::from([0]));
        let code = |faces: &HashMap<&str, Faces>, line| {
            gate_code(Some("a.rs"), Some(line), faces, &old_paths, &Text)
        };
        assert!(
            code(&partial, 5).is_some(),
            "a finished unit's line has its code"
        );
        assert!(
            code(&partial, 15).is_none(),
            "an unfinished unit's line has none"
        );
        let all = faces_of(&batches, &BTreeSet::from([0, 1]));
        assert!(
            code(&all, 15).is_some(),
            "every unit done: the file is whole"
        );
    }

    #[test]
    fn a_failed_read_after_the_batches_falls_back_to_the_finished_units() {
        // [MAGI §6 MS2 D loop 2] the batches are paid when `reviewed_of` reads a unit-split file:
        // a failing read must not throw them away. It registers only the units the trio saw.
        struct Failing;
        impl ContentSource for Failing {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io(format!("{path}: gone")))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let unit = |first_line, last_line| BatchItem::Unit {
            file: "a.rs".into(),
            name: "f".into(),
            text: String::new(),
            first_line,
            last_line,
        };
        let batches = vec![ReviewBatch {
            index: 0,
            items: vec![unit(1, 3), unit(6, 8)],
            tokens: 0,
            companion_tokens: 0,
        }];
        let reviewed = reviewed_of(&batches, &BTreeSet::from([0]), &Failing);
        assert!(reviewed.contains_line("a.rs", 2) && reviewed.contains_line("a.rs", 7));
        assert!(
            !reviewed.contains_line("a.rs", 5),
            "the gap between units is not claimed"
        );
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
    async fn a_deadline_bounds_the_batches_from_where_the_preparation_ended() {
        // [MAGI §6 MS2 D loop 1] the calling run's deadline is read when the batches start: a
        // slow preparation shrinks the review's total instead of pushing its end past the
        // deadline, where the caller's cancellation would drop every paid batch.
        let dir = workspace(1);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        let deadline = Instant::now() + Duration::from_secs(300);
        review_until(
            dir.path(),
            None,
            Some(deadline),
            Duration::from_millis(1_500),
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        let wall = consult.clocks.lock().expect("lock")[0].expect("a deadline is a wall clock");
        assert!(wall <= Duration::from_secs(299), "{wall:?}");
        assert!(wall >= Duration::from_secs(290), "{wall:?}");
    }

    #[tokio::test]
    async fn a_timeout_under_the_deadline_is_the_total() {
        let dir = workspace(1);
        let consult = FakeConsult::new(vec![Ok(real_report(empty()).await)]);
        review_until(
            dir.path(),
            Some(200),
            Some(Instant::now() + Duration::from_secs(1_000)),
            Duration::ZERO,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await
        .expect("runs");
        assert_eq!(
            consult.clocks.lock().expect("lock")[0],
            Some(Duration::from_secs(200))
        );
    }

    #[tokio::test]
    async fn a_deadline_too_close_for_the_batches_fails_before_any_trio_call() {
        let dir = workspace(4);
        let consult = FakeConsult::new(Vec::new());
        let r = review_until(
            dir.path(),
            None,
            Some(Instant::now() + Duration::from_secs(60)),
            Duration::ZERO,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await;
        assert!(
            matches!(r, Err(ReviewError::Budget { batches: 4, .. })),
            "{:?}",
            r.err()
        );
        assert_eq!(consult.calls(), 0);
    }

    #[tokio::test]
    async fn a_deadline_too_close_names_the_calling_run_not_a_timeout_flag() {
        // [MAGI S6 loop 4, D] the agent tool passes its run's deadline, not a `--timeout`: telling
        // the user to pass `--timeout` names a flag they never set.
        let dir = workspace(4);
        let consult = FakeConsult::new(Vec::new());
        let r = review_until(
            dir.path(),
            None,
            Some(Instant::now() + Duration::from_secs(60)),
            Duration::ZERO,
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await;
        let message = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(!message.contains("--timeout"), "{message}");
        assert!(message.contains("left of the calling run"), "{message}");
        let r = review(
            dir.path(),
            Some(60),
            &consult,
            &gate_always("real_defect"),
            &FakeClock::default(),
        )
        .await;
        let message = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("pass --timeout"), "{message}");
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
        // [MAGI §6 MS2 D loop 2] the failed batch ran too: it has its metrics, numbered 1.
        assert_eq!(v["batches"][0]["index"], 1, "{v}");
        assert_eq!(v["batches"][1]["index"], 2, "{v}");
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

    #[cfg(windows)]
    #[test]
    fn dropping_a_verbatim_prefix_keeps_a_path_that_is_not_unicode() {
        // [MAGI §6 MS2 D loop 2] a Windows path may hold an unpaired surrogate; the prefix goes,
        // every other unit stays as it was (no U+FFFD), and a volume GUID path is left alone.
        use std::ffi::OsString;
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let wide = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        let mut raw = wide(r"\\?\C:\w\");
        raw.push(0xD800);
        let got = without_verbatim_prefix(Path::new(&OsString::from_wide(&raw)));
        let mut want = wide(r"C:\w\");
        want.push(0xD800);
        assert_eq!(got.as_os_str().encode_wide().collect::<Vec<u16>>(), want);
        let volume = r"\\?\Volume{00000000-0000-0000-0000-000000000000}\w";
        assert_eq!(
            without_verbatim_prefix(Path::new(volume)),
            Path::new(volume)
        );
    }

    #[cfg(windows)]
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

    #[cfg(windows)]
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
        let v = validate_batch(
            &mut agents,
            0,
            &reviewed,
            &HashMap::new(),
            &index,
            &workspace,
        );
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

    /// Git as the command line runs it in `dir`, for the tests that need a real repository.
    struct PlainGit(PathBuf);
    impl GitCommand for PlainGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&self.0)
                .args(["-c", "core.autocrlf=false", "-c", "diff.noprefix=false"])
                .args(["-c", "diff.renames=true", "-c", "core.quotePath=false"])
                .args(args)
                .output()
                .map_err(|e| ReviewError::Git(e.to_string()))?;
            if out.status.success() {
                Ok(out.stdout)
            } else {
                Err(ReviewError::Git(
                    String::from_utf8_lossy(&out.stderr).trim().to_string(),
                ))
            }
        }
    }

    fn git_in(dir: &Path, args: &[&str]) {
        PlainGit(dir.to_path_buf())
            .run_bytes(args)
            .expect("git runs");
    }

    #[tokio::test]
    async fn a_renamed_and_edited_file_reads_its_old_side_under_its_old_path() {
        // [MAGI §6 MS2 D loop 2] RV21: the deleted lines of a rename live under the old path. The
        // lexical index (the symbol `f18`) and the gate's code (line 17) must both read them there.
        let (dir, git) = renamed_repo();
        let f = serde_json::json!([
            {"severity":"warning","title":"t","detail":"`f18` is never called","file":"new.rs","line":18},
            {"severity":"warning","title":"u","detail":"d","file":"new.rs","line":17}
        ]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let client = gate_always("real_defect");
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW * 10))];
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Uncommitted,
                workspace: dir.path(),
                timeout_secs: None,
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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
        assert!(
            out.summary.discarded.is_empty(),
            "the old side's symbol is indexed: {:?}",
            out.summary.discarded
        );
        let states: Vec<_> = out
            .summary
            .findings
            .iter()
            .map(|(f, v)| (f.line, v.state))
            .collect();
        assert!(
            states.contains(&(Some(17), GateState::Kept)),
            "the gate saw the old side's code: {states:?}"
        );
    }

    /// A repository whose `old.rs` (20 functions) is renamed to `new.rs` and cut to 15, uncommitted.
    fn renamed_repo() -> (tempfile::TempDir, PlainGit) {
        let dir = tempfile::tempdir().expect("tempdir");
        git_in(dir.path(), &["init", "-q", "-b", "main"]);
        for (k, v) in [
            ("user.email", "t@example.invalid"),
            ("user.name", "t"),
            ("commit.gpgsign", "false"),
        ] {
            git_in(dir.path(), &["config", k, v]);
        }
        let old: String = (1..=20).map(|i| format!("fn f{i}() {{}}\n")).collect();
        std::fs::write(dir.path().join("old.rs"), &old).expect("write");
        git_in(dir.path(), &["add", "old.rs"]);
        git_in(dir.path(), &["commit", "-q", "-m", "one"]);
        git_in(dir.path(), &["mv", "old.rs", "new.rs"]);
        let new: String = (1..=15).map(|i| format!("fn f{i}() {{}}\n")).collect();
        std::fs::write(dir.path().join("new.rs"), &new).expect("write");
        let git = PlainGit(dir.path().to_path_buf());
        let diff = git.run(&["diff", "HEAD"]).expect("diff");
        assert!(
            diff.contains("rename from old.rs"),
            "precondition: a rename: {diff}"
        );
        (dir, git)
    }

    #[tokio::test]
    async fn a_finding_on_the_old_path_of_a_rename_is_kept_under_the_new_path() {
        // [MAGI S6 loop 4, B1-9] a mage that names a deleted line by the file's old path cites code
        // the batch showed (the old side of the rename): it is reviewed, under the new path.
        let (dir, git) = renamed_repo();
        let f = serde_json::json!([
            {"severity":"warning","title":"t","detail":"d","file":"old.rs","line":17}
        ]);
        let consult =
            FakeConsult::new(vec![Ok(
                real_report([f.clone(), f, serde_json::json!([])]).await
            )]);
        let client = gate_always("real_defect");
        let guard = |p: &Path| -> Result<PathBuf, ReviewError> { Ok(p.to_path_buf()) };
        let registry = LanguageRegistry::new();
        let pool = vec![("m".to_string(), Some(POOL_WINDOW * 10))];
        let out = run_review(
            ReviewRequest {
                input: ReviewInput::Uncommitted,
                workspace: dir.path(),
                timeout_secs: None,
                deadline: None,
                max_rotations: 2,
                retry_disabled: false,
                agent_timeout_secs: 2_335,
                approved_batches: None,
                approved_digest: None,
                prep_started: None,
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
        assert!(
            out.summary.discarded.is_empty(),
            "the old path names reviewed code: {:?}",
            out.summary.discarded
        );
        let kept: Vec<_> = out
            .summary
            .findings
            .iter()
            .map(|(f, v)| (f.file.clone(), f.line, v.state))
            .collect();
        assert_eq!(
            kept,
            vec![(Some("new.rs".to_string()), Some(17), GateState::Kept)],
            "under the new path, and the gate read the old side"
        );
    }
}
