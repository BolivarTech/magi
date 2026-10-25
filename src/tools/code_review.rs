// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! The `code-review` surface: the agent tool, the runner the three surfaces share (the
//! `magi code-review` subcommand, the TUI `/code-review` command and the tool), and the pure
//! pieces that turn a review's outcome into what each surface prints.
// Red stub: every item below gets its production caller in the Green commit.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use magi_rs::decision::{DecisionClient, ModelSpec};
use magi_rs::review::error::ReviewError;
use magi_rs::review::input::ReviewInput;
use magi_rs::review::pipeline::{BatchConsult, ReviewOutcome};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::tools::{Tool, ToolError, ToolResult};

/// What a review run as a tool leaves its caller before the run's token fires.
///
/// Consolidation is capped (about 0.9 s, `MAX_UNLOCATED_PROBES`); rendering and redaction are
/// linear in the model text, bounded by batches × three mages × the output cap (65 536 tokens,
/// about 256 KiB): a hundred batches are about 75 MiB, under a second at 100 MB/s or more. A
/// larger review needs a generous `--timeout`.
pub(crate) const REVIEW_DEADLINE_MARGIN_SECS: u64 = 30;

/// How many batches a review splits into, and its worst case, computed without any model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReviewPlan {
    /// The batches.
    pub batches: usize,
    /// `batches × (1 + max_rotations) × 2 × agent_timeout_secs`: an upper bound, never the
    /// expected duration.
    pub worst_case_secs: u64,
}

/// What a review produced, rendered for every surface.
#[derive(Debug, Clone)]
pub(crate) struct Rendered {
    /// The text report (or the paid batch reports when consolidation failed).
    pub text: String,
    /// The review's own summary (`code_review` in the JSON output).
    pub json: Value,
    /// The `consult` object of the JSON output; `null` when no consensus exists.
    pub consult: Value,
    /// Why there is no consolidated report, redacted.
    pub failure: Option<String>,
}

/// One review, run by the subcommand, the TUI and the tool alike.
#[async_trait]
pub(crate) trait CodeReviewRunner: Send + Sync {
    /// Plans the review without calling a model.
    ///
    /// # Errors
    /// Any [`ReviewError`] of the input, the filter or the batching.
    fn plan(&self, input: &ReviewInput) -> Result<ReviewPlan, ReviewError>;

    /// Runs the review until it ends or `cancel` fires.
    ///
    /// # Errors
    /// Any [`ReviewError`] before the first trio call, or [`ReviewError::Cancelled`].
    async fn run(
        &self,
        input: ReviewInput,
        timeout: Option<Duration>,
        cancel: &CancellationToken,
    ) -> Result<Rendered, ReviewError>;
}

/// Runs `fut` until it ends or `cancel` fires.
///
/// # Errors
/// The future's own error, or [`ReviewError::Cancelled`].
pub(crate) async fn run_cancellable<T>(
    _cancel: &CancellationToken,
    _fut: impl std::future::Future<Output = Result<T, ReviewError>>,
) -> Result<T, ReviewError> {
    Err(ReviewError::Io(String::new()))
}

/// `(exit code, stdout, stderr line)` of a rendered review.
pub(crate) fn finish_code_review(r: &Rendered) -> (i32, String, Option<String>) {
    (0, r.text.clone(), None)
}

/// A runner's [`ReviewOutcome`] rendered for every surface.
pub(crate) fn render_outcome(_o: ReviewOutcome, _structured: bool) -> Rendered {
    Rendered {
        text: String::new(),
        json: Value::Null,
        consult: Value::Null,
        failure: None,
    }
}

/// The envelope fields of the JSON output: `(response, consult, code_review)`.
pub(crate) fn envelope_fields(r: &Rendered) -> (String, Value, Value) {
    (r.text.clone(), r.json.clone(), r.json.clone())
}

/// The tool arguments as a review input.
///
/// # Errors
/// [`ToolError::InvalidArguments`].
pub(crate) fn parse_tool_input(_args: &Value) -> Result<ReviewInput, ToolError> {
    Ok(ReviewInput::Uncommitted)
}

/// `cause.name` of a review error's run-log entry.
pub(crate) fn review_error_cause_name(_e: &ReviewError) -> &'static str {
    "budget"
}

/// A `File` that names a directory under `workspace` becomes `Directory`.
pub(crate) fn path_input(_workspace: &Path, input: ReviewInput) -> ReviewInput {
    input
}

/// The agent tool `code-review`.
pub(crate) struct CodeReviewTool {
    /// The shared runner.
    runner: Arc<dyn CodeReviewRunner>,
    /// Whether an invocation needs no approval.
    auto_approve: bool,
}

impl CodeReviewTool {
    /// Builds the tool.
    pub(crate) fn new(runner: Arc<dyn CodeReviewRunner>, auto_approve: bool) -> Self {
        Self {
            runner,
            auto_approve,
        }
    }

    /// Bounds every review to what is left of the calling run.
    pub(crate) fn with_run_deadline(self, _deadline: std::time::Instant) -> Self {
        self
    }
}

#[async_trait]
impl Tool for CodeReviewTool {
    fn name(&self) -> &str {
        "code-review"
    }

    fn description(&self) -> &str {
        ""
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
        Ok(Value::Null)
    }

    fn requires_approval(&self) -> bool {
        true
    }
}

/// What the production runner is built from.
pub(crate) struct ReviewRunnerParts {
    /// The workspace root.
    pub workspace: PathBuf,
    /// The pool, as `(model, measured window)`.
    pub pool: Vec<(String, Option<usize>)>,
    /// The trio, one `consult` per batch.
    pub consult: Arc<dyn BatchConsult>,
    /// The gate's decision client, its chain without thresholds.
    pub gate_client: Arc<dyn DecisionClient>,
    /// The gate's chain, with its thresholds.
    pub gate_chain: Vec<ModelSpec>,
    /// `[code_review].max_state_bytes`.
    pub max_state_bytes: usize,
    /// `[code_review].timeout_secs`, per gate call.
    pub call_timeout: Duration,
    /// `[magi].max_rotations`, resolved.
    pub max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    pub retry_disabled: bool,
    /// `[magi].agent_timeout_secs`, resolved.
    pub agent_timeout_secs: u64,
    /// Whether the `consult` object carries the consolidated agents.
    pub structured: bool,
}

/// The production [`CodeReviewRunner`].
pub(crate) struct ProductionReviewRunner {
    /// What it was built from.
    parts: ReviewRunnerParts,
}

impl ProductionReviewRunner {
    /// Builds the runner.
    ///
    /// # Errors
    /// When the workspace cannot be guarded.
    pub(crate) fn new(parts: ReviewRunnerParts) -> anyhow::Result<Self> {
        Ok(Self { parts })
    }
}

#[async_trait]
impl CodeReviewRunner for ProductionReviewRunner {
    fn plan(&self, _input: &ReviewInput) -> Result<ReviewPlan, ReviewError> {
        Ok(ReviewPlan {
            batches: 0,
            worst_case_secs: 0,
        })
    }

    async fn run(
        &self,
        _input: ReviewInput,
        _timeout: Option<Duration>,
        _cancel: &CancellationToken,
    ) -> Result<Rendered, ReviewError> {
        Ok(render_outcome(
            ReviewOutcome {
                consolidated: None,
                batch_reports: Vec::new(),
                summary: magi_rs::review::report::ReviewSummary::default(),
                consolidation_error: None,
            },
            self.parts.structured,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FakeRunner {
        batches: usize,
        worst_case_secs: u64,
        text: String,
        calls: std::sync::atomic::AtomicUsize,
        workspace: Option<std::path::PathBuf>,
        timeout_seen: std::sync::Mutex<Option<std::time::Duration>>,
    }
    #[async_trait::async_trait]
    impl CodeReviewRunner for FakeRunner {
        fn plan(&self, input: &ReviewInput) -> Result<ReviewPlan, ReviewError> {
            if let (Some(ws), ReviewInput::File(p) | ReviewInput::Directory(p)) =
                (&self.workspace, input)
            {
                let guard = crate::system::path_guard::PathGuard::new(ws.clone()).expect("guard");
                guard
                    .validate(p)
                    .map_err(|e| ReviewError::OutsideWorkspace(e.to_string()))?;
            }
            Ok(ReviewPlan {
                batches: self.batches,
                worst_case_secs: self.worst_case_secs,
            })
        }
        async fn run(
            &self,
            _input: ReviewInput,
            timeout: Option<std::time::Duration>,
            _c: &CancellationToken,
        ) -> Result<Rendered, ReviewError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            *self.timeout_seen.lock().expect("lock") = timeout;
            Ok(Rendered {
                text: self.text.clone(),
                json: serde_json::json!({"gate_states": {}}),
                consult: serde_json::Value::Null,
                failure: None,
            })
        }
    }

    #[test]
    fn the_tool_asks_for_approval_unless_auto_approved() {
        assert!(CodeReviewTool::new(Arc::new(FakeRunner::default()), false).requires_approval());
        assert!(!CodeReviewTool::new(Arc::new(FakeRunner::default()), true).requires_approval());
    }

    #[test]
    fn the_approval_detail_shows_batches_and_the_worst_case() {
        let t = CodeReviewTool::new(
            Arc::new(FakeRunner {
                batches: 3,
                worst_case_secs: 42_030,
                ..FakeRunner::default()
            }),
            false,
        );
        assert_eq!(
            t.approval_detail(&serde_json::json!({"path":"src"}))
                .as_deref(),
            Some("3 batches; worst case 42030 s")
        );
    }

    #[tokio::test]
    async fn as_a_tool_the_review_gets_what_is_left_of_the_run() {
        let runner = Arc::new(FakeRunner::default());
        // [CP2 MS2 S2 loop 1] a wide window: under CI load the gap between deadline and execute is far below 10 minutes.
        let t = CodeReviewTool::new(runner.clone(), true)
            .with_run_deadline(std::time::Instant::now() + std::time::Duration::from_secs(3_600));
        t.execute(
            serde_json::json!({"path": "src"}),
            &CancellationToken::new(),
        )
        .await
        .expect("runs");
        let got = runner
            .timeout_seen
            .lock()
            .expect("lock")
            .expect("a timeout reached the runner");
        // [CP2 MS2 S2b loop 8] the margin: the review's own clock ends before the run's token fires.
        assert!(
            got <= std::time::Duration::from_secs(3_600 - REVIEW_DEADLINE_MARGIN_SECS)
                && got > std::time::Duration::from_secs(3_000),
            "{got:?}"
        );
    }

    #[tokio::test]
    async fn a_spent_run_deadline_is_an_error_not_a_panic() {
        // [CP2 MS2 S2 loop 2] deadline already past: no Instant underflow, no runner call.
        let runner = Arc::new(FakeRunner::default());
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("a past instant");
        let t = CodeReviewTool::new(runner.clone(), true).with_run_deadline(past);
        assert!(t
            .execute(
                serde_json::json!({"path": "src"}),
                &CancellationToken::new()
            )
            .await
            .is_err());
        assert!(
            runner.timeout_seen.lock().expect("lock").is_none(),
            "the runner was not called"
        );
        // [CP2 MS2 S2b loop 7] a deadline of exactly now: saturating_duration_since gives zero, which is the same error.
        let t =
            CodeReviewTool::new(runner.clone(), true).with_run_deadline(std::time::Instant::now());
        assert!(t
            .execute(
                serde_json::json!({"path": "src"}),
                &CancellationToken::new()
            )
            .await
            .is_err());
        assert!(
            runner.timeout_seen.lock().expect("lock").is_none(),
            "the runner was not called"
        );
        // [CP2 MS2 S2b loop 8] less than the margin left is spent too.
        let soon = std::time::Instant::now()
            + std::time::Duration::from_secs(REVIEW_DEADLINE_MARGIN_SECS / 2);
        let t = CodeReviewTool::new(runner.clone(), true).with_run_deadline(soon);
        assert!(t
            .execute(
                serde_json::json!({"path": "src"}),
                &CancellationToken::new()
            )
            .await
            .is_err());
        assert!(
            runner.timeout_seen.lock().expect("lock").is_none(),
            "the runner was not called"
        );
    }

    #[test]
    fn the_input_schema_names_the_seven_input_forms() {
        // [CP2 MS2 S2b loop 7] a drifted schema would mislead the agent with every other test green.
        let t = CodeReviewTool::new(Arc::new(FakeRunner::default()), false);
        let keys: std::collections::BTreeSet<String> = t.input_schema()["properties"]
            .as_object()
            .expect("properties")
            .keys()
            .cloned()
            .collect();
        let want = [
            "base",
            "commit",
            "from",
            "head",
            "path",
            "to",
            "uncommitted",
        ]
        .map(String::from);
        assert_eq!(keys, std::collections::BTreeSet::from(want));
    }

    #[test]
    fn a_failed_consolidation_exits_non_zero_with_the_batch_reports_on_stdout() {
        // [CP2 MS2 S2 loop 1] RV-S21 / M-9 at the surface.
        let r = Rendered {
            text: "## [batch 0]\nreport".into(),
            json: serde_json::json!({}),
            consult: serde_json::Value::Null,
            failure: Some("consolidation failed: insufficient agents".into()),
        };
        let (code, stdout, stderr) = finish_code_review(&r);
        assert_eq!(code, 1);
        assert!(stdout.contains("## [batch 0]"));
        assert_eq!(
            stderr.as_deref(),
            Some("error: consolidation failed: insufficient agents")
        );
        let ok = Rendered { failure: None, ..r };
        assert_eq!(finish_code_review(&ok).0, 0);
    }

    #[test]
    fn a_failed_consolidation_in_json_mode_keeps_the_envelope_with_a_null_consult() {
        // [CP2 MS2 S2b loop 10] M-9 × --output-format json: the same three fields as success, consult null.
        let r = Rendered {
            text: "## [batch 0]\nreport".into(),
            json: serde_json::json!({"batches_planned": 1}),
            consult: serde_json::Value::Null,
            failure: Some("consolidation failed: x".into()),
        };
        let (response, consult, code_review) = envelope_fields(&r);
        assert!(response.contains("## [batch 0]"));
        assert_eq!(consult, serde_json::Value::Null);
        assert_eq!(code_review["batches_planned"], 1);
    }

    #[test]
    fn a_failed_consolidation_renders_the_batch_reports_with_a_null_consult() {
        // [CP2 MS2 S2b loop 9] M-9 at the runner: no consensus exists, so `consult` is null; `json` is still the summary.
        let o = magi_rs::review::pipeline::ReviewOutcome {
            consolidated: None,
            batch_reports: vec![],
            summary: magi_rs::review::report::ReviewSummary::default(),
            consolidation_error: Some("consolidation failed: insufficient agents".into()),
        };
        let r = render_outcome(o, true);
        assert_eq!(
            r.failure.as_deref(),
            Some("consolidation failed: insufficient agents")
        );
        assert_eq!(r.consult, serde_json::Value::Null);
        assert_eq!(
            r.json,
            magi_rs::review::report::summary_json(
                &magi_rs::review::report::ReviewSummary::default()
            )
        );
    }

    #[tokio::test]
    async fn the_tool_result_is_redacted_data() {
        let runner = FakeRunner {
            text: "fix https://user:hunter2@example.invalid/x".into(), // allow-secret-scan
            ..FakeRunner::default()
        };
        let t = CodeReviewTool::new(Arc::new(runner), true);
        let v = t
            .execute(
                serde_json::json!({"path": "src"}),
                &CancellationToken::new(),
            )
            .await
            .expect("runs");
        assert!(!v.to_string().contains("hunter2"), "{v}");
        assert!(
            v.get("report").is_some() && v.get("code_review").is_some(),
            "{v}"
        );
    }

    #[tokio::test]
    async fn cancelling_the_token_drops_the_review_in_flight() {
        // [CP2 MS2 S2b loop 6] REQ-H36: --timeout or a cancel stops a multi-batch review, the batch in flight too.
        struct Dropped(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = Dropped(Arc::clone(&flag));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let r: Result<(), ReviewError> = run_cancellable(&cancel, async move {
            let _g = guard;
            std::future::pending::<Result<(), ReviewError>>().await
        })
        .await;
        assert!(matches!(r, Err(ReviewError::Cancelled)), "{r:?}");
        assert!(
            flag.load(std::sync::atomic::Ordering::SeqCst),
            "the review future was not dropped"
        );
        let done: Result<u8, ReviewError> =
            run_cancellable(&CancellationToken::new(), async { Ok(7) }).await;
        assert_eq!(done.expect("not cancelled"), 7);
        // [CP2 MS2 S2b loop 8] a finished review and a fired token at once: the finished review wins.
        let ready: Result<u8, ReviewError> = run_cancellable(&cancel, async { Ok(9) }).await;
        assert_eq!(ready.expect("a finished future wins"), 9);
    }

    #[test]
    fn the_tool_input_takes_one_form_and_complete_pairs() {
        // [CP2 MS2 S2b loop 8] mixed or half forms are an argument error, never a guess.
        assert_eq!(
            parse_tool_input(&serde_json::json!({"path": "src"})).expect("a path"),
            ReviewInput::File("src".into())
        );
        assert_eq!(
            parse_tool_input(&serde_json::json!({})).expect("nothing is HEAD"),
            ReviewInput::Commit(None)
        );
        for bad in [
            serde_json::json!({"path": "src", "commit": "HEAD"}),
            serde_json::json!({"from": "a"}),
            serde_json::json!({"base": "main"}),
            serde_json::json!({"uncommitted": true, "to": "b"}),
            serde_json::json!({"to": "b"}),
            serde_json::json!({"head": "x"}),
        ] {
            // [S2b loop 11] both halves
            assert!(
                matches!(parse_tool_input(&bad), Err(ToolError::InvalidArguments(_))),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_path_names_a_directory_or_a_file_by_what_is_on_disk() {
        // [CP2 MS2 S2b loop 8] one rule for the CLI positional and the tool's `path`.
        let ws = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(ws.path().join("src")).expect("dir");
        std::fs::write(ws.path().join("src").join("a.rs"), "fn a() {}\n").expect("file");
        assert_eq!(
            path_input(ws.path(), ReviewInput::File("src".into())),
            ReviewInput::Directory("src".into())
        );
        assert_eq!(
            path_input(ws.path(), ReviewInput::File("src/a.rs".into())),
            ReviewInput::File("src/a.rs".into())
        );
        assert_eq!(
            path_input(ws.path(), ReviewInput::File("gone.rs".into())),
            ReviewInput::File("gone.rs".into())
        );
        assert_eq!(
            path_input(ws.path(), ReviewInput::Commit(None)),
            ReviewInput::Commit(None)
        );
    }

    #[tokio::test]
    async fn a_hostile_path_or_ref_fails_before_any_model_call() {
        let ws = tempfile::tempdir().expect("tempdir");
        let runner = Arc::new(FakeRunner {
            workspace: Some(dunce::canonicalize(ws.path()).expect("canon")),
            ..FakeRunner::default()
        });
        let t = CodeReviewTool::new(runner.clone(), true);
        let cancel = CancellationToken::new();
        assert!(t
            .execute(serde_json::json!({"path": "../out"}), &cancel)
            .await
            .is_err());
        assert!(t
            .execute(serde_json::json!({"commit": "--output=x"}), &cancel)
            .await
            .is_err());
        assert!(matches!(
            parse_tool_input(&serde_json::json!({"commit": "--output=x"})),
            Err(ToolError::InvalidArguments(_))
        ));
        assert_eq!(runner.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// Counts its `plan` calls, for the plan the tool caches between approval and execution.
    #[derive(Default)]
    struct PlanCounter {
        plans: std::sync::atomic::AtomicUsize,
        runs: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl CodeReviewRunner for PlanCounter {
        fn plan(&self, _input: &ReviewInput) -> Result<ReviewPlan, ReviewError> {
            self.plans.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ReviewPlan {
                batches: 2,
                worst_case_secs: 9,
            })
        }
        async fn run(
            &self,
            _input: ReviewInput,
            _t: Option<std::time::Duration>,
            _c: &CancellationToken,
        ) -> Result<Rendered, ReviewError> {
            self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Rendered {
                text: String::new(),
                json: Value::Null,
                consult: Value::Null,
                failure: None,
            })
        }
    }

    #[tokio::test]
    async fn the_plan_shown_for_approval_is_the_plan_that_runs() {
        // The approved arguments run on the plan computed for the prompt: git is enumerated once.
        let runner = Arc::new(PlanCounter::default());
        let t = CodeReviewTool::new(runner.clone(), false);
        let args = serde_json::json!({"path": "src"});
        assert_eq!(
            t.approval_detail(&args).as_deref(),
            Some("2 batches; worst case 9 s")
        );
        t.execute(args.clone(), &CancellationToken::new())
            .await
            .expect("runs");
        assert_eq!(runner.plans.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(runner.runs.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The cache is single-use: a second execution with the same arguments plans again.
        t.execute(args, &CancellationToken::new())
            .await
            .expect("runs");
        assert_eq!(runner.plans.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// A trio double that counts the batches it is asked to review.
    #[derive(Default)]
    struct CountingConsult(std::sync::atomic::AtomicUsize);
    #[async_trait::async_trait]
    impl BatchConsult for CountingConsult {
        async fn consult(
            &self,
            _prompt: &str,
            _wall_clock: Option<std::time::Duration>,
        ) -> Result<magi_core::reporting::MagiReport, magi_rs::review::pipeline::ConsultFailure>
        {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(magi_rs::review::pipeline::ConsultFailure::Other(
                "not reached".into(),
            ))
        }
    }

    /// A gate client that is never reached by the tests below.
    struct NoDecision;
    #[async_trait::async_trait]
    impl DecisionClient for NoDecision {
        async fn decide(
            &self,
            _request: &magi_rs::decision::DecisionRequest,
        ) -> Result<magi_rs::decision::Decision, magi_rs::decision::DecisionError> {
            Err(magi_rs::decision::DecisionError::ClientUnavailable {
                reason: "not reached".into(),
            })
        }
    }

    /// `<tmp>/ws` with `src/a.rs`, beside `<tmp>/fuera/x.rs`.
    fn guarded_workspace() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = dunce::canonicalize(tmp.path()).expect("canonical");
        let ws = base.join("ws");
        std::fs::create_dir_all(ws.join("src")).expect("ws/src");
        std::fs::write(ws.join("src").join("a.rs"), "fn a() {}\n").expect("a.rs");
        std::fs::create_dir_all(base.join("fuera")).expect("fuera");
        std::fs::write(base.join("fuera").join("x.rs"), "fn x() {}\n").expect("x.rs");
        (tmp, ws)
    }

    fn production_runner(ws: &Path, consult: Arc<CountingConsult>) -> ProductionReviewRunner {
        ProductionReviewRunner::new(ReviewRunnerParts {
            workspace: ws.to_path_buf(),
            pool: vec![("m".into(), None)],
            consult,
            gate_client: Arc::new(NoDecision),
            gate_chain: vec![ModelSpec {
                model: "g".into(),
                min_confidence: None,
            }],
            max_state_bytes: 61_000,
            call_timeout: Duration::from_secs(1),
            max_rotations: 0,
            retry_disabled: false,
            agent_timeout_secs: 120,
            structured: false,
        })
        .expect("runner")
    }

    #[tokio::test]
    async fn the_production_runner_refuses_an_escaping_path_before_any_model_call() {
        // RV-S10 through the production guard (PathGuard), never a test double's.
        let (_tmp, ws) = guarded_workspace();
        let consult = Arc::new(CountingConsult::default());
        let runner = production_runner(&ws, Arc::clone(&consult));
        let hostile = ReviewInput::File("../fuera/x.rs".into());
        assert!(
            matches!(runner.plan(&hostile), Err(ReviewError::OutsideWorkspace(_))),
            "plan"
        );
        let run = runner.run(hostile, None, &CancellationToken::new()).await;
        assert!(
            matches!(run, Err(ReviewError::OutsideWorkspace(ref p)) if p.contains("fuera")),
            "{:?}",
            run.err()
        );
        assert_eq!(consult.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn the_production_runner_plans_a_file_inside_the_workspace() {
        let (_tmp, ws) = guarded_workspace();
        let runner = production_runner(&ws, Arc::new(CountingConsult::default()));
        let plan = runner
            .plan(&ReviewInput::File("src/a.rs".into()))
            .expect("plans");
        assert_eq!(plan.batches, 1);
        assert_eq!(plan.worst_case_secs, 240, "1 × (1 + 0) × 2 × 120");
    }

    #[cfg(unix)] // creating a symlink needs Developer Mode on Windows; the `..` case runs everywhere.
    #[tokio::test]
    async fn the_production_runner_refuses_a_symlinked_directory_that_escapes() {
        let (tmp, ws) = guarded_workspace();
        let base = dunce::canonicalize(tmp.path()).expect("canonical");
        std::os::unix::fs::symlink(base.join("fuera"), ws.join("link")).expect("symlink");
        let consult = Arc::new(CountingConsult::default());
        let runner = production_runner(&ws, Arc::clone(&consult));
        let link = ReviewInput::File("link".into());
        assert!(matches!(
            runner.plan(&link),
            Err(ReviewError::OutsideWorkspace(_))
        ));
        assert!(matches!(
            runner.run(link, None, &CancellationToken::new()).await,
            Err(ReviewError::OutsideWorkspace(_))
        ));
        assert_eq!(consult.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
