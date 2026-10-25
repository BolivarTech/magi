// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! The review pipeline: one `consult code-review` per batch, then one report.

use std::path::{Path, PathBuf};
use std::time::Duration;

use magi_core::reporting::MagiReport;

use super::consolidate::Consolidated;
use super::error::ReviewError;
use super::gate::Gate;
use super::input::ReviewInput;
use super::languages::LanguageRegistry;
use super::report::ReviewSummary;
use super::source::GitCommand;

/// Why one batch's `consult` gave no report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsultFailure {
    /// The batch's wall clock ran out: the batches still pending are not run.
    ClockExpired,
    /// Any other failure, already redacted.
    Other(String),
}

/// One `consult code-review` over the trio.
#[async_trait::async_trait]
pub trait BatchConsult: Send + Sync {
    /// Runs the trio over `prompt`, under `wall_clock` when there is one.
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

/// The production [`ReviewClock`], over an [`std::time::Instant`].
pub struct SystemReviewClock {
    start: std::time::Instant,
}

impl SystemReviewClock {
    /// A clock that starts now.
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
}

impl ReviewClock for SystemReviewClock {
    fn elapsed_secs(&self) -> u64 {
        let _ = self.start;
        todo!()
    }
}

/// Runs one batch under its wall clock.
///
/// # Errors
///
/// [`ConsultFailure::ClockExpired`] when the wall clock elapsed, [`ConsultFailure::Other`]
/// with the future's error otherwise.
pub async fn run_with_wall_clock<F>(
    wall: Option<Duration>,
    fut: F,
) -> Result<MagiReport, ConsultFailure>
where
    F: std::future::Future<Output = Result<MagiReport, String>> + Send,
{
    let _ = (wall, fut);
    todo!()
}

/// What to review and under which clock.
pub struct ReviewRequest<'a> {
    /// What the caller asked to review.
    pub input: ReviewInput,
    /// The workspace root.
    pub workspace: &'a Path,
    /// The explicit `--timeout`, the total of the review.
    pub timeout_secs: Option<u64>,
    /// `[magi].max_rotations`, resolved.
    pub max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    pub retry_disabled: bool,
    /// `[magi].agent_timeout_secs`, resolved.
    pub agent_timeout_secs: u64,
}

/// What a review runs on.
pub struct ReviewDeps<'a> {
    /// The review's git.
    pub git: &'a dyn GitCommand,
    /// The sandbox check of `File`/`Directory` paths.
    pub guard: &'a dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
    /// Whether `.gitignore` excludes a path.
    pub is_ignored: &'a dyn Fn(&str) -> bool,
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
    /// The consolidated report.
    pub consolidated: Option<Consolidated>,
    /// The paid batch reports, when consolidation failed.
    pub batch_reports: Vec<MagiReport>,
    /// What every stage declared.
    pub summary: ReviewSummary,
    /// Why there is no consolidated report.
    pub consolidation_error: Option<String>,
}

/// Runs a review.
///
/// # Errors
///
/// Any [`ReviewError`] before the first trio call.
pub async fn run_review(
    req: ReviewRequest<'_>,
    deps: ReviewDeps<'_>,
) -> Result<ReviewOutcome, ReviewError> {
    let _ = (req, deps);
    todo!()
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
        let git = FakeGit::from(&[]);
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
            ("rev-parse --verify HEAD^{commit}", Ok("abc\n")),
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
            ("rev-parse --verify HEAD^{commit}", Ok("abc\n")),
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
        let git = FakeGit::from(&[]);
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
            text.contains("## [batch 0]") && text.contains("## [batch 1]"),
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
