// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! Builds the production `code-review` runner from what a surface already resolved: the trio
//! that answers each batch and the decision client of the false-positive gate.
//!
//! The three surfaces (`magi code-review`, the TUI and `magi query`'s tool) build it the same way,
//! here. Each batch gets a trio of its own, built with the ceiling derived from that batch's wall
//! clock, so an explicit `--timeout` reaches every mage through the one derivation the headless
//! path uses (REQ-A04: no second timeout knob).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use magi_core::orchestrator::Magi;
use magi_core::reporting::MagiReport;
use magi_core::schema::Mode;
use magi_rs::decision::{DecisionSettings, ModelSpec, SystemOneClient};
use magi_rs::magi::endpoint::Scope;
use magi_rs::magi::kind::ProviderKind;
use magi_rs::magi::probe::Measurement;
use magi_rs::magi::{resolve_run_timeout, BudgetTelemetry, ResolvedCeiling, TimeoutMeasure};
use magi_rs::redact::{redact_foreign_error, redact_foreign_text};
use magi_rs::review::pipeline::{run_with_wall_clock, BatchConsult, ConsultFailure};
use magi_rs::section_keys::{SectionKeys, SharedSecretStore};

use crate::config::MagiConfig;
use crate::system::model_cache::ModelCapabilityCache;
use crate::tools::code_review::{CodeReviewRunner, ProductionReviewRunner, ReviewRunnerParts};
use crate::{
    build_magi_orchestrator, candidate_window_view, code_review_api_key, resolve_backend_model,
    resolve_magi_kind, resolve_section_endpoint, seats_with_env_overrides, timeout_scale,
    trio_unavailable_message, MagiEnvModelOverrides, ProbeOutcome, ResolvedEndpoints, TrioBuild,
};

/// Everything a surface resolved that the review runner is built from.
pub(crate) struct CodeReviewBuild<'a> {
    /// The workspace root (the review's sandbox).
    pub workspace: PathBuf,
    /// The loaded configuration.
    pub cfg: MagiConfig,
    /// The resolved principal kind.
    pub principal_kind: ProviderKind,
    /// The resolved endpoints.
    pub endpoints: ResolvedEndpoints,
    /// The section keys chosen once at startup (the trio's key and the gate's key are copied).
    pub keys: &'a SectionKeys,
    /// The probe-derived warning threshold.
    pub warn_tokens: Option<usize>,
    /// `MAGI_MODEL_*`, read once.
    pub env_overrides: MagiEnvModelOverrides,
    /// The capability cache, when a database is open.
    pub capability_cache: Option<Arc<ModelCapabilityCache>>,
    /// What the startup probe measured.
    pub probe: ProbeOutcome,
    /// The open vault, for the `[code_review]` endpoint placeholders.
    pub secret_store: Option<SharedSecretStore>,
    /// Whether the `consult` object carries the consolidated agents.
    pub structured: bool,
    /// Whether the surface already built this trio successfully; when it did not, the runner
    /// builds it once to report a trio that cannot be built before any review starts.
    pub trio_checked: bool,
}

/// The trio's credentials, copied from the section keys: both kinds take the `[magi]` key, as
/// [`SectionKeys`] hands it to the trio.
struct TrioKey(Option<zeroize::Zeroizing<String>>);

impl crate::Credentials for TrioKey {
    fn openai(&self) -> Option<String> {
        self.0.as_ref().map(|k| k.to_string())
    }
    fn anthropic(&self) -> Option<String> {
        self.0.as_ref().map(|k| k.to_string())
    }
}

/// Builds the trio of one batch from owned startup state.
struct TrioFactory {
    /// The loaded configuration.
    cfg: MagiConfig,
    /// The resolved principal kind.
    principal_kind: ProviderKind,
    /// The resolved endpoints.
    endpoints: ResolvedEndpoints,
    /// The trio's credentials.
    creds: TrioKey,
    /// The probe-derived warning threshold.
    warn_tokens: Option<usize>,
    /// `MAGI_MODEL_*`, read once.
    env_overrides: MagiEnvModelOverrides,
    /// The capability cache, if any.
    capability_cache: Option<Arc<ModelCapabilityCache>>,
    /// What the startup probe measured.
    probe: ProbeOutcome,
}

/// Builds the trio of one batch under a ceiling. Production: [`TrioFactory`].
trait TrioBuilder: Send + Sync {
    /// Builds a trio under `ceiling`.
    ///
    /// # Arguments
    /// * `ceiling` - The batch's resolved ceiling.
    ///
    /// # Returns
    /// The trio.
    ///
    /// # Errors
    /// A message naming why the trio cannot be built (not yet redacted).
    fn build(&self, ceiling: ResolvedCeiling) -> Result<Arc<Magi>, String>;
}

impl TrioBuilder for TrioFactory {
    /// Builds a trio under `ceiling`. Its notices were already announced by the startup build
    /// of the same configuration, so they are dropped here.
    fn build(&self, ceiling: ResolvedCeiling) -> Result<Arc<Magi>, String> {
        let mut notices = Vec::new();
        build_magi_orchestrator(
            &TrioBuild {
                cfg: &self.cfg,
                principal_kind: self.principal_kind,
                endpoints: &self.endpoints,
                creds: Some(&self.creds),
                warn_tokens: self.warn_tokens,
                env_overrides: &self.env_overrides,
                capability_cache: self.capability_cache.as_ref(),
                probe: &self.probe,
                ceiling,
            },
            &mut notices,
        )
        .map_err(|e| trio_unavailable_message(&e))
    }
}

/// The production [`BatchConsult`]: a trio per batch, in the code-review mode.
struct TrioBatchConsult {
    /// Builds each batch's trio.
    factory: Arc<dyn TrioBuilder>,
    /// `[magi].agent_timeout_secs`, resolved.
    configured_ceiling: u64,
    /// `[magi].max_rotations`, resolved.
    max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    retry_disabled: bool,
}

impl TrioBatchConsult {
    /// Builds the trio once under the configured ceiling, so a trio that cannot be built is
    /// reported before any review starts.
    ///
    /// # Errors
    /// The builder's message, redacted as every other error of [`build_code_review_runner`].
    fn check(&self) -> Result<(), String> {
        self.factory.build(self.ceiling(None)).map(|_| ())
    }

    /// The ceiling of a batch: derived from its wall clock as headless derives it from a
    /// `--timeout`, or the configured one when the batch has none.
    fn ceiling(&self, wall_clock: Option<Duration>) -> ResolvedCeiling {
        let decision = wall_clock.map(|w| {
            resolve_run_timeout(
                Some(w.as_secs()),
                self.configured_ceiling,
                self.max_rotations,
                self.retry_disabled,
                TimeoutMeasure::DerivesCeiling,
            )
        });
        BudgetTelemetry::derive(
            decision.as_ref(),
            self.configured_ceiling,
            self.max_rotations,
            self.retry_disabled,
        )
        .0
    }
}

#[async_trait::async_trait]
impl BatchConsult for TrioBatchConsult {
    async fn consult(
        &self,
        prompt: &str,
        wall_clock: Option<Duration>,
    ) -> Result<MagiReport, ConsultFailure> {
        let factory = Arc::clone(&self.factory);
        let ceiling = self.ceiling(wall_clock);
        run_with_wall_clock(wall_clock, async move {
            // MAGI FIX (S6 loop 3): the build is part of the batch. It runs on a blocking thread
            // inside the wall clock, so the clock bounds it and the run's token is not held up by
            // it (a build cut there finishes unread on its thread).
            let magi = tokio::task::spawn_blocking(move || factory.build(ceiling))
                .await
                .map_err(|e| format!("the trio build did not finish: {e}"))?
                .map_err(|e| redact_foreign_text(&e).to_string())?;
            let report = magi
                .analyze(&Mode::CodeReview, prompt)
                .await
                .map_err(|e| redact_foreign_error(&e).to_string())?;
            crate::cut_log::CutAttemptLog::from_report(&report).emit();
            Ok(report)
        })
        .await
    }
}

/// The pool a review sizes its batches against: the three seats and every declared fallback,
/// each with the window the probe measured for it (`None` when not measured).
fn review_pool(
    cfg: &MagiConfig,
    principal_kind: ProviderKind,
    endpoints: &ResolvedEndpoints,
    env_overrides: &MagiEnvModelOverrides,
    probe: &ProbeOutcome,
) -> Result<Vec<(String, Option<usize>)>, String> {
    let kind =
        resolve_magi_kind(cfg, principal_kind).map_err(|e| redact_foreign_error(&e).to_string())?;
    let backend = resolve_backend_model(cfg, kind);
    let view = candidate_window_view(cfg, endpoints, principal_kind, probe);
    let mut models: Vec<String> = seats_with_env_overrides(cfg, backend, env_overrides)
        .into_iter()
        .map(|(_, model)| model)
        .collect();
    models.extend(cfg.fallback_pool().iter().map(|f| f.model.clone()));
    // A model that is a seat and a fallback is one pool member: keep its first place.
    let mut seen = std::collections::HashSet::new();
    models.retain(|m| seen.insert(m.clone()));
    Ok(models
        .into_iter()
        .map(|m| {
            let window = match view.get(&m) {
                Some(Measurement::Measured { window, .. }) => Some(*window),
                _ => None,
            };
            (m, window)
        })
        .collect())
}

/// The gate's decision client: the gate's chain with every threshold removed, because the gate
/// applies each model's `min_confidence` itself. A client that kept them would turn an answer
/// under the threshold into a failed attempt, and the gate's `low_confidence` (with the lean and
/// the confidence) could never be reached.
///
/// # Arguments
/// * `base_url` - The `[code_review]` endpoint, resolved.
/// * `api_key` - Its key, if any.
/// * `chain` - The gate's chain, with its thresholds.
/// * `timeout` - Per call.
///
/// # Returns
/// The client.
///
/// # Errors
/// When the client cannot be built.
fn gate_client(
    base_url: &str,
    api_key: Option<zeroize::Zeroizing<String>>,
    chain: &[ModelSpec],
    timeout: Duration,
) -> Result<SystemOneClient, magi_rs::decision::DecisionError> {
    SystemOneClient::new(DecisionSettings {
        base_url: base_url.to_string(),
        api_key,
        chain: chain
            .iter()
            .map(|m| ModelSpec {
                model: m.model.clone(),
                min_confidence: None,
            })
            .collect(),
        timeout,
        keep_alive: None,
    })
}

/// Builds the review runner a surface shares.
///
/// # Arguments
/// * `b` - What the surface resolved.
///
/// # Returns
/// The runner.
///
/// # Errors
/// A redacted message when the trio cannot be built, the `[code_review]` endpoint cannot be
/// resolved (a missing vault entry, a bad URL), the gate's client cannot be built, or the
/// workspace cannot be guarded.
pub(crate) fn build_code_review_runner(
    b: CodeReviewBuild<'_>,
) -> Result<Arc<dyn CodeReviewRunner>, String> {
    let section = b.cfg.effective_code_review();
    let template = b
        .cfg
        .effective_code_review_base_url()
        .map_err(|e| redact_foreign_error(&e).to_string())?;
    let base_url = resolve_section_endpoint(&template, Scope::CodeReview, b.secret_store.as_ref())
        .map_err(|e| redact_foreign_error(&e).to_string())?;
    let call_timeout = Duration::from_secs(
        section
            .timeout_secs
            .unwrap_or(crate::defaults::DEFAULT_CODE_REVIEW_TIMEOUT_SECS),
    );
    let gate_chain = section.chain();
    let gate_client = gate_client(
        base_url.as_str(),
        code_review_api_key(b.keys),
        &gate_chain,
        call_timeout,
    )
    .map_err(|e| redact_foreign_error(&e).to_string())?;
    let pool = review_pool(
        &b.cfg,
        b.principal_kind,
        &b.endpoints,
        &b.env_overrides,
        &b.probe,
    )?;
    let (configured_ceiling, max_rotations, retry_disabled) = timeout_scale(&b.cfg);
    let factory = TrioFactory {
        cfg: b.cfg,
        principal_kind: b.principal_kind,
        endpoints: b.endpoints,
        creds: TrioKey(
            b.keys
                .magi()
                .value()
                .map(|k| zeroize::Zeroizing::new(k.to_string())),
        ),
        warn_tokens: b.warn_tokens,
        env_overrides: b.env_overrides,
        capability_cache: b.capability_cache,
        probe: b.probe,
    };
    let consult = TrioBatchConsult {
        factory: Arc::new(factory),
        configured_ceiling,
        max_rotations,
        retry_disabled,
    };
    // Built once now, unless the surface already did, so a trio that cannot be built is
    // reported before any review starts.
    if !b.trio_checked {
        consult.check()?;
    }
    let runner = ProductionReviewRunner::new(ReviewRunnerParts {
        workspace: b.workspace,
        pool,
        consult: Arc::new(consult),
        gate_client: Arc::new(gate_client),
        gate_chain,
        max_state_bytes: section
            .max_state_bytes
            .unwrap_or(crate::defaults::DEFAULT_CODE_REVIEW_MAX_STATE_BYTES),
        call_timeout,
        max_rotations,
        retry_disabled,
        agent_timeout_secs: configured_ceiling,
        structured: b.structured,
    })
    .map_err(|e| redact_foreign_text(&e.to_string()).to_string())?;
    Ok(Arc::new(runner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoVaultInScope;
    use magi_rs::magi::endpoint::EndpointTemplate;
    use magi_rs::review::window::pool_window;

    /// A trio build that takes its duration and then fails, as a slow or stuck build would.
    struct SlowBuild(Duration);
    impl TrioBuilder for SlowBuild {
        fn build(&self, _ceiling: ResolvedCeiling) -> Result<Arc<Magi>, String> {
            std::thread::sleep(self.0);
            Err("built too late".into())
        }
    }

    fn slow_consult(build: Duration) -> TrioBatchConsult {
        TrioBatchConsult {
            factory: Arc::new(SlowBuild(build)),
            configured_ceiling: 120,
            max_rotations: 0,
            retry_disabled: false,
        }
    }

    #[test]
    fn the_early_trio_check_redacts_the_build_error() {
        // [MAGI S6 loop 5, D-m1] the one error path of the runner builder that went out raw.
        struct LeakyBuild;
        impl TrioBuilder for LeakyBuild {
            fn build(&self, _ceiling: ResolvedCeiling) -> Result<Arc<Magi>, String> {
                Err("seat failed at https://user:hunter2@example.invalid/v1".into())
            }
        }
        let consult = TrioBatchConsult {
            factory: Arc::new(LeakyBuild),
            configured_ceiling: 120,
            max_rotations: 0,
            retry_disabled: false,
        };
        let err = consult.check().expect_err("the build fails");
        assert!(!err.contains("hunter2"), "{err}");
        assert!(err.contains("seat failed"), "{err}");
    }

    /// How long a slow build takes in these tests, and the bound the review must answer within.
    const SLOW_BUILD: Duration = Duration::from_millis(1_500);
    const PROMPT_ANSWER: Duration = Duration::from_millis(1_000);

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_trio_build_runs_inside_the_batch_wall_clock() {
        // [MAGI §6 MS2 D loop 3] the build is part of the batch: its wall clock bounds it.
        let consult = slow_consult(SLOW_BUILD);
        let started = std::time::Instant::now();
        let r = consult
            .consult("p", Some(Duration::from_millis(100)))
            .await
            .err();
        assert_eq!(r, Some(ConsultFailure::ClockExpired));
        assert!(started.elapsed() < PROMPT_ANSWER, "{:?}", started.elapsed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_the_run_does_not_wait_for_the_trio_build() {
        // [MAGI §6 MS2 D loop 3] the run's token is honored while a batch's trio is built.
        use crate::tools::code_review::run_cancellable;
        use magi_rs::review::error::ReviewError;
        let consult = slow_consult(SLOW_BUILD);
        let cancel = tokio_util::sync::CancellationToken::new();
        let fire = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            fire.cancel();
        });
        let started = std::time::Instant::now();
        let r = run_cancellable(&cancel, async {
            consult
                .consult("p", None)
                .await
                .map(|_| ())
                .map_err(|e| ReviewError::Io(format!("{e:?}")))
        })
        .await;
        assert!(matches!(r, Err(ReviewError::Cancelled)), "{r:?}");
        assert!(started.elapsed() < PROMPT_ANSWER, "{:?}", started.elapsed());
    }

    #[test]
    fn a_seat_repeated_as_a_non_adjacent_fallback_is_listed_once() {
        // `bal` is the second seat and the second fallback: the repeat is not next to it.
        let cfg = MagiConfig::from_toml_str(
            "provider = \"ollama\"\n\
             [magi]\n\
             melchior_model = \"mel\"\nmelchior_lineage = \"lin-m\"\n\
             balthasar_model = \"bal\"\nbalthasar_lineage = \"lin-b\"\n\
             caspar_model = \"cas\"\ncaspar_lineage = \"lin-c\"\n\
             [[magi.fallback]]\n\
             model = \"other\"\nlineage = \"lin-o\"\n\
             [[magi.fallback]]\n\
             model = \"bal\"\nlineage = \"lin-r\"\n",
        )
        .expect("valid");
        let tpl = EndpointTemplate::parse("http://localhost:11434/v1", Scope::Root).expect("url");
        let endpoints = ResolvedEndpoints {
            root: tpl.resolve(&mut NoVaultInScope, Scope::Root).expect("root"),
            magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).expect("magi"),
        };
        let pool = review_pool(
            &cfg,
            ProviderKind::Ollama,
            &endpoints,
            &MagiEnvModelOverrides::default(),
            &ProbeOutcome::default(),
        )
        .expect("pool");
        let unmeasured = pool_window(&pool).unmeasured;
        assert_eq!(unmeasured, vec!["mel", "bal", "cas", "other"]);
    }

    #[test]
    fn a_pool_error_is_redacted_like_every_other_build_error() {
        // [MAGI §6 MS2 D loop 2] an unknown `[magi].kind` is echoed by its parse error; a value
        // shaped like a credential URL must not reach the surface's notice in clear.
        let cfg = MagiConfig::builder()
            .magi(crate::config::MagiSectionConfig {
                kind: Some("https://user:hunter2@example.invalid".to_string()),
                ..crate::config::MagiSectionConfig::default()
            })
            .build_unvalidated();
        let tpl = EndpointTemplate::parse("http://localhost:11434/v1", Scope::Root).expect("url");
        let endpoints = ResolvedEndpoints {
            root: tpl.resolve(&mut NoVaultInScope, Scope::Root).expect("root"),
            magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).expect("magi"),
        };
        let err = review_pool(
            &cfg,
            ProviderKind::Ollama,
            &endpoints,
            &MagiEnvModelOverrides::default(),
            &ProbeOutcome::default(),
        )
        .expect_err("an unknown kind");
        assert!(err.contains("unknown provider"), "{err}");
        assert!(!err.contains("hunter2"), "{err}");
    }

    #[tokio::test]
    async fn an_answer_under_the_threshold_reaches_the_gate_as_low_confidence() {
        // [MAGI §6 MS2 D loop 1, carried from C7] the production client, over HTTP: an answer
        // under the model's threshold must come back as an answer for the gate to weigh, not as
        // a failed attempt of the client's own (which would leave the finding `unfiltered`).
        use magi_core::schema::{AgentName, Category, Severity};
        use magi_rs::review::consolidate::ConsolidatedFinding;
        use magi_rs::review::gate::{Gate, GateState};
        let mut server = mockito::Server::new_async().await;
        // Magi's confidence for two options is 2p - 1: p = 0.8 gives 0.6, under 0.9.
        let body = serde_json::json!({"model": "a", "answers": {"gate": {"type": "choice",
            "choice": "false_positive",
            "probabilities": {"false_positive": 0.8, "real_defect": 0.2}}}});
        let m = server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_body(body.to_string())
            .expect(1)
            .create_async()
            .await;
        let chain = vec![ModelSpec {
            model: "a".into(),
            min_confidence: Some(0.9),
        }];
        let client = gate_client(
            &format!("{}/v1", server.url()),
            None,
            &chain,
            Duration::from_secs(10),
        )
        .expect("client");
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
        let mut gate = Gate::new(&client, chain, 61_000);
        let v = gate.judge(&finding, Some("fn a() {}\n"), None).await;
        assert_eq!(v.state, GateState::LowConfidence, "{v:?}");
        assert_eq!(v.lean.as_deref(), Some("false_positive"));
        assert!(
            v.confidence.is_some_and(|c| (c - 0.6).abs() < 1e-6),
            "{v:?}"
        );
        m.assert_async().await;
    }
}
