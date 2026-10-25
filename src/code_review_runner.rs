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

impl TrioFactory {
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
    factory: TrioFactory,
    /// `[magi].agent_timeout_secs`, resolved.
    configured_ceiling: u64,
    /// `[magi].max_rotations`, resolved.
    max_rotations: u32,
    /// `[magi].retry_disabled`, resolved.
    retry_disabled: bool,
}

impl TrioBatchConsult {
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
        let magi = self
            .factory
            .build(self.ceiling(wall_clock))
            .map_err(|e| ConsultFailure::Other(redact_foreign_text(&e).to_string()))?;
        run_with_wall_clock(wall_clock, async move {
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
    let kind = resolve_magi_kind(cfg, principal_kind).map_err(|e| e.to_string())?;
    let backend = resolve_backend_model(cfg, kind);
    let view = candidate_window_view(cfg, endpoints, principal_kind, probe);
    let mut models: Vec<String> = seats_with_env_overrides(cfg, backend, env_overrides)
        .into_iter()
        .map(|(_, model)| model)
        .collect();
    models.extend(cfg.fallback_pool().iter().map(|f| f.model.clone()));
    models.dedup();
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
    // The gate applies each model's threshold itself, so the client's chain carries none.
    let client_chain = gate_chain
        .iter()
        .map(|m| ModelSpec {
            model: m.model.clone(),
            min_confidence: None,
        })
        .collect();
    let gate_client = SystemOneClient::new(DecisionSettings {
        base_url: base_url.as_str().to_string(),
        api_key: code_review_api_key(b.keys),
        chain: client_chain,
        timeout: call_timeout,
        keep_alive: None,
    })
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
        factory,
        configured_ceiling,
        max_rotations,
        retry_disabled,
    };
    // Built once now, unless the surface already did, so a trio that cannot be built is
    // reported before any review starts.
    if !b.trio_checked {
        consult.factory.build(consult.ceiling(None))?;
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
