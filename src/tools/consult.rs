// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Tool that wraps `magi_core::Magi` to run 3-perspective consensus queries.
//! The agent routes here only for genuine multi-perspective decisions; trivial
//! or factual lookups are answered directly.

use crate::config::MagiConfig;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use magi_core::error::MagiError;
use magi_core::orchestrator::{Magi, MagiConfig as CoreMagiConfig};
use magi_core::reporting::{ExtractionFailure, InputSize, MagiReport};
use magi_core::schema::{AgentName, Mode};
use magi_rs::magi::kind::ProviderKind;
use magi_rs::magi::mode::{read_resolved_mode, ModeResolution, ModeSource};
use magi_rs::magi::TimeoutDecision;
use magi_rs::redact::redact_foreign_error;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Reject oversized consult input before incurring 3 model calls.
/// `pub(crate)` so the forced `/consult` TUI path and the headless direct/forced
/// consult path ([`crate::headless_runner`]) apply the same cap (REQ-H33).
pub(crate) const MAX_QUERY_LEN: usize = 8192;

/// El prefijo ESTABLE con el que magi-core renderiza la causa de un asiento cuando el
/// fallo fue de autenticación — el `Display` de `ProviderError::Auth` (magi-core 3.1.0,
/// `error.rs`: `#[error("auth error: {message}")]`) (REQ-A12c, SC-A12f).
///
/// **Por qué esto, y no `ExternalErrorKind::Auth`.** `ExternalErrorKind` (`error.rs`,
/// junto a `ProviderError::external`) es para providers de TERCEROS implementados
/// FUERA de magi-core — su propia rustdoc lo dice: *"Failure reported by an LlmProvider
/// implemented OUTSIDE this crate"*, y su único constructor alcanzable desde otro crate
/// es `ProviderError::external(...)`. REQ-A01 prohíbe que magi-rs implemente
/// `LlmProvider`, así que esa vía es ESTRUCTURALMENTE inalcanzable para el trío nativo.
/// Lo que el trío SÍ produce, verificado leyendo las dos implementaciones nativas y sus
/// propios tests: `OpenAiCompatibleProvider::map_status_to_error` (magi-core
/// `src/providers/openai_compat.rs:278-289`) y `ClaudeProvider::map_status_to_error`
/// (`src/providers/claude.rs:239-254`) mapean AMBAS 401 y 403 a
/// `ProviderError::Auth{message}` — no a `ProviderError::Http`. Pineado por
/// `map_status_to_error_maps_401_and_403_to_auth` en `openai_compat.rs` y por
/// `map_status_to_error_maps_401_to_auth`/`_403_to_auth` en `claude.rs`.
///
/// **Dónde vive esta cadena, exactamente — y dónde NO llega.** `MagiError::Provider`
/// tiene `#[error(transparent)]`, así que su `Display` delega íntegro al de
/// `ProviderError`. `dispatch_one_agent` (magi-core `orchestrator.rs:1298-1304`)
/// construye la causa del PRIMER intento con
/// `MagiError::Provider(provider_err).to_string()` — sin ningún prefijo adicional en
/// el caso común (sin reintento, que solo se dispara ante `Validation`/
/// `Deserialization`, nunca ante un error de provider) — y esa cadena es exactamente lo
/// que termina en `MagiReport::failed_agents`. `.contains(...)`, no un prefijo exacto,
/// porque el camino de reintento SÍ antepone `"retry-failed: "`.
///
/// **El alcance real de esta detección es MÁS ANGOSTO de lo que parece — ver
/// [`keyless_auth_explanation`] y el reporte de esta tarea.**
const PROVIDER_AUTH_ERROR_MARKER: &str = "auth error: ";

/// El núcleo REUTILIZABLE de la explicación keyless (REQ-A12c, B3) — UNA sola
/// redacción, consumida por los DOS caminos alcanzables de este archivo:
/// [`keyless_auth_explanation`] (evidencia POSITIVA: el marcador
/// [`PROVIDER_AUTH_ERROR_MARKER`] presente en la causa de UN asiento, vía
/// `MagiReport::failed_agents`) y [`explain_magi_error`] (SIN evidencia de status —
/// solo "0 de 3 asientos corrieron bajo un kind keyless", vía
/// `MagiError::InsufficientAgents`). Fix round 3: antes había una sola redacción con
/// una apertura ya diagnosticada ("el endpoint rechazó..."), correcta para el primer
/// camino pero una afirmación no respaldada en el segundo. En vez de escribir una
/// segunda redacción (que B3 prohíbe — dos textos que pueden divergir con el tiempo),
/// esta constante se recortó al núcleo que es VERDADERO en los dos casos, y cada
/// llamador antepone su PROPIA frase de encuadre según la evidencia que realmente
/// tiene.
///
/// **Deliberadamente en modo CONDICIONAL** ("si tu endpoint LA EXIGE...", nunca "tu
/// endpoint LA EXIGIÓ"): por sí sola, ya sirve para el camino SIN evidencia
/// ([`explain_magi_error`]) sin necesitar edición — REQ-A12c pide nombrar la
/// configuración como **causa PROBABLE**, no demostrada ("no se pide una validación
/// imposible... se exige que el fallo inevitable llegue explicado"), y ese registro es
/// el que tiene que sobrevivir en las dos superficies.
const KEYLESS_AUTH_EXPLANATION: &str = "`[magi].kind = \"ollama\"` es keyless y nunca envía \
     credencial. Si tu endpoint la exige, usá `kind = \"openai-compat\"` y declará la \
     clave por env o vault.";

/// Traduce la causa de UN asiento —tal como aparece en `MagiReport::failed_agents`— a
/// un error de configuración accionable, cuando esa causa es una autenticación
/// rechazada BAJO un kind keyless (REQ-A12c, SC-A12f).
///
/// Devuelve `None` para cualquier otra combinación: bajo un kind que SÍ lleva
/// credencial (`openai-compat`/`anthropic`), un 401/403 puede ser una credencial
/// genuinamente mala, y reinterpretarlo como error de configuración mandaría al
/// usuario a revisar el archivo equivocado — un diagnóstico ACTIVAMENTE incorrecto,
/// no un guard de más.
///
/// **No interpola `cause` en el resultado.** El mensaje es texto fijo: no hay nada
/// derivado de la causa —de un TERCERO, por definición no confiable (B11)— en la
/// salida, así que no hay superficie de fuga que redactar. Cubierto por
/// `keyless_auth_explanation_never_echoes_the_raw_cause`.
///
/// # Alcance real — LEER ANTES DE ASUMIR QUE ESTO CUBRE TODO 401 KEYLESS
///
/// `failed_agents` solo existe cuando `Magi::analyze()` devuelve `Ok(MagiReport)`, y
/// eso exige `successful.len() >= min_agents` — 2, el default de `ConsensusConfig`
/// que REQ-A15 prohíbe exponer (`consensus.rs`: `impl Default for ConsensusConfig`).
/// Verificado contra `orchestrator.rs::dispatch_no_rotation` (magi-core 3.1.0,
/// líneas 1058-1065): cuando MENOS de `min_agents` asientos tienen éxito, la función
/// devuelve `Err(MagiError::InsufficientAgents{succeeded, required})` — **y el mapa
/// `failed` que sí tenía cada causa, incluida una de autenticación, se descarta ahí
/// mismo**; nunca llega a construirse un `MagiReport`, así que esta función nunca
/// llega a invocarse para esos asientos.
///
/// Como el trío de MS2 comparte UN `base_url`/`kind` entre los tres asientos (sin
/// rotación, R-A06), un `kind` mal elegido —el escenario que REQ-A12c describe— rechaza
/// a LOS TRES asientos por igual: 0 de 3 exitosos, `Err(InsufficientAgents{succeeded: 0,
/// required: 2})`, y esta función nunca ve la causa. El caso donde SÍ la ve —exactamente
/// 2 de 3 exitosos, degradado pero `Ok`— es real y esta función lo cubre genuinamente,
/// pero es el caso MENOS probable de "kind mal elegido", no el más. Documentado con
/// evidencia completa en el reporte de esta tarea (ronda 2).
#[must_use]
fn keyless_auth_explanation(cause: &str, kind: ProviderKind) -> Option<&'static str> {
    (kind == ProviderKind::Ollama && cause.contains(PROVIDER_AUTH_ERROR_MARKER))
        .then_some(KEYLESS_AUTH_EXPLANATION)
}

/// Añade, al final del texto ya renderizado por magi-core, una nota por cada asiento
/// de `failed_agents` cuya causa se reconoce como autenticación rechazada bajo un kind
/// keyless (REQ-A12c) — en vez de dejar esa causa completamente invisible, que es lo
/// que pasa hoy: `report_format.rs` (magi-core) no la incluye en `report.report`/
/// `report.banner`, y nada en magi-rs la leía antes de esta tarea.
///
/// **Alcance deliberadamente angosto.** Solo agrega una línea cuando
/// [`keyless_auth_explanation`] reconoce el patrón — no vuelca las demás causas de
/// `failed_agents` sin traducir. Surfacing general de `failed_agents` (REQ-A09/A11d)
/// es una responsabilidad de una tarea posterior (Task 6.1, telemetría); esta función
/// no se adelanta a esa forma para no competir con su diseño.
///
/// El nombre del asiento (`AgentName`, vía `{agent:?}`) es seguro — no es texto de un
/// tercero.
///
/// `pub(crate)` (fix round 4, finding 2): además de [`report_to_consult_json`]
/// (`ConsultTool::execute`, `analyze_direct`), el `/consult` explícito de la TUI
/// (`src/tui/mod.rs::tui_consult_success_body`) llama esto directo — es la MISMA
/// anotación en las tres superficies (B3), nunca una cuarta redacción independiente.
pub(crate) fn annotate_report_text(report: &MagiReport, kind: ProviderKind) -> String {
    let mut text = report.report.clone();
    for (agent, cause) in &report.failed_agents {
        if let Some(explanation) = keyless_auth_explanation(cause, kind) {
            // La apertura "rechazado por autenticación" es una AFIRMACIÓN, y acá está
            // respaldada: `keyless_auth_explanation` solo devolvió `Some` porque
            // `cause` (la causa REAL de este asiento) contenía
            // `PROVIDER_AUTH_ERROR_MARKER` — evidencia positiva. `explain_magi_error`
            // (abajo) no tiene esa evidencia y por eso NO usa esta misma apertura.
            text.push_str(&format!(
                "\n\n**{agent:?}** rechazado por autenticación: {explanation}"
            ));
        }
    }
    text
}

/// How much of a report's markdown survived an output-size recort (REQ-A11b).
///
/// **Not a boolean.** A boolean would only answer *"was it truncated?"*, which is the
/// less useful question: a consumer needs to know **what guarantee it is looking at**.
/// With [`Self::Structural`] it can trust the verdict and at least one finding survived;
/// with [`Self::Bytes`] it knows only that the first N bytes did, and anything else may
/// be missing.
///
/// The actual recort logic (`truncate_report`) is a later task's job — this module only
/// needs the vocabulary because [`Truncated`] is part of [`report_to_consult_json`]'s
/// contract, so every caller of it (this task's two production call sites included) has
/// to be able to name a level even before real truncation exists — hence [`Self::None`]
/// being the only level anything currently produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TruncationLevel {
    /// The report was not truncated.
    None,
    // Narrow allow (matches `MagiConfig::effective_max_query_bytes` in `config.rs`):
    // these three variants have no producer yet — `truncate_report` is a later task's
    // job. They are exercised directly by `report_truncated_names_the_level_applied`.
    /// The verdict and at least one finding were located and kept.
    #[allow(dead_code)]
    Structural,
    /// A magi-core contractual anchor was used instead of locating sections.
    #[allow(dead_code)]
    Anchored,
    /// Only the first N bytes survived; anything past that may be missing.
    #[allow(dead_code)]
    Bytes,
}

/// The report text paired with the [`TruncationLevel`] that produced it.
///
/// A struct, not a `(String, TruncationLevel)` tuple, and every field read by name: the
/// two values have to travel **together**, because a caller free to pass the level
/// separately from the text could put `Bytes` next to text that was never actually cut
/// — and then `report_truncated` in the JSON would be asserting a guarantee about text
/// it does not describe.
#[derive(Debug, Clone)]
pub(crate) struct Truncated {
    /// The (possibly annotated, possibly recorted) text that reaches `"report"` in the
    /// JSON — never the raw [`MagiReport::report`] on its own.
    pub(crate) text: String,
    /// The guarantee `text` carries.
    pub(crate) level: TruncationLevel,
}

/// Per-run signals that belong in THIS run's own JSON output, never a startup notice
/// (REQ-A11d) — a batch consumer parsing yesterday's output never sees today's stderr.
///
/// `schema_version` does not move for this telemetry (REQ-A08b): magi-rs is pre-1.0 and
/// the crate version is already the compatibility signal. That is exactly why the SHAPE
/// has to stay stable instead — a field that appears only on some runs is precisely the
/// instability a consumer with no in-band version signal cannot absorb.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunContext {
    /// `true` when the trio ran on an endpoint different from the principal provider
    /// **and** mode classification was attempted — i.e. the content went out to the
    /// principal first.
    pub(crate) endpoint_divergence: bool,
    /// `true` when an explicit `--timeout` landed below the formula's minimum
    /// (SC-A04d).
    pub(crate) timeout_below_formula: bool,
}

impl RunContext {
    /// Builds the context at the point where all three inputs are actually available —
    /// **before** launching the consult, not while rendering: by render time, why the
    /// run was routed the way it was is no longer at hand.
    ///
    /// Takes [`ModeResolution::classification_attempted`], NOT [`ModeSource`]: a
    /// classification that is attempted and then expires still resolves to
    /// [`ModeSource::Default`], but the content has ALREADY gone out to the principal
    /// provider — and that is exactly the run where declaring the data-flow matters
    /// most, not least. Deriving `endpoint_divergence` from the resulting `source`
    /// instead of the attempt flag would give a false negative right where something
    /// went wrong.
    // Narrow allow: not yet called from production. Both current call sites
    // (`ConsultTool::execute`, `headless_runner::analyze_direct`) build `RunContext`
    // directly as an honest placeholder — neither holds a `MagiConfig` +
    // `TimeoutDecision` pair yet. Wiring this in is deferred to a later task (see
    // this task's report). Exercised directly by its own unit tests.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn build(cfg: &MagiConfig, res: &ModeResolution, timeout: &TimeoutDecision) -> Self {
        Self {
            endpoint_divergence: res.classification_attempted && cfg.magi_endpoint_diverges(),
            timeout_below_formula: timeout.below_formula,
        }
    }
}

/// The typed shape of "none of the three mages produced a verdict" (SC-A11g).
///
/// **Never surfaced as an empty report marked `degraded`.** `degraded` describes a
/// *weakened* consensus (two out of three, say); using it for the zero-verdict case
/// would be a lie — it is not a weaker consensus, it is the absence of one, and a
/// consumer that only inspects the JSON's shape cannot tell the two apart unless the
/// zero-verdict case gets its own type.
// Narrow allow: not yet constructed from production. Translating this into a
// `ToolError`/`ConsultRunError` at a real call site is deferred to a later task —
// this predicate's own rustdoc says so explicitly ("prometer una impl que ninguna
// tarea escribe" — this task does not invent one). Exercised directly by its own
// unit tests.
#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
#[error("no mage produced a verdict: {causes}")]
pub(crate) struct AllMagesFailed {
    /// Every known cause, joined — see [`guard_all_failed`] for how it is built.
    pub(crate) causes: String,
}

/// Guards against emitting a report where **zero** mages produced a verdict.
///
/// # Why `agents.is_empty()`, not `failed_agents.len() == 3`
///
/// [`MagiReport::failed_agents`] only records a seat that failed to **execute** —
/// network, credential, timeout. A seat that DID respond but whose output could not be
/// **extracted** (verdict outside the sentinel markers, invalid schema, or one that
/// exhausted its corrective retry) lands in [`MagiReport::extraction_failures`] too —
/// and, per magi-core's own dispatch, the failed-to-extract case also lands in
/// `failed_agents` alongside it (verified against `orchestrator.rs::dispatch_no_rotation`,
/// magi-core 3.1.0). `report.agents` is the one field that is empty in **every** case a
/// seat ends up without a verdict, regardless of which way it lost it — so it is the
/// only field this predicate can trust.
///
/// # Errors
/// [`AllMagesFailed`] naming every known cause, drawn from BOTH `failed_agents`
/// (execution failures) and `extraction_failures` (adherence failures) — a message that
/// only walked `failed_agents` could come back empty in exactly the case this guard
/// exists to catch.
// Narrow allow: not yet called from production for the same reason as
// `AllMagesFailed` above.
#[allow(dead_code)]
pub(crate) fn guard_all_failed(report: &MagiReport) -> Result<(), AllMagesFailed> {
    if !report.agents.is_empty() {
        return Ok(());
    }
    let mut causes: Vec<String> = report
        .failed_agents
        .iter()
        .map(|(seat, why)| format!("{seat:?}: {why}"))
        .collect();
    causes.extend(report.extraction_failures.iter().map(|(seat, fs)| {
        let last = fs.last().map_or_else(
            || "no detail".to_string(),
            |f| format!("{} attempt {} — {:?}", f.model, f.attempt, f.cause),
        );
        format!("{seat:?}: verdict not extractable ({last})")
    }));
    if causes.is_empty() {
        causes.push("no mage produced a verdict and magi-core reported no cause".to_string());
    }
    Err(AllMagesFailed {
        causes: causes.join("; "),
    })
}

/// Stable JSON labels for the effective mode. Kebab-case, matching how magi-core itself
/// serializes [`Mode`] — this is wire contract, never `Debug`, which would change
/// silently if a variant got renamed.
#[must_use]
fn mode_label(m: Mode) -> &'static str {
    match m {
        Mode::CodeReview => "code-review",
        Mode::Design => "design",
        Mode::Analysis => "analysis",
    }
}

/// See [`mode_label`]. `Configured` keeps its own label instead of collapsing into
/// `explicit`: it shares the semantics (someone chose it, so classification is skipped)
/// but not the provenance, and that difference is what makes an odd verdict auditable —
/// `explicit` sends a reader to the command, `configured` sends them to `magi.toml`.
#[must_use]
fn source_label(s: ModeSource) -> &'static str {
    match s {
        ModeSource::Explicit => "explicit",
        ModeSource::Configured => "configured",
        ModeSource::AgentChosen => "agent-chosen",
        ModeSource::Inferred => "inferred",
        ModeSource::Default => "default",
    }
}

/// See [`mode_label`]. Names the LEVEL, not a boolean: the consumer needs to know what
/// guarantee it has, not merely whether truncation happened (SC-A11h).
#[must_use]
fn truncation_label(l: TruncationLevel) -> &'static str {
    match l {
        TruncationLevel::None => "none",
        TruncationLevel::Structural => "structural",
        TruncationLevel::Anchored => "anchored",
        TruncationLevel::Bytes => "bytes",
    }
}

/// Single source of the lowercase seat key shared by [`failures_json`] and
/// [`failed_agents_json`] (B3) — one casing convention for an `AgentName`-keyed JSON
/// object, not two that could quietly diverge.
#[must_use]
fn seat_key(seat: AgentName) -> String {
    format!("{seat:?}").to_lowercase()
}

/// `extraction_failures`, keyed by lowercase seat name — ALWAYS present, even when
/// empty (REQ-A10). An empty map is a positive certificate that every seat adhered to
/// the verdict contract on every attempt; omitting it would make that certificate
/// indistinguishable from "this version doesn't report it".
#[must_use]
fn failures_json(f: &BTreeMap<AgentName, Vec<ExtractionFailure>>) -> Value {
    let mut out = serde_json::Map::new();
    for (seat, failures) in f {
        out.insert(
            seat_key(*seat),
            Value::Array(
                failures
                    .iter()
                    .map(|x| {
                        json!({
                            "model": x.model,
                            "attempt": x.attempt,
                            "cause": format!("{:?}", x.cause),
                        })
                    })
                    .collect(),
            ),
        );
    }
    Value::Object(out)
}

/// `failed_agents`, keyed the SAME way as [`failures_json`] (see [`seat_key`]) — built
/// by hand rather than serialized straight through `report.failed_agents`, so the two
/// maps cannot end up on different casings for what is the same kind of key.
#[must_use]
fn failed_agents_json(agents: &BTreeMap<AgentName, String>) -> Value {
    let mut out = serde_json::Map::new();
    for (seat, cause) in agents {
        out.insert(seat_key(*seat), Value::String(cause.clone()));
    }
    Value::Object(out)
}

/// `input_size` with its three sub-keys ALWAYS present (SC-A10c).
///
/// `None` from magi-core is neither omitted nor emitted as `null`: it reports what the
/// pipeline actually APPLIED — magi-core's own default as the threshold, and `exceeded`
/// evaluated against that — never a bare "I don't know" that a consumer could confuse
/// with an unmeasured `0`.
#[must_use]
fn input_size_json(s: Option<&InputSize>) -> Value {
    match s {
        Some(v) => json!({
            "estimated_tokens": v.estimated_tokens,
            "warn_threshold": v.warn_threshold,
            "exceeded": v.exceeded,
        }),
        None => json!({
            "estimated_tokens": 0,
            "warn_threshold": CoreMagiConfig::default().input_warn_tokens,
            "exceeded": false,
        }),
    }
}

/// Builds the stable `consult` JSON object from a finished MAGI report.
///
/// Single source of truth for the shape shared by the tool-loop [`ConsultTool::execute`]
/// path and the headless direct/forced consult path (REQ-H21/H22, REQ-A11c) — so the
/// on-the-wire shape never drifts between the two entry points.
///
/// `schema_version` does not move for this telemetry (REQ-A08b), which is exactly why
/// every field below — and every sub-key of `input_size` — is always present: a field
/// or sub-key that appears only on some runs is the instability a version-less consumer
/// cannot absorb.
///
/// # Parameters
/// * `report` - the finished multi-perspective consensus report.
/// * `truncated` - the report text actually surfaced, paired with the guarantee it
///   carries — see [`Truncated`] for why the two travel together. Callers that also
///   annotate the text (REQ-A12c, [`annotate_report_text`]) must do so **before**
///   building this value: this function renders `truncated.text` verbatim.
/// * `res` - the resolved mode, its source, and whether classification was attempted.
/// * `ctx` - the per-run signals that belong in this run's own output (REQ-A11d).
///
/// # Returns
/// A JSON object with `report`, `degraded`, `mode`, `mode_source`,
/// `extraction_failures`, `input_size`, `report_truncated`, `endpoint_divergence`,
/// `timeout_below_formula` and `failed_agents` — every key always present.
pub(crate) fn report_to_consult_json(
    report: &MagiReport,
    truncated: &Truncated,
    res: &ModeResolution,
    ctx: &RunContext,
) -> Value {
    json!({
        // The (possibly annotated, possibly truncated) text — never the raw report:
        // text and level travel together in `truncated`, so `report_truncated` can
        // never assert a guarantee about text it does not describe.
        "report": truncated.text,
        "degraded": report.degraded,
        "mode": mode_label(res.mode),
        "mode_source": source_label(res.source),
        // Always present, even when empty: an empty map is the positive certificate
        // that all three seats adhered to the contract (REQ-A10).
        "extraction_failures": failures_json(&report.extraction_failures),
        // Always present WITH its sub-keys: an object that is always there but whose
        // contents vary is the same instability one level down (SC-A10c).
        "input_size": input_size_json(report.input_size.as_ref()),
        "report_truncated": truncation_label(truncated.level),
        // REQ-A11d: what affects how THIS run reads goes in THIS run's JSON, not a
        // notice — a batch consumer never sees stderr from a process that already
        // exited.
        "endpoint_divergence": ctx.endpoint_divergence,
        // SC-A04d: whoever sets an explicit `--timeout` is running in a pipeline, i.e.
        // exactly who is LEAST likely to read stderr. The warning travels here too.
        "timeout_below_formula": ctx.timeout_below_formula,
        // Typed, never collapsed into `degraded`: `failed_agents`, NOT `window_rejected`
        // — the latter does not exist on `MagiReport` (verified against magi-core
        // 3.1.0; rotation-only telemetry, unreachable in MS2 without `FallbackPool`).
        "failed_agents": failed_agents_json(&report.failed_agents),
    })
}

/// Explica —AGREGANDO al mensaje de `err`, nunca reemplazándolo— un
/// `MagiError::InsufficientAgents` cuando el kind efectivo es keyless (REQ-A12c,
/// SC-A12f, fix round 3).
///
/// # Por qué existe: la ventana de `keyless_auth_explanation` excluye justo el
/// escenario que REQ-A12c describe
///
/// `keyless_auth_explanation` solo ve una causa cuando `Magi::analyze()` devuelve
/// `Ok(MagiReport)`, y eso exige `successful.len() >= min_agents` (2). Verificado
/// contra `orchestrator.rs::dispatch_no_rotation` (magi-core 3.1.0, líneas
/// 1058-1065): `if successful.len() < min_agents { return
/// Err(MagiError::InsufficientAgents { succeeded, required }) }` — el mapa `failed`,
/// YA completo con cada causa en ese punto, se descarta ahí mismo; `MagiReport` nunca
/// llega a construirse. Como el trío de MS2 comparte UN `base_url`/`kind` entre los
/// tres asientos (sin rotación, R-A06), un `kind` mal elegido —exactamente el
/// escenario de SC-A12f, `kind = "ollama"` contra un endpoint que exige
/// autenticación— rechaza a LOS TRES por igual: 0 de 3 exitosos, este camino, no el
/// otro.
///
/// # Por qué esto SÍ alcanza sin la causa por asiento
///
/// REQ-A12c pide nombrar la configuración como **causa PROBABLE** ("no se pide una
/// validación imposible... se exige que el fallo inevitable llegue explicado"), no
/// una demostrada. La combinación "cero asientos completaron" + "el kind efectivo es
/// keyless" ya es, por sí sola, evidencia suficiente para ese umbral — sin necesitar
/// el código de status que este camino nunca tiene. Por esto el guard de kind sigue
/// siendo obligatorio acá también: bajo un kind CON credencial, un fallo total dice
/// tan poco sobre configuración como cualquier otro outage, y ofrecer la pista mandaría
/// al usuario al archivo equivocado — el mismo daño que
/// [`keyless_auth_explanation`] evita del otro lado.
///
/// # Parameters
/// * `err` - el error que devolvió `Magi::analyze()`.
/// * `kind` - el `ProviderKind` bajo el que corría el trío.
///
/// # Returns
/// El `Display` de `err` (redactado — B11, ver abajo), con la pista keyless
/// AGREGADA (nunca en su lugar) cuando `err` es `InsufficientAgents` y `kind` es
/// `Ollama`. En cualquier otro caso, solo el `Display` de `err`.
///
/// Consumida por `ConsultTool::execute`, `analyze_direct` (headless), Y —desde el fix
/// round 4, finding 2— el `/consult` explícito de la TUI
/// (`src/tui/mod.rs::tui_consult_error_body`): las tres superficies comparten esta
/// única función en vez de cada una escribiendo su propia traducción (B3).
#[must_use]
pub(crate) fn explain_magi_error(err: &MagiError, kind: ProviderKind) -> String {
    // B11 — `redact_foreign_error`, NUNCA `redact_url`, y la diferencia importa acá:
    // `redact_url` asume que TODA la entrada es una URL y redacta por completo
    // cualquier cosa que no pueda recorrer como tal (`locate_userinfo` devuelve
    // `Unparseable` ante cualquier string sin `://`) — aplicado acá habría reducido
    // CADA mensaje de `MagiError` sin URL (p. ej. "insufficient agents: 0 succeeded,
    // 2 required") a `"***"`, un bug real que atrapó
    // `explain_magi_error_preserves_a_url_free_underlying_message` la primera vez que
    // se corrió este test (ver el reporte de esta tarea). `redact_foreign_error`
    // recorre PROSA buscando URLs EMBEBIDAS y redacta solo esas, dejando el resto
    // intacto — la misma primitiva que `build_native_provider::to_seat` ya usa para
    // el mismo problema: un `Display` foráneo que PODRÍA traer una URL, no que ES una.
    // `MagiError` es `#[non_exhaustive]`, así que una variante futura sí podría
    // interpolar una URL; se redacta siempre, no solo para las variantes de hoy.
    let base = redact_foreign_error(err);
    match (err, kind) {
        (MagiError::InsufficientAgents { .. }, ProviderKind::Ollama) => {
            format!("{base} — posible causa: {KEYLESS_AUTH_EXPLANATION}")
        }
        _ => base.to_string(),
    }
}

/// RAII backstop that aborts a spawned task when the guard is dropped.
///
/// [`ConsultTool::execute`] runs the 3-perspective analysis on a `tokio::spawn`
/// task and awaits it under a `select!`. The explicit cancel arm aborts the
/// task on `--timeout`, but if the `execute` future itself is *dropped* before
/// either arm resolves (e.g. the caller drops the tool call), a bare spawned
/// task would keep running and orphan its three in-flight LLM calls. Holding
/// this guard across the `select!` aborts the task on that drop too, mirroring
/// the `GroupKiller` backstop the `bash` tool uses for its subprocess.
///
/// `pub(crate)` so [`crate::headless_runner`]'s direct `magi consult` path
/// (`analyze_direct`) reuses this exact primitive for its own spawned MAGI
/// analysis rather than duplicating it — same gap, same fix, one guard type.
pub(crate) struct AbortOnDrop {
    /// Abort handle of the guarded task.
    handle: tokio::task::AbortHandle,
}

impl AbortOnDrop {
    /// Wraps a task's abort handle so dropping the guard aborts the task.
    pub(crate) fn new(handle: tokio::task::AbortHandle) -> Self {
        Self { handle }
    }

    /// Aborts the guarded task now. Idempotent: aborting an already-finished or
    /// already-aborted task is a no-op, so `Drop` re-invoking it is harmless.
    pub(crate) fn abort(&self) {
        self.handle.abort();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Notice emitted in the TUI when the `consult` tool is auto-approved.
/// Visible to the user so they know the 3-LLM consensus was launched.
const AUTO_LAUNCH_NOTICE: &str = "launched MAGI multi-perspective consensus — awaiting evaluation…";

/// Resolves the (mode, source) pair to dispatch with (MS2, REQ-A20/REQ-A07d/REQ-A08):
/// reads what the agent's tool loop already resolved and injected under the reserved
/// `__resolved_mode`/`__resolved_mode_source` keys
/// (`magi_rs::magi::mode::inject_resolved_mode`) instead of re-resolving it here
/// — a tool that could disagree with the gate that evaluated the same call
/// would reopen exactly the divergence REQ-A07d closes.
///
/// Falls back to `(`[`Mode::Analysis`]`, `[`ModeSource::Default`]`)` when the keys are
/// **absent**. This is a deliberate, narrow back-compat default, not a re-resolution of
/// untrusted input: **both** real production dispatch paths in `Agent::run_tool_loop`
/// inject the pair before calling [`ConsultTool::execute`] — the model-issued
/// `ToolUse` route (`Agent::dispatch_consult_through_gate`) and the forced
/// pre-loop injection (REQ-H22's `config.force_consult` block, which resolves
/// and injects too, just without ever evaluating the gate: REQ-A20 forbids
/// vetoing a forced consult, but it still needs a resolved mode). So this
/// fallback is reached only by a caller that invokes [`ConsultTool::execute`]
/// directly without going through the loop — exactly what this module's own
/// pre-MS2 tests do, and precisely the unconditional `Mode::Analysis` this
/// tool used before MS2 (the source half is new in Task 6.1, needed to name
/// `mode_source` in [`report_to_consult_json`]'s output). Wiring `ConsultTool`
/// fully into `ConsultToolCfg` (a later task) can tighten this to a hard error
/// once every production caller is confirmed to inject.
fn resolved_mode_and_source(args: &Value) -> (Mode, ModeSource) {
    read_resolved_mode(args).unwrap_or((Mode::Analysis, ModeSource::Default))
}

/// Tool wrapping a `magi_core::Magi`. `execute` runs the 3-perspective consensus
/// (implemented in Task 4) and returns the verbatim report. The `description` is
/// what makes the main LLM self-route here only for multi-perspective decisions.
pub struct ConsultTool {
    magi: Arc<Magi>,
    description: String,
    /// When `true`, autonomous MAGI launches via the agent tool loop are
    /// auto-approved (no `ApprovalRequest` emitted). The explicit `/consult`
    /// TUI command path is NEVER gated regardless of this flag.
    auto_approve: bool,
    /// The `ProviderKind` the trio runs under (REQ-A12c) — feeds
    /// [`keyless_auth_explanation`] via [`report_to_consult_json`]. Defaults to
    /// [`ProviderKind::OpenAiCompat`] (see [`Self::new`]), under which the
    /// explanation never applies — a caller that does not care about this
    /// feature does not need to call [`Self::with_kind`].
    kind: ProviderKind,
}

impl ConsultTool {
    /// Creates a `ConsultTool` over a shared `Magi` orchestrator.
    ///
    /// # Parameters
    /// * `magi` - Shared `Magi` orchestrator that drives the 3-perspective consensus.
    /// * `auto_approve` - When `true`, the tool opts out of the approval gate for
    ///   autonomous launches (the agent tool loop will auto-approve it and emit a
    ///   TUI notice). Default is `false` — the agent asks before each launch.
    ///
    /// # Returns
    /// A new `ConsultTool` instance with a routing-tuned description and `kind`
    /// defaulted to [`ProviderKind::OpenAiCompat`] — call [`Self::with_kind`] to
    /// declare the trio's real kind when REQ-A12c's explanation should apply.
    pub fn new(magi: Arc<Magi>, auto_approve: bool) -> Self {
        Self {
            magi,
            description: "Run a multi-perspective MAGI consensus (three independent \
                analyst agents) on a hard decision. Use ONLY for questions with genuine \
                trade-offs, design/architecture choices, or 'should we X vs Y given these \
                constraints?' decisions where a single answer is risky. Do NOT use for \
                trivial, factual, or lookup questions — answer those directly."
                .to_string(),
            auto_approve,
            // Neutral: `keyless_auth_explanation` only ever fires under `Ollama`, so
            // this default is equivalent to the feature being off until declared.
            kind: ProviderKind::OpenAiCompat,
        }
    }

    /// Declares the `ProviderKind` the trio runs under (REQ-A12c).
    ///
    /// Builder-style (`self` by value) so production call sites read as
    /// `ConsultTool::new(magi, auto_approve).with_kind(kind)` without a second
    /// mutable binding, and so the ~13 existing test call sites that do not care
    /// about this feature do not need to change at all.
    #[must_use]
    pub fn with_kind(mut self, kind: ProviderKind) -> Self {
        self.kind = kind;
        self
    }
}

#[async_trait]
impl Tool for ConsultTool {
    fn name(&self) -> &str {
        "consult"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The decision or content to analyze from three perspectives."
                },
                "mode": {
                    "type": "string",
                    "enum": ["code-review", "design", "analysis"],
                    "description": "Optional lens for the analysis (pick the one that \
                        matches what you're asking about). Omit to let the caller's \
                        configured/inferred lens apply instead."
                }
            },
            "required": ["query"]
        })
    }

    /// When `auto_approve = false` (the default), autonomous MAGI launches are
    /// gated — the agent prompts the user before each 3-LLM consensus call.
    /// When `auto_approve = true`, the agent tool loop auto-approves the call
    /// and emits an [`Self::approval_notice`] in the TUI instead.
    fn requires_approval(&self) -> bool {
        !self.auto_approve
    }

    /// Returns an announcement notice when the tool is auto-approved.
    ///
    /// The notice is sent as a `StreamPiece::Notice` **before** the tool runs,
    /// so the user knows the 3-LLM consensus was launched without a prompt.
    /// Returns `None` when `auto_approve = false` (the gate prompts the user
    /// instead, so no proactive notice is needed).
    fn approval_notice(&self) -> Option<String> {
        if self.auto_approve {
            Some(AUTO_LAUNCH_NOTICE.to_string())
        } else {
            None
        }
    }

    async fn execute(&self, args: Value, cancel: &CancellationToken) -> ToolResult<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("missing 'query' string".to_string()))?;
        if query.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "query must not be empty".to_string(),
            ));
        }
        if query.len() > MAX_QUERY_LEN {
            return Err(ToolError::InvalidArguments(format!(
                "query too large ({} bytes; max {})",
                query.len(),
                MAX_QUERY_LEN
            )));
        }
        let (mode, source) = resolved_mode_and_source(&args);
        let magi = self.magi.clone();
        let q = query.to_string();
        // Joined spawn isolates a panic in magi-core's analyze into a recoverable
        // JoinError instead of unwinding into the agent tool loop.
        let handle = tokio::spawn(async move { magi.analyze(&mode, &q).await });
        // RAII backstop: aborts the spawned analysis if this `execute` future is
        // dropped before the `select!` resolves — a dropped tool call would
        // otherwise orphan the three in-flight LLM calls. The abort handle is
        // taken separately from `handle` (which the `select!` consumes), so
        // there is no borrow conflict.
        let abort_guard = AbortOnDrop::new(handle.abort_handle());
        // A proactive consult runs three MAGI LLM calls; on the run's `--timeout`
        // cancellation (REQ-H36) the task is **aborted** — not merely detached —
        // so those expensive API calls actually stop instead of being orphaned.
        // `biased` polls the cancel arm first, so an already-cancelled token
        // short-circuits before the analysis is awaited.
        let report = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                abort_guard.abort();
                return Err(ToolError::ExecutionError(
                    "consult cancelled by timeout".to_string(),
                ));
            }
            joined = handle => match joined {
                Ok(Ok(report)) => report,
                Ok(Err(e)) => {
                    return Err(ToolError::ExecutionError(explain_magi_error(&e, self.kind)))
                }
                Err(join_err) => {
                    return Err(ToolError::ExecutionError(format!(
                        "consult crashed: {join_err}"
                    )))
                }
            },
        };
        // The annotation (REQ-A12c) is applied BEFORE wrapping in `Truncated` — the new
        // `report_to_consult_json` renders `truncated.text` verbatim, so the annotation
        // step has to happen here, not inside it, or the keyless-auth hint this file
        // already surfaces end-to-end (see `a_keyless_auth_failure_reaches_the_
        // consult_report_end_to_end`) would silently stop reaching the user.
        let truncated = Truncated {
            text: annotate_report_text(&report, self.kind),
            // No truncation logic exists yet (`truncate_report` is a later task's job);
            // this is the untruncated report, honestly labeled as such.
            level: TruncationLevel::None,
        };
        let res = ModeResolution {
            mode,
            source,
            // Not yet threaded through the injected resolution: `inject_resolved_mode`
            // only round-trips mode+source today. Inert here — this call site builds
            // `RunContext` directly rather than via `RunContext::build`, so this field
            // is not read. Wiring it end-to-end is left to a later task.
            classification_attempted: false,
        };
        let ctx = RunContext {
            // Placeholder until `MagiConfig`/`TimeoutDecision` are threaded to this call
            // site (a later wiring task) — this tool does not hold either today.
            endpoint_divergence: false,
            timeout_below_formula: false,
        };
        Ok(report_to_consult_json(&report, &truncated, &res, &ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::error::{ExternalErrorKind, ProviderError};
    use magi_core::provider::{CompletionConfig, LlmProvider};
    use magi_core::test_support::RoutingMockProvider;
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
    use magi_rs::magi::{resolve_run_timeout, AGENT_TIMEOUT_SECS};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Upper bound on how long a *cancelled* `execute` may take to return. The
    /// cancel path aborts the in-flight analysis, so it must resolve almost
    /// immediately; sized generously to absorb scheduler jitter while staying
    /// far below the blocking provider's [`BlockingProvider::SLEEP_SECS`] sleep,
    /// so a regression that awaited the full analysis would blow this budget.
    const CANCEL_RETURN_BUDGET_MS: u128 = 2_000;

    /// A provider whose `complete` blocks far longer than any test tolerates, so
    /// a MAGI analysis over it never finishes within the test. Used to prove that
    /// [`ConsultTool::execute`] returns on cancellation *without* waiting for the
    /// analysis: if the cancel token were ignored, `execute` would block on this
    /// sleep and overrun [`CANCEL_RETURN_BUDGET_MS`].
    struct BlockingProvider;

    impl BlockingProvider {
        /// Sleep duration of each `complete` call — long enough that awaiting the
        /// full analysis is unmistakably distinguishable from a prompt cancel.
        const SLEEP_SECS: u64 = 3_600;
    }

    #[async_trait]
    impl LlmProvider for BlockingProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _config: &CompletionConfig,
        ) -> Result<String, ProviderError> {
            tokio::time::sleep(Duration::from_secs(Self::SLEEP_SECS)).await;
            Ok(String::new())
        }

        fn name(&self) -> &str {
            "blocking"
        }

        fn model(&self) -> &str {
            "blocking"
        }
    }

    /// Helper: constructs a `ConsultTool` with `auto_approve = false` (the default).
    fn dummy_tool() -> ConsultTool {
        ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            false,
        )
    }

    /// Respuesta canónica de un mage, en el formato que magi-core 3.x exige.
    ///
    /// Desde 3.0.0 el veredicto se lee **solo** entre [`VERDICT_OPEN`] y
    /// [`VERDICT_CLOSE`], cada marcador solo en su línea. Un JSON pelado —el
    /// formato que servía en 2.x— ya no se parsea y el mage cuenta como fallido.
    /// Se usan las constantes del crate en vez de literales para que un cambio
    /// de marcador rompa la compilación en vez de degradar el fixture en silencio.
    fn agent_json(agent: &str) -> String {
        let verdict = format!(
            r#"{{"agent":"{agent}","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}}"#
        );
        format!(
            "{VERDICT_OPEN}
{verdict}
{VERDICT_CLOSE}"
        )
    }

    fn magi_all_ok() -> Arc<Magi> {
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Ok(agent_json("caspar"))]);
        Arc::new(Magi::new(Arc::new(provider)))
    }

    /// Two seats succeed, one fails on a REAL `ProviderError::Auth` (magi-core's own
    /// `ClaudeProvider::map_status_to_error(401, ..)` — public, not hand-rolled) —
    /// the ONE case genuinely reachable via `MagiReport::failed_agents`
    /// (`min_agents = 2`, `ConsensusConfig::default()`, verified in
    /// `keyless_auth_explanation`'s own rustdoc). Caspar is the failing seat so the
    /// tests below can assert on its name specifically.
    fn magi_caspar_fails_with_auth_error() -> Arc<Magi> {
        let auth_err = magi_core::providers::claude::ClaudeProvider::map_status_to_error(
            401,
            "x",
            vec![],
            None,
        );
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Err(auth_err)]);
        Arc::new(Magi::new(Arc::new(provider)))
    }

    // -----------------------------------------------------------------------
    // Task 4.4 (fix round 2) — REQ-A12c/SC-A12f: the keyless-auth translation
    // -----------------------------------------------------------------------

    /// SC-A12f: a real `ProviderError::Auth` rendering — as magi-core itself
    /// produces and pins it, not a hand-rolled string — is recognized as a keyless
    /// auth failure under `ollama`.
    ///
    /// **Positive control (mandatory per this task's fix-round-1 review).** Both
    /// `OpenAiCompatibleProvider::map_status_to_error` (magi-core
    /// `src/providers/openai_compat.rs:280`) and `ClaudeProvider::map_status_to_error`
    /// (`src/providers/claude.rs:247`) map 401/403 to `ProviderError::Auth` — the
    /// same enum variant, same crate, same `Display`. `ClaudeProvider`'s mapper is
    /// `pub fn` (the OpenAI-compat one is `pub(crate)`, unreachable from here), so
    /// it is used to construct the REAL value; the contract pinned by this test
    /// (`ProviderError::Auth`'s `"auth error: "` `Display` prefix) is a property of
    /// `error.rs`, shared identically by both providers regardless of which one
    /// built the value. If a future magi-core release changes that wording, THIS
    /// test goes red — not silently false forever.
    #[test]
    fn keyless_auth_explanation_recognizes_a_real_provider_error_auth_rendering() {
        let provider_err = magi_core::providers::claude::ClaudeProvider::map_status_to_error(
            401,
            "x",
            vec![],
            None,
        );
        // The exact composition `dispatch_one_agent` performs on a first-attempt
        // provider failure (magi-core `orchestrator.rs:1298-1304`):
        // `MagiError::Provider(provider_err).to_string()`.
        let cause = magi_core::error::MagiError::Provider(provider_err).to_string();
        assert_eq!(
            keyless_auth_explanation(&cause, ProviderKind::Ollama),
            Some(KEYLESS_AUTH_EXPLANATION),
            "a real magi-core auth rendering must be recognized: {cause:?}"
        );
    }

    /// SC-A12f: under a kind that DOES carry a credential, the same real rendering
    /// is left alone — a 401 there can be a genuinely bad credential, and
    /// reinterpreting it would send the user to the wrong file.
    #[test]
    fn keyless_auth_explanation_does_not_reinterpret_under_a_credentialed_kind() {
        let provider_err = magi_core::providers::claude::ClaudeProvider::map_status_to_error(
            401,
            "x",
            vec![],
            None,
        );
        let cause = magi_core::error::MagiError::Provider(provider_err).to_string();
        for kind in [ProviderKind::OpenAiCompat, ProviderKind::Anthropic] {
            assert_eq!(
                keyless_auth_explanation(&cause, kind),
                None,
                "kind {kind:?} carries a credential: a 401 there is not reinterpreted"
            );
        }
    }

    /// Edge case (B13): a cause that is NOT an auth failure (e.g. a timeout) is
    /// never reinterpreted, even under `ollama`.
    #[test]
    fn keyless_auth_explanation_ignores_causes_without_the_auth_marker() {
        let cause = "timeout: agent timed out after 90s";
        assert_eq!(keyless_auth_explanation(cause, ProviderKind::Ollama), None);
    }

    /// B11: the explanation is FIXED text — it never echoes `cause`, so a secret
    /// that somehow ended up in third-party diagnostic text cannot reach the
    /// surfaced message through this function.
    #[test]
    fn keyless_auth_explanation_never_echoes_the_raw_cause() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let cause = format!("{PROVIDER_AUTH_ERROR_MARKER}token={CANARY}");
        let explanation =
            keyless_auth_explanation(&cause, ProviderKind::Ollama).expect("marker present");
        assert!(!explanation.contains(CANARY), "{explanation}");
    }

    /// SC-A12f, end to end: a seat that fails on auth under `ollama` reaches the
    /// user through the ACTUAL `ConsultTool::execute` → `report_to_consult_json`
    /// path, not just the pure predicate. This is the wiring proof: a correct
    /// `keyless_auth_explanation` nobody calls from `execute` would pass every test
    /// above and still leave the user looking at the raw "auth error: x" text.
    #[tokio::test]
    async fn a_keyless_auth_failure_reaches_the_consult_report_end_to_end() {
        let tool = ConsultTool::new(magi_caspar_fails_with_auth_error(), false)
            .with_kind(ProviderKind::Ollama);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("2 of 3 succeed ⇒ Ok, degraded");
        assert_eq!(out["degraded"], json!(true));
        let report = out["report"].as_str().expect("report string");
        assert!(report.contains("Caspar"), "{report}");
        assert!(
            report.contains("keyless") && report.contains("openai-compat"),
            "the explanation must reach the surfaced report: {report}"
        );
    }

    /// Same failing seat, but under a kind that carries a credential: the raw
    /// cause reaches the report (nothing hides it), but it is NEVER reinterpreted
    /// as a keyless-configuration problem.
    #[tokio::test]
    async fn a_keyless_auth_failure_is_not_annotated_under_a_credentialed_kind() {
        let tool = ConsultTool::new(magi_caspar_fails_with_auth_error(), false)
            .with_kind(ProviderKind::OpenAiCompat);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("2 of 3 succeed ⇒ Ok, degraded");
        assert_eq!(out["degraded"], json!(true));
        let report = out["report"].as_str().expect("report string");
        assert!(
            !report.contains("keyless"),
            "openai-compat carries a credential: no reinterpretation: {report}"
        );
    }

    // -----------------------------------------------------------------------
    // Task 4.4 (fix round 3) — REQ-A12c/SC-A12f: the total-failure window
    // -----------------------------------------------------------------------

    /// All three seats fail on a REAL `ProviderError::Auth` — the case
    /// `keyless_auth_explanation`/`annotate_report_text` CANNOT see (0 of 3
    /// succeeded < `min_agents` 2 ⇒ `Magi::analyze()` returns `Err`, and
    /// `MagiReport`/`failed_agents` is never constructed). This is the scenario
    /// SC-A12f actually describes: a `kind` mismatch rejects every seat identically
    /// because they share one `base_url`/`kind` (no rotation, R-A06).
    fn magi_all_fail_with_auth_errors() -> Arc<Magi> {
        let mk = || {
            magi_core::providers::claude::ClaudeProvider::map_status_to_error(
                401,
                "x",
                vec![],
                None,
            )
        };
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Err(mk())])
            .with_agent_responses(AgentName::Balthasar, vec![Err(mk())])
            .with_agent_responses(AgentName::Caspar, vec![Err(mk())]);
        Arc::new(Magi::new(Arc::new(provider)))
    }

    /// SC-A12f: on the total-failure path (`MagiError::InsufficientAgents`, no
    /// per-agent cause available), a keyless kind is enough on its own to name the
    /// configuration as a **probable** cause — REQ-A12c's own words ("causa
    /// probable", not demostrada). The hint is ADDED to the underlying message, never
    /// in its place.
    #[test]
    fn explain_magi_error_adds_a_probable_cause_hint_for_insufficient_agents_under_a_keyless_kind()
    {
        let err = MagiError::InsufficientAgents {
            succeeded: 0,
            required: 2,
        };
        let msg = explain_magi_error(&err, ProviderKind::Ollama);
        assert!(
            msg.contains("insufficient agents") || msg.contains("0 succeeded"),
            "the underlying message must survive, not be replaced: {msg}"
        );
        assert!(
            msg.contains("keyless") && msg.contains("openai-compat"),
            "the probable-cause hint must be added: {msg}"
        );
        // Register check: this path has NO per-agent evidence, so it must not borrow
        // the confident opening `annotate_report_text` uses when it DOES have
        // evidence (`PROVIDER_AUTH_ERROR_MARKER` present in a real cause).
        assert!(
            !msg.contains("rechazado por autenticación"),
            "no per-agent evidence here: must not claim auth was the cause: {msg}"
        );
    }

    /// SC-A12f: under a kind that carries a credential, a total failure says nothing
    /// about configuration — the hint must NOT appear, or it would send the user to
    /// the wrong file (same guard [`keyless_auth_explanation`] enforces).
    #[test]
    fn explain_magi_error_never_adds_the_hint_under_a_credentialed_kind() {
        let err = MagiError::InsufficientAgents {
            succeeded: 0,
            required: 2,
        };
        for kind in [ProviderKind::OpenAiCompat, ProviderKind::Anthropic] {
            let msg = explain_magi_error(&err, kind);
            assert!(
                !msg.contains("keyless"),
                "kind {kind:?} carries a credential: no hint: {msg}"
            );
        }
    }

    /// Edge case (B13): the hint is specific to `InsufficientAgents` — a DIFFERENT
    /// `MagiError` variant, even under `ollama`, gets no hint, because "input too
    /// large" says nothing about authentication.
    #[test]
    fn explain_magi_error_never_adds_the_hint_for_a_different_magi_error_variant() {
        let err = MagiError::InputTooLarge {
            size: 10_000,
            max: 5_000,
        };
        let msg = explain_magi_error(&err, ProviderKind::Ollama);
        assert!(!msg.contains("keyless"), "{msg}");
        assert!(msg.contains("10000") || msg.contains("5000"), "{msg}");
    }

    /// Regression: `explain_magi_error` must use `redact_foreign_error`, NEVER
    /// `redact_url`, on the underlying message. An earlier version of this function
    /// used `redact_url`, which assumes its ENTIRE input is a URL and fully redacts
    /// anything it cannot parse as one (`locate_userinfo` returns `Unparseable` for
    /// any string without `://`) — applied to a `MagiError` message with no embedded
    /// URL (the common case: "insufficient agents: 0 succeeded, 2 required" has none),
    /// that reduced the entire diagnostic to a bare `"***"`. This test caught that the
    /// first time it ran; it stays here so a future edit can't silently reintroduce
    /// the wrong primitive.
    #[test]
    fn explain_magi_error_preserves_a_url_free_underlying_message_verbatim() {
        let err = MagiError::InsufficientAgents {
            succeeded: 0,
            required: 2,
        };
        let msg = explain_magi_error(&err, ProviderKind::OpenAiCompat);
        assert_eq!(
            msg,
            err.to_string(),
            "a URL-free message must survive unredacted, verbatim: {msg}"
        );
    }

    /// SC-A12f, end to end: a TOTAL failure (0 of 3 seats) under `ollama` reaches the
    /// user through the ACTUAL `ConsultTool::execute` path, not just the pure
    /// `explain_magi_error` predicate — this is the wiring proof for the window
    /// `annotate_report_text` cannot cover.
    #[tokio::test]
    async fn a_total_seat_failure_under_ollama_surfaces_the_keyless_hint_through_consult_tool_execute(
    ) {
        let tool = ConsultTool::new(magi_all_fail_with_auth_errors(), false)
            .with_kind(ProviderKind::Ollama);
        let err = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect_err("0 of 3 succeed ⇒ Err(InsufficientAgents)");
        let msg = match err {
            ToolError::ExecutionError(m) => m,
            other => panic!("expected ExecutionError, got {other:?}"),
        };
        assert!(
            msg.contains("keyless") && msg.contains("openai-compat"),
            "the probable-cause hint must reach the user: {msg}"
        );
    }

    /// Same total failure, but under a kind that carries a credential: the hint must
    /// NOT appear — this is the negative case that proves the guard is real (an
    /// unconditional hint would pass the positive test above too).
    #[tokio::test]
    async fn a_total_seat_failure_under_openai_compat_does_not_surface_the_keyless_hint() {
        let tool = ConsultTool::new(magi_all_fail_with_auth_errors(), false)
            .with_kind(ProviderKind::OpenAiCompat);
        let err = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect_err("0 of 3 succeed ⇒ Err(InsufficientAgents)");
        let msg = match err {
            ToolError::ExecutionError(m) => m,
            other => panic!("expected ExecutionError, got {other:?}"),
        };
        assert!(
            !msg.contains("keyless"),
            "openai-compat carries a credential: no hint: {msg}"
        );
    }

    /// `ConsultTool` with `auto_approve = false` (default) MUST require approval.
    #[test]
    fn test_consult_tool_requires_approval_when_auto_approve_false() {
        let tool = dummy_tool(); // auto_approve = false
        assert!(
            tool.requires_approval(),
            "consult with auto_approve=false must still require approval"
        );
    }

    /// `ConsultTool` with `auto_approve = true` must NOT require approval.
    ///
    /// RED: fails until `requires_approval()` is wired to `!self.auto_approve`.
    #[test]
    fn test_consult_tool_does_not_require_approval_when_auto_approve_true() {
        let tool = ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            true,
        );
        assert!(
            !tool.requires_approval(),
            "consult with auto_approve=true must not require approval (auto-approved)"
        );
    }

    /// `ConsultTool` with `auto_approve = false` must return `None` from `approval_notice`.
    ///
    /// RED: fails until `approval_notice()` is wired to `auto_approve`.
    #[test]
    fn test_consult_approval_notice_is_none_when_auto_approve_false() {
        let tool = dummy_tool(); // auto_approve = false
        assert!(
            tool.approval_notice().is_none(),
            "consult with auto_approve=false must return None — user is prompted instead"
        );
    }

    /// `ConsultTool` with `auto_approve = true` must return `Some(notice)` from `approval_notice`.
    ///
    /// RED: fails until `approval_notice()` is wired to `auto_approve`.
    #[test]
    fn test_consult_approval_notice_is_some_when_auto_approve_true() {
        let tool = ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            true,
        );
        let notice = tool.approval_notice();
        assert!(
            notice.is_some(),
            "consult with auto_approve=true must return Some notice for TUI announcement"
        );
        let msg = notice.unwrap();
        assert!(
            msg.contains("MAGI") || msg.contains("consensus"),
            "auto-launch notice must mention MAGI or consensus; got: {msg:?}"
        );
    }

    #[test]
    fn test_consult_tool_contract() {
        let tool = dummy_tool();
        assert_eq!(tool.name(), "consult");
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        assert_eq!(schema["required"][0], "query");
        // `required` names ONLY `query`: `mode` must stay optional so an agent that
        // doesn't pick a lens still gets to consult (REQ-A07/A07b).
        assert_eq!(schema["required"].as_array().unwrap().len(), 1);
        let lower = tool.description().to_lowercase();
        assert!(!lower.is_empty());
        assert!(lower.contains("trade-off"));
        assert!(lower.contains("perspective") || lower.contains("perspectives"));
        assert!(lower.contains("decision") || lower.contains("decisions"));
    }

    /// REQ-A07b: the tool exposes `mode` in its own input schema so an agent that
    /// decides to consult can also pick the lens, from the same three-label
    /// vocabulary `magi_rs::magi::mode::normalize_label` accepts. No behavior change
    /// to `execute` — it still hardcodes `Mode::Analysis`; wiring the declared value
    /// into dispatch is Task 2.3/2.4's job, not this one's.
    #[test]
    fn test_consult_tool_schema_exposes_an_optional_mode_lens() {
        let tool = dummy_tool();
        let schema = tool.input_schema();
        assert_eq!(schema["properties"]["mode"]["type"], "string");
        assert_eq!(
            schema["properties"]["mode"]["enum"],
            json!(["code-review", "design", "analysis"])
        );
    }

    #[tokio::test]
    async fn test_execute_oversized_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let big = "x".repeat(9000);
        assert!(matches!(
            tool.execute(json!({"query": big}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_returns_consensus_report() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("3 agents → success");
        assert!(!out["report"].as_str().expect("report string").is_empty());
        assert_eq!(out["degraded"], json!(false));
    }

    #[tokio::test]
    async fn test_execute_empty_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({ "query": "   " }), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_missing_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    /// A pre-cancelled token makes `execute` return the cancellation error
    /// promptly, aborting the in-flight 3-LLM analysis instead of running it to
    /// completion (REQ-H36). Uses [`BlockingProvider`] so the analysis would
    /// otherwise block for an hour: returning within [`CANCEL_RETURN_BUDGET_MS`]
    /// proves the cancel path pre-empts the work rather than awaiting it.
    #[tokio::test]
    async fn test_execute_returns_cancellation_error_without_running_full_analysis() {
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(BlockingProvider))), false);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let start = std::time::Instant::now();
        let err = tool
            .execute(json!({ "query": "should we migrate X to Y?" }), &cancel)
            .await
            .expect_err("a cancelled consult must return an error, not a report");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, ToolError::ExecutionError(ref m) if m.contains("cancelled")),
            "cancelled consult must surface a typed cancellation error; got: {err:?}"
        );
        assert!(
            elapsed.as_millis() < CANCEL_RETURN_BUDGET_MS,
            "cancelled consult must return promptly (took {elapsed:?}); it awaited the full analysis"
        );
    }

    /// The `AbortOnDrop` backstop aborts its guarded task the instant the guard
    /// is dropped, so a dropped `execute` future cannot orphan the spawned
    /// analysis (the drop path the explicit cancel arm does not cover). A bare
    /// dropped `JoinHandle`/`AbortHandle` would merely detach the task, leaving
    /// it to run to completion — this asserts the join reports cancellation and
    /// the task never reached its completion store.
    #[tokio::test]
    async fn test_abort_on_drop_aborts_task_when_guard_dropped() {
        let ran_to_completion = Arc::new(AtomicBool::new(false));
        let flag = ran_to_completion.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(BlockingProvider::SLEEP_SECS)).await;
            flag.store(true, Ordering::SeqCst);
        });
        {
            let _guard = AbortOnDrop::new(handle.abort_handle());
            // `_guard` drops here without ever calling `abort()` explicitly.
        }
        let joined = handle.await;
        assert!(
            joined.as_ref().err().map(|e| e.is_cancelled()).unwrap_or(false),
            "dropping the guard must abort the task (join must report cancellation); got {joined:?}"
        );
        assert!(
            !ran_to_completion.load(Ordering::SeqCst),
            "aborted task must not have reached its completion store"
        );
    }

    /// MS2 (REQ-A20/REQ-A07d/REQ-A08): the resolved (mode, source) pair injected by
    /// the agent's tool loop wins over the `(Analysis, Default)` fallback — happy path
    /// (injected) and the two edge cases (absent, corrupt) that must degrade to the
    /// default pair.
    #[test]
    fn resolved_mode_and_source_prefers_the_injected_pair_over_the_fallback() {
        assert_eq!(
            resolved_mode_and_source(&json!({
                "query": "x",
                "__resolved_mode": "code-review",
                "__resolved_mode_source": "explicit",
            })),
            (Mode::CodeReview, ModeSource::Explicit),
            "an injected resolution must win over the (Analysis, Default) fallback"
        );
        assert_eq!(
            resolved_mode_and_source(&json!({"query": "x"})),
            (Mode::Analysis, ModeSource::Default),
            "absent injection falls back to the pre-MS2 unconditional default"
        );
        assert_eq!(
            resolved_mode_and_source(&json!({
                "query": "x",
                "__resolved_mode": "not-a-mode",
            })),
            (Mode::Analysis, ModeSource::Default),
            "a corrupt injection is treated the same as an absent one"
        );
    }

    #[tokio::test]
    async fn test_execute_backend_failure_surfaces_error() {
        let p = RoutingMockProvider::new()
            .with_agent_responses(
                AgentName::Melchior,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            )
            .with_agent_responses(
                AgentName::Balthasar,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            )
            .with_agent_responses(
                AgentName::Caspar,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            );
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(p))), false);
        assert!(matches!(
            tool.execute(json!({"query": "x"}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::ExecutionError(_)
        ));
    }

    // -----------------------------------------------------------------------
    // Task 6.1 (REQ-A08b/A09/A10/A11c/A11d/A11g/A11h) — telemetry:
    // `report_to_consult_json`, `RunContext::build`, `guard_all_failed`.
    // -----------------------------------------------------------------------

    /// Builds a real `MagiReport` via `Deserialize` — the only way to construct one
    /// outside magi-core, since the struct is `#[non_exhaustive]` with no public
    /// constructor and `Magi::analyze()` cannot be coaxed into returning `Ok` with an
    /// empty `agents` (its own `min_agents` gate rejects that before a report is ever
    /// built — verified against `orchestrator.rs::dispatch_no_rotation`, magi-core
    /// 3.1.0). `agents`/`consensus`/`banner` are minimal-but-valid filler except where
    /// a specific test cares about them.
    fn report_fixture(
        agents: Value,
        failed_agents: Value,
        extraction_failures: Value,
        input_size: Value,
        degraded: bool,
        report_text: &str,
    ) -> MagiReport {
        serde_json::from_value(json!({
            "agents": agents,
            "consensus": {
                "consensus": "GO (0-0)",
                "consensus_verdict": "reject",
                "confidence": 0.0,
                "score": 0.0,
                "agent_count": 0,
                "votes": {},
                "majority_summary": "",
                "dissent": [],
                "findings": [],
                "conditions": [],
                "recommendations": {},
            },
            "banner": "",
            "report": report_text,
            "degraded": degraded,
            "failed_agents": failed_agents,
            "extraction_failures": extraction_failures,
            "input_size": input_size,
        }))
        .expect("fixture matches MagiReport's Deserialize shape")
    }

    /// One minimal, validly-shaped `AgentOutput` JSON literal (B4: named once, not
    /// repeated per test).
    fn agent_output_json(seat: &str) -> Value {
        json!({
            "agent": seat, "verdict": "approve", "confidence": 0.9,
            "summary": "s", "reasoning": "r", "findings": [], "recommendation": "rec",
        })
    }

    /// A report where nothing went wrong: no extraction failures, no execution
    /// failures, input under threshold. `agents` is irrelevant to
    /// `report_to_consult_json` (it never reads that field), so it stays empty here —
    /// tests that DO care about `agents` (`guard_all_failed`) build their own fixture.
    fn clean_report() -> MagiReport {
        report_fixture(
            json!([]),
            json!({}),
            json!({}),
            json!({ "estimated_tokens": 10, "warn_threshold": 150_000, "exceeded": false }),
            false,
            "a clean consensus report",
        )
    }

    /// A report with an extraction failure on Caspar AND an input-size overage —
    /// "failures and excess". SC-A09c/A10b/A10c want this to yield the SAME JSON
    /// shape as `clean_report` despite carrying more data; SC-A08 wants a named,
    /// typed failure to read off it.
    fn report_with_failures_and_excess() -> MagiReport {
        report_fixture(
            json!([]),
            json!({}),
            json!({
                "caspar": [
                    { "model": "some-model", "attempt": 1, "cause": "missing-markers" },
                ],
            }),
            json!({ "estimated_tokens": 999_999, "warn_threshold": 100, "exceeded": true }),
            true,
            "a degraded consensus report",
        )
    }

    /// A report where Caspar failed to EXECUTE (network/credential/timeout) — as
    /// opposed to the extraction failure above. SC-A11f's `failed_agents` case.
    fn report_with_one_failed_mage() -> MagiReport {
        report_fixture(
            json!([]),
            json!({ "caspar": "timeout: agent timed out after 90s" }),
            json!({}),
            Value::Null,
            true,
            "a degraded consensus report",
        )
    }

    /// A report where ALL THREE seats end up without a verdict — some by execution
    /// failure, some by extraction failure. `guard_all_failed`'s target case.
    fn report_with_all_three_failed() -> MagiReport {
        report_fixture(
            json!([]),
            json!({
                "melchior": "network: connection refused",
                "balthasar": "timeout: agent timed out after 90s",
            }),
            json!({
                "caspar": [
                    { "model": "some-model", "attempt": 1, "cause": "missing-markers" },
                    { "model": "some-model", "attempt": 2, "cause": "unterminated" },
                ],
            }),
            Value::Null,
            true,
            "",
        )
    }

    /// A report where two of three seats succeeded (`agents` non-empty) —
    /// `guard_all_failed`'s happy path.
    fn report_with_two_successes() -> MagiReport {
        report_fixture(
            json!([
                agent_output_json("melchior"),
                agent_output_json("balthasar")
            ]),
            json!({}),
            json!({}),
            Value::Null,
            true,
            "a degraded but valid consensus report",
        )
    }

    /// Local double: the report's own text, unmodified — no truncation logic exists
    /// yet, so every caller in this module today passes `TruncationLevel::None`.
    fn untruncated(r: &MagiReport) -> Truncated {
        Truncated {
            text: r.report.clone(),
            level: TruncationLevel::None,
        }
    }

    /// Local double: a fixed mode resolution. `classification_attempted` mirrors
    /// `source == Inferred` — the only one of the fixed sources these tests use that
    /// actually implies a classification call was made.
    fn res_of(mode: Mode, source: ModeSource) -> ModeResolution {
        ModeResolution {
            mode,
            source,
            classification_attempted: source == ModeSource::Inferred,
        }
    }

    /// Local double: a `RunContext` with nothing to report.
    fn ctx_plain() -> RunContext {
        RunContext {
            endpoint_divergence: false,
            timeout_below_formula: false,
        }
    }

    /// Local double: a `RunContext` declaring endpoint divergence.
    fn ctx_with_divergent_endpoint() -> RunContext {
        RunContext {
            endpoint_divergence: true,
            timeout_below_formula: false,
        }
    }

    /// SC-A09b: the JSON grows with the new telemetry fields. `schema_version`
    /// (`src/headless/output.rs`, an unrelated module) does not move for it
    /// (REQ-A08b) — verified structurally by this task simply never touching that
    /// file; its `SCHEMA_VERSION` constant is private with no public accessor this
    /// module could assert against.
    #[test]
    fn the_json_grows_while_schema_version_stays() {
        let r = clean_report();
        let v = report_to_consult_json(
            &r,
            &untruncated(&r),
            &res_of(Mode::CodeReview, ModeSource::Inferred),
            &ctx_plain(),
        );
        for key in [
            "report",
            "degraded",
            "mode",
            "mode_source",
            "extraction_failures",
            "input_size",
            "report_truncated",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["mode_source"], "inferred");
    }

    /// SC-A09c / SC-A10b / SC-A10c: the shape does NOT vary with the outcome.
    #[test]
    fn the_shape_does_not_vary_with_the_outcome() {
        let clean_r = clean_report();
        let noisy_r = report_with_failures_and_excess();
        let clean = report_to_consult_json(
            &clean_r,
            &untruncated(&clean_r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        let noisy = report_to_consult_json(
            &noisy_r,
            &untruncated(&noisy_r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        let keys_of = |v: &Value| -> Vec<String> {
            let mut ks: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            ks.sort();
            ks
        };
        assert_eq!(keys_of(&clean), keys_of(&noisy));
        assert_eq!(keys_of(&clean["input_size"]), keys_of(&noisy["input_size"]));

        assert!(
            clean["extraction_failures"].as_object().unwrap().is_empty(),
            "empty is a POSITIVE CERTIFICATE: omitting it would destroy the signal"
        );
        assert!(!clean["input_size"]["exceeded"].as_bool().unwrap());
        assert!(
            !clean["input_size"]["warn_threshold"].is_null(),
            "never null to mean 'I don't know': report what was APPLIED"
        );
    }

    /// SC-A08: a non-adhering model is NAMED, with its attempt and a typed cause.
    #[test]
    fn a_non_adhering_model_is_named_with_attempt_and_typed_cause() {
        let r = report_with_failures_and_excess();
        let v = report_to_consult_json(
            &r,
            &untruncated(&r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        let f = &v["extraction_failures"]["caspar"][0];
        assert!(
            f["model"].is_string(),
            "with rotation the question is WHICH MODEL, not which seat"
        );
        assert!(f["attempt"].is_number());
        assert!(f["cause"].is_string());
    }

    /// SC-A11f: endpoint divergence and a failed seat both travel in THIS run's own
    /// output, not a process-wide notice.
    #[test]
    fn per_run_signals_travel_in_the_runs_own_output() {
        let diverged_r = clean_report();
        let diverged = report_to_consult_json(
            &diverged_r,
            &untruncated(&diverged_r),
            &res_of(Mode::Analysis, ModeSource::Inferred),
            &ctx_with_divergent_endpoint(),
        );
        assert_eq!(
            diverged["endpoint_divergence"], true,
            "a single once-per-process notice does not serve someone auditing a \
             thousand runs afterwards"
        );

        let failed_r = report_with_one_failed_mage();
        let failed = report_to_consult_json(
            &failed_r,
            &untruncated(&failed_r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        assert!(
            failed["failed_agents"]
                .as_object()
                .unwrap()
                .contains_key("caspar"),
            "typed with its cause, not collapsed into `degraded`"
        );
    }

    /// SC-A11h: `report_truncated` names the LEVEL applied, never a boolean.
    #[test]
    fn report_truncated_names_the_level_applied() {
        for (level, expected) in [
            (TruncationLevel::None, "none"),
            (TruncationLevel::Structural, "structural"),
            (TruncationLevel::Anchored, "anchored"),
            (TruncationLevel::Bytes, "bytes"),
        ] {
            let r = clean_report();
            let v = report_to_consult_json(
                &r,
                &Truncated {
                    text: r.report.clone(),
                    level,
                },
                &res_of(Mode::Analysis, ModeSource::Default),
                &ctx_plain(),
            );
            assert_eq!(v["report_truncated"], expected);
        }
    }

    /// Local double: a resolution with only the attempt flag varied — the one field
    /// these `RunContext::build` tests exercise.
    fn resolution(attempted: bool) -> ModeResolution {
        ModeResolution {
            mode: Mode::Analysis,
            source: ModeSource::Default,
            classification_attempted: attempted,
        }
    }

    /// SC-A11f: `endpoint_divergence` is populated where the decision is actually
    /// made — `RunContext::build`, fed by the REAL `crate::config::MagiConfig` and
    /// the REAL `magi_rs::magi::resolve_run_timeout`.
    #[test]
    fn endpoint_divergence_is_populated_where_the_decision_is_made() {
        let diverged = MagiConfig::from_toml_str(
            "base_url = \"http://a/v1\"\n[magi]\nbase_url = \"http://b/v1\"\n",
        )
        .expect("valid toml");
        let dec = resolve_run_timeout(None, AGENT_TIMEOUT_SECS);

        let ctx = RunContext::build(&diverged, &resolution(true), &dec);
        assert!(ctx.endpoint_divergence);

        let ctx = RunContext::build(&diverged, &resolution(false), &dec);
        assert!(
            !ctx.endpoint_divergence,
            "without classification the content never went out to the principal"
        );

        let same =
            MagiConfig::from_toml_str("base_url = \"http://a/v1\"\n[magi]\n").expect("valid toml");
        let ctx = RunContext::build(&same, &resolution(true), &dec);
        assert!(
            !ctx.endpoint_divergence,
            "same endpoint: there is no divergence to declare"
        );
    }

    /// SC-A11f (the case the naive predicate would lose): classification ATTEMPTED
    /// and FAILED.
    ///
    /// This is what distinguishes "attempted" from `ModeSource::Inferred`. A
    /// classification that expires leaves `ModeSource::Default`, but the content has
    /// ALREADY gone out to the principal provider — and that is the run where
    /// declaring the data-flow matters most, not least.
    #[test]
    fn a_failed_classification_still_declares_the_endpoint_divergence() {
        let diverged = MagiConfig::from_toml_str(
            "base_url = \"http://a/v1\"\n[magi]\nbase_url = \"http://b/v1\"\n",
        )
        .expect("valid toml");
        let dec = resolve_run_timeout(None, AGENT_TIMEOUT_SECS);

        let ctx = RunContext::build(&diverged, &resolution(true), &dec);
        assert!(
            ctx.endpoint_divergence,
            "hanging the signal off ModeSource::Inferred would give a false negative here"
        );
    }

    /// SC-A11g: zero verdicts is a TYPED ERROR, never a degraded report.
    #[test]
    fn guard_all_failed_names_every_known_cause_when_nothing_produced_a_verdict() {
        let err = guard_all_failed(&report_with_all_three_failed())
            .expect_err("no agent produced a verdict");
        assert!(err.causes.contains("Melchior"), "{}", err.causes);
        assert!(err.causes.contains("Balthasar"), "{}", err.causes);
        assert!(err.causes.contains("Caspar"), "{}", err.causes);
    }

    /// Happy path (B13): at least one verdict ⇒ the guard passes.
    #[test]
    fn guard_all_failed_passes_when_at_least_one_agent_produced_a_verdict() {
        assert!(guard_all_failed(&report_with_two_successes()).is_ok());
    }
}
