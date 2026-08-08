// Author: Julian Bolivar Version: 1.0.0 Date: 2026-06-07

//! Tool that wraps `magi_core::Magi` to run 3-perspective consensus queries. The agent routes
//! here only for genuine multi-perspective decisions; trivial or factual lookups are answered
//! directly.

use crate::config::MagiConfig;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use magi_core::error::MagiError;
use magi_core::orchestrator::{Magi, MagiConfig as CoreMagiConfig};
use magi_core::reporting::{ExtractionFailure, InputSize, MagiReport};
use magi_core::schema::{AgentName, Mode};
use magi_rs::magi::kind::ProviderKind;
use magi_rs::magi::mode::{read_resolved_mode, ModeResolution, ModeSource};
use magi_rs::magi::report_anchors::{CONTRACTUAL_ANCHORS, SECTION_ANCHORS};
use magi_rs::magi::{
    bytes_to_tokens_est, mark_overhead, TimeoutDecision, MAX_QUERY_BYTES, TOOL_RESULT_CAP_BYTES,
    TRUNCATION_MARK,
};
use magi_rs::redact::{redact_foreign_error, redact_foreign_text};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

/// Why the input to a consult is unacceptable (REQ-A11b).
///
/// Two variants, not one: an empty query and an oversized one are distinct failures (pure waste
/// vs. real cost) and the message must distinguish them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ConsultInputError {
    /// Empty query or one with only whitespace. Calling three models with this is pure waste,
    /// with no chance of a useful verdict.
    #[error("query must not be empty")]
    Empty,
    /// Above the configured cap. **Rejects, NEVER truncates** (REQ-A11b): a silently truncated
    /// payload produces a verdict indistinguishable from a legitimate one.
    #[error("query too large ({size} bytes; max {cap})")]
    TooLarge {
        /// Received size, in bytes.
        size: usize,
        /// The effective cap, so the message says by how much it exceeded.
        cap: usize,
    },
}

/// Checks the input size of a consult (REQ-A11b). **The single cap for the three entry routes**
/// — [`ConsultTool::execute`], the direct headless route
/// (`crate::headless_runner::analyze_direct`), and the explicit TUI `/consult` (`crate::tui`) —
/// call this SAME function with the effective cap each one resolved from
/// `MagiConfig::effective_max_query_bytes`, instead of three parallel copies that could diverge
/// (SC-A11c, B3).
///
/// **Rejects, never truncates.** The numeric criterion is COST, not capacity:
/// magi-core already skips models where the prompt does not fit, so magi-rs does not have to
/// protect the model. What it must bound is the cost: the payload goes to the three mages, so
/// it is paid for three times.
///
/// # Parameters
/// * `query` - the content to validate before spending any model call.
/// * `cap` - the effective limit in bytes (`MagiConfig::effective_max_query_bytes`).
///
/// # Errors
/// [`ConsultInputError::Empty`] if `query` is empty or only whitespace;
/// [`ConsultInputError::TooLarge`] with the size and the limit if it exceeds `cap`.
pub(crate) fn check_query_size(query: &str, cap: usize) -> Result<(), ConsultInputError> {
    if query.trim().is_empty() {
        return Err(ConsultInputError::Empty);
    }
    if query.len() > cap {
        return Err(ConsultInputError::TooLarge {
            size: query.len(),
            cap,
        });
    }
    Ok(())
}

/// Did the anchor of the first finding survive the cut?
///
/// This is the check that turns [`TruncationLevel::Structural`] from intention into fact: it is
/// ASSERTED over the RESULTING TEXT (`kept`), not over whether the attempt to locate the
/// anchors succeeded — the same discipline SC-A11e demands of its own assertions.
/// `SECTION_ANCHORS` may legitimately not appear in the original report when there were no
/// findings (`report_anchors::SectionAnchors:: findings_start`); in that case this function
/// also returns `false`, and the caller steps down to [`TruncationLevel::Anchored`] instead of
/// asserting a guarantee the text does not meet — "there were no findings" and "could not be
/// located" are indistinguishable to this check, but the result (stepping down) is correct in
/// both cases.
#[must_use]
fn kept_has_first_finding(kept: &str) -> bool {
    SECTION_ANCHORS.is_some_and(|a| kept.contains(a.findings_start))
}

/// Trims at most `cap` bytes from `s`, cutting at a CHARACTER boundary.
///
/// `char_indices` gives the INITIAL index of each character, so comparing against the start
/// would let through a multi-byte character that STARTS before the cap and ENDS after it — the
/// result would exceed the limit by up to 3 bytes (the maximum width of a UTF-8 character minus
/// one). The END of the character (`i + c.len_utf8()`) is compared, never its start, and the
/// result is built with `.collect()` over complete `char`s — never with a byte slice — so there
/// is no way it can produce an invalid boundary.
///
/// # Complexity
/// O(n) in the number of characters in `s` up to `cap` — one traversal, no backtracking.
#[must_use]
fn head_chars(s: &str, cap: usize) -> String {
    s.char_indices()
        .take_while(|(i, c)| i + c.len_utf8() <= cap)
        .map(|(_, c)| c)
        .collect()
}

/// Appends the truncation mark to the kept text, so the loss is visible on the text surface
/// just as `report_truncated` makes it visible in the JSON (B9 — a silent truncation is
/// indistinguishable from a complete report).
#[must_use]
fn mark(kept: String) -> String {
    format!("{kept}\n\n{TRUNCATION_MARK}")
}

/// Keeps verdict + findings, if the sections can be located.
///
/// Uses the anchors that the Task 0.6 spike left in [`SECTION_ANCHORS`] (`report_anchors.rs`,
/// sole owner). Returns `None` when the list is empty — the spike concluded there is no stable
/// structure — or when the verdict anchor does not appear in `report`; in both cases the caller
/// steps down a level.
///
/// The right-end cut goes at the THIRD anchor (`findings_end`, `"## Recommended Actions"`), not
/// the second (`findings_start`): the region must SPAN the findings section, not end right
/// where it begins — cutting there would return the verdict and nothing more, while this
/// function promises "verdict AND first finding". If `findings_end` does not appear (a report
/// without that section, a case not observed in production but not dismissible), keep up to the
/// end of the report instead of failing.
///
/// The mark's budget ([`mark_overhead`]) is deducted HERE, because that is where it is known:
/// the caller concatenates [`TRUNCATION_MARK`] after this function, so trimming to exactly
/// `cap` and then concatenating would exceed the cap by `mark_overhead()` bytes.
#[must_use]
fn keep_verdict_and_first_finding(report: &str, cap: usize) -> Option<String> {
    let anchors = SECTION_ANCHORS?; // sin anclas: este nivel no aplica
    let start = report.find(anchors.verdict_start)?;
    let end = report
        .get(start..)
        .and_then(|s| s.find(anchors.findings_end))
        .map_or(report.len(), |i| start + i);
    let slice = report.get(start..end)?;
    if slice.is_empty() {
        return None;
    }
    let budget = cap.checked_sub(mark_overhead())?; // cap ridículo ⇒ este nivel no aplica
    Some(head_chars(slice, budget))
}

/// Keeps from the first CONTRACTUAL anchor of magi-core.
///
/// A contractual anchor (`report_anchors::CONTRACTUAL_ANCHORS`) is stable BY DEFINITION — magi-
/// core always emits it — unlike the section format, which is stable by convention. Hence this
/// level comes after the structural one but before byte counting: it is the fallback for a
/// report where `findings_start` does not appear because there were no findings, or where the
/// findings region did not survive the cut.
#[must_use]
fn keep_contractual_anchor(report: &str, cap: usize) -> Option<String> {
    let start = CONTRACTUAL_ANCHORS.iter().find_map(|a| report.find(a))?;
    let budget = cap.checked_sub(mark_overhead())?; // ver `keep_verdict_and_first_finding`
    Some(head_chars(report.get(start..)?, budget))
}

/// Last resort: the first `cap` bytes of the report, without looking for any structure. Returns
/// `None` only when `cap` is too small to leave room for the mark —
/// `MagiConfig::effective_tool_result_cap` already rejects that configuration on load
/// (`ConfigError::OutputCapTooSmall`), so this path is defense in depth for a hand-built `cap`
/// (tests, or some future caller), not a route reachable from `magi.toml`.
#[must_use]
fn keep_bytes(report: &str, cap: usize) -> Option<String> {
    let budget = cap.checked_sub(mark_overhead())?;
    Some(head_chars(report, budget))
}

/// Truncates the report to the cap, choosing the highest level the shape allows (REQ-A11b, Task
/// 6.2 — completes what [`TruncationLevel`] left pending).
///
/// Three levels, from most to least guarantee — each is tried ONLY if the previous one is not
/// viable, and the returned level (`Truncated::level`, exposed in the JSON via
/// `truncation_label`) says WHICH one was used, so the consumer knows what guarantee it is
/// looking at instead of deducing it (SC-A11e/SC-A11h):
/// - [`TruncationLevel::Structural`] — verdict + at least the first finding.
/// - [`TruncationLevel::Anchored`] — only the verdict; the finding is best-effort.
/// - [`TruncationLevel::Bytes`] — only the first N bytes; nothing else guaranteed.
///
/// The `Structural` level is ASSERTED over the RESULT, not the attempt: cutting to `cap` may
/// cut just before the first finding, and returning `Structural` there would be promising a
/// guarantee the text does not meet — the consumer would trust a finding that is not there. If
/// the finding did not survive, step down a level instead of lying.
///
/// # Parameters
/// * `report` - the ALREADY-ANNOTATED text (REQ-A12c, [`annotate_report_text`]) that is wants to bound. Callers must annotate BEFORE truncating: this function never sees the keyless-auth hint if the order is reversed.
/// * `cap` - the effective byte limit (`MagiConfig::effective_tool_result_cap`).
///
/// # Returns
/// The text that survived (with [`TRUNCATION_MARK`] already applied if there was a truncation)
/// and the guarantee level reached. `report.len() <= cap` returns the report intact with
/// [`TruncationLevel::None`] — never adds the mark when nothing was truncated.
#[must_use]
pub(crate) fn truncate_report(report: &str, cap: usize) -> Truncated {
    if report.len() <= cap {
        return Truncated {
            text: report.to_string(),
            level: TruncationLevel::None,
        };
    }
    if let Some(kept) = keep_verdict_and_first_finding(report, cap) {
        if kept_has_first_finding(&kept) {
            return Truncated {
                text: mark(kept),
                level: TruncationLevel::Structural,
            };
        }
    }
    if let Some(kept) = keep_contractual_anchor(report, cap) {
        return Truncated {
            text: mark(kept),
            level: TruncationLevel::Anchored,
        };
    }
    match keep_bytes(report, cap) {
        Some(kept) => Truncated {
            text: mark(kept),
            level: TruncationLevel::Bytes,
        },
        // `cap` too small even for the mark — unreachable via `magi.toml`
        // (`ConfigError::OutputCapTooSmall` rejects it on load), but a hand-built `cap` CAN
        // reach here. Returning the report intact, honestly labeled `None`, is preferable to a
        // broken fragment that lies about respecting the cap.
        None => Truncated {
            text: report.to_string(),
            level: TruncationLevel::None,
        },
    }
}

/// Bytes the `"\n\n"` separator between a preserved prefix (e.g. the TUI's `[DEGRADED: ...]`
/// banner) and the report text that follows it adds to the combined length.
///
/// Same shape as [`magi_rs::magi::TRUNCATION_SEPARATOR_LEN`] — kept as its OWN constant rather
/// than reused, because it reserves budget for a DIFFERENT join (prefix-to-report, not mark-to-
/// kept-text): the two separators are free to diverge, and folding them into one constant would
/// let a future change to either silently mis-budget the other (B4).
const PREFIX_SEPARATOR_LEN: usize = 2;

/// Truncates `report` to `cap`, reserving room so `prefix` survives the cut UNCONDITIONALLY
/// (fix: the TUI's `[DEGRADED: ...]` banner was silently dropped by [`truncate_report`] alone —
/// its three levels all cut the kept region starting from the verdict anchor onward, which sits
/// AFTER a banner that was being joined ahead of the report BEFORE truncation; a sufficiently
/// long degraded report therefore rendered with NO caveat at all, indistinguishable from a
/// full-strength consensus).
///
/// `prefix` is reserved from `cap` FIRST — `prefix.len() + PREFIX_SEPARATOR_LEN` bytes — and
/// [`truncate_report`] is applied to `report` ALONE under whatever budget remains. The highest-
/// value text (the caveat that qualifies everything below it) is never the thing sacrificed to
/// make room for report content.
///
/// An empty `prefix` is treated as "no prefix at all": the call degenerates to plain
/// [`truncate_report`] rather than joining an empty string with a spurious blank line, so this
/// function is a strict superset of [`truncate_report`]'s behavior, not a parallel
/// implementation of it.
///
/// # Arithmetic
///
/// Whenever `cap` leaves a remaining budget of at least [`mark_overhead`] bytes after reserving
/// the prefix, the returned text is
/// **exactly** `prefix + "\n\n" + <report truncated to that remaining
/// budget>`: each of [`truncate_report`]'s three levels bounds its own output to the cap it was
/// given (`cap.checked_sub(mark_overhead())` per level), so the combined result never exceeds
/// `cap`.
///
/// Below that floor — `cap` too tiny to hold the prefix and still leave room for
/// `truncate_report`'s own mark — this degrades the SAME way [`truncate_report`] itself
/// degrades when its `cap` is below [`magi_rs::magi::min_viable_output_cap`]: the full,
/// untruncated `prefix + "\n\n" + report`, labeled [`TruncationLevel::None`] truthfully
/// (nothing WAS truncated) rather than a fragment that claims to respect a cap it does not. For
/// the project's own banner this floor sits at 113 bytes (`75` reserved for the banner + `38`
/// for `mark_overhead()`), well below [`TOOL_RESULT_CAP_BYTES`]'s default and unreachable at
/// that default; it is documented here because this function accepts an arbitrary `prefix`, not
/// only the banner.
///
/// # Parameters
/// * `prefix` - text that must survive the cut whole; meant for a short, one-line caveat, never the bulk of the reply.
/// * `report` - the (already-annotated) report text to bound.
/// * `cap` - the byte budget the combined `prefix + report` output must respect.
#[must_use]
pub(crate) fn truncate_report_with_preserved_prefix(
    prefix: &str,
    report: &str,
    cap: usize,
) -> Truncated {
    if prefix.is_empty() {
        return truncate_report(report, cap);
    }
    let prefix_reserved = prefix.len() + PREFIX_SEPARATOR_LEN;
    match cap.checked_sub(prefix_reserved) {
        Some(remaining) => {
            let truncated = truncate_report(report, remaining);
            Truncated {
                text: format!("{prefix}\n\n{}", truncated.text),
                level: truncated.level,
            }
        }
        None => Truncated {
            text: format!("{prefix}\n\n{report}"),
            level: TruncationLevel::None,
        },
    }
}

/// The STABLE prefix with which magi-core renders a seat's cause when the failure was
/// authentication — the `Display` of `ProviderError::Auth` (magi-core 3.1.0, `error.rs`:
/// `#[error("auth error: {message}")]`) (REQ-A12c, SC-A12f).
///
/// **Why this, and not `ExternalErrorKind::Auth`.** `ExternalErrorKind` (`error.rs`,
/// along with `ProviderError::external`) is for THIRD-PARTY providers implemented OUTSIDE magi-
/// core — its own rustdoc says so: *"Failure reported by an LlmProvider implemented OUTSIDE
/// this crate"*, and its only constructor reachable from another crate is
/// `ProviderError::external(...)`. REQ-A01 forbids magi-rs from implementing `LlmProvider`, so
/// that route is STRUCTURALLY unreachable for the native trio. What the trio DOES produce,
/// verified by reading the two native implementations and their own tests:
/// `OpenAiCompatibleProvider::map_status_to_error` (magi-core
/// `src/providers/openai_compat.rs:278-289`) and `ClaudeProvider::map_status_to_error`
/// (`src/providers/claude.rs:239-254`) BOTH map 401 and 403 to `ProviderError::Auth{message}` —
/// not to `ProviderError::Http`. Pinned by `map_status_to_error_maps_401_and_403_to_auth` in
/// `openai_compat.rs` and by `map_status_to_error_maps_401_to_auth`/`_403_to_auth` in
/// `claude.rs`.
///
/// **Where this chain lives, exactly — and where it does NOT reach.** `MagiError::Provider`
/// has `#[error(transparent)]`, so its `Display` delegates wholly to `ProviderError`'s.
/// `dispatch_one_agent` (magi-core `orchestrator.rs:1298-1304`) builds the cause of the FIRST
/// attempt with `MagiError::Provider(provider_err).to_string()` — without any additional prefix
/// in the common case (no retry, which only fires on `Validation`/`Deserialization`, never on a
/// provider error) — and that string is exactly what ends up in `MagiReport::failed_agents`.
/// `.contains(...)`, not an exact prefix, because the retry path DOES prepend `"retry-failed:
/// "`.
///
/// **The real scope of this detection is NARROWER than it looks — see
/// [`keyless_auth_explanation`] and this task's report.**
const PROVIDER_AUTH_ERROR_MARKER: &str = "auth error: ";

/// The REUSABLE core of the keyless explanation (REQ-A12c, B3) — ONE single wording, consumed
/// by the TWO reachable paths in this file: [`keyless_auth_explanation`] (POSITIVE evidence:
/// the [`PROVIDER_AUTH_ERROR_MARKER`] present in the cause of ONE seat, via
/// `MagiReport::failed_agents`) and [`explain_magi_error`] (NO status evidence — only "0 of 3
/// seats ran under a keyless kind", via `MagiError::InsufficientAgents`). Fix round 3:
/// previously there was a single wording with an opening already diagnosed ("the endpoint
/// rejected..."), correct for the first path but an unsupported claim in the second. Instead of
/// writing a second wording (which B3 forbids — two texts that could diverge over time), this
/// constant was cut to the core that is TRUE in both cases, and each caller prepends its OWN
/// framing phrase according to the evidence it actually has.
///
/// **Deliberately in CONDITIONAL mode** ("if your endpoint REQUIRES it...", never "your
/// endpoint REQUIRED it"): by itself, it already serves the path WITHOUT evidence
/// ([`explain_magi_error`]) without needing editing — REQ-A12c asks to name the configuration
/// as a **probable cause**, not a demonstrated one ("no impossible validation is requested...
/// the inevitable failure is required to arrive explained"), and that record is the one that
/// has to survive on the two surfaces.
const KEYLESS_AUTH_EXPLANATION: &str = "`[magi].kind = \"ollama\"` is keyless and never sends \
     a credential. If your endpoint requires one, use `kind = \"openai-compat\"` and declare \
     the key via env or vault.";

/// Translates the cause of ONE seat —as it appears in `MagiReport::failed_agents`— into an
/// actionable configuration error, when that cause is a rejected authentication UNDER a keyless
/// kind (REQ-A12c, SC-A12f).
///
/// Returns `None` for any other combination: under a kind that DOES carry credentials (`openai-
/// compat`/`anthropic`), a 401/403 can be a genuinely bad credential, and reinterpreting it as
/// a configuration error would send the user to check the wrong file — an ACTIVELY incorrect
/// diagnosis, not an extra guard.
///
/// **Does not interpolate `cause` into the result.** The message is fixed text: there is
/// nothing
/// derived from the cause —from a THIRD PARTY, by definition untrusted (B11)— in the output, so
/// there is no leakage surface to redact. Covered by
/// `keyless_auth_explanation_never_echoes_the_raw_cause`.
///
/// # Real scope — READ BEFORE ASSUMING THIS COVERS EVERY KEYLESS 401
///
/// `failed_agents` only exists when `Magi::analyze()` returns `Ok(MagiReport)`, and that
/// requires `successful.len() >= min_agents` — 2, the `ConsensusConfig` default that REQ-A15
/// forbids exposing (`consensus.rs`: `impl Default for ConsensusConfig`). Verified against
/// `orchestrator.rs::dispatch_no_rotation` (magi-core 3.1.0, lines 1058-1065): when FEWER than
/// `min_agents` seats succeed, the function returns
/// `Err(MagiError::InsufficientAgents{succeeded, required})` — **and the `failed` map that did
/// hold each cause, including an authentication one, is discarded right there**; a `MagiReport`
/// is never built, so this function is never invoked for those seats.
///
/// Because the MS2 trio shares ONE `base_url`/`kind` across the three seats (no rotation,
/// R-A06), a badly chosen `kind` —the scenario REQ-A12c describes— rejects ALL THREE seats
/// equally: 0 of 3 succeeded, `Err(InsufficientAgents{succeeded: 0, required: 2})`, and this
/// function never sees the cause. The case where it DOES see it —exactly 2 of 3 succeeded,
/// degraded but `Ok`— is real and this function genuinely covers it, but it is the LESS likely
/// case of "badly chosen kind", not the more likely one. Documented with full evidence in this
/// task's report (round 2).
#[must_use]
fn keyless_auth_explanation(cause: &str, kind: ProviderKind) -> Option<&'static str> {
    (kind == ProviderKind::Ollama && cause.contains(PROVIDER_AUTH_ERROR_MARKER))
        .then_some(KEYLESS_AUTH_EXPLANATION)
}

/// Appends, at the end of the text already rendered by magi-core, one note for each seat in
/// `failed_agents` whose cause is recognized as rejected authentication under a keyless kind
/// (REQ-A12c) — instead of leaving that cause completely invisible, which is what happens
/// today: `report_format.rs` (magi-core) does not include it in
/// `report.report`/`report.banner`, and nothing in magi-rs read it before this task.
///
/// **Deliberately narrow scope.** It only adds a line when
/// [`keyless_auth_explanation`] recognizes the pattern — it does not dump the other
/// `failed_agents` causes untranslated. General surfacing of `failed_agents` (REQ-A09/A11d) is
/// the responsibility of a later task (Task 6.1, telemetry); this function does not get ahead
/// of that shape so as not to compete with its design.
///
/// The seat name (`AgentName`, via `{agent:?}`) is safe — it is not third-party text.
///
/// `pub(crate)` (fix round 4, finding 2): besides [`report_to_consult_json`]
/// (`ConsultTool::execute`, `analyze_direct`), the explicit TUI `/consult`
/// (`src/tui/mod.rs::tui_consult_success_body`) calls this directly — it is the SAME annotation
/// on the three surfaces (B3), never a fourth independent wording.
pub(crate) fn annotate_report_text(report: &MagiReport, kind: ProviderKind) -> String {
    let mut text = report.report.clone();
    for (agent, cause) in &report.failed_agents {
        if let Some(explanation) = keyless_auth_explanation(cause, kind) {
            // The opening "rejected by authentication" is an ASSERTION, and here it is backed:
            // `keyless_auth_explanation` only returned `Some` because `cause` (the REAL cause
            // of this seat) contained `PROVIDER_AUTH_ERROR_MARKER` — positive evidence.
            // `explain_magi_error` (below) does not have that evidence and therefore does NOT
            // use this same opening.
            text.push_str(&format!(
                "\n\n**{agent:?}** rejected due to authentication: {explanation}"
            ));
        }
    }
    text
}

/// How much of a report's markdown survived an output-size recort (REQ-A11b).
///
/// **Not a boolean.** A boolean would only answer *"was it truncated?"*, which is the
/// less useful question: a consumer needs to know **what guarantee it is looking at**. With
/// [`Self::Structural`] it can trust the verdict and at least one finding survived; with
/// [`Self::Bytes`] it knows only that the first N bytes did, and anything else may be missing.
///
/// [`truncate_report`] chooses among the four variants, from most to least guarantee — see its
/// own rustdoc for how each level is selected and what it promises. [`Self::None`] is what any
/// caller still gets when the report already fits under the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TruncationLevel {
    /// The report was not truncated.
    None,
    /// The verdict and at least one finding were located and kept ([`truncate_report`]'s first,
    /// highest-guarantee level).
    Structural,
    /// A magi-core contractual anchor was used instead of locating sections.
    Anchored,
    /// Only the first N bytes survived; anything past that may be missing.
    Bytes,
}

/// The report text paired with the [`TruncationLevel`] that produced it.
///
/// A struct, not a `(String, TruncationLevel)` tuple, and every field read by name: the two
/// values have to travel **together**, because a caller free to pass the level separately from
/// the text could put `Bytes` next to text that was never actually cut — and then
/// `report_truncated` in the JSON would be asserting a guarantee about text it does not
/// describe.
#[derive(Debug, Clone)]
pub(crate) struct Truncated {
    /// The (possibly annotated, possibly recorted) text that reaches `"report"` in the JSON —
    /// never the raw [`MagiReport::report`] on its own.
    pub(crate) text: String,
    /// The guarantee `text` carries.
    pub(crate) level: TruncationLevel,
}

/// Per-run signals that belong in THIS run's own JSON output, never a startup notice (REQ-A11d)
/// — a batch consumer parsing yesterday's output never sees today's stderr.
///
/// `schema_version` does not move for this telemetry (REQ-A08b): magi-rs is pre-1.0 and the
/// crate version is already the compatibility signal. That is exactly why the SHAPE has to stay
/// stable instead — a field that appears only on some runs is precisely the instability a
/// consumer with no in-band version signal cannot absorb.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunContext {
    /// `true` when the trio ran on an endpoint different from the principal provider
    /// **and** mode classification was attempted — i.e. the content went out to the
    /// principal first.
    pub(crate) endpoint_divergence: bool,
    /// `true` when an explicit `--timeout` landed below the formula's minimum (SC-A04d).
    pub(crate) timeout_below_formula: bool,
    /// Honest replacement for the `0` [`input_size_json`] used to fabricate when
    /// `MagiReport::input_size` is `None` (Loop 2 gate, S4 finding 3).
    ///
    /// `report.input_size` is `None` only for a `MagiReport` magi-core itself did not measure
    /// — verified against `orchestrator.rs::analyze` (magi-core 3.1.0): `measure_input` runs
    /// UNCONDITIONALLY and the real report always carries `Some`, so this branch is
    /// unreachable from any live `Magi::analyze()` call today. It stays reachable through the
    /// type (`Option<InputSize>`, `#[non_exhaustive]`), and this project's own fixtures already
    /// exercise it (`report_with_one_failed_mage`), so it must not lie when it does fire.
    ///
    /// `Some(bytes_to_tokens_est(query.len()))` at the ONE call site that has the query text at
    /// hand (`ConsultTool::execute`'s literal) — an honest, already-used-elsewhere estimator
    /// (SC-A24i), never a fabricated `0` that a consumer could misread as "empty payload".
    /// `RunContext::build` (the headless `analyze_direct` call site) sets this to `None`: its
    /// three existing parameters (`cfg`, `res`, `timeout`) carry no query text, and widening its
    /// signature would require editing `headless_runner.rs`, out of scope for this fix round.
    /// `input_size_json` falls back to `0` only when BOTH `report.input_size` and this field are
    /// `None` — the dead branch described above, now narrowed instead of silently accepted.
    pub(crate) unmeasured_fallback_tokens: Option<usize>,
}

impl RunContext {
    /// Builds the context at the point where all three inputs are actually available —
    /// **before** launching the consult, not while rendering: by render time, why the
    /// run was routed the way it was is no longer at hand.
    ///
    /// Takes [`ModeResolution::classification_attempted`], NOT [`ModeSource`]: a classification
    /// that is attempted and then expires still resolves to [`ModeSource::Default`], but the
    /// content has ALREADY gone out to the principal provider — and that is exactly the run
    /// where declaring the data-flow matters most, not least.
    ///
    /// Called from production at [`headless_runner::analyze_direct`]'s call site
    /// (`crate::headless_runner`), fed by the real `crate::config::MagiConfig` and a real
    /// `magi_rs::magi::TimeoutDecision` threaded through `MagiRuntimeParams` (fix round 1,
    /// Finding 1). `ConsultTool::execute` cannot call this: it has no
    /// `&MagiConfig`/`&TimeoutDecision` to give it (see its own rustdoc for why that is a
    /// proven invariant, not a gap) — it builds a `RunContext` literal instead.
    #[must_use]
    pub(crate) fn build(cfg: &MagiConfig, res: &ModeResolution, timeout: &TimeoutDecision) -> Self {
        Self {
            endpoint_divergence: res.classification_attempted && cfg.magi_endpoint_diverges(),
            timeout_below_formula: timeout.below_formula,
            // No query text reaches this constructor (see the field's own rustdoc) — the
            // headless path keeps the old fallback-of-last-resort in `input_size_json` for the
            // dead `None` branch, rather than a fabricated non-zero number this function has no
            // basis for.
            unmeasured_fallback_tokens: None,
        }
    }
}

/// Stable JSON labels for the effective mode. Kebab-case, matching how magi-core itself
/// serializes [`Mode`] — this is wire contract, never `Debug`, which would change silently if a
/// variant got renamed.
#[must_use]
fn mode_label(m: Mode) -> &'static str {
    match m {
        Mode::CodeReview => "code-review",
        Mode::Design => "design",
        Mode::Analysis => "analysis",
    }
}

/// See [`mode_label`]. `Configured` keeps its own label instead of collapsing into `explicit`:
/// it shares the semantics (someone chose it, so classification is skipped) but not the
/// provenance, and that difference is what makes an odd verdict auditable — `explicit` sends a
/// reader to the command, `configured` sends them to `magi.toml`.
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
/// [`failed_agents_json`] (B3) — one casing convention for an `AgentName`-keyed JSON object,
/// not two that could quietly diverge.
#[must_use]
fn seat_key(seat: AgentName) -> String {
    format!("{seat:?}").to_lowercase()
}

/// `extraction_failures`, keyed by lowercase seat name — ALWAYS present, even when empty
/// (REQ-A10). An empty map is a positive certificate that every seat adhered to the verdict
/// contract on every attempt; omitting it would make that certificate indistinguishable from
/// "this version doesn't report it".
///
/// **`model` and `cause` are safe to interpolate verbatim — verified against magi-core
/// 3.1.0, not assumed (fix round 3):**
/// - `model` is `agent.provider_model().to_string()` — the CONFIGURED model identifier for the seat (`orchestrator.rs::dispatch_one_agent`), never text derived from a network/provider error. Not third-party free text.
/// - `cause: ExtractionFailureCause` is `#[non_exhaustive]` but every variant is a **fieldless** unit case (`verdict_markers.rs`) — `format!("{:?}", x.cause)` can only ever produce one of a fixed, closed set of Debug strings (`"MissingMarkers"`, …, `"Other"`). There is structurally no way for it to carry a URL or a credential.
///
/// Contrast [`failed_agents_json`], whose `cause` is genuinely third-party free text and DOES
/// need redaction.
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

/// `failed_agents`, keyed the SAME way as [`failures_json`] (see [`seat_key`]) — built by hand
/// rather than serialized straight through `report.failed_agents`, so the two maps cannot end
/// up on different casings for what is the same kind of key.
///
/// **`cause` is redacted before it enters the JSON (B11, fix round 3, CRITICAL).** Unlike
/// [`failures_json`]'s `cause`, this one is genuinely third-party free text: verified against
/// `orchestrator.rs::dispatch_one_agent` (magi-core 3.1.0), it is literally
/// `MagiError::Provider(e).to_string()`, a `format!("timeout: …")`, or a `format!("retry-
/// failed: …")` wrapping either — none of which magi-rs controls the content of.
///
/// **Fix round 4 correction — the mechanism, not the fix, was wrong in the original
/// writeup.** magi-core 3.1.0's `Network`/`Timeout` variants are already redacted upstream:
/// `provider.rs::to_provider_error` composes their `message` from an ALREADY-redacted URL plus
/// `cause_chain(e)`, which starts at `e.source()` and so
/// **skips** the top-level client error whose `Display` embeds the raw request URL —
/// pinned by magi-core's own `cause_chain_skips_the_top_level_error` test. So an ordinary
/// connection failure against a `[user]:[password]`-substituted `base_url` (REQ-A16c) does NOT
/// leak through that specific path in 3.1.0. The real exposure this redaction guards is:
/// - **`ProviderError::Http { body }`** — the response body from whatever a request reached (a misconfigured proxy echoing headers, a captive portal, a compromised endpoint) is genuinely unredacted, server-controlled text, and it reaches `MagiReport::failed_agents` the same way `Network`/`Timeout` do.
/// - **`ProviderError` (and `Http` itself) is `#[non_exhaustive]`.** A future minor version of magi-core can add a variant that interpolates free text, and this code would start leaking again silently, with nothing on our side to change.
///
/// So this redaction is defence in depth over a boundary magi-rs does not control, not a patch
/// for a hole that is live in the `Network`/`Timeout` path today — it stays correct even where
/// the specific mechanism first suspected does not apply, and stays correct if magi-core's
/// internals change. `redact_foreign_text` is the SAME positional redaction
/// `explain_magi_error` already applies to foreign error text elsewhere in this file (B3: one
/// redaction path, not two) — it just takes the already-formatted `&str` this field already is,
/// instead of a `&dyn Error` there is none of here.
#[must_use]
fn failed_agents_json(agents: &BTreeMap<AgentName, String>) -> Value {
    let mut out = serde_json::Map::new();
    for (seat, cause) in agents {
        out.insert(
            seat_key(*seat),
            Value::String(redact_foreign_text(cause).as_str().to_string()),
        );
    }
    Value::Object(out)
}

/// `input_size` with its three sub-keys ALWAYS present (SC-A10c).
///
/// `None` from magi-core is neither omitted nor emitted as `null`: it reports what the pipeline
/// actually APPLIED — magi-core's own default as the threshold, and `exceeded` evaluated
/// against that.
///
/// **`estimated_tokens` is never a fabricated `0` (Loop 2 gate, S4 finding 3).** magi-core's own
/// rustdoc on `MagiReport::input_size` names exactly this trap: a `Default` reading
/// `estimated_tokens: 0` "is not a neutral value but the false claim that the input was empty" —
/// which is precisely what the old unconditional `0` here asserted, and provably wrong whenever
/// this function is reachable at all: every entry route validates the query non-empty
/// (`check_query_size`, REQ-A11b) before a `MagiReport` can exist. `fallback_tokens` — computed
/// by the caller from the real query where it has one at hand (see
/// [`RunContext::unmeasured_fallback_tokens`]) — replaces it when present; only when BOTH
/// `s` and `fallback_tokens` are `None` (magi-core did not measure, AND the caller had no query
/// text to fall back on — today only `RunContext::build`'s headless call site, itself dead code
/// per that field's rustdoc) does this still report `0`, as a documented last resort rather than
/// a silent one.
#[must_use]
fn input_size_json(s: Option<&InputSize>, fallback_tokens: Option<usize>) -> Value {
    match s {
        Some(v) => json!({
            "estimated_tokens": v.estimated_tokens,
            "warn_threshold": v.warn_threshold,
            "exceeded": v.exceeded,
        }),
        None => {
            let warn_threshold = CoreMagiConfig::default().input_warn_tokens;
            let estimated_tokens = fallback_tokens.unwrap_or(0);
            json!({
                "estimated_tokens": estimated_tokens,
                "warn_threshold": warn_threshold,
                "exceeded": estimated_tokens > warn_threshold,
            })
        }
    }
}

/// Builds the stable `consult` JSON object from a finished MAGI report.
///
/// Single source of truth for the shape shared by the tool-loop [`ConsultTool::execute`] path
/// and the headless direct/forced consult path (REQ-H21/H22, REQ-A11c) — so the on-the-wire
/// shape never drifts between the two entry points.
///
/// `schema_version` does not move for this telemetry (REQ-A08b), which is exactly why every
/// field below — and every sub-key of `input_size` — is always present: a field or sub-key that
/// appears only on some runs is the instability a version-less consumer cannot absorb.
///
/// # Parameters
/// * `report` - the finished multi-perspective consensus report.
/// * `truncated` - the report text actually surfaced, paired with the guarantee it carries — see [`Truncated`] for why the two travel together. Callers that also annotate the text (REQ-A12c, [`annotate_report_text`]) must do so **before** building this value: this function renders `truncated.text` verbatim.
/// * `res` - the resolved mode, its source, and whether classification was attempted.
/// * `ctx` - the per-run signals that belong in this run's own output (REQ-A11d).
///
/// # Returns
/// A JSON object with `report`, `degraded`, `mode`, `mode_source`, `extraction_failures`,
/// `input_size`, `report_truncated`, `endpoint_divergence`, `timeout_below_formula` and
/// `failed_agents` — every key always present.
pub(crate) fn report_to_consult_json(
    report: &MagiReport,
    truncated: &Truncated,
    res: &ModeResolution,
    ctx: &RunContext,
) -> Value {
    json!({
        // The (possibly annotated, possibly truncated) text — never the raw report: text and
        // level travel together in `truncated`, so `report_truncated` can never assert a
        // guarantee about text it does not describe.
        "report": truncated.text,
        "degraded": report.degraded,
        "mode": mode_label(res.mode),
        "mode_source": source_label(res.source),
        // Always present, even when empty: an empty map is the positive certificate that all
        // three seats adhered to the contract (REQ-A10).
        "extraction_failures": failures_json(&report.extraction_failures),
        // Always present WITH its sub-keys: an object that is always there but whose contents
        // vary is the same instability one level down (SC-A10c).
        "input_size": input_size_json(report.input_size.as_ref(), ctx.unmeasured_fallback_tokens),
        "report_truncated": truncation_label(truncated.level),
        // REQ-A11d: what affects how THIS run reads goes in THIS run's JSON, not a notice — a
        // batch consumer never sees stderr from a process that already exited.
        "endpoint_divergence": ctx.endpoint_divergence,
        // SC-A04d: whoever sets an explicit `--timeout` is running in a pipeline, i.e. exactly
        // who is LEAST likely to read stderr. The warning travels here too.
        "timeout_below_formula": ctx.timeout_below_formula,
        // Typed, never collapsed into `degraded`: `failed_agents`, NOT `window_rejected` — the
        // latter does not exist on `MagiReport` (verified against magi-core 3.1.0; rotation-
        // only telemetry, unreachable in MS2 without `FallbackPool`).
        "failed_agents": failed_agents_json(&report.failed_agents),
    })
}

/// Explains —ADDING to `err`'s message, never replacing it— a `MagiError::InsufficientAgents`
/// when the effective kind is keyless (REQ-A12c, SC-A12f, fix round 3).
///
/// # Why it exists: the `keyless_auth_explanation` window excludes exactly the
/// scenario that REQ-A12c describes
///
/// `keyless_auth_explanation` only sees a cause when `Magi::analyze()` returns
/// `Ok(MagiReport)`, and that requires `successful.len() >= min_agents` (2). Verified against
/// `orchestrator.rs::dispatch_no_rotation` (magi-core 3.1.0, lines 1058-1065): `if
/// successful.len() < min_agents { return Err(MagiError::InsufficientAgents { succeeded,
/// required }) }` — the `failed` map, ALREADY complete with each cause at that point, is
/// discarded right there; `MagiReport` is never built. Because the MS2 trio shares ONE
/// `base_url`/`kind` across the three seats (no rotation, R-A06), a badly chosen `kind`
/// —exactly SC-A12f's scenario, `kind = "ollama"` against an endpoint that demands
/// authentication— rejects ALL THREE equally: 0 of 3 succeeded, this path, not the other.
///
/// # Why this DOES reach without the per-agent cause
///
/// REQ-A12c asks to name the configuration as a **probable cause** ("no impossible validation
/// is requested... the inevitable failure is required to arrive explained"), not a demonstrated
/// one. The combination "zero seats completed" + "the effective kind is keyless" is, by itself,
/// sufficient evidence for that threshold — without needing the status code this path never
/// has. That is why the kind guard is still mandatory here too: under a kind WITH credentials,
/// a total failure says as little about configuration as any other outage, and offering the
/// hint would send the user to check the wrong file — the same harm
/// [`keyless_auth_explanation`] avoids on the other side.
///
/// # Parameters
/// * `err` - the error returned by `Magi::analyze()`.
/// * `kind` - the `ProviderKind` the trio ran under.
///
/// # Returns
/// The `Display` of `err` (redacted — B11, see below), with the keyless hint ADDED (never in
/// its place) when `err` is `InsufficientAgents` and `kind` is `Ollama`. In any other case,
/// only the `Display` of `err`.
///
/// Consumed by `ConsultTool::execute`, `analyze_direct` (headless), AND —since fix round 4,
/// finding 2— the explicit TUI `/consult` (`src/tui/mod.rs::tui_consult_error_body`): the three
/// surfaces share this single function instead of each one writing its own translation (B3).
#[must_use]
pub(crate) fn explain_magi_error(err: &MagiError, kind: ProviderKind) -> String {
    // B11 — `redact_foreign_error`, NEVER `redact_url`, and the difference matters here:
    // `redact_url` assumes the ENTIRE input is a URL and fully redacts anything it cannot
    // traverse as such (`locate_userinfo` returns `Unparseable` for any string without `://`) —
    // applied here it would have reduced EVERY `MagiError` message without a URL (e.g.
    // "insufficient agents: 0 succeeded, 2 required") to `"***"`, a real bug that caught
    // `explain_magi_error_preserves_a_url_free_underlying_message` the first time this test ran
    // (see this task's report). `redact_foreign_error` traverses PROSE looking for EMBEDDED
    // URLs and redacts only those, leaving the rest intact — the same primitive
    // `build_native_provider::to_seat` already uses for the same problem: a foreign `Display`
    // that MIGHT bring a URL, not one that IS one. `MagiError` is `#[non_exhaustive]`, so a
    // future variant indeed could interpolate a URL; it is always redacted, not only for
    // today's variants.
    let base = redact_foreign_error(err);
    match (err, kind) {
        (MagiError::InsufficientAgents { .. }, ProviderKind::Ollama) => {
            format!("{base} — possible cause: {KEYLESS_AUTH_EXPLANATION}")
        }
        _ => base.to_string(),
    }
}

/// RAII backstop that aborts a spawned task when the guard is dropped.
///
/// [`ConsultTool::execute`] runs the 3-perspective analysis on a `tokio::spawn` task and awaits
/// it under a `select!`. The explicit cancel arm aborts the task on `--timeout`, but if the
/// `execute` future itself is *dropped* before either arm resolves (e.g. the caller drops the
/// tool call), a bare spawned task would keep running and orphan its three in-flight LLM calls.
/// Holding this guard across the `select!` aborts the task on that drop too, mirroring the
/// `GroupKiller` backstop the `bash` tool uses for its subprocess.
///
/// `pub(crate)` so [`crate::headless_runner`]'s direct `magi consult` path (`analyze_direct`)
/// reuses this exact primitive for its own spawned MAGI analysis rather than duplicating it —
/// same gap, same fix, one guard type.
pub(crate) struct AbortOnDrop {
    /// Abort handle of the guarded task.
    handle: tokio::task::AbortHandle,
}

impl AbortOnDrop {
    /// Wraps a task's abort handle so dropping the guard aborts the task.
    pub(crate) fn new(handle: tokio::task::AbortHandle) -> Self {
        Self { handle }
    }

    /// Aborts the guarded task now. Idempotent: aborting an already-finished or already-aborted
    /// task is a no-op, so `Drop` re-invoking it is harmless.
    pub(crate) fn abort(&self) {
        self.handle.abort();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Notice emitted in the TUI when the `consult` tool is auto-approved. Visible to the user so
/// they know the 3-LLM consensus was launched.
const AUTO_LAUNCH_NOTICE: &str = "launched MAGI multi-perspective consensus — awaiting evaluation…";

/// Resolves the (mode, source) pair to dispatch with (MS2, REQ-A20/REQ-A07d/REQ-A08): reads
/// what the agent's tool loop already resolved and injected under the reserved
/// `__resolved_mode`/`__resolved_mode_source` keys
/// (`magi_rs::magi::mode::inject_resolved_mode`) instead of re-resolving it here — a tool that
/// could disagree with the gate that evaluated the same call would reopen exactly the
/// divergence REQ-A07d closes.
///
/// Falls back to `(`[`Mode::Analysis`]`, `[`ModeSource::Default`]`)` when the keys are
/// **absent**. This is a deliberate, narrow back-compat default, not a re-resolution of
/// untrusted input: **both** real production dispatch paths in `Agent::run_tool_loop` inject
/// the pair before calling [`ConsultTool::execute`] — the model-issued `ToolUse` route
/// (`Agent::dispatch_consult_through_gate`) and the forced pre-loop injection (REQ-H22's
/// `config.force_consult` block, which resolves and injects too, just without ever evaluating
/// the gate: REQ-A20 forbids vetoing a forced consult, but it still needs a resolved mode). So
/// this fallback is reached only by a caller that invokes [`ConsultTool::execute`] directly
/// without going through the loop — exactly what this module's own pre-MS2 tests do, and
/// precisely the unconditional `Mode::Analysis` this tool used before MS2 (the source half is
/// new in Task 6.1, needed to name `mode_source` in [`report_to_consult_json`]'s output).
/// Wiring `ConsultTool` fully into `ConsultToolCfg` (a later task) can tighten this to a hard
/// error once every production caller is confirmed to inject.
fn resolved_mode_and_source(args: &Value) -> (Mode, ModeSource) {
    read_resolved_mode(args).unwrap_or((Mode::Analysis, ModeSource::Default))
}

/// Tool wrapping a `magi_core::Magi`. `execute` runs the 3-perspective consensus (implemented
/// in Task 4) and returns the verbatim report. The `description` is what makes the main LLM
/// self-route here only for multi-perspective decisions.
pub struct ConsultTool {
    magi: Arc<Magi>,
    description: String,
    /// When `true`, autonomous MAGI launches via the agent tool loop are auto-approved (no
    /// `ApprovalRequest` emitted). The explicit `/consult` TUI command path is NEVER gated
    /// regardless of this flag.
    auto_approve: bool,
    /// The `ProviderKind` the trio runs under (REQ-A12c) — feeds [`keyless_auth_explanation`]
    /// via [`report_to_consult_json`]. Defaults to [`ProviderKind::OpenAiCompat`] (see
    /// [`Self::new`]), under which the explanation never applies — a caller that does not care
    /// about this feature does not need to call [`Self::with_kind`].
    kind: ProviderKind,
    /// `crate::config::MagiConfig::magi_endpoint_diverges()`, resolved ONCE at construction
    /// (fix round 1, Finding 1) — feeds `RunContext.endpoint_divergence` in [`Tool::execute`].
    /// Defaults to `false` (see [`Self::new`]).
    ///
    /// **This alone never flips `endpoint_divergence` to `true` at this call site**,
    /// and that is not a bug: `Tool::execute`'s own rustdoc proves `classification_attempted`
    /// is structurally always `false` here, so the AND is always `false` regardless of this
    /// field's value. It is wired anyway — correctness should not depend on an invariant
    /// elsewhere in the codebase never changing, and a future change that makes it real needs
    /// no further plumbing.
    magi_endpoint_diverges: bool,
    /// Effective input cap (`MagiConfig::effective_max_query_bytes`, REQ-A11b), checked by
    /// [`check_query_size`] before any model call. Defaults to [`MAX_QUERY_BYTES`] (see
    /// [`Self::new`]); call [`Self::with_max_query_bytes`] to declare an operator-configured
    /// value.
    max_query_bytes: usize,
    /// Effective output cap (`MagiConfig::effective_tool_result_cap`, REQ-A11b), applied to the
    /// annotated report via [`truncate_report`] before it becomes this call's `ToolResult` —
    /// the string that re-enters the conversation history and gets re-sent on every subsequent
    /// turn of the session, which is why this one call site bounds BOTH the TUI's auto-routed
    /// consult and the headless `magi query` tool loop (`Agent::authorize_and_execute_tool`
    /// serializes every tool's `Value` result the same way, for every caller). Defaults to
    /// [`TOOL_RESULT_CAP_BYTES`]; call [`Self::with_output_cap`] to declare an operator-
    /// configured value.
    output_cap: usize,
}

impl ConsultTool {
    /// Creates a `ConsultTool` over a shared `Magi` orchestrator.
    ///
    /// # Parameters
    /// * `magi` - Shared `Magi` orchestrator that drives the 3-perspective consensus.
    /// * `auto_approve` - When `true`, the tool opts out of the approval gate for autonomous launches (the agent tool loop will auto-approve it and emit a TUI notice). Default is `false` — the agent asks before each launch.
    ///
    /// # Returns
    /// A new `ConsultTool` instance with a routing-tuned description and `kind` defaulted to
    /// [`ProviderKind::OpenAiCompat`] — call [`Self::with_kind`] to declare the trio's real
    /// kind when REQ-A12c's explanation should apply.
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
            // Neutral: `keyless_auth_explanation` only ever fires under `Ollama`, so this
            // default is equivalent to the feature being off until declared.
            kind: ProviderKind::OpenAiCompat,
            // Safe default: "no divergence" until a caller declares otherwise via
            // `Self::with_magi_endpoint_diverges`, mirroring `kind`'s own default.
            magi_endpoint_diverges: false,
            // Built-ins until a caller declares an operator-configured value via
            // `Self::with_max_query_bytes`/`Self::with_output_cap` — the ~13 existing test call
            // sites that do not care about either cap keep working unchanged, same reasoning as
            // `kind`'s default above.
            max_query_bytes: MAX_QUERY_BYTES,
            output_cap: TOOL_RESULT_CAP_BYTES,
        }
    }

    /// Declares the `ProviderKind` the trio runs under (REQ-A12c).
    ///
    /// Builder-style (`self` by value) so production call sites read as `ConsultTool::new(magi,
    /// auto_approve).with_kind(kind)` without a second mutable binding, and so the ~13 existing
    /// test call sites that do not care about this feature do not need to change at all.
    #[must_use]
    pub fn with_kind(mut self, kind: ProviderKind) -> Self {
        self.kind = kind;
        self
    }

    /// Declares whether the trio runs on an endpoint different from the principal provider
    /// (`crate::config::MagiConfig::magi_endpoint_diverges()`), fix round 1 Finding 1. Same
    /// builder shape as [`Self::with_kind`], for the same reason.
    #[must_use]
    pub fn with_magi_endpoint_diverges(mut self, diverges: bool) -> Self {
        self.magi_endpoint_diverges = diverges;
        self
    }

    /// Declares the effective input cap (`MagiConfig::effective_max_query_bytes`, REQ-A11b).
    /// Same builder shape as [`Self::with_kind`], for the same reason.
    #[must_use]
    pub fn with_max_query_bytes(mut self, cap: usize) -> Self {
        self.max_query_bytes = cap;
        self
    }

    /// Declares the effective output cap (`MagiConfig::effective_tool_result_cap`, REQ-A11b).
    /// Same builder shape as [`Self::with_kind`], for the same reason.
    #[must_use]
    pub fn with_output_cap(mut self, cap: usize) -> Self {
        self.output_cap = cap;
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

    /// When `auto_approve = false` (the default), autonomous MAGI launches are gated — the
    /// agent prompts the user before each 3-LLM consensus call. When `auto_approve = true`, the
    /// agent tool loop auto-approves the call and emits an [`Self::approval_notice`] in the TUI
    /// instead.
    fn requires_approval(&self) -> bool {
        !self.auto_approve
    }

    /// Returns an announcement notice when the tool is auto-approved.
    ///
    /// The notice is sent as a `StreamPiece::Notice` **before** the tool runs, so the user
    /// knows the 3-LLM consensus was launched without a prompt. Returns `None` when
    /// `auto_approve = false` (the gate prompts the user instead, so no proactive notice is
    /// needed).
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
        // SC-A11c: the SAME `check_query_size` the direct headless path and the TUI's explicit
        // `/consult` call, so the three routes reject with the same limit instead of three
        // copies that could drift apart.
        check_query_size(query, self.max_query_bytes)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let (mode, source) = resolved_mode_and_source(&args);
        let magi = self.magi.clone();
        let q = query.to_string();
        // Joined spawn isolates a panic in magi-core's analyze into a recoverable JoinError
        // instead of unwinding into the agent tool loop.
        let handle = tokio::spawn(async move { magi.analyze(&mode, &q).await });
        // RAII backstop: aborts the spawned analysis if this `execute` future is dropped before
        // the `select!` resolves — a dropped tool call would otherwise orphan the three in-
        // flight LLM calls. The abort handle is taken separately from `handle` (which the
        // `select!` consumes), so there is no borrow conflict.
        let abort_guard = AbortOnDrop::new(handle.abort_handle());
        // A proactive consult runs three MAGI LLM calls; on the run's `--timeout` cancellation
        // (REQ-H36) the task is **aborted** — not merely detached — so those expensive API
        // calls actually stop instead of being orphaned. `biased` polls the cancel arm first,
        // so an already-cancelled token short-circuits before the analysis is awaited.
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
        // The annotation (REQ-A12c) is applied BEFORE truncating — `report_to_consult_json`
        // renders `truncated.text` verbatim, so the annotation step has to happen here, not
        // inside `truncate_report`, or the keyless-auth hint this file already surfaces end-to-
        // end (see `a_keyless_auth_failure_reaches_the_ consult_report_end_to_end`) would
        // silently stop reaching the user, and a report cut right at the annotation boundary
        // could drop it either way.
        let annotated = annotate_report_text(&report, self.kind);
        // REQ-A11b/SC-A11d: bounds the string that becomes this call's `ToolResult` — the text
        // re-sent on every subsequent turn of the session — with the same truncation-level
        // vocabulary the JSON exposes via `report_truncated`.
        let truncated = truncate_report(&annotated, self.output_cap);
        // `classification_attempted: false` is a PROVEN invariant here, not a placeholder (fix
        // round 1, Finding 1). `Tool::execute` is reachable from production through exactly two
        // funnel call sites — `Agent::dispatch_consult_through_gate` and the forced pre-loop
        // injection in `Agent::run_tool_loop` (REQ-H22) — and BOTH call
        // `magi_rs::magi::mode::resolve_mode_guarded(..., classifier: None, ...)`. That is not
        // incidental: REQ-A07d/SC-A07u require the agent tool loop to NEVER pay for a
        // classification call — the agent's own free `AgentChosen` level (via this tool's
        // `mode` input-schema field) already covers this route at zero cost, which is the
        // entire point of that level existing. So no `ModeResolution` reaching this function
        // via the funnel can ever carry `classification_attempted: true`, and
        // `inject_resolved_mode` does not even round-trip the flag (only mode+source) — there
        // being nothing honest to read here is a consequence of the design, not a gap in it.
        let res = ModeResolution {
            mode,
            source,
            classification_attempted: false,
        };
        let ctx = RunContext {
            // `res.classification_attempted` is always `false` (see above), so this is always
            // `false` too, REGARDLESS of `self.magi_endpoint_diverges` — computed via the real
            // field rather than hardcoded, so a future change to the invariant above does not
            // silently leave this wrong.
            endpoint_divergence: res.classification_attempted && self.magi_endpoint_diverges,
            // No `--timeout` concept reaches a tool-loop-dispatched consult at all: `execute`
            // receives a `CancellationToken` shared with the WHOLE agent run, not a `Duration`
            // dedicated to this one consult, so REQ-A04's "does `--timeout` leave room for the
            // trio's own retry formula" comparison — which assumes a budget dedicated to one
            // consult — does not apply here the way it does to `magi consult`'s direct path
            // (`headless_runner::analyze_direct`, which DOES wire this for real).
            timeout_below_formula: false,
            // Loop 2 gate, S4 finding 3: `query` is right here, non-empty (`check_query_size`
            // already rejected an empty one above) — an honest fallback for `input_size_json`'s
            // `None` arm instead of the `0` it used to fabricate.
            unmeasured_fallback_tokens: Some(bytes_to_tokens_est(query.len())),
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

    /// Upper bound on how long a *cancelled* `execute` may take to return. The cancel path
    /// aborts the in-flight analysis, so it must resolve almost immediately; sized generously
    /// to absorb scheduler jitter while staying far below the blocking provider's
    /// [`BlockingProvider::SLEEP_SECS`] sleep, so a regression that awaited the full analysis
    /// would blow this budget.
    const CANCEL_RETURN_BUDGET_MS: u128 = 2_000;

    /// A provider whose `complete` blocks far longer than any test tolerates, so a MAGI
    /// analysis over it never finishes within the test. Used to prove that
    /// [`ConsultTool::execute`] returns on cancellation *without* waiting for the analysis: if
    /// the cancel token were ignored, `execute` would block on this sleep and overrun
    /// [`CANCEL_RETURN_BUDGET_MS`].
    struct BlockingProvider;

    impl BlockingProvider {
        /// Sleep duration of each `complete` call — long enough that awaiting the full analysis
        /// is unmistakably distinguishable from a prompt cancel.
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

    /// Canonical response from a mage, in the format magi-core 3.x requires.
    ///
    /// Since 3.0.0 the verdict is read **only** between [`VERDICT_OPEN`] and [`VERDICT_CLOSE`],
    /// each marker on its own line. A bare JSON —the format that worked in 2.x— is no longer
    /// parsed and the mage counts as failed. The crate constants are used instead of literals
    /// so a marker change breaks compilation instead of silently degrading the fixture.
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
    /// `ClaudeProvider::map_status_to_error(401, ..)` — public, not hand-rolled) — the ONE case
    /// genuinely reachable via `MagiReport::failed_agents` (`min_agents = 2`,
    /// `ConsensusConfig::default()`, verified in `keyless_auth_explanation`'s own rustdoc).
    /// Caspar is the failing seat so the tests below can assert on its name specifically.
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

    /// SC-A12f: a real `ProviderError::Auth` rendering — as magi-core itself produces and pins
    /// it, not a hand-rolled string — is recognized as a keyless auth failure under `ollama`.
    ///
    /// **Positive control (mandatory per this task's fix-round-1 review).** Both
    /// `OpenAiCompatibleProvider::map_status_to_error` (magi-core
    /// `src/providers/openai_compat.rs:280`) and `ClaudeProvider::map_status_to_error`
    /// (`src/providers/claude.rs:247`) map 401/403 to `ProviderError::Auth` — the same enum
    /// variant, same crate, same `Display`. `ClaudeProvider`'s mapper is `pub fn` (the OpenAI-
    /// compat one is `pub(crate)`, unreachable from here), so it is used to construct the REAL
    /// value; the contract pinned by this test (`ProviderError::Auth`'s `"auth error: "`
    /// `Display` prefix) is a property of `error.rs`, shared identically by both providers
    /// regardless of which one built the value. If a future magi-core release changes that
    /// wording, THIS test goes red — not silently false forever.
    #[test]
    fn keyless_auth_explanation_recognizes_a_real_provider_error_auth_rendering() {
        let provider_err = magi_core::providers::claude::ClaudeProvider::map_status_to_error(
            401,
            "x",
            vec![],
            None,
        );
        // The exact composition `dispatch_one_agent` performs on a first-attempt provider
        // failure (magi-core `orchestrator.rs:1298-1304`):
        // `MagiError::Provider(provider_err).to_string()`.
        let cause = magi_core::error::MagiError::Provider(provider_err).to_string();
        assert_eq!(
            keyless_auth_explanation(&cause, ProviderKind::Ollama),
            Some(KEYLESS_AUTH_EXPLANATION),
            "a real magi-core auth rendering must be recognized: {cause:?}"
        );
    }

    /// SC-A12f: under a kind that DOES carry a credential, the same real rendering is left
    /// alone — a 401 there can be a genuinely bad credential, and reinterpreting it would send
    /// the user to the wrong file.
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

    /// Edge case (B13): a cause that is NOT an auth failure (e.g. a timeout) is never
    /// reinterpreted, even under `ollama`.
    #[test]
    fn keyless_auth_explanation_ignores_causes_without_the_auth_marker() {
        let cause = "timeout: agent timed out after 90s";
        assert_eq!(keyless_auth_explanation(cause, ProviderKind::Ollama), None);
    }

    /// B11: the explanation is FIXED text — it never echoes `cause`, so a secret that somehow
    /// ended up in third-party diagnostic text cannot reach the surfaced message through this
    /// function.
    #[test]
    fn keyless_auth_explanation_never_echoes_the_raw_cause() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let cause = format!("{PROVIDER_AUTH_ERROR_MARKER}token={CANARY}");
        let explanation =
            keyless_auth_explanation(&cause, ProviderKind::Ollama).expect("marker present");
        assert!(!explanation.contains(CANARY), "{explanation}");
    }

    /// SC-A12f, end to end: a seat that fails on auth under `ollama` reaches the user through
    /// the ACTUAL `ConsultTool::execute` → `report_to_consult_json` path, not just the pure
    /// predicate. This is the wiring proof: a correct `keyless_auth_explanation` nobody calls
    /// from `execute` would pass every test above and still leave the user looking at the raw
    /// "auth error: x" text.
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

    /// Same failing seat, but under a kind that carries a credential: the raw cause reaches the
    /// report (nothing hides it), but it is NEVER reinterpreted as a keyless-configuration
    /// problem.
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
    /// `keyless_auth_explanation`/`annotate_report_text` CANNOT see (0 of 3 succeeded <
    /// `min_agents` 2 ⇒ `Magi::analyze()` returns `Err`, and `MagiReport`/`failed_agents` is
    /// never constructed). This is the scenario SC-A12f actually describes: a `kind` mismatch
    /// rejects every seat identically because they share one `base_url`/`kind` (no rotation,
    /// R-A06).
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

    /// SC-A12f: on the total-failure path (`MagiError::InsufficientAgents`, no per-agent cause
    /// available), a keyless kind is enough on its own to name the configuration as a
    /// **probable** cause — REQ-A12c's own words ("probable cause", not demonstrated). The hint
    /// is ADDED to the underlying message, never in its place.
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
        // Register check: this path has NO per-agent evidence, so it must not borrow the
        // confident opening `annotate_report_text` uses when it DOES have evidence
        // (`PROVIDER_AUTH_ERROR_MARKER` present in a real cause).
        assert!(
            !msg.contains("rejected due to authentication"),
            "no per-agent evidence here: must not claim auth was the cause: {msg}"
        );
    }

    /// SC-A12f: under a kind that carries a credential, a total failure says nothing about
    /// configuration — the hint must NOT appear, or it would send the user to the wrong file
    /// (same guard [`keyless_auth_explanation`] enforces).
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

    /// Edge case (B13): the hint is specific to `InsufficientAgents` — a DIFFERENT `MagiError`
    /// variant, even under `ollama`, gets no hint, because "input too large" says nothing about
    /// authentication.
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

    /// Regression: `explain_magi_error` must use `redact_foreign_error`, NEVER `redact_url`, on
    /// the underlying message. An earlier version of this function used `redact_url`, which
    /// assumes its ENTIRE input is a URL and fully redacts anything it cannot parse as one
    /// (`locate_userinfo` returns `Unparseable` for any string without `://`) — applied to a
    /// `MagiError` message with no embedded URL (the common case: "insufficient agents: 0
    /// succeeded, 2 required" has none), that reduced the entire diagnostic to a bare `"***"`.
    /// This test caught that the first time it ran; it stays here so a future edit can't
    /// silently reintroduce the wrong primitive.
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

    /// SC-A12f, end to end: a TOTAL failure (0 of 3 seats) under `ollama` reaches the user
    /// through the ACTUAL `ConsultTool::execute` path, not just the pure `explain_magi_error`
    /// predicate — this is the wiring proof for the window `annotate_report_text` cannot cover.
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

    /// Same total failure, but under a kind that carries a credential: the hint must NOT appear
    /// — this is the negative case that proves the guard is real (an unconditional hint would
    /// pass the positive test above too).
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

    /// `ConsultTool` with `auto_approve = true` must return `Some(notice)` from
    /// `approval_notice`.
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
        // `required` names ONLY `query`: `mode` must stay optional so an agent that doesn't
        // pick a lens still gets to consult (REQ-A07/A07b).
        assert_eq!(schema["required"].as_array().unwrap().len(), 1);
        let lower = tool.description().to_lowercase();
        assert!(!lower.is_empty());
        assert!(lower.contains("trade-off"));
        assert!(lower.contains("perspective") || lower.contains("perspectives"));
        assert!(lower.contains("decision") || lower.contains("decisions"));
    }

    /// REQ-A07b: the tool exposes `mode` in its own input schema so an agent that decides to
    /// consult can also pick the lens, from the same three-label vocabulary
    /// `magi_rs::magi::mode::normalize_label` accepts. No behavior change to `execute` — it
    /// still hardcodes `Mode::Analysis`; wiring the declared value into dispatch is Task
    /// 2.3/2.4's job, not this one's.
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
        // REQ-A11b raised the default cap from the old 8 KiB `MAX_QUERY_LEN` to
        // `MAX_QUERY_BYTES` (256 KiB, SC-A11): a 9 KiB string that used to be rejected now
        // legitimately fits, so the fixture must genuinely exceed the cap this tool now
        // enforces.
        let tool = ConsultTool::new(magi_all_ok(), false);
        let big = "x".repeat(MAX_QUERY_BYTES + 1);
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

    /// A pre-cancelled token makes `execute` return the cancellation error promptly, aborting
    /// the in-flight 3-LLM analysis instead of running it to completion (REQ-H36). Uses
    /// [`BlockingProvider`] so the analysis would otherwise block for an hour: returning within
    /// [`CANCEL_RETURN_BUDGET_MS`] proves the cancel path pre-empts the work rather than
    /// awaiting it.
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

    /// The `AbortOnDrop` backstop aborts its guarded task the instant the guard is dropped, so
    /// a dropped `execute` future cannot orphan the spawned analysis (the drop path the
    /// explicit cancel arm does not cover). A bare dropped `JoinHandle`/`AbortHandle` would
    /// merely detach the task, leaving it to run to completion — this asserts the join reports
    /// cancellation and the task never reached its completion store.
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

    /// MS2 (REQ-A20/REQ-A07d/REQ-A08): the resolved (mode, source) pair injected by the agent's
    /// tool loop wins over the `(Analysis, Default)` fallback — happy path (injected) and the
    /// two edge cases (absent, corrupt) that must degrade to the default pair.
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
    // Task 6.1 (REQ-A08b/A09/A10/A11c/A11d/A11g/A11h) — telemetry: `report_to_consult_json`,
    // `RunContext::build`.
    // -----------------------------------------------------------------------

    /// Builds a real `MagiReport` via `Deserialize` — the only way to construct one outside
    /// magi-core, since the struct is `#[non_exhaustive]` with no public constructor and
    /// `Magi::analyze()` cannot be coaxed into returning `Ok` with an empty `agents` (its own
    /// `min_agents` gate rejects that before a report is ever built — verified against
    /// `orchestrator.rs::dispatch_no_rotation`, magi-core 3.1.0). `agents`/`consensus`/`banner`
    /// are minimal-but-valid filler except where a specific test cares about them.
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

    /// A report where nothing went wrong: no extraction failures, no execution failures, input
    /// under threshold. `agents` is irrelevant to `report_to_consult_json` (it never reads that
    /// field), so it stays empty here.
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

    /// A report with an extraction failure on Caspar AND an input-size overage — "failures and
    /// excess". SC-A09c/A10b/A10c want this to yield the SAME JSON shape as `clean_report`
    /// despite carrying more data; SC-A08 wants a named, typed failure to read off it.
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

    /// A report where Caspar failed to EXECUTE (network/credential/timeout) — as opposed to the
    /// extraction failure above. SC-A11f's `failed_agents` case.
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

    /// Local double: the report's own text, unmodified — no truncation logic exists yet, so
    /// every caller in this module today passes `TruncationLevel::None`.
    fn untruncated(r: &MagiReport) -> Truncated {
        Truncated {
            text: r.report.clone(),
            level: TruncationLevel::None,
        }
    }

    /// Local double: a fixed mode resolution. `classification_attempted` mirrors `source ==
    /// Inferred` — the only one of the fixed sources these tests use that actually implies a
    /// classification call was made.
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
            unmeasured_fallback_tokens: None,
        }
    }

    /// Local double: a `RunContext` declaring endpoint divergence.
    fn ctx_with_divergent_endpoint() -> RunContext {
        RunContext {
            endpoint_divergence: true,
            timeout_below_formula: false,
            unmeasured_fallback_tokens: None,
        }
    }

    /// Local double: a `RunContext` carrying an honest fallback estimate, as
    /// `ConsultTool::execute` would supply from a real, non-empty query
    /// (Loop 2 gate, S4 finding 3).
    fn ctx_with_fallback(tokens: usize) -> RunContext {
        RunContext {
            endpoint_divergence: false,
            timeout_below_formula: false,
            unmeasured_fallback_tokens: Some(tokens),
        }
    }

    /// SC-A09b and SC-A07 (the magi-rs half): the JSON grows with the new telemetry fields. `schema_version`
    /// (`src/headless/output.rs`, an unrelated module) does not move for it (REQ-A08b) —
    /// verified structurally by this task simply never touching that file; its `SCHEMA_VERSION`
    /// constant is private with no public accessor this module could assert against.
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
        // SC-A07 (magi-rs half): the mode that comes out is the one that was resolved, and
        // it carries its origin. Asserting only the key's presence would pass even if the
        // builder emitted a different lens than the caller resolved.
        assert_eq!(v["mode"], "code-review");
        assert_eq!(v["mode_source"], "inferred");
    }

    /// SC-A09 / SC-A09c / SC-A10b / SC-A10c: the shape does NOT vary with the outcome, and a
    /// clean run's empty `extraction_failures` is the positive certificate SC-A09 asks for.
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

    /// Loop 2 gate, S4 finding 3: an honest fallback replaces the fabricated `0`.
    ///
    /// `report_with_one_failed_mage` carries `input_size: None` — magi-core did not measure.
    /// With a caller-supplied fallback (the shape `ConsultTool::execute` builds from a real,
    /// non-empty query), the JSON must report THAT number, never `0` — a query that passed
    /// `check_query_size` is provably non-empty, so `estimated_tokens: 0` would be a knowingly
    /// false claim, exactly the trap magi-core's own `InputSize` rustdoc names.
    #[test]
    fn an_unmeasured_input_reports_the_callers_honest_fallback_not_a_fabricated_zero() {
        let r = report_with_one_failed_mage();
        let v = report_to_consult_json(
            &r,
            &untruncated(&r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_with_fallback(42),
        );
        assert_eq!(
            v["input_size"]["estimated_tokens"], 42,
            "the honest fallback must reach the JSON, not a fabricated 0"
        );
        assert!(
            !v["input_size"]["exceeded"].as_bool().unwrap(),
            "42 tokens sits far under magi-core's default warn threshold"
        );
        assert!(
            !v["input_size"]["warn_threshold"].is_null(),
            "the threshold actually applied is still magi-core's own default"
        );
    }

    /// The dead branch (no magi-core measurement AND no caller fallback — today only
    /// `RunContext::build`'s headless call site) is a documented last resort, not a silent one:
    /// it still reports `0`, but the shape stays stable and `exceeded` is derived from that same
    /// `0` rather than independently hardcoded — pinning the fallback-of-last-resort behavior so
    /// a future change to it is deliberate, not incidental.
    #[test]
    fn the_last_resort_fallback_still_yields_a_stable_shape() {
        let r = report_with_one_failed_mage();
        let v = report_to_consult_json(
            &r,
            &untruncated(&r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        assert_eq!(v["input_size"]["estimated_tokens"], 0);
        assert!(!v["input_size"]["exceeded"].as_bool().unwrap());
        assert!(!v["input_size"]["warn_threshold"].is_null());
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

    /// SC-A11f: endpoint divergence and a failed seat both travel in THIS run's own output, not
    /// a process-wide notice.
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

    /// SC-A13 / B11 (fix round 3, CRITICAL): a `cause` string reaching `failed_ agents` is
    /// THIRD-PARTY text — magi-core's own `ProviderError` rendering, e.g.
    /// `MagiError::Provider(e).to_string()` (verified against `orchestrator.
    /// rs::dispatch_one_agent`) — and `failed_agents_json` must redact it before it ever
    /// reaches the JSON. A test that redacts a benign string proves nothing about this: the
    /// credential must actually be present in the input and actually absent from the output.
    ///
    /// Fix round 4 correction: the URL-with-userinfo canary below is REPRESENTATIVE of the
    /// shape a `ProviderError::Http { body }` (an unredacted, server-controlled response body —
    /// `#[non_exhaustive]`, so a future magi-core variant could carry free text too) could
    /// produce, not a pin of a mechanism that actually occurs on the `Network`/`Timeout` path —
    /// magi-core 3.1.0 already redacts THAT path upstream (`provider.rs:: to_provider_error`,
    /// `cause_chain_skips_the_top_level_error`). This test exercises the redaction generically,
    /// at the boundary, regardless of which `ProviderError` variant a future magi-core version
    /// routes a URL through.
    #[test]
    fn a_credential_bearing_cause_never_reaches_the_json() {
        const CANARY: &str = "hunter2-s3cr3t";
        let r = report_fixture(
            json!([]),
            json!({
                "caspar": format!(
                    "network error: connect to https://user:{CANARY}@host:8443/v1 failed"
                ),
            }),
            json!({}),
            Value::Null,
            true,
            "a degraded consensus report",
        );
        let v = report_to_consult_json(
            &r,
            &untruncated(&r),
            &res_of(Mode::Analysis, ModeSource::Default),
            &ctx_plain(),
        );
        let rendered = v.to_string();
        assert!(
            !rendered.contains(CANARY),
            "credential leaked into the consult JSON: {rendered}"
        );
        assert!(
            rendered.contains("host:8443"),
            "the host must survive redaction — still accionable: {rendered}"
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

    /// Local double: a resolution with only the attempt flag varied — the one field these
    /// `RunContext::build` tests exercise.
    fn resolution(attempted: bool) -> ModeResolution {
        ModeResolution {
            mode: Mode::Analysis,
            source: ModeSource::Default,
            classification_attempted: attempted,
        }
    }

    /// SC-A11f: `endpoint_divergence` is populated where the decision is actually made —
    /// `RunContext::build`, fed by the REAL `crate::config::MagiConfig` and the REAL
    /// `magi_rs::magi::resolve_run_timeout`.
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

    /// SC-A11f (the case the naive predicate would lose): classification ATTEMPTED and FAILED.
    ///
    /// This is what distinguishes "attempted" from `ModeSource::Inferred`. A classification
    /// that expires leaves `ModeSource::Default`, but the content has ALREADY gone out to the
    /// principal provider — and that is the run where declaring the data-flow matters most, not
    /// least.
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

    /// Fix round 1, Finding 1 (SC-A11f/REQ-A11d): `endpoint_divergence` and
    /// `timeout_below_formula` stay `false` through the REAL `Tool::execute` call site — driven
    /// end to end, not asserted against a hand-built `RunContext`, which is exactly what let
    /// the previous hardcoded-`false` version of this code sit under a green suite. Setting
    /// `.with_magi_endpoint_diverges(true)` and still observing `false` in the output is the
    /// proof that `classification_attempted` is what gates it, not a value this call site
    /// simply forgot to read.
    #[tokio::test]
    async fn tool_loop_dispatch_never_reports_endpoint_divergence_or_timeout_below_formula() {
        let tool = ConsultTool::new(magi_all_ok(), false).with_magi_endpoint_diverges(true);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("3 agents → success");
        assert_eq!(
            out["endpoint_divergence"], false,
            "the agent tool loop never classifies (REQ-A07d/SC-A07u), so this must \
             stay false even with magi_endpoint_diverges declared true"
        );
        assert_eq!(out["timeout_below_formula"], false);
    }

    // -----------------------------------------------------------------------
    // Task 6.2 (REQ-A11b) — the unified input cap and the output truncation levels:
    // `check_query_size`, `truncate_report`, `head_chars`.
    // -----------------------------------------------------------------------

    /// SC-A11b: an empty (or whitespace-only) query is rejected — spending three model calls on
    /// nothing to analyze is pure waste.
    #[test]
    fn check_query_size_rejects_an_empty_or_whitespace_only_query() {
        assert_eq!(check_query_size("   ", 100), Err(ConsultInputError::Empty));
        assert_eq!(check_query_size("", 100), Err(ConsultInputError::Empty));
    }

    /// SC-A11: the raised default cap lets a realistic review diff through — with the old 8 KiB
    /// `MAX_QUERY_LEN` this would have been rejected outright.
    #[test]
    fn a_realistic_diff_now_goes_through() {
        let diff = "x".repeat(200 * 1024);
        assert!(
            check_query_size(&diff, MAX_QUERY_BYTES).is_ok(),
            "with the old 8 KiB cap this would have been rejected: ~50k tokens \
             against a real review diff"
        );
    }

    /// SC-A11b: over the cap is REJECTED — with the exact size/cap named, never silently
    /// truncated (a silently truncated payload produces a verdict indistinguishable from a
    /// legitimate one, per this function's own rustdoc).
    #[test]
    fn check_query_size_rejects_over_the_cap_and_names_the_amount() {
        let over = "x".repeat(101);
        assert_eq!(
            check_query_size(&over, 100),
            Err(ConsultInputError::TooLarge {
                size: 101,
                cap: 100
            })
        );
        // Exactly at the cap is accepted — the boundary is inclusive.
        assert_eq!(check_query_size(&"x".repeat(100), 100), Ok(()));
    }

    /// SC-A11c: `ConsultTool::execute` rejects with ITS configured cap — the same function the
    /// other two routes call, not a re-derived copy.
    #[tokio::test]
    async fn execute_rejects_a_query_over_its_configured_input_cap() {
        let tool = ConsultTool::new(magi_all_ok(), false).with_max_query_bytes(50);
        let err = tool
            .execute(json!({"query": "x".repeat(51)}), &CancellationToken::new())
            .await
            .expect_err("51 bytes over a 50-byte cap must reject");
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        // Edge case (B13): exactly at the cap must proceed, not reject.
        let ok = tool
            .execute(json!({"query": "x".repeat(50)}), &CancellationToken::new())
            .await;
        assert!(
            ok.is_ok(),
            "exactly at the configured cap must be accepted, not rejected"
        );
    }

    /// UTF-8 boundary safety (project invariant): `head_chars` never splits a multi-byte
    /// character, whatever the cap.
    ///
    /// `"a🦀b"` is 1 + 4 + 1 = 6 bytes; the crab occupies bytes 1..5. A cap of 2, 3, or 4 lands
    /// INSIDE the crab's encoding — a naive `&s[..cap]` would panic (`byte index N is not a
    /// char boundary`) at those caps. `head_chars` must back off to the last COMPLETE character
    /// instead of splitting one.
    #[test]
    fn head_chars_never_splits_a_multibyte_character() {
        let s = "a🦀b";
        assert_eq!(head_chars(s, 0), "");
        assert_eq!(head_chars(s, 1), "a", "cap=1: only the ASCII byte fits");
        for cap in 2..=4 {
            assert_eq!(
                head_chars(s, cap),
                "a",
                "cap={cap}: the 4-byte crab cannot fit partially, so it is dropped \
                 whole rather than split mid-encoding"
            );
        }
        assert_eq!(head_chars(s, 5), "a🦀", "cap=5: the crab fits exactly");
        assert_eq!(head_chars(s, 6), "a🦀b", "cap=6: the whole string fits");
        assert_eq!(
            head_chars(s, 100),
            "a🦀b",
            "a cap above the string's length keeps it whole"
        );
    }

    /// `truncate_report` is a no-op below the cap: no mark, level `None` — never adds the mark
    /// to a report that was not actually cut.
    #[test]
    fn truncate_report_is_a_no_op_when_the_report_already_fits() {
        let report = "short report";
        let out = truncate_report(report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(out.level, TruncationLevel::None);
        assert_eq!(out.text, report);
        assert!(
            !out.text.contains(TRUNCATION_MARK),
            "a report that fits must not carry the mark"
        );
    }

    /// Builds a report whose kept region (verdict + a real first finding) fits comfortably
    /// inside `cap`, padded PAST `cap` with filler placed AFTER `findings_end` —
    /// `keep_verdict_and_first_finding` cuts at `findings_end`, so the filler is excluded from
    /// the kept slice regardless of its size; its only job is to push `report.len()` past `cap`
    /// so `truncate_report` actually recorts instead of returning the report whole.
    fn report_with_locatable_sections(cap: usize) -> String {
        let anchors = SECTION_ANCHORS.expect("this fixture assumes Structural is reachable");
        format!(
            "some estimation preamble\n\n{}\nthe consensus is GO\n\n{}\n- first finding\n- \
             second finding\n\n{}\n- do X\n\n## Dissenting Opinion\n{}",
            anchors.verdict_start,
            anchors.findings_start,
            anchors.findings_end,
            "z".repeat(cap),
        )
    }

    /// SC-A11e (`Structural`): the highest-guarantee level — the verdict AND at least the first
    /// finding survive the recort, plus the mark.
    #[test]
    fn structural_level_keeps_the_verdict_and_the_first_finding() {
        let anchors = SECTION_ANCHORS.expect("this test assumes Structural is reachable");
        let report = report_with_locatable_sections(TOOL_RESULT_CAP_BYTES);
        assert!(
            report.len() > TOOL_RESULT_CAP_BYTES,
            "test setup: the fixture must actually exceed the cap"
        );
        let out = truncate_report(&report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(out.level, TruncationLevel::Structural);
        assert!(
            out.text.contains(anchors.verdict_start),
            "guarantee (a): the verdict block"
        );
        assert!(
            out.text.contains(anchors.findings_start),
            "guarantee (b): at least the first finding"
        );
        assert!(
            out.text.contains(TRUNCATION_MARK),
            "guarantee (c): the truncation mark"
        );
        assert!(out.text.len() <= TOOL_RESULT_CAP_BYTES);
    }

    /// Builds a report where the verdict + first-finding region GENUINELY EXISTS — unlike
    /// [`report_with_only_the_anchor`] — but the text between `verdict_start` and
    /// `findings_start` alone is twice `cap`, so no `cap`'s worth of budget can ever reach
    /// `findings_start`. Distinguishes the BUDGET-driven cause of `Anchored` from the ABSENT-
    /// findings cause the sibling test below covers.
    fn report_with_findings_beyond_the_budget(cap: usize) -> String {
        let anchors =
            SECTION_ANCHORS.expect("this fixture assumes Structural/Anchored are reachable");
        // 2x cap guarantees the offset of `findings_start` from `verdict_start` alone exceeds
        // `cap - mark_overhead()`, however small `mark_overhead()` is.
        let long_verdict_body = "z".repeat(cap * 2);
        format!(
            "preamble\n\n{}\n{long_verdict_body}\n\n{}\n- first finding\n\n{}\n- do X",
            anchors.verdict_start, anchors.findings_start, anchors.findings_end,
        )
    }

    /// SC-A11e (`Anchored`, the BUDGET-driven cause — B13 edge case). Pins the `Structural` ->
    /// `Anchored` downgrade: `keep_verdict_and_first_finding` successfully locates and slices
    /// the verdict-through-`findings_end` region (`Some`), but `head_chars` cuts it to `budget`
    /// BEFORE reaching `findings_start`, so `kept_has_first_finding` correctly reports `false`
    /// on the RESULT — the level steps down rather than claiming guarantee (b) dishonestly.
    /// This is a DIFFERENT cause of `Anchored` than
    /// `anchored_level_keeps_only_the_verdict_when_there_were_no_findings` (findings genuinely
    /// absent from the whole report): here they exist, they are just too far from the verdict
    /// to survive the cut. Without the `kept_has_first_finding` check — i.e. if
    /// `truncate_report` returned `Structural` whenever `keep_verdict_and_first_finding`
    /// returns `Some`, regardless of content — this test goes red: it would observe
    /// `Structural` where it asserts `Anchored`.
    #[test]
    fn insufficient_budget_downgrades_structural_to_anchored_even_when_findings_exist() {
        let anchors = SECTION_ANCHORS.expect("this test assumes Structural/Anchored are reachable");
        let report = report_with_findings_beyond_the_budget(TOOL_RESULT_CAP_BYTES);
        assert!(
            report.contains(anchors.findings_start),
            "test setup: findings genuinely exist in the SOURCE report — this is the \
             budget-driven cause of Anchored, not the absent-findings cause"
        );
        assert!(
            report.len() > TOOL_RESULT_CAP_BYTES,
            "test setup: the fixture must actually exceed the cap"
        );
        let out = truncate_report(&report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(
            out.level,
            TruncationLevel::Anchored,
            "the kept slice was cut before reaching findings_start, so guarantee (b) \
             cannot be honestly claimed even though the SOURCE report has findings"
        );
        assert!(
            !out.text.contains(anchors.findings_start),
            "confirms the cut genuinely landed before the finding, proving THIS is the \
             budget-driven downgrade and not some other path to Anchored"
        );
        assert!(
            out.text.contains(anchors.verdict_start),
            "guarantee (a): the verdict block"
        );
        assert!(
            out.text.contains(TRUNCATION_MARK),
            "guarantee (c): the truncation mark"
        );
        assert!(out.text.len() <= TOOL_RESULT_CAP_BYTES);
    }

    /// A report where findings genuinely never rendered — magi-core omits the section entirely
    /// when there are none (`report_anchors::SectionAnchors:: findings_start`'s own rustdoc) —
    /// still exposes the verdict and the unconditional `findings_end` anchor. Padded past `cap`
    /// the same way as [`report_with_locatable_sections`].
    fn report_with_only_the_anchor(cap: usize) -> String {
        let anchors = SECTION_ANCHORS.expect("this fixture assumes Anchored is reachable");
        format!(
            "preamble\n\n{}\nno findings section in this report at all\n\n{}\n- do X\n\n## \
             Dissenting Opinion\n{}",
            anchors.verdict_start,
            anchors.findings_end,
            "z".repeat(cap),
        )
    }

    /// SC-A11e (`Anchored`)/SC-A11h/SC-A11i: the verdict and the mark survive, but the promise
    /// stops there — "no findings" (this fixture) and "could not locate" both degrade to this
    /// same level, which is the point: the consumer only needs to know it does NOT have
    /// guarantee (b), not which of the two caused it
    /// (`report_anchors::SectionAnchors::findings_start`'s own rustdoc). SC-A11i specifically:
    /// the sections could not be fully located (no findings), but the contractual anchor COULD
    /// — so the level lands on `Anchored`, one step down, never straight to `Bytes`.
    #[test]
    fn anchored_level_keeps_only_the_verdict_when_there_were_no_findings() {
        let anchors = SECTION_ANCHORS.expect("this test assumes Anchored is reachable");
        let report = report_with_only_the_anchor(TOOL_RESULT_CAP_BYTES);
        assert!(
            report.len() > TOOL_RESULT_CAP_BYTES,
            "test setup: the fixture must actually exceed the cap"
        );
        assert!(
            !report.contains(anchors.findings_start),
            "test setup: this fixture must genuinely have NO findings section, not \
             merely one that got cut off — 'no findings' and 'could not locate' are \
             two different things and this fixture is testing the former"
        );
        let out = truncate_report(&report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(out.level, TruncationLevel::Anchored);
        assert!(
            out.text.contains(anchors.verdict_start),
            "guarantee (a): the verdict block"
        );
        assert!(
            out.text.contains(TRUNCATION_MARK),
            "guarantee (c): the truncation mark"
        );
        assert!(out.text.len() <= TOOL_RESULT_CAP_BYTES);
    }

    /// A report with no recognizable anchor at all — neither the verdict box nor the
    /// unconditional `findings_end`/`## Recommended Actions` anywhere.
    fn report_with_no_recognizable_structure(cap: usize) -> String {
        format!(
            "garbled output with no recognizable anchors at all\n{}",
            "y".repeat(cap)
        )
    }

    /// SC-A11e (`Bytes`)/SC-A11h: the last-resort level promises ONLY the mark — no veredict,
    /// no finding — and declaring that honestly (rather than pretending to preserve the
    /// verdict, which was already the guarantee that failed to be located) is the entire point
    /// of this level existing.
    #[test]
    fn bytes_level_promises_only_the_mark() {
        let report = report_with_no_recognizable_structure(TOOL_RESULT_CAP_BYTES);
        assert!(
            report.len() > TOOL_RESULT_CAP_BYTES,
            "test setup: the fixture must actually exceed the cap"
        );
        let out = truncate_report(&report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(out.level, TruncationLevel::Bytes);
        assert!(
            out.text.contains(TRUNCATION_MARK),
            "the ONE thing this level promises, it must deliver"
        );
        assert!(out.text.len() <= TOOL_RESULT_CAP_BYTES);
    }

    /// A `cap` too small even for the mark itself is unreachable via `magi.toml`
    /// (`ConfigError::OutputCapTooSmall` rejects it at load — `src/config.rs`), but
    /// `truncate_report` is a pure function any caller can hand an arbitrary `cap`. Rather than
    /// emit a fragment that LIES about respecting the cap (text longer than `cap` because the
    /// mark alone does not fit), the report comes back WHOLE, honestly labeled `None` — see
    /// `truncate_report`'s own rustdoc for why this beats a broken fragment.
    #[test]
    fn a_cap_too_small_for_the_mark_returns_the_report_whole_instead_of_a_broken_fragment() {
        let report = "y".repeat(1_000);
        let tiny_cap = 1; // far below `mark_overhead()`
        let out = truncate_report(&report, tiny_cap);
        assert_eq!(out.level, TruncationLevel::None);
        assert_eq!(
            out.text, report,
            "an unenforceable cap must not truncate mid-report"
        );
    }

    // -----------------------------------------------------------------------
    // `truncate_report_with_preserved_prefix` — the TUI `[DEGRADED: ...]` banner fix: a prefix
    // reserved from `cap` BEFORE `truncate_report` runs on the report alone, so the prefix is
    // never the thing sacrificed to make room.
    // -----------------------------------------------------------------------

    /// An empty prefix degenerates to plain `truncate_report` — no spurious leading separator,
    /// no different budgeting. Proves the function is a strict superset of `truncate_report`,
    /// not a parallel reimplementation that could silently diverge from it (B3).
    #[test]
    fn preserved_prefix_with_an_empty_prefix_matches_plain_truncate_report() {
        let report = "y".repeat(1_000);
        for cap in [10, 50, TOOL_RESULT_CAP_BYTES] {
            let expected = truncate_report(&report, cap);
            let actual = truncate_report_with_preserved_prefix("", &report, cap);
            assert_eq!(actual.level, expected.level, "cap={cap}");
            assert_eq!(actual.text, expected.text, "cap={cap}");
        }
    }

    /// Happy path: a report too large for `cap` on its own still yields a combined `prefix +
    /// report` that (a) starts with the prefix intact and (b) never exceeds `cap` — the two
    /// properties the TUI banner fix exists to guarantee together.
    #[test]
    fn preserved_prefix_survives_truncation_and_the_combined_text_respects_the_cap() {
        let prefix = "[DEGRADED: fewer than 3 agents responded]";
        let report = report_with_no_recognizable_structure(TOOL_RESULT_CAP_BYTES);
        let cap = 300;
        assert!(
            prefix.len() + 2 + report.len() > cap,
            "test setup: the combined text must actually exceed cap"
        );

        let out = truncate_report_with_preserved_prefix(prefix, &report, cap);
        assert!(out.text.len() <= cap, "{}", out.text.len());
        assert!(
            out.text.starts_with(prefix),
            "the prefix must survive whole: {}",
            out.text
        );
        assert!(
            out.text.contains(TRUNCATION_MARK),
            "the report side must still carry its own truncation mark: {}",
            out.text
        );
    }

    /// A report that already fits under `cap` (after reserving room for the prefix) is returned
    /// intact, `TruncationLevel::None`, no mark — mirrors
    /// `truncate_report_is_a_op_when_the_report_already_fits`'s own guarantee, now with a
    /// prefix in the mix.
    #[test]
    fn preserved_prefix_is_a_no_op_when_the_combined_text_already_fits() {
        let prefix = "[DEGRADED: fewer than 3 agents responded]";
        let report = "short report";
        let out = truncate_report_with_preserved_prefix(prefix, report, TOOL_RESULT_CAP_BYTES);
        assert_eq!(out.level, TruncationLevel::None);
        assert_eq!(out.text, format!("{prefix}\n\n{report}"));
        assert!(!out.text.contains(TRUNCATION_MARK));
    }

    /// Boundary (B13): `cap` set EXACTLY to `prefix.len() + 2 + mark_overhead()` — the smallest
    /// cap that still leaves the report side enough room for its own mark. One byte below this
    /// is covered by the next test; this one proves the arithmetic does not off-by-one at the
    /// floor itself.
    #[test]
    fn preserved_prefix_survives_exactly_at_the_viable_floor() {
        let prefix = "[DEGRADED: fewer than 3 agents responded]";
        let report = report_with_no_recognizable_structure(TOOL_RESULT_CAP_BYTES);
        let cap = prefix.len() + 2 + mark_overhead();

        let out = truncate_report_with_preserved_prefix(prefix, &report, cap);
        assert!(out.text.len() <= cap, "{}", out.text.len());
        assert!(out.text.starts_with(prefix));
        assert!(out.text.contains(TRUNCATION_MARK));
    }

    /// Below the viable floor — `cap` too small to hold the prefix and still leave the report
    /// side room for its own mark — this degrades the SAME way `truncate_report` itself
    /// degrades when handed an unenforceable cap: the full, untruncated `prefix + "\n\n" +
    /// report`, honestly labeled `None` rather than a fragment that lies about respecting `cap`
    /// (documented on `truncate_report_with_preserved_prefix`'s own rustdoc).
    #[test]
    fn preserved_prefix_below_the_viable_floor_returns_everything_whole_instead_of_lying() {
        let prefix = "[DEGRADED: fewer than 3 agents responded]";
        let report = "y".repeat(1_000);
        let cap = prefix.len(); // no room even for the separator, let alone a mark

        let out = truncate_report_with_preserved_prefix(prefix, &report, cap);
        assert_eq!(out.level, TruncationLevel::None);
        assert_eq!(
            out.text,
            format!("{prefix}\n\n{report}"),
            "an unenforceable cap must not silently drop the prefix either"
        );
    }

    /// SC-A11d (the `ConsultTool::execute` route specifically): a report that exceeds a tiny
    /// effective output cap comes back bounded, with the level surfaced in the JSON — proving
    /// `execute` actually calls `truncate_report` with `self.output_cap`, not merely that the
    /// function exists in isolation.
    #[tokio::test]
    async fn execute_bounds_the_report_when_it_exceeds_the_configured_output_cap() {
        let cap = magi_rs::magi::mark_overhead() + 20;
        let tool = ConsultTool::new(magi_all_ok(), false).with_output_cap(cap);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("3 agents → success");
        let report = out["report"].as_str().expect("report string");
        assert!(
            report.len() <= cap,
            "the ToolResult text must respect the configured cap: {} > {cap}",
            report.len()
        );
        assert_ne!(
            out["report_truncated"], "none",
            "a report bigger than the cap must be marked truncated"
        );
    }
}
