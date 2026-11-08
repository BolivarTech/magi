// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! The false-positive gate (REQ-RV10): one `decide` question per consolidated finding.
//!
//! The gate fails open by design, the only exception to the fail-closed rule of the review: a
//! finding the gate could not judge passes as `unfiltered`, with its cause, and leaves one WARN
//! entry in the run log. A `filtered` finding is listed apart and never deleted, and a finding
//! the trio reported as `critical` is never filtered.
//!
//! The memory of a model that answered HTTP 400 lives in [`Gate`], one per review: the next
//! review asks that model again.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use magi_core::schema::Severity;
use serde_json::{json, Value};

use crate::decision::{
    Answer, AttemptOutcome, Cause, Decision, DecisionClient, DecisionError, DecisionRequest,
    ModelSpec, Question, Questions,
};

use super::config::DEFAULT_CALL_TIMEOUT_SECS;
use super::consolidate::ConsolidatedFinding;
use super::secrets::find_secret;

/// Name of the gate's only question.
pub const GATE_QUESTION_NAME: &str = "gate";
/// The question's instructions, verbatim from the calibration: changing them invalidates it.
pub const GATE_INSTRUCTIONS: &str = "Does this finding point to a real defect in the code shown?";
/// Description of the `real_defect` option, verbatim from the calibration.
pub const GATE_REAL_DEFECT: &str = "The finding describes a defect that exists in the code shown";
/// Description of the `false_positive` option, verbatim from the calibration.
pub const GATE_FALSE_POSITIVE: &str =
    "The finding is wrong about the code shown, or what it describes is not a defect";
/// Lines of code shown on each side of the finding's line, as in the calibration.
pub const GATE_CONTEXT_LINES: u32 = 40;
/// Run-log target of the gate's WARN entries.
pub const GATE_LOG_TARGET: &str = "magi_rs::review::gate";

/// Option name of a real defect.
const REAL_DEFECT: &str = "real_defect";
/// Option name of a false positive.
const FALSE_POSITIVE: &str = "false_positive";
/// HTTP status that rotates the chain and is remembered for the rest of the review.
const BAD_REQUEST: u16 = 400;

/// Cause of a finding without a line or without its code.
const CAUSE_NO_LOCATION: &str = "no location";
/// Cause of a state over `max_state_bytes`; the state is never truncated.
const CAUSE_STATE_TOO_LARGE: &str = "state too large";
/// Cause of a state in which the secret detector found a pattern; it is never sent.
const CAUSE_SECRET: &str = "secret";
/// Cause of a finding the review clock left no time to judge.
const CAUSE_TIMEOUT: &str = "timeout";
/// Cause of a chain where every model called failed in the transport class.
const CAUSE_TRANSPORT: &str = "transport";
/// Cause of an endpoint that refused the credentials.
const CAUSE_UNAUTHORIZED: &str = "unauthorized";
/// Cause of a chain that ended without a usable answer.
const CAUSE_CHAIN_EXHAUSTED: &str = "chain exhausted";
/// Confidence at or above which a `false_positive` answer filters even a `critical` finding
/// (RV10 amendment, user decision 2026-11-06, T3.7 criterion 2). Below it the finding stays in the
/// main list as `low_confidence` (`critical never filtered`).
///
/// A round number declared post hoc: without a bound the rule protected 6 `critical` false
/// positives and the gate removed 16/42 (38.1 %, criterion 2 asks 40 %). The risk to real
/// `critical` defects was not measured: neither `tune` nor `verify` holds a real `critical`
/// judged `false_positive`.
pub const CRITICAL_MIN_CONFIDENCE: f64 = 0.75;
/// Cause of a critical finding judged a false positive with low confidence: it stays in the main list.
const CAUSE_CRITICAL_NEVER_FILTERED: &str = "critical never filtered";
/// Cause of a finding whose asked pairs were all judged false positives while pairs past
/// [`MAX_GATE_QUESTIONS_PER_FINDING`] went unasked: what was never judged is never filtered.
const CAUSE_NOT_EVERY_CLAIM_JUDGED: &str = "not every claim judged";
/// Cause of a finding judged a false positive on a line whose number names both a line its hunk
/// deleted and a live line: the finding does not say which side it meant, so a judgment made on
/// the live side may be about other code, and it stays in the main list.
const CAUSE_AMBIGUOUS_DIFF_SIDE: &str = "ambiguous diff side";

/// The most distinct (title, detail) pairs of one finding put to the chain: one per mage of the
/// trio. Each pair is a walk of the chain, so a finding merged from many rewordings (several
/// batches, several mages) would otherwise spend the review clock on itself; pairs past the cap
/// are not asked. A finding is `filtered` only when it has no unasked pair and every pair is;
/// with pairs left unasked it stays `low_confidence`.
///
/// RV10's "once per consolidated finding" is read as one gate verdict per finding, not one
/// call: a finding merged from different claims (RV12 keeps every title and detail) asks about
/// each, because filtering on the answer to one claim would drop the others unjudged. The walk
/// stops at the first answer that is not a confident false positive: a `kept` with later claims
/// unasked stays `kept` (it is in the main list, so nothing was lost) and names them in
/// [`GateVerdict::causes`] (`not every claim judged`, the rule settled in S6 loop 6). The cost is
/// bounded by this cap and by the review clock, which every call of every walk is charged to.
// MAGI FIX (S6 loop 7, C1-c1): the per-claim walk is a declared reading of RV10, not a defect.
pub const MAX_GATE_QUESTIONS_PER_FINDING: usize = 3;

/// What the gate made of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateState {
    /// A model judged it a real defect.
    Kept,
    /// A model judged it a false positive with enough confidence; listed apart, never deleted.
    Filtered,
    /// No model reached its threshold, a critical finding was judged a false positive, a finding
    /// on an ambiguous diff line was judged one ([`Gate::judge_ambiguous_side`]), or every pair
    /// asked was judged one while pairs past the question cap went unasked.
    LowConfidence,
    /// The gate could not judge it; it passes with its cause.
    Unfiltered,
}

/// The gate's verdict on one finding.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GateVerdict {
    /// The state.
    pub state: GateState,
    /// Why, for `unfiltered` and `low_confidence`: the first of [`Self::causes`]. Always `None`
    /// for `kept` and `filtered`, even when [`Self::causes`] names unasked claims.
    pub cause: Option<&'static str>,
    /// Every cause that applies, the state's own first: a finding with (title, detail) pairs
    /// that were never put to the chain also carries `not every claim judged`.
    pub causes: Vec<&'static str>,
    /// The model whose answer decided the state (or, for `low_confidence`, answered last).
    pub model: Option<String>,
    /// The option that answer leaned to.
    pub lean: Option<String>,
    /// Magi's confidence of that answer.
    pub confidence: Option<f64>,
    /// Models skipped because they answered HTTP 400 in this review, in chain order. Each entry
    /// is a skip with cause `rejected` (RV10, RV-S20), the only cause that skips a model, so the
    /// field carries the cause by its meaning; the summary counts them under `rejected_skips`.
    pub skipped: Vec<String>,
    /// The HTTP status of the provider refusal that ended the chain (413 or 422), so a
    /// `chain exhausted` from a refused request is told apart from one where every model
    /// answered under its threshold; absent otherwise (and from the JSON form).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
}

/// The `state` sent to `decide` for a finding's first (title, detail) pair.
///
/// # Arguments
/// * `finding` - the consolidated finding.
/// * `code` - the whole file the finding points into.
///
/// # Returns
/// `{"finding": {file, line, severity, title, detail}, "code": "<numbered lines>"}`; see
/// [`gate_state_of`].
pub fn gate_state(finding: &ConsolidatedFinding, code: &str) -> Value {
    gate_state_of(finding, 0, code)
}

/// The `state` sent to `decide` for the `i`-th distinct (title, detail) pair of a finding.
///
/// Each pair goes in the calibration's single-finding shape: the file, the line, the severity the
/// trio reported (before the agreement weight), one title, one detail, and the lines of `code`
/// from `line - 40` to `line + 40`, each as `"<n>: <text>\n"`. Line ends (`\n` or `\r\n`) are
/// not part of a line, so the numbers match an editor's. An `i` past the last pair takes the last
/// one. O(l) for the l lines up to the window's end, plus the finding's text.
///
/// # Arguments
/// * `finding` - the consolidated finding.
/// * `i` - index among its distinct (title, detail) pairs, in order of appearance.
/// * `code` - the whole file the finding points into.
///
/// # Returns
/// The state as JSON.
///
/// # Examples
/// ```
/// use magi_core::schema::{AgentName, Category, Severity};
/// use magi_rs::review::consolidate::ConsolidatedFinding;
/// use magi_rs::review::gate::gate_state_of;
///
/// let f = ConsolidatedFinding {
///     file: Some("a.rs".into()), line: Some(2), severity: Severity::Info,
///     reported_severity: Severity::Info, titles: vec!["t".into()], details: vec!["d".into()],
///     categories: vec![Category::Other], sources: vec![AgentName::Melchior], marks: vec![],
///     batches: vec![0],
/// };
/// let state = gate_state_of(&f, 0, "fn a() {}\nfn b() {}\n");
/// assert_eq!(state["code"], "1: fn a() {}\n2: fn b() {}\n");
/// assert_eq!(state["finding"]["severity"], "info");
/// ```
pub fn gate_state_of(finding: &ConsolidatedFinding, i: usize, code: &str) -> Value {
    let pairs = constituents(finding);
    let (title, detail) = pairs
        .get(i)
        .or_else(|| pairs.last())
        .copied()
        .unwrap_or_default();
    json!({
        "finding": {
            "file": finding.file,
            "line": finding.line,
            "severity": finding.reported_severity,
            "title": title,
            "detail": detail,
        },
        "code": finding.line.map(|line| numbered_window(code, line)).unwrap_or_default(),
    })
}

/// The gate's question set: one `choice` between a real defect and a false positive.
///
/// # Returns
/// The question of the calibration, verbatim.
pub fn gate_questions() -> Questions {
    let criteria = [
        (REAL_DEFECT.to_string(), GATE_REAL_DEFECT.to_string()),
        (FALSE_POSITIVE.to_string(), GATE_FALSE_POSITIVE.to_string()),
    ]
    .into_iter()
    .collect();
    [(
        GATE_QUESTION_NAME.to_string(),
        Question::Choice {
            instructions: Some(GATE_INSTRUCTIONS.to_string()),
            criteria: Some(criteria),
        },
    )]
    .into_iter()
    .collect()
}

/// The distinct (title, detail) pairs of a finding, in order of appearance. O(k^2) for k merged
/// findings, a handful per location.
fn constituents(finding: &ConsolidatedFinding) -> Vec<(&str, &str)> {
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    for pair in finding
        .titles
        .iter()
        .map(String::as_str)
        .zip(finding.details.iter().map(String::as_str))
    {
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    if pairs.is_empty() {
        pairs.push(("", ""));
    }
    pairs
}

/// The lines `line +/- GATE_CONTEXT_LINES` of `code`, numbered from 1.
fn numbered_window(code: &str, line: u32) -> String {
    let first = line.saturating_sub(GATE_CONTEXT_LINES).max(1);
    let last = line.saturating_add(GATE_CONTEXT_LINES);
    let mut out = String::new();
    for (n, text) in (1u32..).zip(code.lines()) {
        if n > last {
            break;
        }
        if n >= first {
            // Writing into a `String` cannot fail.
            let _ = writeln!(out, "{n}: {text}");
        }
    }
    out
}

/// Whether `code` has a line numbered `line` (from 1). O(bytes up to that line).
fn holds_line(code: &str, line: u32) -> bool {
    usize::try_from(line)
        .ok()
        .and_then(|l| l.checked_sub(1))
        .is_some_and(|i| code.lines().nth(i).is_some())
}

/// Whether the secret detector finds a pattern in any string of `value`. O(bytes).
fn carries_secret(value: &Value) -> bool {
    match value {
        Value::String(text) => find_secret(text).is_some(),
        Value::Array(items) => items.iter().any(carries_secret),
        Value::Object(fields) => fields.values().any(carries_secret),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// The gate's single answer in `decision`, when it is one of the two options.
fn gate_answer(decision: &Decision) -> Option<(&'static str, f64)> {
    match decision.answers.get(GATE_QUESTION_NAME)? {
        Answer::Choice {
            choice, confidence, ..
        } => [REAL_DEFECT, FALSE_POSITIVE]
            .into_iter()
            .find(|option| option == choice)
            .map(|option| (option, *confidence)),
        Answer::Noul { .. } | Answer::Score { .. } => None,
    }
}

/// The cause of the last attempt an error recorded, if any.
fn last_cause(error: &DecisionError) -> Option<Cause> {
    match &error.trace()?.attempts.last()?.outcome {
        AttemptOutcome::Failed { cause, .. } => Some(*cause),
        AttemptOutcome::Answered | AttemptOutcome::LowConfidence { .. } => None,
    }
}

/// Records `model` as skipped by a 400 once, in chain order. O(m) for m models.
fn note_skip(skipped: &mut Vec<String>, model: &str) {
    if !skipped.iter().any(|s| s == model) {
        skipped.push(model.to_string());
    }
}

/// The HTTP timeout of one gate call, and whether the review clock set it.
///
/// # Arguments
/// * `left` - What is left of the review clock, if there is one.
/// * `call_timeout` - The per-call timeout (`[code_review].timeout_secs`).
///
/// # Returns
/// `(cap, cut_by_clock)`: the smaller of the two, and whether a timeout of that call is the
/// review clock running out rather than a slow model.
fn call_cap(left: Option<Duration>, call_timeout: Duration) -> (Duration, bool) {
    match left {
        // MAGI FIX (S6 loop 3): equality is the review clock too, so its timeout is `timeout`.
        Some(left) if left <= call_timeout => (left, true),
        _ => (call_timeout, false),
    }
}

/// What one question (one pair of a finding) came to.
struct Outcome {
    state: GateState,
    cause: Option<&'static str>,
    model: Option<String>,
    lean: Option<String>,
    confidence: Option<f64>,
    /// The provider's HTTP status, for the WARN entry of a 413/422.
    status: Option<u16>,
    /// Whether `model`'s answer reached its threshold and decided the state.
    decided: bool,
    /// Whether the chain was asked about this finding and some of its (title, detail) pairs
    /// were never put to it.
    unasked: bool,
    /// The cause that kept one of its pairs from being sent (`state too large`, `secret`) while
    /// another pair was asked.
    withheld: Option<&'static str>,
}

impl Outcome {
    /// An `unfiltered` outcome with its cause.
    fn unfiltered(cause: &'static str) -> Self {
        Self {
            state: GateState::Unfiltered,
            cause: Some(cause),
            model: None,
            lean: None,
            confidence: None,
            status: None,
            decided: false,
            unasked: false,
            withheld: None,
        }
    }

    /// An outcome decided (or leaned) by `model`'s answer.
    fn answered(
        state: GateState,
        cause: Option<&'static str>,
        model: &str,
        lean: &str,
        confidence: f64,
    ) -> Self {
        Self {
            state,
            cause,
            model: Some(model.to_string()),
            lean: Some(lean.to_string()),
            confidence: Some(confidence),
            status: None,
            decided: false,
            unasked: false,
            withheld: None,
        }
    }
}

/// The gate of one review.
///
/// Build one per review: it remembers, for the rest of that review, the models that answered
/// HTTP 400 and the cause that stopped it (`unauthorized`, or `timeout` when the review clock ran
/// out). The thresholds are applied here, so the client must be built with no `min_confidence`
/// in its chain; a model without one decides with any confidence, as in `[decision]`.
///
/// That rule is not prose only: the one production client, `gate_client` in
/// `code_review_runner.rs`, clears every `min_confidence` of the configured chain, and
/// `an_answer_under_the_threshold_reaches_the_gate_as_low_confidence` there drives a
/// below-threshold `false_positive` through that client into this gate and asserts
/// `low_confidence`, not `filtered`.
///
/// # Examples
/// ```no_run
/// # async fn demo(client: &dyn magi_rs::decision::DecisionClient,
/// #               finding: &magi_rs::review::consolidate::ConsolidatedFinding) {
/// use magi_rs::decision::ModelSpec;
/// use magi_rs::review::gate::Gate;
///
/// let chain = vec![ModelSpec { model: "clef:cloud".into(), min_confidence: Some(0.40) }];
/// let mut gate = Gate::new(client, chain, 61_000)
///     .with_call_timeout(std::time::Duration::from_secs(120));
/// let verdict = gate.judge(finding, Some("fn main() {}\n"), None).await;
/// println!("{:?} {:?}", verdict.state, verdict.cause);
/// # }
/// ```
// MAGI FIX (S6 loop 10, C1-m2): the threshold-free client is pinned by a wiring test, named above.
pub struct Gate<'a> {
    client: &'a dyn DecisionClient,
    /// Shared, so a walk holds it without cloning every model per question.
    chain: Arc<[ModelSpec]>,
    max_state_bytes: usize,
    call_timeout: Duration,
    rejected: BTreeSet<String>,
    stopped: Option<&'static str>,
    per_model: BTreeMap<String, usize>,
    questions: usize,
}

impl<'a> Gate<'a> {
    /// A gate over `chain`, with no model rejected yet.
    ///
    /// # Arguments
    /// * `client` - the decision client (its chain without thresholds).
    /// * `chain` - the models, in order, each with its threshold.
    /// * `max_state_bytes` - the largest serialized state sent, in bytes.
    ///
    /// # Returns
    /// The gate. Until [`Self::with_call_timeout`] sets another, each call's timeout is
    /// [`DEFAULT_CALL_TIMEOUT_SECS`]: a call always carries a timeout of its own, so a slow
    /// model is never read as the review clock running out.
    pub fn new(
        client: &'a dyn DecisionClient,
        chain: Vec<ModelSpec>,
        max_state_bytes: usize,
    ) -> Self {
        Self {
            client,
            chain: chain.into(),
            max_state_bytes,
            call_timeout: Duration::from_secs(DEFAULT_CALL_TIMEOUT_SECS),
            rejected: BTreeSet::new(),
            stopped: None,
            per_model: BTreeMap::new(),
            questions: 0,
        }
    }

    /// Sets the per-call HTTP timeout (`[code_review].timeout_secs`).
    ///
    /// # Arguments
    /// * `timeout` - timeout of each call; with a review clock, a call gets the smaller of this
    ///   and what is left.
    ///
    /// # Returns
    /// The gate.
    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = timeout;
        self
    }

    /// Judges one finding, never failing: what cannot be judged passes as `unfiltered`.
    ///
    /// In order: no file, no line, no code, or code without that line (`no location`); a cause that already stopped the gate; a
    /// spent review clock (`timeout`). Each distinct (title, detail) pair's state is then checked
    /// on its own: one over `max_state_bytes` (`state too large`, never truncated) or carrying a
    /// secret (`secret`, never sent) is withheld, and a finding whose every pair is withheld is
    /// `unfiltered` with the first such cause. A withheld pair beside asked ones is named in
    /// [`GateVerdict::causes`] with `not every claim judged`, and the finding is never `filtered`.
    /// Then each distinct (title,
    /// detail) pair, up to [`MAX_GATE_QUESTIONS_PER_FINDING`], is asked once, walking the chain: a confident `false_positive` filters it
    /// (a `critical` stays `low_confidence` instead), a confident `real_defect` keeps it, an
    /// answer under the threshold passes to the next model, a 400 rotates and is remembered, a
    /// 413/422 ends the chain, a 401/403 stops the gate, and a call cut by the review clock stops
    /// it as `timeout`. A merged finding is `filtered` only when every pair was asked and judged a
    /// false positive; with pairs past the cap unasked it is `low_confidence` (`not every claim
    /// judged`). A walk that ends early (a pair kept, under every threshold, a critical, a stop)
    /// leaves the later pairs unasked: once the chain was asked about the finding, any unasked
    /// pair adds `not every claim judged` to [`GateVerdict::causes`], after the state's own cause.
    /// Every `unfiltered` writes one WARN entry. At most one call per model per pair.
    ///
    /// # Arguments
    /// * `finding` - the consolidated finding.
    /// * `code` - the file it points into, if it was read.
    /// * `remaining` - what is left of the review clock, if there is one; a deadline is taken at
    ///   entry, so each call of the chain gets only what is still left.
    ///
    /// # Returns
    /// The verdict.
    pub async fn judge(
        &mut self,
        finding: &ConsolidatedFinding,
        code: Option<&str>,
        remaining: Option<Duration>,
    ) -> GateVerdict {
        self.judge_with(finding, code, remaining, None).await
    }

    /// Judges a finding as [`Self::judge`] does, except that it is never `filtered`: the code is
    /// the live side of a diff line whose number also names a line its hunk deleted, and the
    /// finding does not say which side it meant (RV21). A confident `false_positive` is
    /// `low_confidence` with cause `ambiguous diff side` (a `critical` keeps `critical never
    /// filtered` as its cause, with `ambiguous diff side` after it in `causes`); every other answer is read as by [`Self::judge`], and the deciding model is
    /// counted the same way.
    ///
    /// The caller reads `code` from the reviewed revision (RV21) and routes an ambiguous line
    /// here: `run_review` does both, pinned by the pipeline tests
    /// `a_line_the_hunk_deleted_whose_number_is_live_is_never_filtered`,
    /// `a_finding_on_a_deleted_file_reaches_the_gate_with_old_side_code` and
    /// `a_renamed_and_edited_file_reads_its_old_side_under_its_old_path`.
    ///
    /// # Arguments
    /// * `finding` - the consolidated finding.
    /// * `code` - the live side of the file it points into.
    /// * `remaining` - what is left of the review clock, if there is one.
    ///
    /// # Returns
    /// The verdict.
    // MAGI FIX (S6 loop 10, C1-m2): the RV21 wiring of the gate's code is named, not assumed.
    // MAGI FIX (S6 loop 7, D-b3/D-c3): what may have been judged on the wrong side is never
    // filtered, as what was never judged is never filtered.
    pub async fn judge_ambiguous_side(
        &mut self,
        finding: &ConsolidatedFinding,
        code: &str,
        remaining: Option<Duration>,
    ) -> GateVerdict {
        self.judge_with(
            finding,
            Some(code),
            remaining,
            Some(CAUSE_AMBIGUOUS_DIFF_SIDE),
        )
        .await
    }

    /// [`Self::judge`], with `guard` the cause a confident `false_positive` takes instead of
    /// `filtered` (`None`: it filters, unless the finding is critical).
    async fn judge_with(
        &mut self,
        finding: &ConsolidatedFinding,
        code: Option<&str>,
        remaining: Option<Duration>,
        guard: Option<&'static str>,
    ) -> GateVerdict {
        let deadline = remaining.and_then(|r| Instant::now().checked_add(r));
        let mut skipped = Vec::new();
        let outcome = self
            .decide(finding, code, deadline, guard, &mut skipped)
            .await;
        // MAGI FIX (S6 loop 3): unasked claims are named beside the state's own cause, never
        // replaced by it (`critical never filtered`) nor left out (a lean that ended the walk).
        let mut causes: Vec<&'static str> = outcome.cause.into_iter().collect();
        // MAGI FIX (S6 loop 9, C1-c2): the critical rule sets the state's cause, but the guard
        // applied too (it is only ever asked for by a confident `false_positive`): both are named.
        if let Some(guard) = guard.filter(|_| outcome.cause == Some(CAUSE_CRITICAL_NEVER_FILTERED))
        {
            causes.push(guard);
        }
        causes.extend(outcome.withheld);
        if outcome.unasked && !causes.contains(&CAUSE_NOT_EVERY_CLAIM_JUDGED) {
            causes.push(CAUSE_NOT_EVERY_CLAIM_JUDGED);
        }
        if outcome.state == GateState::Unfiltered {
            tracing::event!(
                target: GATE_LOG_TARGET,
                tracing::Level::WARN,
                cause = outcome.cause.unwrap_or(CAUSE_CHAIN_EXHAUSTED),
                status = outcome.status,
                "finding passed the gate unjudged"
            );
        }
        // MAGI FIX (S6 loop 6): `kept` has no cause of its own; its unasked claims stay in `causes`.
        let cause = match outcome.state {
            GateState::Kept => None,
            _ => causes.first().copied(),
        };
        GateVerdict {
            state: outcome.state,
            cause,
            causes,
            model: outcome.model,
            lean: outcome.lean,
            confidence: outcome.confidence,
            skipped,
            // MAGI FIX (S6 loop 5): the refusal's status reaches the report, not only the log.
            status: outcome.status,
        }
    }

    /// Findings counted per deciding model.
    ///
    /// A finding counts once, for the model whose answer decided its final state (`kept`,
    /// `filtered`, or `low_confidence` for a critical), however many of its (title, detail)
    /// pairs were asked. A finding whose chain ended with every model under its threshold, that
    /// passed `unfiltered`, or that is `low_confidence` (`not every claim judged`) because pairs
    /// past [`MAX_GATE_QUESTIONS_PER_FINDING`] went unasked, counts for no model: in that last
    /// case the question cap, not an answer, set its final state. [`Self::questions`] counts the questions.
    ///
    /// # Returns
    /// Model name to the number of findings whose state its answer decided.
    pub fn per_model(&self) -> &BTreeMap<String, usize> {
        &self.per_model
    }

    /// Questions asked in this review: one per (title, detail) pair put to the chain, at most
    /// [`MAX_GATE_QUESTIONS_PER_FINDING`] per finding.
    ///
    /// # Returns
    /// The count.
    pub fn questions(&self) -> usize {
        self.questions
    }

    /// The cause that stopped the gate for the rest of the review, if any.
    ///
    /// # Returns
    /// `unauthorized` or `timeout`, the first that happened, or `None`.
    pub fn stopped(&self) -> Option<&'static str> {
        self.stopped
    }

    /// Records the cause that stops the gate; only the first one is kept.
    fn stop(&mut self, cause: &'static str) -> Outcome {
        let cause = *self.stopped.get_or_insert(cause);
        Outcome::unfiltered(cause)
    }

    /// The outcome of a whole finding: its checks, then each pair until one is not `filtered`.
    async fn decide(
        &mut self,
        finding: &ConsolidatedFinding,
        code: Option<&str>,
        deadline: Option<Instant>,
        guard: Option<&'static str>,
        skipped: &mut Vec<String>,
    ) -> Outcome {
        // A finding with nothing to show is `no location` whatever stopped the gate: no call is
        // made either way, and the cause says what the finding lacks.
        // MAGI FIX (S6 loop 9, C1-c3): a location is a file and a line; a line alone is none.
        let (Some(_), Some(line), Some(code)) = (&finding.file, finding.line, code) else {
            return Outcome::unfiltered(CAUSE_NO_LOCATION);
        };
        // MAGI FIX (S6 loop 7, C1-c3): code that does not reach the finding's line (empty, or
        // shorter than it) would send an excerpt without that line: nothing to show either.
        if !holds_line(code, line) {
            return Outcome::unfiltered(CAUSE_NO_LOCATION);
        }
        if let Some(cause) = self.stopped {
            return Outcome::unfiltered(cause);
        }
        if deadline.is_some_and(|d| d <= Instant::now()) {
            return self.stop(CAUSE_TIMEOUT);
        }
        let pairs = constituents(finding).len();
        let asked = pairs.min(MAX_GATE_QUESTIONS_PER_FINDING);
        // MAGI FIX (S6 loop 10, C1-c2): the checks run per pair. A pair too large or carrying a
        // secret is withheld (never sent, never truncated) and the others are still asked; only
        // a finding whose every pair is withheld leaves `unfiltered` with that pair's cause.
        let mut withheld = None;
        let states: Vec<Value> = (0..asked)
            .map(|i| gate_state_of(finding, i, code))
            .filter(|state| match self.withheld_cause(state) {
                Some(cause) => {
                    withheld.get_or_insert(cause);
                    false
                }
                None => true,
            })
            .collect();
        if let (true, Some(cause)) = (states.is_empty(), withheld) {
            return Outcome::unfiltered(cause);
        }
        let critical = finding.reported_severity == Severity::Critical;
        let mut outcome = Outcome::unfiltered(CAUSE_CHAIN_EXHAUSTED);
        let mut put = 0;
        for state in states {
            self.questions += 1;
            put += 1;
            outcome = self.ask(state, guard, critical, deadline, skipped).await;
            if outcome.state != GateState::Filtered {
                break;
            }
        }
        outcome.unasked = put < pairs;
        outcome.withheld = withheld;
        if outcome.state == GateState::Filtered && outcome.unasked {
            // The cap, not the model's answer, sets this state: it counts for no model.
            outcome.state = GateState::LowConfidence;
            outcome.cause = Some(CAUSE_NOT_EVERY_CLAIM_JUDGED);
            outcome.decided = false;
        }
        if outcome.decided {
            if let Some(model) = &outcome.model {
                *self.per_model.entry(model.clone()).or_default() += 1;
            }
        }
        outcome
    }

    /// Why `state` cannot be sent, if it cannot: over `max_state_bytes` (`state too large`,
    /// checked first) or carrying a secret pattern (`secret`). O(bytes of the state).
    fn withheld_cause(&self, state: &Value) -> Option<&'static str> {
        if state.to_string().len() > self.max_state_bytes {
            Some(CAUSE_STATE_TOO_LARGE)
        } else if carries_secret(state) {
            Some(CAUSE_SECRET)
        } else {
            None
        }
    }

    /// Asks one state down the chain. A confident `false_positive` is `low_confidence` when the
    /// finding is `critical` and the confidence is under [`CRITICAL_MIN_CONFIDENCE`] (cause
    /// `critical never filtered`), or when `guard` names another protection (that cause);
    /// otherwise it is `filtered`.
    async fn ask(
        &mut self,
        state: Value,
        guard: Option<&'static str>,
        critical: bool,
        deadline: Option<Instant>,
        skipped: &mut Vec<String>,
    ) -> Outcome {
        let mut lean: Option<Outcome> = None;
        let mut called = false;
        let mut all_transport = true;
        // MAGI FIX (S6 loop 4): a reference count, not a copy of the chain, per question.
        let chain = Arc::clone(&self.chain);
        for spec in chain.iter() {
            if self.rejected.contains(&spec.model) {
                note_skip(skipped, &spec.model);
                continue;
            }
            let left = match deadline {
                Some(d) => {
                    let left = d.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return self.stop(CAUSE_TIMEOUT);
                    }
                    Some(left)
                }
                None => None,
            };
            let (cap, cut_by_clock) = call_cap(left, self.call_timeout);
            // MAGI FIX (S6 loop 10, C1-m3): rebuilt per model on purpose, not a hot path. Each
            // build clones a state already capped at `max_state_bytes` (61 000 by default), at
            // most chain length x MAX_GATE_QUESTIONS_PER_FINDING times per finding (9 with the
            // default chain), and the +/-40-line window is built once per pair; both are O(bytes)
            // microseconds against a model call of seconds (p95 2.9 s, E7).
            // `new` only fails on a malformed state or question set; both are built here.
            let Ok(request) = DecisionRequest::new(state.clone(), gate_questions()) else {
                return Outcome::unfiltered(CAUSE_CHAIN_EXHAUSTED);
            };
            let request = request.with_model(spec.model.clone()).with_timeout(cap);
            called = true;
            match self.client.decide(&request).await {
                Ok(decision) => {
                    all_transport = false;
                    let Some((choice, confidence)) = gate_answer(&decision) else {
                        continue;
                    };
                    if spec.min_confidence.is_some_and(|min| confidence < min) {
                        lean = Some(Outcome::answered(
                            GateState::LowConfidence,
                            None,
                            &spec.model,
                            choice,
                            confidence,
                        ));
                        continue;
                    }
                    // T3.7-fix1: the critical rule yields to a confident answer; `guard` does not.
                    let protected = if critical && confidence < CRITICAL_MIN_CONFIDENCE {
                        Some(CAUSE_CRITICAL_NEVER_FILTERED)
                    } else {
                        guard
                    };
                    let (state, cause) = match (choice, protected) {
                        (FALSE_POSITIVE, Some(cause)) => (GateState::LowConfidence, Some(cause)),
                        (FALSE_POSITIVE, None) => (GateState::Filtered, None),
                        _ => (GateState::Kept, None),
                    };
                    return Outcome {
                        decided: true,
                        ..Outcome::answered(state, cause, &spec.model, choice, confidence)
                    };
                }
                Err(DecisionError::Rejected { status, .. }) if status == BAD_REQUEST => {
                    // A 400 is this model's refusal, classified by status and never by its text:
                    // the model is skipped for the rest of this review.
                    all_transport = false;
                    self.rejected.insert(spec.model.clone());
                    note_skip(skipped, &spec.model);
                }
                Err(DecisionError::Rejected { status, .. }) => {
                    // 413/422: the request itself is at fault; every model would repeat it.
                    // MAGI FIX (S6 loop 8, C1-c1): refuted, not changed. Only a 400 or a 413/422
                    // reaches this arm: for a request pinned to one model (`with_model`), the
                    // client ends a 404 or a 5xx as `ModelUnavailable`, which the last arm rotates
                    // past without the 400 memory. Pinned end to end over the production client by
                    // `a_404_or_5xx_head_rotates_to_the_fallback_and_is_never_remembered_as_a_400`.
                    return Outcome {
                        status: Some(status),
                        ..Outcome::unfiltered(CAUSE_CHAIN_EXHAUSTED)
                    };
                }
                Err(DecisionError::Unauthorized { .. }) => {
                    return self.stop(CAUSE_UNAUTHORIZED);
                }
                Err(error) => {
                    let cause = last_cause(&error);
                    if cause == Some(Cause::Timeout) && cut_by_clock {
                        return self.stop(CAUSE_TIMEOUT);
                    }
                    if !cause.is_some_and(Cause::is_transport_class) {
                        all_transport = false;
                    }
                }
            }
        }
        match lean {
            Some(lean) => lean,
            None if called && all_transport => Outcome::unfiltered(CAUSE_TRANSPORT),
            None => Outcome::unfiltered(CAUSE_CHAIN_EXHAUSTED),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::client::MockDecisionClient;
    use crate::decision::{Answer, Attempt, AttemptOutcome, Cause, Decision, DecisionError, Trace};
    use magi_core::schema::{AgentName, Category, Severity};

    fn finding(sev: Severity, line: Option<u32>) -> ConsolidatedFinding {
        ConsolidatedFinding {
            file: Some("a.rs".into()),
            line,
            severity: sev,
            reported_severity: sev,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior, AgentName::Caspar],
            marks: vec![],
            batches: vec![0],
        }
    }
    fn chain() -> Vec<ModelSpec> {
        vec![
            ModelSpec {
                model: "clef:cloud".into(),
                min_confidence: Some(0.40),
            },
            ModelSpec {
                model: "clef-flash".into(),
                min_confidence: Some(0.32),
            },
            ModelSpec {
                model: "nimble".into(),
                min_confidence: Some(0.80),
            },
        ]
    }
    fn answer(model: &str, choice: &str, confidence: f64) -> Result<Decision, DecisionError> {
        let other = if choice == "real_defect" {
            "false_positive"
        } else {
            "real_defect"
        };
        let p = (1.0 + confidence) / 2.0; // Magi's confidence for two options: 2·p − 1
        let answers = [(
            GATE_QUESTION_NAME.to_string(),
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
    /// What T2.2 returns for a single explicit model that answered 400.
    fn bad_request(model: &str) -> Result<Decision, DecisionError> {
        Err(DecisionError::Rejected {
            status: 400,
            provider_message: format!("{model}: requires a local model"),
            trace: Trace::default(),
        })
    }

    #[tokio::test]
    async fn a_merged_finding_is_filtered_only_when_every_constituent_is() {
        // [CP2 MS2 cierre (sin MAGI)] RV-S31 meets the gate: each constituent goes in the calibration's single-finding shape.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["noise".into(), "real".into()];
        f.details = vec!["dn".into(), "dr".into()];
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(2).returning(|r| {
            let fp = r.state()["finding"]["title"].as_str() == Some("noise");
            answer(
                r.model().unwrap_or(""),
                if fp { "false_positive" } else { "real_defect" },
                0.9,
            )
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        assert_eq!(
            g.judge(&f, Some("x\nx\nx\n"), None).await.state,
            GateState::Kept,
            "one real defect keeps it"
        );
        let mut c2 = MockDecisionClient::new();
        c2.expect_decide()
            .times(2)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g2 = Gate::new(&c2, chain(), 61_000);
        assert_eq!(
            g2.judge(&f, Some("x\nx\nx\n"), None).await.state,
            GateState::Filtered
        );
    }

    #[tokio::test]
    async fn a_merged_finding_asks_at_most_the_question_cap() {
        // Each distinct (title, detail) pair is a walk of the chain; a finding merged from many
        // rewordings must not spend the review clock on all of them.
        let mut f = finding(Severity::Warning, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 2;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(MAX_GATE_QUESTIONS_PER_FINDING)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(g.questions(), MAX_GATE_QUESTIONS_PER_FINDING);
    }

    #[tokio::test]
    async fn a_finding_with_unasked_pairs_is_never_filtered() {
        // Controller ruling (MS2 §6): pairs past the cap were never judged, so a finding whose
        // asked pairs are all false positives stays in the main list, like a critical does.
        let mut f = finding(Severity::Warning, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 1;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(MAX_GATE_QUESTIONS_PER_FINDING)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("not every claim judged"))
        );
        assert_eq!(v.lean.as_deref(), Some("false_positive"));
    }

    #[tokio::test]
    async fn a_finding_demoted_for_unasked_pairs_counts_for_no_model() {
        // [MAGI MS2 S6 loop 2, C] the model answered `filtered`, but the final state comes from
        // the question cap, so no model decided it.
        let mut f = finding(Severity::Warning, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 1;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(MAX_GATE_QUESTIONS_PER_FINDING)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(v.cause, Some("not every claim judged"));
        assert!(g.per_model().is_empty(), "{:?}", g.per_model());
    }

    #[tokio::test]
    async fn a_merged_finding_kept_on_its_first_claim_stays_kept_and_names_its_unasked_claims() {
        // [MAGI S6 loop 6, C1] a confident `real_defect` on the first pair ends the walk. One
        // claim confirmed as a real defect keeps the finding (README: the walk stops at the first
        // pair not judged a false positive), credited to the model that answered; the unasked
        // claims are named in `causes`, beside its own, but `kept` has no cause of its own, so
        // `cause` stays empty, as its doc says (only `unfiltered` and `low_confidence` have one).
        let mut f = finding(Severity::Warning, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 1;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (GateState::Kept, None, vec!["not every claim judged"])
        );
        assert_eq!(v.lean.as_deref(), Some("real_defect"));
        assert_eq!(
            g.per_model().values().sum::<usize>(),
            1,
            "{:?}",
            g.per_model()
        );
    }

    #[tokio::test]
    async fn a_critical_with_unasked_claims_carries_both_causes() {
        // [MAGI §6 MS2 C loop 3] `critical never filtered` does not replace the unasked claims.
        let mut f = finding(Severity::Critical, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 2;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.70));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("critical never filtered"),
                vec!["critical never filtered", "not every claim judged"]
            )
        );
    }

    #[tokio::test]
    async fn a_confident_critical_with_unasked_claims_is_still_never_filtered() {
        // [T3.7-fix1] above the critical threshold the other guard still rules: unasked claims.
        let mut f = finding(Severity::Critical, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 2;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("not every claim judged"),
                vec!["not every claim judged"]
            )
        );
    }

    #[tokio::test]
    async fn a_lean_that_ends_a_merged_finding_names_its_unasked_claims() {
        // [MAGI §6 MS2 C loop 3] the first pair under every threshold ends the walk: the second
        // pair is never asked, and the `low_confidence` verdict says so.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["a".into(), "b".into()];
        f.details = vec!["da".into(), "db".into()];
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(chain().len())
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.10));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("not every claim judged"),
                vec!["not every claim judged"]
            )
        );
        assert_eq!(v.lean.as_deref(), Some("false_positive"));
    }

    #[tokio::test]
    async fn a_finding_whose_every_pair_was_asked_names_no_unasked_claim() {
        let f = finding(Severity::Critical, Some(1));
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.70));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(v.causes, vec!["critical never filtered"]);
    }

    #[tokio::test]
    async fn a_critical_kept_in_the_list_counts_for_the_model_that_judged_it() {
        let f = finding(Severity::Critical, Some(1));
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.70));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(v.cause, Some("critical never filtered"));
        assert_eq!(g.per_model().get("clef:cloud"), Some(&1));
    }

    #[tokio::test]
    async fn a_finding_whose_every_pair_was_asked_can_be_filtered() {
        let mut f = finding(Severity::Warning, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(n)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!((v.state, v.cause), (GateState::Filtered, None));
    }

    #[tokio::test]
    async fn per_model_counts_findings_not_questions() {
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["a".into(), "b".into()];
        f.details = vec!["da".into(), "db".into()];
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(2)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        assert_eq!(
            g.judge(&f, Some("x\nx\nx\n"), None).await.state,
            GateState::Filtered
        );
        assert_eq!(g.per_model().get("clef:cloud"), Some(&1));
        assert_eq!(g.questions(), 2);
    }

    #[test]
    fn the_question_is_verbatim() {
        let q = gate_questions();
        let v = serde_json::to_value(&q).expect("serializes");
        assert_eq!(v["gate"]["type"], "choice");
        assert_eq!(v["gate"]["instructions"], GATE_INSTRUCTIONS);
        assert_eq!(v["gate"]["criteria"]["real_defect"], GATE_REAL_DEFECT);
        assert_eq!(v["gate"]["criteria"]["false_positive"], GATE_FALSE_POSITIVE);
    }

    #[tokio::test]
    async fn the_request_sent_holds_forty_lines_each_side_not_the_whole_file() {
        // [MAGI S6 loop 7, D-m1, refuted] the pipeline hands the gate the whole side
        // (`gate_code`); what reaches `decide` is only RV10's +/-40 numbered lines around it.
        let code: String = (1..=200).map(|i| format!("l{i}\n")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .withf(|r| {
                let shown = r.state()["code"].as_str().unwrap_or_default();
                shown.starts_with("60: l60\n")
                    && shown.trim_end().ends_with("140: l140")
                    && shown.lines().count() == 81
            })
            .returning(|r| answer(r.model().unwrap_or(""), REAL_DEFECT, 0.95));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(&finding(Severity::Warning, Some(100)), Some(&code), None)
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[test]
    fn the_state_carries_the_finding_and_forty_numbered_lines_each_side() {
        let code: String = (1..=200).map(|i| format!("l{i}\n")).collect();
        let s = gate_state(&finding(Severity::Warning, Some(100)), &code);
        let text = s["code"].as_str().expect("code");
        assert!(text.starts_with("60: l60\n"), "{}", &text[..20]);
        assert!(text.trim_end().ends_with("140: l140"));
        assert_eq!(s["finding"]["line"], 100);
        assert_eq!(s["finding"]["severity"], "warning");
        // [CP2 MS2 S2a1 loop 10] the rest of RV10's state, and the title a merged finding sends.
        assert_eq!(s["finding"]["file"], "a.rs");
        assert_eq!(
            (
                s["finding"]["title"].as_str(),
                s["finding"]["detail"].as_str()
            ),
            (Some("t"), Some("d"))
        );
        let mut merged = finding(Severity::Warning, Some(100));
        merged.titles.push("t2".into());
        merged.details.push("d2".into());
        assert_eq!(gate_state(&merged, &code)["finding"]["title"], "t");
    }

    #[tokio::test]
    async fn a_confident_false_positive_is_filtered_and_a_real_defect_is_kept() {
        let mut c = MockDecisionClient::new();
        let mut seq = mockall::Sequence::new();
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| answer("clef:cloud", "false_positive", 0.9));
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| answer("clef:cloud", "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        assert_eq!(
            g.judge(
                &finding(Severity::Warning, Some(3)),
                Some("x\nx\nx\n"),
                None
            )
            .await
            .state,
            GateState::Filtered
        );
        assert_eq!(
            g.judge(
                &finding(Severity::Warning, Some(3)),
                Some("x\nx\nx\n"),
                None
            )
            .await
            .state,
            GateState::Kept
        );
    }

    #[tokio::test]
    async fn a_model_without_min_confidence_has_no_threshold_like_in_decision() {
        // [CP2 MS2 S2a loop 8] a user-written chain entry that omits min_confidence.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|_| answer("own", "false_positive", 0.05));
        let mut g = Gate::new(
            &c,
            vec![ModelSpec {
                model: "own".into(),
                min_confidence: None,
            }],
            61_000,
        );
        assert_eq!(
            g.judge(
                &finding(Severity::Warning, Some(3)),
                Some("x\nx\nx\n"),
                None
            )
            .await
            .state,
            GateState::Filtered
        );
    }

    #[tokio::test]
    async fn a_critical_with_a_confident_false_positive_is_filtered() {
        // [T3.7-fix1] formerly `a_critical_is_never_filtered` (0.99 stayed in the main list).
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Critical, Some(3)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Filtered, None));
    }

    #[tokio::test]
    async fn a_critical_below_the_critical_threshold_is_never_filtered() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.74));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Critical, Some(3)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("critical never filtered"))
        );
    }

    #[tokio::test]
    async fn a_critical_exactly_at_the_critical_threshold_is_filtered() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.75));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Critical, Some(3)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Filtered, None));
    }

    #[tokio::test]
    async fn a_confident_critical_on_an_ambiguous_side_is_not_filtered() {
        // [T3.7-fix1] above the critical threshold the other guard still rules, and names itself.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge_ambiguous_side(&finding(Severity::Critical, Some(3)), "x\nx\nx\n", None)
            .await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("ambiguous diff side"),
                vec!["ambiguous diff side"]
            )
        );
    }

    #[tokio::test]
    async fn a_finding_without_a_line_is_unfiltered_without_a_call() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&finding(Severity::Warning, None), None, None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::Unfiltered, Some("no location"))
        );
    }

    #[tokio::test]
    async fn a_line_past_the_end_of_the_code_shown_is_no_location_without_a_call() {
        // [MAGI S6 loop 7, C1-c3] the excerpt must hold the finding's line: code that ends before
        // it, or no code at all, is nothing to judge, so no call is made.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        for (line, code) in [(100, "one\ntwo\n"), (1, "")] {
            let v = g
                .judge(&finding(Severity::Warning, Some(line)), Some(code), None)
                .await;
            assert_eq!(
                (v.state, v.cause),
                (GateState::Unfiltered, Some("no location")),
                "line {line}"
            );
        }
    }

    #[tokio::test]
    async fn a_state_over_the_limit_is_unfiltered_not_truncated() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 10);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!(v.cause, Some("state too large"));
    }

    #[tokio::test]
    async fn a_state_carrying_a_one_line_secret_is_sent_masked() {
        // RV22: every byte that leaves the process passes the secret detector, the gate's state too.
        // [MAGI S6 MS3 loop2, R7b Caspar] RV22 amended masks a one-line secret at the state's
        // egress, as at every other: the finding is judged on the masked state, never withheld.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(2).returning(|r| {
            let sent = r.state().to_string();
            assert!(
                !sent.contains("sk-ant-") && !sent.contains("ghp_"),
                "{sent}"
            );
            assert!(sent.contains("<redacted: "), "{sent}");
            answer(r.model().unwrap_or(""), "real_defect", 0.9)
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        let code = "let k = \"sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some(code), None)
            .await;
        assert_eq!((v.state, v.cause), (GateState::Kept, None));
        let mut f = finding(Severity::Warning, Some(1));
        f.details = vec!["token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaks".into()]; // allow-secret-scan
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!((v.state, v.cause), (GateState::Kept, None));
    }

    #[tokio::test]
    async fn a_state_carrying_a_private_key_block_is_never_sent() {
        // RV22's exception: a key body has no pattern of its own, so masking cannot promise that
        // none of it leaves; the state is withheld whole.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let code = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\n{end}\n");
        let v = g
            .judge(&finding(Severity::Warning, Some(2)), Some(&code), None)
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("secret")));
    }

    #[tokio::test]
    async fn a_merged_claim_carrying_a_one_line_secret_is_asked_masked_with_the_others() {
        // [MAGI S6 loop 10, C1-c2] one reworded claim with a secret pattern made the whole merged
        // finding `unfiltered` with no question asked and no `not every claim judged`.
        // [MAGI S6 MS3 loop2] Masked now, so every claim is asked and the finding can filter.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["a".into(), "b".into()];
        f.details = vec![
            "token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaks".into(), // allow-secret-scan
            "db".into(),
        ];
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(2)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (GateState::Filtered, None, Vec::<&str>::new())
        );
        assert_eq!(g.questions(), 2);
    }
    #[tokio::test]
    async fn a_merged_claim_over_the_state_limit_is_withheld_and_a_kept_one_names_it() {
        // [MAGI S6 loop 10, C1-c2] the oversized claim is never sent nor truncated; the other
        // claim is asked, and the verdict names both what withheld one and that it went unasked.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["a".into(), "b".into()];
        f.details = vec!["da".into(), "y".repeat(400)];
        let small = gate_state_of(&f, 0, "x\nx\nx\n").to_string().len();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), small + 10);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::Kept,
                None,
                vec!["state too large", "not every claim judged"]
            )
        );
    }

    #[tokio::test]
    async fn a_bad_request_head_is_skipped_for_the_rest_of_the_review() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .withf(|r| r.model() == Some("clef:cloud"))
            .times(1)
            .returning(|_| bad_request("clef:cloud"));
        c.expect_decide()
            .withf(|r| r.model() == Some("clef-flash"))
            .times(2)
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            let v = g
                .judge(
                    &finding(Severity::Warning, Some(1)),
                    Some("x\nx\nx\n"),
                    None,
                )
                .await;
            assert_eq!(v.model.as_deref(), Some("clef-flash"));
            assert_eq!(v.skipped, vec!["clef:cloud".to_string()]); // [CP2 MS2 S2 loop 2] RV-S15: the skip is visible
        }
        assert_eq!(g.per_model().get("clef-flash"), Some(&2));
    }

    #[tokio::test]
    async fn a_new_review_tries_the_rejected_head_again() {
        // [CP2 MS2 S2a loop 7] U23 / RV-S20: the 400 memory lives in the Gate, and run_review builds one Gate per review
        // (so one per tool call, T2.10). Moving it to the client or a static would skip the head in the next review.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .withf(|r| r.model() == Some("clef:cloud"))
            .times(2)
            .returning(|_| bad_request("clef:cloud"));
        c.expect_decide()
            .withf(|r| r.model() == Some("clef-flash"))
            .times(2)
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        for _ in 0..2 {
            let mut g = Gate::new(&c, chain(), 61_000);
            let v = g
                .judge(
                    &finding(Severity::Warning, Some(1)),
                    Some("x\nx\nx\n"),
                    None,
                )
                .await;
            assert_eq!(v.model.as_deref(), Some("clef-flash")); // `times(2)` on the head is the check: asked once per review
        }
    }

    #[tokio::test]
    async fn a_provider_413_or_422_is_chain_exhausted_and_never_remembered_as_a_400() {
        // [CP2 MS2 S1 loop 6] the memory keys on the typed status, never on the shared `rejected` label.
        // [CP2 MS2 S2 loop 1] 422 too: the spec has no better unfiltered cause for an unprocessable request.
        for status in [413, 422] {
            let mut c = MockDecisionClient::new();
            c.expect_decide()
                .withf(|r| r.model() == Some("clef:cloud"))
                .times(2)
                .returning(move |_| {
                    Err(DecisionError::Rejected {
                        status,
                        provider_message: "too large".into(),
                        trace: Trace::default(),
                    })
                });
            c.expect_decide()
                .withf(|r| r.model() == Some("clef-flash"))
                .times(0);
            let mut g = Gate::new(&c, chain(), 61_000);
            for _ in 0..2 {
                assert_eq!(
                    g.judge(
                        &finding(Severity::Warning, Some(1)),
                        Some("x\nx\nx\n"),
                        None
                    )
                    .await
                    .cause,
                    Some("chain exhausted"),
                    "{status}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_413_or_422_verdict_carries_the_provider_status() {
        // [MAGI S6 loop 5, C1-c3] the status of the refusal that ended the chain reaches the
        // verdict, not only the WARN entry, so the report tells it from an under-threshold end.
        for status in [413, 422] {
            let mut c = MockDecisionClient::new();
            c.expect_decide().returning(move |_| {
                Err(DecisionError::Rejected {
                    status,
                    provider_message: "too large".into(),
                    trace: Trace::default(),
                })
            });
            let mut g = Gate::new(&c, chain(), 61_000);
            let v = g
                .judge(
                    &finding(Severity::Warning, Some(1)),
                    Some(
                        "x
",
                    ),
                    None,
                )
                .await;
            assert_eq!((v.cause, v.status), (Some("chain exhausted"), Some(status)));
        }
    }

    #[tokio::test]
    async fn a_chain_where_every_model_answers_400_is_exhausted_and_never_called_again() {
        // [CP2 MS2 S2a loop 6] every model lands in `rejected`: chain exhausted, all skipped, later findings cost nothing.
        let mut c = MockDecisionClient::new();
        for m in ["clef:cloud", "clef-flash", "nimble"] {
            c.expect_decide()
                .withf(move |r| r.model() == Some(m))
                .times(1)
                .returning(move |_| bad_request(m));
        }
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            let v = g
                .judge(
                    &finding(Severity::Warning, Some(1)),
                    Some("x\nx\nx\n"),
                    None,
                )
                .await;
            assert_eq!(
                (v.state, v.cause),
                (GateState::Unfiltered, Some("chain exhausted"))
            );
            assert_eq!(
                v.skipped,
                vec![
                    "clef:cloud".to_string(),
                    "clef-flash".into(),
                    "nimble".into()
                ]
            );
        }
    }

    #[tokio::test]
    async fn a_policy_rejection_is_classified_by_its_http_status_never_by_its_text() {
        // [CP2 MS2 S2a loop 7] RV-S34 (user, 2026-10-19): `account policy` is unreachable; a ZDR message on a 400 is a 400.
        let zdr = |m: &str| -> Result<Decision, DecisionError> {
            Err(DecisionError::Rejected {
                status: 400,
                provider_message: format!(
                    "{m}: blocked by your account's zero data retention policy"
                ),
                trace: Trace::default(),
            })
        };
        let mut c = MockDecisionClient::new();
        for m in ["clef:cloud", "clef-flash", "nimble"] {
            c.expect_decide()
                .withf(move |r| r.model() == Some(m))
                .times(1)
                .returning(move |_| zdr(m));
        }
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::Unfiltered, Some("chain exhausted"))
        );
        assert_eq!(v.skipped.len(), 3, "each 400 rotated like any other 400");
    }

    /// A single-model chain that ran out of time.
    fn timed_out(model: &str) -> Result<Decision, DecisionError> {
        Err(DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![Attempt {
                    model: model.into(),
                    outcome: AttemptOutcome::Failed {
                        status: None,
                        cause: Cause::Timeout,
                        detail: None,
                    },
                    usage: Default::default(),
                }],
                usage: Default::default(),
            },
        })
    }

    #[tokio::test]
    async fn each_call_is_capped_by_what_is_left_and_running_out_mid_gate_is_timeout() {
        // [CP2 MS2 S2 loop 1] N-10: min(call timeout, remaining); a call capped by `remaining` that times out is RV-S16's `timeout`.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            // [CP2 MS2 cierre (sin MAGI)] a range: the cap is what is left of the deadline, a little under 5 s by the time it is sent.
            .withf(|r| {
                r.timeout().is_some_and(|t| {
                    t > std::time::Duration::from_secs(4) && t <= std::time::Duration::from_secs(5)
                })
            })
            .returning(|r| timed_out(r.model().unwrap_or("")));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::from_secs(5)),
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("timeout")));
        // the review clock is spent: the next finding is not sent at all (times(1) above)
        let v = g
            .judge(
                &finding(Severity::Warning, Some(2)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::from_secs(5)),
            )
            .await;
        assert_eq!(v.cause, Some("timeout"));
        assert_eq!(g.stopped(), Some("timeout")); // [CP2 MS2 S2 loop 4]
    }

    #[tokio::test]
    async fn the_review_clock_is_re_read_before_each_call_of_the_chain() {
        // [CP2 MS2 S2a loop 9] `remaining` is a deadline taken at entry: the second model gets what is left after the
        // first one answered, never the whole `remaining` again.
        let mut c = MockDecisionClient::new();
        let mut seq = mockall::Sequence::new();
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| {
                std::thread::sleep(std::time::Duration::from_millis(1_100));
                answer("clef:cloud", "false_positive", 0.10) // under clef:cloud's 0.40: next model
            });
        // [CP2 MS2 S2a1 loop 10] 30 s of review left: reusing the entry `remaining` passes 30 s and fails; a stall has
        // to reach ~29 s before the honest implementation flakes (it was 0.9 s with a 2 s budget).
        c.expect_decide()
            .times(1)
            .in_sequence(&mut seq)
            .withf(|r| {
                r.timeout()
                    .is_some_and(|t| t < std::time::Duration::from_secs(29))
            })
            .returning(|_| answer("clef-flash", "real_defect", 0.9));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::from_secs(30)),
            )
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[test]
    fn a_review_clock_equal_to_the_call_timeout_is_the_clock_that_cuts_the_call() {
        // [MAGI §6 MS2 C loop 3] when what is left equals the per-call timeout, a timeout of that
        // call is the review clock running out (`timeout`), not a slow model (`transport`).
        let t = std::time::Duration::from_secs(120);
        assert_eq!(call_cap(Some(t), t), (t, true));
        assert_eq!(
            call_cap(Some(t + std::time::Duration::from_secs(1)), t),
            (t, false)
        );
        assert_eq!(call_cap(None, t), (t, false));
    }

    #[tokio::test]
    async fn a_spent_review_clock_never_reaches_the_http_client() {
        // [CP2 MS2 S2a2 loop 10] a cap of zero is never sent as a 0 s HTTP timeout: the call is skipped as `timeout`.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::ZERO),
            )
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("timeout")));
    }

    #[tokio::test]
    async fn the_first_stop_cause_wins_and_a_spent_clock_does_not_replace_a_401() {
        // [CP2 MS2 S2 loop 4] precedence: stopped is write-once.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        g.judge(
            &finding(Severity::Warning, Some(1)),
            Some("x\nx\nx\n"),
            Some(std::time::Duration::from_secs(60)),
        )
        .await;
        let v = g
            .judge(
                &finding(Severity::Warning, Some(2)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::ZERO),
            )
            .await;
        assert_eq!(
            (v.cause, g.stopped()),
            (Some("unauthorized"), Some("unauthorized"))
        );
    }

    #[tokio::test]
    async fn every_model_timing_out_on_its_own_call_timeout_is_transport() {
        // [CP2 MS2 S2 loop 4] per-call timeouts are transport-class; `timeout` is reserved for the review clock (RV-S16).
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(3)
            .returning(|r| timed_out(r.model().unwrap_or("")));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!(
            (v.state, v.cause, g.stopped()),
            (GateState::Unfiltered, Some("transport"), None)
        );
    }

    #[tokio::test]
    async fn without_a_configured_call_timeout_each_call_uses_the_default_one() {
        // A gate built without `with_call_timeout` must still cap each call by its own timeout,
        // or a slow model would be read as the review clock running out.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .withf(|r| {
                r.timeout()
                    == Some(std::time::Duration::from_secs(
                        crate::review::config::DEFAULT_CALL_TIMEOUT_SECS,
                    ))
            })
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::from_secs(5_000)),
            )
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[tokio::test]
    async fn after_a_401_a_finding_without_a_line_is_still_no_location() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        g.judge(
            &finding(Severity::Warning, Some(1)),
            Some("x\nx\nx\n"),
            None,
        )
        .await;
        let v = g.judge(&finding(Severity::Warning, None), None, None).await;
        assert_eq!(
            (v.state, v.cause, g.stopped()),
            (
                GateState::Unfiltered,
                Some("no location"),
                Some("unauthorized")
            )
        );
    }

    #[tokio::test]
    async fn with_time_to_spare_each_call_uses_the_configured_timeout() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .withf(|r| r.timeout() == Some(std::time::Duration::from_secs(120)))
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g =
            Gate::new(&c, chain(), 61_000).with_call_timeout(std::time::Duration::from_secs(120));
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                Some(std::time::Duration::from_secs(5_000)),
            )
            .await;
        assert_eq!(v.state, GateState::Kept);
    }

    #[tokio::test]
    async fn every_unfiltered_finding_logs_one_warn_never_deduplicated() {
        // [CP2 MS2 S2 loop 1] RV16. A counting layer on GATE_LOG_TARGET at WARN.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Count(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Count {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if e.metadata().target() == GATE_LOG_TARGET
                    && *e.metadata().level() == tracing::Level::WARN
                {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let count = Count::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(count.clone()));
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..2 {
            g.judge(&finding(Severity::Warning, None), None, None).await; // the same cause twice
        }
        assert_eq!(count.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unauthorized_stops_the_gate_for_the_whole_review() {
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        for _ in 0..3 {
            assert_eq!(
                g.judge(
                    &finding(Severity::Warning, Some(1)),
                    Some("x\nx\nx\n"),
                    None
                )
                .await
                .cause,
                Some("unauthorized")
            );
        }
        assert_eq!(g.stopped(), Some("unauthorized")); // [CP2 MS2 S2 loop 3] what the report declares once
    }

    #[tokio::test]
    async fn a_low_confidence_chain_passes_with_its_lean() {
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.10));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("x\nx\nx\n"),
                None,
            )
            .await;
        assert_eq!(
            (v.state, v.lean.as_deref()),
            (GateState::LowConfidence, Some("false_positive"))
        );
    }

    /// The production client against `url`, with the default chain and no threshold of its own.
    fn served_gate_client(url: &str) -> crate::decision::client::SystemOneClient {
        crate::decision::client::SystemOneClient::new(crate::decision::client::DecisionSettings {
            base_url: url.to_string(),
            api_key: None,
            chain: chain(),
            timeout: std::time::Duration::from_secs(10),
            keep_alive: None,
        })
        .expect("client")
    }

    /// A mocked `/systemone` answer of `model` with `status` and `body`, expected `hits` times.
    async fn served(
        server: &mut mockito::ServerGuard,
        model: &str,
        status: usize,
        body: String,
        hits: usize,
    ) -> mockito::Mock {
        server
            .mock("POST", "/v1/systemone")
            .match_body(mockito::Matcher::PartialJson(json!({ "model": model })))
            .with_status(status)
            .with_body(body)
            .expect(hits)
            .create_async()
            .await
    }

    #[tokio::test]
    async fn a_404_or_5xx_head_rotates_to_the_fallback_and_is_never_remembered_as_a_400() {
        // [MAGI S6 loop 8, C1-c1] refutation, end to end over the production client: for a
        // pinned model a 404 or a 5xx ends as `ModelUnavailable`, never as `Rejected`, so the
        // gate rotates past it and the per-review 400 memory stays empty (the head is asked
        // again for the next finding).
        for status in [404, 500] {
            let mut server = mockito::Server::new_async().await;
            let head = served(&mut server, "clef:cloud", status, "{}".into(), 2).await;
            let body = json!({"model": "clef-flash", "answers": {GATE_QUESTION_NAME: {
                "type": "choice", "choice": "real_defect",
                "probabilities": {"real_defect": 0.95, "false_positive": 0.05}}}})
            .to_string();
            let fallback = served(&mut server, "clef-flash", 200, body, 2).await;
            let client = served_gate_client(&server.url());
            let mut g = Gate::new(&client, chain(), 61_000);
            for line in [1, 2] {
                let v = g
                    .judge(
                        &finding(Severity::Warning, Some(line)),
                        Some("x\nx\nx\n"),
                        None,
                    )
                    .await;
                assert_eq!(
                    (v.state, v.model.as_deref()),
                    (GateState::Kept, Some("clef-flash")),
                    "HTTP {status}"
                );
                assert!(v.skipped.is_empty(), "HTTP {status}: {:?}", v.skipped);
            }
            head.assert_async().await;
            fallback.assert_async().await;
        }
    }

    #[tokio::test]
    async fn a_413_or_422_stops_the_finding_on_its_status_and_is_never_remembered() {
        // [MAGI S6 loop 8, C2-c3] the memory keys on `DecisionError::Rejected { status }`, never
        // on the serialized `rejected` token a 413/422 shares with a 400: the head is asked
        // again for the next finding and the fallback never.
        for status in [413_u16, 422] {
            let mut server = mockito::Server::new_async().await;
            let head = served(&mut server, "clef:cloud", status.into(), "{}".into(), 2).await;
            let fallback = served(&mut server, "clef-flash", 200, "{}".into(), 0).await;
            let client = served_gate_client(&server.url());
            let mut g = Gate::new(&client, chain(), 61_000);
            for line in [1, 2] {
                let v = g
                    .judge(
                        &finding(Severity::Warning, Some(line)),
                        Some("x\nx\nx\n"),
                        None,
                    )
                    .await;
                assert_eq!(
                    (v.state, v.cause, v.status),
                    (GateState::Unfiltered, Some("chain exhausted"), Some(status))
                );
                assert!(v.skipped.is_empty(), "HTTP {status}: {:?}", v.skipped);
            }
            head.assert_async().await;
            fallback.assert_async().await;
        }
    }

    #[tokio::test]
    async fn an_ambiguous_side_turns_a_confident_false_positive_into_low_confidence() {
        // [MAGI S6 loop 9, C1-m2] RV10's replaced-line amendment, pinned at the gate: on a line
        // whose number also names a deleted line, a confident `false_positive` is never filtered.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge_ambiguous_side(&finding(Severity::Warning, Some(3)), "x\nx\nx\n", None)
            .await;
        assert_eq!(
            (v.state, v.cause, v.causes.clone()),
            (
                GateState::LowConfidence,
                Some("ambiguous diff side"),
                vec!["ambiguous diff side"]
            )
        );
        assert_eq!(g.per_model().get("clef:cloud"), Some(&1));
    }

    #[tokio::test]
    async fn a_critical_on_an_ambiguous_side_keeps_both_causes() {
        // [MAGI S6 loop 9, C1-c2] the critical rule set the cause and the ambiguity was dropped,
        // though `causes` lists every cause that applies.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.70));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g
            .judge_ambiguous_side(&finding(Severity::Critical, Some(3)), "x\nx\nx\n", None)
            .await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("critical never filtered"),
                vec!["critical never filtered", "ambiguous diff side"]
            )
        );
    }

    #[tokio::test]
    async fn a_finding_with_a_line_but_no_file_is_no_location_without_a_call() {
        // [MAGI S6 loop 9, C1-c3] the gate checked the line and the code, never the file: a
        // finding with `file: None` sent a state with a null file.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 61_000);
        let mut f = finding(Severity::Warning, Some(1));
        f.file = None;
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::Unfiltered, Some("no location"))
        );
    }
}
