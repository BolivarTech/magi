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
use sha2::{Digest, Sha256};

use crate::decision::{
    Answer, AttemptOutcome, Cause, Decision, DecisionClient, DecisionError, DecisionRequest,
    ModelSpec, Question, Questions,
};

use super::config::DEFAULT_CALL_TIMEOUT_SECS;
use super::consolidate::ConsolidatedFinding;
use super::secrets::{find_secret, holds_key_block, mask_secrets, REDACTED_OPEN};

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
/// Cause of a finding whose code window carries a mask marker: the gate saw less than the file
/// holds, so a confident `false_positive` is not trusted and the finding is never `filtered`.
const CAUSE_MASKED_CODE: &str = "masked code";
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
/// judged `false_positive`. A constant, not configuration, by the user's decision (U21, bound at
/// 0.75): moving it is a recalibration, decided with evidence, never an operator knob.
///
/// Interaction with a model's `min_confidence` (declared, MS3 §6 loop16): the two compose. For a
/// model whose `min_confidence` is at or above this bound, an answer under the bound is already
/// `low_confidence` by the model's own threshold, so this rule adds nothing for it; that is the
/// specified behavior (the 2026-11-06 amendment filters a `critical` at confidence >= 0.75), not
/// a gap. For a model with a lower `min_confidence` (the built-in chain: 0.40, 0.32, 0.80) the
/// window between its threshold and this bound is where a `critical` stays in the main list.
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
// Invariant: every object key here is one of this function's own literals, never foreign text;
// `masked_strings` masks values only and relies on it (a key built from a path or a title would
// make the state withheld). Pinned by `the_gate_state_keys_are_the_gates_own_literals`.
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

/// The first and last line (numbered from 1, both included) of the window around `line`: the one
/// definition [`numbered_window`] cuts and [`window_meets_block`] tests against, so what is shown
/// to the gate and what is checked for a key block cannot drift apart
/// (`the_window_shown_and_the_window_checked_share_their_bounds`). O(1).
fn window_bounds(line: u32) -> (u32, u32) {
    (
        line.saturating_sub(GATE_CONTEXT_LINES).max(1),
        line.saturating_add(GATE_CONTEXT_LINES),
    )
}

/// The lines `line +/- GATE_CONTEXT_LINES` of `code`, numbered from 1.
fn numbered_window(code: &str, line: u32) -> String {
    let (first, last) = window_bounds(line);
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

/// The five dashes of PEM and OpenPGP armor.
const ARMOR_DASHES: &str = "-----";
/// The four dashes of SSH2 armor.
const SSH2_DASHES: &str = "----";
/// What opens an armor `END` line, as `(dashes, word right after them)`: five dashes then
/// `END ` (PEM, OpenPGP), four then ` END ` (SSH2). Built from named dashes because the review
/// tool showed the joined literal mangled; pinned by `armor_end_lines_close_a_block`.
const KEY_BLOCK_END_OPENERS: [(&str, &str); 2] = [(ARMOR_DASHES, "END "), (SSH2_DASHES, " END ")];

/// Whether `text` is an armor `END` line: after leading whitespace it opens with an END opener
/// ([`KEY_BLOCK_END_OPENERS`]), its dash run exactly that long. An END quoted inside a line (an
/// OpenPGP `Comment:` header, a glued `x-----END`) or a five-dash ` END` never closes a block, so
/// a block stays open over its body: the safe direction (`armor_end_must_open_its_line`). O(1).
///
/// Declared (MS3 §6 loop12): `secrets::mask_secrets` has its own notion of where a block ends
/// (it masks a block whole for the report). The two are kept apart on purpose: that one decides
/// what text is masked, this one only how far a withheld window reaches inside a file the filter
/// has ALREADY left out for holding a block (RV22's file-level exception), a backstop. A drift
/// loosens at most this backstop, never the file-level omission, and each side is pinned by its
/// own tests (`armor_end_lines_close_a_block`, `armor_end_must_open_its_line`), and the relation
/// between them by `footer_rules_agree`: on every canonical PEM, OpenPGP and SSH2 block, where the
/// mask stops is a line this reads as an END line.
// MAGI FIX (MS3 §6 loop10): any occurrence closed a block, `----- END` as an SSH2 footer.
fn holds_armor_end(text: &str) -> bool {
    let line = text.trim_start();
    KEY_BLOCK_END_OPENERS.iter().any(|(dashes, word)| {
        line.strip_prefix(dashes)
            .is_some_and(|r| r.starts_with(word))
    })
}

/// Whether the window [`numbered_window`] cuts around `line` overlaps a private-key block of
/// `code` (RV10's state is the finding plus ±[`GATE_CONTEXT_LINES`]; RV22's exception keeps a
/// block out of it).
///
/// A block runs from a line that opens one ([`holds_key_block`] on that line) to the first later
/// line holding an armor `END` line ([`KEY_BLOCK_END_OPENERS`]), or to the end of `code` when it
/// has none: a window inside a long block holds body lines only, which no pattern names, and is
/// still withheld (`a_key_block_longer_than_the_window_is_found_in_the_whole_code_and_never_sent`),
/// while a finding far from the block is judged
/// (`a_key_block_far_from_the_window_does_not_withhold_the_finding`). When the code outside the
/// blocks tracked by their lines still holds a block (a header the detector reads only across
/// lines), the whole code counts as the block, whatever else was tracked: the safe direction
/// (`a_header_read_only_across_lines_is_not_hidden_by_another_tracked_block`). Declared limit: a body line copied far from
/// its block is not recognized here; the filter leaves every file holding a block out first
/// (RV22's file-level exception), so this is a backstop. O(bytes of `code`).
// MAGI FIX (MS3 §6 loop6): the whole-code check withheld every finding of such a file.
// MAGI FIX (MS3 §6 loop8): `scan` is the per-file part, memoized by `Gate`: no second whole-code
// pass and no allocation per finding.
fn window_overlaps_key_block(line: u32, scan: &KeyBlockScan) -> bool {
    scan.holds && (scan.untracked || window_meets_block(&scan.blocks, line))
}

/// What the gate knows of one file's private-key blocks, whatever the finding: computed once
/// per file ([`Gate::key_block_scan`]).
#[derive(Debug, Clone)]
struct KeyBlockScan {
    /// The code holds a block ([`holds_key_block`]).
    holds: bool,
    /// The detector finds one outside the blocks its lines open and close ([`untracked_block`]).
    untracked: bool,
    /// The line intervals `(first, last)` of the blocks its lines open and close, computed once
    /// per file: each finding then costs O(blocks), not a pass over the code.
    // MAGI FIX (MS3 §6 loop15): `overlaps_tracked_block` rescanned the whole file per finding
    // (R6b2).
    blocks: Vec<(u32, u32)>,
}

impl KeyBlockScan {
    /// The scan of `code` with `opens` the block detector (the production one is
    /// [`holds_key_block`]; a test gives one that reads a header only across lines, which none
    /// does today, so the `untracked` fallback is exercised through the production path of
    /// [`Gate::key_block_scan`]). O(bytes of `code`) times the cost of `opens`.
    fn of(code: &str, opens: &dyn Fn(&str) -> bool) -> Self {
        let holds = opens(code);
        let untracked = holds && untracked_block(code, opens);
        let blocks = if holds && !untracked {
            tracked_blocks(code, opens)
        } else {
            Vec::new()
        };
        Self {
            holds,
            untracked,
            blocks,
        }
    }
}

/// [`window_overlaps_key_block`] with the block detector `opens` given, so the cases the real
/// detector cannot produce today (a header read only across lines) are testable.
#[cfg(test)]
fn window_overlaps_block_of(code: &str, line: u32, opens: &dyn Fn(&str) -> bool) -> bool {
    opens(code)
        && (untracked_block(code, opens) || window_meets_block(&tracked_blocks(code, opens), line))
}

/// Whether the window around `line` overlaps one of `blocks` ([`tracked_blocks`]). O(blocks).
fn window_meets_block(blocks: &[(u32, u32)], line: u32) -> bool {
    let (first, last) = window_bounds(line);
    blocks
        .iter()
        .any(|&(start, end)| start <= last && end >= first)
}

/// The line intervals `(first, last)` of the blocks of `code` opened by one of its lines and
/// closed by its armor `END` line; a block still open at the end runs to the last line
/// (`u32::MAX`). O(bytes of `code`) times the cost of `opens`.
fn tracked_blocks(code: &str, opens: &dyn Fn(&str) -> bool) -> Vec<(u32, u32)> {
    let mut blocks = Vec::new();
    let mut opened: Option<u32> = None;
    for (n, text) in (1u32..).zip(code.lines()) {
        match opened {
            None if opens(text) => opened = Some(n),
            Some(start) if holds_armor_end(text) => {
                blocks.push((start, n));
                opened = None;
            }
            _ => {}
        }
    }
    if let Some(start) = opened {
        blocks.push((start, u32::MAX));
    }
    blocks
}

/// Whether the detector still finds a block in the lines outside every block tracked by its
/// lines (a header read only across lines): then the whole code counts as the block, whatever the
/// window. Independent of the finding, so computed once per file. O(bytes of `code`).
// MAGI FIX (MS3 §6 loop7): that fallback was off once any block was tracked by its lines.
fn untracked_block(code: &str, opens: &dyn Fn(&str) -> bool) -> bool {
    let mut opened = false;
    let mut outside = String::new();
    for text in code.lines() {
        if !opened && opens(text) {
            opened = true;
        } else if opened && holds_armor_end(text) {
            opened = false;
        } else if !opened {
            outside.push_str(text);
            outside.push('\n');
        }
    }
    opens(&outside)
}

/// Whether `code` has a line numbered `line` (from 1). O(bytes up to that line).
fn holds_line(code: &str, line: u32) -> bool {
    usize::try_from(line)
        .ok()
        .and_then(|l| l.checked_sub(1))
        .is_some_and(|i| code.lines().nth(i).is_some())
}

/// Whether `holds` is true of any string of `value`, an object's keys included. O(bytes) times
/// the cost of `holds`.
// MAGI FIX (MS3 §6 loop3): the keys were not read, so the residual check after masking (which
// masks values only) could not see a key holding a secret.
fn any_string(value: &Value, holds: &dyn Fn(&str) -> bool) -> bool {
    match value {
        Value::String(text) => holds(text),
        Value::Array(items) => items.iter().any(|item| any_string(item, holds)),
        Value::Object(fields) => fields
            .iter()
            .any(|(key, item)| holds(key) || any_string(item, holds)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// `value` with every string value passed through [`mask_secrets`]. Keys are not masked, but
/// they are scanned: a state's keys are the gate's own field names today, and the residual
/// check ([`any_string`], which reads every key) withholds a state whose key holds a secret.
/// So a key built from foreign text would make the state withheld, never masked: build keys
/// from the gate's own literals only. O(bytes).
fn masked_strings(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(mask_secrets(&text)),
        Value::Array(items) => Value::Array(items.into_iter().map(masked_strings).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k, masked_strings(v)))
                .collect(),
        ),
        other => other,
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
    /// The guard the state itself raised (`masked code`) when the critical rule's cause took
    /// the state's cause, so both are named in the causes.
    guard_also: Option<&'static str>,
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
            guard_also: None,
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
            guard_also: None,
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
    /// The SHA-256 of the last code scanned for private-key blocks, and what the scan found: the
    /// findings of one file come together, so its whole-code scans run once, not once per
    /// finding. Keyed by the digest, so no copy of a file (up to `MAX_REVIEW_FILE_BYTES`) is
    /// held (hashing costs more per byte than a comparison would: the gain is the memory, not
    /// the time); a digest that matches is the same code, since SHA-256 does not collide in
    /// practice (a miss only rescans). One slot: the O(files) bound holds for findings grouped by
    /// file, as the pipeline hands them; in any other order each file change rescans (correct,
    /// `scan_memo_any_order`, only slower).
    // MAGI FIX (MS3 §6 loop12): it kept a copy of the file and compared it whole (R6b Balthasar).
    last_scan: Option<([u8; 32], KeyBlockScan)>,
    /// Whole-code scans done (tests count them: `scan_memo_is_keyed_by_digest`).
    #[cfg(test)]
    scans: usize,
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
            last_scan: None,
            #[cfg(test)]
            scans: 0,
        }
    }

    /// The [`KeyBlockScan`] of `code`, from [`Self::last_scan`] when `code` is the code scanned
    /// last (same SHA-256). O(bytes of `code`): one digest, plus the scans when the file changed;
    /// no copy of the file is kept.
    /// Correct in any order (`scan_memo_any_order`); O(files) scans only for findings grouped
    /// by file, as the pipeline hands them.
    // MAGI FIX (MS3 §6 loop7): the whole-code scan ran once per finding (R6b Melchior).
    fn key_block_scan(&mut self, code: &str) -> KeyBlockScan {
        let digest: [u8; 32] = Sha256::digest(code.as_bytes()).into();
        if let Some((last, scan)) = &self.last_scan {
            if *last == digest {
                return scan.clone();
            }
        }
        #[cfg(test)]
        {
            self.scans += 1;
        }
        let scan = KeyBlockScan::of(code, &holds_key_block);
        self.last_scan = Some((digest, scan.clone()));
        scan
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
    /// * `code` - the file it points into, if it was read: the side the finding's line lies on
    ///   (the caller's `gate_code` reads a removed line from the old side, RV21; a line it cannot
    ///   place gives `None`, which is `no location` and never a call). The gate cannot tell the
    ///   sides apart and must not be handed the other one.
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
        if let Some(also) = outcome.guard_also.filter(|a| !causes.contains(a)) {
            causes.push(also);
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
        // MAGI FIX (MS3 §6 loop6): a stopped gate keeps its cause (C-6: `unauthorized` for every
        // later finding, declared once); the key-block check ran first and said `secret`.
        if let Some(cause) = self.stopped {
            return Outcome::unfiltered(cause);
        }
        // MAGI FIX (MS3 §6 loop12): the clock first, so a spent review does no scan work; nothing
        // is sent either way (the timeout stops the gate before any call).
        if deadline.is_some_and(|d| d <= Instant::now()) {
            return self.stop(CAUSE_TIMEOUT);
        }
        // MAGI FIX (MS3 §6 loop5): the block is looked for in the whole code, before the window
        // clips it: a window inside a longer block holds body lines only, which no pattern
        // names, so [`Self::sendable`]'s check on the state could not see it.
        // MAGI FIX (MS3 §6 loop6): only a finding whose window overlaps a block's lines is
        // withheld; one far from it is judged (R6b).
        if window_overlaps_key_block(line, &self.key_block_scan(code)) {
            return Outcome::unfiltered(CAUSE_SECRET);
        }
        let pairs = constituents(finding).len();
        let asked = pairs.min(MAX_GATE_QUESTIONS_PER_FINDING);
        // MAGI FIX (S6 loop 10, C1-c2): the checks run per pair. A pair too large or carrying a
        // secret is withheld (never sent, never truncated) and the others are still asked; only
        // a finding whose every pair is withheld leaves `unfiltered` with that pair's cause.
        // MAGI FIX (MS3 §6 loop2): RV22 amended masks a one-line secret at the state's egress
        // instead of withholding the pair; only a private-key block still withholds it.
        // The first withheld pair's cause is the one reported, on purpose: RV10 names "that
        // pair's cause" in the singular, and one cause already tells the finding was not fully
        // judged; a second cause (another pair over `max_state_bytes`) is not listed.
        let mut withheld = None;
        let states: Vec<Value> = (0..asked)
            .filter_map(|i| match self.sendable(gate_state_of(finding, i, code)) {
                Ok(state) => Some(state),
                Err(cause) => {
                    withheld.get_or_insert(cause);
                    None
                }
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

    /// `state` as it may be sent, or why it cannot be. The code was already checked for a
    /// private-key block overlapping the window before the window was cut ([`Self::decide`],
    /// `window_overlaps_key_block`), so here a string holding one is the finding's own text
    /// (`a_state_whose_own_text_quotes_a_key_block_is_never_sent`). Such a string
    /// withholds it (`secret`: RV22's exception, a key body has no pattern of its own); otherwise
    /// every one-line secret of every string is masked ([`mask_secrets`]), and the masked state
    /// is withheld when it is over `max_state_bytes` (`state too large`) or masking left a match
    /// (`secret`). O(bytes of the state).
    ///
    /// Declared limit (RV22): the check is per string, so key-body lines quoted in a detail
    /// without their header in that same string are not recognized here; the filter's file-level
    /// rule (a file holding a block is left out) and the window check above come first.
    ///
    /// # Errors
    /// The cause the state is withheld for.
    fn sendable(&self, state: Value) -> Result<Value, &'static str> {
        if any_string(&state, &holds_key_block) {
            return Err(CAUSE_SECRET);
        }
        let state = masked_strings(state);
        if state.to_string().len() > self.max_state_bytes {
            Err(CAUSE_STATE_TOO_LARGE)
        } else if any_string(&state, &|text| find_secret(text).is_some()) {
            Err(CAUSE_SECRET)
        } else {
            Ok(state)
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
        // MAGI FIX (MS3 §6 loop13): a state that carries a mask marker is never `filtered`.
        // MAGI FIX (MS3 §6 loop14): the trigger is the marker's text in the state, not the act of
        // masking (`sendable` returns a `Value`, and widening it for the flag would change every
        // caller). A finding that itself quotes the marker therefore also stays out of `filtered`
        // (`a_quoted_marker_is_never_filtered`): the safe direction, one `low_confidence`.
        // MAGI FIX (MS3 §6 loop16, R6b2): the mask guard is kept apart from the caller's, so it
        // is named even when another guard (or the critical rule) took the state's own cause.
        let masked = state
            .to_string()
            .contains(REDACTED_OPEN)
            .then_some(CAUSE_MASKED_CODE);
        let guard = guard.or(masked);
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
                    // MAGI FIX (MS3 §6 loop15): the guard the state raised is named beside the
                    // critical cause that replaced it.
                    let guard_also = masked.filter(|m| cause.is_some_and(|c| c != *m));
                    return Outcome {
                        decided: true,
                        guard_also,
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
    async fn a_critical_with_unasked_pairs_and_a_confident_false_positive_is_low_confidence() {
        // MAGI FIX (MS3 §6 loop14): a critical whose confidence reaches CRITICAL_MIN_CONFIDENCE is
        // not protected by its severity, so only the question cap keeps it from `filtered`: it
        // stays in the main list as `low_confidence` with `not every claim judged`.
        let mut f = finding(Severity::Critical, Some(1));
        let n = MAX_GATE_QUESTIONS_PER_FINDING + 1;
        f.titles = (0..n).map(|i| format!("t{i}")).collect();
        f.details = (0..n).map(|i| format!("d{i}")).collect();
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(MAX_GATE_QUESTIONS_PER_FINDING)
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("not every claim judged"))
        );
        assert!(g.per_model().is_empty(), "{:?}", g.per_model());
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

    #[test]
    fn a_state_whose_key_holds_a_secret_is_withheld_by_the_residual_check() {
        // [MAGI S6 MS3 loop3, R6b Caspar, Balthasar] Masking reads the values only (a state's
        // keys are the gate's own field names), and the residual check after it read the values
        // only too: a key holding a secret would have left as written. The residual check now
        // reads the keys, and this is the branch it guards.
        let c = MockDecisionClient::new();
        let g = Gate::new(&c, chain(), 61_000);
        let key = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        assert_eq!(
            g.sendable(serde_json::json!({ "finding": { key: "x" } })),
            Err(CAUSE_SECRET)
        );
        assert!(g
            .sendable(serde_json::json!({ "finding": { "file": "a.rs" } }))
            .is_ok());
    }

    #[test]
    fn the_state_size_is_measured_after_masking() {
        // [MAGI S6 MS3 loop3, R6b Melchior] A mask can be longer than the secret it replaces;
        // what is sent is the masked state, so that is what `max_state_bytes` bounds.
        let c = MockDecisionClient::new();
        let raw = "postgres://u:p@db/app";
        let masked = mask_secrets(raw);
        assert!(masked.len() > raw.len(), "precondition: {masked}");
        let state = serde_json::json!({ "code": raw });
        let g = Gate::new(&c, chain(), state.to_string().len());
        assert_eq!(g.sendable(state), Err(CAUSE_STATE_TOO_LARGE));
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
    async fn a_key_block_longer_than_the_window_is_found_in_the_whole_code_and_never_sent() {
        // [MAGI S6 MS3 loop5, R6b Balthasar] The state shows RV10's ±40 lines; a finding in the
        // middle of a longer block sees only body lines, which match no pattern of their own, so
        // the windowed check alone would send them. The block is looked for in the whole code.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(0);
        let mut g = Gate::new(&c, chain(), 1_000_000);
        let begin = concat!("-----", "BEGIN ", "RSA PRIVATE KEY-----");
        let end = concat!("-----", "END ", "RSA PRIVATE KEY-----");
        let body: String = (0..200)
            .map(|_| "MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\n")
            .collect();
        let code = format!("fn a() {{}}\n{begin}\n{body}{end}\n");
        let line = 102;
        let window = gate_state_of(&finding(Severity::Warning, Some(line)), 0, &code);
        assert!(
            !any_string(&window, &holds_key_block),
            "precondition: the window holds body lines only"
        );
        let v = g
            .judge(&finding(Severity::Warning, Some(line)), Some(&code), None)
            .await;
        assert_eq!((v.state, v.cause), (GateState::Unfiltered, Some("secret")));
    }

    #[test]
    fn the_gate_state_keys_are_the_gates_own_literals() {
        // [MAGI S6 MS3 loop7, R6b Balthasar] `masked_strings` masks values, not keys: whatever
        // the finding holds, the state's keys are the gate's fixed field names.
        let mut f = finding(Severity::Warning, Some(1));
        f.file = Some("sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.rs".into()); // allow-secret-scan
        f.titles = vec!["title".into()];
        let state = gate_state_of(&f, 0, "fn a() {}\n");
        let keys = |v: &Value| -> Vec<String> {
            v.as_object()
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default()
        };
        assert_eq!(keys(&state), ["code", "finding"]);
        assert_eq!(
            keys(&state["finding"]),
            ["detail", "file", "line", "severity", "title"]
        );
    }

    #[test]
    fn armor_end_lines_close_a_block() {
        // [MAGI S6 MS3 loop8] The openers are built from named dashes (the review tool mangled
        // the joined literal): checked against an independent count, and every canonical footer
        // closes a block, so a finding past it is outside.
        assert_eq!(
            (ARMOR_DASHES, SSH2_DASHES),
            ("-".repeat(5).as_str(), "-".repeat(4).as_str())
        );
        let d = ARMOR_DASHES;
        for footer in [
            format!("{d}END PRIVATE KEY{d}"),
            format!("{d}END PGP PRIVATE KEY BLOCK{d}"),
            format!("  {d}END RSA PRIVATE KEY{d}"),
            format!("{SSH2_DASHES} END SSH2 ENCRYPTED PRIVATE KEY {SSH2_DASHES}"),
        ] {
            assert!(holds_armor_end(&footer), "{footer}");
            let tail: String = (0..100).map(|i| format!("f{i}\n")).collect();
            let code = format!("OPEN\nbody\n{footer}\n{tail}");
            assert!(
                !window_overlaps_block_of(&code, 90, &|t| t.contains("OPEN")),
                "{footer}"
            );
        }
        assert!(!holds_armor_end("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC+/="));
    }

    #[test]
    fn footer_rules_agree() {
        // [MAGI S6 MS3 loop13, R6b] The gate's END rule and `secrets::mask_secrets`' are two
        // implementations kept apart on purpose; this pins their relation. Over the canonical
        // PEM, OpenPGP and SSH2 blocks, the point where the mask stops (the text after the block
        // survives) is a line the gate reads as an END line, so the gate's backstop never
        // reaches less far than the mask on any footer the mask recognises.
        let (d, s) = (ARMOR_DASHES, SSH2_DASHES);
        let blocks = [
            (
                format!("{d}BEGIN PRIVATE KEY{d}"),
                format!("{d}END PRIVATE KEY{d}"),
            ),
            (
                format!("{d}BEGIN PGP PRIVATE KEY BLOCK{d}"),
                format!("{d}END PGP PRIVATE KEY BLOCK{d}"),
            ),
            (
                format!("{s} BEGIN SSH2 ENCRYPTED PRIVATE KEY {s}"),
                format!("{s} END SSH2 ENCRYPTED PRIVATE KEY {s}"),
            ),
            (
                format!("{s} BEGIN SSH2 PRIVATE KEY {s}"),
                format!("{s} END SSH2 PRIVATE KEY {s}"),
            ),
        ];
        for (header, footer) in blocks {
            let masked =
                crate::review::secrets::mask_secrets(&format!("{header}\nAAAA\n{footer}\nAFTER\n"));
            assert!(
                masked.contains("AFTER"),
                "the mask ran past {footer}: {masked}"
            );
            assert!(!masked.contains("AAAA"), "{masked}");
            assert!(holds_armor_end(&footer), "the gate misses {footer}");
        }
    }
    #[tokio::test]
    async fn deadline_precedes_key_scan() {
        // MAGI FIX (MS3 §6 loop12): the whole-code key-block scan ran before the deadline check, so
        // a spent clock still paid for it (and said `secret`); the clock is read first.
        let c = MockDecisionClient::new();
        let mut g = Gate::new(&c, chain(), 61_000);
        let key = format!("{}\nQQQQ\n", concat!("-----", "BEGIN ", "PRIVATE KEY-----"));
        let past = Instant::now();
        let mut f = finding(Severity::Warning, Some(1));
        f.file = Some("a.rs".into());
        let out = g
            .decide(&f, Some(key.as_str()), Some(past), None, &mut Vec::new())
            .await;
        assert_eq!(out.cause, Some(CAUSE_TIMEOUT), "{:?}", out.cause);
        assert!(g.last_scan.is_none(), "no scan after the deadline");
    }

    #[test]
    fn scan_memo_is_keyed_by_digest() {
        // MAGI FIX (MS3 §6 loop12): the memo compared (and held a copy of) the whole file. Equal
        // code scans once however many findings ask; another file scans again; the copy is gone.
        let c = MockDecisionClient::new();
        let mut g = Gate::new(&c, chain(), 61_000);
        let (a, b) = ("fn a() {}\n".repeat(1_000), "fn b() {}\n".repeat(1_000));
        for _ in 0..3 {
            g.key_block_scan(&a);
        }
        assert_eq!(g.scans, 1, "one scan for one file");
        g.key_block_scan(&b);
        g.key_block_scan(&b);
        assert_eq!(g.scans, 2, "another file scans once");
        // Type-level pin: the memo is a 32-byte digest and a scan, never a copy of the file.
        let _: &Option<([u8; 32], KeyBlockScan)> = &g.last_scan;
    }

    #[test]
    fn the_tracked_block_intervals_are_scanned_once_per_file() {
        // [MAGI S6 MS3 loop15, R6b2] every finding rescanned the whole file for its tracked
        // blocks; the intervals are part of the per-file scan now, and a finding costs O(blocks).
        let c = MockDecisionClient::new();
        let mut g = Gate::new(&c, chain(), 61_000);
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let code = format!("fn a() {{}}\n{begin}\nQQQQ\n{end}\nfn b() {{}}\n{begin}\nRRRR\n");
        let scan = g.key_block_scan(&code);
        assert_eq!(scan.blocks, vec![(2, 4), (6, u32::MAX)]);
        let found = |line: u32| window_overlaps_key_block(line, &scan);
        assert!(
            found(1) && found(5) && found(7),
            "a window within reach of a block"
        );
        let far = format!("{}{begin}\nQQQQ\n{end}\n", "fn x() {}\n".repeat(200));
        let scan = g.key_block_scan(&far);
        assert!(!window_overlaps_key_block(1, &scan));
        assert!(window_overlaps_key_block(201 + 1, &scan));
    }

    #[test]
    fn the_window_shown_and_the_window_checked_share_their_bounds() {
        // [MAGI S6 MS3 loop16, R6b2] `numbered_window` and `window_meets_block` each computed the
        // window: a block whose first line is the last shown line, or whose last line is the first
        // shown, must be both shown and found, one line further neither.
        for line in [1u32, 5, GATE_CONTEXT_LINES, GATE_CONTEXT_LINES + 1, 200] {
            let (first, last) = window_bounds(line);
            let code: String = (1..=line + GATE_CONTEXT_LINES + 3)
                .map(|n| format!("l{n}\n"))
                .collect();
            let shown = numbered_window(&code, line);
            let shown_has = |n: u32| shown.lines().any(|l| l.starts_with(&format!("{n}: ")));
            assert!(shown_has(first) && shown_has(last), "line {line}");
            assert!(!shown_has(last + 1), "line {line}");
            assert!(first == 1 || !shown_has(first - 1), "line {line}");
            assert!(window_meets_block(&[(last, last + 5)], line), "line {line}");
            assert!(!window_meets_block(&[(last + 1, last + 5)], line));
            assert!(window_meets_block(&[(1, first)], line));
            if first > 1 {
                assert!(!window_meets_block(&[(1, first - 1)], line));
            }
        }
        assert_eq!(
            window_bounds(GATE_CONTEXT_LINES),
            (1, 2 * GATE_CONTEXT_LINES)
        );
    }

    #[test]
    fn a_header_read_only_across_lines_withholds_a_window_far_from_it() {
        // [MAGI S6 MS3 loop16, R6b2] The `untracked` term of the per-file scan: when the detector
        // finds a block only across lines, no line opens one and `blocks` is empty, so without
        // that term every window would be judged and the block's body sent.
        let opens = |t: &str| t.contains("SPLIT\nHEADER");
        let tail: String = (0..200).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let code = format!("{tail}SPLIT\nHEADER\nbody\n");
        let scan = KeyBlockScan::of(&code, &opens);
        assert!(scan.holds && scan.untracked && scan.blocks.is_empty());
        assert!(window_overlaps_key_block(1, &scan), "whole code counts");
        let clean = KeyBlockScan::of(&tail, &opens);
        assert!(!window_overlaps_key_block(1, &clean));
    }

    #[test]
    fn scan_memo_any_order() {
        // [MAGI S6 MS3 loop10, R6b Melchior] The memo is keyed by the code itself, so findings
        // alternating between files rescan (never reuse another file's scan).
        let c = MockDecisionClient::new();
        let mut g = Gate::new(&c, chain(), 61_000);
        let key = format!("{}\nQQQQ\n", concat!("-----", "BEGIN ", "PRIVATE KEY-----"));
        for _ in 0..2 {
            assert!(g.key_block_scan(&key).holds);
            assert!(!g.key_block_scan("fn a() {}\n").holds);
        }
    }

    #[test]
    fn armor_end_must_open_its_line() {
        // [MAGI S6 MS3 loop10, R6b Caspar, Melchior] `----- END` matched the SSH2 opener one dash
        // in, and an END quoted inside a block (`Comment: …`, `x-----END`) closed it early, so
        // the body after it was sent. An END line closes only when it opens its line.
        let d = ARMOR_DASHES;
        for not_end in [
            format!("{d} END PRIVATE KEY{d}"),
            format!("Comment: {d}END PGP PRIVATE KEY BLOCK{d}"),
            format!("x{d}-END RSA PRIVATE KEY{d}"),
        ] {
            assert!(!holds_armor_end(&not_end), "{not_end}");
            let body: String = (0..60).map(|i| format!("b{i}\n")).collect();
            let code = format!("OPEN\n{not_end}\n{body}");
            assert!(
                window_overlaps_block_of(&code, 55, &|t| t.contains("OPEN")),
                "{not_end}"
            );
        }
    }

    #[test]
    fn a_header_read_only_across_lines_is_not_hidden_by_another_tracked_block() {
        // [MAGI S6 MS3 loop7, R6b Melchior, Caspar] With one block opened and closed by lines,
        // the whole-code fallback was off, so a second header the detector reads only across
        // lines (none does today; a detector that does is given here) let its body be sent.
        let opens = |t: &str| t.contains("OPEN") || t.contains("SPLIT\nHEADER");
        let tail: String = (0..100).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let code = format!("OPEN\nbody\n{ARMOR_DASHES}END x\n{tail}SPLIT\nHEADER\nbody2\n");
        let header = 104;
        assert!(window_overlaps_block_of(&code, header, &opens));
        assert!(
            !window_overlaps_block_of(&tail, 50, &opens),
            "no block at all"
        );
    }

    #[tokio::test]
    async fn a_key_block_far_from_the_window_does_not_withhold_the_finding() {
        // [MAGI S6 MS3 loop6, R6b Melchior, Caspar] The whole-code check withheld every finding
        // of a file holding a block. Only a finding whose ±40-line window overlaps the block's
        // lines is withheld; one far from it is judged.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .times(1)
            .returning(|r| answer(r.model().unwrap_or(""), "real_defect", 0.9));
        let mut g = Gate::new(&c, chain(), 1_000_000);
        let begin = concat!("-----", "BEGIN ", "RSA PRIVATE KEY-----");
        let end = concat!("-----", "END ", "RSA PRIVATE KEY-----");
        let tail: String = (0..200).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let code = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}\n{tail}");
        let far = g
            .judge(&finding(Severity::Warning, Some(150)), Some(&code), None)
            .await;
        assert_eq!(far.state, GateState::Kept, "{far:?}");
        let near = g
            .judge(&finding(Severity::Warning, Some(40)), Some(&code), None)
            .await;
        assert_eq!(
            (near.state, near.cause),
            (GateState::Unfiltered, Some("secret"))
        );
    }

    #[tokio::test]
    async fn a_stopped_gate_keeps_its_cause_for_a_finding_near_a_key_block() {
        // [MAGI S6 MS3 loop6, R6b Caspar] After a 401 every later finding is `unauthorized` with
        // no call (C-6); the key-block check ran first and said `secret` instead.
        let mut c = MockDecisionClient::new();
        c.expect_decide().times(1).returning(|_| {
            Err(DecisionError::Unauthorized {
                status: 401,
                trace: Trace::default(),
            })
        });
        let mut g = Gate::new(&c, chain(), 61_000);
        g.judge(&finding(Severity::Warning, Some(1)), Some("x\nx\n"), None)
            .await;
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let code = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let v = g
            .judge(&finding(Severity::Warning, Some(1)), Some(&code), None)
            .await;
        assert_eq!(v.cause, Some("unauthorized"));
    }

    #[test]
    fn the_gate_state_is_keyed_by_the_gates_own_literals_only() {
        // [MAGI S6 MS3 loop6, R6b Balthasar] `masked_strings` masks values, not keys, on the
        // invariant that every key is one of the gate's literals; this pins that invariant
        // (a key built from model or code text would show here) and the net behind it: a state
        // whose key holds a secret is withheld, never sent with the key raw.
        fn keys(value: &Value, out: &mut BTreeSet<String>) {
            match value {
                Value::Object(fields) => {
                    for (k, v) in fields {
                        out.insert(k.clone());
                        keys(v, out);
                    }
                }
                Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
                _ => {}
            }
        }
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec![token.into()];
        f.details = vec![token.into()];
        f.file = Some(format!("src/{token}.rs"));
        let mut seen = BTreeSet::new();
        keys(
            &gate_state_of(&f, 0, &format!("let t = \"{token}\";\n")),
            &mut seen,
        );
        let mut clean = BTreeSet::new();
        keys(
            &gate_state_of(&finding(Severity::Info, Some(1)), 0, "x\n"),
            &mut clean,
        );
        assert_eq!(
            seen, clean,
            "the keys never depend on the finding or the code"
        );
        assert!(seen.iter().all(|k| find_secret(k).is_none()), "{seen:?}");
        let c = MockDecisionClient::new();
        let g = Gate::new(&c, chain(), 1_000_000);
        assert_eq!(
            g.sendable(serde_json::json!({ token: "v" })),
            Err(CAUSE_SECRET)
        );
    }

    #[test]
    fn a_state_whose_own_text_quotes_a_key_block_is_never_sent() {
        // [MAGI S6 MS3 loop6, R6b Melchior] The state-level branch of `sendable` alone: the code
        // is clean, the block is in the finding's own detail.
        let c = MockDecisionClient::new();
        let g = Gate::new(&c, chain(), 1_000_000);
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let mut f = finding(Severity::Warning, Some(1));
        f.details = vec![format!(
            "the key is {begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w"
        )];
        let state = gate_state_of(&f, 0, "fn a() {}\n");
        assert_eq!(g.sendable(state), Err(CAUSE_SECRET));
    }

    #[tokio::test]
    async fn a_merged_claim_carrying_a_one_line_secret_is_asked_masked_with_the_others() {
        // [MAGI S6 loop 10, C1-c2] one reworded claim with a secret pattern made the whole merged
        // finding `unfiltered` with no question asked and no `not every claim judged`.
        // [MAGI S6 MS3 loop2] Masked now, so the claim is asked. [loop13] A masked claim is
        // never `filtered`: a confident `false_positive` leaves as `low_confidence`.
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec!["a".into(), "b".into()];
        f.details = vec![
            "token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaks".into(), // allow-secret-scan
            "db".into(),
        ];
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|r| answer(r.model().unwrap_or(""), "false_positive", 0.9));
        let mut g = Gate::new(&c, chain(), 61_000);
        let v = g.judge(&f, Some("x\nx\nx\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("masked code"),
                vec!["masked code", "not every claim judged"]
            )
        );
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
    async fn masked_state_is_never_filtered() {
        // [MAGI S6 MS3 loop13, R6b] a masked code window shows the gate less than the file holds
        // (the secret is replaced), so a confident `false_positive` on it is not trusted: the
        // finding stays in the main list as `low_confidence`.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let masked = format!(
            "let k = \"{}github-token>\";\n",
            crate::review::secrets::REDACTED_OPEN
        );
        let v = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some(masked.as_str()),
                None,
            )
            .await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("masked code"))
        );
        let plain = g
            .judge(
                &finding(Severity::Warning, Some(1)),
                Some("let k = 1;\n"),
                None,
            )
            .await;
        assert_eq!(plain.state, GateState::Filtered);
    }

    #[tokio::test]
    async fn masked_code_stays_named_beside_an_ambiguous_side() {
        // [MAGI S6 MS3 loop16, R6b2] `guard.or_else(masked)` dropped the masked-code cause as soon
        // as another guard applied, so the report named one protection of two.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let masked = format!(
            "let k = \"{}github-token>\";\n",
            crate::review::secrets::REDACTED_OPEN
        );
        let v = g
            .judge_ambiguous_side(&finding(Severity::Warning, Some(1)), masked.as_str(), None)
            .await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("ambiguous diff side"),
                vec!["ambiguous diff side", "masked code"]
            )
        );
    }

    #[tokio::test]
    async fn a_quoted_marker_is_never_filtered() {
        // MAGI FIX (MS3 §6 loop14): the declared false trigger of the masked-code guard: clean
        // code, but the finding's own title quotes the mask marker.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.99));
        let mut g = Gate::new(&c, chain(), 61_000);
        let mut f = finding(Severity::Warning, Some(1));
        f.titles = vec![format!(
            "the value {}github-token> is fine",
            crate::review::secrets::REDACTED_OPEN
        )];
        let v = g.judge(&f, Some("let k = 1;\n"), None).await;
        assert_eq!(
            (v.state, v.cause),
            (GateState::LowConfidence, Some("masked code"))
        );
    }

    #[tokio::test]
    async fn a_critical_with_masked_code_carries_both_causes() {
        // [MAGI S6 MS3 loop15, R6b2 Melchior/Caspar] `critical never filtered` replaced the
        // guard the state itself raised (a mask marker): both are named, the critical one first.
        let mut c = MockDecisionClient::new();
        c.expect_decide()
            .returning(|_| answer("clef:cloud", "false_positive", 0.60));
        let mut g = Gate::new(&c, chain(), 61_000);
        let mut f = finding(Severity::Critical, Some(1));
        f.titles = vec![format!(
            "the value {}github-token> is fine",
            crate::review::secrets::REDACTED_OPEN
        )];
        let v = g.judge(&f, Some("let k = 1;\n"), None).await;
        assert_eq!(
            (v.state, v.cause, v.causes),
            (
                GateState::LowConfidence,
                Some("critical never filtered"),
                vec!["critical never filtered", "masked code"]
            )
        );
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
