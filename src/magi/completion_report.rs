// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! Per-attempt completion telemetry, composed for output.
//!
//! # What a record answers that a counter cannot
//!
//! magi-core records one [`CompletionRecord`] **per completion ATTEMPT**, not per kept verdict.
//! With rotation a single seat may spend several attempts on several models, and *which model was
//! cut* is the question that decides what leaves the pool — a per-seat total cannot be
//! disaggregated back into that attribution, while the totals are trivially derivable from these
//! records. So the array under a seat label is the attempt series, in the order magi-core
//! recorded it, and its length is a fact about the run.
//!
//! # Redaction happens HERE, at composition, not at the call sites
//!
//! `CompletionRecord::model` is **composed by another crate** — it is whatever string the seat's
//! provider reported as having served the attempt, and a rotation resolves it from configuration
//! magi-rs does not own. That is the same shape that produced five separate leak findings in an
//! earlier milestone, every one of them green against all seven build gates. Redacting once,
//! where the value is turned into output, means the output surfaces cannot get it wrong.
//!
//! **`redact_url` is NOT interchangeable with `redact_foreign_text` here, and the direction of
//! the mistake is the expensive one.** A model tag carries no authority: `redact_url` treats its
//! whole input as a URL and would collapse `glm-5.2:cloud` to `***`, destroying every record's
//! most actionable field while leaking nothing — a silent, total loss of the diagnostic.
//! `redact_foreign_text` treats the tag as prose that may *embed* a URL, which is the only shape
//! in which a credential can reach this field at all.
//!
//! # Why this takes the completions MAP and not the `MagiReport`
//!
//! `MagiReport` and `CompletionRecord` are both `#[non_exhaustive]`, so neither can be built with
//! a struct literal from outside magi-core. Taking the map keeps this module testable through the
//! one door that does exist — `serde` deserialization, whose derive lives inside the defining
//! crate and can therefore reach private fields — and it is also the smaller dependency: nothing
//! here needs a report, only its completions.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::collections::BTreeMap;

use magi_core::provider::{FinishReason, ReasoningControl, ReasoningState};
use magi_core::reporting::CompletionRecord;
use magi_core::rotation::{AgentRotation, RotationEvent, RotationKind};
use magi_core::schema::AgentName;
use serde_json::{json, Map, Value};

use crate::magi::seat_label;
use crate::redact::{foreign_serde_label, redact_foreign_text};

/// Renders the per-attempt completion telemetry as JSON.
///
/// One key per seat that has records, holding the seat's attempts **in magi-core's order**, one
/// object per attempt with exactly seven keys: `model`, `cap`, `finish`, `completion_tokens`,
/// `prompt_tokens`, `reasoning` and `control`.
///
/// # An unreported measurement renders as `null`, never as a word
///
/// `finish`, `completion_tokens` and `prompt_tokens` are all optional at the source, because a
/// backend may report none of them. Each keeps that distinction into the JSON: a consumer must be
/// able to tell *"the backend did not say why the model stopped"* from *"the backend said it
/// stopped normally"*. Substituting a default — `"stop"`, or a zero — asserts a measurement that
/// nobody took, which is exactly the confusion that made a scraping-by output cap look healthy
/// until it started failing.
///
/// # `reasoning` and `control`: a state, not a number
///
/// [`CompletionRecord`] also carries `reasoning: ReasoningState` and `control:
/// ReasoningControl`. Both now have a consumer — MAGI-Claude's reasoning-budget instrumentation
/// (E-E) needs, per attempt, what the caller asked the reasoning channel to do and what that
/// channel measured, so a cut attempt (`finish: length` with an exhausted budget) can be told
/// apart from a completion the backend simply spent on tokens that never came back. `reasoning`
/// is rendered through `reasoning_value` as magi-core's own serde form of [`ReasoningState`],
/// **never flattened to a character or token count**: `"NotMeasured"` (nobody looked),
/// `{"Measured": {"chars": N, "text": null}}` (the channel was read) and `{"Unsupported": {…}}`
/// (the backend cannot honour the control) stay distinguishable states, and every state's `text`
/// is forced to `null` — model text never enters this envelope. `control` is rendered through
/// `control_label` as the kebab-case tag [`ReasoningControl`] serializes (`"default"`,
/// `"disabled"`, `"enabled"`), so a consumer can tell an honoured `disabled` (a measured zero)
/// from an ignored one (`Unsupported` with a non-zero count) apart, which a bare number could
/// not.
///
/// # Arguments
/// * `completions` - magi-core's per-seat attempt records, exactly as the report carries them.
///
/// # Returns
///
/// A JSON object keyed by lowercase seat label, one key per seat magi-core recorded. The two
/// empty cases are different inputs and render differently: a seat ABSENT from the map contributes
/// no key, while a seat PRESENT with no attempts renders an empty array. Neither is synthesized —
/// the object mirrors what the report carries.
///
/// # Complexity
///
/// `O(seats x records)` for the traversal, plus `O(k)` per attempt for walking the serialized
/// `reasoning` state (`k` is its own small, fixed-depth JSON structure — a handful of fields at
/// most). The map is a trio and the function runs **once per consult**, so the whole traversal is
/// a handful of items on a path that already spent seconds in HTTP. An index or a precomputed
/// lookup would buy nothing measurable and would be over-engineering against a workload this
/// small.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::completion_report::render_completions;
/// let json = render_completions(&BTreeMap::new());
/// assert!(json.as_object().is_some_and(|seats| seats.is_empty()));
/// ```
#[must_use]
pub fn render_completions(completions: &BTreeMap<AgentName, Vec<CompletionRecord>>) -> Value {
    let seats: Map<String, Value> = completions
        .iter()
        .map(|(agent, records)| {
            let attempts: Vec<Value> = records
                .iter()
                .map(|record| {
                    // The ONLY foreign-composed string here, and the reason this module does its
                    // own redaction. Bound rather than inlined: `SafeErrorText` deliberately does
                    // NOT implement `Serialize`, so reaching the JSON has to go through an
                    // explicit `as_str` — which is the point. A type that serialized itself would
                    // let an unredacted `String` take its place with nothing to notice.
                    let model = redact_foreign_text(&record.model);
                    json!({
                        "model": model.as_str(),
                        "cap": record.cap,
                        "finish": finish_label(record.finish.as_ref()),
                        "completion_tokens": record.completion_tokens,
                        "prompt_tokens": record.prompt_tokens,
                        "reasoning": reasoning_value(&record.reasoning),
                        "control": control_label(&record.control),
                    })
                })
                .collect();
            (seat_label(*agent), Value::Array(attempts))
        })
        .collect();

    Value::Object(seats)
}

/// Stable JSON value for a finish reason, DERIVED from the crate.
///
/// # Why the serde form and not a hand-written match
///
/// [`FinishReason`] is `#[non_exhaustive]` and carries an open `Other(String)` variant, so a
/// hand-written match here would need a wildcard — and a wildcard is what turns a value the
/// backend actually sent into a label magi-rs invented. Its `Serialize` impl lives inside the
/// defining crate, where the enum is still closed, so magi-rs inherits the wire vocabulary
/// instead of maintaining a second copy that drifts.
///
/// # Arguments
/// * `finish` - the reason magi-core reported, or `None` when the backend did not say.
///
/// # Returns
///
/// The `snake_case` wire label as a JSON string, or [`Value::Null`] when nothing was reported.
/// **`None` is the only path that yields null**: the unreachable serialization-failure branch
/// falls back to the variant's debug form rather than null, so "not reported" stays a value only
/// an absent reason can produce.
fn finish_label(finish: Option<&FinishReason>) -> Value {
    let Some(reason) = finish else {
        return Value::Null;
    };
    match serde_json::to_value(reason) {
        // REDACTED, and the reason is not defensive. `FinishReason` is `#[non_exhaustive]` and its
        // fourth variant is `Other(String)`, which carries whatever the backend put on the wire —
        // so this is NOT the closed set of three literals it looks like, and an un-redacted branch
        // would let a foreign string reach the run JSON. `redact_foreign_text` rather than
        // `redact_url`: the value is prose that MAY embed a URL, and treating a bare `stop` as a
        // URL would collapse it to `***`.
        Ok(Value::String(label)) => Value::String(redact_foreign_text(&label).as_str().to_string()),
        // Unreachable: the impl writes a plain string and cannot fail into `serde_json`. Handled
        // rather than unwrapped because a panic here would take down a whole report over one
        // diagnostic field, and it must NOT degrade to null — that value is spoken for.
        _ => Value::String(
            redact_foreign_text(&format!("{reason:?}"))
                .as_str()
                .to_string(),
        ),
    }
}

/// Renders a reasoning channel measurement as magi-core's own serde form of it, with the model
/// text stripped and every embedded string redacted.
///
/// # Why the JSON is walked structurally instead of matching on the enum
///
/// [`ReasoningState`] is `#[non_exhaustive]`, so a hand-written match here would need a wildcard
/// — the same trap `finish_label` avoids for [`FinishReason`]. Serializing first and then
/// transforming the resulting [`Value`] sidesteps it entirely: the walk only ever asks "is this
/// key named `text`?" and "is this a string?", so it applies unchanged to a future variant this
/// crate does not know yet, without inventing a label for it.
///
/// # Arguments
/// * `reasoning` - the measurement magi-core recorded for one completion attempt.
///
/// # Returns
///
/// The serde shape of `reasoning`, with two transformations and only those: every `text` field
/// becomes [`Value::Null`] (model text never enters the envelope), and every string in the
/// resulting structure — today only `backend`, plus a unit-variant tag such as `"NotMeasured"`,
/// which is the identity under redaction — passes through [`redact_foreign_text`]. The
/// unreachable serialization-failure branch falls back to a redacted form of magi-core's own
/// `Debug` for [`ReasoningState`], which elides the trace field itself, and never to
/// [`Value::Null`] — that value is reserved for "no measurement was reported" and must not be
/// produced by a path that did measure something.
fn reasoning_value(reasoning: &ReasoningState) -> Value {
    match serde_json::to_value(reasoning) {
        Ok(value) => redact_reasoning_json(value),
        // Unreachable: the derived impl writes plain JSON and cannot fail into `serde_json`.
        // Handled rather than unwrapped because a panic here would take down a whole report
        // over one telemetry field. `ReasoningState`'s own `Debug` elides the trace, so this
        // fallback can never leak model text even though it is not the field-by-field walk
        // above.
        _ => Value::String(
            redact_foreign_text(&format!("{reasoning:?}"))
                .as_str()
                .to_string(),
        ),
    }
}

/// Structural transform applied to a serialized [`ReasoningState`]: null every `text` field,
/// redact every string.
///
/// # Arguments
/// * `value` - the raw [`serde_json::to_value`] output for a [`ReasoningState`].
///
/// # Returns
///
/// The same shape, recursively: an object keeps its keys with `text` forced to
/// [`Value::Null`] and every other value walked the same way; a string is redacted with
/// [`redact_foreign_text`]; every other value passes through unchanged.
fn redact_reasoning_json(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, val)| {
                    let val = if key == "text" {
                        Value::Null
                    } else {
                        redact_reasoning_json(val)
                    };
                    (key, val)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(redact_reasoning_json).collect()),
        Value::String(s) => Value::String(redact_foreign_text(&s).as_str().to_string()),
        other => other,
    }
}

/// Stable JSON value for the reasoning control a completion attempt ran under, DERIVED from the
/// crate exactly as `finish_label` derives a finish reason.
///
/// # Arguments
/// * `control` - the control magi-core recorded as sent for one completion attempt.
///
/// # Returns
///
/// The kebab-case wire tag (`"default"`, `"disabled"`, `"enabled"`) [`ReasoningControl`]
/// serializes, via [`foreign_serde_label`] — the same helper `rotation_report::cause_label`
/// uses — so a future variant this crate does not know yet still renders its own tag rather
/// than a wildcard label this crate invented.
fn control_label(control: &ReasoningControl) -> Value {
    Value::String(
        foreign_serde_label(control, || format!("{control:?}"))
            .as_str()
            .to_string(),
    )
}

/// Prefix of magi-core's `ProviderError::EmptyCompletion` `Display` (magi-core 4.2.0,
/// `error.rs:418`). A seat that runs out of rotations on an empty completion carries it inside
/// its `failed_agents` cause, which is the only place that fact survives for the seat's LAST
/// attempt: the record itself cannot say the content was empty.
pub const EMPTY_COMPLETION_MARKER: &str = "empty completion: ";

/// Why an attempt counts as cut (REQ-EE-2, REQ-EE-4, REQ-EE-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutCause {
    /// The backend reported `finish == length`: the output budget ran out.
    Length,
    /// The attempt came back with empty content under any other finish.
    EmptyContent,
}

impl CutCause {
    /// Stable label for the log line: `"length"` or `"empty_content"`.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::EmptyContent => "empty_content",
        }
    }
}

/// One cut attempt, located in magi-core's per-seat attempt series.
#[derive(Debug, Clone, Copy)]
pub struct CutAttempt<'a> {
    /// The seat the attempt belongs to.
    pub seat: AgentName,
    /// 0-based index into `completions[seat]` — the same position the consult JSON renders.
    pub index: usize,
    /// The record itself, borrowed from the report.
    pub record: &'a CompletionRecord,
    /// Why it counts as cut. `Length` wins when both hold.
    pub cause: CutCause,
}

/// THE cut-attempt predicate, defined once (spec §0: shared by REQ-EE-2, REQ-EE-4, REQ-EE-7).
///
/// An attempt is cut when its record says `finish == length`, or when it is the attempt that
/// came back EMPTY — which the record alone cannot say (magi-core 4.2.0 builds an empty
/// attempt's record exactly like a successful one), so it is located structurally: the LAST
/// record of a model's run is the empty one when the hop that left that model has kind
/// `EmptyCompletion`, or, for the seat's final run, when its `failed_agents` cause contains
/// [`EMPTY_COMPLETION_MARKER`]. A timed-out attempt (no finish, nothing measured, spec §1
/// criterion 3) is never a cut.
///
/// Returns the cut attempts in `completions`' seat order, then attempt order. `O(records +
/// hops)`: a trio, once per consult.
#[must_use]
pub fn cut_attempts<'a>(
    completions: &'a BTreeMap<AgentName, Vec<CompletionRecord>>,
    rotations: &BTreeMap<AgentName, AgentRotation>,
    failed_agents: &BTreeMap<AgentName, String>,
) -> Vec<CutAttempt<'a>> {
    let mut cuts = Vec::new();
    for (seat, records) in completions {
        let hops: &[RotationEvent] = rotations
            .get(seat)
            .map_or(&[], |rotation| rotation.chain.as_slice());
        let lost_empty = failed_agents
            .get(seat)
            .is_some_and(|cause| cause.contains(EMPTY_COMPLETION_MARKER));
        // `run` is the 0-based index of the contiguous run of one model (magi-core never
        // re-seats a model on the same seat, so each run maps to one hop, in order).
        let mut run = 0usize;
        for (index, record) in records.iter().enumerate() {
            let next = records.get(index + 1);
            let last_of_run = next.is_none_or(|n| n.model != record.model);
            let final_run = next.is_none();
            let cause = if matches!(record.finish, Some(FinishReason::Length)) {
                Some(CutCause::Length)
            } else if last_of_run
                && (hops
                    .get(run)
                    .is_some_and(|hop| hop.kind() == RotationKind::EmptyCompletion)
                    || (final_run && lost_empty))
            {
                Some(CutCause::EmptyContent)
            } else {
                None
            };
            if let Some(cause) = cause {
                cuts.push(CutAttempt {
                    seat: *seat,
                    index,
                    record,
                    cause,
                });
            }
            if last_of_run {
                run += 1;
            }
        }
    }
    cuts
}

/// Unit tests for the completion telemetry composition.
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a [`CompletionRecord`] through **deserialization**, the only door magi-core leaves
    /// open: the type is `#[non_exhaustive]`, so a struct literal does not compile from here,
    /// while `serde`'s derive lives inside the defining crate and can reach its fields.
    fn record(json: Value) -> CompletionRecord {
        serde_json::from_value(json).expect("the fixture must match magi-core's shape")
    }

    /// One attempt served by `model`, stopping for `finish`, having spent `completion_tokens`.
    ///
    /// `finish` and `completion_tokens` are taken as raw [`Value`]s so a test can express the
    /// absent case — `Value::Null` — which is the distinction this module exists to preserve.
    fn attempt(model: &str, finish: Value, completion_tokens: Value) -> CompletionRecord {
        record(json!({
            "model": model,
            "cap": 16_384,
            "finish": finish,
            "completion_tokens": completion_tokens,
            "prompt_tokens": 1_200,
            "reasoning": "NotMeasured",
        }))
    }

    /// A single seat holding `records`, the shape every assertion below reads.
    fn seat(
        agent: AgentName,
        records: Vec<CompletionRecord>,
    ) -> BTreeMap<AgentName, Vec<CompletionRecord>> {
        let mut map = BTreeMap::new();
        map.insert(agent, records);
        map
    }

    /// The distinction the whole record exists for: a reply cut by the output budget and a reply
    /// the model genuinely had nothing more to add to both arrive as short text, and only `finish`
    /// separates them. Their remedies are opposite — raise the cap, or do not — so a consumer that
    /// cannot tell them apart will fix the wrong one.
    #[test]
    fn a_truncated_completion_is_distinguishable_from_a_genuinely_empty_one() {
        let map = seat(
            AgentName::Caspar,
            vec![
                attempt("glm-5.2:cloud", json!("length"), json!(16_384)),
                attempt("glm-5.2:cloud", json!("stop"), json!(3)),
            ],
        );

        let json = render_completions(&map);
        let truncated = &json["caspar"][0];
        let empty = &json["caspar"][1];
        assert_eq!(
            truncated["finish"], "length",
            "the budget ran out: the remedy is a larger cap"
        );
        assert_eq!(
            empty["finish"], "stop",
            "the model ended on its own terms: raising the cap would change nothing"
        );
        assert_ne!(
            truncated["finish"], empty["finish"],
            "collapsing these two is what makes a scraping-by cap look healthy"
        );
        assert_eq!(truncated["completion_tokens"], 16_384);
        assert_eq!(empty["completion_tokens"], 3);
    }

    /// One entry per ATTEMPT, not per kept completion. A seat that was cut twice before a third
    /// model answered has three records, and with rotation those may be three different models —
    /// which model was cut is the attribution that decides what leaves the pool, and it is not
    /// derivable from a per-seat total.
    #[test]
    fn every_attempt_renders_its_own_record_including_the_ones_that_were_discarded() {
        let map = seat(
            AgentName::Balthasar,
            vec![
                attempt("qwen3.5:397b-cloud", json!("length"), json!(16_384)),
                attempt("gpt-oss:120b-cloud", json!("length"), json!(16_384)),
                attempt("deepseek-v4-pro:cloud", json!("stop"), json!(900)),
            ],
        );

        let json = render_completions(&map);
        let attempts = json["balthasar"]
            .as_array()
            .expect("the seat holds an array");
        assert_eq!(
            attempts.len(),
            3,
            "three attempts must render three records, not one for the verdict that survived"
        );
        assert_eq!(attempts[0]["model"], "qwen3.5:397b-cloud");
        assert_eq!(attempts[1]["model"], "gpt-oss:120b-cloud");
        assert_eq!(
            attempts[2]["model"], "deepseek-v4-pro:cloud",
            "the order magi-core recorded is the attempt series and must survive"
        );
    }

    /// A backend that never said why the model stopped renders `null`, never a word. Substituting
    /// `"stop"` would assert that the turn ended on the backend's terms, which nobody observed —
    /// and a consumer counting clean stops would count this one among them.
    #[test]
    fn an_unreported_finish_reason_renders_as_json_null_and_not_as_a_word() {
        let map = seat(
            AgentName::Melchior,
            vec![attempt("kimi-k2.6:cloud", Value::Null, Value::Null)],
        );

        let json = render_completions(&map);
        let entry = &json["melchior"][0];
        assert!(
            entry["finish"].is_null(),
            "an unreported reason must stay absent, got {}",
            entry["finish"]
        );
        assert!(
            !entry["finish"].is_string(),
            "no word may stand in for a measurement nobody took"
        );
        assert!(
            entry["completion_tokens"].is_null(),
            "an uncounted total is absent too, never a zero that reads as a real count"
        );
        assert_eq!(
            entry["prompt_tokens"], 1_200,
            "absence is decided per field: the one the backend DID count must survive, or the \
             assertion above would also pass against an output that renders nothing at all"
        );
    }

    /// A reported reason survives as its own wire label, so the null above means *absent* rather
    /// than *this module renders nothing*. Without this the test above passes against a function
    /// that emits null unconditionally.
    #[test]
    fn a_reported_finish_reason_renders_its_snake_case_wire_label() {
        for (wire, expected) in [("stop", "stop"), ("length", "length"), ("load", "load")] {
            let map = seat(
                AgentName::Melchior,
                vec![attempt("m", json!(wire), json!(1))],
            );
            let json = render_completions(&map);
            assert_eq!(
                json["melchior"][0]["finish"], expected,
                "{wire} renders wrong"
            );
        }
    }

    /// The model field is composed by another crate, so it is redacted at composition.
    ///
    /// The surviving-tag assertion is what makes this a guardian rather than a no-leak check that
    /// an EMPTY output would also satisfy — absence of a secret proves nothing on its own.
    /// Mutation-verified: dropping the redaction turns this test red on the canary.
    ///
    /// **It does NOT discriminate the wrong helper, and saying so is the point.** `redact_url`
    /// locates this input's userinfo and rewrites it in place, so the canary disappears and the
    /// tag survives here too — this test stays green under that mutation. What catches it is
    /// [`an_ordinary_model_tag_reaches_the_output_unchanged`], because `redact_url` collapses a
    /// plain `glm-5.2:cloud` (no authority to find) to `***`. That test is mutation-verified for
    /// exactly this substitution; keep the pair together.
    #[test]
    fn a_credential_embedded_in_a_model_string_is_redacted_without_losing_the_tag() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let map = seat(
            AgentName::Caspar,
            vec![attempt(
                &format!("glm-5.2:cloud via http://alice:{CANARY}@host:11434/v1"),
                json!("stop"),
                json!(7),
            )],
        );

        let json = render_completions(&map).to_string();
        assert!(!json.contains(CANARY), "the JSON surface leaked: {json}");
        assert!(
            json.contains("glm-5.2:cloud"),
            "redaction must remove the credential, NOT collapse the tag — that is what \
             `redact_url` would do to a model string: {json}"
        );
    }

    /// An ordinary tag reaches the output byte-for-byte. A model string is the most actionable
    /// field in the record, and mangling it would be as damaging as leaking.
    ///
    /// **This is the test that catches `redact_url` in place of `redact_foreign_text`**, the one
    /// substitution a reader is most likely to make: an ordinary tag has no authority to find, so
    /// `redact_url` treats the whole thing as unparseable and returns `***`. Mutation-verified —
    /// swapping the helper turns this red while the credential canary above stays green.
    #[test]
    fn an_ordinary_model_tag_reaches_the_output_unchanged() {
        let map = seat(
            AgentName::Melchior,
            vec![attempt("kimi-k2.6:cloud", json!("stop"), json!(42))],
        );

        let json = render_completions(&map);
        assert_eq!(json["melchior"][0]["model"], "kimi-k2.6:cloud");
    }

    /// One attempt with the telemetry `attempt("m", json!("length"), json!(16_384))` would carry,
    /// but with the given `reasoning` (magi-core's serde shape of `ReasoningState`, passed as raw
    /// JSON so a test can express every variant and a captured `text`) and the given `control`
    /// tag. Built through `record`, i.e. through magi-core's own `Deserialize`.
    fn attempt_with(reasoning: Value, control: &str) -> CompletionRecord {
        record(json!({
            "model": "m",
            "cap": 16_384,
            "finish": "length",
            "completion_tokens": 16_384,
            "prompt_tokens": 1_200,
            "reasoning": reasoning,
            "control": control,
        }))
    }

    /// The consult JSON's key set is a CONTRACT. It grew from five to seven in v0.20.0, and on
    /// purpose: `reasoning` and `control` now have a consumer (MAGI-Claude, E-E), which is the
    /// condition the old omission was waiting for. The next change must break this test the
    /// same way — deliberately, naming the key.
    #[test]
    fn the_attempt_object_carries_exactly_the_declared_keys() {
        let map = seat(
            AgentName::Caspar,
            vec![attempt("glm-5.2:cloud", json!("length"), json!(5))],
        );

        let json = render_completions(&map);
        let object = json["caspar"][0].as_object().expect("attempt is an object");
        let mut keys: Vec<String> = object.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "cap",
                "completion_tokens",
                "control",
                "finish",
                "model",
                "prompt_tokens",
                "reasoning"
            ],
            "a key was added or removed without updating the contract"
        );
    }

    /// S-2, JSON half: a cut reads as a cut, with its cause. The budget ran out
    /// (`completion_tokens == cap`, `finish: length`) and the reasoning channel says where it
    /// went. That pair is what tells "the model reasoned the whole budget away" from "the
    /// daemon spent tokens it did not return".
    #[test]
    fn a_cut_attempt_carries_its_measured_reasoning_and_the_control_it_ran_under() {
        let map = seat(
            AgentName::Caspar,
            vec![record(json!({
                "model": "glm-5.2:cloud",
                "cap": 16_384,
                "finish": "length",
                "completion_tokens": 16_384,
                "prompt_tokens": 14_000,
                "reasoning": { "Measured": { "chars": 81_920, "text": null } },
                "control": "default",
            }))],
        );

        let json = render_completions(&map);
        let entry = &json["caspar"][0];
        assert_eq!(entry["finish"], "length");
        assert_eq!(
            entry["completion_tokens"], entry["cap"],
            "the cut spent exactly the budget it was given"
        );
        assert_eq!(
            entry["reasoning"],
            json!({ "Measured": { "chars": 81_920, "text": null } }),
            "the state is kept as magi-core serializes it, never flattened to a number"
        );
        assert_eq!(entry["control"], "default");
    }

    /// `text` is ALWAYS null in the JSON, even when a trace was captured: model text never
    /// enters the envelope (spec §0, REQ-EE-1). The count must survive the nulling — a renderer
    /// that blanked the whole state would pass the no-leak assertion and lose the measurement.
    ///
    /// MUTATION (required): stop nulling `text` and this goes red with the canary visible.
    #[test]
    fn a_captured_reasoning_trace_never_reaches_the_json() {
        const TRACE: &str = "the model weighed c4n4ry-tr4ce for a long while";
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Melchior,
            vec![attempt_with(
                json!({ "Measured": { "chars": 48, "text": TRACE } }),
                "default",
            )],
        );
        map.insert(
            AgentName::Caspar,
            vec![attempt_with(
                json!({ "Unsupported": { "backend": "ollama", "chars": 48, "text": TRACE } }),
                "disabled",
            )],
        );

        let json = render_completions(&map);
        assert!(json["melchior"][0]["reasoning"]["Measured"]["text"].is_null());
        assert_eq!(
            json["melchior"][0]["reasoning"]["Measured"]["chars"], 48,
            "nulling the text must not blank the measurement"
        );
        assert!(json["caspar"][0]["reasoning"]["Unsupported"]["text"].is_null());
        assert_eq!(json["caspar"][0]["reasoning"]["Unsupported"]["chars"], 48);
        let rendered = json.to_string();
        assert!(
            !rendered.contains("c4n4ry-tr4ce"),
            "model text reached the envelope: {rendered}"
        );
    }

    /// S-3: no channel is not zero. `NotMeasured` and a measured zero are different facts —
    /// "nobody looked" against "somebody looked and the model did not reason" — and a consumer
    /// deciding whether reasoning ate the budget needs to tell them apart.
    #[test]
    fn an_unread_channel_renders_as_not_measured_and_a_measured_zero_as_zero() {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Melchior,
            vec![attempt_with(json!("NotMeasured"), "default")],
        );
        map.insert(
            AgentName::Caspar,
            vec![attempt_with(
                json!({ "Measured": { "chars": 0, "text": null } }),
                "default",
            )],
        );

        let json = render_completions(&map);
        assert_eq!(json["melchior"][0]["reasoning"], json!("NotMeasured"));
        assert_ne!(
            json["melchior"][0]["reasoning"],
            json!({ "Measured": { "chars": 0, "text": null } }),
            "an unread channel must never read as a measured zero"
        );
        assert_eq!(
            json["caspar"][0]["reasoning"],
            json!({ "Measured": { "chars": 0, "text": null } }),
            "the pair: a real zero keeps its shape, so the assertion above is not satisfied by a \
             renderer that writes NotMeasured for everything"
        );
    }

    /// S-4, record half: with `disabled`, a model that honours the switch reads as a measured
    /// zero and one that ignores it (the `gpt-oss:120b` case) reads as `Unsupported` with the
    /// count that came back anyway — never as a count that looks like the switch worked.
    #[test]
    fn a_disabled_control_reads_as_honoured_or_as_unsupported_never_as_a_count_that_worked() {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Melchior,
            vec![attempt_with(
                json!({ "Measured": { "chars": 0, "text": null } }),
                "disabled",
            )],
        );
        map.insert(
            AgentName::Balthasar,
            vec![attempt_with(
                json!({ "Unsupported": { "backend": "ollama", "chars": 40_000, "text": null } }),
                "disabled",
            )],
        );

        let json = render_completions(&map);
        assert_eq!(json["melchior"][0]["control"], "disabled");
        assert_eq!(
            json["melchior"][0]["reasoning"],
            json!({ "Measured": { "chars": 0, "text": null } })
        );
        assert_eq!(json["balthasar"][0]["control"], "disabled");
        assert_eq!(
            json["balthasar"][0]["reasoning"],
            json!({ "Unsupported": { "backend": "ollama", "chars": 40_000, "text": null } })
        );
        assert!(
            json["balthasar"][0]["reasoning"].get("Measured").is_none(),
            "an ignored switch must not read as a measurement under the control that was asked"
        );
    }

    /// An `Unsupported` channel that could not be read keeps its count ABSENT (`null`), which
    /// magi-core distinguishes from `Some(0)` on purpose (`provider.rs:465-478`).
    #[test]
    fn an_unreadable_unsupported_channel_keeps_its_count_absent() {
        let map = seat(
            AgentName::Caspar,
            vec![attempt_with(
                json!({ "Unsupported": { "backend": "anthropic", "chars": null, "text": null } }),
                "disabled",
            )],
        );

        let json = render_completions(&map);
        let state = &json["caspar"][0]["reasoning"]["Unsupported"];
        assert!(state["chars"].is_null(), "unreadable is not zero: {state}");
        assert_eq!(state["backend"], "anthropic");
    }

    /// `control` is the kebab-case tag magi-core serializes, for every variant it defines —
    /// derived, never a hand-written label that drifts.
    #[test]
    fn every_control_renders_as_the_kebab_case_tag_magi_core_serializes() {
        for tag in ["default", "disabled", "enabled"] {
            let map = seat(
                AgentName::Melchior,
                vec![attempt_with(json!("NotMeasured"), tag)],
            );
            assert_eq!(
                render_completions(&map)["melchior"][0]["control"],
                tag,
                "{tag} renders wrong"
            );
        }
    }

    /// `backend` is a foreign string (spec §0): redacted at composition, and the name
    /// survives. The surviving-name assertion is what makes this a guardian and not a no-leak
    /// check an empty output would also pass.
    ///
    /// MUTATION (required): drop the redaction of `backend` and this goes red on the canary.
    #[test]
    fn a_credential_embedded_in_a_backend_name_is_redacted_without_losing_the_name() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let map = seat(
            AgentName::Caspar,
            vec![attempt_with(
                json!({ "Unsupported": {
                    "backend": format!("ollama via http://alice:{CANARY}@host:11434"),
                    "chars": 7,
                    "text": null,
                } }),
                "disabled",
            )],
        );

        let json = render_completions(&map);
        let rendered = json.to_string();
        assert!(!rendered.contains(CANARY), "the backend leaked: {rendered}");
        assert!(
            json["caspar"][0]["reasoning"]["Unsupported"]["backend"]
                .as_str()
                .expect("a string")
                .starts_with("ollama"),
            "redaction must remove the credential, not the backend's name: {rendered}"
        );
    }

    /// An ordinary backend name reaches the output byte-for-byte.
    ///
    /// **This is the test that catches `redact_url` in place of `redact_foreign_text`**:
    /// `ollama` has no authority to find, so `redact_url` would return `***`.
    /// MUTATION (required): swap the helper and this goes red while the canary test above
    /// stays green — keep the pair together, as for `model`.
    #[test]
    fn an_ordinary_backend_name_reaches_the_output_unchanged() {
        let map = seat(
            AgentName::Balthasar,
            vec![attempt_with(
                json!({ "Unsupported": { "backend": "ollama", "chars": 3, "text": null } }),
                "disabled",
            )],
        );
        assert_eq!(
            render_completions(&map)["balthasar"][0]["reasoning"]["Unsupported"]["backend"],
            "ollama"
        );
    }

    /// Every seat that has records gets its own key, under the identity magi-core serializes.
    #[test]
    fn each_seat_renders_under_its_own_lowercase_label() {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Melchior,
            vec![attempt("a", json!("stop"), json!(1))],
        );
        map.insert(
            AgentName::Balthasar,
            vec![attempt("b", json!("stop"), json!(1))],
        );
        map.insert(
            AgentName::Caspar,
            vec![attempt("c", json!("stop"), json!(1))],
        );

        let json = render_completions(&map);
        let seats = json.as_object().expect("the render is an object");
        let mut labels: Vec<String> = seats.keys().cloned().collect();
        labels.sort();
        assert_eq!(labels, vec!["balthasar", "caspar", "melchior"]);
    }

    /// A seat magi-core recorded with no attempts renders an empty array rather than being
    /// dropped: the seat ran and produced no measurable attempt, which is not the same fact as
    /// the seat being absent from the report.
    #[test]
    fn a_seat_with_no_attempts_renders_an_empty_array() {
        let map = seat(AgentName::Melchior, Vec::new());

        let json = render_completions(&map);
        assert!(
            json["melchior"].as_array().is_some_and(Vec::is_empty),
            "the key must be present and its array empty"
        );
    }

    /// The declared cap travels with the attempt it applied to, because it is the number a
    /// consumer edits in response to a `length` finish.
    #[test]
    fn the_requested_output_cap_travels_with_the_attempt_it_applied_to() {
        let map = seat(
            AgentName::Caspar,
            vec![attempt("glm-5.2:cloud", json!("length"), json!(16_384))],
        );

        let json = render_completions(&map);
        assert_eq!(json["caspar"][0]["cap"], 16_384);
    }

    /// `FinishReason` is `#[non_exhaustive]` and its fourth variant is `Other(String)`, which
    /// carries whatever the backend put on the wire. The plan's redaction table called this field
    /// a closed set of three literals and that was wrong; a credential reaching the run JSON
    /// through a finish reason is the same leak class the v0.12.0 gate found five of.
    ///
    /// MUTATION (required): drop `redact_foreign_text` from `finish_label` and this goes red with
    /// the canary visible in the message.
    #[test]
    fn a_credential_embedded_in_an_unrecognised_finish_reason_is_redacted() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let map = seat(
            AgentName::Caspar,
            vec![attempt(
                "glm-5.2:cloud",
                json!(format!("stopped by http://alice:{CANARY}@host/x")),
                json!(7),
            )],
        );

        let json = render_completions(&map).to_string();
        assert!(!json.contains(CANARY), "the finish reason leaked: {json}");
    }

    /// `FinishReason::Other(String)` must reach the JSON as the wire label the backend sent, not
    /// as a `Debug` rendering that would wrap it in the variant's name. The gate asked for this
    /// explicitly: the redaction test proves nothing leaks, but it does not prove the value that
    /// survives is the right SHAPE.
    #[test]
    fn an_unrecognised_finish_reason_renders_its_wire_string_not_the_debug_form() {
        let map = seat(
            AgentName::Melchior,
            vec![attempt("m", json!("interrupted_by_operator"), json!(3))],
        );
        let rendered = render_completions(&map);
        let finish = &rendered["melchior"][0]["finish"];
        assert_eq!(
            finish,
            &json!("interrupted_by_operator"),
            "the wire string must survive; a Debug form would read like Other(\"...\")"
        );
        assert!(
            !finish.as_str().expect("a string").starts_with("Other"),
            "the variant name leaked into a value consumers parse: {finish}"
        );
    }
    // ── The cut-attempt predicate (REQ-EE-2, REQ-EE-4, REQ-EE-7) ──────────────────────────────

    /// Builds an [`AgentRotation`] through magi-core's `Deserialize` — the same door
    /// `rotation_report::tests::rotation` uses.
    fn rotation(json: Value) -> AgentRotation {
        serde_json::from_value(json).expect("the fixture must match magi-core's shape")
    }

    /// A rotation of one or more hops: `model_configured` is `models[0]`, hop k goes to
    /// `models[k+1]` with kind `kinds[k]` (snake_case tags), `detail: "d"`, lineages `lin-k`.
    fn rotated(models: &[&str], kinds: &[&str]) -> AgentRotation {
        let chain: Vec<Value> = kinds
            .iter()
            .enumerate()
            .map(|(k, kind)| {
                json!({
                    "from": format!("lin-{k}"),
                    "to": format!("lin-{}", k + 1),
                    "model_resolved": models[k + 1],
                    "kind": kind,
                    "detail": "d",
                })
            })
            .collect();
        rotation(json!({
            "model_configured": models[0],
            "model_used": models[models.len() - 1],
            "ran_unmeasured": false,
            "chain": chain,
        }))
    }

    /// A seat that never rotated (`chain: []`) on `model`.
    fn never_rotated_on(model: &str) -> AgentRotation {
        rotation(json!({
            "model_configured": model,
            "model_used": model,
            "ran_unmeasured": false,
            "chain": [],
        }))
    }

    /// One attempt served by `model` with the given raw `finish` (a wire string or `Value::Null`),
    /// `completion_tokens`, and nothing measured about reasoning — a timed-out or failed attempt is
    /// `attempt_on(model, Value::Null, Value::Null)`, the shape `CompletionRecord::new` produces.
    fn attempt_on(model: &str, finish: Value, completion_tokens: Value) -> CompletionRecord {
        record(json!({
            "model": model,
            "cap": 16_384,
            "finish": finish,
            "completion_tokens": completion_tokens,
            "prompt_tokens": Value::Null,
            "reasoning": "NotMeasured",
        }))
    }

    /// A `length` finish is a cut even when the seat went on to answer: the budget ran out on
    /// that attempt, which is the fact REQ-EE-2 logs.
    #[test]
    fn a_length_finish_is_a_cut_even_when_the_seat_went_on_to_answer() {
        let completions = seat(
            AgentName::Caspar,
            vec![
                attempt_on("glm-5.2:cloud", json!("length"), json!(16_384)),
                attempt_on("glm-5.2:cloud", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(AgentName::Caspar, never_rotated_on("glm-5.2:cloud"));

        let cuts = cut_attempts(&completions, &rotations, &BTreeMap::new());
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        assert_eq!(cuts[0].seat, AgentName::Caspar);
        assert_eq!(cuts[0].index, 0);
        assert_eq!(cuts[0].cause, CutCause::Length);
    }

    /// The empty attempt cannot be read off its record (magi-core builds it like a success),
    /// so it is located by the hop that left its model: kind `empty_completion` marks the LAST
    /// record of that model's run.
    #[test]
    fn an_empty_completion_hop_marks_the_last_attempt_of_the_model_it_left() {
        let completions = seat(
            AgentName::Caspar,
            vec![
                attempt_on("down-model", json!("stop"), json!(0)),
                attempt_on("rescue-model", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(
            AgentName::Caspar,
            rotated(&["down-model", "rescue-model"], &["empty_completion"]),
        );

        let cuts = cut_attempts(&completions, &rotations, &BTreeMap::new());
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        assert_eq!(cuts[0].index, 0);
        assert_eq!(cuts[0].cause, CutCause::EmptyContent);
        assert_eq!(cuts[0].record.model, "down-model");
    }

    /// With the schema retry a model can run twice before it is left; only the attempt that
    /// CAUSED the hop — the last of the run — came back empty.
    #[test]
    fn the_empty_attempt_is_the_last_of_its_model_run_not_the_first() {
        let completions = seat(
            AgentName::Melchior,
            vec![
                attempt_on("down-model", json!("stop"), json!(40)),
                attempt_on("down-model", json!("stop"), json!(0)),
                attempt_on("rescue-model", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(
            AgentName::Melchior,
            rotated(&["down-model", "rescue-model"], &["empty_completion"]),
        );

        let cuts = cut_attempts(&completions, &rotations, &BTreeMap::new());
        let indices: Vec<usize> = cuts.iter().map(|c| c.index).collect();
        assert_eq!(
            indices,
            vec![1],
            "only the attempt that caused the hop: {cuts:?}"
        );
    }

    /// A seat that ran out of rotations has no hop for its last model; the empty completion
    /// survives only inside its `failed_agents` cause, which contains magi-core's `Display`.
    #[test]
    fn a_seat_lost_to_an_empty_completion_marks_its_final_attempt() {
        let completions = seat(
            AgentName::Caspar,
            vec![attempt_on("down-model", json!("stop"), json!(0))],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(AgentName::Caspar, never_rotated_on("down-model"));
        let mut failed = BTreeMap::new();
        failed.insert(
            AgentName::Caspar,
            format!("no_fitting_candidate: {EMPTY_COMPLETION_MARKER}the model returned no content"),
        );

        let cuts = cut_attempts(&completions, &rotations, &failed);
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        assert_eq!(cuts[0].index, 0);
        assert_eq!(cuts[0].cause, CutCause::EmptyContent);
    }

    /// PM-S-1 / spec §1 criterion 3: an attempt the clock killed carries no measurement and is
    /// its own class. Counting it as a cut would merge the two failures the replays must count
    /// apart.
    #[test]
    fn a_timed_out_attempt_is_its_own_class_and_never_a_cut() {
        let completions = seat(
            AgentName::Balthasar,
            vec![
                attempt_on("slow-model", Value::Null, Value::Null),
                attempt_on("rescue-model", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(
            AgentName::Balthasar,
            rotated(&["slow-model", "rescue-model"], &["timeout"]),
        );
        let mut failed = BTreeMap::new();
        failed.insert(
            AgentName::Balthasar,
            "timeout: agent exceeded 27s".to_string(),
        );

        assert!(cut_attempts(&completions, &rotations, &failed).is_empty());
    }

    /// A schema rotation after a clean stop is not a cut: the model answered, badly.
    #[test]
    fn a_schema_rotation_after_a_clean_stop_is_not_a_cut() {
        let completions = seat(
            AgentName::Caspar,
            vec![
                attempt_on("chatty-model", json!("stop"), json!(700)),
                attempt_on("chatty-model", json!("stop"), json!(650)),
                attempt_on("rescue-model", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(
            AgentName::Caspar,
            rotated(&["chatty-model", "rescue-model"], &["schema"]),
        );

        assert!(cut_attempts(&completions, &rotations, &BTreeMap::new()).is_empty());
    }

    /// When both hold, `length` names the cause: it is the one the record itself proves.
    #[test]
    fn length_names_the_cause_when_the_cut_attempt_was_also_empty() {
        let completions = seat(
            AgentName::Caspar,
            vec![
                attempt_on("down-model", json!("length"), json!(16_384)),
                attempt_on("rescue-model", json!("stop"), json!(900)),
            ],
        );
        let mut rotations = BTreeMap::new();
        rotations.insert(
            AgentName::Caspar,
            rotated(&["down-model", "rescue-model"], &["empty_completion"]),
        );

        let cuts = cut_attempts(&completions, &rotations, &BTreeMap::new());
        assert_eq!(
            cuts.len(),
            1,
            "one attempt, one entry — never counted twice: {cuts:?}"
        );
        assert_eq!(cuts[0].cause, CutCause::Length);
    }

    /// Empty inputs: no records ⇒ no cuts; a seat present with zero attempts has nothing to
    /// attribute its failure to, even with the marker in its cause.
    #[test]
    fn no_records_means_no_cut_attempts_even_for_a_failed_seat() {
        assert!(cut_attempts(&BTreeMap::new(), &BTreeMap::new(), &BTreeMap::new()).is_empty());

        let completions = seat(AgentName::Melchior, Vec::new());
        let mut failed = BTreeMap::new();
        failed.insert(
            AgentName::Melchior,
            format!("{EMPTY_COMPLETION_MARKER}the model returned no content"),
        );
        assert!(cut_attempts(&completions, &BTreeMap::new(), &failed).is_empty());
    }

    /// Order: the map's seat order, then magi-core's attempt order — the order the JSON renders.
    #[test]
    fn cut_attempts_keep_seat_order_then_attempt_order() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![attempt_on("c", json!("length"), json!(16_384))],
        );
        completions.insert(
            AgentName::Melchior,
            vec![
                attempt_on("m", json!("length"), json!(16_384)),
                attempt_on("m", json!("length"), json!(16_384)),
            ],
        );

        let cuts = cut_attempts(&completions, &BTreeMap::new(), &BTreeMap::new());
        let got: Vec<(AgentName, usize)> = cuts.iter().map(|c| (c.seat, c.index)).collect();
        let expected: Vec<(AgentName, usize)> = completions
            .iter()
            .flat_map(|(seat, records)| (0..records.len()).map(move |i| (*seat, i)))
            .collect();
        assert_eq!(got, expected);
    }

    /// The two labels the log line prints.
    #[test]
    fn each_cut_cause_has_a_stable_label() {
        assert_eq!(CutCause::Length.label(), "length");
        assert_eq!(CutCause::EmptyContent.label(), "empty_content");
    }
}
