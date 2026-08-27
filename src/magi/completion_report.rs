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

use magi_core::provider::FinishReason;
use magi_core::reporting::CompletionRecord;
use magi_core::schema::AgentName;
use serde_json::{json, Map, Value};

use crate::redact::redact_foreign_text;

/// Renders the per-attempt completion telemetry as JSON.
///
/// One key per seat that has records, holding the seat's attempts **in magi-core's order**, one
/// object per attempt with exactly five keys: `model`, `cap`, `finish`, `completion_tokens` and
/// `prompt_tokens`.
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
/// # The sixth field, `reasoning`, is DELIBERATELY NOT RENDERED
///
/// [`CompletionRecord`] also carries `reasoning: ReasoningState`. **Nothing in magi-rs consumes
/// it**, and the project standard forbids shipping public surface that no existing consumer
/// reads: a key in this object is a contract a CI consumer may pin, so emitting one speculatively
/// costs a removal later that a plain addition would not. The day a consumer exists, adding the
/// key is additive and cheap; that is the trade this omission takes.
///
/// # Arguments
/// * `completions` - magi-core's per-seat attempt records, exactly as the report carries them.
///
/// # Returns
///
/// A JSON object keyed by lowercase seat label. A seat with no records contributes no key, and an
/// empty input renders an empty object — the array under a seat is never synthesized.
///
/// # Complexity
///
/// `O(seats x records)`: plain iteration over every attempt, once. The map is a trio and the
/// function runs **once per consult**, so the whole traversal is a handful of items on a path that
/// already spent seconds in HTTP. An index or a precomputed lookup would buy nothing measurable
/// and would be over-engineering against a workload this small.
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
                    })
                })
                .collect();
            (seat_label(*agent), Value::Array(attempts))
        })
        .collect();

    Value::Object(seats)
}

/// Lowercase seat label, matching the identity magi-core serializes.
fn seat_label(agent: AgentName) -> String {
    agent.display_name().to_lowercase()
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

    /// The consult JSON's key set is a CONTRACT, and `reasoning` is deliberately absent from it:
    /// nothing consumes it, and a key nobody reads is public surface that costs a removal later.
    /// This test breaking is a versioned decision, not an accident — but it must break
    /// deliberately, naming the added key.
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
                "finish",
                "model",
                "prompt_tokens"
            ],
            "a key was added or removed without updating the contract"
        );
        assert!(
            !object.contains_key("reasoning"),
            "the sixth field stays unrendered until a consumer for it exists"
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
}
