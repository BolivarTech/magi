// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! Pool eligibility telemetry, composed for output.
//!
//! # This report is PURELY DIAGNOSTIC
//!
//! Nothing in magi-rs may branch on what this module renders. It answers *"which pool candidates
//! could each seat have rotated into, and which were ruled out and why"* — a snapshot magi-core
//! takes **before dispatch**, for a human reading a run afterwards. The moment a magi-rs decision
//! reads it, a diagnostic that is allowed to be incomplete becomes a control input, and the
//! accepted gaps in it (the three causes magi-core documents as unreachable pre-dispatch) turn
//! from footnotes into silent behaviour changes. Filtering stays entirely with magi-core, which is
//! the same boundary REQ-R26 draws for an assumed window.
//!
//! # Why this takes the eligibility MAP and not the `MagiReport`
//!
//! Same reasoning as [`crate::magi::rotation_report`]: `MagiReport` and `CandidateEligibility` are
//! both `#[non_exhaustive]`, so neither can be built with a literal from outside magi-core. Taking
//! the map keeps this module testable through the one door that does exist — `serde`
//! deserialization, whose derive lives inside the defining crate and can therefore reach private
//! fields — and it is the smaller dependency: nothing here needs a report, only its eligibility.

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

use magi_core::rotation::{CandidateEligibility, IneligibilityCause};
use magi_core::schema::AgentName;
use serde_json::{json, Value};

use crate::magi::seat_label;
use crate::redact::redact_foreign_text;

/// Renders the pre-dispatch pool eligibility as JSON.
///
/// Produces one key per seat label, whose value is an array with one object per candidate carrying
/// `model` and `causes`. An **empty** `causes` array means the candidate was eligible; magi-core
/// reports every failing condition rather than the first, so the array is a complete answer and not
/// an arbitrary one of several true reasons.
///
/// # The map is EMITTED EVEN WHEN EMPTY, and never as null or absent
///
/// An absent map and a map in which every candidate is eligible are **different facts**: the first
/// says the snapshot was not computed, the second says it was computed and found nothing to reject.
/// A consumer has to be able to tell them apart — a run with no pool declared and a run with a
/// healthy pool would otherwise read identically, and the inert-pool case is exactly what this
/// telemetry exists to make visible. So the object is always present, empty included, and a caller
/// that omits it on emptiness has broken the contract rather than tidied the output.
///
/// # Redaction: `model`, and only `model`
///
/// `model` is a string **composed by another crate**, so it goes through
/// [`redact_foreign_text`] at composition — once, here, rather than at each output surface where
/// there would be one chance per surface to forget. `redact_url` is **not** interchangeable: a
/// model tag carries no authority, and `redact_url` treats its whole input as a URL and would
/// collapse the tag to `"***"`, destroying the diagnostic.
///
/// # Complexity
///
/// `O(seats x candidates)` — plain iteration over the map and each seat's candidate vector. Run
/// over a trio, once per consult, so the input is a handful of entries and anything cleverer would
/// buy a measurable nothing at the price of a structure to keep in step.
///
/// # Arguments
/// * `eligibility` - magi-core's pre-dispatch snapshot, keyed by seat.
///
/// # Returns
///
/// A JSON object mapping each seat label to its candidate array. Always an object, never null.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::eligibility_report::render_pool_eligibility;
/// let json = render_pool_eligibility(&BTreeMap::new());
/// assert!(json.as_object().is_some_and(|m| m.is_empty()));
/// ```
#[must_use]
pub fn render_pool_eligibility(
    eligibility: &BTreeMap<AgentName, Vec<CandidateEligibility>>,
) -> Value {
    let mut out = serde_json::Map::new();
    for (agent, candidates) in eligibility {
        let rendered: Vec<Value> = candidates
            .iter()
            .map(|candidate| {
                // The ONLY foreign-composed string here. Bound rather than inlined because
                // `SafeErrorText` deliberately does NOT implement `Serialize`: reaching the JSON
                // has to go through an explicit `as_str`, which is what stops an unredacted
                // `String` from taking its place with nothing to notice.
                let model = redact_foreign_text(&candidate.model);
                json!({
                    "model": model.as_str(),
                    // The causes need NO redaction: every `IneligibilityCause` variant is either
                    // a unit variant or carries only `u32`/`usize` payloads -- verified against
                    // the pinned `magi-core = "=4.0.0"`, where not one of the eight holds a
                    // `String`, `&str` or `Cow`. RE-VERIFY THIS AT EVERY PIN BUMP: a payload can
                    // gain a field in a minor release, and a new foreign-composed string would
                    // then reach the output unredacted with nothing here to fail.
                    "causes": candidate
                        .causes
                        .iter()
                        .map(cause_label)
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        out.insert(seat_label(*agent), Value::Array(rendered));
    }
    Value::Object(out)
}

/// Stable label for an ineligibility cause, DERIVED from the crate.
///
/// # Why the serde form and not a hand-written match
///
/// `#[non_exhaustive]` binds CONSUMERS, not the defining crate: inside magi-core the enum is still
/// closed and its serde derive enumerates every variant, so a new cause cannot ship without
/// magi-core rendering it. A match here would need a wildcard, and a wildcard is what renders a
/// newly added cause under some older variant's name — a wrong answer that looks like a right one.
///
/// The two data-carrying variants (`RotationBudgetExhausted`, `WindowBelowCoarseEstimate`)
/// serialize as an object rather than a bare string, so their payload reaches the report instead of
/// being flattened away.
///
/// # Arguments
/// * `cause` - the failing condition magi-core reported.
///
/// # Returns
///
/// The `snake_case` serde form. Falls back to the variant's `Debug` rendering, so no placeholder is
/// ever invented and none can leak into the JSON.
fn cause_label(cause: &IneligibilityCause) -> Value {
    // Handled rather than unwrapped: a panic here would take down a whole report over one
    // diagnostic field.
    serde_json::to_value(cause).unwrap_or_else(|_| json!(format!("{cause:?}")))
}

/// Unit tests for the pool eligibility composition.
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a [`CandidateEligibility`] through **deserialization**, the only door magi-core
    /// leaves open: the type is `#[non_exhaustive]`, so a struct literal does not compile from
    /// here, while `serde`'s derive lives inside the defining crate and can reach its fields.
    fn candidate(json: Value) -> CandidateEligibility {
        serde_json::from_value(json).expect("the fixture must match magi-core's shape")
    }

    /// One seat with one candidate carrying `causes`.
    fn one_seat(model: &str, causes: Value) -> BTreeMap<AgentName, Vec<CandidateEligibility>> {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Caspar,
            vec![candidate(json!({ "model": model, "causes": causes }))],
        );
        map
    }

    /// A candidate refused because its window could not be measured names that exact condition,
    /// so a reader can tell "rejected for being unmeasurable" from "rejected for being too small".
    #[test]
    fn a_candidate_rejected_under_the_strict_guard_names_that_cause() {
        let map = one_seat(
            "glm-5.2:cloud",
            json!(["window_unmeasured_under_strict_guard"]),
        );

        let json = render_pool_eligibility(&map);
        let entry = &json["caspar"][0];
        assert_eq!(entry["model"], "glm-5.2:cloud");
        assert_eq!(
            entry["causes"][0], "window_unmeasured_under_strict_guard",
            "the cause must render by its snake_case serde label"
        );
    }

    /// An absent map and a computed-but-empty one are DIFFERENT facts, so the key must exist and
    /// be an object with zero entries -- never null, never omitted.
    #[test]
    fn an_empty_map_still_renders_as_a_json_object_with_no_entries() {
        let json = render_pool_eligibility(&BTreeMap::new());

        assert!(
            json.is_object(),
            "an empty snapshot must stay an object, not become null: {json}"
        );
        assert!(!json.is_null(), "null would mean 'not computed'");
        assert_eq!(
            json.as_object().map(serde_json::Map::len),
            Some(0),
            "computed and empty, not absent"
        );
    }

    /// An eligible candidate is REPORTED, with an empty cause array. Dropping it would make the
    /// snapshot unable to say a pool was healthy, which is half of what it exists to answer.
    #[test]
    fn an_eligible_candidate_is_reported_with_an_empty_cause_array() {
        let map = one_seat("primary-model", json!([]));

        let json = render_pool_eligibility(&map);
        assert_eq!(
            json["caspar"].as_array().map(Vec::len),
            Some(1),
            "an eligible candidate must not be filtered out of the snapshot"
        );
        assert!(json["caspar"][0]["causes"]
            .as_array()
            .is_some_and(Vec::is_empty));
    }

    /// magi-core reports EVERY failing condition, not the first, so all of them must survive
    /// composition -- and a data-carrying variant must keep its payload rather than flatten to a
    /// bare label.
    #[test]
    fn every_cause_survives_including_the_payload_of_a_data_carrying_variant() {
        let map = one_seat(
            "small-model",
            json!([
                { "window_below_coarse_estimate": {
                    "measured_window": 8192,
                    "estimated_need": 40000
                }},
                "model_already_used_by_this_mage",
            ]),
        );

        let json = render_pool_eligibility(&map);
        let causes = json["caspar"][0]["causes"]
            .as_array()
            .expect("causes is an array");
        assert_eq!(causes.len(), 2, "no cause may be dropped: {causes:?}");
        assert_eq!(
            causes[0]["window_below_coarse_estimate"]["measured_window"],
            json!(8192),
            "the payload must reach the report, not be flattened away"
        );
        assert_eq!(causes[1], "model_already_used_by_this_mage");
    }

    /// Every seat present in the snapshot gets its own lowercase key, matching the identity
    /// magi-core serializes and the labels `rotation_report` already emits.
    #[test]
    fn each_seat_renders_under_its_own_lowercase_label() {
        let mut map = BTreeMap::new();
        map.insert(AgentName::Melchior, vec![]);
        map.insert(
            AgentName::Balthasar,
            vec![candidate(json!({ "model": "m", "causes": [] }))],
        );

        let json = render_pool_eligibility(&map);
        let object = json.as_object().expect("the snapshot is an object");
        let mut keys: Vec<&String> = object.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["balthasar", "melchior"]);
        assert!(
            json["melchior"].as_array().is_some_and(Vec::is_empty),
            "a seat with no candidates keeps its key with an empty array"
        );
    }

    /// A model tag is a foreign-composed string, so it passes through `redact_foreign_text`. The
    /// surviving-diagnostic assertion is what makes the no-leak one mean anything: an EMPTY output
    /// contains no secret either, and `redact_url` -- the wrong helper here -- would collapse the
    /// whole tag to `"***"`.
    #[test]
    fn a_credential_embedded_in_a_model_tag_is_redacted_without_collapsing_it() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let map = one_seat(
            &format!("registry http://alice:{CANARY}@host:11434/v1 qwen3.5:397b-cloud"),
            json!([]),
        );

        let json = render_pool_eligibility(&map).to_string();
        assert!(!json.contains(CANARY), "the model tag leaked: {json}");
        assert!(
            json.contains("qwen3.5:397b-cloud"),
            "redaction must remove the credential, NOT collapse the tag -- which is what using \
             `redact_url` on it would do: {json}"
        );
    }

    /// The candidate object's key set is a CONTRACT. This test breaking is a versioned decision,
    /// not an accident -- but it must break deliberately, naming the added key.
    #[test]
    fn the_candidate_object_carries_exactly_the_declared_keys() {
        let json = render_pool_eligibility(&one_seat("m", json!([])));
        let mut keys: Vec<String> = json["caspar"][0]
            .as_object()
            .expect("a candidate is an object")
            .keys()
            .cloned()
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["causes", "model"],
            "a key was added or removed without updating the contract"
        );
    }
}
