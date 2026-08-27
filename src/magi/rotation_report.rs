// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-11

//! Rotation telemetry, composed for output (REQ-R06/R07/R08/R09/R16).
//!
//! # Redaction happens HERE, at composition, not at the call sites
//!
//! Every string in this module's input is **composed by another crate**: `RotationEvent::detail`
//! carries whatever magi-core assembled from a transport failure, which can be an HTTP error body
//! or a URL with a resolved credential in it. That is the exact shape that produced **five**
//! separate findings in the previous milestone's gate, every one of them green against all seven
//! build gates (B17). Redacting at each output surface instead would mean four chances to forget;
//! redacting once, where the value is turned into output, means the surfaces cannot get it wrong.
//!
//! **`redact_url` is NOT interchangeable with `redact_foreign_text`.** The first treats its whole
//! input as a URL; a `detail` is prose that may *embed* one, so `redact_url` would collapse the
//! whole message to `***`. Using the wrong one either destroys the diagnostic or leaks.
//!
//! # Why this takes the rotations MAP and not the `MagiReport`
//!
//! `MagiReport`, `AgentRotation` and `RotationEvent` are all `#[non_exhaustive]` with
//! `pub(crate)` constructors, so **none of them can be built from outside magi-core**. Taking the
//! map keeps this module testable through the one door that does exist — `serde` deserialization,
//! whose derive lives inside the defining crate and can therefore reach private fields — and it
//! is also the smaller dependency: nothing here needs a report, only its rotations.

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

use magi_core::rotation::{AgentRotation, RotationKind};
use magi_core::schema::AgentName;
use serde_json::{json, Value};

use crate::redact::redact_foreign_text;

/// Renders the rotation telemetry as JSON (REQ-R07/R08).
///
/// # The two keys are separate on purpose
///
/// `rotations` carries **only the mages that actually hopped**, because that is what makes an
/// empty array a *positive certificate* that nobody rotated (SC-R13) — magi-core's own map is
/// populated for every agent, so echoing it would make "empty" unreachable and the certificate
/// meaningless. `ran_unmeasured` is a different fact about a different question: it says a mage
/// ran **without a measured window**, which qualifies the confidence in its verdict (REQ-R08) and
/// is true whether or not it rotated.
///
/// **Both keys are ALWAYS present, empty included.** A field that appears and disappears changes
/// the JSON's shape between runs, and a consumer with a strict schema cannot declare it — the
/// same criterion `extraction_failures` established in the previous milestone.
///
/// # The one case where `model_used` can name the configured model anyway
///
/// REQ-R06 says the report names the model that actually produced the verdict, and that holds
/// **except** for a mage that rotated and whose task then **panicked or was cancelled**. magi-core
/// documents this as an accepted limitation: that entry shows the *pre-seed* — an empty `chain`
/// and `model_used == model_configured` — because the hops it made lived on the stack of the task
/// that died, and recovering them would need a second lock outside the rotation registry, which
/// its single-lock concurrency design deliberately forbids.
///
/// **This cannot be corrected from outside the crate, so it is declared instead of hidden.** It is
/// exactly the kind of detail somebody discovers while reading a report and concludes magi-rs has
/// a bug. The signal that such an entry is not to be trusted does exist and is worth naming: that
/// mage also appears in `failed_agents`, so a `rotations` entry with an empty `chain` for a seat
/// that failed is *"unknown"*, not *"did not rotate"*.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::rotation_report::render_rotations;
/// let json = render_rotations(&BTreeMap::new());
/// assert!(json["rotations"].as_array().is_some_and(Vec::is_empty));
/// assert!(json["ran_unmeasured"].as_array().is_some_and(Vec::is_empty));
/// ```
#[must_use]
pub fn render_rotations(rotations: &BTreeMap<AgentName, AgentRotation>) -> Value {
    let hops: Vec<Value> = rotations
        .iter()
        .filter(|(_, r)| !r.chain.is_empty())
        .map(|(agent, r)| {
            json!({
                "agent": seat_label(*agent),
                "model_configured": r.model_configured,
                // REQ-R06: what ACTUALLY produced the verdict. A report naming the configured
                // model when the fallback ran lies about its own evidence base.
                "model_used": r.model_used,
                "ran_unmeasured": r.ran_unmeasured,
                "chain": r.chain.iter().map(|hop| {
                    // The ONLY foreign-composed string here, and the reason this module exists.
                    // Bound rather than inlined: `SafeErrorText` deliberately does NOT implement
                    // `Serialize`, so reaching the JSON has to go through an explicit `as_str`
                    // — which is the point. A type that serialized itself would let an
                    // unredacted `String` take its place with nothing to notice.
                    let detail = redact_foreign_text(hop.detail());
                    json!({
                        "from_lineage": hop.from().as_str(),
                        "to_lineage": hop.to().as_str(),
                        "model_resolved": hop.model_resolved(),
                        "cause": cause_label(hop.kind()),
                        "detail": detail.as_str(),
                    })
                }).collect::<Vec<_>>(),
            })
        })
        .collect();

    let unmeasured: Vec<Value> = rotations
        .iter()
        .filter(|(_, r)| r.ran_unmeasured)
        .map(|(agent, _)| json!(seat_label(*agent)))
        .collect();

    json!({ "rotations": hops, "ran_unmeasured": unmeasured })
}

/// Renders the rotation telemetry as text lines, already redacted (REQ-R07/R16).
///
/// Returns an empty vector when nobody rotated, so a caller appends nothing rather than a heading
/// with nothing under it.
///
/// # The `mage-local:` marker is preserved and never parsed (SC-R57)
///
/// magi-core marks a cause that is local to one mage — as opposed to run-wide — with a prefix in
/// `detail`, because its cause enum is public and exhaustive; it plans dedicated variants for the
/// next major. So the text is carried through **verbatim** (after redaction) and no behaviour
/// here branches on it. Parsing it would bind us to a shape the crate has already announced it
/// will change, and would break on a minor upgrade, far from its cause.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::rotation_report::rotation_lines;
/// assert!(rotation_lines(&BTreeMap::new()).is_empty());
/// ```
#[must_use]
pub fn rotation_lines(rotations: &BTreeMap<AgentName, AgentRotation>) -> Vec<String> {
    rotations
        .iter()
        .filter(|(_, r)| !r.chain.is_empty())
        .map(|(agent, r)| {
            let hops = r
                .chain
                .iter()
                .map(|hop| {
                    format!(
                        "{} -> {} ({}, {}): {}",
                        hop.from().as_str(),
                        hop.to().as_str(),
                        hop.model_resolved(),
                        cause_label(hop.kind()),
                        redact_foreign_text(hop.detail()).as_str()
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            let unmeasured = if r.ran_unmeasured {
                " [ran unmeasured]"
            } else {
                ""
            };
            format!(
                "{}: {} -> {}{unmeasured} | {hops}",
                seat_label(*agent),
                r.model_configured,
                r.model_used
            )
        })
        .collect()
}

/// Lowercase seat label, matching the identity magi-core serializes.
fn seat_label(agent: AgentName) -> String {
    agent.display_name().to_lowercase()
}

/// Stable label for a rotation cause.
///
/// The rotation cause as it reaches the run JSON, DERIVED from the crate (REQ-V4-01).
///
/// # Why the serde form and not `Display`
///
/// Both render the identical string and 4.0.0 pins both with tests, but those tests promise
/// different things: the serde one exists so a `rename_all` change cannot silently alter an
/// existing variant's serialized form, while the `Display` one only asserts current behaviour and
/// a reworded `Display` is conventionally not a breaking change. This string reaches the run JSON,
/// which CI consumers parse, so the wire contract is the right anchor.
///
/// # Why deriving PRESERVES the compile-time guard (REQ-V4-02)
///
/// `#[non_exhaustive]` binds CONSUMERS, not the defining crate. Inside magi-core the enum is still
/// closed and its serde derive enumerates every variant, so a new cause cannot ship without
/// magi-core rendering it. magi-rs inherits that instead of maintaining a second copy that drifts
/// — and a hand-written match here would need a wildcard, which is what rendered
/// `OversizedResponse` as `"transport"` while this was scaffolded.
///
/// # Arguments
/// * `kind` - the cause magi-core reported.
///
/// # Returns
///
/// The `snake_case` label. Falls back to `Display`, which renders the same string, so no
/// placeholder is ever invented and none can leak into the JSON.
fn cause_label(kind: RotationKind) -> String {
    match serde_json::to_value(kind) {
        Ok(Value::String(s)) => s,
        // Unreachable for a unit-variant enum, and handled rather than unwrapped because a panic
        // here would take down a whole report over one diagnostic field.
        _ => kind.to_string(),
    }
}

/// Unit tests for the rotation telemetry composition.
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an [`AgentRotation`] through **deserialization**, the only door magi-core leaves
    /// open: the type is `#[non_exhaustive]` with a `pub(crate)` constructor, so a struct literal
    /// does not compile from here, while `serde`'s derive lives inside the defining crate and can
    /// reach the private fields.
    fn rotation(json: Value) -> AgentRotation {
        serde_json::from_value(json).expect("the fixture must match magi-core's shape")
    }

    /// A rotation with one hop, from `from` to `to`, ending on `model_used`.
    fn one_hop(from: &str, to: &str, model_used: &str, detail: &str) -> AgentRotation {
        rotation(json!({
            "model_configured": "primary-model",
            "model_used": model_used,
            "ran_unmeasured": false,
            "chain": [{
                "from": from,
                "to": to,
                "model_resolved": model_used,
                "kind": "transport",
                "detail": detail,
            }],
        }))
    }

    /// A one-hop rotation whose cause is `kind`, for the locality assertions.
    fn one_hop_with(kind: &str) -> BTreeMap<AgentName, AgentRotation> {
        let mut m = BTreeMap::new();
        m.insert(
            AgentName::Melchior,
            rotation(json!({
                "model_configured": "primary-model",
                "model_used": "fallback-model",
                "ran_unmeasured": false,
                "chain": [{
                    "from": "alibaba",
                    "to": "moonshot",
                    "model_resolved": "fallback-model",
                    "kind": kind,
                    "detail": "d",
                }],
            })),
        );
        m
    }

    /// A mage that never rotated: magi-core still records an entry, with an empty chain.
    fn never_rotated() -> AgentRotation {
        rotation(json!({
            "model_configured": "primary-model",
            "model_used": "primary-model",
            "ran_unmeasured": false,
            "chain": [],
        }))
    }

    /// SC-R13: the field is ALWAYS present, even empty. A field that appears and disappears
    /// changes the JSON's shape between runs, and a consumer with a strict schema cannot declare
    /// it. Empty is a POSITIVE CERTIFICATE that nobody rotated, not silence.
    #[test]
    fn rotations_is_present_and_empty_when_nobody_rotated() {
        let mut map = BTreeMap::new();
        map.insert(AgentName::Melchior, never_rotated());
        map.insert(AgentName::Balthasar, never_rotated());
        map.insert(AgentName::Caspar, never_rotated());

        let json = render_rotations(&map);
        assert!(
            json.get("rotations").is_some(),
            "the field must never be omitted"
        );
        assert_eq!(
            json["rotations"].as_array().map(Vec::len),
            Some(0),
            "magi-core records an entry per agent; only a HOP counts as a rotation, or `empty` \
             would be unreachable and the certificate meaningless"
        );
    }

    /// SC-R03 + SC-R06: from which lineage to which, why, and WITH WHICH MODEL IT ENDED.
    #[test]
    fn a_rotation_reports_cause_lineages_and_the_model_that_produced_the_verdict() {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Caspar,
            one_hop("deepseek", "zhipu", "glm-5.2:cloud", "503 from upstream"),
        );

        let json = render_rotations(&map);
        let entry = &json["rotations"][0];
        assert_eq!(entry["agent"], "caspar");
        assert_eq!(
            entry["model_used"], "glm-5.2:cloud",
            "REQ-R06: the model that ACTUALLY produced the verdict"
        );
        let hop = &entry["chain"][0];
        assert_eq!(hop["from_lineage"], "deepseek");
        assert_eq!(hop["to_lineage"], "zhipu");
        assert_eq!(hop["cause"], "transport");
    }

    /// REQ-R08: `ran_unmeasured` surfaces on its own key, for EVERY mage it is true of —
    /// including one that never rotated, because running without a measured window qualifies a
    /// verdict whether or not a hop happened.
    #[test]
    fn ran_unmeasured_surfaces_even_for_a_mage_that_never_rotated() {
        let mut unmeasured = never_rotated();
        unmeasured.ran_unmeasured = true;
        let mut map = BTreeMap::new();
        map.insert(AgentName::Melchior, unmeasured);

        let json = render_rotations(&map);
        assert_eq!(json["rotations"].as_array().map(Vec::len), Some(0));
        assert_eq!(
            json["ran_unmeasured"][0], "melchior",
            "the flag is about the window, not about rotating"
        );
    }

    /// SC-R57: the local-vs-run-wide distinction is PRESERVED but never PARSED. §7 forbids
    /// treating the marker as a stable contract — the crate declares it transitory and plans
    /// dedicated variants for the next major, and the literal even carries a trailing space.
    #[test]
    fn the_mage_local_marker_is_preserved_verbatim_and_never_branched_on() {
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Caspar,
            one_hop("a", "b", "m", "mage-local: endpoint refused"),
        );

        let lines = rotation_lines(&map);
        assert!(
            lines.iter().any(|l| l.contains("mage-local")),
            "the marker must survive into the output: {lines:?}"
        );
    }

    /// REQ-R16/B17: a `detail` composed by magi-core that embeds a credentialed URL is redacted
    /// at composition, on **both** surfaces.
    ///
    /// This is the unit-level half. The end-to-end half — driving a REAL failure and rotation
    /// through a credentialed endpoint — lives in `main.rs`, because a canary that builds the
    /// error by hand is exactly the guardian B16 declares useless, and this project already
    /// shipped one.
    #[test]
    fn a_credentialed_url_inside_a_detail_is_redacted_on_both_surfaces() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let mut map = BTreeMap::new();
        map.insert(
            AgentName::Caspar,
            one_hop(
                "a",
                "b",
                "m",
                &format!("POST http://alice:{CANARY}@host:11434/v1/chat/completions failed"),
            ),
        );

        let json = render_rotations(&map).to_string();
        assert!(!json.contains(CANARY), "the JSON surface leaked: {json}");
        let text = rotation_lines(&map).join("\n");
        assert!(!text.contains(CANARY), "the text surface leaked: {text}");
        assert!(
            text.contains("chat/completions"),
            "redaction must remove the credential, NOT collapse the diagnostic — that is what \
             using `redact_url` on prose would do: {text}"
        );
        // ^ THIS assertion is what makes the two above mean anything. An EMPTY output contains no
        // secret either, so a no-leak test that only checks absence passes against a function
        // that produces nothing at all — which is precisely what the Red stubs for this task
        // produced, and precisely the shape B16 exists to reject. Requiring the surviving
        // diagnostic is what separates "redacted" from "gone".
    }

    /// REQ-V4-01: all seven labels, AND the three that already existed unchanged. The second half
    /// is the regression that matters -- those strings reach the run JSON, which CI consumers parse.
    #[test]
    fn every_rotation_cause_renders_its_own_snake_case_label() {
        let cases = [
            (RotationKind::Transport, "transport"),
            (RotationKind::Timeout, "timeout"),
            (RotationKind::Schema, "schema"),
            (RotationKind::OversizedResponse, "oversized_response"),
            (RotationKind::ExternalFailure, "external_failure"),
            (RotationKind::EmptyCompletion, "empty_completion"),
            (RotationKind::ResponseContract, "response_contract"),
        ];
        for (kind, expected) in cases {
            assert_eq!(cause_label(kind), expected, "{kind:?} renders wrong");
        }
    }

    /// S3: the three labels magi-rs emitted before this milestone are byte-identical afterwards.
    /// Separate from the test above on purpose: that one would still pass if all seven changed
    /// together, and these three are the ones with consumers today.
    #[test]
    fn the_three_pre_existing_labels_are_byte_identical_to_the_hand_written_match() {
        assert_eq!(cause_label(RotationKind::Transport), "transport");
        assert_eq!(cause_label(RotationKind::Timeout), "timeout");
        assert_eq!(cause_label(RotationKind::Schema), "schema");
    }

    /// The serde error path falls back to `Display`, which renders the identical string -- so no
    /// placeholder can be invented and none can leak into the run JSON.
    #[test]
    fn the_label_never_renders_a_placeholder() {
        for kind in [RotationKind::Transport, RotationKind::EmptyCompletion] {
            let label = cause_label(kind);
            assert!(!label.is_empty());
            assert_ne!(label, "unknown");
            assert!(
                !label.contains('"'),
                "a quoted JSON string leaked into the label: {label}"
            );
        }
    }

    /// REQ-V4-09: the distinction the empty-content request was filed about -- did the cause
    /// condemn ONE seat or the whole run. Load-bearing now rather than merely useful, because the
    /// derived `retry_after_cap` makes `RetryAbandoned`'s run-wide condemnation reachable in
    /// ordinary operation, and it arrives as `Transport`.
    #[test]
    fn every_hop_reports_whether_its_cause_was_mage_local() {
        let v = render_rotations(&one_hop_with("empty_completion"));
        assert_eq!(v["rotations"][0]["chain"][0]["mage_local"], json!(true));

        let v = render_rotations(&one_hop_with("transport"));
        assert_eq!(
            v["rotations"][0]["chain"][0]["mage_local"],
            json!(false),
            "Transport condemns the lineage run-wide, and a Retry-After abandonment arrives as one"
        );
    }

    /// The consult JSON's key set is a CONTRACT. This test breaking is the versioned decision, not
    /// an accident -- but it must break deliberately, naming the added key.
    #[test]
    fn the_hop_object_carries_exactly_the_declared_keys() {
        let v = render_rotations(&one_hop_with("schema"));
        let mut keys: Vec<String> = v["rotations"][0]["chain"][0]
            .as_object()
            .expect("hop is an object")
            .keys()
            .cloned()
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "cause",
                "detail",
                "from_lineage",
                "mage_local",
                "model_resolved",
                "to_lineage"
            ],
            "a key was added or removed without updating the contract"
        );
    }
}
