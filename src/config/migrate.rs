// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! Validation pass that reports all migration incompatibilities of a `magi.toml` **together**
//! (REQ-A21b, REQ-R22).
//!
//! Serde cannot produce that message on its own: `deny_unknown_fields` **aborts on the FIRST
//! unknown key**, and a missing field is reported one at a time. So this pass collects the
//! incompatibilities it knows — before deserializing for the ones serde would abort on, after it
//! for the ones serde stays silent about — and emits a single message.
//!
//! # Every pattern SET has a retirement date; the MECHANISM has none
//!
//! The v0.11.0 set — `provider = "openai"`, `[openai].base_url`,
//! `[headless].tool_result_cap_bytes` — was retired in v0.13.0 (MS3, REQ-R22 / SC-R20). That
//! debt was dated on purpose: those patterns duplicated schema knowledge about a shape that is
//! no longer in the code, so they could only rot. The same release then put the mechanism
//! straight back to work on the break it introduced itself — mandatory seat lineages.
//!
//! **The mechanism carries no such date, and retiring it with them would be the wrong reading of
//! that note.** Its reason to exist outlives any single generation: `deny_unknown_fields` aborts
//! on the FIRST unknown key and serde reports one missing field at a time, so without a pass
//! that collects them all, the user pays one edit-start-fail cycle per incompatibility — start,
//! read one complaint, edit, start again. That is true of every future break, not only of the
//! one that motivated the pass.
//!
//! # Two halves, because a break arrives in one of two shapes
//!
//! **Unknown keys — [`detect_migrations`], before serde.** `deny_unknown_fields` aborts on the
//! first one, so the only way to collect them all is to read the document ahead of serde. **No
//! such pattern is declared today**: the v0.11.0 set retired with v0.13.0, and this half waits for
//! the next break that arrives in that shape.
//!
//! **Missing keys — [`missing_seat_lineages`], after serde.** An absent key with an `Option` field
//! produces no serde complaint at all, so there is nothing to get ahead of, and reading the typed
//! struct beats re-reading the raw document: renaming a field breaks the compile, whereas a
//! string-keyed pattern would go on matching nothing in silence. That silent-rot property is
//! precisely the debt REQ-R22 retired, so this half deliberately does not re-incur it.
//!
//! Both halves produce [`Migration`] values and share one [`render_migration_error`] frame, which
//! is what keeps "reported whole, in one message" a property of the mechanism rather than of
//! whichever half happens to be loaded.
//!
//! # The frame names ONE break at a time — move it with the patterns
//!
//! `VERSION_FROM`, `VERSION_TO`, `V0_11_X_NOTE` and `MINIMAL_VALID_CONFIG` describe the
//! v0.12.0 → v0.13.0 break (mandatory seat lineages, REQ-R22 / SC-R46 / SC-R47). Whoever declares
//! the patterns of the NEXT break retargets them in the same change: a correct list of
//! incompatibilities under a header naming the wrong pair of versions is worse than no message,
//! because it tells the user their file is a generation older than it is.
//!
//! # Any correction that echoes a value read from the file is redacted first
//!
//! The retired `[openai].base_url` correction pasted the user's own URL back, so it went through
//! `redact_url`/`locate_userinfo` and said, in the message, that the value was masked (SC-A21e).
//! A migration message that leaks a credential to the terminal, the scrollback and CI logs is a
//! worse problem than a line the user has to complete by hand. That rule belongs to the
//! mechanism, not to the pattern that first needed it — the current lineage corrections honour it
//! by quoting nothing from the file at all.
//!
//! # The line-locating helpers are gone, and the v0.13.0 patterns do not want them back
//!
//! **They went with their patterns**, rather than staying behind an
//! `#[allow(dead_code)]` that would claim a caller they no longer have: `line_of`,
//! `line_of_in_section` and `is_key_at` are in git history at the v0.13.0 boundary, together
//! with why they matched on a key boundary and why the section-scoped lookup was needed for a
//! half-migrated file. [`Migration::line`] still accepts `0` for "could not be located", which
//! is what a pattern about a MISSING key reports.

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

use crate::defaults::{
    DEFAULT_MAGI_BALTHASAR_LINEAGE, DEFAULT_MAGI_CASPAR_LINEAGE, DEFAULT_MAGI_MELCHIOR_LINEAGE,
};

use super::MagiSectionConfig;

/// Source version of the migration the message frame describes.
///
/// Retargeted from `v0.11.0` when the v0.13.0 break took over the frame: the version pair belongs
/// to whichever break the pass is actually reporting, and a correct list of incompatibilities under
/// a header naming the wrong pair sends the user to migrate something they already migrated.
const VERSION_FROM: &str = "v0.12.0";

/// Target version of the migration the message frame describes.
///
/// Retargeted together with [`VERSION_FROM`] — the pair only ever moves as a pair.
const VERSION_TO: &str = "v0.13.0";

/// Unconditional note about the jump from two generations back.
///
/// **There is NO detection for a generation older than the declared set**: a file that old may
/// additionally carry earlier incompatibilities that nobody audited, and it would receive the
/// generic error exactly when it most needs help. Supporting two generations would double the
/// debt for a jump the user makes in two steps.
///
/// Retargeted with the rest of the frame: under v0.13.0 the generation two steps back is v0.11.x,
/// whose own keys (`provider = "openai"`, `[openai].base_url`,
/// `[headless].tool_result_cap_bytes`) no longer have patterns of their own (REQ-R22 retired them)
/// and now surface as plain serde errors.
const V0_11_X_NOTE: &str =
    "If you're coming from v0.11.x, migrate to v0.12.0 first and then to v0.13.0: this pass only knows\nwhat changed in v0.13.0, and a v0.11.0 file also gets plain serde errors for its own keys.";

/// Line number reported for a key that is **missing**.
///
/// A key the file never wrote has no line to point at — this is the `0` case
/// [`Migration::line`] documents, and the reason the line-locating helpers of the retired pattern
/// set were not needed here.
const MISSING_KEY_HAS_NO_LINE: usize = 0;

/// Trailing hint appended to every lineage correction.
///
/// One shared literal because the same sentence is true of all three seats, and because a
/// correction repeated three times with three slightly different wordings reads like three
/// different rules.
const LINEAGE_CORRECTION_HINT: &str =
    "   # any label you choose; no two seats may share one, and it is never inferred";

/// Backup advice, in the body of the error and not only in the CHANGELOG.
/// Whoever hits this error got here **by starting the binary**, not by reading release notes.
/// It is the only moment when they can still make the copy — that is, before editing.
const BACKUP_ADVISORY: &str =
    "Save a copy of your magi.toml BEFORE editing it: this migration is one-way.";

/// A minimal and valid `magi.toml`, ready to paste.
///
/// **It goes in the body of the error and not in `docs/magi.toml.example`**: whoever installed
/// with `cargo install` or downloaded a release binary does NOT have the example file, and
/// without an escape flag (REQ-A23) this message is the only defense. It is six lines.
///
/// Unlike the rest of the frame, this one is **not** dormant: it is checked against the live
/// schema by `the_minimal_config_the_error_hands_out_actually_parses_today`, so it cannot rot
/// silently.
///
/// **Revisited for the v0.13.0 break and deliberately left unchanged.** It declares no seat model,
/// so it owes no lineage and stays valid under the new rule — which is also what makes it the
/// escape hatch it is meant to be: whoever cannot decide three lineage labels right now can delete
/// their `[magi]` block, run on the built-in trio, and come back to it. The per-key corrections
/// above it are what teach the declaration; this is what unblocks the user who wants to start.
const MINIMAL_VALID_CONFIG: &str = "provider = \"ollama\"\nbase_url = \"http://localhost:11434/v1\"\n\n[openai]\nmodel = \"kimi-k2.6:cloud\"\n";

/// A detected migration incompatibility, with its correction.
/// The shape is the contract shared by [`detect_migrations`] and [`render_migration_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Affected key, as it appears in the file (e.g., `"[openai].base_url"`).
    pub key: &'static str,
    /// 1-indexed line where it was found, for the message. `0` if it could not be located.
    pub line: usize,
    /// Correction text, already redacted if the original value carried credentials.
    pub correction: String,
}

/// Detects the migration patterns declared for the current generation in a raw `magi.toml`.
///
/// **No PRE-parse pattern is declared today** (REQ-R22, SC-R20): the v0.11.0 set retired with
/// v0.13.0, so this reports nothing for every input, and a v0.11.0 file gets serde's own errors
/// for its own keys.
///
/// **That is not the same as saying nothing is reported.** The v0.13.0 break is about keys the
/// file is MISSING, which serde never aborts on, so it is collected after parsing by
/// [`missing_seat_lineages`] — through the same [`Migration`] contract and the same
/// [`render_migration_error`] frame. This half of the pass stays for the next break that arrives
/// as **unknown** keys, where reading the document before serde does is the only way to collect
/// more than one.
///
/// The parameter keeps its place for the same reason, with the leading underscore recording that
/// an empty table reads nothing. A reloaded table reads the document as a **parsed TOML value**,
/// never by textual search: a syntactically broken file must get its syntax error, with line and
/// column, instead of advice about a shape nobody can tell which one it is (SC-A21g).
///
/// Whatever it reports must be **only what that file has wrong**: a half-migrated file receives
/// only the missing correction (SC-A21h). Repeating one the user already applied would make them
/// doubt whether they applied it correctly, which is the opposite mental state from what this
/// message aims for.
#[must_use]
pub fn detect_migrations(_raw: &str) -> Vec<Migration> {
    Vec::new()
}

/// One seat's lineage declaration, as this pass needs to see it.
///
/// The four pieces travel together on purpose: two parallel arrays (one of values, one of key
/// names) would let a later edit reorder one and not the other, and the resulting message would
/// name the wrong seat while every test still passed.
struct SeatLineage<'a> {
    /// Model declared for this seat, if any. Absent ⇒ the seat runs the built-in model, whose
    /// lineage is built in too, so nothing is owed.
    model: Option<&'a str>,
    /// Lineage declared for this seat, if any.
    lineage: Option<&'a str>,
    /// Fully-qualified key, so the message says WHERE the line goes, not only what it is called.
    key: &'static str,
    /// The bare `key = "value"` line the user has to add, without the section prefix.
    declaration: &'static str,
    /// Label shown in the correction — the lineage of the model this seat runs by default, so
    /// what `magi init` writes and what the error suggests cannot drift apart.
    example: &'static str,
}

/// Whether a declared text value counts as **absent**.
///
/// Blank is absent, never invalid — the rule every text key in this file has followed since MS2.
/// An exported-but-unfilled variable in a CI script is an everyday accident, and answering it with
/// an "invalid value" error the user cannot act on punishes the accident instead of guiding it.
fn is_absent(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim().is_empty())
}

/// Detects the v0.12.0 → v0.13.0 break: seats that declare a model but not its lineage.
///
/// **Why this one is checked AFTER parsing while [`detect_migrations`] runs before.** The two
/// halves of the reporting problem are not the same shape. An **unknown** key aborts serde on the
/// first one it meets, so collecting them all requires reading the document before serde does. A
/// **missing** key with an `Option` field does not abort serde at all — it produces no complaint
/// whatsoever — so there is nothing to get ahead of, and reading the typed struct is strictly
/// better than re-reading the raw document: a rename of `melchior_lineage` breaks this function's
/// compile, whereas a string-keyed pattern would keep matching nothing, silently. That is the
/// exact debt REQ-R22 just retired, and re-incurring it one release later would be the wrong
/// lesson to draw from the retirement.
///
/// It reports **only what the file still lacks** (SC-A21h): a half-migrated file gets the two
/// corrections it is missing and no mention of the seat it already fixed. Repeating a correction
/// the user already applied makes them doubt whether they applied it correctly.
///
/// # Redaction (SC-A21e)
///
/// Nothing here echoes a value read from the file: the corrections are built from the built-in
/// lineage labels, not from the user's models or lineages. The rule the module docs state — any
/// correction that echoes a value read from the file is redacted first — is therefore honoured by
/// having nothing to redact, which is the cheapest way to honour it. **A later pattern that wants
/// to quote the user's own value must route it through `redact_url`/`redact_foreign_text` first**;
/// a migration message that leaks a credential to the terminal, the scrollback and CI logs is a
/// worse problem than a line the user completes by hand.
#[must_use]
pub fn missing_seat_lineages(magi: &MagiSectionConfig) -> Vec<Migration> {
    let seats = [
        SeatLineage {
            model: magi.melchior_model.as_deref(),
            lineage: magi.melchior_lineage.as_deref(),
            key: "[magi].melchior_lineage",
            declaration: "melchior_lineage",
            example: DEFAULT_MAGI_MELCHIOR_LINEAGE,
        },
        SeatLineage {
            model: magi.balthasar_model.as_deref(),
            lineage: magi.balthasar_lineage.as_deref(),
            key: "[magi].balthasar_lineage",
            declaration: "balthasar_lineage",
            example: DEFAULT_MAGI_BALTHASAR_LINEAGE,
        },
        SeatLineage {
            model: magi.caspar_model.as_deref(),
            lineage: magi.caspar_lineage.as_deref(),
            key: "[magi].caspar_lineage",
            declaration: "caspar_lineage",
            example: DEFAULT_MAGI_CASPAR_LINEAGE,
        },
    ];

    seats
        .into_iter()
        .filter(|seat| !is_absent(seat.model) && is_absent(seat.lineage))
        .map(|seat| Migration {
            key: seat.key,
            line: MISSING_KEY_HAS_NO_LINE,
            correction: format!(
                "{} = \"{}\"{LINEAGE_CORRECTION_HINT}",
                seat.declaration, seat.example
            ),
        })
        .collect()
}

/// Renders the full migration error from the found incompatibilities.
///
/// The message is **self-contained** and does not send the user to any file in the repo: it
/// includes each correction, the backup advice, a minimal valid `magi.toml` to paste, and the
/// unconditional note about the older generation. Whoever installed via `cargo install` or
/// downloaded a binary **does not have** the source tree, and sending them there leaves them just
/// as stuck as no message.
#[must_use]
pub fn render_migration_error(found: &[Migration]) -> String {
    let mut out = format!(
        "error: .magi/magi.toml is not compatible with magi-rs {VERSION_TO} (coming from {VERSION_FROM})\n\n"
    );

    for m in found {
        if m.line == 0 {
            out.push_str(&format!("  {}\n           {}\n\n", m.key, m.correction));
        } else {
            out.push_str(&format!(
                "  line {}  {}\n           {}\n\n",
                m.line, m.key, m.correction
            ));
        }
    }

    out.push_str(BACKUP_ADVISORY);
    // Interpolated, never spelled out: a hardcoded version here would keep naming the previous
    // break while the frame constants above moved on, which is the failure this whole frame exists
    // to avoid.
    out.push_str(&format!("\n\nA minimal, valid {VERSION_TO} magi.toml:\n\n"));
    out.push_str(MINIMAL_VALID_CONFIG);
    out.push('\n');
    out.push_str(V0_11_X_NOTE);
    out.push('\n');
    out
}

/// Unit tests for the retired pattern table and the surviving reporting mechanism.
///
/// SC-R20: the v0.11.0 patterns are gone; the pass that reports incompatibilities is not.
#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A21d, second half: **what the message proposes parses without error today**.
    ///
    /// This is the part that makes the message useful and the part most likely to rot: the
    /// minimal TOML is a literal, so nothing ties it to the schema except this test. If a later
    /// task changes a key, the advice we give the stuck user stops working — and without this,
    /// silently.
    #[test]
    fn the_minimal_config_the_error_hands_out_actually_parses_today() {
        super::super::MagiConfig::from_toml_str(MINIMAL_VALID_CONFIG)
            .expect("the minimal magi.toml the error proposes must parse today");
    }

    /// A `magi.toml` carrying **all three** retired v0.11.0 patterns at once: `provider =
    /// "openai"` at the root, `[openai].base_url` and `[headless].tool_result_cap_bytes`.
    ///
    /// One shared literal because both retirement tests need the same file and a second copy
    /// could drift from this one, which would let one of them keep passing for the wrong reason.
    const V0_11_0_SAMPLE: &str = "provider = \"openai\"\n\
                                  [openai]\n\
                                  base_url = \"http://localhost:11434/v1\"\n\
                                  [headless]\n\
                                  tool_result_cap_bytes = 65536\n";

    /// SC-R20: a v0.11.0 file no longer receives ITS guided error — the patterns retired with the
    /// milestone that owed them. It gets generic serde errors for its own keys instead, and the
    /// CHANGELOG says so.
    ///
    /// **What this test can and cannot catch (B16 honesty).** Once the pattern table is empty,
    /// [`detect_migrations`] reports nothing for *any* input, so no mutation of the current body
    /// turns this red — it is a **regression pin**, not a behavioral guardian: it goes red the day
    /// someone re-adds a v0.11.0 pattern to the table (verified by mutation: restoring the
    /// `provider = "openai"` arm alone makes it fail).
    #[test]
    fn the_three_v0_11_0_patterns_are_no_longer_detected() {
        let found = detect_migrations(V0_11_0_SAMPLE);
        assert!(
            found.is_empty(),
            "v0.11.0 patterns must no longer be detected, got {found:?}"
        );
    }

    /// The retirement removes the **patterns**, never the **pass**: [`detect_migrations`] and
    /// [`render_migration_error`] keep their signatures and their behavior, so a later task can
    /// reload the table with a new generation's patterns without rebuilding the reporting side.
    ///
    /// **Why the emptiness precondition is part of THIS test and not just a duplicate of the one
    /// above.** It is what makes the assertion a *survival* claim — "the renderer still works even
    /// though the pass itself now declares no pattern" — instead of a plain rendering test that
    /// would have been just as green before the retirement. It is also what makes this test red
    /// before Green, so the guardian was watched failing like any other.
    ///
    /// Mutation that verifies it: deleting the per-migration loop, the backup advisory or the
    /// minimal-config block from [`render_migration_error`] turns it red.
    #[test]
    fn the_reporting_mechanism_still_renders_after_the_patterns_are_retired() {
        assert!(
            detect_migrations(V0_11_0_SAMPLE).is_empty(),
            "precondition: no pattern is declared today"
        );

        let rendered = render_migration_error(&[Migration {
            key: "[magi].melchior_lineage",
            line: 7,
            correction: "melchior_lineage = \"alibaba\"".to_owned(),
        }]);

        assert!(
            rendered.contains("[magi].melchior_lineage"),
            "the caller's key must reach the message: {rendered}"
        );
        assert!(
            rendered.contains("line 7"),
            "the caller's line must reach the message: {rendered}"
        );
        assert!(
            rendered.contains("melchior_lineage = \"alibaba\""),
            "the caller's correction must reach the message: {rendered}"
        );
        assert!(
            rendered.contains(BACKUP_ADVISORY),
            "the backup advisory is part of the frame, not of any pattern: {rendered}"
        );
        assert!(
            rendered.contains(MINIMAL_VALID_CONFIG),
            "so is the ready-to-paste minimal config: {rendered}"
        );
    }

    /// A `magi.toml` exactly as v0.12.0 wrote it: the three seats declared **by model**, and not
    /// one lineage anywhere.
    ///
    /// One shared literal for every test of the v0.13.0 break, for the same reason
    /// [`V0_11_0_SAMPLE`] is shared: a second copy could drift from this one and let a test keep
    /// passing for the wrong reason.
    const V0_12_0_SAMPLE: &str = "[magi]\n\
                                  melchior_model = \"qwen3.5:397b-cloud\"\n\
                                  balthasar_model = \"gpt-oss:120b-cloud\"\n\
                                  caspar_model = \"deepseek-v4-pro:cloud\"\n";

    /// SC-R47: the break is reported WHOLE, never one incompatibility per start.
    ///
    /// That is the mechanism's reason to exist: serde reports **one** missing field at a time (and
    /// `deny_unknown_fields` aborts on the FIRST unknown key), so without a pass that collects them
    /// all the user pays three edit-start-fail cycles for one migration.
    ///
    /// The variant assertion is not decoration: it is what separates "the file failed" from "the
    /// file failed through the GUIDED path". A bare serde error would satisfy `expect_err` and
    /// leave the user exactly as stuck as SC-R46 forbids.
    #[test]
    fn all_three_missing_lineages_are_reported_in_one_message() {
        let err = super::super::MagiConfig::from_toml_str(V0_12_0_SAMPLE)
            .expect_err("a v0.12.0 magi.toml must not start under v0.13.0");
        assert!(
            matches!(err, super::super::ConfigError::NeedsMigration(_)),
            "the break must take the guided path, not serde's generic one: {err}"
        );
        let msg = err.to_string();
        for key in ["melchior_lineage", "balthasar_lineage", "caspar_lineage"] {
            assert!(msg.contains(key), "the message must name {key}: {msg}");
        }
    }

    /// SC-R46: and it must EXPLAIN — naming a key the user has never seen, without showing the
    /// line that declares it, only tells them they are stuck in more words.
    #[test]
    fn the_v0_12_0_error_says_how_to_declare_the_missing_keys() {
        let msg = super::super::MagiConfig::from_toml_str(V0_12_0_SAMPLE)
            .expect_err("a v0.12.0 magi.toml must not start under v0.13.0")
            .to_string();
        for key in ["melchior_lineage", "balthasar_lineage", "caspar_lineage"] {
            assert!(
                msg.contains(&format!("{key} = \"")),
                "the message must show HOW to declare {key}, not just that it is missing: {msg}"
            );
        }
    }

    /// SC-A21h: a half-migrated file is told **only what it still lacks**.
    ///
    /// Repeating a correction the user already applied makes them doubt whether they applied it
    /// correctly, which is the opposite of the mental state this message exists to produce.
    #[test]
    fn a_half_migrated_file_is_told_only_what_it_still_lacks() {
        let half = "[magi]\n\
                    melchior_model = \"qwen3.5:397b-cloud\"\n\
                    melchior_lineage = \"alibaba\"\n\
                    balthasar_model = \"gpt-oss:120b-cloud\"\n\
                    caspar_model = \"deepseek-v4-pro:cloud\"\n";
        let msg = super::super::MagiConfig::from_toml_str(half)
            .expect_err("two lineages are still missing")
            .to_string();
        assert!(
            !msg.contains("melchior_lineage"),
            "the seat that IS declared must not be mentioned at all: {msg}"
        );
        assert!(
            msg.contains("balthasar_lineage") && msg.contains("caspar_lineage"),
            "the two that are still missing must both appear: {msg}"
        );
    }

    /// SC-R19: blank is **absent**, never invalid — the same rule every text key has followed
    /// since MS2. An exported-but-unfilled variable in a CI script is an everyday accident, and a
    /// blank lineage must therefore land in the guided migration message rather than in a separate
    /// "invalid value" error the user cannot act on.
    #[test]
    fn a_blank_lineage_is_absent_and_therefore_reported_as_missing() {
        let blank = "[magi]\n\
                     melchior_model = \"qwen3.5:397b-cloud\"\n\
                     melchior_lineage = \"   \"\n";
        let msg = super::super::MagiConfig::from_toml_str(blank)
            .expect_err("a blank lineage is an absent lineage")
            .to_string();
        assert!(
            msg.contains("melchior_lineage = \""),
            "a blank lineage must get the same guided correction as an absent one: {msg}"
        );
    }

    /// The **scope fence** of the new rule, and the reason it is phrased per seat: a file that
    /// declares no seat model declares no lineage either, and must keep starting.
    ///
    /// **What it can and cannot catch (B16 honesty).** It is green before Green, because today
    /// nothing rejects these files. Its job is the opposite direction: it turns red the moment the
    /// implementation over-reaches — demanding lineages from every `[magi]` section, or from every
    /// file — which would break every user who never configured the trio, for a break that is not
    /// theirs.
    #[test]
    fn a_file_that_declares_no_seat_model_is_not_broken_by_the_new_rule() {
        super::super::MagiConfig::from_toml_str("")
            .expect("an empty magi.toml still means 'all defaults'");
        super::super::MagiConfig::from_toml_str("provider = \"ollama\"\n")
            .expect("a file that never configures the trio has no lineage to declare");
        super::super::MagiConfig::from_toml_str("[magi]\nauto_approve = true\n")
            .expect("a [magi] section without seat models declares no seat to give a lineage to");
    }

    /// The message frame must name **this** break, not the retired one.
    ///
    /// A correct list of incompatibilities under a header naming the wrong pair of versions is
    /// worse than no message: it tells the user their file is a generation older than it is, and
    /// sends them to migrate something they already migrated.
    ///
    /// Mutation that verifies it: reverting `VERSION_TO` to `"v0.12.0"` (or `VERSION_FROM` to
    /// `"v0.11.0"`) turns it red.
    #[test]
    fn the_message_frame_names_the_versions_of_the_break_it_reports() {
        let msg = super::super::MagiConfig::from_toml_str(V0_12_0_SAMPLE)
            .expect_err("a v0.12.0 magi.toml must not start under v0.13.0")
            .to_string();
        assert!(
            msg.contains("magi-rs v0.13.0 (coming from v0.12.0)"),
            "the frame must name the break it is actually reporting: {msg}"
        );
    }

    /// SC-A21g: a syntactically broken file gets its **syntax error**, with line and column, not
    /// advice about a shape nobody can tell which one it is.
    ///
    /// Green before Green, like the scope fence above: it turns red if the implementation ever
    /// starts guessing at text it could not parse.
    #[test]
    fn a_syntactically_broken_file_gets_its_syntax_error_not_migration_advice() {
        let err = super::super::MagiConfig::from_toml_str("[magi\nmelchior_model = ")
            .expect_err("broken TOML must not parse");
        assert!(
            matches!(err, super::super::ConfigError::Parse(_)),
            "a syntax error must stay a syntax error: {err}"
        );
    }
}
