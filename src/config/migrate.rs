// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! PRE-parse validation pass, to report all migration incompatibilities of a `magi.toml`
//! together (REQ-A21b).
//!
//! `deny_unknown_fields` **aborts on the FIRST unknown key**, so "all together" is impossible
//! to achieve from serde's error. This pass reads the TOML as a generic document **before**
//! deserializing, collects the patterns it knows, and emits a single message.
//!
//! # Every pattern SET has a retirement date; the MECHANISM has none
//!
//! The v0.11.0 set — `provider = "openai"`, `[openai].base_url`,
//! `[headless].tool_result_cap_bytes` — was retired in v0.13.0 (MS3, REQ-R22 / SC-R20). That
//! debt was dated on purpose: those patterns duplicated schema knowledge about a shape that is
//! no longer in the code, so they could only rot.
//!
//! **The mechanism carries no such date, and retiring it with them would be the wrong reading of
//! that note.** Its reason to exist outlives any single generation: `deny_unknown_fields` aborts
//! on the FIRST unknown key and serde reports one missing field at a time, so without a pass
//! that collects them all, the user pays one edit-start-fail cycle per incompatibility — start,
//! read one complaint, edit, start again. That is true of every future break, not only of the
//! one that motivated the pass.
//!
//! # What the next author needs to know before reloading the table
//!
//! **The table is empty today.** [`detect_migrations`] reports nothing for every input, and
//! [`render_migration_error`] is reachable only from a caller that supplies its own
//! [`Migration`] values. `MagiConfig::from_toml_str` still runs the pass first, so reloading the
//! table is enough to bring the guided error back — no call site has to change.
//!
//! **The message frame still describes the v0.11.0 → v0.12.0 break.** `VERSION_FROM`,
//! `VERSION_TO`, `MINIMAL_VALID_CONFIG` and `V0_10_X_NOTE` were left as they were rather than
//! guessed forward, because the wording of the next break belongs to the task that declares its
//! patterns. Retarget them in the same change that reloads the table, or the user gets a correct
//! list of incompatibilities under a header naming the wrong pair of versions.
//!
//! **Any correction that echoes a value read from the file must be redacted first.** The retired
//! `[openai].base_url` correction pasted the user's own URL back, so it went through
//! `redact_url`/`locate_userinfo` and said, in the message, that the value was masked (SC-A21e).
//! A migration message that leaks a credential to the terminal, the scrollback and CI logs is a
//! worse problem than a line the user has to complete by hand. That rule belongs to the
//! mechanism, not to the pattern that first needed it.
//!
//! **The line-locating helpers went with their patterns**, rather than staying behind an
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

/// Source version of the migration the message frame describes.
///
/// Frame content of the RETIRED v0.11.0 set, kept dormant on purpose — see the module docs:
/// whoever reloads the pattern table retargets this pair in the same change.
const VERSION_FROM: &str = "v0.11.0";

/// Target version of the migration the message frame describes.
///
/// Same dormant frame content as [`VERSION_FROM`]; retarget both together.
const VERSION_TO: &str = "v0.12.0";

/// Unconditional note about the jump from two generations back.
///
/// **There is NO detection for a generation older than the declared set**: a file that old may
/// additionally carry earlier incompatibilities that nobody audited, and it would receive the
/// generic error exactly when it most needs help. Supporting two generations would double the
/// debt for a jump the user makes in two steps.
///
/// Dormant frame content of the retired v0.11.0 set, like [`VERSION_FROM`].
const V0_10_X_NOTE: &str =
    "If you're coming from v0.10.x, migrate to v0.11.0 first and then to v0.12.0: this pass only knows\nthe v0.11.0 patterns.";

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
/// silently while the table is empty.
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
/// **No pattern is declared today** (REQ-R22, SC-R20): the v0.11.0 set retired with v0.13.0, so
/// this reports nothing for every input, and a v0.11.0 file gets serde's own errors for its own
/// keys. The signature, the [`Migration`] contract and [`render_migration_error`] survive that
/// retirement deliberately — see the module docs.
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
    out.push_str("\n\nA minimal, valid v0.12.0 magi.toml:\n\n");
    out.push_str(MINIMAL_VALID_CONFIG);
    out.push('\n');
    out.push_str(V0_10_X_NOTE);
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
}
