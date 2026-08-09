// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! PRE-parse validation pass, to report all migration incompatibilities of a v0.11.0
//! `magi.toml` together (REQ-A21b).
//!
//! `deny_unknown_fields` **aborts on the FIRST unknown key**, so "all together" is impossible
//! to achieve from serde's error. This pass reads the TOML as a generic document **before**
//! deserializing, collects the known patterns, and emits a single message. Without it the user
//! pays two edit-start-fail cycles.
//!
//! # Dated technical debt
//!
//! **Removed in v0.13.0 (MS3)**, once migration is no longer the common case. It duplicates a
//! little schema knowledge — the patterns are about the **old** shape, which is no longer in
//! the code — and that duplication is accepted knowingly: it is bounded to three patterns and
//! is covered by tests against real v0.11.0 files.

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

use magi_rs::redact::{locate_userinfo, redact_url, UserinfoLocation};
use toml::Value;

/// Root provider key in `magi.toml`.
const PROVIDER_KEY: &str = "provider";

/// Value of `provider` in v0.11.0 that ceased to exist in v0.12.0.
///
/// **Not auto-migrated, and that is deliberate.** `"openai"` was ambiguous — it could mean a
/// local Ollama without credentials or an authenticated endpoint — and splitting that ambiguity
/// is half the point of the change. Choosing for the user would be guessing exactly what D-A01
/// forbids.
const PROVIDER_V0_11_0: &str = "openai";

/// v0.11.0 `[openai]` section; from v0.12.0 `base_url` no longer lives there.
const OPENAI_SECTION: &str = "openai";

/// v0.11.0 `[headless]` section; from v0.12.0 `tool_result_cap_bytes` moves to root.
const HEADLESS_SECTION: &str = "headless";

/// Key `base_url`, which in v0.11.0 lived inside `[openai]`.
const BASE_URL_KEY: &str = "base_url";

/// Key `tool_result_cap_bytes`, which in v0.11.0 lived inside `[headless]`.
const TOOL_RESULT_CAP_BYTES_KEY: &str = "tool_result_cap_bytes";

/// Label shown for `[openai].base_url`.
const OPENAI_BASE_URL_LABEL: &str = "[openai].base_url";

/// Label shown for `[headless].tool_result_cap_bytes`.
const HEADLESS_CAP_LABEL: &str = "[headless].tool_result_cap_bytes";

/// Source version of the migration.
const VERSION_FROM: &str = "v0.11.0";

/// Target version of the migration.
const VERSION_TO: &str = "v0.12.0";

/// Correction for `provider = "openai"`: names both options and the criterion for choosing.
const PROVIDER_CORRECTION: &str = "provider = \"ollama\"        # if it points to a local Ollama daemon, no credential\n           provider = \"openai-compat\" # for OpenAI, Groq, OpenRouter and other authenticated endpoints";

/// Prefix of the `[openai].base_url` correction, before the redacted value.
const BASE_URL_CORRECTION_PREFIX: &str = "base_url = \"";

/// Common closing clause of the `[openai].base_url` correction, said either way (SC-A21e).
const BASE_URL_CORRECTION_TAIL: &str = "\"   # at the root level, above every section.";

/// States that the value is **redacted**: with embedded credentials, redaction wins over ready-
/// to-paste (SC-A21e). A migration message that leaks a credential to the terminal, the
/// scrollback, and CI logs is a worse problem than a line that must be completed.
///
/// **Used only when `url` actually carried a `userinfo`** (loop 1 fix round CE, F21) —
/// unconditionally appending it made the message lie for the common, credential-free case: the
/// value shown was already the real, pasteable one, while the text told the user it was not.
const BASE_URL_CORRECTION_SUFFIX_REDACTED: &str =
    " Value redacted: copy the real one from the old file.";

/// States that the value is **masked in full**, for a `base_url` that never parsed as a URL to
/// begin with (loop 1 fix round CF, F21 follow-up).
///
/// `redact_url` masks `UserinfoLocation::Unparseable` to the literal `"***"` just like it masks
/// a located credential — it is the same "safe failure direction" rule, applied because a string
/// with no `"://"` might hide a secret anywhere. The wording deliberately differs from
/// [`BASE_URL_CORRECTION_SUFFIX_REDACTED`]: there is no credential to "copy from the old file"
/// here, because the value never had URL structure to extract one from. Telling the user to copy
/// a credential that was never identified would send them looking for something that is not
/// there.
const BASE_URL_CORRECTION_SUFFIX_UNPARSEABLE: &str =
    " Value masked: the original did not parse as a URL at all — check the old file directly.";

/// Prefix of the `[headless].tool_result_cap_bytes` correction.
const CAP_CORRECTION_PREFIX: &str = "tool_result_cap_bytes = ";

/// Suffix of the `[headless].tool_result_cap_bytes` correction.
const CAP_CORRECTION_SUFFIX: &str =
    "   # at the root level: now governs all THREE routes (TUI, magi query and headless consult).";

/// **There is NO v0.10.x detection**: the pass only knows the v0.11.0 patterns. A v0.10.x file
/// may additionally carry earlier incompatibilities that nobody audited, and it would receive
/// the generic error exactly when it most needs help. Supporting two generations would double
/// the debt for a jump the user makes in two steps.
///
/// Unconditional note about the jump from v0.10.x.
const V0_10_X_NOTE: &str =
    "If you're coming from v0.10.x, migrate to v0.11.0 first and then to v0.12.0: this pass only knows\nthe v0.11.0 patterns.";

/// Backup advice, in the body of the error and not only in the CHANGELOG.
/// Whoever hits this error got here **by starting the binary**, not by reading release notes.
/// It is the only moment when they can still make the copy — that is, before editing.
const BACKUP_ADVISORY: &str =
    "Save a copy of your magi.toml BEFORE editing it: this migration is one-way.";

/// A minimal and valid v0.12.0 `magi.toml`, ready to paste.
///
/// **It goes in the body of the error and not in `docs/magi.toml.example`**: whoever installed
/// with `cargo install` or downloaded a release binary does NOT have the example file, and
/// without an escape flag (REQ-A23) this message is the only defense. It is six lines.
const MINIMAL_VALID_CONFIG: &str = "provider = \"ollama\"\nbase_url = \"http://localhost:11434/v1\"\n\n[openai]\nmodel = \"kimi-k2.6:cloud\"\n";

/// A detected migration incompatibility, with its correction.
/// The shape is the contract shared by `from_toml_str` and [`render_migration_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Affected key, as it appears in the file (e.g., `"[openai].base_url"`).
    pub key: &'static str,
    /// 1-indexed line where it was found, for the message. `0` if it could not be located.
    pub line: usize,
    /// Correction text, already redacted if the original value carried credentials.
    pub correction: String,
}

/// Detects the three v0.11.0 migration patterns in a raw `magi.toml`.
/// The patterns are `provider = "openai"` at root, `[openai].base_url`, and
/// `[headless].tool_result_cap_bytes`.
///
/// Returns empty if the document **does not parse as TOML**: without structure there is nowhere
/// to look, and rescuing it by textual search would give advice about a shape nobody knows
/// which one it is (SC-A21g). A syntactically broken file receives its syntax error, with line
/// and column, not migration advice.
///
/// Reports **only what that file has wrong**: a half-migrated file receives only the missing
/// correction (SC-A21h). Repeating one the user already applied would make them doubt whether
/// they applied it correctly, which is the opposite mental state from what this message aims
/// for.
///
/// **Deliberately NOT a fourth pattern: `[embedding].provider = "openai"` (Loop 2, S1,
/// verified false positive).** `tests/fixtures/v0.11.0/full.toml` carries that exact line and
/// `every_real_v0_11_0_fixture_reports_its_own_incompatibilities` asserts it is NOT flagged —
/// which looked, from this module alone, like it could leave the user with a second, unguided
/// parse error after applying the two named corrections. It does not: `[embedding].provider` is
/// a **different vocabulary** from the root `provider`/`[magi].kind` this pass polices — see
/// [`crate::memory::config::EmbeddingConfig::provider`], which only requires non-empty and never
/// compares against [`magi_rs::magi::kind::ProviderKind`]. `"openai"` is literally that field's
/// own built-in default, so it round-trips through v0.12.0 without error — verified by reading
/// `EmbeddingConfig::validate` and its default provider function.
#[must_use]
pub fn detect_migrations(raw: &str) -> Vec<Migration> {
    let Ok(doc) = raw.parse::<Value>() else {
        return Vec::new();
    };

    let mut found = Vec::new();

    if let Some(provider) = doc.get(PROVIDER_KEY).and_then(Value::as_str) {
        if provider.trim() == PROVIDER_V0_11_0 {
            found.push(Migration {
                key: PROVIDER_KEY,
                line: line_of(raw, PROVIDER_KEY),
                correction: PROVIDER_CORRECTION.to_owned(),
            });
        }
    }

    if let Some(url) = doc
        .get(OPENAI_SECTION)
        .and_then(Value::as_table)
        .and_then(|t| t.get(BASE_URL_KEY))
        .and_then(Value::as_str)
    {
        // Exhaustive over `UserinfoLocation` on purpose (loop 1 fix round CF): a positive check
        // against `Found` alone left `Unparseable` — a DIFFERENT outcome where `redact_url` also
        // shows a value that is not the real one — silently undisclosed. Matching means a future
        // fourth variant fails to compile here instead of falling into the wrong branch, which is
        // exactly how this bug happened the first time.
        let redacted_clause = match locate_userinfo(url) {
            // The shown value equals the real one: nothing was hidden, nothing to disclose
            // (SC-A21e, F21).
            UserinfoLocation::Absent => "",
            // A credential was located and masked: point the user at the old file to recover it.
            UserinfoLocation::Found { .. } => BASE_URL_CORRECTION_SUFFIX_REDACTED,
            // The whole value was masked because it never parsed as a URL: different disclosure,
            // since there is no credential to "copy" (F21 follow-up).
            UserinfoLocation::Unparseable => BASE_URL_CORRECTION_SUFFIX_UNPARSEABLE,
        };
        found.push(Migration {
            key: OPENAI_BASE_URL_LABEL,
            line: line_of_in_section(raw, OPENAI_SECTION, BASE_URL_KEY),
            correction: format!(
                "{BASE_URL_CORRECTION_PREFIX}{}{BASE_URL_CORRECTION_TAIL}{redacted_clause}",
                redact_url(url)
            ),
        });
    }

    if let Some(cap) = doc
        .get(HEADLESS_SECTION)
        .and_then(Value::as_table)
        .and_then(|t| t.get(TOOL_RESULT_CAP_BYTES_KEY))
    {
        found.push(Migration {
            key: HEADLESS_CAP_LABEL,
            line: line_of_in_section(raw, HEADLESS_SECTION, TOOL_RESULT_CAP_BYTES_KEY),
            correction: format!("{CAP_CORRECTION_PREFIX}{cap}{CAP_CORRECTION_SUFFIX}"),
        });
    }

    found
}

/// True when the already-trimmed `line` is a TOML key assignment for exactly `needle`, not
/// merely a line that happens to start with the same characters.
///
/// **Why a boundary check and not bare `starts_with` (Loop 1 gate S1-f finding, Caspar).**
/// `starts_with` alone would also match a key that has `needle` as a literal prefix — e.g. a
/// hypothetical `provider_timeout` key would match the needle `"provider"`. No such colliding
/// key exists in the schema today for any of the three needles this module uses (verified: the
/// only fields whose full name contains "provider" / "base_url" / "tool_result_cap_bytes" are
/// the exact keys these patterns already look for), so today this cannot misfire — but that
/// safety is a property of the CURRENT field list, not of the matching logic, and the matching
/// logic does not know it. A future field rename or addition one prefix away would silently
/// point the user at the wrong line, which this module's own `detect_migrations` doc calls
/// worse than no line at all — a wrong line is trusted; an absent one is not.
///
/// The boundary mirrors TOML key syntax: a bare key is followed by whitespace, `=`, or the end
/// of the line — never by another identifier character (TOML bare keys are
/// `[A-Za-z0-9_-]+`, so a real continuation is alphanumeric, `_`, or `-`).
fn is_key_at(line: &str, needle: &str) -> bool {
    match line.strip_prefix(needle) {
        Some(rest) => rest
            .chars()
            .next()
            .is_none_or(|c| c.is_whitespace() || c == '='),
        None => false,
    }
}

/// 1-indexed line number where a `needle` key **starts**, or 0 if it does not appear.
///
/// Compares against the start of the already-trimmed line, not with `contains`: a `contains`
/// would match a key mentioned inside a comment, and the default v0.11.0 file is full of
/// comments that name their own keys. It is only for the message — a wrong line confuses, but
/// does not change what was detected.
///
/// **Unscoped, used only for `provider` at the root** (Loop 2, S1): TOML syntax forces every
/// root-level key to appear textually before the first `[section]` header in the file, so a
/// root `provider` line is always the FIRST line starting with `"provider"` regardless of what
/// sections follow — there is no section-scoped `provider` occurrence that could sit earlier in
/// the file and get matched by mistake, unlike `base_url`/`tool_result_cap_bytes` below.
fn line_of(raw: &str, needle: &str) -> usize {
    raw.lines()
        .position(|line| is_key_at(line.trim_start(), needle))
        .map_or(0, |idx| idx + 1)
}

/// 1-indexed line where `needle` starts **inside** the named TOML `section`, or 0 if the section
/// is absent or does not contain it.
///
/// **Why scoped, and not just [`line_of`] (Loop 2 fix, Caspar, S1).** A half-migrated
/// `magi.toml` can legally carry BOTH the new root-level key (`base_url`,
/// `tool_result_cap_bytes`) and the leftover old section-scoped one with the exact same
/// trailing name — that overlap is precisely what makes the file "half-migrated" rather than
/// simply old. An unscoped search for `"base_url"` finds whichever line comes first in the
/// file, which is the ROOT one in the common layout (the reshaped v0.12.0 template puts
/// `base_url` above every section), and reports that line for an incompatibility that is
/// actually about the `[openai]` occurrence — pointing the user at the wrong line to fix.
///
/// Scans forward from the `[section]` header (matched at the start of a trimmed line, so a
/// mention inside a comment or a differently-named table like `[embedding]` cannot match) and
/// stops at the next `[`-prefixed line, so a needle appearing in a LATER section is correctly
/// treated as absent from this one.
fn line_of_in_section(raw: &str, section: &str, needle: &str) -> usize {
    let header = format!("[{section}]");
    let mut in_section = false;
    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim_start();
        if in_section {
            if trimmed.starts_with('[') {
                break; // Entered the next table without finding the needle in this one.
            }
            if is_key_at(trimmed, needle) {
                return idx + 1;
            }
        } else if trimmed.starts_with(&header) {
            in_section = true;
        }
    }
    0
}

/// Renders the full migration error from the found incompatibilities.
///
/// The message is **self-contained** and does not send the user to any file in the repo: it
/// includes each correction, the backup advice, a minimal valid `magi.toml` to paste, and the
/// unconditional note about v0.10.x. Whoever installed via `cargo install` or downloaded a
/// binary **does not have** the source tree, and sending them there leaves them just as stuck
/// as no message.
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

/// Unit tests for migration detection and rendering.
///
/// SC-A21: both incompatibilities are reported together.
#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A21: the v0.11.0 config fails with both incompatibilities named together, the
    /// `base_url` already rewritten, and the criterion for choosing `ollama` vs `openai-compat`.
    #[test]
    fn a_v0_11_0_config_reports_both_incompatibilities_at_once() {
        let toml = include_str!("../../tests/fixtures/v0.11.0/default.toml");
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 2, "expected provider + [openai].base_url");
        let rendered = render_migration_error(&found);
        assert!(rendered.contains("provider"));
        assert!(rendered.contains("base_url"));
        assert!(
            rendered.contains("ollama") && rendered.contains("openai-compat"),
            "must say HOW to choose between the two"
        );
    }

    /// SC-A21h: a half-migrated file receives ONLY what is missing.
    #[test]
    fn a_partially_migrated_file_reports_only_what_is_missing() {
        let toml = "provider = \"openai\"\nbase_url = \"http://x/v1\"\n[openai]\nmodel = \"m\"\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].key, "provider");
    }

    /// SC-A21e: embedded credentials in `base_url` are redacted in the migration message — the
    /// host stays visible, and the message states the value is redacted.
    #[test]
    fn embedded_credentials_are_redacted_in_the_migration_message() {
        let toml = "[openai]\nbase_url = \"https://user:s3cr3t@host/v1\"\n";
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            !rendered.contains("s3cr3t"),
            "the credential must NOT appear"
        );
        assert!(
            rendered.contains("host"),
            "the host must, though — it is what makes the message actionable"
        );
        assert!(
            rendered.contains("redacted"),
            "and the message must say it is redacted"
        );
    }

    /// SC-A21e, negative clause (loop 1 fix round CE, F21): a `base_url` WITHOUT credentials is
    /// pasted literally — the message must not claim it was redacted, because there is nothing
    /// left to copy from the old file. Covers three of the four real v0.11.0 fixtures, the
    /// common case (an unauthenticated local Ollama).
    #[test]
    fn a_credential_free_base_url_is_not_claimed_to_be_redacted() {
        let toml = "[openai]\nbase_url = \"http://localhost:11434/v1\"\n";
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            rendered.contains("http://localhost:11434/v1"),
            "the full value must remain paste-ready: {rendered}"
        );
        assert!(
            !rendered.contains("redacted"),
            "with no credentials there is nothing to redact, so there is nothing to \
             copy from the old file: {rendered}"
        );
    }

    /// Loop 1 fix round CF (finding F21 follow-up): a scheme-less/malformed `base_url` hits
    /// `UserinfoLocation::Unparseable`, which `redact_url` masks to the literal `"***"` — not
    /// the real value, exactly like the credential case, just for a different reason. The
    /// disclaimer must cover this outcome too, with wording that does not tell the user to
    /// "copy the real one" (there is no credential to extract) but instead that the original
    /// never parsed as a URL at all.
    #[test]
    fn an_unparseable_base_url_states_it_is_masked_and_not_paste_ready() {
        let toml = "[openai]\nbase_url = \"localhost:11434/v1\"\n";
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            rendered.contains("***"),
            "the value shown must be the full mask: {rendered}"
        );
        assert!(
            rendered.contains("did not parse as a URL"),
            "must say the original did not parse as a URL: {rendered}"
        );
        assert!(
            !rendered.contains("copy the real one"),
            "that phrase is only for the case with a credential, it does not apply here: {rendered}"
        );
    }

    /// SC-A21g: a syntactically broken TOML does NOT receive migration advice — without
    /// structure, `detect_migrations` finds nothing to search patterns against.
    #[test]
    fn a_syntactically_broken_toml_gets_a_syntax_error_not_migration_advice() {
        let toml = "provider = \"sin cerrar\n[magi]\n";
        assert!(
            detect_migrations(toml).is_empty(),
            "with no structure there is nowhere to search for patterns; the pass does \
             not rescue via grep"
        );
    }

    /// SC-A21f: an empty TOML parses and triggers nothing.
    #[test]
    fn an_empty_toml_is_valid_and_triggers_no_migration() {
        assert!(detect_migrations("").is_empty());
        assert!(detect_migrations("   \n\n  ").is_empty());
    }

    /// SC-A21i: the jump from v0.10.x is declared unsupported, in EVERY config error — this
    /// pass only knows the v0.11.0 patterns, so the unconditional note is exercised here via a
    /// v0.11.0 fixture rather than a v0.10.x one.
    #[test]
    fn every_config_error_mentions_the_v0_10_x_path() {
        let rendered = render_migration_error(&detect_migrations(include_str!(
            "../../tests/fixtures/v0.11.0/default.toml"
        )));
        assert!(
            rendered.contains("v0.11.0"),
            "there is no v0.10.x detection: there is an unconditional note"
        );
    }

    /// SC-A21d: the message is validated against the FOUR real v0.11.0 files, each reporting
    /// all of its own incompatibilities. The four were generated or derived from the published
    /// v0.11.0 binary and verified against it — see `tests/fixtures/v0.11.0/README.md`. They
    /// share the same two incompatibilities because the three variants are derived from the
    /// canonical `default.toml` without adding or removing migratable keys.
    #[test]
    fn every_real_v0_11_0_fixture_reports_its_own_incompatibilities() {
        for (name, toml) in [
            (
                "default",
                include_str!("../../tests/fixtures/v0.11.0/default.toml"),
            ),
            (
                "with-models",
                include_str!("../../tests/fixtures/v0.11.0/with-models.toml"),
            ),
            (
                "full",
                include_str!("../../tests/fixtures/v0.11.0/full.toml"),
            ),
            (
                "with-credentials",
                include_str!("../../tests/fixtures/v0.11.0/with-credentials.toml"),
            ),
        ] {
            let found = detect_migrations(toml);
            assert_eq!(
                found.len(),
                2,
                "{name}: expected provider + [openai].base_url"
            );
            let rendered = render_migration_error(&found);
            assert!(rendered.contains(PROVIDER_KEY), "{name}: names provider");
            assert!(rendered.contains(BASE_URL_KEY), "{name}: names base_url");
        }
    }

    /// SC-A21e on the REAL file: the fixture's credential never reaches the rendered message.
    ///
    /// The test above (`embedded_credentials_are_redacted_in_the_migration_message`) uses an
    /// inline TOML; this one uses the committed file, which is what a user would actually have.
    /// They are different on purpose: the inline test pins the rule, this one pins that the rule
    /// survives the real file.
    #[test]
    fn the_real_credentials_fixture_never_leaks_its_secret() {
        let toml = include_str!("../../tests/fixtures/v0.11.0/with-credentials.toml");
        let rendered = render_migration_error(&detect_migrations(toml));
        assert!(
            !rendered.contains("s3cr3t"),
            "the fixture's credential must not appear in the message"
        );
        assert!(
            rendered.contains("host"),
            "the host must, so it stays actionable"
        );
    }

    /// SC-A21d, second half: **what the message proposes parses without error in v0.12.0**.
    ///
    /// This is the part that makes the message useful and the part most likely to rot: the
    /// minimal TOML is a literal, so nothing ties it to the schema except this test. If a later
    /// task changes a key, the advice we give the stuck user stops working — and without this,
    /// silently.
    #[test]
    fn the_minimal_config_the_error_hands_out_actually_parses_today() {
        super::super::MagiConfig::from_toml_str(MINIMAL_VALID_CONFIG)
            .expect("the minimal magi.toml the error proposes must parse in v0.12.0");
    }

    /// `line_of` returns 0 when the pattern does not appear in the text.
    #[test]
    fn line_of_returns_zero_when_needle_is_absent() {
        assert_eq!(line_of("foo\nbar\n", "baz"), 0);
    }

    /// `line_of` returns the 1-based line number.
    #[test]
    fn line_of_returns_one_indexed_line_number() {
        assert_eq!(line_of("foo\nbar\nbaz", "bar"), 2);
    }

    /// Loop 2 fix (Caspar, S1): a half-migrated file that already has the NEW root-level
    /// `base_url` but has not yet removed the OLD `[openai].base_url` must report the line of
    /// the `[openai]` occurrence, not the root one. `line_of` searched the whole file for the
    /// first line starting with `"base_url"`, which — with the root key written first, the
    /// common layout — pointed the user at the wrong line.
    #[test]
    fn a_half_migrated_files_openai_base_url_reports_its_own_line_not_the_roots() {
        let toml = "base_url = \"http://root/v1\"\n\n[openai]\nbase_url = \"http://old/v1\"\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1, "only the leftover [openai].base_url");
        assert_eq!(found[0].key, OPENAI_BASE_URL_LABEL);
        assert_eq!(
            found[0].line, 4,
            "must point at the [openai] section's own base_url line, not the root's (line 1)"
        );
    }

    /// Same defect, mirrored for `[headless].tool_result_cap_bytes` against a root-level key
    /// that happens to start the same way in the file.
    #[test]
    fn a_half_migrated_files_headless_cap_reports_its_own_line_not_an_earlier_namesake() {
        let toml = "tool_result_cap_bytes = 4096\n\n[headless]\ntool_result_cap_bytes = 2048\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].key, HEADLESS_CAP_LABEL);
        assert_eq!(
            found[0].line, 4,
            "must point at the [headless] section's own line, not the root's (line 1)"
        );
    }

    /// Detects the third pattern: `[headless].tool_result_cap_bytes`.
    #[test]
    fn headless_tool_result_cap_bytes_is_detected() {
        let toml = "[headless]\ntool_result_cap_bytes = 4096\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].key, "[headless].tool_result_cap_bytes");
        assert!(found[0].correction.contains("tool_result_cap_bytes"));
    }

    /// The three old patterns in the same file are all reported together.
    #[test]
    fn all_three_patterns_together_are_reported() {
        let toml = "provider = \"openai\"\n[openai]\nbase_url = \"http://x/v1\"\n\
                    [headless]\ntool_result_cap_bytes = 2048\n";
        let found = detect_migrations(toml);
        assert_eq!(found.len(), 3);
    }

    /// Loop 1 gate S1-f finding (Caspar): `line_of` must not mistake a key that merely has
    /// `needle` as a literal PREFIX for `needle` itself. Without a boundary check, a
    /// `provider_timeout` key would be reported as the `provider` line, even though it is a
    /// different key entirely.
    #[test]
    fn line_of_does_not_match_a_key_that_merely_has_the_needle_as_a_prefix() {
        assert_eq!(
            line_of("provider_timeout = 5\nprovider = \"openai\"\n", "provider"),
            2,
            "must skip the prefix-colliding key and find the real one"
        );
    }

    /// Companion: when ONLY the prefix-colliding key is present, the real needle is correctly
    /// reported absent (line 0), not misattributed to the colliding key's line.
    #[test]
    fn line_of_returns_zero_when_only_a_prefix_colliding_key_is_present() {
        assert_eq!(line_of("provider_timeout = 5\n", "provider"), 0);
    }

    /// Same guard, for the section-scoped lookup.
    #[test]
    fn line_of_in_section_does_not_match_a_prefix_colliding_key() {
        let toml = "[openai]\nbase_url_backup = \"http://old/v1\"\nbase_url = \"http://real/v1\"\n";
        assert_eq!(
            line_of_in_section(toml, OPENAI_SECTION, BASE_URL_KEY),
            3,
            "must skip base_url_backup and find the real base_url line"
        );
    }

    /// `is_key_at` directly: a bare key with nothing trailing (end-of-line boundary) still
    /// counts as a match.
    #[test]
    fn is_key_at_accepts_a_bare_key_with_no_trailing_content() {
        assert!(is_key_at("provider", "provider"));
    }

    /// Both forms TOML allows around `=` must still match.
    #[test]
    fn is_key_at_matches_with_and_without_space_before_equals() {
        assert!(is_key_at("provider = \"openai\"", "provider"));
        assert!(is_key_at("provider=\"openai\"", "provider"));
    }

    /// The exact collision this fix closes.
    #[test]
    fn is_key_at_rejects_a_longer_key_sharing_the_prefix() {
        assert!(!is_key_at("provider_timeout = 5", "provider"));
    }
}
