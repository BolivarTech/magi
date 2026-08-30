// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! `base_url` credentials as **placeholder**, resolved from in-memory vault (REQ-A16c).
//!
//! # Why placeholders and not redaction
//!
//! The previous design accepted the credential in the file and redacted it before displaying
//! it. That leaves security depending on **every** output path remembering to redact, and one
//! was already found that did not: the `toml` parse error quotes the offending line, so a
//! malformed `magi.toml` with `base_url = "https://u:p@host/v1"` spat the credential to stderr
//! and CI logs. Closing that path does not close the class; it closes the path.
//!
//! With placeholders the property is **structural**: if the file cannot contain the secret, no
//! output path can leak it, including those nobody audited. It is the same reason API keys
//! never lived in `magi.toml` (REQ-A14) — `base_url` was the hole through which a credential
//! did get in.

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

use std::fmt;

use crate::redact::{locate_userinfo, redact_url, UserinfoLocation};
use crate::vault::SecretStore;

/// User placeholder. Exact literal, **not** a pattern: this is not a template engine.
const USER_PLACEHOLDER: &str = "[user]";
/// See [`USER_PLACEHOLDER`].
const PASSWORD_PLACEHOLDER: &str = "[password]";

/// The `userinfo` that a credential template must have, exactly.
const EXPECTED_USERINFO: &str = "[user]:[password]";

/// Which `base_url` is being resolved. Determines the prefix of the vault entries.
///
/// Each `base_url` resolves **its own** credentials: two distinct endpoints may have different
/// users, and sharing one entry would silently couple them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Root `base_url` ⇒ `BASE_URL_USER` / `BASE_URL_PASSWORD`.
    Root,
    /// `[magi].base_url` ⇒ `MAGI_BASE_URL_*`.
    Magi,
    /// `[embedding].base_url` ⇒ `EMBEDDING_BASE_URL_*`.
    Embedding,
}

impl Scope {
    /// Name of the vault entry with the user.
    #[must_use]
    pub fn user_entry(self) -> &'static str {
        match self {
            Self::Root => "BASE_URL_USER",
            Self::Magi => "MAGI_BASE_URL_USER",
            Self::Embedding => "EMBEDDING_BASE_URL_USER",
        }
    }

    /// Name of the vault entry with the password.
    #[must_use]
    pub fn password_entry(self) -> &'static str {
        match self {
            Self::Root => "BASE_URL_PASSWORD",
            Self::Magi => "MAGI_BASE_URL_PASSWORD",
            Self::Embedding => "EMBEDDING_BASE_URL_PASSWORD",
        }
    }
}

/// What can go wrong when reading or resolving a `base_url`.
///
/// **No message repeats the offending value**: a security error that prints the secret it
/// is rejecting is useless.
#[derive(Debug, thiserror::Error)]
pub enum EndpointError {
    /// The `base_url` carries a literal credential instead of placeholders.
    #[error(
        "`base_url` carries a literal credential. Replace it with \
         `{USER_PLACEHOLDER}:{PASSWORD_PLACEHOLDER}` and store the values in the vault: \
         `magi-rs vault set {user_entry}` and `magi-rs vault set {password_entry}`"
    )]
    LiteralCredential {
        /// Vault entry for the user, according to scope.
        user_entry: &'static str,
        /// Vault entry for the password, according to scope.
        password_entry: &'static str,
    },

    /// The placeholder is declared and the vault entry does not exist.
    #[error(
        "`base_url` declares a placeholder but entry {entry} is missing from the vault. \
         Create it with `magi-rs vault set {entry}`"
    )]
    MissingVaultEntry {
        /// The missing entry.
        entry: &'static str,
    },

    /// A placeholder that is neither of the two known ones.
    #[error(
        "unknown placeholder in `base_url`: only `{USER_PLACEHOLDER}` and \
         `{PASSWORD_PLACEHOLDER}` are accepted, in the `userinfo` position"
    )]
    UnknownPlaceholder,

    /// The URL could not be traversed, so it cannot be asserted that it does not carry a
    /// secret.
    #[error("`base_url` does not have a recognizable form (`scheme://host/...`)")]
    Unparseable,
}

/// `base_url` **as it is in the file**: with `[user]`/`[password]`, never the secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointTemplate(String);

impl EndpointTemplate {
    /// Reads a `base_url` from the file and rejects any literal credential (REQ-A16c).
    ///
    /// Reuses the authority locator from [`crate::redact`], which is the same one redaction
    /// uses: the rule of where `userinfo` lives is written **once**. If it were written twice,
    /// drifting out of sync would mean one of the two stops seeing a credential.
    ///
    /// `scope` is what makes a [`EndpointError::LiteralCredential`] actionable: it names the
    /// vault entries of the `base_url` actually being parsed (`BASE_URL_USER` for the root,
    /// `MAGI_BASE_URL_USER` for `[magi]`, `EMBEDDING_BASE_URL_USER` for `[embedding]`), not
    /// always the root's — a fix round CE (loop 1, F22) correction: the previous signature took
    /// no scope and always named the root entries, misdirecting an operator fixing `[magi]` or
    /// `[embedding]` to create the wrong vault entry.
    ///
    /// # Errors
    ///
    /// [`EndpointError::LiteralCredential`] if the `userinfo` are not exactly the two
    /// placeholders; [`EndpointError::UnknownPlaceholder`] if it carries another placeholder;
    /// [`EndpointError::Unparseable`] if the URL could not be traversed.
    pub fn parse(raw: &str, scope: Scope) -> Result<Self, EndpointError> {
        match locate_userinfo(raw) {
            // Without `userinfo` there is no credential to validate — the common case pays
            // nothing.
            UserinfoLocation::Absent => Ok(Self(raw.to_string())),
            UserinfoLocation::Unparseable => Err(EndpointError::Unparseable),
            UserinfoLocation::Found { start, end } => {
                let Some(userinfo) = raw.get(start..end) else {
                    return Err(EndpointError::Unparseable);
                };
                if userinfo == EXPECTED_USERINFO {
                    return Ok(Self(raw.to_string()));
                }
                // A misspelled placeholder is named as such; anything else is a literal
                // credential. The distinction matters because the fixes are different.
                if userinfo.contains('[') || userinfo.contains(']') {
                    return Err(EndpointError::UnknownPlaceholder);
                }
                Err(EndpointError::LiteralCredential {
                    user_entry: scope.user_entry(),
                    password_entry: scope.password_entry(),
                })
            }
        }
    }

    /// The template text — **safe by construction, does NOT need redaction**.
    ///
    /// What is here is `https://[user]:[password]@host/v1`: by REQ-A16c a literal credential is
    /// a configuration error, so the template cannot contain a secret. The one that does need
    /// redaction is [`ResolvedEndpoint`], which is the one after.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Replaces the placeholders with the vault values, **in memory**.
    ///
    /// Fails **closed**: a missing entry stops the process by naming it. Substituting empty
    /// would produce a 401 on the first request, with no apparent relation to the configuration
    /// — the same class of late failure that D-A01 eliminated.
    ///
    /// # Errors
    ///
    /// [`EndpointError::MissingVaultEntry`] with the missing entry and the command that creates
    /// it.
    pub fn resolve(
        &self,
        vault: &mut dyn SecretStore,
        scope: Scope,
    ) -> Result<ResolvedEndpoint, EndpointError> {
        // Substitution is limited to the `userinfo` of the AUTHORITY, not the whole string.
        //
        // Searching for the placeholder across the whole text is what the first version did,
        // and a `https://host/v1/[user]` —where `[user]` is a literal path segment— would go
        // look for a credential in the vault and fail closed on an entry nobody had any reason
        // to create. `parse` already uses the same locator; resolving with a different rule
        // would desynchronize them.
        let UserinfoLocation::Found { start, end } = locate_userinfo(&self.0) else {
            // Without `userinfo` there is nothing to substitute, and the common case —local
            // Ollama, keyless— does not pay even a lookup.
            return Ok(ResolvedEndpoint(self.0.clone()));
        };
        let (Some(prefix), Some(userinfo), Some(tail)) = (
            self.0.get(..start),
            self.0.get(start..end),
            self.0.get(end..),
        ) else {
            return Err(EndpointError::Unparseable);
        };
        if userinfo != EXPECTED_USERINFO {
            // INVARIANT, not reachable in practice (Loop 2 note, Melchior, S1): `Self(String)`'s
            // field is private to this module, and the ONLY public constructor is `parse`, which
            // already rejects (as `LiteralCredential`/`UnknownPlaceholder`) any `userinfo` that
            // is not exactly `EXPECTED_USERINFO` before an `EndpointTemplate` can exist. So by
            // construction, every `EndpointTemplate` this method is ever called on already
            // satisfies this check.
            //
            // Kept as a real branch rather than `unreachable!()` or removed: it costs nothing on
            // the happy path, and it is deliberately defensive against a FUTURE change that
            // loosens the invariant (a second constructor, a `From<String>`, a refactor of
            // `parse` that widens what it accepts) without updating this comment — `unreachable!`
            // would turn that same future mistake into a panic instead of a graceful no-op, and a
            // config module reachable from the same run loop as the TUI/headless surfaces should
            // not be the one to introduce a new panic path over what is, worst case, an
            // unexpected but harmless template string.
            return Ok(ResolvedEndpoint(self.0.clone()));
        }

        let user = vault
            .get(scope.user_entry())
            .map_err(|_| EndpointError::MissingVaultEntry {
                entry: scope.user_entry(),
            })?;
        let password =
            vault
                .get(scope.password_entry())
                .map_err(|_| EndpointError::MissingVaultEntry {
                    entry: scope.password_entry(),
                })?;

        // Loop 2 fix (Caspar, S1): PERCENT-ENCODED, never raw. A vault value is an opaque
        // secret the operator chose — nothing here constrains it to characters that are safe
        // to sit unescaped inside a URL authority. `locate_userinfo`/`redact_url` compute the
        // authority window by scanning for `/`, `?`, `#` (terminators) and the LAST `@`
        // (userinfo/host split); a raw `/`, `?`, `#` or extra `@` from the credential shifts
        // that window, which is both a redaction bypass (the truncated "authority" no longer
        // contains an `@`, so `locate_userinfo` returns `Absent` and nothing gets masked) and a
        // mis-route (the actual host substring changes). `urlencoding::encode` escapes every
        // byte outside the RFC 3986 unreserved set (ALPHA / DIGIT / `-` / `.` / `_` / `~`),
        // which includes `/ ? # @ %` — so no substituted byte can ever be mistaken for a URL
        // structural character, regardless of what the operator's secret contains. `%` itself
        // is escaped too, so a literal `%2F` typed into a password cannot smuggle a decoded
        // `/` past a consumer that decodes once.
        // **Through the shared encoder, not a second call to the same crate.**
        // `crate::encoding`'s whole reason to exist is that this side and the
        // auditor's registration must produce the SAME bytes: the auditor scans
        // for the encoded form, and if the two ever diverge it scans for a form
        // that never appears. Two independent `urlencoding::encode` calls are
        // not a shared function -- either one can move to a different `AsciiSet`
        // and nothing stops compiling, while the failure is a credential encoded
        // one way in the URL and hashed another way in the auditor.
        let user_encoded = crate::encoding::percent_encode(user.as_str());
        let password_encoded = crate::encoding::percent_encode(password.as_str());

        // REQ-L49's MAIN case, registered where the value is resolved. A
        // password with reserved characters NEVER appears raw inside a
        // `base_url` -- it is encoded on the way in -- so an auditor that only
        // knew the raw form would be blind in the one place credentials live.
        // `register_process_secrets` composes the encoded variant from the raw
        // value using this same encoder, which is what the paragraph above is
        // guarding.
        // **Named by SCOPE, not by the root's name.** `Scope` already spells the
        // three pairs the vault uses, and the lookup above went through them.
        // Registering a `MAGI_BASE_URL_PASSWORD` under `BASE_URL_PASSWORD`
        // masks the value correctly and then names the wrong config key in the
        // alarm -- sending the reader to an entry that is not the one at fault.
        for short in crate::logging::register_process_secrets(&[
            (
                crate::logging::auditor::SecretName::new(scope.user_entry()),
                user.as_str(),
            ),
            (
                crate::logging::auditor::SecretName::new(scope.password_entry()),
                password.as_str(),
            ),
        ]) {
            eprintln!(
                "warning: {} is too short to be matched exactly in the log; it is still masked by shape, which is weaker.",
                short.as_str()
            );
        }

        let mut out = String::with_capacity(self.0.len());
        out.push_str(prefix);
        out.push_str(user_encoded.as_str());
        out.push(':');
        out.push_str(password_encoded.as_str());
        out.push_str(tail);
        Ok(ResolvedEndpoint(out))
    }
}

impl fmt::Display for EndpointTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// URL with the placeholders already substituted.
///
/// **Only [`EndpointTemplate::resolve`] constructs it**, and resolving requires the vault —
/// that is why an
/// unresolved `&str` cannot reach a provider by accident.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint(String);

impl ResolvedEndpoint {
    /// The effective URL. Whoever displays it must run it through
    /// [`crate::redact::redact_url`].
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ResolvedEndpoint {
    /// Hand-written and redacted: a `derive(Debug)` here is the easiest way to leak the
    /// credential without noticing — a `{:?}` in an error or trace is enough.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResolvedEndpoint({})", redact_url(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::VaultError;
    use crate::vault::{SecretEntry, SecretStore};
    use std::collections::BTreeMap;
    use zeroize::Zeroizing;

    /// Test vault: an in-memory map, no crypto or SQLite.
    struct StubVault {
        /// Available entries.
        entries: BTreeMap<String, String>,
    }

    impl StubVault {
        fn with(pairs: &[(&str, &str)]) -> Self {
            Self {
                entries: pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            }
        }
        fn empty() -> Self {
            Self {
                entries: BTreeMap::new(),
            }
        }
    }

    impl SecretStore for StubVault {
        fn set(&mut self, name: &str, value: &str) -> Result<(), VaultError> {
            self.entries.insert(name.to_string(), value.to_string());
            Ok(())
        }
        fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
            self.entries
                .get(name)
                .map(|v| Zeroizing::new(v.clone()))
                .ok_or_else(|| VaultError::SecretNotFound(name.to_string()))
        }
        fn remove(&mut self, name: &str) -> Result<(), VaultError> {
            self.entries.remove(name);
            Ok(())
        }
        fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
            Ok(Vec::new())
        }
        fn contains(&mut self, name: &str) -> Result<bool, VaultError> {
            Ok(self.entries.contains_key(name))
        }
    }

    /// The URL and the auditor must encode a credential to the SAME bytes.
    ///
    /// `src/encoding.rs`'s module doc states this as the reason it exists, and
    /// until this test it was a claim rather than a property: `endpoint.rs`
    /// called `urlencoding::encode` directly, so the two were independent calls
    /// that happened to agree. Either could have moved to a different
    /// `AsciiSet` with nothing failing to compile, and the failure would be a
    /// credential encoded one way in the URL and hashed another way in the
    /// auditor -- which reads, from the outside, as an auditor that simply does
    /// not catch it.
    ///
    /// Every character in the fixture is one that changes a URL authority's
    /// parse if it survives unencoded, which is what makes them worth pinning.
    #[test]
    fn the_url_and_the_auditor_encode_a_hostile_password_identically() {
        let hostile = "p4ss@word/with?reserved#chars%and:more";
        let mut vault = StubVault::with(&[
            ("BASE_URL_USER", "operator"),
            ("BASE_URL_PASSWORD", hostile),
        ]);
        let template =
            EndpointTemplate::parse("https://[user]:[password]@api.example.com/v1", Scope::Root)
                .expect("parse");
        let resolved = template.resolve(&mut vault, Scope::Root).expect("resolve");

        let shared = crate::encoding::percent_encode(hostile);
        assert!(
            resolved.as_str().contains(shared.as_str()),
            "the URL does not carry the bytes the auditor scans for.\n  url: {}\n  auditor: {shared}",
            resolved.as_str()
        );
        // Without this the assertion above would also hold for an encoder that
        // did nothing at all on both sides.
        assert!(
            !resolved.as_str().contains(hostile),
            "the hostile password survived unencoded: {}",
            resolved.as_str()
        );
    }

    /// SC-A16d: LITERAL credential is an error, and the message does not repeat it.
    #[test]
    fn a_literal_credential_is_a_config_error_that_does_not_echo_it() {
        let err = EndpointTemplate::parse("https://juan:s3cr3t@host/v1", Scope::Root).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("[user]") && msg.contains("[password]"),
            "names the placeholder: {msg}"
        );
        assert!(
            msg.contains("vault set"),
            "and the command that stores it: {msg}"
        );
        assert!(
            !msg.contains("s3cr3t"),
            "a security error that repeats the secret is useless: {msg}"
        );
        assert!(!msg.contains("juan"), "{msg}");
    }

    /// SC-A16e: the placeholder resolves from the vault, and the template displays nothing.
    #[test]
    fn placeholders_resolve_from_the_vault_in_memory() {
        let mut vault =
            StubVault::with(&[("BASE_URL_USER", "juan"), ("BASE_URL_PASSWORD", "s3cr3t")]);
        let tpl =
            EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root).unwrap();

        let resolved = tpl.resolve(&mut vault, Scope::Root).unwrap();
        assert_eq!(resolved.as_str(), "https://juan:s3cr3t@host/v1");
        // The template is what is displayed: it is already safe, no need to redact it.
        assert_eq!(tpl.as_str(), "https://[user]:[password]@host/v1");
    }

    /// The `Debug` of the resolved URL redacts: a `derive` here is the easiest way to leak.
    #[test]
    fn the_resolved_endpoints_debug_never_shows_the_credential() {
        let mut vault =
            StubVault::with(&[("BASE_URL_USER", "juan"), ("BASE_URL_PASSWORD", "s3cr3t")]);
        let resolved = EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root)
            .unwrap()
            .resolve(&mut vault, Scope::Root)
            .unwrap();
        let shown = format!("{resolved:?}");
        assert!(!shown.contains("s3cr3t"), "leaked via Debug: {shown}");
        assert!(
            shown.contains("host"),
            "and the host stays visible: {shown}"
        );
    }

    /// SC-A16f: placeholder without entry fails CLOSED, does not substitute empty.
    #[test]
    fn a_missing_vault_entry_fails_closed_naming_the_entry() {
        let mut vault = StubVault::with(&[("BASE_URL_USER", "juan")]); // password missing
        let err = EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root)
            .unwrap()
            .resolve(&mut vault, Scope::Root)
            .unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("BASE_URL_PASSWORD"), "names the entry: {msg}");
        assert!(msg.contains("vault set"), "and how to create it: {msg}");
    }

    /// Loop 2 fix (Caspar, S1): a reserved character in a vault-stored credential must be
    /// percent-encoded before substitution, not inserted raw.
    ///
    /// Without encoding, a password containing `/` truncates the AUTHORITY: `locate_userinfo`
    /// stops at the first `/`, `?` or `#` it sees when computing where the authority ends, so
    /// `"p/ss"` as a password yields `https://juan:p/ss@host/v1` — the authority becomes
    /// `juan:p` (ends at the `/`), which has no `@` in it, so `locate_userinfo` returns
    /// `Absent` and `redact_url` returns the string UNCHANGED. The rest of the password
    /// (`ss@host/v1`) reads as path text in cleartext, and the string no longer denotes the
    /// intended host either — this is simultaneously a redaction bypass and a mis-route, the
    /// exact defect the placeholder design exists to make structurally impossible.
    ///
    /// Percent-encoding both `user` and `password` for the `userinfo` position closes this: no
    /// substituted byte can ever be interpreted as `/`, `?`, `#` or `@` by a URL parser, so the
    /// authority window `locate_userinfo` computes always matches the one the template declared.
    #[test]
    fn a_reserved_character_in_the_password_is_percent_encoded_not_left_raw() {
        let mut vault = StubVault::with(&[
            ("BASE_URL_USER", "juan"),
            ("BASE_URL_PASSWORD", "p/ss@word?#evil"),
        ]);
        let resolved = EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root)
            .unwrap()
            .resolve(&mut vault, Scope::Root)
            .unwrap();

        // The raw reserved characters must not survive substitution unescaped: if they did,
        // they would still be sitting where a URL parser looks for authority terminators.
        assert!(
            !resolved.as_str().contains("p/ss@word?#evil"),
            "the raw password leaked into the resolved URL unescaped: {}",
            resolved.as_str()
        );

        // The redaction that this whole design is built around must still find EXACTLY the
        // credential and mask it, leaving the host visible — proving the authority is still
        // parsed the way the template intended, not truncated by an embedded '/'.
        let redacted = redact_url(resolved.as_str());
        assert_eq!(
            redacted, "https://***@host/v1",
            "encoding must preserve both redaction AND correct host routing: got {redacted}"
        );
    }

    /// Each `base_url` resolves ITS OWN credentials: two endpoints may have different users.
    #[test]
    fn each_scope_reads_its_own_vault_entries() {
        let mut vault = StubVault::with(&[
            ("BASE_URL_USER", "root-u"),
            ("BASE_URL_PASSWORD", "root-p"),
            ("MAGI_BASE_URL_USER", "trio-u"),
            ("MAGI_BASE_URL_PASSWORD", "trio-p"),
            ("EMBEDDING_BASE_URL_USER", "emb-u"),
            ("EMBEDDING_BASE_URL_PASSWORD", "emb-p"),
        ]);
        let tpl =
            EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root).unwrap();
        assert!(tpl
            .resolve(&mut vault, Scope::Root)
            .unwrap()
            .as_str()
            .contains("root-u"));
        assert!(tpl
            .resolve(&mut vault, Scope::Magi)
            .unwrap()
            .as_str()
            .contains("trio-u"));
        assert!(tpl
            .resolve(&mut vault, Scope::Embedding)
            .unwrap()
            .as_str()
            .contains("emb-u"));
    }

    /// Only those two placeholders, and only in the authority. It is not a template engine.
    #[test]
    fn only_the_two_known_placeholders_in_the_authority_are_recognized() {
        assert!(EndpointTemplate::parse("https://[banana]@host/v1", Scope::Root).is_err());
        // Outside the authority it is literal path text, not a placeholder.
        let tpl = EndpointTemplate::parse("https://host/v1/[user]", Scope::Root).unwrap();
        assert_eq!(
            tpl.resolve(&mut StubVault::empty(), Scope::Root)
                .unwrap()
                .as_str(),
            "https://host/v1/[user]"
        );
    }

    /// A URL without credentials passes too: the common case pays nothing.
    #[test]
    fn a_plain_url_without_userinfo_resolves_to_itself() {
        let tpl = EndpointTemplate::parse("http://localhost:11434/v1", Scope::Root).unwrap();
        assert_eq!(
            tpl.resolve(&mut StubVault::empty(), Scope::Root)
                .unwrap()
                .as_str(),
            "http://localhost:11434/v1"
        );
    }

    /// Without `://` there is no authority to traverse: `locate_userinfo` returns `Unparseable`
    /// and `parse` propagates it as [`EndpointError::Unparseable`], instead of assuming "no
    /// credential".
    ///
    /// Covers the arm `UserinfoLocation::Unparseable => Err(EndpointError::Unparseable)` of
    /// `EndpointTemplate::parse`, which had no test case: verified by reading `locate_userinfo`
    /// (`src/redact.rs`) — the first `let Some(scheme_end) = raw.find("://") else { return
    /// Unparseable }` is reachable with any text not containing `"://"`.
    #[test]
    fn a_url_without_a_scheme_separator_is_rejected_as_unparseable() {
        let err = EndpointTemplate::parse("localhost:11434/v1", Scope::Root).unwrap_err();
        assert!(
            matches!(err, EndpointError::Unparseable),
            "expected Unparseable, got {err:?}"
        );
    }

    /// The template's `Display` emits exactly what it stores — it is the same text as
    /// `as_str()`, so a consumer doing `format!("{tpl}")` sees the complete template.
    #[test]
    fn display_renders_the_same_text_as_as_str() {
        let tpl =
            EndpointTemplate::parse("https://[user]:[password]@host/v1", Scope::Root).unwrap();
        assert_eq!(format!("{tpl}"), tpl.as_str());
        assert_eq!(tpl.to_string(), "https://[user]:[password]@host/v1");
    }

    /// Loop 1 fix round CE, F22: a literal credential in `[magi].base_url` or
    /// `[embedding].base_url` must name THAT scope's vault entries, not the root's. Before this
    /// fix `parse` always named `BASE_URL_USER`/`BASE_URL_PASSWORD`, so an operator fixing
    /// `[magi]` was sent to create the wrong entry and only discovered the real one when
    /// `resolve` failed a second time against `MAGI_BASE_URL_USER`.
    #[test]
    fn a_literal_credential_names_the_entries_of_its_own_scope() {
        let cases = [
            (Scope::Root, "BASE_URL_USER", "BASE_URL_PASSWORD"),
            (Scope::Magi, "MAGI_BASE_URL_USER", "MAGI_BASE_URL_PASSWORD"),
            (
                Scope::Embedding,
                "EMBEDDING_BASE_URL_USER",
                "EMBEDDING_BASE_URL_PASSWORD",
            ),
        ];
        for (scope, expected_user, expected_password) in cases {
            let err = EndpointTemplate::parse("https://juan:s3cr3t@host/v1", scope).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(expected_user),
                "{scope:?}: expected {expected_user} in {msg}"
            );
            assert!(
                msg.contains(expected_password),
                "{scope:?}: expected {expected_password} in {msg}"
            );
        }
    }
}
