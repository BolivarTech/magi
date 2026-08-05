// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! Redacting credentials in URLs, **by position and never by content** (REQ-A16).
//!
//! # Why it lives in the LIB and not under `system/`
//!
//! It processes untrusted input, so it is the candidate from §0.3 for `cargo fuzz`. `system/`
//! belongs to the binary and is not reachable from either a fuzz target or `tests/`.

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

use std::error::Error;
use std::fmt;

/// What replaces a credential, and an entire URL when it could not be traversed.
const FULLY_REDACTED: &str = "***";

/// Separator that opens the authority of a URL.
const SCHEME_SEPARATOR: &str = "://";

/// Redacts the `userinfo` of a URL **by POSITION, never by content** (REQ-A16).
///
/// Exact rule, in three steps:
/// 1. The **authority** begins after `://` and ends at the first `/`, `?`, or `#`.
/// 2. Inside that window —and only there— the `userinfo` is everything before the **last** `@`.
/// The last, not the first: `user:p@ss@host` is a password containing `@`, legal in RFC 3986.
/// 3. Without an `@` inside the authority there is no `userinfo`, and nothing is touched.
///
/// **Why positional and not by content:** «decode and then redact» loses to
/// double percent-encoding — `%2570` decodes once to `%70`, which is still encoded, and
/// decoding in a loop invites a decoding bomb. The position of `userinfo` **does not depend on
/// the encoding of its content**, so the positional rule works for any encoding, present or
/// future.
///
/// IPv6 hosts in brackets enter without special casing: the last `@` of the authority falls
/// before the `[`, and the rule never looks for `:`, so the colons in the address are not
/// confused with the `usuario:clave` separator.
///
/// A URL that does not parse is redacted **whole**: that is exactly where a secret might be in
/// an unexpected place, so the safe failure direction is to hide too much.
///
/// # Examples
///
/// ```
/// use magi_rs::redact::redact_url;
///
/// assert_eq!(redact_url("https://user:pass@host/v1"), "https://***@host/v1");
/// assert_eq!(redact_url("https://host/ruta@cosa"), "https://host/ruta@cosa");
/// ```
#[must_use]
pub fn redact_url(raw: &str) -> String {
    match locate_userinfo(raw) {
        UserinfoLocation::Unparseable => FULLY_REDACTED.to_string(),
        UserinfoLocation::Absent => raw.to_string(),
        UserinfoLocation::Found { start, end } => {
            let (Some(prefix), Some(tail)) = (raw.get(..start), raw.get(end..)) else {
                return FULLY_REDACTED.to_string();
            };
            let mut out = String::with_capacity(raw.len());
            out.push_str(prefix);
            out.push_str(FULLY_REDACTED);
            out.push_str(tail);
            out
        }
    }
}

/// Where the `userinfo` of a URL falls, if there is one.
///
/// It is exposed because **two** modules need the same authority rule: this one redacts what it
/// finds and `magi::endpoint` rejects what it finds if they are not placeholders. Writing the
/// traversal twice is how they get out of sync (B3), and here getting out of sync means one of
/// the two stops seeing a credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserinfoLocation {
    /// The URL could not be traversed. Whoever receives it must **fail toward hiding**: that is
    /// where a secret might be in an unexpected place.
    Unparseable,
    /// The authority has no `@`, so there is no `userinfo` and nothing to hide.
    Absent,
    /// Range `[start, end)` of the `userinfo`, **without** including the closing `@`.
    Found {
        /// First byte of the `userinfo`, right after `://`.
        start: usize,
        /// Byte of the `@` that closes the `userinfo`.
        end: usize,
    },
}

/// Locates the `userinfo` by applying the RFC 3986 authority rule.
///
/// The complete rule is documented in [`redact_url`], which is its first consumer.
///
/// It does not use `&raw[a..b]`: this module's attribute block includes
/// `deny(clippy::string_slice, clippy::indexing_slicing)`, and `str::get` returns `Option`
/// instead of panicking at a character boundary — which is the guarantee wanted in a function
/// that traverses untrusted input.
#[must_use]
pub fn locate_userinfo(raw: &str) -> UserinfoLocation {
    let Some(scheme_end) = raw.find(SCHEME_SEPARATOR) else {
        return UserinfoLocation::Unparseable;
    };
    let authority_start = scheme_end + SCHEME_SEPARATOR.len();
    let Some(rest) = raw.get(authority_start..) else {
        return UserinfoLocation::Unparseable;
    };
    // The authority ends at the first `/`, `?`, or `#`; with none, it is all the rest.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let Some(authority) = rest.get(..authority_end) else {
        return UserinfoLocation::Unparseable;
    };
    // LAST `@` of the authority: `user:p@ss@host` is a password containing `@`, legal in RFC
    // 3986. Without `@` there is no userinfo — an `@` in the path is not a credential.
    let Some(at) = authority.rfind('@') else {
        return UserinfoLocation::Absent;
    };
    UserinfoLocation::Found {
        start: authority_start,
        end: authority_start + at,
    }
}

/// Error text already redacted. **Its only constructors are [`redact_foreign_error`] and
/// [`redact_foreign_text`].**
///
/// It is what prevents an unredacted `String` from reaching a domain error: without the newtype
/// the unredacted path is one `.into()` away, and the defense goes back to depending on someone
/// remembering at every site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeErrorText(String);

impl SafeErrorText {
    /// The text, already safe to display.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SafeErrorText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Characters that may be part of a URL scheme (RFC 3986: `ALPHA *( ALPHA / DIGIT / "+" / "-" /
/// "." )`).
fn is_scheme_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
}

/// Characters that terminate a URL embedded in prose.
///
/// Deliberately generous on the ending side: including too much truncates the URL earlier and
/// at most leaves a stretch of host visible, while including too little could leave the
/// credential outside the redacted window.
fn ends_embedded_url(c: char) -> bool {
    c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`' | ',' | ';' | ')')
}

/// Redacts URLs **embedded in any third-party prose**, keeping the rest — the engine behind
/// [`redact_foreign_error`], exposed directly for callers that do not have a `&dyn Error` to
/// wrap.
///
/// # Why it exists SEPARATE from [`redact_foreign_error`] (fix round 3, CRITICAL finding)
///
/// `redact_foreign_error` takes `&dyn Error` because its first consumer (`explain_magi_error`)
/// had one at hand. But the whole logic operates on the `String` produced by `err.to_string()`
/// — nothing that follows uses the `Error` itself. A second caller with third-party text that
/// **is already `String`** (e.g. `MagiReport::failed_agents`, whose value is literally
/// `MagiError::Provider(e).to_string()` built by magi-core, verified against
/// `orchestrator.rs::dispatch_one_agent`) does not have an `Error` to wrap — wrapping it in a
/// single-use `impl Error` to satisfy a signature would be a workaround that leaves the real
/// hole intact: the day someone has a foreign `String` and not an `Error`, they will write THAT
/// `.to_string()` without going through here, exactly the defect this fix closes.
///
/// **Fix round 4 — the concrete mechanism, corrected.** The original motivator for this
/// function
/// (`failed_agents_json`, `src/tools/consult.rs`) described an ORDINARY CONNECTION FAILURE
/// leaking the resolved URL through the reqwest/hyper `Display` — verified against magi-core
/// 3.1.0 and **it is incorrect**: `provider.rs::to_provider_error` builds `Network`/`Timeout`
/// from an ALREADY REDACTED URL plus `cause_chain(e)`, which starts at `e.source()` and
/// therefore
/// **skips** the top-level error (the one that interpolates the raw URL) — pinned by the very
/// magi-core test `cause_chain_skips_the_top_level_error`. That specific path is already
/// covered upstream. The real exposure this redaction covers is `ProviderError::Http { body }`
/// (SERVER-CONTROLLED response text, unredacted) and the fact that `ProviderError` is
/// `#[non_exhaustive]` — a future magi-core variant might interpolate free text without this
/// code changing a line. That is why redaction happens at the EDGE (every foreign `String`) and
/// not by variant: it remains correct even if the originally suspected mechanism does not
/// apply, and remains correct if magi-core changes.
///
/// # Why this needs to be redacted, and why a list of sites is not enough
///
/// The `format!`s we write ourselves can be enumerated and audited. This path cannot: the text
/// is built by **another crate** with the URL we pass it, so no review of our formatters sees
/// it. Every `String` that packages a foreign error goes through here.
///
/// It is not a second implementation of [`redact_url`]: it sweeps the URLs in the message and
/// applies **the same positional rule** to each one.
#[must_use]
pub fn redact_foreign_text(raw: &str) -> SafeErrorText {
    let mut out = String::with_capacity(raw.len());
    let mut cursor = 0usize;

    while let Some(found) = raw.get(cursor..).and_then(|r| r.find(SCHEME_SEPARATOR)) {
        let sep_at = cursor + found;
        // The scheme starts where valid scheme characters end, working backwards.
        let Some(before) = raw.get(cursor..sep_at) else {
            break;
        };
        let scheme_len = before
            .chars()
            .rev()
            .take_while(|c| is_scheme_char(*c))
            .map(char::len_utf8)
            .sum::<usize>();
        let url_start = sep_at - scheme_len;

        let after_sep = sep_at + SCHEME_SEPARATOR.len();
        let Some(tail) = raw.get(after_sep..) else {
            break;
        };
        let url_end = after_sep + tail.find(ends_embedded_url).unwrap_or(tail.len());

        let (Some(lead), Some(url)) = (raw.get(cursor..url_start), raw.get(url_start..url_end))
        else {
            break;
        };
        out.push_str(lead);
        out.push_str(&redact_url(url));
        cursor = url_end;
    }

    if let Some(rest) = raw.get(cursor..) {
        out.push_str(rest);
    }
    SafeErrorText(out)
}

/// Redacts URLs **embedded in the prose of a foreign error**, keeping the rest.
///
/// Wrapper of [`redact_foreign_text`] over `err.to_string()` — see its rustdoc for why the
/// separation. It is kept for compatibility with callers that do have a `&dyn Error`
/// (`explain_magi_error`).
#[must_use]
pub fn redact_foreign_error(err: &dyn Error) -> SafeErrorText {
    redact_foreign_text(&err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A13: simple userinfo.
    #[test]
    fn userinfo_is_redacted_and_the_host_survives() {
        assert_eq!(
            redact_url("https://user:pass@host/v1"),
            "https://***@host/v1"
        );
    }

    /// SC-A13c: double percent-encoding does NOT evade, because the rule is POSITIONAL.
    #[test]
    fn double_percent_encoding_does_not_evade_redaction() {
        let doubly = "https://%2575%2573%2565%2572:%2570@host/v1";
        let out = redact_url(doubly);
        assert!(!out.contains("%2570"), "quedó credencial: {out}");
        assert!(out.contains("host"));
    }

    /// SC-A13d: an `@` in the PATH does not trigger redaction.
    #[test]
    fn an_at_sign_in_the_path_is_not_userinfo() {
        assert_eq!(
            redact_url("https://host/ruta@cosa"),
            "https://host/ruta@cosa"
        );
    }

    /// SC-A13d: password containing `@` — the LAST `@` of the authority wins.
    #[test]
    fn the_last_at_within_the_authority_wins() {
        assert_eq!(
            redact_url("https://user:p@ss@host/v1"),
            "https://***@host/v1"
        );
    }

    /// IPv6 in brackets: enters the rule without special casing.
    #[test]
    fn bracketed_ipv6_hosts_are_handled_without_a_special_case() {
        assert_eq!(redact_url("http://[::1]:11434/v1"), "http://[::1]:11434/v1");
        assert_eq!(
            redact_url("http://u:p@[::1]:11434/v1"),
            "http://***@[::1]:11434/v1"
        );
    }

    /// Safe failure direction: what does not parse is redacted WHOLE.
    #[test]
    fn an_unparseable_url_is_redacted_whole() {
        assert_eq!(redact_url("no es una url"), "***");
    }

    /// A FOREIGN error brings the URL embedded in its prose, and there it is also redacted.
    ///
    /// It is the path that a list of our own `format!`s cannot see: the text is built by
    /// another crate with the URL we pass it.
    #[test]
    fn a_foreign_errors_embedded_url_is_redacted_while_its_prose_survives() {
        let err = std::io::Error::other("connect to https://user:hunter2@host:8443/v1 failed");
        let safe = redact_foreign_error(&err);
        assert!(
            !safe.as_str().contains("hunter2"),
            "filtró: {}",
            safe.as_str()
        );
        assert!(
            safe.as_str().contains("host:8443"),
            "el host sigue siendo accionable"
        );
        assert!(safe.as_str().contains("failed"), "y la prosa se conserva");
    }

    /// Without a URL inside, the text passes through intact: redacting too much would make it
    /// useless.
    #[test]
    fn a_foreign_error_without_a_url_is_left_alone() {
        let err = std::io::Error::other("connection reset by peer");
        assert_eq!(
            redact_foreign_error(&err).as_str(),
            "connection reset by peer"
        );
    }

    /// Several URLs in the same message: ALL are redacted, not just the first.
    #[test]
    fn every_embedded_url_is_redacted_not_just_the_first() {
        let err =
            std::io::Error::other("tried https://a:b@one/v1 then https://c:d@two/v1 and gave up");
        let safe = redact_foreign_error(&err);
        assert!(!safe.as_str().contains("a:b"), "primera: {}", safe.as_str());
        assert!(!safe.as_str().contains("c:d"), "segunda: {}", safe.as_str());
        assert!(safe.as_str().contains("one") && safe.as_str().contains("two"));
    }

    /// Fix round 3: [`redact_foreign_text`] is the `&str` engine that [`redact_foreign_error`]
    /// wraps — it is tested DIRECTLY, without going through a `&dyn Error`, for the caller that
    /// already has a foreign `String` (e.g. `MagiReport::failed_agents`, which IS
    /// `MagiError::Provider(e).to_string()`).
    #[test]
    fn redact_foreign_text_redacts_a_string_with_no_error_to_wrap() {
        let raw = "network error: connect to https://user:hunter2@host:8443/v1 failed";
        let safe = redact_foreign_text(raw);
        assert!(
            !safe.as_str().contains("hunter2"),
            "filtró: {}",
            safe.as_str()
        );
        assert!(safe.as_str().contains("host:8443"), "sigue accionable");
    }

    /// Edge case (B13): without a URL inside, it passes intact — same contract as
    /// [`redact_foreign_error`] on the same text.
    #[test]
    fn redact_foreign_text_leaves_url_free_text_alone() {
        assert_eq!(
            redact_foreign_text("connection reset by peer").as_str(),
            "connection reset by peer"
        );
    }
}
