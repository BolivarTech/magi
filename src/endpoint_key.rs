// Author: Julian Bolivar
// Version: 0.23.0
// Date: 2026-10-12

//! Same-origin comparison and API key selection for every section that can point at its own endpoint.
#![deny(missing_docs)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use reqwest::Url;

/// Reduces a URL to its origin: scheme, host (no trailing dot) and port.
///
/// Returns `None` when the URL does not parse or has no host.
fn origin_of(raw: &str) -> Option<(String, String, u16)> {
    let url = Url::parse(raw).ok()?;
    let host = url.host_str()?.trim_end_matches('.');
    if host.is_empty() {
        return None;
    }
    Some((
        url.scheme().to_string(),
        host.to_string(),
        url.port_or_known_default()?,
    ))
}

/// Tells whether two URLs share scheme, host and effective port.
///
/// Both are parsed with the WHATWG parser (host lower-cased, IDN already in
/// punycode); path, query and userinfo do not count. A URL that does not
/// parse, or has no host, never matches anything. Textual comparison is
/// never used: it is what `http://localhost:11434@evil.com` defeats.
///
/// Complexity: O(n) in the combined length of the inputs.
///
/// # Arguments
/// * `a`, `b` - the URLs to compare.
///
/// # Returns
/// `true` only when both parse and have the same origin.
pub fn same_origin(a: &str, b: &str) -> bool {
    match (origin_of(a), origin_of(b)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// Keeps a key only when it is not blank.
fn non_blank(key: Option<&str>) -> Option<&str> {
    key.filter(|k| !k.trim().is_empty())
}

/// Picks the API key to send to a section's endpoint.
///
/// A non-blank `own_key` (the key dedicated to the section) wins. Otherwise
/// `inherited_key` (the key of the endpoint the section falls back to, such
/// as the principal provider) is used only when `home_url` and `target_url`
/// share an origin, so a key never reaches a host that is not its own. Blank
/// keys count as absent.
///
/// The key is resolved once, when the client is built: a vault change made
/// by another process is not seen until the next start.
///
/// Complexity: O(n) in the combined length of the two URLs.
///
/// # Arguments
/// * `own_key` - the key dedicated to the section, if any.
/// * `inherited_key` - the fallback key, if any.
/// * `home_url` - endpoint the inherited key belongs to.
/// * `target_url` - the endpoint the selected key will be sent to.
///
/// # Returns
/// The key to send, or `None` for no authentication header.
pub fn select_api_key(
    own_key: Option<&str>,
    inherited_key: Option<&str>,
    home_url: &str,
    target_url: &str,
) -> Option<String> {
    if let Some(key) = non_blank(own_key) {
        return Some(key.to_string());
    }
    non_blank(inherited_key)
        .filter(|_| same_origin(home_url, target_url))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adversarial pairs that must NOT inherit a key (REQ-D04, MAGI loops 1 and 6).
    #[test]
    fn different_origins_never_match() {
        for (a, b) in [
            ("http://localhost.evil.com", "http://localhost"),
            ("http://localhost:11434@evil.com", "http://localhost:11434"),
            ("http://example.com", "https://example.com"),
            ("http://localhost:11434", "http://localhost:11435"),
            ("http://evil-localhost", "http://localhost"),
            ("http://localhost", "http://localhostx"),
            ("http://[::ffff:127.0.0.1]:11434", "http://127.0.0.1:11434"),
            ("not a url", "not a url"),
            ("http://localhost:11434", "not a url"),
            // CP2 MS1 r1 (Caspar): spellings that have fooled URL comparators.
            ("http://evil.com\\@localhost", "http://localhost"), // WHATWG reads `\` as `/`: host is evil.com
            ("http://localhost%40evil.com", "http://localhost"), // a percent-encoded `@` is not userinfo
            ("http://", "http://localhost"),                     // no host at all
            ("http://:11434", "http://localhost:11434"),         // empty host
        ] {
            assert!(!same_origin(a, b), "{a} vs {b}");
            assert!(!same_origin(b, a), "{b} vs {a}");
        }
    }

    #[test]
    fn equivalent_spellings_of_the_same_origin_match() {
        for (a, b) in [
            ("http://LOCALHOST:11434", "http://localhost:11434"),
            ("http://localhost.:11434", "http://localhost:11434"),
            ("https://host:443/v1", "https://host/x"),
            ("http://host:80", "http://host"),
            ("http://localhost:11434", "http://localhost:11434/v1"),
            ("http://bücher.example", "http://xn--bcher-kva.example"),
            ("https://[user]:[password]@host/v1", "https://host"),
            // CP2 MS1 r1 (Caspar): an `@` in the PATH is not userinfo; the host is still `good`.
            ("http://good/@evil", "http://good"),
        ] {
            assert!(same_origin(a, b), "{a} vs {b}");
        }
    }

    #[test]
    fn the_decision_key_wins_even_on_the_same_origin() {
        let k = select_api_key(
            Some("dec-key"),
            Some("prin-key"),
            "http://h:1/v1",
            "http://h:1",
        );
        assert_eq!(k.as_deref(), Some("dec-key"));
    }

    #[test]
    fn the_principal_key_is_inherited_only_on_the_same_origin() {
        assert_eq!(
            select_api_key(
                None,
                Some("prin-key"),
                "http://localhost:11434/v1",
                "http://localhost:11434"
            )
            .as_deref(),
            Some("prin-key")
        );
        assert_eq!(
            select_api_key(None, Some("prin-key"), "http://a:1", "http://a:2"),
            None
        );
        assert_eq!(
            select_api_key(
                None,
                Some("prin-key"),
                "http://a:1",
                "http://a:1@evil.com:1"
            ),
            None
        );
    }

    #[test]
    fn blank_keys_are_absent() {
        assert_eq!(
            select_api_key(Some("  "), Some("prin-key"), "http://a:1", "http://a:1").as_deref(),
            Some("prin-key")
        );
        assert_eq!(
            select_api_key(None, Some(" "), "http://a:1", "http://a:1"),
            None
        );
        assert_eq!(select_api_key(None, None, "http://a:1", "http://a:1"), None);
    }
}
