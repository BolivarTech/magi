//! System One URL derivation, same-origin comparison and API key selection
//! (REQ-D03, REQ-D04).

use reqwest::Url;

/// Path segment that marks a `base_url` already carrying the API version.
const API_VERSION_SUFFIX: &str = "/v1";

/// Route of the System One protocol, appended to a base that ends in `/v1`.
const SYSTEMONE_ROUTE: &str = "/systemone";

/// Where Magi tells the operator an OpenRouter-style base URL should point.
const OPENROUTER_BASE_URL_EXAMPLE: &str = "https://openrouter.ai/api/v1";

/// Builds the System One URL from a configured `base_url`.
///
/// Surrounding whitespace and every trailing `/` are dropped; a base that
/// ends in `/v1` gets `/systemone`, any other gets `/v1/systemone`. The
/// provider is never detected.
///
/// Complexity: O(n) in the length of `base_url`.
///
/// # Arguments
/// * `base_url` - the configured endpoint, with or without `/v1`.
///
/// # Returns
/// The full protocol URL.
///
/// # Examples
/// ```
/// use magi_rs::decision::endpoint::systemone_url;
/// assert_eq!(systemone_url("http://localhost:11434"), "http://localhost:11434/v1/systemone");
/// ```
pub fn systemone_url(base_url: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if base.ends_with(API_VERSION_SUFFIX) {
        format!("{base}{SYSTEMONE_ROUTE}")
    } else {
        format!("{base}{API_VERSION_SUFFIX}{SYSTEMONE_ROUTE}")
    }
}

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

/// Picks the API key to send to the decision endpoint.
///
/// A non-blank `decision_key` wins. Otherwise the principal key is
/// inherited only when both endpoints share an origin, so a key never
/// reaches a host that is not its own. Blank keys count as absent.
///
/// The key is resolved once, when the client is built: a vault change made
/// by another process is not seen until the next start.
///
/// # Arguments
/// * `decision_key` - the dedicated decision key, if any.
/// * `principal_key` - the principal provider key, if any.
/// * `principal_base_url` - endpoint the principal key belongs to.
/// * `decision_base_url` - the decision endpoint.
///
/// # Returns
/// The key to send, or `None` for no authentication header.
pub fn select_api_key(
    decision_key: Option<&str>,
    principal_key: Option<&str>,
    principal_base_url: &str,
    decision_base_url: &str,
) -> Option<String> {
    if let Some(key) = non_blank(decision_key) {
        return Some(key.to_string());
    }
    non_blank(principal_key)
        .filter(|_| same_origin(principal_base_url, decision_base_url))
        .map(str::to_string)
}

/// Diagnostic for a System One route that did not answer as one (a 404, or a body that is not JSON).
///
/// # Arguments
/// * `tried_url_redacted` - the URL that was tried, already redacted.
///
/// # Returns
/// A message naming the URL and both probable causes.
pub fn not_found_hint(tried_url_redacted: &str) -> String {
    format!(
        concat!(
        "no System One endpoint answered at {tried_url_redacted}; check that base_url includes the ",
        "provider's API prefix (OpenRouter: {OPENROUTER_BASE_URL_EXAMPLE}) and that the server ",
        "supports System One (Ollama 0.35 or later)"
        ),
        tried_url_redacted = tried_url_redacted,
        OPENROUTER_BASE_URL_EXAMPLE = OPENROUTER_BASE_URL_EXAMPLE
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_url_follows_the_req_d03_table() {
        for (base, expected) in [
            (
                "http://localhost:11434",
                "http://localhost:11434/v1/systemone",
            ),
            (
                "http://localhost:11434/v1",
                "http://localhost:11434/v1/systemone",
            ),
            (
                "http://localhost:11434/v1/",
                "http://localhost:11434/v1/systemone",
            ),
            (
                "http://localhost:11434/v1//",
                "http://localhost:11434/v1/systemone",
            ),
            (
                " https://openrouter.ai/api/v1 ",
                "https://openrouter.ai/api/v1/systemone",
            ),
            (
                "https://api.typesafe.ai",
                "https://api.typesafe.ai/v1/systemone",
            ),
            (
                "https://openrouter.ai",
                "https://openrouter.ai/v1/systemone",
            ),
        ] {
            assert_eq!(systemone_url(base), expected, "{base:?}");
        }
    }

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

    #[test]
    fn the_not_found_hint_names_the_url_and_both_causes() {
        let hint = not_found_hint("https://openrouter.ai/v1/systemone");
        assert!(!hint.contains("  ") && !hint.contains('\n'), "{hint}");
        // SMOKE #1: OpenRouter answers a prefix-less path with a non-JSON 200, not a 404, so the
        // hint must not claim a status it may not have seen.
        assert!(!hint.contains("404"), "{hint}");
        assert!(
            hint.contains("https://openrouter.ai/v1/systemone"),
            "{hint}"
        );
        assert!(hint.contains("base_url"), "{hint}");
        assert!(hint.contains("https://openrouter.ai/api/v1"), "{hint}");
        assert!(hint.contains("Ollama 0.35"), "{hint}");
    }
}
