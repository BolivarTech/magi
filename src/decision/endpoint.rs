//! System One URL derivation, same-origin comparison and API key selection
//! (REQ-D03, REQ-D04).

/// Builds the System One URL from a configured `base_url`.
///
/// # Arguments
/// * `base_url` - the configured endpoint, with or without `/v1`.
///
/// # Returns
/// The full protocol URL.
pub fn systemone_url(base_url: &str) -> String {
    let _ = base_url;
    todo!()
}

/// Tells whether two URLs share scheme, host and effective port.
///
/// # Arguments
/// * `a`, `b` - the URLs to compare.
///
/// # Returns
/// `true` only when both parse and have the same origin.
pub fn same_origin(a: &str, b: &str) -> bool {
    let _ = (a, b);
    todo!()
}

/// Picks the API key to send to the decision endpoint.
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
    let _ = (
        decision_key,
        principal_key,
        principal_base_url,
        decision_base_url,
    );
    todo!()
}

/// Diagnostic for a 404 on the System One route.
///
/// # Arguments
/// * `tried_url_redacted` - the URL that was tried, already redacted.
///
/// # Returns
/// A message naming both probable causes.
pub fn not_found_hint(tried_url_redacted: &str) -> String {
    let _ = tried_url_redacted;
    todo!()
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
        assert!(
            hint.contains("https://openrouter.ai/v1/systemone"),
            "{hint}"
        );
        assert!(hint.contains("base_url"), "{hint}");
        assert!(hint.contains("https://openrouter.ai/api/v1"), "{hint}");
        assert!(hint.contains("Ollama 0.35"), "{hint}");
    }
}
