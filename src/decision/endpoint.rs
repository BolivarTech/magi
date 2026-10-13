// Author: Julian Bolivar
// Version: 0.24.0
// Date: 2026-10-13

//! System One URL derivation (REQ-D03).

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
            "{marker}{tried_url_redacted}; check that base_url includes the ",
            "provider's API prefix (OpenRouter: {OPENROUTER_BASE_URL_EXAMPLE}) and that the server ",
            "supports System One (Ollama 0.35 or later)"
        ),
        marker = NOT_FOUND_HINT_MARKER,
        tried_url_redacted = tried_url_redacted,
        OPENROUTER_BASE_URL_EXAMPLE = OPENROUTER_BASE_URL_EXAMPLE
    )
}

/// Start of every [`not_found_hint`]; lets the envelope find the hint inside an attempt's detail.
pub const NOT_FOUND_HINT_MARKER: &str = "no System One endpoint answered at ";

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
