// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-09

//! Built-in default backend profile (Ollama-first). MANUAL MAINTENANCE: these
//! `:cloud` tags reflect the Ollama catalog at release time and rot as it changes
//! (e.g. `qwen3-max` never existed; `qwen3.6` appeared). Refresh per release; users
//! override via `magi.toml`/env. All default literals live HERE, in one place.

// These default literals are consumed by `config.rs` (provider/base_url/model
// resolution) and `agent::magi_wiring` (the MAGI trio) in later wiring steps; the
// module is the single source of truth, so allow them to exist ahead of their call
// sites without tripping the dead-code lint on the non-test build.
#![allow(dead_code)]

/// Default provider when no `magi.toml`/env is present (RF-1).
pub const DEFAULT_PROVIDER: &str = "openai";
/// Default OpenAI-compatible base URL — local Ollama (RF-2).
pub const DEFAULT_OPENAI_BASE_URL: &str = "http://localhost:11434/v1";
/// Default principal model on the openai path (RF-3).
pub const DEFAULT_OPENAI_MODEL: &str = "kimi-k2.6:cloud";
/// Default MAGI trio (openai path only, RF-4). Lineages: Alibaba / OpenAI / DeepSeek.
pub const DEFAULT_MAGI_MELCHIOR: &str = "qwen3.5:397b-cloud";
pub const DEFAULT_MAGI_BALTHASAR: &str = "gpt-oss:120b-cloud";
pub const DEFAULT_MAGI_CASPAR: &str = "deepseek-v4-pro:cloud";
/// Default Anthropic model on the opt-in path (RF-5). Was `main.rs::DEFAULT_MODEL`.
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";

/// Startup notice shown when no `magi.toml` is present (RF-9). Built by
/// interpolating the default constants (RF-8 DRY) so it tracks any constant edit.
pub fn no_config_notice() -> String {
    format!(
        "No magi.toml — using Ollama defaults ({base}, {model}, \
         Melchior: {mel}, Balthasar: {bal}, Caspar: {cas}). Copy \
         magi.toml.example to customize, or set provider=\"anthropic\" \
         for Anthropic.",
        base = DEFAULT_OPENAI_BASE_URL,
        model = DEFAULT_OPENAI_MODEL,
        mel = DEFAULT_MAGI_MELCHIOR,
        bal = DEFAULT_MAGI_BALTHASAR,
        cas = DEFAULT_MAGI_CASPAR,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constants_are_the_ollama_first_profile() {
        assert_eq!(DEFAULT_PROVIDER, "openai");
        assert_eq!(DEFAULT_OPENAI_BASE_URL, "http://localhost:11434/v1");
        assert_eq!(DEFAULT_OPENAI_MODEL, "kimi-k2.6:cloud");
        assert_eq!(DEFAULT_MAGI_MELCHIOR, "qwen3.5:397b-cloud");
        assert_eq!(DEFAULT_MAGI_BALTHASAR, "gpt-oss:120b-cloud");
        assert_eq!(DEFAULT_MAGI_CASPAR, "deepseek-v4-pro:cloud");
        assert_eq!(DEFAULT_ANTHROPIC_MODEL, "claude-sonnet-4-6");
    }

    #[test]
    fn test_no_config_notice_interpolates_all_defaults() {
        // S-9: the notice is built from the constants (DRY), not hardcoded strings.
        let n = no_config_notice();
        assert!(n.contains(DEFAULT_OPENAI_BASE_URL));
        assert!(n.contains(DEFAULT_OPENAI_MODEL));
        assert!(n.contains(DEFAULT_MAGI_MELCHIOR));
        assert!(n.contains(DEFAULT_MAGI_BALTHASAR));
        assert!(n.contains(DEFAULT_MAGI_CASPAR));
    }
}
