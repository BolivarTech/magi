// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Resolution of parameters for a headless run: per-field precedence and the operator cost
//! clamp (REQ-H12/H12b).
//!
//! [`resolve`] combines three sources with precedence **CLI flag > envelope > defaults
//! (toml/built-in)** for preferences (`model`/`provider`/ `consult`), and applies two
//! **safety** limits:
//! - `max_tool_calls` is a **cost ceiling**: the requested value is clamped to `operator_ceiling` (the envelope may request less, never more). The ceiling is computed by the *caller* (bin), not this function.
//! - `system` is a **prompt-injection limit**: the envelope's `system` is ignored unless the operator enables it with `allow_system_override`.
//!
//! The function is **pure and has no binary dependencies** (it takes lib-local structs, not
//! `MagiConfig`/`Args`), so it is tested in isolation (R-H05).
//!
//! `pub` visibility: [`Resolved`] embeds [`SystemPolicy`]/[`AppliedCaps`] from `types.rs`,
//! which are `pub` for the same reason — the MS2 runner lives in the binary crate and can only
//! reach `pub` APIs from the lib.

use super::input::Envelope;
use super::types::{AppliedCaps, SystemPolicy};

/// Defaults derived from `magi.toml` (plus built-in) filled in by the bin.
///
/// The already-resolved fields (`model`/`provider`/`system`) are `String` because the bin
/// collapses the toml→built-in fallback before constructing this struct; the optionals remain
/// `None` when neither toml nor built-in sets a value.
#[derive(Debug, Clone)]
pub struct ConfigDefaults {
    /// Default LLM model (toml or built-in already collapsed).
    pub model: String,
    /// Default LLM provider (toml or built-in already collapsed).
    pub provider: String,
    /// Tool-call ceiling from toml, if set.
    pub max_tool_calls: Option<u32>,
    /// `consult` preference from toml, if set.
    pub consult: Option<bool>,
    /// Operator system-prompt (governs unless override enabled).
    pub system: String,
}

/// Overrides coming from operator CLI flags (win over the envelope).
///
/// All `None` means "the operator did not pass that flag": resolution falls to the envelope and
/// then to defaults.
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    /// Model forced by flag, if passed.
    pub model: Option<String>,
    /// Provider forced by flag, if passed.
    pub provider: Option<String>,
    /// Tool-call ceiling forced by flag, if passed.
    pub max_tool_calls: Option<u32>,
    /// `consult` preference forced by flag, if passed.
    pub consult: Option<bool>,
}

/// Effective run parameters after applying precedence and clamps.
///
/// `applied_caps` makes the result of the safety limits **visible** (cost clamp, system
/// override) so the caller does not silently fail.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// Effective LLM model.
    pub model: String,
    /// Effective LLM provider.
    pub provider: String,
    /// Effective system-prompt together with its origin (operator vs. caller).
    pub system: SystemPolicy,
    /// Effective tool-call ceiling (already clamped to the operator's ceiling).
    pub max_tool_calls: u32,
    /// Effective `consult` preference; `None` lets the agent decide.
    pub consult: Option<bool>,
    /// Effective limits applied (cost clamp, system override).
    pub applied_caps: AppliedCaps,
}

/// Resolves the effective run parameters by precedence and clamp.
///
/// Per-field precedence for preferences: `overrides` (CLI flag) > `env` (envelope) > `defaults`
/// (toml/built-in). `max_tool_calls` follows the same chain for the *requested value*, then is
/// clamped to `operator_ceiling` (`min`), setting `max_tool_calls_clamped` if the request
/// exceeded it; if nobody requests a value, the effective is the ceiling. The envelope's
/// `system` is only honored (`SystemPolicy::CallerOverride`) if `allow_system_override` is
/// `true`; otherwise the operator's governs (`SystemPolicy::Operator`).
///
/// The caller computes `operator_ceiling` (not recomputed here); `timeout_secs` is left `None`
/// here because this function has no wall-clock ceiling to report — `run_query`/`run_consult`
/// (`headless_runner.rs`) know the effective one (`RunWiring::timeout` / their own `timeout`
/// parameter) and stamp it into `AppliedCaps` themselves before returning the `RunOutcome`.
///
/// # Examples
///
/// An envelope requesting `max_tool_calls: Some(999)` with `operator_ceiling = 15` produces a
/// [`Resolved`] with `max_tool_calls == 15` and `applied_caps.max_tool_calls_clamped == true`
/// (the request is trimmed to the ceiling). See the module tests for runnable cases; this
/// example is illustrative (`ignore`), not a run doctest.
pub fn resolve(
    env: Envelope,
    defaults: &ConfigDefaults,
    overrides: &CliOverrides,
    operator_ceiling: u32,
    allow_system_override: bool,
) -> Resolved {
    // Preferences: overrides (flag) > env (envelope) > defaults (toml/built-in).
    let model = overrides
        .model
        .clone()
        .or(env.model)
        .unwrap_or_else(|| defaults.model.clone());
    let provider = overrides
        .provider
        .clone()
        .or(env.provider)
        .unwrap_or_else(|| defaults.provider.clone());
    let consult = overrides.consult.or(env.consult).or(defaults.consult);

    // Cost (SAFETY): the request follows precedence; the effective is clamped to the operator's
    // ceiling. No request ⇒ the effective is the ceiling, no clamp.
    let requested_max = overrides
        .max_tool_calls
        .or(env.max_tool_calls)
        .or(defaults.max_tool_calls);
    let (max_tool_calls, max_tool_calls_clamped) = match requested_max {
        Some(requested) => (
            requested.min(operator_ceiling),
            requested > operator_ceiling,
        ),
        None => (operator_ceiling, false),
    };

    // System (SAFETY): the envelope's is only honored with the operator's flag.
    let (system, system_override_applied) = match (allow_system_override, env.system) {
        (true, Some(caller)) => (SystemPolicy::CallerOverride(caller), true),
        _ => (SystemPolicy::Operator(defaults.system.clone()), false),
    };

    Resolved {
        model,
        provider,
        system,
        max_tool_calls,
        consult,
        applied_caps: AppliedCaps {
            max_tool_calls,
            max_tool_calls_clamped,
            timeout_secs: None,
            system_override_applied,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope with fixed `prompt` and the rest of the fields empty, for tests.
    fn base_env() -> Envelope {
        Envelope {
            prompt: "p".to_string(),
            system: None,
            model: None,
            provider: None,
            max_tool_calls: None,
            consult: None,
            mode: None,
            untrusted_content: None,
        }
    }

    /// Defaults with values distinguishable from the operator's, for tests.
    fn base_defaults() -> ConfigDefaults {
        ConfigDefaults {
            model: "def-model".to_string(),
            provider: "def-provider".to_string(),
            max_tool_calls: None,
            consult: None,
            system: "operator-system".to_string(),
        }
    }

    /// An envelope requesting MORE than the operator's ceiling is clamped to the ceiling and
    /// marks the clamp (REQ-H12b: the caller cannot raise the budget).
    #[test]
    fn test_envelope_max_tool_calls_clamped_to_operator_ceiling() {
        let mut env = base_env();
        env.max_tool_calls = Some(999);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 15);
        assert!(r.applied_caps.max_tool_calls_clamped);
        assert_eq!(r.applied_caps.max_tool_calls, 15);
    }

    /// The CLI flag wins over the envelope value for preferences.
    #[test]
    fn test_cli_flag_wins_over_envelope() {
        let mut env = base_env();
        env.model = Some("A".to_string());
        env.provider = Some("prov-env".to_string());
        env.consult = Some(false);

        let overrides = CliOverrides {
            model: Some("B".to_string()),
            provider: Some("prov-flag".to_string()),
            consult: Some(true),
            ..CliOverrides::default()
        };

        let r = resolve(env, &base_defaults(), &overrides, 15, false);

        assert_eq!(r.model, "B");
        assert_eq!(r.provider, "prov-flag");
        assert_eq!(r.consult, Some(true));
    }

    /// Without flag or envelope, preferences fall to defaults; with envelope (and no flag), the
    /// envelope wins over defaults.
    #[test]
    fn test_precedence_envelope_over_defaults_and_fallthrough() {
        // Nothing requested ⇒ defaults.
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            false,
        );
        assert_eq!(r.model, "def-model");
        assert_eq!(r.provider, "def-provider");
        assert_eq!(r.consult, None);

        // Envelope requested, no flag ⇒ envelope wins over defaults.
        let mut env = base_env();
        env.model = Some("env-model".to_string());
        let r2 = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);
        assert_eq!(r2.model, "env-model");
    }

    /// The envelope's `system` is IGNORED without the operator's flag: the operator's governs
    /// and `system_override_applied` remains `false` (REQ-H12b/H37).
    #[test]
    fn test_envelope_system_ignored_without_override_flag() {
        let mut env = base_env();
        env.system = Some("caller-sys".to_string());

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert!(matches!(r.system, SystemPolicy::Operator(ref s) if s == "operator-system"));
        assert!(!r.applied_caps.system_override_applied);
    }

    /// The envelope's `system` is HONORED with the operator's flag: it becomes `CallerOverride`
    /// and `system_override_applied` remains `true`.
    #[test]
    fn test_envelope_system_honored_with_override_flag() {
        let mut env = base_env();
        env.system = Some("caller-sys".to_string());

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, true);

        assert!(matches!(r.system, SystemPolicy::CallerOverride(ref s) if s == "caller-sys"));
        assert!(r.applied_caps.system_override_applied);
    }

    /// The enabled flag without a `system` in the envelope does not invent an override: the
    /// operator's governs and it is not marked applied.
    #[test]
    fn test_override_flag_without_envelope_system_uses_operator() {
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            true,
        );

        assert!(matches!(r.system, SystemPolicy::Operator(ref s) if s == "operator-system"));
        assert!(!r.applied_caps.system_override_applied);
    }

    /// An envelope requesting LESS than the ceiling is not clamped; the effective is the
    /// request (edge of the cost rule).
    #[test]
    fn test_envelope_max_tool_calls_below_ceiling_not_clamped() {
        let mut env = base_env();
        env.max_tool_calls = Some(5);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 5);
        assert!(!r.applied_caps.max_tool_calls_clamped);
    }

    /// Without a `max_tool_calls` request from any source, the effective falls to the
    /// operator's ceiling and no clamp is marked.
    #[test]
    fn test_max_tool_calls_falls_through_to_operator_ceiling() {
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            false,
        );

        assert_eq!(r.max_tool_calls, 15);
        assert!(!r.applied_caps.max_tool_calls_clamped);
        assert_eq!(r.applied_caps.timeout_secs, None);
    }

    /// A request EXACTLY equal to the ceiling is not clamped (strict `>`).
    #[test]
    fn test_max_tool_calls_equal_to_ceiling_not_clamped() {
        let mut env = base_env();
        env.max_tool_calls = Some(15);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 15);
        assert!(!r.applied_caps.max_tool_calls_clamped);
    }
}
