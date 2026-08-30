// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Numeric limits and default caps for headless mode — the single source of
//! truth, lib-visible so both the `headless` lib modules and the bin
//! (`main.rs`/`config.rs`) reference the same values.
//!
//! These live here (not in the bin-only `src/defaults.rs`) because the crate
//! is split: `headless` is a library module while `defaults` sits in the
//! binary crate. The bin re-exports these via `crate::defaults` so a
//! `magi.toml` `[headless]` key still maps to one constant (REQ-H00 DRY).

/// Cap on `-i`/stdin input bytes read (DoS bound, REQ-H29). 10 MiB. Read with
/// `take(cap+1)` so a hostile unbounded source never buffers fully.
pub const MAX_INPUT_BYTES: usize = 10 * 1024 * 1024;

/// Max JSON nesting depth of the envelope parser (stack-DoS bound, REQ-H29).
/// The default `serde_json` limit (128) exceeds this, so our counter fires first.
pub const MAX_JSON_DEPTH: u32 = 64;

/// Cap on each tool `result` in the rich output, with a truncation marker
/// (REQ-H14). 64 KiB.
pub const TOOL_RESULT_CAP: usize = 64 * 1024;

/// Hard wall-clock ceiling (seconds) applied by default under `--full-auto`
/// (and any tool-executing tier) when no `--timeout` is given (REQ-H36). 900 s.
pub const FULL_AUTO_TIMEOUT_SECS: u64 = 900;

/// Normal per-query tool-call cap; the operator ceiling when `magi.toml` sets
/// none (REQ-H08/H12b). 15.
pub const NORMAL_MAX_TOOL_CALLS: u32 = 15;

/// Elevated tool-call cap under `--full-auto`; a hard backstop, not infinite
/// (REQ-H08). 50.
pub const FULL_AUTO_MAX_TOOL_CALLS: u32 = 50;

/// Effective headless numeric caps for one run, after the `[headless]`
/// `magi.toml` overrides are applied over the built-in constant defaults
/// (spec §11).
///
/// Built once per run (see `main.rs::resolve_headless_limits`); each field
/// defaults to its module constant and is overridden only when the operator
/// sets the matching `[headless]` key. Non-numeric knobs (`log_level`,
/// `allow_system_override`) resolve at their own sites, not here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadlessLimits {
    /// Effective `-i`/stdin byte cap (default [`MAX_INPUT_BYTES`]).
    pub max_input_bytes: usize,
    /// Effective `--full-auto` tool-call cap (default [`FULL_AUTO_MAX_TOOL_CALLS`]).
    pub full_auto_max_tool_calls: u32,
    /// Effective per-tool-result byte cap (default [`TOOL_RESULT_CAP`]).
    pub tool_result_cap: usize,
    /// Effective default wall-clock timeout secs for tool-executing tiers
    /// (default [`FULL_AUTO_TIMEOUT_SECS`]).
    pub full_auto_timeout_secs: u64,
}

impl Default for HeadlessLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: MAX_INPUT_BYTES,
            full_auto_max_tool_calls: FULL_AUTO_MAX_TOOL_CALLS,
            tool_result_cap: TOOL_RESULT_CAP,
            full_auto_timeout_secs: FULL_AUTO_TIMEOUT_SECS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headless caps keep their documented contract values (spec §11).
    #[test]
    fn test_headless_limit_values_match_spec_contract() {
        assert_eq!(MAX_INPUT_BYTES, 10 * 1024 * 1024);
        assert_eq!(MAX_JSON_DEPTH, 64);
        assert_eq!(TOOL_RESULT_CAP, 64 * 1024);
        assert_eq!(FULL_AUTO_TIMEOUT_SECS, 900);
        assert_eq!(NORMAL_MAX_TOOL_CALLS, 15);
        assert_eq!(FULL_AUTO_MAX_TOOL_CALLS, 50);
        // Elevated cap is a genuine elevation over the normal cap (50 > 15),
        // never a silent downgrade — pinned by the two values above.
    }
}
