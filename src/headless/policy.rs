// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-18

//! Tool authorization policy per headless tier (REQ-H06/H07/H08/H09).
//!
//! [`Policy`] translates the headless tier (`default`/`--auto`/`--full-auto`) into a per-tool
//! approval decision: **fail-closed** — an unrecognized tool never auto-approves, in any tier.
//! The module is **pure** logic, with no dependency on `Agent` or the tools (`headless` is a
//! library [`crate`] module, while `src/tools/` lives only in the binary) — the runner (later
//! MS2 task) wires this decision to the agent's `approval_tx`.
//!
//! **This policy never touches the HARD barriers** (`bash::is_command_allowed`,
//! the prohibition of metacharacters, `PathGuard::validate`): these are applied inside each
//! tool and remain active regardless of the tier (REQ-H09). What this policy decides is
//! exclusively the **soft** per-tier approval.

use super::limits::{FULL_AUTO_MAX_TOOL_CALLS, NORMAL_MAX_TOOL_CALLS};

/// READ-ONLY tool names — single source of truth for the set (REQ-H06, DRY). Verified against
/// the real registry in `main.rs` (`ListTool`/ `FileReadTool`/`GrepTool`, whose `Tool::name()`
/// returns exactly these three literals — see `src/tools/{ls,read,grep}.rs`).
pub const READ_ONLY_TOOLS: &[&str] = &["ls", "view", "grep"];

/// Names of tools that mutate state or run additional processes/LLMs — approved only in
/// `Auto`/`FullAuto` (REQ-H07). Verified against the real registry in `main.rs`
/// (`FileWriteTool`/`BashTool`/`ConsultTool`/`ProjectFactTool`, whose `Tool::name()` returns
/// these four literals — see `src/tools/{write,bash,consult,knowledge}.rs`).
const READ_WRITE_TOOLS: &[&str] = &["edit", "bash", "consult", "project_knowledge"];

/// Tool authorization tier for a headless run.
///
/// Determines only the **soft** approval matrix; each tool's hard barriers are identical across
/// all three tiers (REQ-H09).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Only tools in [`READ_ONLY_TOOLS`] auto-approve (REQ-H06).
    Default,
    /// All registered tools auto-approve, hard barriers intact (REQ-H07).
    Auto,
    /// Like `Auto`, it also raises `max_tool_calls` and silences the agent's soft guards
    /// (REQ-H08).
    FullAuto,
}

/// Effective authorization policy for a headless run.
///
/// `max_tool_calls`/`timeout` travel here so the runner (later MS2 task) can consume them
/// together with the approval decision; neither one participates in the logic of
/// [`Policy::approves`], which depends only on the tier.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Active tier; determines the approval matrix and the warnings emitted.
    tier: Tier,
    /// Tool-call ceiling already resolved by the caller (no clamping here).
    max_tool_calls: u32,
    /// Wall-clock timeout in seconds, if set (REQ-H36; applied by T4).
    timeout: Option<u64>,
}

impl Policy {
    /// Builds a policy for `tier` with the limits already resolved.
    ///
    /// `max_tool_calls`/`timeout` are taken as-is from the caller — this function applies no
    /// cost clamp (that already happened during parameter resolution, `resolution::resolve`,
    /// previous MS1 task).
    #[must_use]
    pub fn new(tier: Tier, max_tool_calls: u32, timeout: Option<u64>) -> Self {
        Self {
            tier,
            max_tool_calls,
            timeout,
        }
    }

    /// Active tier of this policy.
    #[must_use]
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Tool-call ceiling for this run.
    #[must_use]
    pub fn max_tool_calls(&self) -> u32 {
        self.max_tool_calls
    }

    /// Wall-clock timeout in seconds, if one was set.
    #[must_use]
    pub fn timeout(&self) -> Option<u64> {
        self.timeout
    }

    /// Decides whether `tool_name` auto-approves under this policy's tier.
    ///
    /// **Fail-closed:** a name that belongs neither to [`READ_ONLY_TOOLS`] nor
    /// to the known read-write tools is never approved, in any tier — so a future tool
    /// registered in `main.rs` but not classified here is denied by default instead of auto-
    /// approved by omission.
    ///
    /// This function does **not** evaluate or relax any hard barrier: approval here is a
    /// necessary but not sufficient condition — the tool can still fail inside itself (`bash`
    /// allowlist, `PathGuard`).
    #[must_use]
    pub fn approves(&self, tool_name: &str) -> bool {
        let is_read_only = READ_ONLY_TOOLS.contains(&tool_name);
        match self.tier {
            Tier::Default => is_read_only,
            Tier::Auto | Tier::FullAuto => is_read_only || READ_WRITE_TOOLS.contains(&tool_name),
        }
    }

    /// Warnings to emit (stderr + log) for this policy, at the start of the run.
    ///
    /// Non-empty only under `FullAuto` (REQ-H08): the privilege elevation (raised cap +
    /// silenced soft guards) is never silent. `Default`/`Auto` do not raise anything and emit
    /// no warning.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        match self.tier {
            Tier::FullAuto => vec![format!(
                "WARNING: --full-auto is active — max_tool_calls is elevated \
                 to {FULL_AUTO_MAX_TOOL_CALLS} (from the normal \
                 {NORMAL_MAX_TOOL_CALLS}), and the repetitive-call soft guard \
                 is silenced. Hard barriers (bash allowlist, metacharacter \
                 ban, PathGuard) remain fully enforced in every tier."
            )],
            Tier::Default | Tier::Auto => Vec::new(),
        }
    }

    /// `true` if the runner should disable the `Agent`'s SOFT guards.
    ///
    /// Only `FullAuto` silences (a) the detection of 3 consecutive identical calls and (b) the
    /// normal `max_tool_calls` cap (replaced by the raised one) — REQ-H08. This policy only
    /// **declares** the intent: it has no reference to the `Agent` to apply it, the runner
    /// wires that. No hard barrier is affected by this signal.
    #[must_use]
    pub fn silences_soft_guards(&self) -> bool {
        matches!(self.tier, Tier::FullAuto)
    }
}

/// Fuzz entrypoint for the `fuzz_policy` target (MS2 Task 10 / REQ-H35): maps arbitrary bytes
/// to `(tier, nombre_de_tool)` and exercises the entire public surface of [`Policy`].
///
/// The first byte selects the tier (`0` ⇒ [`Tier::Default`], `1` ⇒ [`Tier::Auto`], any other ⇒
/// [`Tier::FullAuto`]) and the rest of the bytes are the tool name, converted with
/// `String::from_utf8_lossy` — so the input covers non-UTF8 names. Invariants verified over
/// EVERY input: **never panics** (the matrix is pure, total logic), and **fail-closed** — an
/// approval implies the name belongs to the known tool set in any tier (an unknown name never
/// returns `true`).
///
/// `#[doc(hidden)] pub` mirrors the convention of the vault's `fuzz_*_entrypoint` and of
/// [`output`](super::output): it exposes the boundary to the `fuzz/` crate without widening the
/// module's documented public API.
///
/// # Panics
///
/// Panics (under `debug_assertions`, which `cargo-fuzz` enables) only if the fail-closed
/// invariant is violated — that is the genuine bug the fuzzer is looking for, not a spurious
/// abort.
#[doc(hidden)]
pub fn fuzz_policy_entrypoint(data: &[u8]) {
    // First byte ⇒ tier; rest ⇒ tool name (lossy, covers non-UTF8). The fallback `(&0, &[])`
    // covers the empty input without indexing (fail-closed).
    let (&tier_byte, name_bytes) = data.split_first().unwrap_or((&0, &[]));
    let tier = match tier_byte {
        0 => Tier::Default,
        1 => Tier::Auto,
        _ => Tier::FullAuto,
    };

    // `max_tool_calls`/`timeout` derived from the tail to exercise the accessors with varied
    // values; they do not participate in the approval logic.
    let max_tool_calls = u32::try_from(name_bytes.len()).unwrap_or(u32::MAX);
    let timeout = name_bytes.first().map(|&b| u64::from(b));

    let name = String::from_utf8_lossy(name_bytes);
    let policy = Policy::new(tier, max_tool_calls, timeout);

    // The entire public surface must be total (never panic) over the input.
    let approved = policy.approves(&name);
    let _ = policy.silences_soft_guards();
    let _ = policy.warnings();
    let _ = policy.tier();
    let _ = policy.max_tool_calls();
    let _ = policy.timeout();

    // Fail-closed: an approval implies a known tool name, in any tier — an unknown name is
    // never auto-approved (REQ-H09).
    let name_ref: &str = name.as_ref();
    let is_known = READ_ONLY_TOOLS.contains(&name_ref) || READ_WRITE_TOOLS.contains(&name_ref);
    debug_assert!(
        !approved || is_known,
        "fail-closed violated: approved unknown tool name {name_ref:?} in tier {tier:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All tools registered in `main.rs`, as returned by their actual `Tool::name()` (verified
    /// in `src/tools/{ls,read,write,grep, bash,consult,knowledge}.rs` and in the `main.rs`
    /// registry on this file's date). It is the reference list for the maintenance test below:
    /// a new tool registered in `main.rs` without being added here (and to
    /// [`READ_ONLY_TOOLS`]/`READ_WRITE_TOOLS`) remains uncovered by this guard until a
    /// maintainer updates both lists — the `headless` module is pure and cannot import
    /// `crate::tools` to verify it dynamically (see module rustdoc).
    const REAL_REGISTERED_TOOL_NAMES: &[&str] = &[
        "ls",
        "view",
        "edit",
        "grep",
        "bash",
        "consult",
        "project_knowledge",
    ];

    /// The per-tier approval matrix is exhaustive over the known tools and fail-closed over any
    /// unknown name, in ALL tiers (REQ-H06/H07/H09; MS2.md Task 1 Step 1 verbatim).
    fn policy(tier: Tier) -> Policy {
        Policy::new(tier, NORMAL_MAX_TOOL_CALLS, None)
    }

    #[test]
    fn test_tier_approval_matrix_is_exhaustive_and_fail_closed() {
        let default = policy(Tier::Default);
        for ro in READ_ONLY_TOOLS {
            assert!(
                default.approves(ro),
                "{ro} debe auto-aprobarse en default (read-only)"
            );
        }
        for rw in READ_WRITE_TOOLS {
            assert!(!default.approves(rw), "{rw} NO debe aprobarse en default");
        }
        assert!(
            !default.approves("tool_que_no_existe"),
            "fail-closed: un tool desconocido nunca se aprueba en default"
        );

        for tier in [Tier::Auto, Tier::FullAuto] {
            let p = policy(tier);
            for known in READ_ONLY_TOOLS.iter().chain(READ_WRITE_TOOLS.iter()) {
                assert!(
                    p.approves(known),
                    "{known} debe aprobarse en {tier:?} (todos los registrados)"
                );
            }
            assert!(
                !p.approves("tool_que_no_existe"),
                "fail-closed: un tool desconocido nunca se aprueba, ni en {tier:?}"
            );
        }
    }

    /// The known set of [`READ_ONLY_TOOLS`] + `READ_WRITE_TOOLS` must match exactly (same size
    /// and same elements) the real registry in `main.rs`, so that a tool added there but
    /// forgotten in this policy's classification fails this test instead of being silently
    /// denied or approved by omission.
    #[test]
    fn test_known_tool_set_matches_real_tool_registry() {
        let mut known: Vec<&str> = READ_ONLY_TOOLS
            .iter()
            .copied()
            .chain(READ_WRITE_TOOLS.iter().copied())
            .collect();
        known.sort_unstable();

        let mut real: Vec<&str> = REAL_REGISTERED_TOOL_NAMES.to_vec();
        real.sort_unstable();

        assert_eq!(
            known, real,
            "READ_ONLY_TOOLS + READ_WRITE_TOOLS debe coincidir con el registro \
             real de main.rs — actualizar ambas listas al registrar un tool nuevo"
        );
    }

    /// `silences_soft_guards` is `true` only under `FullAuto`.
    #[test]
    fn test_silences_soft_guards_true_only_for_full_auto() {
        assert!(!policy(Tier::Default).silences_soft_guards());
        assert!(!policy(Tier::Auto).silences_soft_guards());
        assert!(policy(Tier::FullAuto).silences_soft_guards());
    }

    /// `warnings()` is empty in `Default`/`Auto` and contains the limit-elevation warning in
    /// `FullAuto` (REQ-H08).
    #[test]
    fn test_warnings_nonempty_only_for_full_auto_and_mentions_elevation() {
        assert!(policy(Tier::Default).warnings().is_empty());
        assert!(policy(Tier::Auto).warnings().is_empty());

        let warnings = policy(Tier::FullAuto).warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--full-auto"));
        assert!(warnings[0].contains(&FULL_AUTO_MAX_TOOL_CALLS.to_string()));
        assert!(warnings[0].contains("soft guard"));
    }

    /// Edge: `approves` with an empty string (never a real tool name) is denied in all tiers —
    /// fail-closed even on empty input.
    #[test]
    fn test_approves_denies_empty_tool_name_in_every_tier() {
        for tier in [Tier::Default, Tier::Auto, Tier::FullAuto] {
            assert!(!policy(tier).approves(""));
        }
    }

    /// The accessors expose the values passed to `new` exactly as-is (no clamping), including
    /// the edge case `timeout: None`.
    #[test]
    fn test_new_accessors_expose_constructor_values_unmodified() {
        let p = Policy::new(Tier::Auto, 42, Some(900));
        assert_eq!(p.tier(), Tier::Auto);
        assert_eq!(p.max_tool_calls(), 42);
        assert_eq!(p.timeout(), Some(900));

        let no_timeout = Policy::new(Tier::Default, 15, None);
        assert_eq!(no_timeout.timeout(), None);
    }

    /// Unit-smoke of the `fuzz_policy` fuzz entrypoint (REQ-H35): degenerate inputs (empty,
    /// out-of-range tier, non-UTF8 tail, unknown tool) never panic and respect fail-closed.
    /// This is the local version that DOES run on every §0.1, complementing CI's coverage-
    /// guided run.
    #[test]
    fn test_fuzz_policy_entrypoint_never_panics_on_arbitrary_input() {
        let cases: &[&[u8]] = &[
            b"",
            b"\x00",
            b"\x01",
            b"\x00ls",
            b"\x01bash",
            b"\xffedit",
            b"\x00tool_que_no_existe",
            &[0x02, 0xff, 0xfe, 0xfd],
        ];
        for case in cases {
            fuzz_policy_entrypoint(case);
        }
    }
}
