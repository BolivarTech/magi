// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Mapping of a headless outcome to an actionable exit code (REQ-H23/H23b).
//!
//! [`exit_code`] is a **pure and total** function: there is no branch that fails to produce a
//! code, and the internal `match` over [`HeadlessError`] is **exhaustive with no `_` wildcard**
//! — a new variant from the source breaks the build instead of silently degrading to a default
//! exit code (same pattern as `HeadlessError::from(VaultError)`, T0).
//!
//! **Precedence** (REQ-H23, "when several conditions co-occur"): a
//! typed error present always dominates — `InputInvalid`/`InputTooLarge` ⇒ `EXIT_MISUSE`, any
//! other variant ⇒ `EXIT_RUNTIME`. Only in
//! **absence** of error, the deterministic criterion for exit 3 is evaluated
//! (REQ-H23b): the tier denied at least one tool **and** the agent's final turn produced no
//! response (`response_empty`) **and** the resulting `stop_reason` is [`StopReason::Denied`] —
//! the three signals coincide by construction (`Denied` is assigned precisely under that
//! condition), but all three are verified so as not to rely on a single channel. Any other case
//! is success (`EXIT_OK`).
//!
//! `exit_code` is `pub`: the MS2 runner lives in the binary crate and can only reach `pub` APIs
//! from the lib.

use super::types::StopReason;
use super::HeadlessError;

/// Exit code of a successful run (REQ-H23).
const EXIT_OK: i32 = 0;

/// Exit code of a runtime/agent error: corrupt DB, wrong or unavailable passphrase, I/O,
/// storage, cancellation, timeout (REQ-H23).
const EXIT_RUNTIME: i32 = 1;

/// Exit code for CLI misuse or invalid input: envelope without `prompt`, input that exceeds the
/// size cap, invalid format (REQ-H23).
const EXIT_MISUSE: i32 = 2;

/// Exit code when the authorization tier blocked the task: at least one essential tool was
/// denied and the agent produced no response (REQ-H23b).
const EXIT_TIER_DENIED: i32 = 3;

/// Computes the exit code of a headless run according to its outcome.
///
/// # Parameters
/// - `err`: typed error from the run, if any; its presence dominates the
/// precedence (see the module rustdoc).
/// - `stop_reason`: reason the agent loop ended.
/// - `response_empty`: whether the agent's final turn produced no
/// text block (REQ-H23b — "empty" means *zero* `TextDelta`, not just whitespace; the caller
/// resolves that distinction before invoking this function).
/// - `tier_denied`: whether at least one tool was denied by the tier during the
/// run (regardless of whether the agent managed to work around the denial).
///
/// Returns one of `EXIT_OK`, `EXIT_RUNTIME`, `EXIT_MISUSE` or `EXIT_TIER_DENIED`.
pub fn exit_code(
    err: Option<&HeadlessError>,
    stop_reason: StopReason,
    response_empty: bool,
    tier_denied: bool,
) -> i32 {
    if let Some(e) = err {
        return exit_code_for_error(e);
    }
    if tier_denied && response_empty && stop_reason == StopReason::Denied {
        return EXIT_TIER_DENIED;
    }
    EXIT_OK
}

/// Translates a concrete [`HeadlessError`] to its exit-code class.
///
/// The `match` is **exhaustive with no `_` wildcard**: adding a variant to [`HeadlessError`]
/// forces its mapping to be decided here instead of silently defaulting to [`EXIT_RUNTIME`].
fn exit_code_for_error(err: &HeadlessError) -> i32 {
    match err {
        HeadlessError::InputInvalid(_) | HeadlessError::InputTooLarge(_) => EXIT_MISUSE,
        HeadlessError::Io(_)
        | HeadlessError::Storage(_)
        | HeadlessError::Aborted
        | HeadlessError::PassphraseUnavailable
        | HeadlessError::Db(_) => EXIT_RUNTIME,
    }
}

#[cfg(test)]
mod tests {
    use super::{exit_code, EXIT_MISUSE, EXIT_OK, EXIT_RUNTIME, EXIT_TIER_DENIED};
    use crate::headless::types::StopReason;
    use crate::headless::HeadlessError;
    use crate::vault::VaultError;

    /// Success with no error, no denial ⇒ 0.
    #[test]
    fn test_exit_codes_taxonomy() {
        assert_eq!(exit_code(None, StopReason::Done, false, false), EXIT_OK);
        assert_eq!(
            exit_code(
                Some(&HeadlessError::InputInvalid("x".into())),
                StopReason::Error,
                true,
                false
            ),
            EXIT_MISUSE
        );
        // tier blocked the task: denied + no response + stop_reason=denied.
        assert_eq!(
            exit_code(None, StopReason::Denied, true, true),
            EXIT_TIER_DENIED
        );
        // the agent worked around the denial: produced a response ⇒ success.
        assert_eq!(exit_code(None, StopReason::Done, false, true), EXIT_OK);
    }

    /// `InputTooLarge` is CLI misuse (input that exceeds the cap) ⇒ 2.
    #[test]
    fn test_input_too_large_maps_to_misuse() {
        let err = HeadlessError::InputTooLarge(10 * 1024 * 1024);
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_MISUSE
        );
    }

    /// Passphrase unavailable (fail-closed without TTY) ⇒ runtime error.
    #[test]
    fn test_passphrase_unavailable_maps_to_runtime_error() {
        let err = HeadlessError::PassphraseUnavailable;
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Corrupt DB (wrapped from `VaultError`) ⇒ runtime error.
    #[test]
    fn test_db_corrupt_maps_to_runtime_error() {
        let err: HeadlessError = VaultError::DbCorrupt {
            db_path: std::path::PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".into(),
        }
        .into();
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Wrong passphrase (wrapped from `VaultError`) ⇒ runtime error.
    #[test]
    fn test_wrong_passphrase_maps_to_runtime_error() {
        let err: HeadlessError = VaultError::WrongPassphrase.into();
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// I/O and storage are runtime errors, not misuse.
    #[test]
    fn test_io_and_storage_map_to_runtime_error() {
        let io = HeadlessError::Io("disk full".into());
        let storage = HeadlessError::Storage("locked".into());
        assert_eq!(
            exit_code(Some(&io), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
        assert_eq!(
            exit_code(Some(&storage), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Explicit cancellation (`Aborted`) is a runtime error, not success.
    #[test]
    fn test_aborted_maps_to_runtime_error() {
        let err = HeadlessError::Aborted;
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// A present typed error dominates over the tier-denied signal: even with
    /// `tier_denied=true` and `response_empty=true`, the error wins (2, not 3).
    #[test]
    fn test_error_precedence_wins_over_tier_denied_signal() {
        let err = HeadlessError::InputInvalid("bad envelope".into());
        assert_eq!(
            exit_code(Some(&err), StopReason::Denied, true, true),
            EXIT_MISUSE
        );
    }

    /// `tier_denied` without `response_empty` never produces exit 3, even if the reported
    /// `stop_reason` is `Denied` (defensive inconsistency).
    #[test]
    fn test_tier_denied_without_empty_response_is_success() {
        assert_eq!(exit_code(None, StopReason::Denied, false, true), EXIT_OK);
    }
}
