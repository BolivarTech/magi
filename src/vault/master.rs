// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! Master passphrase resolution and strength enforcement (MS2, zero-knowledge).
//!
//! The passphrase is the ONLY key: there is no keyring, no recovery, no backdoor
//! (REQ-V23). It resolves with a deterministic precedence (`-p`, then
//! `MAGI_PASSPHRASE`, then interactive prompt) and is used only to derive the KEK,
//! then zeroized (the caller holds it in [`Zeroizing`]). A moved `.db` is
//! brute-forced offline
//! with no rate-limiting, creation/rotation enforce a **hard** strength floor
//! (`zxcvbn` score ≥ 3, ≥ 12 chars) with **no override** (REQ-V18).

use std::io::IsTerminal;

use zeroize::Zeroizing;
use zxcvbn::zxcvbn;

use crate::vault::VaultError;

/// Environment variable that supplies the master passphrase in headless/CI use.
pub const PASSPHRASE_ENV: &str = "MAGI_PASSPHRASE";

/// Absolute minimum passphrase length (chars, not bytes). 12 is the floor below
/// which even a high-`zxcvbn` estimate is untrustworthy for offline brute-force.
pub const MIN_PASSPHRASE_CHARS: usize = 12;

/// Minimum `zxcvbn` score to accept (0–4). 3 = "safely unguessable: moderate
/// protection from an offline slow-hash scenario" — the correct floor for a
/// portable `.db` attacked offline without rate-limiting (REQ-V18).
const MIN_ZXCVBN_SCORE: u8 = 3;

/// Injectable interactive input (R-V07: tests never touch a real TTY).
pub trait PassphrasePrompt {
    /// Whether stdin is an interactive terminal (`IsTerminal`, D-V08).
    fn is_interactive(&self) -> bool;

    /// Reads one passphrase. `show=false` ⇒ hidden (no echo); `show=true` ⇒ live
    /// echo (REQ-V21, for `passwd --show`). `show` is ignored without a TTY.
    ///
    /// # Errors
    /// [`VaultError::Io`] if the terminal read fails.
    fn read_passphrase(&mut self, msg: &str, show: bool) -> Result<Zeroizing<String>, VaultError>;
}

/// Production [`PassphrasePrompt`] over the real stdin/stderr.
pub struct TtyPrompt;

impl PassphrasePrompt for TtyPrompt {
    fn is_interactive(&self) -> bool {
        std::io::stdin().is_terminal()
    }

    fn read_passphrase(&mut self, msg: &str, show: bool) -> Result<Zeroizing<String>, VaultError> {
        if show {
            eprint!("{msg}");
            use std::io::Write;
            std::io::stderr().flush().ok();
            let mut line = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .map_err(|e| VaultError::Io(e.to_string()))?;
            Ok(Zeroizing::new(strip_trailing_newline(line)))
        } else {
            rpassword::prompt_password(msg)
                .map(Zeroizing::new)
                .map_err(|e| VaultError::Io(e.to_string()))
        }
    }
}

/// Removes at most ONE trailing `\n` or `\r\n` (never other whitespace).
///
/// A submitted line's newline is never part of an intended passphrase, so
/// stripping it aligns `-p`/env with the interactive prompt and prevents the
/// silent lockout of `MAGI_PASSPHRASE=$(cat f)` (trailing `\n`). Inner or
/// deliberate trailing spaces/tabs are preserved (REQ-V18 / MAGI run 10).
fn strip_trailing_newline(s: String) -> String {
    if let Some(stripped) = s.strip_suffix("\r\n") {
        stripped.to_string()
    } else if let Some(stripped) = s.strip_suffix('\n') {
        stripped.to_string()
    } else {
        s
    }
}

/// Resolves the master passphrase: `-p` > `MAGI_PASSPHRASE` > interactive prompt.
///
/// A trailing newline is stripped from `-p`/env. An empty `MAGI_PASSPHRASE`
/// counts as absent. **Without a TTY and without `-p`/env, fails closed with
/// [`VaultError::PassphraseUnavailable`] and NEVER reads stdin** (REQ-V40:
/// stdin is reserved for the secret value). An empty prompt entry aborts the
/// same way (user cancelled).
///
/// # Errors
/// [`VaultError::PassphraseUnavailable`] as described; [`VaultError::Io`] on a
/// terminal read failure.
pub fn resolve_passphrase(
    flag: Option<Zeroizing<String>>,
    prompt: &mut dyn PassphrasePrompt,
) -> Result<Zeroizing<String>, VaultError> {
    if let Some(p) = flag {
        return Ok(Zeroizing::new(strip_trailing_newline(
            p.as_str().to_string(),
        )));
    }
    if let Ok(env) = std::env::var(PASSPHRASE_ENV) {
        if !env.is_empty() {
            return Ok(Zeroizing::new(strip_trailing_newline(env)));
        }
    }
    if !prompt.is_interactive() {
        return Err(VaultError::PassphraseUnavailable);
    }
    let entered = prompt.read_passphrase("Passphrase: ", false)?;
    if entered.is_empty() {
        return Err(VaultError::PassphraseUnavailable);
    }
    Ok(entered)
}

/// Enforces the hard strength floor (REQ-V18): `< MIN_PASSPHRASE_CHARS` chars or
/// `zxcvbn` score `< MIN_ZXCVBN_SCORE` ⇒ rejected. No override, no composition
/// rules. Runs only on create/rotate, never on unlock.
///
/// # Errors
/// [`VaultError::WeakPassphrase`] with the reasons + tips (never the passphrase).
pub fn check_strength(passphrase: &str) -> Result<(), VaultError> {
    if passphrase.chars().count() < MIN_PASSPHRASE_CHARS {
        return Err(VaultError::WeakPassphrase(format!(
            "too short (need at least {MIN_PASSPHRASE_CHARS} characters); \
             a passphrase of 4+ random words is strong and easy to recall"
        )));
    }
    let estimate = zxcvbn(passphrase, &[]);
    if u8::from(estimate.score()) < MIN_ZXCVBN_SCORE {
        let mut reason = String::from("too easy to guess");
        if let Some(feedback) = estimate.feedback() {
            if let Some(warning) = feedback.warning() {
                reason = format!("{reason}: {warning}");
            }
            let tips: Vec<String> = feedback
                .suggestions()
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            if !tips.is_empty() {
                reason = format!("{reason} ({})", tips.join("; "));
            }
        }
        return Err(VaultError::WeakPassphrase(format!(
            "{reason}. Try 4+ random words; length matters more than symbols"
        )));
    }
    Ok(())
}

/// The verbatim zero-knowledge warning shown on first-run and `passwd` (REQ-V23).
const ZK_WARNING: &str =
    "no recovery: if you forget the passphrase, the data is lost. There is no backdoor.";

/// First-run (REQ-V17) and `passwd`: double entry + [`check_strength`] + the
/// zero-knowledge no-recovery warning. `show` (REQ-V21) is threaded to both reads.
///
/// A mismatch re-prompts; an empty entry aborts. The strength floor is enforced
/// before acceptance — with no override.
///
/// # Errors
/// [`VaultError::PassphraseUnavailable`] if the user aborts (empty entry);
/// [`VaultError::WeakPassphrase`] if the floor is not met; [`VaultError::Io`] on
/// a terminal read failure.
pub fn create_passphrase(
    prompt: &mut dyn PassphrasePrompt,
    show: bool,
) -> Result<Zeroizing<String>, VaultError> {
    loop {
        let first = prompt.read_passphrase(&format!("New passphrase ({ZK_WARNING}): "), show)?;
        if first.is_empty() {
            return Err(VaultError::PassphraseUnavailable);
        }
        check_strength(first.as_str())?;
        let second = prompt.read_passphrase("Confirm passphrase: ", show)?;
        if first.as_str() == second.as_str() {
            return Ok(first);
        }
        // Mismatch: loop and re-prompt (a transient user error, not a failure).
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_strength, create_passphrase, resolve_passphrase, PassphrasePrompt, PASSPHRASE_ENV,
    };
    use crate::vault::VaultError;
    use zeroize::Zeroizing;

    /// Fake prompt: scripted answers in order; records `show` and `msg` per read.
    struct FakePrompt {
        interactive: bool,
        answers: Vec<String>,
        reads: usize,
        shows: Vec<bool>,
        msgs: Vec<String>,
    }
    impl PassphrasePrompt for FakePrompt {
        fn is_interactive(&self) -> bool {
            self.interactive
        }
        fn read_passphrase(
            &mut self,
            msg: &str,
            show: bool,
        ) -> Result<Zeroizing<String>, VaultError> {
            let i = self.reads;
            self.reads += 1;
            self.shows.push(show);
            self.msgs.push(msg.to_string());
            Ok(Zeroizing::new(
                self.answers.get(i).cloned().unwrap_or_default(),
            ))
        }
    }
    fn fp(interactive: bool, answers: Vec<&str>) -> FakePrompt {
        FakePrompt {
            interactive,
            answers: answers.into_iter().map(Into::into).collect(),
            reads: 0,
            shows: vec![],
            msgs: vec![],
        }
    }

    /// RAII env guard (no `temp_env` dep); restores the prior value on drop.
    fn with_var<R>(key: &str, val: Option<&str>, f: impl FnOnce() -> R) -> R {
        struct Guard {
            key: String,
            prev: Option<String>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self.prev {
                    Some(v) => std::env::set_var(&self.key, v),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
        let _g = Guard {
            key: key.to_string(),
            prev: std::env::var(key).ok(),
        };
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        f()
    }

    #[test]
    #[serial_test::serial]
    fn test_flag_takes_precedence_over_env_and_prompt() {
        let mut p = fp(true, vec!["from-prompt"]);
        with_var(PASSPHRASE_ENV, Some("from-env"), || {
            let r =
                resolve_passphrase(Some(Zeroizing::new("from-flag".into())), &mut p).expect("ok");
            assert_eq!(r.as_str(), "from-flag");
            assert_eq!(p.reads, 0);
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_env_takes_precedence_over_prompt_and_empty_env_counts_as_absent() {
        let mut p = fp(true, vec!["from-prompt"]);
        with_var(PASSPHRASE_ENV, Some("from-env"), || {
            assert_eq!(
                resolve_passphrase(None, &mut p).expect("ok").as_str(),
                "from-env"
            );
        });
        with_var(PASSPHRASE_ENV, Some(""), || {
            assert_eq!(
                resolve_passphrase(None, &mut p).expect("ok").as_str(),
                "from-prompt"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_no_tty_without_flag_or_env_fails_closed_without_reading_stdin() {
        let mut p = fp(false, vec!["pipe-data"]);
        with_var(PASSPHRASE_ENV, None, || {
            let e = resolve_passphrase(None, &mut p).expect_err("fail-closed");
            assert!(matches!(e, VaultError::PassphraseUnavailable));
            assert_eq!(p.reads, 0);
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_trailing_newline_stripped_from_flag_and_env_but_not_inner_whitespace() {
        let mut p = fp(true, vec!["x"]);
        let r =
            resolve_passphrase(Some(Zeroizing::new("pass phrase\n".into())), &mut p).expect("ok");
        assert_eq!(r.as_str(), "pass phrase");
        with_var(PASSPHRASE_ENV, Some("secret \r\n"), || {
            let e = resolve_passphrase(None, &mut p).expect("ok");
            assert_eq!(e.as_str(), "secret ");
        });
    }

    #[test]
    fn test_short_or_low_score_passphrases_are_hard_rejected_without_override() {
        for weak in ["short", "password123!", "qwertyuiop12"] {
            assert!(
                matches!(check_strength(weak), Err(VaultError::WeakPassphrase(_))),
                "should reject: {weak}"
            );
        }
    }

    #[test]
    fn test_diceware_lowercase_words_are_accepted_without_composition_rules() {
        // Verified empirically with zxcvbn 3.1.1: score 4 (>= floor 3).
        check_strength("correct horse battery staple").expect("diceware >= 3");
    }

    #[test]
    fn test_weak_passphrase_message_never_contains_the_passphrase() {
        let probe = "hunter2hunter2";
        if let Err(VaultError::WeakPassphrase(msg)) = check_strength(probe) {
            assert!(!msg.contains(probe));
        }
    }

    #[test]
    fn test_create_passphrase_requires_matching_double_entry() {
        let mut p = fp(
            true,
            vec![
                "correct horse battery staple",
                "MISMATCH-XYZ",
                "correct horse battery staple",
                "correct horse battery staple",
            ],
        );
        let r = create_passphrase(&mut p, false).expect("ok");
        assert_eq!(r.as_str(), "correct horse battery staple");
        assert_eq!(p.reads, 4);
    }

    #[test]
    fn test_create_passphrase_threads_show_flag_to_both_reads() {
        let mut p = fp(true, vec!["correct horse battery staple"; 2]);
        create_passphrase(&mut p, true).expect("ok");
        assert_eq!(p.shows, vec![true, true]);
    }

    #[test]
    fn test_create_passphrase_emits_zero_knowledge_no_recovery_warning() {
        let mut p = fp(true, vec!["correct horse battery staple"; 2]);
        create_passphrase(&mut p, false).expect("ok");
        assert!(p
            .msgs
            .iter()
            .any(|m| m.to_lowercase().contains("no recovery")
                || m.to_lowercase().contains("data is lost")));
    }
}
