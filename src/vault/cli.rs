// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! `magi-rs vault` subcommands: `ls` / `set` / `rm` / `passwd` (REQ-V07..V11,
//! V21, V22, V40).
//!
//! There is deliberately no `get`/`cat`/`show <name>`/`reveal`/`export`
//! subcommand — nothing in this module can ever print a *stored* secret's
//! value (REQ-V09; mechanized by `test_no_subcommand_exposes_a_stored_value`,
//! SC-V05), and a value is never accepted as a CLI argument (SC-V11): it is
//! only ever supplied through [`VaultIo::read_value`], hidden behind a TTY
//! prompt or read from stdin.
//!
//! [`run_vault_cmd`] is the single entry point; it drives an already-unlocked
//! [`SecretStore`] through an injected [`VaultIo`] (R-V07: no test here ever
//! touches a real terminal). [`TtyIo`] is the production implementation, over
//! the real stdin/stdout/stderr.

use std::io::{self, Read, Write};

use cryptovault::MAX_PLAINTEXT_LEN;
use zeroize::Zeroizing;

use crate::vault::{create_passphrase, PassphrasePrompt, SecretStore, TtyPrompt, VaultError};

/// Suffix appended to every deliberate `[Y/n]` confirmation prompt (REQ-V22),
/// factored out because both `set`'s overwrite prompt and `rm`'s deletion
/// prompt use it verbatim.
const CONFIRM_SUFFIX: &str = "? [Y/n] ";

/// Placeholder line `ls` prints for an empty vault, instead of a silently
/// empty table (REQ-V08).
const EMPTY_VAULT_LINE: &str = "(vault empty)";

/// Fails with [`VaultError::ValueTooLarge`] if `len` (bytes) exceeds
/// `cryptovault::MAX_PLAINTEXT_LEN`. Shared by the bounded-stdin read and the
/// TTY-entered value check so the same ceiling applies identically
/// regardless of which path supplied the value.
///
/// # Errors
/// [`VaultError::ValueTooLarge`] as described.
fn enforce_value_ceiling(len: usize) -> Result<(), VaultError> {
    if len > MAX_PLAINTEXT_LEN {
        Err(VaultError::ValueTooLarge(MAX_PLAINTEXT_LEN))
    } else {
        Ok(())
    }
}

/// Removes at most one trailing `\n` or `\r\n` from a line read from a
/// prompt or a pipe (mirrors `crate::vault::master`'s identical rule for the
/// passphrase): a submitted line's terminator is never part of an intended
/// value, but inner or deliberately trailing spaces/tabs are preserved.
fn strip_trailing_newline(s: String) -> String {
    if let Some(stripped) = s.strip_suffix("\r\n") {
        stripped.to_string()
    } else if let Some(stripped) = s.strip_suffix('\n') {
        stripped.to_string()
    } else {
        s
    }
}

/// SC-V22: a `[Y/n]` confirmation proceeds if and only if `line` is EXACTLY
/// the single uppercase letter `"Y"` — `"y"`, an empty line (a bare Enter),
/// `"yes"`, and anything else return `false`. See [`VaultIo::confirm`] for
/// the full rationale.
fn is_affirmative(line: &str) -> bool {
    line == "Y"
}

/// Reads all of `reader`, bounded to `MAX_PLAINTEXT_LEN + 1` bytes so a
/// hostile unbounded source (a pipe of gigabytes) can never be buffered in
/// full before the size check runs (anti-DoS). Strips at most one trailing
/// `\n`/`\r\n`; inner newlines are preserved, so multi-line values such as
/// PEM keys round-trip intact.
///
/// The bytes are decoded **strictly** as UTF-8: invalid input fails with a
/// typed error rather than being lossily coerced to U+FFFD and stored as a
/// silently-wrong secret (this mirrors the strict decode of the read path —
/// REQ-V03). The size ceiling is enforced on the **stored** (newline-stripped)
/// length, so a value of exactly `MAX_PLAINTEXT_LEN` bytes with a trailing
/// newline is accepted, consistent with the TTY path.
///
/// # Errors
/// [`VaultError::Io`] if the underlying read fails or the bytes are not valid
/// UTF-8; [`VaultError::ValueTooLarge`] if the stored value would exceed
/// `MAX_PLAINTEXT_LEN` — checked without ever buffering more than
/// `MAX_PLAINTEXT_LEN + 1` bytes.
fn read_bounded_value(reader: impl Read) -> Result<Zeroizing<String>, VaultError> {
    let mut buf = Vec::new();
    reader
        .take(MAX_PLAINTEXT_LEN as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| VaultError::Io(e.to_string()))?;
    let decoded = String::from_utf8(buf)
        .map_err(|_| VaultError::Io("piped value is not valid UTF-8".to_string()))?;
    let text = strip_trailing_newline(decoded);
    enforce_value_ceiling(text.len())?;
    Ok(Zeroizing::new(text))
}

/// Subcommands of `magi-rs vault` (REQ-V07). There is deliberately no
/// `get`/`cat`/`show <name>`/`reveal`/`export` subcommand — see the module
/// docs (REQ-V09 / SC-V05).
#[derive(clap::Subcommand, Debug)]
pub enum VaultCmd {
    /// Lists every secret's name and timestamps — never a value (REQ-V08).
    Ls,
    /// Adds a new secret or overwrites an existing one (idempotent). The
    /// value is entered hidden, either via a TTY prompt or via stdin
    /// (REQ-V10).
    Set {
        /// The secret's name (the `vault` table's primary key).
        name: String,
        /// Echoes the value live WHILE it is typed (TTY only; ignored
        /// without a TTY, where the value comes from stdin instead). Shows
        /// what is being ENTERED now, never a previously stored value
        /// (REQ-V21).
        #[arg(long)]
        show: bool,
        /// Skips the `[Y/n]` overwrite confirmation (scripts/CI).
        #[arg(short, long)]
        force: bool,
    },
    /// Deletes a secret (with a `[Y/n]` confirmation, REQ-V22).
    Rm {
        /// The secret's name to delete.
        name: String,
        /// Skips the `[Y/n]` deletion confirmation (scripts/CI).
        #[arg(short, long)]
        force: bool,
    },
    /// Changes the master passphrase: re-wraps the existing data key in
    /// O(1) time. Does **not** re-encrypt any stored data (D-V07) — see the
    /// CLI documentation for exactly what that limitation does and does not
    /// protect against.
    Passwd {
        /// Echoes the new passphrase live while it is typed (TTY only;
        /// REQ-V21).
        #[arg(long)]
        show: bool,
    },
}

/// Injectable CLI I/O (R-V07): value entry, deliberate confirmations, and
/// output — everything [`run_vault_cmd`] needs beyond the store itself.
///
/// Supertrait of [`PassphrasePrompt`] so the SAME `io` also serves
/// `passwd`'s new-passphrase entry: `run_vault_cmd` passes
/// `io as &mut dyn PassphrasePrompt` to [`create_passphrase`] via trait
/// upcasting (stable since rustc 1.86; this project targets 1.97.0).
/// [`TtyIo`] implements both traits for production use; tests use a fake
/// double that implements both as well.
pub trait VaultIo: PassphrasePrompt {
    /// Reads a secret's value: hidden (no echo) or with live echo if
    /// `show`, from a TTY; the *entire* stdin stream if there is no TTY
    /// (`show` is then ignored — the value comes from a pipe, not a
    /// keystroke echo). At most one trailing `\n`/`\r\n` is stripped either
    /// way; inner newlines are preserved so multi-line values (e.g. PEM
    /// keys) round-trip intact.
    ///
    /// # Errors
    /// [`VaultError::Io`] if the underlying read fails;
    /// [`VaultError::ValueTooLarge`] if the value exceeds
    /// `cryptovault::MAX_PLAINTEXT_LEN` — checked without ever buffering
    /// more than that limit plus one byte (anti-DoS).
    fn read_value(&mut self, prompt_msg: &str, show: bool)
        -> Result<Zeroizing<String>, VaultError>;

    /// A deliberate `[Y/n]` confirmation (REQ-V22/SC-V22): returns
    /// `Ok(true)` if and only if the answer is EXACTLY the uppercase letter
    /// `"Y"` — `"y"`, a bare Enter, `"yes"`, and anything else return
    /// `Ok(false)`, never an error (the caller decides whether "no" means
    /// [`VaultError::Aborted`]).
    ///
    /// # Errors
    /// [`VaultError::Io`] if the underlying read fails.
    fn confirm(&mut self, msg: &str) -> Result<bool, VaultError>;

    /// Writes one line of output. Never used to print a *stored* secret's
    /// value (REQ-V09).
    fn println(&mut self, line: &str);
}

/// Production [`VaultIo`] over the real stdin/stdout/stderr.
///
/// Composes [`TtyPrompt`] for the [`PassphrasePrompt`] half by delegation,
/// so the two traits share one TTY-detection rule instead of duplicating
/// it.
pub struct TtyIo {
    /// The shared TTY-aware passphrase prompt this type delegates to.
    prompt: TtyPrompt,
}

impl TtyIo {
    /// Creates a new production I/O handle over the real terminal/stdin.
    pub fn new() -> Self {
        Self { prompt: TtyPrompt }
    }
}

impl Default for TtyIo {
    fn default() -> Self {
        Self::new()
    }
}

impl PassphrasePrompt for TtyIo {
    fn is_interactive(&self) -> bool {
        self.prompt.is_interactive()
    }

    fn read_passphrase(&mut self, msg: &str, show: bool) -> Result<Zeroizing<String>, VaultError> {
        self.prompt.read_passphrase(msg, show)
    }
}

impl VaultIo for TtyIo {
    fn read_value(
        &mut self,
        prompt_msg: &str,
        show: bool,
    ) -> Result<Zeroizing<String>, VaultError> {
        if self.is_interactive() {
            let value = if show {
                eprint!("{prompt_msg}");
                io::stderr().flush().ok();
                let mut line = String::new();
                io::stdin()
                    .read_line(&mut line)
                    .map_err(|e| VaultError::Io(e.to_string()))?;
                strip_trailing_newline(line)
            } else {
                rpassword::prompt_password(prompt_msg).map_err(|e| VaultError::Io(e.to_string()))?
            };
            enforce_value_ceiling(value.len())?;
            Ok(Zeroizing::new(value))
        } else {
            read_bounded_value(io::stdin())
        }
    }

    fn confirm(&mut self, msg: &str) -> Result<bool, VaultError> {
        eprint!("{msg}");
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| VaultError::Io(e.to_string()))?;
        Ok(is_affirmative(&strip_trailing_newline(line)))
    }

    fn println(&mut self, line: &str) {
        println!("{line}");
    }
}

/// Executes one `vault` subcommand against an already-unlocked store.
///
/// `rekey` is `passwd`'s effect on the envelope, injected as a closure
/// (Task 7 builds the real one by partially applying
/// `crate::vault::rekey_envelope` with the connection, the crypto pipeline,
/// and the ALREADY-VERIFIED current passphrase — verifying the current
/// passphrase happens when the caller unlocks the envelope, not here); it
/// receives the NEW passphrase.
///
/// # Errors
/// [`VaultError::Aborted`] if a `[Y/n]` confirmation did not answer exactly
/// `"Y"`, or if a destructive operation was attempted without a TTY and
/// without `-f`; otherwise propagates whatever the store, `io`, or `rekey`
/// returned.
pub fn run_vault_cmd(
    cmd: VaultCmd,
    store: &mut dyn SecretStore,
    io: &mut dyn VaultIo,
    rekey: &mut dyn FnMut(&str) -> Result<(), VaultError>,
) -> Result<(), VaultError> {
    match cmd {
        VaultCmd::Ls => run_ls(store, io),
        VaultCmd::Set { name, show, force } => run_set(&name, show, force, store, io),
        VaultCmd::Rm { name, force } => run_rm(&name, force, store, io),
        VaultCmd::Passwd { show } => run_passwd(show, io, rekey),
    }
}

/// `ls`: prints `NAME · CREATED · UPDATED`, column-aligned to the longest
/// name, or a single placeholder line if the vault is empty. Never a value
/// (REQ-V08 / SC-V04).
///
/// O(n) in the number of entries: one pass to find the longest name, one
/// pass to print.
///
/// # Errors
/// Propagates whatever [`SecretStore::list`] returned.
fn run_ls(store: &mut dyn SecretStore, io: &mut dyn VaultIo) -> Result<(), VaultError> {
    let entries = store.list()?;
    if entries.is_empty() {
        io.println(EMPTY_VAULT_LINE);
        return Ok(());
    }
    let width = entries
        .iter()
        .map(|e| e.name.chars().count())
        .max()
        .unwrap_or(0);
    for entry in &entries {
        io.println(&format!(
            "{name:width$} · {created} · {updated}",
            name = entry.name,
            created = entry.created_at,
            updated = entry.updated_at,
            width = width
        ));
    }
    Ok(())
}

/// `set`: confirms an overwrite of an EXISTING name BEFORE reading the value
/// (never makes the user type a secret only to abort afterwards), then
/// stores it.
///
/// # Errors
/// [`VaultError::Aborted`] as described in [`confirm_destructive`];
/// otherwise propagates whatever the store or `io` returned.
fn run_set(
    name: &str,
    show: bool,
    force: bool,
    store: &mut dyn SecretStore,
    io: &mut dyn VaultIo,
) -> Result<(), VaultError> {
    if store.contains(name)? {
        confirm_destructive(
            io,
            force,
            &format!("overwrite existing secret '{name}'{CONFIRM_SUFFIX}"),
        )?;
    }
    let value = io.read_value(&format!("value for '{name}': "), show)?;
    store.set(name, value.as_str())?;
    io.println(&format!("secret '{name}' stored"));
    Ok(())
}

/// `rm`: always confirms, then deletes.
///
/// # Errors
/// [`VaultError::Aborted`] as described in [`confirm_destructive`];
/// otherwise propagates whatever the store returned (including
/// [`VaultError::SecretNotFound`] if `name` was absent).
fn run_rm(
    name: &str,
    force: bool,
    store: &mut dyn SecretStore,
    io: &mut dyn VaultIo,
) -> Result<(), VaultError> {
    confirm_destructive(
        io,
        force,
        &format!("remove secret '{name}'{CONFIRM_SUFFIX}"),
    )?;
    store.remove(name)?;
    io.println(&format!("secret '{name}' removed"));
    Ok(())
}

/// `passwd`: reads and strength-gates the NEW passphrase (double entry via
/// [`create_passphrase`]), then applies `rekey`. Verifying the CURRENT
/// passphrase already happened when the caller unlocked the envelope to
/// obtain `store`/`rekey` in the first place — this function never sees it.
///
/// # Errors
/// [`VaultError::PassphraseUnavailable`] if there is no TTY (a new passphrase
/// can only be entered interactively — it never comes from `-p`/env/stdin,
/// which carry the CURRENT passphrase and the secret value respectively) or if
/// entry is aborted (an empty line); [`VaultError::WeakPassphrase`] if the new
/// passphrase does not meet the strength floor; [`VaultError::Io`] on a
/// terminal read failure; otherwise propagates whatever `rekey` returned.
fn run_passwd(
    show: bool,
    io: &mut dyn VaultIo,
    rekey: &mut dyn FnMut(&str) -> Result<(), VaultError>,
) -> Result<(), VaultError> {
    // A new passphrase can only be prompted for (double entry); without a TTY
    // there is nowhere to read it from, so fail closed with a clear error
    // rather than surfacing an opaque terminal-read failure (REQ-V40 spirit).
    if !io.is_interactive() {
        return Err(VaultError::PassphraseUnavailable);
    }
    let new_passphrase = create_passphrase(io as &mut dyn PassphrasePrompt, show)?;
    rekey(new_passphrase.as_str())?;
    io.println("passphrase changed");
    Ok(())
}

/// Deliberate `[Y/n]` gate for a destructive operation (REQ-V22).
///
/// `force` skips it outright. Otherwise: without a TTY, a `[Y/n]` cannot be
/// read safely (stdin is the value's channel, or there is no terminal at
/// all), so this fails closed with [`VaultError::Aborted`] rather than
/// hanging or silently consuming the value's input. With a TTY,
/// [`VaultIo::confirm`] decides.
///
/// # Errors
/// [`VaultError::Aborted`] if not forced and either there is no TTY, or the
/// answer was not exactly `"Y"`; [`VaultError::Io`] if the underlying read
/// fails.
fn confirm_destructive(io: &mut dyn VaultIo, force: bool, prompt: &str) -> Result<(), VaultError> {
    if force {
        return Ok(());
    }
    if !io.is_interactive() {
        io.println("destructive operation requires -f in non-interactive mode");
        return Err(VaultError::Aborted);
    }
    if io.confirm(prompt)? {
        Ok(())
    } else {
        Err(VaultError::Aborted)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::vault::{wire, MaskedDek};

    /// Scripted double implementing both [`VaultIo`] and [`PassphrasePrompt`]
    /// (R-V07: these tests never touch a real terminal).
    struct FakeIo {
        interactive: bool,
        confirm_answers: Vec<String>,
        confirm_reads: usize,
        value_answers: Vec<String>,
        value_reads: usize,
        passphrase_answers: Vec<String>,
        passphrase_reads: usize,
        stdin_reader: Option<Box<dyn Read>>,
        output: Vec<String>,
        calls: Vec<String>,
        last_show: Option<bool>,
    }

    impl FakeIo {
        fn interactive(value_answers: Vec<String>) -> Self {
            Self {
                interactive: true,
                confirm_answers: Vec::new(),
                confirm_reads: 0,
                value_answers,
                value_reads: 0,
                passphrase_answers: Vec::new(),
                passphrase_reads: 0,
                stdin_reader: None,
                output: Vec::new(),
                calls: Vec::new(),
                last_show: None,
            }
        }

        fn non_interactive_with_value(value: &str) -> Self {
            let mut io = Self::interactive(vec![value.to_string()]);
            io.interactive = false;
            io
        }

        fn with_stdin_reader(reader: impl Read + 'static) -> Self {
            let mut io = Self::interactive(Vec::new());
            io.interactive = false;
            io.stdin_reader = Some(Box::new(reader));
            io
        }

        fn with_confirm_answers(mut self, answers: Vec<&str>) -> Self {
            self.confirm_answers = answers.into_iter().map(str::to_string).collect();
            self
        }

        fn with_passphrase_answers(mut self, answers: Vec<&str>) -> Self {
            self.passphrase_answers = answers.into_iter().map(str::to_string).collect();
            self
        }
    }

    impl PassphrasePrompt for FakeIo {
        fn is_interactive(&self) -> bool {
            self.interactive
        }

        fn read_passphrase(
            &mut self,
            msg: &str,
            _show: bool,
        ) -> Result<Zeroizing<String>, VaultError> {
            self.calls.push(format!("read_passphrase:{msg}"));
            let idx = self.passphrase_reads;
            self.passphrase_reads += 1;
            Ok(Zeroizing::new(
                self.passphrase_answers
                    .get(idx)
                    .cloned()
                    .unwrap_or_default(),
            ))
        }
    }

    impl VaultIo for FakeIo {
        fn read_value(
            &mut self,
            prompt_msg: &str,
            show: bool,
        ) -> Result<Zeroizing<String>, VaultError> {
            self.calls.push(format!("read_value:{prompt_msg}"));
            self.last_show = Some(if self.interactive { show } else { false });
            if let Some(reader) = self.stdin_reader.take() {
                return read_bounded_value(reader);
            }
            let idx = self.value_reads;
            self.value_reads += 1;
            let value = self.value_answers.get(idx).cloned().unwrap_or_default();
            enforce_value_ceiling(value.len())?;
            Ok(Zeroizing::new(value))
        }

        fn confirm(&mut self, msg: &str) -> Result<bool, VaultError> {
            self.calls.push(format!("confirm:{msg}"));
            let idx = self.confirm_reads;
            self.confirm_reads += 1;
            let raw = self.confirm_answers.get(idx).cloned().unwrap_or_default();
            Ok(is_affirmative(&raw))
        }

        fn println(&mut self, line: &str) {
            self.output.push(line.to_string());
        }
    }

    fn fixture_store() -> impl SecretStore {
        let conn = rusqlite::Connection::open_in_memory().expect("mem db");
        let dek = MaskedDek::new(Zeroizing::new(vec![7u8; 32])).expect("32B dek");
        wire(Arc::new(Mutex::new(conn)), dek).expect("wire")
    }

    fn no_rekey() -> impl FnMut(&str) -> Result<(), VaultError> {
        |_| panic!("rekey should not be called")
    }

    #[test]
    fn test_rm_requires_uppercase_y_exactly() {
        // SC-V22: 'y', '', 'n', 'YES' abort; only "Y" proceeds.
        for (answer, proceeds) in [
            ("y", false),
            ("", false),
            ("n", false),
            ("YES", false),
            ("Y", true),
        ] {
            let mut store = fixture_store();
            store.set("K", "v").expect("seed");
            let mut io = FakeIo::interactive(Vec::new()).with_confirm_answers(vec![answer]);
            let cmd = VaultCmd::Rm {
                name: "K".to_string(),
                force: false,
            };
            let result = run_vault_cmd(cmd, &mut store, &mut io, &mut no_rekey());
            if proceeds {
                assert!(result.is_ok(), "answer {answer:?} should proceed");
                assert!(matches!(store.get("K"), Err(VaultError::SecretNotFound(_))));
            } else {
                assert!(
                    matches!(result, Err(VaultError::Aborted)),
                    "answer {answer:?} should abort"
                );
                assert!(store.get("K").is_ok(), "secret should remain after abort");
            }
        }
    }

    #[test]
    fn test_set_rejects_oversized_stdin_without_buffering_all() {
        // MAGI run 8/12: an unbounded reader by stdin ⇒ ValueTooLarge, and no
        // more than MAX_PLAINTEXT_LEN + 1 bytes are ever buffered (take()).
        let mut io = FakeIo::with_stdin_reader(std::io::repeat(b'a'));
        let r = io.read_value("value: ", false);
        assert!(matches!(r, Err(VaultError::ValueTooLarge(_))));
    }

    #[test]
    fn test_set_rejects_non_utf8_piped_value_instead_of_silently_corrupting() {
        // Loop-1 I-1: a piped secret with invalid UTF-8 bytes must fail with a
        // typed error, NOT be lossily coerced to U+FFFD and stored wrong
        // (matching the strict decode of the read path — REQ-V03).
        let mut io = FakeIo::with_stdin_reader(&[0xff, 0xfe, 0x00, 0x80][..]);
        let r = io.read_value("value: ", false);
        assert!(matches!(r, Err(VaultError::Io(_))));
    }

    #[test]
    fn test_set_on_existing_name_asks_before_reading_the_value() {
        let mut store = fixture_store();
        store.set("K", "old").expect("seed");
        let mut io =
            FakeIo::interactive(vec!["new-value".to_string()]).with_confirm_answers(vec!["Y"]);
        run_vault_cmd(
            VaultCmd::Set {
                name: "K".to_string(),
                show: false,
                force: false,
            },
            &mut store,
            &mut io,
            &mut no_rekey(),
        )
        .expect("set ok");
        let confirm_idx = io
            .calls
            .iter()
            .position(|c| c.starts_with("confirm"))
            .expect("confirm called");
        let read_idx = io
            .calls
            .iter()
            .position(|c| c.starts_with("read_value"))
            .expect("read_value called");
        assert!(
            confirm_idx < read_idx,
            "must confirm BEFORE reading the value"
        );
        assert_eq!(store.get("K").expect("get").as_str(), "new-value");
    }

    #[test]
    fn test_force_skips_confirmation_for_set_and_rm() {
        let mut store = fixture_store();
        store.set("K", "v1").expect("seed");

        let mut io = FakeIo::interactive(vec!["v2".to_string()]);
        run_vault_cmd(
            VaultCmd::Set {
                name: "K".to_string(),
                show: false,
                force: true,
            },
            &mut store,
            &mut io,
            &mut no_rekey(),
        )
        .expect("set ok");
        assert!(!io.calls.iter().any(|c| c.starts_with("confirm")));
        assert_eq!(store.get("K").expect("get").as_str(), "v2");

        let mut io2 = FakeIo::interactive(Vec::new());
        run_vault_cmd(
            VaultCmd::Rm {
                name: "K".to_string(),
                force: true,
            },
            &mut store,
            &mut io2,
            &mut no_rekey(),
        )
        .expect("rm ok");
        assert!(!io2.calls.iter().any(|c| c.starts_with("confirm")));
        assert!(matches!(store.get("K"), Err(VaultError::SecretNotFound(_))));
    }

    #[test]
    fn test_ls_prints_names_and_dates_never_values() {
        let mut store = fixture_store();
        store
            .set("OPENAI_API_KEY", "sk-super-secret-VALUE")
            .expect("seed");
        let mut io = FakeIo::interactive(Vec::new());
        run_vault_cmd(VaultCmd::Ls, &mut store, &mut io, &mut no_rekey()).expect("ls ok");
        assert_eq!(io.output.len(), 1);
        assert!(io.output[0].contains("OPENAI_API_KEY"));
        assert!(!io
            .output
            .iter()
            .any(|l| l.contains("sk-super-secret-VALUE")));

        let mut empty_store = fixture_store();
        let mut io2 = FakeIo::interactive(Vec::new());
        run_vault_cmd(VaultCmd::Ls, &mut empty_store, &mut io2, &mut no_rekey()).expect("ls ok");
        assert_eq!(io2.output, vec![EMPTY_VAULT_LINE.to_string()]);
    }

    #[test]
    fn test_set_show_echoes_input_and_is_ignored_without_tty() {
        // REQ-V21: --show ⇒ read_value(.., show=true) in a TTY; without a
        // TTY the flag is ignored (the value comes from a pipe, not a
        // keystroke echo).
        let mut store = fixture_store();
        let mut io = FakeIo::interactive(vec!["v".to_string()]);
        run_vault_cmd(
            VaultCmd::Set {
                name: "K".to_string(),
                show: true,
                force: false,
            },
            &mut store,
            &mut io,
            &mut no_rekey(),
        )
        .expect("set ok");
        assert_eq!(io.last_show, Some(true));

        let mut store2 = fixture_store();
        let mut io2 = FakeIo::non_interactive_with_value("v2");
        run_vault_cmd(
            VaultCmd::Set {
                name: "K2".to_string(),
                show: true,
                force: true,
            },
            &mut store2,
            &mut io2,
            &mut no_rekey(),
        )
        .expect("set ok");
        assert_eq!(io2.last_show, Some(false));
    }

    #[test]
    fn test_no_subcommand_exposes_a_stored_value() {
        // SC-V05 mechanized: the clap subcommand tree never contains
        // get/cat/show-of-a-stored-value/reveal/export.
        use clap::CommandFactory;

        #[derive(clap::Parser)]
        struct Probe {
            #[command(subcommand)]
            cmd: VaultCmd,
        }

        let names: Vec<String> = Probe::command()
            .get_subcommands()
            .map(|c| c.get_name().to_string())
            .collect();
        for banned in ["get", "cat", "show", "reveal", "export"] {
            assert!(
                !names.iter().any(|n| n == banned),
                "forbidden subcommand: {banned}"
            );
        }
    }

    #[test]
    fn test_passwd_double_entry_and_strength_gate_apply() {
        let mut store = fixture_store();

        // A weak new passphrase ⇒ WeakPassphrase, rekey NOT called.
        let mut io = FakeIo::interactive(Vec::new()).with_passphrase_answers(vec!["short"]);
        let mut rekey_calls = 0usize;
        let mut rekey = |_p: &str| -> Result<(), VaultError> {
            rekey_calls += 1;
            Ok(())
        };
        let result = run_vault_cmd(
            VaultCmd::Passwd { show: false },
            &mut store,
            &mut io,
            &mut rekey,
        );
        assert!(matches!(result, Err(VaultError::WeakPassphrase(_))));
        assert_eq!(rekey_calls, 0);

        // A strong, matching double entry succeeds and rekey receives it.
        let mut io2 = FakeIo::interactive(Vec::new()).with_passphrase_answers(vec![
            "correct horse battery staple",
            "correct horse battery staple",
        ]);
        let mut received: Option<String> = None;
        let mut rekey2 = |p: &str| -> Result<(), VaultError> {
            received = Some(p.to_string());
            Ok(())
        };
        run_vault_cmd(
            VaultCmd::Passwd { show: false },
            &mut store,
            &mut io2,
            &mut rekey2,
        )
        .expect("passwd ok");
        assert_eq!(received.as_deref(), Some("correct horse battery staple"));
    }

    #[test]
    fn test_passwd_without_tty_fails_closed_without_rekey() {
        // Loop-1 M-4: a new passphrase can only be entered interactively; with
        // no TTY, passwd fails with PassphraseUnavailable and never rekeys,
        // instead of surfacing an opaque terminal-read error.
        let mut store = fixture_store();
        let mut io = FakeIo::non_interactive_with_value("ignored");
        let result = run_vault_cmd(
            VaultCmd::Passwd { show: false },
            &mut store,
            &mut io,
            &mut no_rekey(),
        );
        assert!(matches!(result, Err(VaultError::PassphraseUnavailable)));
    }

    #[test]
    fn test_non_tty_destructive_without_force_aborts_without_reading_confirm() {
        let mut store = fixture_store();
        store.set("K", "v").expect("seed");
        let mut io = FakeIo::non_interactive_with_value("ignored");
        let result = run_vault_cmd(
            VaultCmd::Rm {
                name: "K".to_string(),
                force: false,
            },
            &mut store,
            &mut io,
            &mut no_rekey(),
        );
        assert!(matches!(result, Err(VaultError::Aborted)));
        assert!(
            !io.calls.iter().any(|c| c.starts_with("confirm")),
            "must never attempt confirm() without a TTY"
        );
        assert!(store.get("K").is_ok(), "secret must survive an aborted rm");
    }
}
