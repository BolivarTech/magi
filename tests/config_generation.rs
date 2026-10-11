// Author: Julian Bolivar
// Version: 0.23.0
// Date: 2026-10-11

//! S6/S19 end to end: the generation check runs before any other startup work, so an old or empty
//! `magi.toml` is refused before the encrypted database is opened or a model is called.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Exit code for invalid input (`headless_error_exit_code`).
const EXIT_MISUSE: i32 = 2;

/// The message `HeadlessError::PassphraseUnavailable` prints. Seeing it means the run went on to the
/// vault instead of stopping at the generation check.
const PASSPHRASE_UNAVAILABLE: &str = "no passphrase: use -p or MAGI_PASSPHRASE";

fn magi(cwd: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_magi-rs"));
    c.current_dir(cwd);
    c.env_remove("MAGI_PASSPHRASE");
    c.env_remove("DECISION_API_KEY");
    c
}

fn run(cmd: &mut Command) -> (i32, String) {
    let out = cmd.output().expect("spawn magi-rs");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A workspace scaffolded by the current binary whose `magi.toml` is then replaced by `toml_text`.
fn workspace_with(toml_text: &str) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = dunce::canonicalize(tmp.path()).expect("canonical");
    let (code, err) = run(magi(&root).arg("init"));
    assert_eq!(code, 0, "init: {err}");
    std::fs::write(root.join(".magi").join("magi.toml"), toml_text).expect("write toml");
    (tmp, root)
}

#[test]
fn query_consult_and_vault_refuse_an_old_toml_before_touching_the_vault() {
    let (_tmp, ws) = workspace_with("provider = \"ollama\"\n");
    for args in [
        vec!["query"],
        vec!["consult"],
        vec!["vault", "ls"],
        vec!["vault", "set", "X"],
        vec!["vault", "rm", "X"],
        vec!["vault", "passwd"],
    ] {
        let (code, err) = run(magi(&ws).args(&args));
        assert_eq!(code, EXIT_MISUSE, "{args:?}: {err}");
        assert!(
            err.contains("[decision]") && err.contains("magi-rs init"),
            "{args:?}: {err}"
        );
        assert!(
            !err.contains(PASSPHRASE_UNAVAILABLE),
            "{args:?} reached the vault first: {err}"
        );
    }
}

#[test]
fn query_with_an_explicit_workdir_refuses_an_old_toml_there() {
    let (_tmp, ws) = workspace_with("provider = \"ollama\"\n");
    let elsewhere = tempfile::tempdir().expect("tempdir");
    let (code, err) = run(magi(elsewhere.path()).args(["query", "-w"]).arg(&ws));
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(err.contains("[decision]"), "{err}");
}

/// CP2 MS1 r1 (Melchior): the two entry points that reach the DB without `prepare_headless`.
#[test]
fn logout_and_the_tui_refuse_an_old_toml_before_touching_the_vault() {
    let (_tmp, ws) = workspace_with("provider = \"ollama\"\n");
    for args in [vec!["--logout"], vec![]] {
        let (code, err) = run(magi(&ws).args(&args));
        assert_eq!(code, EXIT_MISUSE, "{args:?}: {err}");
        assert!(err.contains("[decision]"), "{args:?}: {err}");
        assert!(
            !err.contains(PASSPHRASE_UNAVAILABLE),
            "{args:?} reached the vault first: {err}"
        );
    }
}

/// CP2 MS1 r3 (Caspar): help and version never hit the generation check.
#[test]
fn help_and_version_work_with_an_old_toml() {
    let (_tmp, ws) = workspace_with("provider = \"ollama\"\n");
    for flag in ["--help", "--version"] {
        let out = magi(&ws).arg(flag).output().expect("spawn");
        assert!(
            out.status.success(),
            "{flag}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// CP2 MS1 r1 (Caspar): D-P3 holds in the preflight too: a literal credential in an old file is reported
/// before the missing section, and its value is never repeated.
#[test]
fn a_literal_credential_is_reported_before_the_missing_section() {
    let (_tmp, ws) = workspace_with("base_url = \"https://alice:s3cr3t@host/v1\"\n");
    let (code, err) = run(magi(&ws).args(["vault", "ls"]));
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("BASE_URL_USER"),
        "the credential error comes first: {err}"
    );
    assert!(
        !err.contains("[decision] section"),
        "the upgrade message does not mask it: {err}"
    );
    assert!(!err.contains("s3cr3t"), "{err}");
}

/// CP2 MS1 r1 (Melchior): the remedy the messages prescribe works: what `init` writes passes the check.
/// Passes in Red by design once T5 Green is in (declared in §0.3); its red is the mutation "scaffold
/// without [decision]".
#[test]
fn what_init_writes_passes_the_generation_check() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = dunce::canonicalize(tmp.path()).expect("canonical");
    let (code, err) = run(magi(&root).arg("init"));
    assert_eq!(code, 0, "init: {err}");
    let (_, err) = run(magi(&root).args(["vault", "ls"]));
    assert!(
        err.contains(PASSPHRASE_UNAVAILABLE),
        "a fresh workspace goes on to the vault: {err}"
    );
}

#[test]
fn an_empty_toml_is_refused_with_both_ways_out() {
    let (_tmp, ws) = workspace_with("# nothing here\n");
    let (code, err) = run(magi(&ws).args(["vault", "ls"]));
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("Delete") && err.contains("magi-rs init"),
        "{err}"
    );
}
