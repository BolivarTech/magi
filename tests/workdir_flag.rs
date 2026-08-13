// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-12

//! End-to-end guardian of `-w`/`--workdir` on the `init` and `vault` subcommands.
//!
//! # Why this file spawns the real binary
//!
//! The unit tests in `src/main.rs` prove two separate things: that clap *accepts* the flag,
//! and that `resolve_workspace_root` *prefers* it over the current directory. Neither proves
//! the one property that matters — that `run()` actually calls the resolver instead of
//! reading [`std::env::current_dir`] the way it did before. A flag that parses, resolves
//! correctly, and is then quietly ignored passes both unit tests and ships broken.
//!
//! That failure mode is not hypothetical in this codebase: MS2's `AutonomousRunConfig` was a
//! configuration bundle that existed, was threaded past every call site, and was never
//! applied — an operator writing `untrusted_content = true` got no protection at all, and
//! every gate was green. The fix there was a type that cannot be built without the config;
//! the equivalent here is a test that observes the **filesystem**, from outside the process,
//! with a current directory deliberately different from the `-w` target.
//!
//! Each test therefore asserts on both sides: the directory that should have been touched,
//! and the one that must not have been.

use std::path::Path;
use std::process::Command;

/// A passphrase that clears the `check_strength` floor (zxcvbn ≥ 3, ≥ 12 chars) so the
/// `vault` cases run non-interactively. Same value the `src/main.rs` unit tests use.
const TEST_PASSPHRASE: &str = "correct horse battery staple";

/// Builds a `magi-rs` invocation whose current directory is `cwd`.
///
/// The current directory is always set explicitly, never inherited: every assertion in this
/// file depends on `cwd` and the `-w` target being two different, known places.
fn magi_in(cwd: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_magi-rs"));
    c.current_dir(cwd);
    // A passphrase in the environment would silently satisfy prompts these tests never
    // intend to answer; an inherited one from the developer's shell would make the result
    // depend on who ran it.
    c.env_remove("MAGI_PASSPHRASE");
    c
}

/// Runs `cmd`, returning `(exit code, stdout, stderr)` with both streams as lossy UTF-8.
fn run(cmd: &mut Command) -> (i32, String, String) {
    let out = cmd
        .output()
        .expect("spawning the magi-rs binary must succeed");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn init_scaffolds_into_the_named_workdir_and_not_the_current_directory() {
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let target = tempfile::tempdir().expect("target tempdir");

    let (code, stdout, stderr) = run(magi_in(cwd.path()).args(["init", "-w"]).arg(target.path()));

    assert_eq!(code, 0, "init -w must succeed; stderr: {stderr}");
    assert!(
        target.path().join(".magi").is_dir(),
        "the .magi/ must be created under the -w target, not elsewhere; stdout: {stdout}"
    );
    assert!(
        !cwd.path().join(".magi").exists(),
        "the current directory must be left untouched — a `.magi/` here means the flag \
         parsed and was then ignored, which is the exact defect this test exists for"
    );
    assert!(
        stdout.contains(&target.path().join(".magi").display().to_string()),
        "stdout must name the directory that was actually created (got: {stdout:?})"
    );
}

#[test]
fn init_on_a_missing_workdir_fails_naming_the_directory() {
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let missing = cwd.path().join("no-such-directory");

    let (code, _stdout, stderr) = run(magi_in(cwd.path()).args(["init", "-w"]).arg(&missing));

    assert_ne!(
        code, 0,
        "a -w that does not exist must fail, not be created"
    );
    assert!(
        stderr.contains("no-such-directory"),
        "the error must name the offending directory rather than surfacing a bare I/O \
         message from deep inside the scaffolder (got: {stderr:?})"
    );
    assert!(
        !missing.exists(),
        "a missing -w must never be created: `init` refuses to nest and refuses to \
         overwrite, and silently building a directory tree does not belong with that"
    );
}

#[test]
fn init_on_a_workdir_that_is_a_file_fails_instead_of_reporting_no_workspace() {
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let file = cwd.path().join("a-file.txt");
    std::fs::write(&file, b"not a directory").expect("fixture file");

    let (code, _stdout, stderr) = run(magi_in(cwd.path()).args(["init", "-w"]).arg(&file));

    assert_ne!(code, 0, "a -w pointing at a file must fail");
    assert!(
        stderr.contains("a-file.txt"),
        "the error must name the path (got: {stderr:?})"
    );
}

#[test]
fn vault_ls_resolves_the_workspace_from_the_named_workdir() {
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let target = tempfile::tempdir().expect("target tempdir");

    let (init_code, _, init_err) = run(magi_in(cwd.path())
        .args(["-p", TEST_PASSPHRASE, "init", "-w"])
        .arg(target.path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );

    // Without `-w` this same invocation fails: `cwd` has no `.magi/` and the walk-up finds
    // nothing. Asserting the negative first is what makes the positive meaningful — it rules
    // out a pass that comes from some unrelated workspace being discovered.
    let (without, _, _) = run(magi_in(cwd.path()).args(["-p", TEST_PASSPHRASE, "vault", "ls"]));
    assert_ne!(
        without, 0,
        "precondition: no .magi/ under the current directory"
    );

    let (with, _stdout, stderr) = run(magi_in(cwd.path())
        .args(["-p", TEST_PASSPHRASE, "vault", "ls", "-w"])
        .arg(target.path()));
    assert_eq!(
        with, 0,
        "vault must resolve its workspace from -w; stderr: {stderr}"
    );
}

#[test]
fn vault_accepts_the_workdir_before_its_own_subcommand_too() {
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let target = tempfile::tempdir().expect("target tempdir");

    let (init_code, _, init_err) = run(magi_in(cwd.path())
        .args(["-p", TEST_PASSPHRASE, "init", "-w"])
        .arg(target.path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );
    // Without this the test passes for the wrong reason: if `-w` were ignored, `init` would
    // have scaffolded into `cwd`, and the `vault` call below would then succeed by finding
    // that workspace under its own current directory — never consulting the flag at all.
    assert!(
        !cwd.path().join(".magi").exists(),
        "precondition: the workspace must exist only under the -w target"
    );

    // `vault -w <dir> ls` and `vault ls -w <dir>` must both work. Only one of the two orders
    // is natural to type, and which one that is depends on the person; accepting a single
    // order turns a coin flip into a usage error.
    let (code, _stdout, stderr) = run(magi_in(cwd.path())
        .args(["-p", TEST_PASSPHRASE, "vault", "-w"])
        .arg(target.path())
        .arg("ls"));

    assert_eq!(
        code, 0,
        "the flag must parse ahead of the vault subcommand too; stderr: {stderr}"
    );
}
