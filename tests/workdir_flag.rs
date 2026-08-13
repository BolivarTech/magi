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

/// The state directory every assertion looks for.
const MAGI_DIR: &str = ".magi";

/// Exit code for operator misuse — a mistyped flag value, an unparseable argument.
///
/// `headless_error_exit_code` splits input/misuse (`2`) from every other failure class (`1`),
/// so a caller can tell "you invoked me wrong" from "the run failed". A `-w` that names
/// nothing is the first kind.
const EXIT_MISUSE: i32 = 2;

/// A temp directory whose path is **canonical**.
///
/// Not cosmetic. `tempfile::tempdir()` on macOS returns a path under `/var/folders/…`, and
/// `/var` is an OS symlink — passed to `-w` unresolved it trips
/// `ensure_raw_chain_symlink_free` and the test fails for a reason that has nothing to do with
/// what it asserts. The pre-existing tests never hit this because they hand the path to
/// `current_dir`, where `getcwd` resolves it for them; these tests hand it to `-w` instead, so
/// they have to resolve it themselves. CI runs Linux and Windows today, so this is
/// future-proofing rather than a fix — which is why it is worth a comment and not just a call.
fn tempdir_canonical() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dunce::canonicalize(dir.path()).expect("canonicalize tempdir");
    (dir, path)
}

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

/// Creates a workspace **in** `dir` without going through `-w`.
///
/// The distinction is the whole reason this helper exists. Seeding with `init -w <dir>` makes
/// a test's setup depend on the very mechanism the test evaluates: with `-w` ignored, the
/// `.magi/` lands in the current directory instead, the assertion then resolves that same
/// current directory, finds the workspace there, and the test passes while measuring nothing.
/// A mutation run proved it — the first version of `vault_diagnose_also_honors_the_workdir`
/// stayed green with the flag disabled, and no amount of strengthening its assertions fixed
/// that, because setup and assertion moved together.
///
/// Running `init` with `dir` as the process's current directory removes the coupling: the
/// workspace exists at a known place no matter what `-w` does.
fn seed_workspace(dir: &Path) -> Command {
    let mut c = magi_in(dir);
    c.args(["-p", TEST_PASSPHRASE, "init"]);
    c
}

#[test]
fn init_scaffolds_into_the_named_workdir_and_not_the_current_directory() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_target_guard, target) = tempdir_canonical();

    let (code, stdout, stderr) = run(magi_in(cwd.as_path())
        .args(["init", "-w"])
        .arg(target.as_path()));

    assert_eq!(code, 0, "init -w must succeed; stderr: {stderr}");
    assert!(
        target.as_path().join(MAGI_DIR).is_dir(),
        "the .magi/ must be created under the -w target, not elsewhere; stdout: {stdout}"
    );
    assert!(
        !cwd.as_path().join(MAGI_DIR).exists(),
        "the current directory must be left untouched — a `.magi/` here means the flag \
         parsed and was then ignored, which is the exact defect this test exists for"
    );
    assert!(
        stdout.contains(&target.as_path().join(MAGI_DIR).display().to_string()),
        "stdout must name the directory that was actually created (got: {stdout:?})"
    );
}

#[test]
fn init_on_a_missing_workdir_fails_naming_the_directory() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let missing = cwd.as_path().join("no-such-directory");

    let (code, _stdout, stderr) = run(magi_in(cwd.as_path()).args(["init", "-w"]).arg(&missing));

    assert_eq!(
        code, EXIT_MISUSE,
        "a mistyped -w is operator misuse (2), not a runtime failure (1) — a caller has to \
         be able to tell them apart, and the sibling symlink rejection already returns 2"
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
    let (_cwd_guard, cwd) = tempdir_canonical();
    let file = cwd.as_path().join("a-file.txt");
    std::fs::write(&file, b"not a directory").expect("fixture file");

    let (code, _stdout, stderr) = run(magi_in(cwd.as_path()).args(["init", "-w"]).arg(&file));

    assert_eq!(
        code, EXIT_MISUSE,
        "a -w pointing at a file is the same class of misuse as one naming nothing"
    );
    assert!(
        stderr.contains("a-file.txt"),
        "the error must name the path (got: {stderr:?})"
    );
}

#[test]
fn vault_ls_resolves_the_workspace_from_the_named_workdir() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_target_guard, target) = tempdir_canonical();

    let (init_code, _, init_err) = run(&mut seed_workspace(target.as_path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );

    // Without `-w` this same invocation fails: `cwd` has no `.magi/` and the walk-up finds
    // nothing. Asserting the negative first is what makes the positive meaningful — it rules
    // out a pass that comes from some unrelated workspace being discovered.
    let (without, _, _) = run(magi_in(cwd.as_path()).args(["-p", TEST_PASSPHRASE, "vault", "ls"]));
    assert_ne!(
        without, 0,
        "precondition: no .magi/ under the current directory"
    );

    let (with, _stdout, stderr) = run(magi_in(cwd.as_path())
        .args(["-p", TEST_PASSPHRASE, "vault", "ls", "-w"])
        .arg(target.as_path()));
    assert_eq!(
        with, 0,
        "vault must resolve its workspace from -w; stderr: {stderr}"
    );
}

#[test]
fn vault_accepts_the_workdir_before_its_own_subcommand_too() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_target_guard, target) = tempdir_canonical();

    let (init_code, _, init_err) = run(&mut seed_workspace(target.as_path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );
    // Without this the test passes for the wrong reason: if `-w` were ignored, `init` would
    // have scaffolded into `cwd`, and the `vault` call below would then succeed by finding
    // that workspace under its own current directory — never consulting the flag at all.
    assert!(
        !cwd.as_path().join(MAGI_DIR).exists(),
        "precondition: the workspace must exist only under the -w target"
    );

    // `vault -w <dir> ls` and `vault ls -w <dir>` must both work. Only one of the two orders
    // is natural to type, and which one that is depends on the person; accepting a single
    // order turns a coin flip into a usage error.
    let (code, _stdout, stderr) = run(magi_in(cwd.as_path())
        .args(["-p", TEST_PASSPHRASE, "vault", "-w"])
        .arg(target.as_path())
        .arg("ls"));

    assert_eq!(
        code, 0,
        "the flag must parse ahead of the vault subcommand too; stderr: {stderr}"
    );
}

#[test]
fn vault_diagnose_also_honors_the_workdir() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_target_guard, target) = tempdir_canonical();

    let (init_code, _, init_err) = run(&mut seed_workspace(target.as_path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );

    // `diagnose` is the one vault arm with its own dispatch branch, intercepted before a
    // passphrase is ever resolved (REQ-H32) so a structural probe never unlocks anything. It
    // reads the same resolved root as the others, which is a property of where the resolution
    // happens rather than of anything `diagnose` does — and that is exactly the kind of thing
    // a later refactor breaks silently, since nothing about the early return mentions `-w`.
    let (code, stdout, stderr) = run(magi_in(cwd.as_path())
        .args(["vault", "diagnose", "-w"])
        .arg(target.as_path()));

    assert_eq!(
        code, 0,
        "vault diagnose must resolve -w too; stderr: {stderr}"
    );
    // The exit code alone proves NOTHING here, and asserting only on it is how the first
    // version of this test came to guard nothing: `run_vault_diagnose` treats an absent DB as
    // a report, not an error, and returns 0 after printing "no database found at …". So with
    // `-w` ignored it would resolve the empty current directory, find no database, print that
    // line, exit 0 — and a code-only assertion would pass. Both halves below are needed: the
    // positive says it read the target's DB, the negative says it did not fall through.
    assert!(
        stdout.contains("envelope:"),
        "diagnose must have read the workspace under -w (got: {stdout:?})"
    );
    assert!(
        !stdout.contains("no database found"),
        "diagnose resolved somewhere other than -w — this is the exact pass-for-the-wrong-\
         reason the assertion above exists to rule out (got: {stdout:?})"
    );
}

#[test]
fn vault_takes_the_innermost_workdir_when_given_twice() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_outer_guard, outer) = tempdir_canonical();
    let (_inner_guard, inner) = tempdir_canonical();

    // Only `inner` gets a workspace, so "which one won" is observable rather than inferred.
    let (init_code, _, init_err) = run(&mut seed_workspace(inner.as_path()));
    assert_eq!(
        init_code, 0,
        "precondition: init -w must succeed ({init_err})"
    );

    // `-w` before the nested subcommand and again after it: the innermost occurrence wins.
    // This is clap's own semantics for a `global` arg, and it is the convention every CLI
    // that carries one already follows (`git -C`, `docker`, `kubectl`) — so it is DECIDED
    // here rather than merely inherited, and pinned so it cannot drift into the opposite.
    //
    // It is a deliberate asymmetry with `init`, whose `-w` is not global and which clap
    // therefore rejects outright when given twice. Erring the other way — rejecting the
    // double here — would need the parse restructured away from the derive API to see both
    // occurrences at all, in exchange for being stricter than the ecosystem norm on a
    // mistake nobody has made yet.
    let (code, _stdout, stderr) = run(magi_in(cwd.as_path())
        .args(["-p", TEST_PASSPHRASE, "vault", "-w"])
        .arg(outer.as_path())
        .arg("ls")
        .arg("-w")
        .arg(inner.as_path()));

    assert_eq!(
        code, 0,
        "the innermost -w must win, and it is the one with the workspace; stderr: {stderr}"
    );
}

#[test]
fn workdir_is_not_accepted_ahead_of_the_vault_subcommand_itself() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let (_target_guard, target) = tempdir_canonical();

    // `-p` IS global on the top-level command, so `magi -p … vault ls` works and a reader
    // reasonably expects `-w` to behave the same way. It does not: `-w` is declared on the
    // `vault` subcommand, not on the root, so `magi -w <dir> vault ls` is a usage error.
    //
    // Pinned as a NEGATIVE on purpose. The tempting "fix" is to promote `-w` to a top-level
    // global for symmetry with `-p`, and that collides with the `-w` already owned by
    // `query`/`consult` — a panic while clap builds the command under `debug_assertions`,
    // which reads as "the binary stopped starting" rather than as a parse error. This test
    // turns that into a red line instead of a discovery.
    let (code, _stdout, _stderr) = run(magi_in(cwd.as_path())
        .args(["-w"])
        .arg(target.as_path())
        .args(["-p", TEST_PASSPHRASE, "vault", "ls"]));

    assert_eq!(
        code, EXIT_MISUSE,
        "-w before the subcommand must be a usage error, not silently accepted"
    );
}

#[cfg(unix)]
#[test]
fn a_symlinked_workdir_is_rejected_and_creates_nothing() {
    let (_cwd_guard, cwd) = tempdir_canonical();
    let real = cwd.as_path().join("real");
    std::fs::create_dir_all(&real).expect("fixture dir");
    let link = cwd.as_path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("fixture symlink");

    // `is_dir()` follows symlinks, so the resolver's own check passes and the rejection comes
    // from `ensure_raw_chain_symlink_free`, which tests EVERY component of the absolute path
    // including the last one. Worth pinning because it is a real behavior difference from the
    // current directory: `getcwd` hands back the resolved physical path, so reaching the same
    // directory with `cd` never triggered this, while naming it with `-w` does.
    let (code, _stdout, stderr) = run(magi_in(cwd.as_path()).args(["init", "-w"]).arg(&link));

    assert_eq!(
        code, EXIT_MISUSE,
        "a symlinked -w must be rejected; stderr: {stderr}"
    );
    assert!(
        !real.join(MAGI_DIR).exists(),
        "nothing may be created through the rejected symlink"
    );
}
