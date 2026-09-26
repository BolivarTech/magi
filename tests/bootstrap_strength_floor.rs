// Author: Julian Bolivar
// Version: 0.19.1
// Date: 2026-09-24

//! End-to-end guardian of the passphrase strength floor on an envelope-less database.
//!
//! `magi init` run without a passphrase scaffolds `.magi/` and leaves the database with no
//! envelope, so the first open creates it. The floor (REQ-V17: `zxcvbn` score >= 3 and at
//! least 12 characters, no override) used to be checked only when the database FILE was
//! absent, so `-p abc vault ls` on such a workspace made "abc" the master secret. This file
//! reproduces that sequence against the real binary, in a temp directory only.
//!
//! The precondition does not use the mechanism under test: the workspace is scaffolded by
//! `init` with no passphrase at all, which never reaches the envelope bootstrap, and the
//! envelope-less state is read straight from SQLite rather than through the binary.

use std::path::Path;
use std::process::{Command, Stdio};

/// A passphrase well below the floor.
const WEAK_PASSPHRASE: &str = "abc";

/// A passphrase that clears the floor. Same value `tests/workdir_flag.rs` uses.
const STRONG_PASSPHRASE: &str = "correct horse battery staple";

/// The database file `magi init` creates under `.magi/`.
const DB_RELATIVE_PATH: &str = ".magi/.magi-rs-memory.db";

/// Builds a `magi-rs` invocation in `cwd` with no inherited passphrase and no terminal on
/// stdin, so nothing can be answered interactively.
fn magi_in(cwd: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_magi-rs"));
    c.current_dir(cwd);
    c.env_remove("MAGI_PASSPHRASE");
    c.stdin(Stdio::null());
    c
}

/// Runs `cmd`, returning `(exit code, stdout, stderr)` as lossy UTF-8.
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

/// Number of rows in `vault_meta` — zero means the database has no envelope.
fn vault_meta_rows(db: &Path) -> i64 {
    let conn = rusqlite::Connection::open(db).expect("open the scaffolded database");
    conn.query_row("SELECT COUNT(*) FROM vault_meta", [], |r| r.get(0))
        .expect("vault_meta exists after init")
}

#[test]
fn a_weak_passphrase_cannot_create_the_envelope_of_an_initialized_workspace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path();
    let db = cwd.join(DB_RELATIVE_PATH);

    let (code, _out, err) = run(magi_in(cwd).arg("init"));
    assert_eq!(
        code, 0,
        "init without a passphrase must succeed; stderr: {err}"
    );
    assert_eq!(
        vault_meta_rows(&db),
        0,
        "precondition: init left no envelope"
    );

    let (code, out, err) = run(magi_in(cwd).args(["-p", WEAK_PASSPHRASE, "vault", "ls"]));
    assert_ne!(
        code, 0,
        "a weak passphrase must not unlock (and so create) the vault; stdout: {out}"
    );
    assert!(
        err.contains("passphrase rejected:"),
        "the refusal must be the strength-floor error; stderr: {err}"
    );
    assert!(
        !err.contains(WEAK_PASSPHRASE) && !out.contains(WEAK_PASSPHRASE),
        "the passphrase must never be echoed"
    );
    assert_eq!(
        vault_meta_rows(&db),
        0,
        "a refused bootstrap leaves the database envelope-less"
    );

    let (code, out, err) = run(magi_in(cwd).args(["-p", STRONG_PASSPHRASE, "vault", "ls"]));
    assert_eq!(
        code, 0,
        "a strong passphrase still bootstraps the envelope; stdout: {out}; stderr: {err}"
    );
    assert_eq!(
        vault_meta_rows(&db),
        2,
        "salt and wrapped_dek are installed"
    );
}
