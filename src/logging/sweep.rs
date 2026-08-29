// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Executing what `retention` decided, and sweeping what a crash left behind.
//!
//! `retention::plan` says *what* should happen to each file; this module makes
//! it happen. The split is what lets "on day 8 it compresses and on day 31 it is
//! deleted" be tested with two dates instead of thirty-one days of real files.
//!
//! # Everything here is best effort
//!
//! A delete that fails — Windows with the file still open — does not abort the
//! run. The file stays a candidate for next time. Retention exists to bound disk
//! use, and failing to bound it this minute is not worth taking the process down.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::logging::retention::Action;
use crate::logging::xz::{compress_verified, compressed_path};
use crate::logging::LoggingError;

/// How often a live compression touches its temporary, so the sweep can tell it
/// apart from an orphan.
pub const TMP_HEARTBEAT_SECS: u64 = 900;
/// How stale a temporary must be before the sweep removes it.
pub const TMP_GRACE_SECS: u64 = 3600;

/// **The relation is the real invariant, and it is asserted at COMPILE time.**
///
/// Dropping the grace to 600 would make every in-flight compression deletable,
/// silently, with no test failing. A `debug_assert!` would not do: `release`
/// removes it, so the precondition that matters in production would be guarded
/// only in `debug` — the trap this repository already documents elsewhere. A
/// `const` assertion costs nothing at runtime and breaks the build instead of a
/// test, which is the earliest anyone can find out.
const _: () = assert!(TMP_HEARTBEAT_SECS < TMP_GRACE_SECS);

/// Compressions started per process launch.
///
/// Two months unused leaves sixty files pending, and without a cap the first
/// launch fires sixty LZMA2 runs before the agent answers anything.
pub const MAX_COMPRESSIONS_PER_START: usize = 5;

/// Extension of a staged compression.
const TMP_EXTENSION: &str = "tmp";

/// Applies `actions` to `files`, in order, best effort.
///
/// # Parameters
///
/// * `dir` — the log directory.
/// * `names` — file names, in the same order the plan was computed over.
/// * `actions` — one per name.
///
/// # Returns
///
/// How many files were compressed. Deletions are not counted because nothing
/// downstream bounds them.
///
/// # Errors
///
/// Never. Individual failures are absorbed; the signature returns `Result` so a
/// caller that wants to react to a systemic failure can, and today none does.
///
/// # Complexity
///
/// `O(n)` over the files, plus the cost of each compression.
pub fn execute_plan(
    dir: &Path,
    names: &[String],
    actions: &[Action],
) -> Result<usize, LoggingError> {
    let mut compressed = 0usize;
    for (name, action) in names.iter().zip(actions.iter()) {
        let path = dir.join(name);
        match action {
            Action::Keep => {}
            Action::Compress => {
                if compressed >= MAX_COMPRESSIONS_PER_START {
                    continue;
                }
                let staged = staging_path(dir, name);
                if compress_verified(&path, &staged).is_ok() {
                    compressed += 1;
                    // The original goes only once the .xz is proven good.
                    let _ = fs::remove_file(&path);
                } else {
                    let _ = fs::remove_file(&staged);
                }
            }
            Action::Delete => {
                // Best effort (REQ-L17): a locked file stays a candidate.
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(compressed_path(&path));
            }
        }
    }
    Ok(compressed)
}

/// Where a compression stages its output.
///
/// Same directory as the source, so the final rename is atomic.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn staging_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{TMP_EXTENSION}"))
}

/// Removes staged files a crash left behind.
///
/// # Why this exists at all
///
/// Without it every crash during a compression leaves a `.tmp` that **no
/// retention rule recognises** — it does not match `magi-<date>.log` — and the
/// directory grows without bound.
///
/// # What distinguishes an orphan from live work
///
/// Its `mtime`. A running compression touches its temporary every
/// [`TMP_HEARTBEAT_SECS`]; anything untouched for [`TMP_GRACE_SECS`] is
/// abandoned. **The mtime alone carries correctness** — a PID check would be an
/// optimisation, and on a recycled PID a wrong one.
///
/// # Returns
///
/// How many temporaries were removed.
///
/// # Errors
///
/// Never: an unreadable directory yields zero, because failing to tidy up is not
/// worth failing a startup over.
///
/// # Complexity
///
/// `O(n)` over the directory.
pub fn sweep_orphan_temps(dir: &Path, now: SystemTime) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let grace = Duration::from_secs(TMP_GRACE_SECS);
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(TMP_EXTENSION) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        // A future mtime means a clock problem, not an orphan: fail toward
        // keeping the file, the same direction retention's skew guard takes.
        let abandoned = now
            .duration_since(mtime)
            .map(|elapsed| elapsed > grace)
            .unwrap_or(false);
        if abandoned && fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Reads the log directory into the entries `retention::plan` decides over.
///
/// The date comes from the file NAME, which is the contract `rotation::file_name`
/// writes. A name that does not carry one yields `None`, and retention treats
/// that as older than every real date — the same direction it takes a future
/// date, and for the same reason: an unrecognised file must not become immortal.
///
/// # Returns
///
/// One entry per `.log` or `.xz` file, in directory order.
///
/// # Complexity
///
/// `O(n)` over the directory.
#[must_use]
pub fn scan(dir: &Path) -> Vec<crate::logging::retention::FileEntry> {
    use crate::logging::retention::FileEntry;
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_log = name.ends_with(".log") || name.ends_with(".log.xz");
        if !is_log {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        out.push(FileEntry {
            date: date_in(&name),
            mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            size: meta.len(),
            name,
        });
    }
    out
}

/// Parses the `YYYY-MM-DD` a log file's name carries.
///
/// # Complexity
///
/// `O(1)`.
fn date_in(name: &str) -> Option<time::Date> {
    let rest = name.strip_prefix("magi-")?;
    let (y, rest) = rest.split_once('-')?;
    let (m, rest) = rest.split_once('-')?;
    let d = rest.get(..2)?;
    time::Date::from_calendar_date(
        y.parse().ok()?,
        time::Month::try_from(m.parse::<u8>().ok()?).ok()?,
        d.parse().ok()?,
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, body).expect("fixture");
        p
    }

    /// Sets a file's mtime `age` into the past.
    fn age(path: &Path, age: Duration) {
        let when = SystemTime::now() - age;
        let f = fs::File::options().write(true).open(path).expect("open");
        f.set_modified(when).expect("set mtime");
    }

    // The plan asked for a RUNTIME test of `TMP_HEARTBEAT_SECS < TMP_GRACE_SECS`
    // alongside the `const` assertion, to document the reason where someone
    // reads it. It is not here, and the reason is that clippy is right:
    // `constant value` fires, because the assertion IS constant — the `const`
    // above already proved it at compile time, and a second copy asserts nothing
    // a build could reach. Keeping it would take an `#[allow]` fabricated purely
    // to quiet a lint, which this project forbids. The explanation lives on the
    // `const` instead, which is where the guarantee is.

    #[test]
    fn an_orphan_temp_is_swept_but_a_live_one_is_not() {
        let dir = tempdir().unwrap();
        let orphan = write(dir.path(), "old.tmp", b"abandoned");
        let live = write(dir.path(), "new.tmp", b"in flight");
        age(&orphan, Duration::from_secs(TMP_GRACE_SECS + 60));

        let removed = sweep_orphan_temps(dir.path(), SystemTime::now());

        assert_eq!(removed, 1, "exactly the abandoned one");
        assert!(!orphan.exists());
        assert!(
            live.exists(),
            "mtime alone carries correctness; the PID would be an optimisation"
        );
    }

    #[test]
    fn the_sweep_leaves_everything_that_is_not_a_temporary_alone() {
        let dir = tempdir().unwrap();
        let log = write(dir.path(), "magi-2026-01-01.log", b"old but not a temp");
        age(&log, Duration::from_secs(TMP_GRACE_SECS * 10));
        assert_eq!(sweep_orphan_temps(dir.path(), SystemTime::now()), 0);
        assert!(log.exists(), "retention decides about logs, not this sweep");
    }

    #[test]
    fn the_startup_compression_burst_is_capped() {
        let dir = tempdir().unwrap();
        let mut names = Vec::new();
        for d in 1..=(MAX_COMPRESSIONS_PER_START + 3) {
            let name = format!("magi-2026-06-{d:02}.log");
            write(
                dir.path(),
                &name,
                b"compress me, i am repetitive\n".repeat(50).as_slice(),
            );
            names.push(name);
        }
        let actions = vec![Action::Compress; names.len()];

        let done = execute_plan(dir.path(), &names, &actions).unwrap();

        assert_eq!(
            done,
            MAX_COMPRESSIONS_PER_START,
            "{} pending must not fire {} LZMA2 runs at startup",
            names.len(),
            names.len()
        );
        assert!(
            names.len() > MAX_COMPRESSIONS_PER_START,
            "the fixture must exceed the cap or the assertion above is free"
        );
    }

    #[test]
    fn a_blocked_delete_does_not_abort_the_run() {
        let dir = tempdir().unwrap();
        let first = "magi-2026-01-01.log".to_string();
        let second = "magi-2026-01-02.log".to_string();
        write(dir.path(), &first, b"old");
        let kept = write(dir.path(), &second, b"also old");

        // Windows refuses to remove a file that is still open.
        let _guard = fs::File::open(&kept).unwrap();

        let outcome = execute_plan(
            dir.path(),
            &[first.clone(), second.clone()],
            &[Action::Delete, Action::Delete],
        );

        assert!(outcome.is_ok(), "retention is best effort");
        assert!(
            !dir.path().join(&first).exists(),
            "the deletable one still went, so the failure did not abort the loop"
        );
    }

    #[test]
    fn a_compressed_file_replaces_its_original_only_after_verification() {
        let dir = tempdir().unwrap();
        let name = "magi-2026-05-05.log".to_string();
        let body = b"a line that repeats\n".repeat(80);
        write(dir.path(), &name, &body);

        let done =
            execute_plan(dir.path(), std::slice::from_ref(&name), &[Action::Compress]).unwrap();

        assert_eq!(done, 1);
        assert!(
            !dir.path().join(&name).exists(),
            "the original goes once the archive is proven"
        );
        let xz = compressed_path(&dir.path().join(&name));
        assert!(xz.exists(), "and the archive is there");
        assert!(
            !staging_path(dir.path(), &name).exists(),
            "no staging file is left behind"
        );
    }
}
