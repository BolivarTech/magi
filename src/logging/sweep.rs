// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

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

/// A source file's identity for the purpose of REQ-L14's delete guard.
///
/// Size and modification time together: either alone misses a case. A rewrite
/// that keeps the length changes only the time; a filesystem with coarse time
/// granularity can leave the time equal across an append that changes only the
/// length.
type SourceStamp = Option<(u64, std::time::SystemTime)>;

/// Whether the original may be deleted now that its archive verified.
///
/// # Parameters
///
/// * `before` — the source's stamp taken before the compression read it.
/// * `after` — its stamp taken after the archive verified.
///
/// # Returns
///
/// `true` only when the file is still the one that was archived.
///
/// # Why an absent stamp is a refusal
///
/// `None` means the metadata could not be read, which is not evidence that
/// nothing changed. REQ-L14 guards an action that cannot be undone, so the
/// unknown case keeps the file: a redundant copy costs disk, and a wrong delete
/// costs the data.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
fn may_delete_source(before: SourceStamp, after: SourceStamp) -> bool {
    match (before, after) {
        (Some(b), Some(a)) => b == a,
        _ => false,
    }
}

/// Why one file did not get compressed.
///
/// Two variants and not one message, because REQ-L61 turns on telling them
/// apart: a failure to compress is a resource problem, and a source that moved
/// underneath is a data problem. One string for both sends the operator to the
/// wrong place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// Compression or its verification failed; carries the file and the cause.
    Compression(String, String),
    /// The archive verified, but the original changed while it was being read,
    /// so deleting it would lose bytes the archive never covered.
    SourceMoved(String),
}

/// What one retention pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Swept {
    /// Files compressed and whose originals were removed.
    pub compressed: usize,
    /// Files that were not, with the reason for each.
    pub failures: Vec<Failure>,
}

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
/// What was compressed and what refused to be. Deletions are not counted
/// because nothing downstream bounds them.
///
/// # Errors
///
/// Never. Per-file failures travel in the return value instead, because they
/// are what the CALLER has to announce: REQ-L61 requires a compression failure
/// to be reported apart from the total-size warning, and swallowing it here
/// leaves the operator reading "the directory is over its ceiling" run after
/// run with no hint that the cause is no room to compress.
///
/// # Complexity
///
/// `O(n)` over the files, plus the cost of each compression.
pub fn execute_plan(
    dir: &Path,
    names: &[String],
    actions: &[Action],
) -> Result<Swept, LoggingError> {
    let mut compressed = 0usize;
    let mut failures = Vec::new();
    for (name, action) in names.iter().zip(actions.iter()) {
        let path = dir.join(name);
        match action {
            Action::Keep => {}
            Action::Compress => {
                if compressed >= MAX_COMPRESSIONS_PER_START {
                    continue;
                }
                // REQ-L14: what the source looked like BEFORE, so the delete
                // below can tell "the file I compressed" from "a file that
                // moved under me". `Delete` is not undoable, so the check has
                // to be about the bytes that were actually read.
                let before = fs::metadata(&path)
                    .ok()
                    .and_then(|m| m.modified().ok().map(|t| (m.len(), t)));
                let staged = staging_path(dir, name);
                match compress_verified(&path, &staged) {
                    Ok(()) => {
                        compressed += 1;
                        let after = fs::metadata(&path)
                            .ok()
                            .and_then(|m| m.modified().ok().map(|t| (m.len(), t)));
                        // The original goes only once the .xz is proven good AND
                        // the original is still what was proven. A file that grew
                        // or was rewritten during the compression is NOT in the
                        // archive, and deleting it loses data the verification
                        // never covered.
                        if may_delete_source(before, after) {
                            let _ = fs::remove_file(&path);
                        } else {
                            failures.push(Failure::SourceMoved(name.clone()));
                        }
                    }
                    Err(e) => {
                        let _ = fs::remove_file(&staged);
                        // REQ-L61: announced APART from the total-size warning,
                        // and the two causes stay distinguishable. A nearly full
                        // disk makes compression fail, retention cannot shrink
                        // the directory, and the size warning then describes the
                        // symptom while hiding the cause.
                        failures.push(Failure::Compression(name.clone(), e.to_string()));
                    }
                }
            }
            Action::Delete => {
                // Best effort (REQ-L17): a locked file stays a candidate.
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(compressed_path(&path));
            }
        }
    }
    Ok(Swept {
        compressed,
        failures,
    })
}

/// Where a compression stages its output.
///
/// Same directory as the source, so the final rename is atomic.
///
/// # Why the name carries the process
///
/// `<name>.tmp` is one path, and two processes compressing the same day collide
/// on it: each writes over the other, and the error branch of either removes a
/// temporary the other is still filling. A network `log_dir` makes this ordinary
/// rather than exotic, which is why D-L08 lists the suffix among what is
/// explicitly NOT deferred.
///
/// The PID is a disambiguator and never a liveness signal -- REQ-L55 decides
/// abandonment by `mtime` alone, so a recycled PID or one belonging to another
/// host confuses nothing.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn staging_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{}.{TMP_EXTENSION}", std::process::id()))
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
    fn a_source_that_changed_during_compression_is_kept_not_deleted() {
        // REQ-L14. The verification proves the ARCHIVE matches what was read;
        // it says nothing about a file that grew afterwards, and deleting on
        // the strength of it loses the bytes it never covered. A delete is the
        // one action in this module that cannot be undone.
        //
        // The DECISION is tested rather than the race, because the race cannot
        // be produced from outside: it needs a writer appending between the
        // read and the delete, inside a call this test makes. Extracting the
        // decision is what makes the part that can be wrong assertable, and
        // leaves one line at the call site to review.
        let t0 = std::time::SystemTime::UNIX_EPOCH;
        let t1 = t0 + std::time::Duration::from_secs(1);

        assert!(
            may_delete_source(Some((10, t0)), Some((10, t0))),
            "an unchanged file is the one that was archived"
        );
        assert!(
            !may_delete_source(Some((10, t0)), Some((20, t0))),
            "an append that leaves the mtime alone must still be caught"
        );
        assert!(
            !may_delete_source(Some((10, t0)), Some((10, t1))),
            "and a rewrite that keeps the length must be caught by the time"
        );
        assert!(
            !may_delete_source(Some((10, t0)), None),
            "metadata that cannot be read is not evidence that nothing changed"
        );
        assert!(
            !may_delete_source(None, Some((10, t0))),
            "and neither is a missing baseline"
        );
    }

    #[test]
    fn a_compression_failure_is_reported_rather_than_swallowed() {
        // REQ-L61. Announced APART from the total-size warning: on a nearly
        // full disk compression fails, retention cannot shrink the directory,
        // and the size warning alone describes the symptom while hiding the
        // cause. Returning the failure is what lets the caller say both.
        let dir = tempdir().unwrap();
        let name = "magi-2026-08-03.log";
        // A name the plan will try to compress, with nothing behind it: the
        // source cannot be opened, which is the shape a disk error takes here.
        let done = execute_plan(dir.path(), &[name.to_string()], &[Action::Compress]).unwrap();

        assert_eq!(done.compressed, 0);
        assert_eq!(done.failures.len(), 1, "the failure must reach the caller");
        assert!(
            matches!(done.failures[0], Failure::Compression(ref n, _) if n == name),
            "and name the file and its cause: {:?}",
            done.failures[0]
        );
    }

    #[test]
    fn two_processes_do_not_stage_into_the_same_temporary() {
        // D-L08 lists the unique suffix among what is explicitly NOT deferred.
        // With one `<name>.tmp` for everyone, two processes compressing the
        // same day write over each other, and the error branch of either
        // removes a temporary the other is still filling. A network log_dir
        // makes that ordinary rather than exotic.
        let dir = std::path::Path::new("/logs");
        let mine = staging_path(dir, "magi-2026-08-04.log");
        assert!(
            mine.to_string_lossy()
                .contains(&std::process::id().to_string()),
            "the staging path does not distinguish this process: {mine:?}"
        );
        assert!(
            mine.to_string_lossy().ends_with(TMP_EXTENSION),
            "and it must still be recognisable to the orphan sweep: {mine:?}"
        );
    }

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
            done.compressed,
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

        assert_eq!(done.compressed, 1);
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
