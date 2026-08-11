// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-18
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Panic/bounds-safety lints: ONLY in production. Tests use `unwrap`/`expect`/indexing
// idiomatically (a failure in a test IS the test failing, which is the correct behavior).
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]
//! Discovery of the unified `.magi/` state directory (REQ-H16/H30/H31).
//!
//! The project state (config, encrypted DB, logs) lives under a single `.magi/` directory,
//! discovered by **walk-up** from the working directory in the style of `.git`. This module
//! provides:
//!
//! - [`Workspace`]: the discovered `.magi/` and the paths of its artifacts.
//! - [`discover`]: the hardened walk-up (symlink rejection, fs boundary limit).
//! - [`detect_legacy_files`]: the **primitive** for detecting legacy files loose in the cwd (the **emission** of the warning is MS2).

use std::fs;
use std::io;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use magi_rs::headless::HeadlessError;

/// Name of the unified state directory searched for during the walk-up.
const MAGI_DIR_NAME: &str = ".magi";

/// Name of the encrypted database file inside `.magi/`.
const DB_FILE_NAME: &str = ".magi-rs-memory.db";

/// Name of the configuration file inside `.magi/`.
const CONFIG_FILE_NAME: &str = "magi.toml";

/// Name of the logs subdirectory inside `.magi/`.
const LOGS_DIR_NAME: &str = "logs";

/// Error message when a component of the discovered path is a symlink.
const SYMLINK_COMPONENT_MSG: &str = "symlinked path component in .magi discovery";

/// Prefix of the sibling temporary directory for atomic `init`, used on
/// **all** platforms (`.magi.tmp.<rand>`), non-replacing rename
/// onto the final `.magi/` (Linux: `renameat2(RENAME_NOREPLACE)`; others: `std::fs::rename`,
/// see [`rename_no_replace`]).
const TMP_DIR_PREFIX: &str = ".magi.tmp.";

/// Restrictive directory mode (`0700`): rwx only for the owner (REQ-H38, unix).
#[cfg(unix)]
const RESTRICTIVE_DIR_MODE: u32 = 0o700;

/// Restrictive file mode (`0600`): rw only for the owner (REQ-H38, unix).
#[cfg(unix)]
const RESTRICTIVE_FILE_MODE: u32 = 0o600;

/// Windows `GENERIC_ALL` access mask — full control for the user the ACL is restricted to
/// (REQ-H38, Windows).
#[cfg(windows)]
const WINDOWS_FULL_CONTROL_MASK: u32 = 0x1000_0000;

/// The discovered `.magi/` state directory and the paths of its artifacts.
///
/// Constructed exclusively by [`discover`], which guarantees that no component of the path is a
/// symlink and that `magi_dir` is a real directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    /// Directory that **contains** the `.magi/` (its direct ancestor).
    pub root: PathBuf,
    /// Absolute and validated path of the `.magi/` directory.
    pub magi_dir: PathBuf,
}

impl Workspace {
    /// Path of the encrypted database (`.magi/.magi-rs-memory.db`).
    #[must_use]
    pub fn db_path(&self) -> PathBuf {
        self.magi_dir.join(DB_FILE_NAME)
    }

    /// Path of the configuration file (`.magi/magi.toml`).
    ///
    /// Consumed by `magi_toml_exists` (`main.rs`, the RF-9 notice for "no magi.toml") and,
    /// since Task 1.4, by the two call sites of `crate::config::MagiConfig::load` (TUI and
    /// headless) — replacing the `dir.join("magi.toml")` that those call sites were redoing,
    /// duplicating the `CONFIG_FILE_NAME` that this module already defines.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.magi_dir.join(CONFIG_FILE_NAME)
    }

    /// Path of the logs subdirectory (`.magi/logs`).
    ///
    /// Consumed by `main.rs`'s `build_run_log` (fix round 2, coordinator, 2026-08-03: the
    /// `#[allow(dead_code)]` this carried was already stale — the caller had arrived and nobody
    /// removed it).
    #[must_use]
    pub fn logs_dir(&self) -> PathBuf {
        self.magi_dir.join(LOGS_DIR_NAME)
    }
}

/// Discovers the `.magi/` of the closest ancestor to `start`, with a hardened walk-up of a
/// **single** mechanism (without canonicalizing `start`).
///
/// The walk is **raw** (it does not resolve symlinks of `start`, so it can reject them instead
/// of following them): (1) `start` is made absolute with [`std::path::absolute`] **without**
/// resolving `..`; (2) it is validated that **no** `Normal` component of that **raw** path is a
/// symlink, traversing it left to right and resolving `..` lexically by *pop*ping the prefix —
/// so a symlinked component is rejected **before** a later `..` erases it lexically
/// (`<root>/link/../sub` would normalize to `<root>/sub`, hiding the symlinked `link` if the
/// check ran after normalizing); (3) only then is it **lexically** normalized
/// (`lexical_normalize`, already guaranteed symlink-free) and walks up component by component,
/// stopping at the **filesystem boundary**, looking for a `.magi` that is a directory (and not
/// a symlink). The result is the already-validated absolute path, **without re-canonicalizing**
/// (anti-TOCTOU: a second fs resolution would reopen the check→use window).
///
/// # Complexity
/// `O(d)` fs accesses, with `d` = depth of `start` (one `symlink_metadata` call per component
/// and per candidate).
///
/// # Platform residual (macOS)
/// The strict policy rejects **any** ancestor component that is a symlink. On macOS
/// `/tmp`→`/private/tmp`, `/var`, `/etc` are operating-system symlinks, so discovering directly
/// under `/tmp` fails; real use (projects under `/Users/...`, which is not a symlink) is
/// unaffected. It is an accepted trade-off in favor of security; it is not solved by relaxing
/// the rejection.
///
/// # Errors
/// Returns [`HeadlessError::InputInvalid`] if any component of the path (including `.magi`
/// itself) is a symlink, or [`HeadlessError::Io`] on an I/O error while making the path
/// absolute or reading metadata.
pub fn discover(start: &Path) -> Result<Option<Workspace>, HeadlessError> {
    let absolute = std::path::absolute(start).map_err(|e| HeadlessError::Io(e.to_string()))?;
    // Symlink check on the RAW absolute path BEFORE lexical `..` resolution, so a symlinked
    // component is caught at its own depth even when a later `..` would lexically erase it
    // (`<root>/link/../sub`).
    ensure_raw_chain_symlink_free(&absolute)?;
    let start_norm = lexical_normalize(&absolute);

    for dir in collect_search_dirs(&start_norm)? {
        let candidate = dir.join(MAGI_DIR_NAME);
        match classify_magi_candidate(&candidate)? {
            MagiCandidate::Directory => {
                return Ok(Some(Workspace {
                    root: dir,
                    magi_dir: candidate,
                }));
            }
            MagiCandidate::Absent => {}
        }
    }
    Ok(None)
}

/// Detects the presence of loose legacy files in `cwd` — the **primitive** of REQ-H31 (emitting
/// the warning to the user is MS2's responsibility).
///
/// Returns `true` if and only if **no** `.magi/` exists in `cwd` **and** at least one of the
/// legacy-layout files (`.magi-rs-memory.db` or `magi.toml`) is loose in `cwd`. With a `.magi/`
/// present the layout is already migrated and there is nothing to warn about.
///
/// Wired in MS2 T7: `main::run` emits the REQ-H31 startup warning when this returns `true`
/// (detect only — the legacy files are never read or migrated).
#[must_use]
pub fn detect_legacy_files(cwd: &Path) -> bool {
    !cwd.join(MAGI_DIR_NAME).is_dir()
        && (cwd.join(DB_FILE_NAME).exists() || cwd.join(CONFIG_FILE_NAME).exists())
}

/// Classification of a `.magi` candidate without following its final link.
enum MagiCandidate {
    /// The candidate exists and is a real directory (valid workspace).
    Directory,
    /// The candidate does not exist or is not a directory (the walk-up continues).
    Absent,
}

/// Classifies a `.magi` candidate by inspecting its metadata **without following** the final
/// component's link.
///
/// # Errors
/// Returns [`HeadlessError::InputInvalid`] if the candidate is a symlink, or
/// [`HeadlessError::Io`] on an I/O error other than "does not exist".
fn classify_magi_candidate(candidate: &Path) -> Result<MagiCandidate, HeadlessError> {
    match fs::symlink_metadata(candidate) {
        Ok(md) => {
            let file_type = md.file_type();
            if file_type.is_symlink() {
                Err(HeadlessError::InputInvalid(
                    SYMLINK_COMPONENT_MSG.to_owned(),
                ))
            } else if file_type.is_dir() {
                Ok(MagiCandidate::Directory)
            } else {
                Ok(MagiCandidate::Absent)
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(MagiCandidate::Absent),
        Err(e) => Err(HeadlessError::Io(e.to_string())),
    }
}

/// Validates that **no** `Normal` component of the **raw** `absolute` path is a symlink,
/// traversing it from left to right and resolving `..` lexically by *pop*ping the accumulated
/// prefix.
///
/// Running over the **raw** path (before [`lexical_normalize`]) is what closes the
/// `..`-through-symlink bypass: each `Normal` component is `symlink_metadata`-tested at the
/// instant it is *push*ed onto the prefix —at its own depth—, so a symlink is rejected
/// **before** a later `..` erases it lexically (`<root>/link/../sub`). A
/// [`Component::ParentDir`] *pop*s a component that was already validated when pushed, so a
/// legitimate `..` in the operator's path keeps working;
/// [`Component::Prefix`]/[`Component::RootDir`] anchor the traversal and [`Component::CurDir`]
/// is ignored.
///
/// # Complexity
/// `O(d)` with `d` = number of components of `absolute` (one `symlink_metadata` per `Normal`
/// component).
///
/// # Errors
/// Returns [`HeadlessError::InputInvalid`] if any `Normal` component is a symlink, or
/// [`HeadlessError::Io`] on an I/O error other than "does not exist".
fn ensure_raw_chain_symlink_free(absolute: &Path) -> Result<(), HeadlessError> {
    let mut prefix = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // The popped component was already symlink-checked when pushed.
                prefix.pop();
            }
            Component::Prefix(_) | Component::RootDir => {
                prefix.push(component.as_os_str());
            }
            Component::Normal(name) => {
                prefix.push(name);
                match fs::symlink_metadata(&prefix) {
                    Ok(md) if md.file_type().is_symlink() => {
                        return Err(HeadlessError::InputInvalid(
                            SYMLINK_COMPONENT_MSG.to_owned(),
                        ));
                    }
                    Ok(_) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(HeadlessError::Io(e.to_string())),
                }
            }
        }
    }
    Ok(())
}

/// Collects candidate directories from `start` toward the root (closest first), stopping at the
/// **filesystem boundary**.
///
/// # Complexity
/// `O(d)` with `d` = depth of `start` (two `symlink_metadata` calls per level for the fs-
/// boundary check).
///
/// # Errors
/// Returns [`HeadlessError::Io`] if it cannot read the metadata needed for the filesystem-
/// boundary check.
fn collect_search_dirs(start: &Path) -> Result<Vec<PathBuf>, HeadlessError> {
    let mut dirs = Vec::new();
    let mut current = start.to_path_buf();
    loop {
        dirs.push(current.clone());
        match current.parent() {
            Some(parent) => {
                let parent = parent.to_path_buf();
                if is_fs_boundary(&current, &parent)? {
                    break;
                }
                current = parent;
            }
            None => break,
        }
    }
    Ok(dirs)
}

/// Normalizes `path` **lexically** (without touching the fs): discards `.` components and
/// resolves `..` purely over the component string.
///
/// Resolving `..` lexically is safe here because [`discover`] later rejects any symlink
/// component of the resulting chain.
///
/// # MAGI re-gate WARNING — verified FALSE POSITIVE
/// Caspar flagged that `out.pop()` might pop `..` past the root, turning an absolute input into
/// a relative one. It does not: `PathBuf::pop()` truncates to `self.parent()`, and
/// `Path::parent()` returns `None` — so `pop()` is a documented no-op — exactly when the path
/// terminates in a `RootDir`/`Prefix` (verified empirically:
/// `lexical_normalize(Path::new(r"C:\..\..\etc"))` stays `C:\etc`, never `etc` or `..\etc`).
/// This module's only two callers ([`discover`], [`init`]) always feed it the output of
/// `std::path::absolute`, which is always rooted, so the root/prefix boundary is never at risk
/// here.
///
/// # Complexity
/// `O(d)` with `d` = number of components of `path`.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Indicates whether `dir` is the root of its filesystem relative to `parent` (POSIX): compares
/// the device number of both.
///
/// # Errors
/// Returns [`HeadlessError::Io`] if it cannot read the metadata of `dir` or `parent`.
#[cfg(unix)]
fn is_fs_boundary(dir: &Path, parent: &Path) -> Result<bool, HeadlessError> {
    use std::os::unix::fs::MetadataExt;

    let dir_dev = fs::symlink_metadata(dir)
        .map_err(|e| HeadlessError::Io(e.to_string()))?
        .dev();
    let parent_dev = fs::symlink_metadata(parent)
        .map_err(|e| HeadlessError::Io(e.to_string()))?
        .dev();
    Ok(dir_dev != parent_dev)
}

/// Indicates whether `dir` crosses a volume boundary relative to `parent` (Windows): compares
/// the volume root **lexically** via [`Component::Prefix`], without a raw syscall (honoring
/// `#![forbid(unsafe_code)]`).
///
/// # Test coverage note (MAGI re-gate INFO)
///
/// The walk-up's boundary stop is exercised generically by the discovery tests, but there is no
/// dedicated Windows test that asserts a drive/UNC volume change (e.g. `C:\...` vs `D:\...` or
/// a differing `\\server\share`) specifically halts the walk. The lexical comparison is total
/// and side-effect-free, so this is a coverage gap, not a correctness concern.
///
/// # Errors
/// Never fails; the `Result` signature unifies with the POSIX variant.
#[cfg(windows)]
fn is_fs_boundary(dir: &Path, parent: &Path) -> Result<bool, HeadlessError> {
    Ok(volume_prefix(dir) != volume_prefix(parent))
}

/// Extracts the lexical volume root of `path` (drive `C:` or UNC `\\server\share`), or `None`
/// if the path has no prefix.
#[cfg(windows)]
fn volume_prefix(path: &Path) -> Option<std::ffi::OsString> {
    path.components().find_map(|component| match component {
        Component::Prefix(prefix) => Some(prefix.as_os_str().to_os_string()),
        _ => None,
    })
}

/// Scaffolds a fresh `.magi/` state directory under `cwd` and returns the resulting
/// [`Workspace`] (REQ-H01/H38/H41).
///
/// Creates `cwd/.magi/` holding `magi.toml` (rendered defaults), an empty `logs/` subdirectory,
/// and the encrypted-store database with **all five tables** created empty and **no envelope**
/// (the first real open bootstraps it, MS1 Task 3). The directory is placed **atomically and
/// no-replace** on
/// **every** platform: the whole tree is built inside a randomly-named sibling
/// temp directory (`.magi.tmp.<rand>`, never a half-populated `.magi/` visible to a concurrent
/// reader — REQ-H07) and then moved into place with a single, platform-appropriate no-replace
/// rename (Linux: `renameat2(RENAME_NOREPLACE)`; elsewhere: `std::fs::rename`, see
/// [`rename_no_replace`]) that refuses to replace an existing `.magi/`. Every created object is
/// restricted to the current user (`0700`/`0600` on unix, an ACL restricted to the current user
/// on Windows).
///
/// Rejects a symlinked ANCESTOR component of `cwd` itself (parity with [`discover`]'s REQ-H30
/// check, via the same [`ensure_raw_chain_symlink_free`]): without it, `init` would silently
/// scaffold `.magi/` inside whatever a symlinked component resolves to instead of refusing —
/// the atomic no-replace rename only protects against a pre-existing `.magi/` itself being a
/// symlink, not an ancestor directory on the path leading to it.
///
/// # Errors
/// - [`HeadlessError::InputInvalid`] if an ancestor component of `cwd` is a symlink.
/// - [`HeadlessError::Aborted`] if `cwd/.magi/` already exists.
/// - [`HeadlessError::Io`] on a filesystem or ACL error (bad parent, rename).
/// - [`HeadlessError::Storage`] if the database schema cannot be created.
pub fn init(cwd: &Path) -> Result<Workspace, HeadlessError> {
    let absolute = std::path::absolute(cwd).map_err(|e| HeadlessError::Io(e.to_string()))?;
    ensure_raw_chain_symlink_free(&absolute)?;
    let root = lexical_normalize(&absolute);
    let magi_dir = root.join(MAGI_DIR_NAME);
    place_magi_dir(&magi_dir)?;
    Ok(Workspace { root, magi_dir })
}

/// Places a populated `.magi/` at `magi_dir` atomically and no-replace, on every platform:
/// builds the whole tree in a sibling `.magi.tmp.<rand>` (never a half-populated `.magi/`
/// visible to a reader) and renames it into place via [`rename_no_replace`]; on a populate
/// error removes only its own freshly-created scaffold.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `magi_dir` exists; [`HeadlessError::Io`] /
/// [`HeadlessError::Storage`] on a filesystem or schema error.
fn place_magi_dir(magi_dir: &Path) -> Result<(), HeadlessError> {
    let parent = magi_dir
        .parent()
        .ok_or_else(|| HeadlessError::Io("target .magi has no parent directory".to_owned()))?;
    let tmp = parent.join(format!("{TMP_DIR_PREFIX}{:016x}", rand::random::<u64>()));
    create_gate_dir(&tmp)?;
    populate_or_cleanup(&tmp)?;
    rename_no_replace(&tmp, magi_dir)
}

/// Populates the freshly-created, restricted `scaffold` directory and, on any populate error,
/// best-effort removes the scaffold `init` itself just created, returning the **original**
/// error (never the cleanup error).
///
/// Removing `init`'s own half-built scaffold does **not** violate never-delete (REQ-H20/H41):
/// the scaffold holds no user data yet — only `logs/`, a defaults `magi.toml`, and an empty
/// envelope-less DB. Leaving it orphaned would make a later `init` refuse (the no-replace
/// gate), so the cleanup keeps a crashed/failed `init` retryable.
///
/// # Errors
/// The original [`HeadlessError`] from [`populate_in_place`] (I/O, storage, or a pre-existing
/// child), unchanged; the cleanup outcome is intentionally ignored.
fn populate_or_cleanup(scaffold: &Path) -> Result<(), HeadlessError> {
    populate_or_cleanup_with(scaffold, populate_in_place)
}

/// [`populate_or_cleanup`] with an injectable populate step, so the failure-cleanup path is
/// unit-testable without forcing a real populate error.
///
/// # Errors
/// The original error returned by `populate`, unchanged (cleanup errors are swallowed).
fn populate_or_cleanup_with<F>(scaffold: &Path, populate: F) -> Result<(), HeadlessError>
where
    F: FnOnce(&Path) -> Result<(), HeadlessError>,
{
    match populate(scaffold) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best-effort: the scaffold contains NO user data (never-delete safe); return the
            // ORIGINAL error, not the cleanup outcome.
            let _ = fs::remove_dir_all(scaffold);
            Err(e)
        }
    }
}

/// Renames `tmp` onto `final_dir` atomically without replacing an existing target
/// (`renameat2(RENAME_NOREPLACE)`), falling back to the portable mkdir-gate if the
/// kernel/filesystem does not support the flag.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `final_dir` already exists; [`HeadlessError::Io`] /
/// [`HeadlessError::Storage`] on any other error.
#[cfg(target_os = "linux")]
fn rename_no_replace(tmp: &Path, final_dir: &Path) -> Result<(), HeadlessError> {
    use rustix::fs::{renameat_with, RenameFlags, CWD};
    use rustix::io::Errno;

    match renameat_with(CWD, tmp, CWD, final_dir, RenameFlags::NOREPLACE) {
        Ok(()) => Ok(()),
        Err(errno) => {
            let _ = fs::remove_dir_all(tmp);
            if errno == Errno::EXIST {
                Err(HeadlessError::Aborted)
            } else if errno == Errno::NOSYS || errno == Errno::INVAL || errno == Errno::OPNOTSUPP {
                // RENAME_NOREPLACE unsupported (pre-3.15 kernel or exotic FS): fail closed
                // rather than degrade to a non-atomic in-place gate, which would briefly expose
                // a half-populated `.magi/` at the well-known path (REQ-H07 atomicity; MAGI re-
                // gate WARNING).
                Err(HeadlessError::Io(format!(
                    "atomic no-replace directory creation is unsupported by this \
                     kernel/filesystem (renameat2 RENAME_NOREPLACE: {errno:?}); \
                     refusing to create .magi/ non-atomically (REQ-H07)"
                )))
            } else {
                Err(HeadlessError::Io(format!("rename failed: {errno:?}")))
            }
        }
    }
}

/// Renames `tmp` onto `final_dir` atomically via `std::fs::rename` (Windows, macOS, other unix)
/// — the portable counterpart of the Linux `renameat2` variant above.
///
/// A directory rename's OS semantics already refuse to replace an existing
/// **non-empty** destination (`ErrorKind::DirectoryNotEmpty` on Windows,
/// `ENOTEMPTY`-mapped errors on other unix): every real `.magi/` this guards against is non-
/// empty (`init` never leaves one with fewer than its three children), so that failure is
/// mapped to [`HeadlessError::Aborted`], the same refusal Linux's `RENAME_NOREPLACE` gives
/// directly.
///
/// # Residual (documented, not fixed)
/// An existing but **empty** `.magi/` — never `init`'s own output, only a directory created by
/// something else (e.g. a bare `mkdir`) — is *replaced* rather than refused:
/// `std::fs::rename`'s directory semantics allow renaming onto an empty destination on Windows
/// and (for non-Linux) other unix targets. Closing this fully needs an atomic "create-with-
/// initial-content" primitive (Windows: a `CreateFile`/`MoveFileEx` sequence with explicit
/// flags), which requires raw platform calls this crate's `#![forbid(unsafe_code)]` does not
/// allow; the narrow case carries no data loss (an empty directory holds nothing to lose).
///
/// # MAGI re-gate INFO — a SYMLINKED `.magi` is verified FALSE POSITIVE
/// Balthasar asked whether this fallback could replace or follow a *symlinked* `.magi`
/// (distinct from the empty-real-dir residual above). It cannot, on every platform this branch
/// covers: a directory rename requires the destination to be either absent or itself a
/// directory by type — a symlink is neither, regardless of what it points to or whether that
/// target is empty. Verified empirically on this Windows host: renaming a fresh directory onto
/// a `.magi` NTFS junction (both an empty and a non-empty target were tried) fails with
/// `PermissionDenied`, never replacing or writing through the junction. POSIX `rename(2)`
/// documents the same type check via `lstat` semantics, so macOS/BSD reject it the same way
/// Linux's `renameat2` does above.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `final_dir` exists and is non-empty; [`HeadlessError::Io`] on
/// any other rename failure.
#[cfg(not(target_os = "linux"))]
fn rename_no_replace(tmp: &Path, final_dir: &Path) -> Result<(), HeadlessError> {
    match fs::rename(tmp, final_dir) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_dir_all(tmp);
            if matches!(
                e.kind(),
                io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
            ) {
                Err(HeadlessError::Aborted)
            } else {
                Err(HeadlessError::Io(e.to_string()))
            }
        }
    }
}

/// Creates a directory as an atomic no-replace gate with restrictive permissions from creation
/// (`0700` on unix, a current-user ACL on Windows).
///
/// # Platform note: Windows ACL is applied post-creation (best-effort, REQ-H38)
///
/// On unix the mode is set atomically at `mkdir` time (`DirBuilder::mode`), so the directory is
/// never visible with looser permissions. On **Windows** the directory is created with the
/// default (inherited) DACL and then tightened by [`restrict_to_current_user`], leaving a brief
/// post-creation TOCTOU window in which the object carries its inherited ACL (MAGI re-gate
/// INFO). This is a
/// **best-effort** protection: the window is on the random-named temp sibling
/// (`.magi.tmp.<rand>`), not yet renamed into place as `.magi/`, so an attacker would have to
/// both guess the random name and win a sub-millisecond race. Closing it fully would require
/// creating the directory with an explicit `SECURITY_ATTRIBUTES` via `CreateDirectoryW` (out of
/// scope); the residual is documented rather than eliminated, consistent with the crate's other
/// best-effort OS-hardening measures.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `path` already exists; [`HeadlessError::Io`] on any other
/// filesystem or ACL error.
fn create_gate_dir(path: &Path) -> Result<(), HeadlessError> {
    create_restricted_dir_impl(path).map_err(map_create_err)?;
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Maps a directory/file creation [`io::Error`] to a [`HeadlessError`], turning an
/// `AlreadyExists` into [`HeadlessError::Aborted`] (the no-replace refusal).
fn map_create_err(e: io::Error) -> HeadlessError {
    if e.kind() == io::ErrorKind::AlreadyExists {
        HeadlessError::Aborted
    } else {
        HeadlessError::Io(e.to_string())
    }
}

/// Creates a single directory restricted to the owner (`0700`) from creation.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(unix)]
fn create_restricted_dir_impl(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new()
        .mode(RESTRICTIVE_DIR_MODE)
        .create(path)
}

/// Creates a single directory (non-recursive, no-replace); permissions are tightened separately
/// per platform.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(not(unix))]
fn create_restricted_dir_impl(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

/// Populates an already-created, restricted `.magi/` directory with `logs/`, `magi.toml`
/// (rendered defaults) and the empty five-table database.
///
/// # Errors
/// [`HeadlessError::Io`] on a filesystem/ACL error, [`HeadlessError::Storage`] if the schema
/// cannot be created, or [`HeadlessError::Aborted`] on an unexpected pre-existing child (should
/// not occur in a fresh directory).
fn populate_in_place(dir: &Path) -> Result<(), HeadlessError> {
    create_gate_dir(&dir.join(LOGS_DIR_NAME))?;
    write_restricted_file(
        &dir.join(CONFIG_FILE_NAME),
        crate::defaults::render_default_magi_toml().as_bytes(),
    )?;
    create_db(&dir.join(DB_FILE_NAME))?;
    Ok(())
}

/// Writes `contents` to a new owner-restricted file (`0600` on unix, current-user ACL on
/// Windows), refusing to overwrite an existing file.
///
/// # Errors
/// [`HeadlessError::Aborted`] if the file already exists; [`HeadlessError::Io`] on any other
/// filesystem or ACL error.
fn write_restricted_file(path: &Path, contents: &[u8]) -> Result<(), HeadlessError> {
    let mut file = open_new_restricted(path).map_err(map_create_err)?;
    file.write_all(contents)
        .map_err(|e| HeadlessError::Io(e.to_string()))?;
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Creates and opens a new file restricted to the owner (`0600`) from creation, failing if it
/// already exists.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(unix)]
fn open_new_restricted(path: &Path) -> io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(RESTRICTIVE_FILE_MODE)
        .open(path)
}

/// Creates and opens a new file (no-replace); permissions are tightened separately per
/// platform.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(not(unix))]
fn open_new_restricted(path: &Path) -> io::Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Creates the state database at `path` with all five schema tables empty (no envelope),
/// restricted to the owner.
///
/// Pre-creates the file with restrictive permissions so `rusqlite` never materializes it world-
/// readable under the umask; SQLite then treats the empty file as a fresh database. No `PRAGMA`
/// is set here — the first real open (MS1 Task 3) configures WAL and bootstraps the envelope.
///
/// # Errors
/// [`HeadlessError::Aborted`] if the file already exists; [`HeadlessError::Io`] on an ACL
/// error; [`HeadlessError::Storage`] if the schema cannot be created.
fn create_db(path: &Path) -> Result<(), HeadlessError> {
    open_new_restricted(path).map_err(map_create_err)?;
    {
        let conn =
            rusqlite::Connection::open(path).map_err(|e| HeadlessError::Storage(e.to_string()))?;
        crate::system::database::init_schema(&conn)
            .map_err(|e| HeadlessError::Storage(e.to_string()))?;
    }
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Restricts `path`'s DACL to the current user only — the Windows equivalent of unix
/// `0700`/`0600` (REQ-H38) — using the safe `windows-acl` crate.
///
/// Grants the current user full control (which writes a PROTECTED DACL, severing inheritance)
/// then removes every other ACE, leaving exactly one allow entry.
///
/// # Errors
/// [`HeadlessError::Io`] if the path is not valid UTF-8 or any Win32 ACL call fails (the
/// numeric error code is included, never a secret).
#[cfg(windows)]
fn restrict_to_current_user(path: &Path) -> Result<(), HeadlessError> {
    use windows_acl::acl::{AceType, ACL};
    use windows_acl::helper::{current_user, name_to_sid, sid_to_string, string_to_sid};

    let path_str = path.to_str().ok_or_else(|| {
        HeadlessError::Io("path is not valid UTF-8 for ACL application".to_owned())
    })?;
    let user = current_user()
        .ok_or_else(|| HeadlessError::Io("cannot resolve current Windows user".to_owned()))?;
    let user_sid = name_to_sid(&user, None)
        .map_err(|code| HeadlessError::Io(format!("name_to_sid failed (code {code})")))?;
    let user_string = sid_to_string(user_sid.as_ptr() as _)
        .map_err(|code| HeadlessError::Io(format!("sid_to_string failed (code {code})")))?;

    let mut acl = ACL::from_file_path(path_str, false)
        .map_err(|code| HeadlessError::Io(format!("read ACL failed (code {code})")))?;

    // Grant the current user full control; windows-acl writes a PROTECTED DACL, severing
    // inheritance so no parent ACE leaks in.
    acl.add_entry(
        user_sid.as_ptr() as _,
        AceType::AccessAllow,
        0,
        WINDOWS_FULL_CONTROL_MASK,
    )
    .map_err(|code| HeadlessError::Io(format!("grant user ACE failed (code {code})")))?;

    // Remove every ACE that is not the current user's, restricting access to exactly this user
    // (drops inherited SYSTEM/Administrators/Users entries).
    let entries = acl
        .all()
        .map_err(|code| HeadlessError::Io(format!("enumerate ACL failed (code {code})")))?;
    for entry in entries {
        if entry.string_sid == user_string {
            continue;
        }
        let sid = string_to_sid(&entry.string_sid)
            .map_err(|code| HeadlessError::Io(format!("string_to_sid failed (code {code})")))?;
        acl.remove(sid.as_ptr() as _, None, None)
            .map_err(|code| HeadlessError::Io(format!("remove ACE failed (code {code})")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        create_gate_dir, detect_legacy_files, discover, init, populate_or_cleanup_with, Workspace,
    };
    use magi_rs::headless::HeadlessError;
    use std::path::PathBuf;

    #[test]
    fn test_discover_finds_nearest_ancestor_magi_dir() {
        // Canonical root ONCE (resolves /tmp→/private/tmp on macOS BEFORE the walk); discover
        // does NOT canonicalize. The `tmp` guard is kept alive.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("a/b/.magi")).unwrap();
        let sub = root.join("a/b/c/d");
        std::fs::create_dir_all(&sub).unwrap();

        let ws = discover(&sub).unwrap().expect("found");

        assert_eq!(ws.magi_dir, root.join("a/b/.magi"));
        assert_eq!(ws.root, root.join("a/b"));
    }

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_symlinked_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("elsewhere");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join(".magi")).unwrap();

        assert!(matches!(
            discover(&root),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_symlinked_ancestor_component() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("real");
        std::fs::create_dir_all(real.join(".magi")).unwrap();
        let link = root.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let sub = link.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        // The `link` ancestor is a symlink ⇒ strict rejection.
        assert!(matches!(
            discover(&sub),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    #[test]
    fn test_discover_returns_none_when_no_magi_dir_in_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let sub = root.join("x/y/z");
        std::fs::create_dir_all(&sub).unwrap();

        assert_eq!(discover(&sub).unwrap(), None);
    }

    #[test]
    fn test_workspace_path_helpers_are_under_magi_dir() {
        let magi_dir = PathBuf::from("/proj/.magi");
        let ws = Workspace {
            root: PathBuf::from("/proj"),
            magi_dir: magi_dir.clone(),
        };

        assert_eq!(ws.db_path(), magi_dir.join(".magi-rs-memory.db"));
        assert_eq!(ws.config_path(), magi_dir.join("magi.toml"));
        assert_eq!(ws.logs_dir(), magi_dir.join("logs"));
    }

    #[test]
    fn test_init_creates_structure_and_refuses_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).expect("init");
        assert!(ws.magi_dir.join("magi.toml").exists());
        assert!(ws.magi_dir.join("logs").is_dir());
        assert!(ws.db_path().exists());
        // A second init must refuse (never overwrite) — atomic no-replace gate.
        assert!(matches!(init(tmp.path()), Err(HeadlessError::Aborted)));
    }

    /// SC-A02d: the REAL scaffold of `magi init` (not just isolated `render_default_magi_toml`)
    /// brings the v0.12.0 shape — `base_url` at the root, `provider` with the new vocabulary,
    /// WITHOUT `[openai].base_url` — and, the assertion that ties it all together, the file it
    /// writes PARSES through the SAME validation (`MagiConfig::load`, NOT `from_toml_str`) that
    /// the binary applies on every startup. Without this `magi init` could write a `magi.toml`
    /// that the binary itself rejects on the next startup — the worst possible first-use
    /// experience, and exactly what this milestone's migration exists to prevent.
    #[test]
    fn magi_init_scaffolds_a_magi_toml_the_binary_can_read_back() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).expect("init");
        let raw = std::fs::read_to_string(ws.config_path()).unwrap();

        // Checked separately, BEFORE parsing: `OpenAiConfig` no longer has a `base_url` field
        // (REQ-A21), so a regression here would not leave a loose key — it would make the
        // parsing below fail with a generic `unknown field`. This textual assertion fails with
        // a message that names the cause, instead of a parse panic that has to be interpreted.
        let raw_value: toml::Value = raw
            .parse()
            .expect("the scanned magi.toml must be valid TOML");
        if let Some(openai) = raw_value.get("openai") {
            assert!(
                openai.get("base_url").is_none(),
                "[openai].base_url moved to the root in REQ-A21; the scaffolder must \
                 not emit it"
            );
        }

        // The assertion that ties it all together: the scaffolder's output goes through the
        // SAME validation that the binary applies on every startup — `MagiConfig::load`, NOT
        // `from_toml_str`. `load` is strictly stricter: in addition to vocabulary and ranges,
        // it validates the THREE `base_url` templates with `EndpointTemplate::parse`
        // (REQ-A16c/SC-A16d). `from_toml_str` deliberately does not do this (its own malformed-
        // template tests depend on it not doing so), so only `load` tests what the binary
        // really applies on every startup. Fix round 2 (coordinator, 2026-08-03, I1): with
        // `from_toml_str` here, a `base_url` without a scheme (e.g. "localhost:11434/v1") would
        // have left this test green while every real `magi init` died with
        // `EndpointError::Unparseable` on the next startup — the exact first-use failure this
        // test exists to prevent.
        let (parsed, _notices) = crate::config::MagiConfig::load(&ws.config_path())
            .expect("magi init must never write a magi.toml the binary rejects");

        assert!(
            parsed.base_url().is_some(),
            "base_url must be declared at the root (REQ-A21)"
        );
        assert!(
            matches!(
                parsed.provider(),
                Some("ollama") | Some("openai-compat") | Some("anthropic")
            ),
            "provider must be one of the three REQ-A01b vocabulary values, got {:?}",
            parsed.provider()
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_init_sets_restrictive_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();
        let dir_mode = std::fs::metadata(&ws.magi_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let db_mode = std::fs::metadata(ws.db_path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
        assert_eq!(db_mode, 0o600);
    }

    #[test]
    #[cfg(windows)]
    fn test_init_restricts_acl_to_current_user() {
        use windows_acl::acl::ACL;
        use windows_acl::helper::{current_user, name_to_sid, sid_to_string};

        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();

        let user = current_user().unwrap();
        let user_sid = name_to_sid(&user, None).unwrap();
        let user_string = sid_to_string(user_sid.as_ptr() as _).unwrap();

        let acl = ACL::from_file_path(ws.magi_dir.to_str().unwrap(), false).unwrap();
        let entries = acl.all().unwrap();
        assert!(!entries.is_empty(), "the DACL must contain the user's ACE");
        for entry in entries {
            assert_eq!(
                entry.string_sid, user_string,
                "only the current user may hold an ACE on .magi/"
            );
        }
        // Owner access is retained: the DB the process just wrote is still there.
        assert!(ws.db_path().exists());
    }

    #[test]
    fn test_init_concurrent_yields_exactly_one_magi_dir() {
        // REQ-H03/H41: two threads racing `init` on the same fresh directory must resolve to
        // exactly ONE `.magi/` — the atomic no-replace gate lets one win and refuses the other
        // with `Aborted`, never a half-populated dir or two envelopes. A `Barrier` maximizes
        // the overlap of the two `init`s.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let dir = dir.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    init(&dir)
                })
            })
            .collect();

        let results: Vec<Result<Workspace, HeadlessError>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        let winners = results.iter().filter(|r| r.is_ok()).count();
        let aborted = results
            .iter()
            .filter(|r| matches!(r, Err(HeadlessError::Aborted)))
            .count();
        assert_eq!(winners, 1, "exactly one concurrent init must win");
        assert_eq!(
            aborted, 1,
            "the loser must be refused with Aborted (atomic no-replace)"
        );

        // The single `.magi/` is COMPLETE (not half-populated) and unique.
        let magi = dir.join(".magi");
        assert!(magi.join("magi.toml").exists(), "config present");
        assert!(magi.join(".magi-rs-memory.db").exists(), "DB present");
        assert!(magi.join("logs").is_dir(), "logs/ present");
    }

    #[test]
    fn test_init_never_exposes_a_partially_populated_final_dir() {
        // REQ-H07 / MAGI re-gate finding: on EVERY platform (not just Linux), a concurrent
        // observer polling `.magi/`'s existence must never see it present without ALREADY being
        // fully populated (`magi.toml`, `logs/`, and the DB all present) — the final directory
        // must come into existence pre-built via a single atomic rename, never via an in-place
        // `mkdir` followed by separate population steps that leave a visible half-built window.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let magi = dir.join(".magi");
        let seen_partial = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        let watcher = {
            let magi = magi.clone();
            let seen_partial = std::sync::Arc::clone(&seen_partial);
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                // Bounded hot-spin: stop as soon as `.magi/` is observed, or after a generous
                // cap so a slow/CI machine can't hang.
                for _ in 0..2_000_000 {
                    if magi.is_dir() {
                        let complete = magi.join("magi.toml").exists()
                            && magi.join("logs").is_dir()
                            && magi.join(".magi-rs-memory.db").exists();
                        if !complete {
                            seen_partial.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        break;
                    }
                }
            })
        };

        barrier.wait();
        let ws = init(&dir).expect("init");
        watcher.join().unwrap();

        assert!(
            !seen_partial.load(std::sync::atomic::Ordering::SeqCst),
            "a concurrent observer must never see .magi/ half-populated"
        );
        assert!(ws.db_path().exists());
    }

    #[test]
    fn test_orphan_tmp_dir_does_not_break_a_later_init() {
        let tmp = tempfile::tempdir().unwrap();
        // Simulate a crashed prior run that left a stray sibling temp dir behind.
        std::fs::create_dir(tmp.path().join(".magi.tmp.deadbeef")).unwrap();

        let ws = init(tmp.path()).expect("init succeeds despite the orphan tmp");
        assert!(ws.magi_dir.is_dir());
        assert!(ws.db_path().exists());
        // The `.magi/` is complete, not half-populated.
        assert!(ws.magi_dir.join("magi.toml").exists());
    }

    #[test]
    fn test_populate_failure_cleans_up_scaffold_and_allows_retry() {
        // REQ-H41 / Fix: a populate error AFTER the scaffold dir is created must remove only
        // that just-built scaffold (no user data yet), return the ORIGINAL error, and leave no
        // orphan that a retry would refuse.
        let tmp = tempfile::tempdir().unwrap();
        let scaffold = tmp.path().join(".magi");
        create_gate_dir(&scaffold).expect("gate dir created");
        assert!(
            scaffold.is_dir(),
            "the scaffold exists before populate runs"
        );

        let err = populate_or_cleanup_with(&scaffold, |_| {
            Err(HeadlessError::Storage(
                "injected populate failure".to_owned(),
            ))
        })
        .expect_err("the injected populate failure must propagate");
        assert!(
            matches!(err, HeadlessError::Storage(_)),
            "the ORIGINAL populate error is returned, not the cleanup outcome"
        );
        assert!(
            !scaffold.exists(),
            "the half-built scaffold must be removed on populate failure"
        );

        // The cleaned-up failure does not block a subsequent real init.
        let ws = init(tmp.path()).expect("init proceeds after the cleaned-up failure");
        assert!(ws.db_path().exists());
        assert!(ws.magi_dir.join("magi.toml").exists());
    }

    // T2↔T3 lock-in (MS1 Task 2 Step 4c-bis / Task 3 Step 9b): a freshly-`init`ed DB has
    // exactly the five empty tables and NO envelope row — the precondition under which Task 3's
    // never-delete state machine bootstraps cleanly (never `DbCorrupt`). Now that
    // `open_with_state_machine` exists (T3), the final assertion drives it directly, closing
    // the T2↔T3 coupling executably.
    #[test]
    fn test_fresh_init_db_bootstraps_cleanly_under_state_machine() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();
        // Structural precondition: five empty tables, no envelope.
        let conn = rusqlite::Connection::open(ws.db_path()).unwrap();
        for table in [
            "sessions",
            "messages",
            "knowledge",
            "memories",
            "vault_meta",
        ] {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                .unwrap_or_else(|_| panic!("table `{table}` must exist on fresh init"));
            assert_eq!(count, 0, "table `{table}` must be empty on fresh init");
        }
        let has_envelope: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vault_meta WHERE key = 'wrapped_dek'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_envelope, 0, "a fresh init has no envelope yet");
        drop(conn);

        // T3 lock-in: the state machine opens the fresh DB cleanly (bootstraps the envelope),
        // NEVER `DbCorrupt`.
        match crate::system::database::EncryptedSqliteMemory::open_with_state_machine(
            ws.db_path(),
            zeroize::Zeroizing::new("fresh-init-state-machine-master".to_string()),
        ) {
            Ok(_) => {}
            Err(e) => panic!("fresh init must bootstrap cleanly under the state machine: {e:?}"),
        }
    }

    #[test]
    fn test_detect_legacy_files_true_for_loose_db_without_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".magi-rs-memory.db"), b"x").unwrap();

        assert!(detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_true_for_loose_config_without_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("magi.toml"), b"x").unwrap();

        assert!(detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_false_when_magi_dir_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".magi")).unwrap();
        // A loose legacy file is ignored if `.magi/` already exists.
        std::fs::write(tmp.path().join(".magi-rs-memory.db"), b"x").unwrap();

        assert!(!detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_false_when_directory_is_clean() {
        let tmp = tempfile::tempdir().unwrap();

        assert!(!detect_legacy_files(tmp.path()));
    }

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_parentdir_through_symlink_component() {
        // The `..`-through-symlink bypass: a start path that traverses a symlink and then `..`
        // back out. Lexical normalization would rewrite `<root>/link/../real/sub` to
        // `<root>/real/sub`, erasing the symlinked `link` before it is ever checked. The raw-
        // component check catches `link` at its own depth, BEFORE the `..` pops it.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("real");
        std::fs::create_dir_all(real.join(".magi")).unwrap();
        std::fs::create_dir_all(real.join("sub")).unwrap();
        std::os::unix::fs::symlink(&real, root.join("link")).unwrap();

        let start = root.join("link").join("..").join("real").join("sub");
        assert!(
            matches!(discover(&start), Err(HeadlessError::InputInvalid(_))),
            "a `..` that first traverses a symlinked component must be rejected"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_init_rejects_symlinked_ancestor_component() {
        // MAGI re-gate WARNING (parity with `discover`, REQ-H30): `discover` rejects a
        // symlinked ancestor component before walking up, but `init` used to just
        // `std::path::absolute` + lexically normalize the target `cwd` with no symlink check at
        // all — so a symlinked ancestor component would be silently followed by the OS at
        // directory-creation time instead of rejected up front. `init` now runs the same
        // `ensure_raw_chain_symlink_free` check `discover` does, on the same raw absolute
        // `cwd`, before ever touching the filesystem.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = root.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        assert!(
            matches!(init(&link), Err(HeadlessError::InputInvalid(_))),
            "init through a symlinked ancestor component must be rejected"
        );
        assert!(
            !real.join(".magi").exists(),
            "no .magi/ must be created through the rejected symlink"
        );
    }

    #[test]
    fn test_discover_allows_parentdir_on_non_symlinked_path() {
        // A legitimate `..` on a fully non-symlinked path still resolves and finds the ancestor
        // `.magi/`: `<root>/a/b/../c` normalizes to `<root>/a/c`, and the walk-up discovers
        // `<root>/a/.magi`.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("a").join(".magi")).unwrap();
        std::fs::create_dir_all(root.join("a").join("b")).unwrap();
        std::fs::create_dir_all(root.join("a").join("c")).unwrap();

        let start = root.join("a").join("b").join("..").join("c");
        let ws = discover(&start).unwrap().expect("found");
        assert_eq!(ws.magi_dir, root.join("a").join(".magi"));
        assert_eq!(ws.root, root.join("a"));
    }
}
