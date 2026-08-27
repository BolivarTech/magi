// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! This module provides a security layer for path validation and sandboxing.

use anyhow::{anyhow, Result};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Utility to ensure paths are safe and stay within the workspace boundary.
pub struct PathGuard {
    workspace_root: PathBuf,
}

impl PathGuard {
    /// Returns a reference to the (canonicalized) workspace root.
    ///
    /// Used by the `bash` tool's rm-root guard to detect a destructive `rm`
    /// whose target resolves to the workspace root itself.
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Creates a new `PathGuard` with the given workspace root.
    pub fn new(root: PathBuf) -> Result<Self> {
        let root = Self::canonicalize_robust(&root)
            .map_err(|e| anyhow!("Invalid workspace root: {}", e))?;
        Ok(Self {
            workspace_root: root,
        })
    }

    /// Validates and canonicalizes a path, ensuring it is within the workspace.
    /// Returns the fully canonicalized path to avoid TOCTOU races in the caller.
    ///
    /// # Security: why the ancestor search runs on RAW components
    ///
    /// A purely lexical `..` collapse (as [`Self::lexical_normalize`] performs)
    /// is only sound on a path that is already known to be symlink-free. This
    /// method used to collapse `..` on the RAW input first — but
    /// `workspace/sym/../x`, with `sym` a symlink to an outside directory,
    /// lexically collapses to `workspace/x` (validates fine) while the OS
    /// actually resolves the original by following `sym` FIRST and only then
    /// applying `..` on the far side, landing outside the workspace entirely
    /// (MAGI re-gate CRITICAL finding).
    ///
    /// The fix: find the closest EXISTING ancestor by testing raw
    /// component-slices of the **un-normalized** path directly against
    /// `fs::symlink_metadata` (letting the OS itself resolve any embedded `..`
    /// or symlink along the way — this is what makes it sound), canonicalize
    /// only that existing prefix (resolving every symlink in it), verify it is
    /// inside the workspace, and only THEN lexically normalize the
    /// (non-existent, and therefore necessarily symlink-free) tail appended to
    /// it. A final `starts_with` check catches a tail `..` that would
    /// otherwise walk back out of the now-canonical ancestor.
    pub fn validate(&self, input_path: &Path) -> Result<PathBuf> {
        let path_str = input_path.to_string_lossy();

        // Security: Disallow null bytes which can be used to bypass filename checks
        if path_str.contains('\0') {
            return Err(anyhow!("Security violation: null bytes in path"));
        }

        let absolute_path = if input_path.is_absolute() {
            input_path.to_path_buf()
        } else {
            self.workspace_root.join(input_path)
        };

        let comps: Vec<Component> = absolute_path.components().collect();

        let (ancestor, tail_start) = Self::closest_existing_ancestor(&comps)
            .ok_or_else(|| anyhow!("Security violation: path has no valid anchor in workspace"))?;

        let canonical_ancestor = Self::canonicalize_robust(&ancestor)
            .map_err(|e| anyhow!("Security check failed: unreachable ancestor: {}", e))?;

        // Critical check: the EXISTING prefix (fully symlink-resolved) must
        // already be inside the workspace, before any tail is appended.
        if !canonical_ancestor.starts_with(&self.workspace_root) {
            return Err(anyhow!(
                "Security violation: path escapes sandbox via traversal"
            ));
        }

        // The tail is guaranteed non-existent (else it would have been part
        // of the ancestor), so it cannot contain a symlink: lexically
        // normalizing it here, anchored on the already-canonical ancestor, is
        // sound.
        let mut reconstructed = canonical_ancestor;
        for comp in comps.get(tail_start..).into_iter().flatten() {
            reconstructed.push(comp.as_os_str());
        }
        let final_path = self.lexical_normalize(&reconstructed);

        // Final check: a tail `..` could still walk back out of the
        // canonical ancestor (`starts_with` alone never resolves `..`).
        if !final_path.starts_with(&self.workspace_root) {
            return Err(anyhow!("Security violation: final path escapes sandbox"));
        }

        Ok(final_path)
    }

    /// Finds the closest existing ancestor of `comps` by testing raw
    /// component-slices — from the full path down to the root — directly
    /// against `fs::symlink_metadata`, letting the OS resolve any `..` or
    /// symlink embedded in the candidate itself.
    ///
    /// Returns `(ancestor_path, tail_start)` where `comps[tail_start..]` is
    /// the non-existent (and therefore symlink-free) tail, or `None` if not
    /// even the shortest candidate (the volume root/prefix) exists.
    fn closest_existing_ancestor(comps: &[Component]) -> Option<(PathBuf, usize)> {
        for k in (1..=comps.len()).rev() {
            let candidate: PathBuf = comps.get(..k)?.iter().collect();
            if fs::symlink_metadata(&candidate).is_ok() {
                return Some((candidate, k));
            }
        }
        None
    }

    /// Purely lexical normalization of a path.
    /// Preserves RootDir and Prefix to prevent popping out of root boundaries.
    fn lexical_normalize(&self, path: &Path) -> PathBuf {
        let mut components = Vec::new();
        for component in path.components() {
            match component {
                Component::ParentDir => match components.last() {
                    Some(Component::Normal(_)) => {
                        components.pop();
                    }
                    Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                    _ => {
                        components.push(Component::ParentDir);
                    }
                },
                Component::CurDir => {}
                c => {
                    components.push(c);
                }
            }
        }
        components.iter().collect()
    }

    /// Robust canonicalization that handles Windows verbatim paths (UNC prefixes) consistently.
    ///
    /// Delegates to the `dunce` crate rather than a hand-rolled prefix strip:
    /// a naive `strip_prefix(r"\\?\")` is only correct for the local-disk
    /// verbatim shape (`\\?\C:\...`) — applied to a verbatim-**UNC** canonical
    /// form (`\\?\UNC\server\share\...`) it produces `UNC\server\share\...`,
    /// which is missing its leading `\\` and is not a valid absolute path at
    /// all. `dunce::canonicalize` only strips when the path's prefix is
    /// `Prefix::VerbatimDisk`, leaving any other verbatim shape (including
    /// `VerbatimUNC`) untouched, so the returned path is always valid. On
    /// non-Windows this is a plain passthrough to `std::fs::canonicalize`.
    fn canonicalize_robust(path: &Path) -> io::Result<PathBuf> {
        dunce::canonicalize(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_path_validation() {
        let dir = tempdir().expect("Failed to create temp dir");
        let root = dir.path().to_path_buf();
        let guard = PathGuard::new(root.clone()).expect("Failed to create guard");

        // Safe path
        let safe_file = root.join("safe.txt");
        fs::write(&safe_file, "data").unwrap();
        assert!(guard.validate(&safe_file).is_ok());

        // Unsafe path (traversal)
        let unsafe_path = root.join("dir/../../outside.txt");
        assert!(guard.validate(&unsafe_path).is_err());

        // Null byte attack
        let null_path = Path::new("test\0file.txt");
        assert!(guard.validate(null_path).is_err());

        // Non-existent safe path with multiple levels
        let new_file = root.join("a/b/c/new_file.txt");
        assert!(guard.validate(&new_file).is_ok());
    }

    #[test]
    #[cfg(unix)]
    fn test_validate_rejects_path_through_dangling_symlink_component() {
        // MAGI re-gate finding: the closest-existing-ancestor walk used
        // `Path::exists()`, which FOLLOWS the final symlink and returns
        // `false` for a DANGLING symlink (link present, target missing). A
        // dangling symlink component was therefore misclassified as "doesn't
        // exist yet" (a future, safely-creatable path segment) and appended
        // lexically onto the ancestor without ever being resolved — but it
        // DOES exist as a filesystem object right now, and an attacker who
        // controls its target (anywhere on the filesystem) can redirect a
        // later real write through it, straight out of the sandbox. Using
        // `fs::symlink_metadata` (which does not follow the final symlink)
        // instead of `exists()` fixes this: the dangling link is found as a
        // real ancestor, `canonicalize_robust` then fails to resolve it (or
        // resolves it outside the workspace), and `validate` rejects it.
        //
        // Windows dev host cannot exercise this (symlink creation requires a
        // privilege this environment lacks — verified empirically); CI runs
        // ubuntu-latest, where this test executes for real.
        let dir = tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let guard = PathGuard::new(root.clone()).expect("guard");

        std::os::unix::fs::symlink(
            "/nonexistent-target-for-magi-pathguard-test",
            root.join("danglink"),
        )
        .expect("create dangling symlink");

        let target = root.join("danglink").join("sub").join("file.txt");
        assert!(
            guard.validate(&target).is_err(),
            "a path through a dangling symlink component must be rejected, not silently accepted"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_validate_rejects_symlink_dotdot_escape() {
        // MAGI re-gate CRITICAL: `validate` used to lexically collapse `..`
        // (`lexical_normalize`) BEFORE ever walking the filesystem for an
        // existing ancestor. That collapse is only sound on a symlink-free
        // path: `root/sym/../x`, with `sym` a symlink to an OUTSIDE directory,
        // lexically becomes `root/x` (validates fine), but the OS resolves the
        // ORIGINAL string by first following `sym` to the outside target and
        // THEN applying `..` there — landing outside the workspace entirely.
        // The fix finds the closest EXISTING ancestor by testing raw
        // component-slices (so an intermediate symlink is followed by the OS,
        // not lexically erased), canonicalizes ONLY that existing prefix, and
        // only then lexically normalizes the non-existent (therefore
        // symlink-free) tail.
        let dir = tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let guard = PathGuard::new(root.clone()).expect("guard");

        let outside = tempdir().expect("outside temp dir");
        std::fs::create_dir_all(outside.path().join("etc")).expect("outside/etc");
        std::fs::write(outside.path().join("etc").join("passwd"), b"root:x:0:0").unwrap();

        std::os::unix::fs::symlink(outside.path(), root.join("sym")).expect("create symlink");

        let escape_one = root.join("sym").join("..").join("x");
        assert!(
            guard.validate(&escape_one).is_err(),
            "root/sym/../x must not escape via a symlinked `sym`"
        );

        let escape_two = root
            .join("sym")
            .join("..")
            .join("..")
            .join("etc")
            .join("passwd");
        assert!(
            guard.validate(&escape_two).is_err(),
            "root/sym/../../etc/passwd must not escape via a symlinked `sym`"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_validate_rejects_existing_symlink_to_outside_component() {
        // A symlink component that IS immediately followed by a real,
        // existing file on the far side (no `..` involved at all) must still
        // be rejected: the ancestor-search finds the full path as "existing"
        // (the OS follows `sym` transparently mid-path), and canonicalizing
        // it resolves fully outside the workspace root.
        let dir = tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let guard = PathGuard::new(root.clone()).expect("guard");

        let outside = tempdir().expect("outside temp dir");
        std::fs::write(outside.path().join("file"), b"secret").unwrap();

        std::os::unix::fs::symlink(outside.path(), root.join("sym")).expect("create symlink");

        let target = root.join("sym").join("file");
        assert!(
            guard.validate(&target).is_err(),
            "a path through an existing symlink to an outside directory must be rejected"
        );
    }

    #[test]
    fn test_validate_allows_legit_dotdot_within_workspace() {
        // A `..` that never leaves a symlink-free workspace must still
        // resolve fine: `root/a/../b.txt` with `a` a REAL (non-symlink)
        // directory is legitimate, even though `b.txt` doesn't exist yet.
        // Canonicalize `root` up front (as the verbatim-path tests above do)
        // to avoid Windows 8.3 short-name aliasing between `root` and the
        // fully-canonicalized path `validate` returns.
        let dir = tempdir().expect("temp dir");
        let root = dunce::canonicalize(dir.path()).expect("canonicalize root");
        let guard = PathGuard::new(root.clone()).expect("guard");

        std::fs::create_dir_all(root.join("a")).unwrap();

        let legit = root.join("a").join("..").join("b.txt");
        let validated = guard
            .validate(&legit)
            .expect("a legitimate .. within the workspace must validate");
        assert!(validated.starts_with(&root));
        assert_eq!(validated, root.join("b.txt"));
    }

    #[test]
    #[cfg(windows)]
    fn test_canonicalize_robust_preserves_valid_verbatim_unc_form() {
        // MAGI re-gate finding: the OLD `canonicalize_robust` did an
        // unconditional `path_str.strip_prefix(r"\\?\")`, which for a
        // verbatim-UNC canonical form (`\\?\UNC\server\share\file`) produces
        // `UNC\server\share\file` — missing the leading `\\`, not a valid
        // absolute path at all (`starts_with`/`starts_with` sandbox checks
        // downstream would then compare against a malformed path). `dunce`
        // (which now backs `canonicalize_robust`) only strips the `\\?\`
        // prefix for `Prefix::VerbatimDisk` (local drives, e.g. `\\?\C:\...`)
        // and deliberately leaves any other prefix kind — including
        // `VerbatimUNC` — untouched. This is exercised directly against
        // `dunce::simplified`, which performs no I/O, since a real UNC path
        // requires a live network share this environment cannot provide.
        let unc = Path::new(r"\\?\UNC\server\share\file.txt");

        let simplified = dunce::simplified(unc);
        assert_eq!(
            simplified, unc,
            "a VerbatimUNC path must be left untouched (still valid & absolute)"
        );

        // What the OLD unconditional strip would have produced, for contrast.
        let naive_stripped = PathBuf::from(
            unc.to_string_lossy()
                .strip_prefix(r"\\?\")
                .expect("input starts with the verbatim prefix by construction"),
        );
        assert_eq!(naive_stripped, Path::new(r"UNC\server\share\file.txt"));
        assert!(
            !naive_stripped.to_string_lossy().starts_with(r"\\"),
            "the old naive strip produces a non-absolute, invalid path for UNC input"
        );
    }

    #[test]
    #[cfg(windows)]
    fn test_validate_accepts_verbatim_prefixed_local_disk_path() {
        let dir = tempdir().expect("temp dir");
        // Resolve the temp root once up front (avoids 8.3 short-name
        // aliasing, which verbatim paths do not expand).
        let root = dunce::canonicalize(dir.path()).expect("canonicalize root");
        let guard = PathGuard::new(root.clone()).expect("guard");

        let safe_file = root.join("safe.txt");
        fs::write(&safe_file, "data").unwrap();

        let verbatim_input = PathBuf::from(format!(r"\\?\{}", safe_file.display()));
        let validated = guard
            .validate(&verbatim_input)
            .expect("a verbatim in-workspace path must validate");
        assert!(validated.starts_with(&root));
        assert!(
            !validated.to_string_lossy().starts_with(r"\\?\"),
            "the validated path should be in compatible (non-verbatim) form"
        );
    }

    #[test]
    #[cfg(windows)]
    fn test_validate_rejects_verbatim_prefixed_path_outside_workspace() {
        let dir = tempdir().expect("temp dir");
        let root = dunce::canonicalize(dir.path()).expect("canonicalize root");
        let guard = PathGuard::new(root).expect("guard");

        let outside = tempdir().expect("outside temp dir");
        let outside_root = dunce::canonicalize(outside.path()).expect("canonicalize outside");
        let outside_file = outside_root.join("secret.txt");
        fs::write(&outside_file, "nope").unwrap();

        let verbatim_outside = PathBuf::from(format!(r"\\?\{}", outside_file.display()));
        assert!(
            guard.validate(&verbatim_outside).is_err(),
            "a verbatim path entirely outside the workspace must be rejected"
        );
    }
}
