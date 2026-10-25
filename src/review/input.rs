// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What a review reviews: its input kinds and how they resolve to files and a content source.
//!
//! Every ref from the caller is validated ([`validate_ref`]) and pinned to the SHA `rev-parse`
//! returns before any other git call, so a branch that moves during the review cannot mix two
//! commits. Diffs are split per file on bytes, so one section that is not UTF-8 is set aside
//! without sinking the rest.

use std::path::{Component, Path, PathBuf};

use super::error::ReviewError;
use super::source::GitCommand;

/// What the caller asked to review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewInput {
    /// One file, reviewed whole from the worktree.
    File(PathBuf),
    /// A directory, every file reviewed whole from the worktree.
    Directory(PathBuf),
    /// One commit against its parent (`None` is `HEAD`; a root commit against the empty tree).
    Commit(Option<String>),
    /// The diff from `from` to `to`.
    Range {
        /// The old end.
        from: String,
        /// The new end.
        to: String,
    },
    /// The diff of `head` against its merge base with `base`.
    Branches {
        /// The base branch.
        base: String,
        /// The reviewed branch.
        head: String,
    },
    /// Staged, unstaged and untracked (not ignored) changes against `HEAD`.
    Uncommitted,
}

/// One file to review.
#[derive(Debug)]
pub struct ReviewFile {
    /// Path relative to the workspace, with `/` separators.
    pub path: String,
    /// Its own diff section, or `None` when it is reviewed whole.
    pub diff: Option<String>,
}

/// Where the content of a review comes from.
pub enum SourceKind {
    /// The worktree, with an optional old side (the SHA of `HEAD` for uncommitted changes).
    Worktree {
        /// The old side's commit.
        old_rev: Option<String>,
    },
    /// A git revision, never the worktree.
    Revision {
        /// The new side's commit.
        new_rev: String,
        /// The old side's commit (`None` for a root commit).
        old_rev: Option<String>,
    },
}

/// A resolved input.
pub struct ResolvedInput {
    /// The files to review, in the order git or the walk produced them.
    pub files: Vec<ReviewFile>,
    /// Where their content comes from.
    pub source_kind: SourceKind,
    /// Diff sections and file names that are not UTF-8, by their lossily decoded path; their
    /// content is never sent.
    pub not_utf8: Vec<String>,
}

/// The diff flags every diff of a review carries: no colour, no external driver, no textconv,
/// paths relative to the workspace (and only what lies under it).
const DIFF_FLAGS: &[&str] = &["--no-color", "--no-ext-diff", "--no-textconv", "--relative"];
/// What `ls-files` lists: tracked and untracked files git does not ignore, NUL-separated.
const LS_FILES_ALL: &[&str] = &[
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
    "-z",
];
/// The opening of git's header line for a binary section (`Binary files … differ`).
const BINARY_MARKER: &str = "Binary files ";
/// git's header line for a binary section of a `--binary` diff.
const BINARY_PATCH_MARKER: &str = "GIT binary patch";
/// The mode of a gitlink (a submodule): its "content" is a commit id, never reviewed.
const GITLINK_MODE: &str = "160000";

/// Rejects a ref git could read as an option, a range or several arguments.
///
/// # Errors
///
/// [`ReviewError::InvalidRef`] for an empty ref, one that starts with `-`, or one that contains
/// `..`, whitespace or a control character (NUL included).
///
/// # Examples
///
/// ```
/// use magi_rs::review::input::validate_ref;
///
/// assert!(validate_ref("feature/x").is_ok());
/// assert!(validate_ref("--output=x").is_err());
/// ```
pub fn validate_ref(r: &str) -> Result<(), ReviewError> {
    let hostile = r.is_empty()
        || r.starts_with('-')
        || r.contains("..")
        || r.chars().any(|c| c.is_whitespace() || c.is_control());
    if hostile {
        return Err(ReviewError::InvalidRef(r.escape_debug().to_string()));
    }
    Ok(())
}

/// Resolves an input to its files and its single content source.
///
/// # Arguments
///
/// * `input` - What to review.
/// * `workspace` - The workspace root; git runs there and every path is relative to it.
/// * `git` - The git runner.
/// * `guard` - The sandbox check for `File`/`Directory` paths; returns the resolved path.
///
/// # Returns
///
/// The files, the [`SourceKind`] to read them from and the sections set aside as not UTF-8.
/// Linear in the size of the diff or the listing.
///
/// # Errors
///
/// [`ReviewError::InvalidRef`] before any git call for a hostile ref,
/// [`ReviewError::RefNotFound`] for a ref that names no commit, [`ReviewError::OutsideWorkspace`]
/// from the guard, [`ReviewError::NotReviewable`] for a `File` that is a directory,
/// [`ReviewError::Git`] when git fails (inside a repository, a failing git never falls back to a
/// walk that would ignore `.gitignore`), [`ReviewError::Io`] for a failing walk, and
/// [`ReviewError::NothingToReview`] when nothing is left.
pub fn resolve_input(
    input: &ReviewInput,
    workspace: &Path,
    git: &dyn GitCommand,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let resolved = match input {
        ReviewInput::File(p) => resolve_file(p, workspace, guard)?,
        ReviewInput::Directory(d) => resolve_directory(d, workspace, git, guard)?,
        ReviewInput::Commit(r) => {
            let r = r.as_deref().unwrap_or("HEAD");
            validate_ref(r)?;
            let sha = verify(git, r)?;
            let old_rev = match git.run(&["rev-parse", "--verify", &format!("{r}^^{{commit}}")]) {
                Ok(parent) => Some(parent.trim().to_string()),
                Err(ReviewError::Git(_)) => None,
                Err(other) => return Err(other),
            };
            let mut args = vec!["diff-tree", "-p", "-r", "--root"];
            args.extend_from_slice(DIFF_FLAGS);
            args.push(&sha);
            from_diff(&git.run_bytes(&args)?, sha.clone(), old_rev)
        }
        ReviewInput::Range { from, to } => {
            validate_ref(from)?;
            validate_ref(to)?;
            let (old, new) = (verify(git, from)?, verify(git, to)?);
            from_diff(&diff(git, &format!("{old}..{new}"))?, new, Some(old))
        }
        ReviewInput::Branches { base, head } => {
            validate_ref(base)?;
            validate_ref(head)?;
            let (base_sha, head_sha) = (verify(git, base)?, verify(git, head)?);
            let merge_base = merge_base(git, base, head, &base_sha, &head_sha)?;
            let text = diff(git, &format!("{merge_base}..{head_sha}"))?;
            from_diff(&text, head_sha, Some(merge_base))
        }
        ReviewInput::Uncommitted => resolve_uncommitted(git, workspace)?,
    };
    if resolved.files.is_empty() && resolved.not_utf8.is_empty() {
        return Err(ReviewError::NothingToReview);
    }
    Ok(resolved)
}

/// `input` with every ref pinned to the commit it names now, so a review approved on one plan
/// runs on the same commits even if a branch or `HEAD` moves before it runs. A bare commit input
/// pins `HEAD`. Working-tree inputs (`File`, `Directory`, `Uncommitted`) have no ref and come
/// back unchanged.
///
/// # Arguments
///
/// * `input` - What the caller asked to review.
/// * `git` - The review's git.
///
/// # Returns
///
/// The input with full SHAs in place of its refs.
///
/// # Errors
///
/// [`ReviewError::InvalidRef`] before any git call for a hostile ref,
/// [`ReviewError::RefNotFound`] for a ref that names no commit, and any other error of `git`.
pub fn pin_refs(input: &ReviewInput, git: &dyn GitCommand) -> Result<ReviewInput, ReviewError> {
    let pin = |r: &str| -> Result<String, ReviewError> {
        validate_ref(r)?;
        verify(git, r)
    };
    Ok(match input {
        ReviewInput::Commit(r) => ReviewInput::Commit(Some(pin(r.as_deref().unwrap_or("HEAD"))?)),
        ReviewInput::Range { from, to } => ReviewInput::Range {
            from: pin(from)?,
            to: pin(to)?,
        },
        ReviewInput::Branches { base, head } => ReviewInput::Branches {
            base: pin(base)?,
            head: pin(head)?,
        },
        other => other.clone(),
    })
}

/// Whether `dir` or one of its ancestors holds a `.git` entry (a directory, or a file as in a
/// linked worktree). Decides "no repository" by the disk, never by the text of a git error.
fn inside_a_repository(dir: &Path) -> bool {
    dir.ancestors()
        .any(|a| std::fs::symlink_metadata(a.join(".git")).is_ok())
}

/// Whether the review's git runs inside a work tree (`rev-parse --is-inside-work-tree`).
///
/// # Arguments
///
/// * `git` - The review's git.
/// * `workspace` - The workspace, named in the error.
/// * `probe` - Where the `.git` search starts when git fails.
///
/// # Returns
///
/// `true` inside a work tree, `false` when git fails and no `.git` exists in `probe` or an
/// ancestor (there is no repository).
///
/// # Errors
///
/// [`ReviewError::Git`] when git answers anything but `true`, or fails while a `.git` exists
/// (treating that as "no repository" would bypass the repository's rules), and any other error
/// of `git`.
pub(crate) fn in_work_tree(
    git: &dyn GitCommand,
    workspace: &Path,
    probe: &Path,
) -> Result<bool, ReviewError> {
    match git.run(&["rev-parse", "--is-inside-work-tree"]) {
        Ok(out) if out.trim() == "true" => Ok(true),
        Ok(out) => Err(ReviewError::Git(format!(
            "{} is not inside a git work tree (git answered {:?})",
            workspace.display(),
            out.trim()
        ))),
        Err(ReviewError::Git(msg)) if inside_a_repository(probe) => Err(ReviewError::Git(msg)),
        Err(ReviewError::Git(_)) => Ok(false),
        Err(other) => Err(other),
    }
}

/// `rev-parse --verify <r>^{commit}`, trimmed: the SHA every later call uses.
fn verify(git: &dyn GitCommand, r: &str) -> Result<String, ReviewError> {
    match git.run(&["rev-parse", "--verify", &format!("{r}^{{commit}}")]) {
        Ok(sha) if !sha.trim().is_empty() => Ok(sha.trim().to_string()),
        Ok(_) | Err(ReviewError::Git(_)) => Err(ReviewError::RefNotFound(r.to_string())),
        Err(other) => Err(other),
    }
}

/// `diff <DIFF_FLAGS> <spec>`.
fn diff(git: &dyn GitCommand, spec: &str) -> Result<Vec<u8>, ReviewError> {
    let mut args = vec!["diff"];
    args.extend_from_slice(DIFF_FLAGS);
    args.push(spec);
    git.run_bytes(&args)
}

/// The merge base of two pinned SHAs, or an error that names both refs and what to do instead.
fn merge_base(
    git: &dyn GitCommand,
    base: &str,
    head: &str,
    base_sha: &str,
    head_sha: &str,
) -> Result<String, ReviewError> {
    let failure = |detail: String| {
        let git_says = if detail.trim().is_empty() {
            String::new()
        } else {
            format!(" ({})", detail.trim())
        };
        ReviewError::Git(format!(
            "{base} and {head} have no common ancestor to compare from; review them as a range \
             with --from {base} --to {head}{git_says}"
        ))
    };
    match git.run(&["merge-base", base_sha, head_sha]) {
        Ok(sha) if !sha.trim().is_empty() => Ok(sha.trim().to_string()),
        Ok(_) => Err(failure(String::new())),
        Err(ReviewError::Git(msg)) => Err(failure(msg)),
        Err(other) => Err(other),
    }
}

/// A `File` input: one path from the worktree, through the guard.
fn resolve_file(
    p: &Path,
    workspace: &Path,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let guarded = guard(&workspace.join(p))?;
    if std::fs::symlink_metadata(&guarded).is_ok_and(|m| m.is_dir()) {
        return Err(ReviewError::NotReviewable {
            path: p.display().to_string(),
            cause: "directory",
        });
    }
    let path = relative(workspace, &guarded)?;
    Ok(ResolvedInput {
        files: vec![ReviewFile { path, diff: None }],
        source_kind: SourceKind::Worktree { old_rev: None },
        not_utf8: Vec::new(),
    })
}

/// A `Directory` input: what git lists (tracked and untracked, not ignored) and still exists
/// as a non-directory, or, with no repository anywhere above it, an ordered walk of the disk.
fn resolve_directory(
    d: &Path,
    workspace: &Path,
    git: &dyn GitCommand,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let dir = guard(&workspace.join(d))?;
    let rel = relative(workspace, &dir)?;
    let mut resolved = ResolvedInput {
        files: Vec::new(),
        source_kind: SourceKind::Worktree { old_rev: None },
        not_utf8: Vec::new(),
    };
    if in_work_tree(git, workspace, &dir)? {
        let pathspec = if rel.is_empty() { "." } else { rel.as_str() };
        let mut args = LS_FILES_ALL.to_vec();
        args.extend_from_slice(&["--", pathspec]);
        let (names, not_utf8) = split_nul(&git.run_bytes(&args)?);
        resolved.not_utf8 = not_utf8;
        // A tracked file deleted from the tree, a link whose target is gone (which reads the
        // same) or a submodule (a directory) is not read.
        resolved.files = names
            .into_iter()
            .filter(|name| {
                let path = workspace.join(name);
                std::fs::symlink_metadata(&path).is_ok_and(|m| !m.is_dir())
                    && !is_dangling_link(&path)
            })
            .map(|path| ReviewFile { path, diff: None })
            .collect();
    } else {
        let mut found = Vec::new();
        walk(&dir, &rel, &mut found, &mut resolved.not_utf8)?;
        found.sort();
        resolved.files = found
            .into_iter()
            .map(|path| ReviewFile { path, diff: None })
            .collect();
    }
    Ok(resolved)
}

/// Walks `dir` (whose workspace-relative path is `rel`) without following symlinks, collecting
/// every non-directory entry except a link whose target is gone. O(entries).
fn walk(
    dir: &Path,
    rel: &str,
    found: &mut Vec<String>,
    not_utf8: &mut Vec<String>,
) -> Result<(), ReviewError> {
    let io = |e: std::io::Error| ReviewError::Io(format!("{}: {e}", dir.display()));
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let name = entry.file_name();
        let path = match (name.to_str(), rel.is_empty()) {
            (Some(n), true) => n.to_string(),
            (Some(n), false) => format!("{rel}/{n}"),
            (None, _) => {
                not_utf8.push(format!("{rel}/{}", name.to_string_lossy()));
                continue;
            }
        };
        if entry.file_type().map_err(io)?.is_dir() {
            walk(&entry.path(), &path, found, not_utf8)?;
        } else if !is_dangling_link(&entry.path()) {
            found.push(path);
        }
    }
    Ok(())
}

/// Uncommitted changes: the diff against the SHA of `HEAD` plus untracked files whole, or, in a
/// repository without commits, every tracked and untracked file whole.
///
/// An untracked link whose target is gone is not listed: it reads as a file that does not
/// exist (`workspace` is where the names resolve).
fn resolve_uncommitted(
    git: &dyn GitCommand,
    workspace: &Path,
) -> Result<ResolvedInput, ReviewError> {
    let present = |name: &String| !is_dangling_link(&workspace.join(name));
    match verify(git, "HEAD") {
        Ok(head) => {
            let mut resolved = from_diff(&diff(git, &head)?, String::new(), None);
            resolved.source_kind = SourceKind::Worktree {
                old_rev: Some(head),
            };
            let (names, not_utf8) =
                split_nul(&git.run_bytes(&["ls-files", "--others", "--exclude-standard", "-z"])?);
            resolved.not_utf8.extend(not_utf8);
            resolved.files.extend(
                names
                    .into_iter()
                    .filter(present)
                    .map(|path| ReviewFile { path, diff: None }),
            );
            Ok(resolved)
        }
        Err(ReviewError::RefNotFound(_)) => {
            let (names, not_utf8) = split_nul(&git.run_bytes(LS_FILES_ALL)?);
            Ok(ResolvedInput {
                files: names
                    .into_iter()
                    .filter(present)
                    .map(|path| ReviewFile { path, diff: None })
                    .collect(),
                source_kind: SourceKind::Worktree { old_rev: None },
                not_utf8,
            })
        }
        Err(other) => Err(other),
    }
}

/// Whether `path` is a link (or a junction) whose target does not exist. Such a link reads as a
/// file that does not exist, so an enumeration leaves it out the way it leaves out a tracked
/// file deleted from the tree. A path that does not exist at all is not a dangling link.
fn is_dangling_link(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
        && std::fs::metadata(path).is_err()
}

/// Splits NUL-separated names; the ones that are not UTF-8 come back apart, decoded lossily.
pub(crate) fn split_nul(bytes: &[u8]) -> (Vec<String>, Vec<String>) {
    let mut names = Vec::new();
    let mut not_utf8 = Vec::new();
    for name in bytes.split(|b| *b == 0).filter(|n| !n.is_empty()) {
        match std::str::from_utf8(name) {
            Ok(n) => names.push(n.to_string()),
            Err(_) => not_utf8.push(String::from_utf8_lossy(name).into_owned()),
        }
    }
    (names, not_utf8)
}

/// The workspace-relative, `/`-separated form of a guarded path.
fn relative(workspace: &Path, guarded: &Path) -> Result<String, ReviewError> {
    let outside = || ReviewError::OutsideWorkspace(guarded.display().to_string());
    let tail = guarded
        .strip_prefix(workspace)
        .ok()
        .map(Path::to_path_buf)
        .or_else(|| {
            let canonical = std::fs::canonicalize(workspace).ok()?;
            guarded.strip_prefix(canonical).ok().map(Path::to_path_buf)
        })
        .ok_or_else(outside)?;
    let mut parts = Vec::new();
    for component in tail.components() {
        match component {
            Component::Normal(part) => {
                parts.push(part.to_str().ok_or_else(|| ReviewError::NotReviewable {
                    path: guarded.display().to_string(),
                    cause: "not utf-8",
                })?)
            }
            Component::CurDir => {}
            _ => return Err(outside()),
        }
    }
    Ok(parts.join("/"))
}

/// A diff-based input: the per-file split of `bytes`, read from `new_rev` against `old_rev`.
fn from_diff(bytes: &[u8], new_rev: String, old_rev: Option<String>) -> ResolvedInput {
    let (files, not_utf8) = split_diff(bytes);
    ResolvedInput {
        files,
        source_kind: SourceKind::Revision { new_rev, old_rev },
        not_utf8,
    }
}

/// Splits a diff into one [`ReviewFile`] per `diff --git` section, on bytes. Sections that are
/// not UTF-8 go to the second list by their lossy path; sections with nothing to review (a mode
/// change, a pure rename, a submodule) are dropped. O(n) in the diff.
pub(crate) fn split_diff(bytes: &[u8]) -> (Vec<ReviewFile>, Vec<String>) {
    let mut starts = Vec::new();
    let mut offset = 0;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        if line.starts_with(b"diff --git ") {
            starts.push(offset);
        }
        offset += line.len();
    }
    let ends = starts.iter().skip(1).copied().chain([bytes.len()]);
    let mut files = Vec::new();
    let mut not_utf8 = Vec::new();
    for (start, end) in starts.iter().copied().zip(ends) {
        let Some(section) = bytes.get(start..end) else {
            continue;
        };
        match std::str::from_utf8(section) {
            Ok(text) => {
                if let Some(path) = reviewable_path(text) {
                    files.push(ReviewFile {
                        path,
                        diff: Some(text.to_string()),
                    });
                }
            }
            Err(_) => {
                if let Some(path) = reviewable_path(&String::from_utf8_lossy(section)) {
                    not_utf8.push(path);
                }
            }
        }
    }
    (files, not_utf8)
}

/// Whether git marks a diff section binary: a `Binary files … differ` or `GIT binary patch`
/// line among its header lines, before the first `@@`. The same line inside a hunk is text.
///
/// # Examples
///
/// ```
/// use magi_rs::review::input::is_binary_section;
///
/// assert!(is_binary_section("diff --git a/x b/x\nBinary files a/x and b/x differ\n"));
/// assert!(!is_binary_section("@@ -1 +1 @@\n+Binary files a/x and b/x differ\n"));
/// ```
pub fn is_binary_section(section: &str) -> bool {
    section
        .lines()
        .take_while(|l| !l.starts_with("@@"))
        .any(|l| l.starts_with(BINARY_MARKER) || l == BINARY_PATCH_MARKER)
}

/// The path of a section worth reviewing (it has hunks, or it is binary and gets listed), or
/// `None` for a gitlink, a header-only section with nothing to review, or an unparseable header.
fn reviewable_path(section: &str) -> Option<String> {
    let mut lines = section.lines();
    let first = lines.next()?;
    let header: Vec<&str> = lines.take_while(|l| !l.starts_with("@@")).collect();
    let has_hunk = section.lines().any(|l| l.starts_with("@@"));
    let gitlink = header.iter().any(|l| {
        (l.starts_with("index ") || l.contains("mode "))
            && l.split(' ').next_back() == Some(GITLINK_MODE)
    });
    if gitlink || !(has_hunk || is_binary_section(section)) {
        return None;
    }
    section_path(first, &header).filter(|p| !p.is_empty())
}

/// The path of a section: `rename to`, then `+++`, then `---` for a deletion, then the
/// symmetric split of the `diff --git` header.
fn section_path(first: &str, header: &[&str]) -> Option<String> {
    let after = |prefix: &str| header.iter().find_map(|l| l.strip_prefix(prefix));
    if let Some(p) = after("rename to ").and_then(unquote) {
        return Some(p);
    }
    let side = |line: Option<&str>| -> Option<String> {
        let raw = line?;
        let raw = raw.strip_suffix('\t').unwrap_or(raw);
        if raw == "/dev/null" {
            return None;
        }
        let path = unquote(raw)?;
        let path = path
            .strip_prefix("b/")
            .or_else(|| path.strip_prefix("a/"))?;
        (!path.is_empty()).then(|| path.to_string())
    };
    let new_side = after("+++ ");
    let deleted = new_side.is_some_and(|l| l.strip_suffix('\t').unwrap_or(l) == "/dev/null");
    let from_sides = if deleted {
        side(after("--- "))
    } else {
        side(new_side)
    };
    from_sides.or_else(|| header_path(first))
}

/// The path of a `diff --git <a> <b>` header whose two halves are the same path: the text has
/// odd length `2k + 1`, byte `k` is a space, and the halves differ only in `a/` against `b/`
/// (quoted or not). Never cut at the first or last ` b/`: a path may contain one.
fn header_path(first: &str) -> Option<String> {
    let rest = first.strip_prefix("diff --git ")?;
    let k = rest.len() / 2;
    if rest.len() % 2 == 0 || rest.as_bytes().get(k) != Some(&b' ') {
        return None;
    }
    let (left, right) = (rest.get(..k)?, rest.get(k + 1..)?);
    for (a, b) in [("a/", "b/"), ("\"a/", "\"b/")] {
        if let (Some(l), Some(r)) = (left.strip_prefix(a), right.strip_prefix(b)) {
            if l == r {
                return unquote(left)?.strip_prefix("a/").map(str::to_string);
            }
        }
    }
    None
}

/// Undoes git's C-style quoting (`"..."` with `\\ \" \a \b \t \n \v \f \r` and octal `\ooo`
/// bytes); an unquoted text comes back as is. `None` for a malformed quote or escape, or bytes
/// that are not UTF-8.
fn unquote(text: &str) -> Option<String> {
    let Some(inner) = text.strip_prefix('"') else {
        return Some(text.to_string());
    };
    let inner = inner.strip_suffix('"')?;
    let mut out = Vec::with_capacity(inner.len());
    let mut bytes = inner.bytes();
    while let Some(b) = bytes.next() {
        match b {
            b'"' => return None,
            b'\\' => {
                let escaped = match bytes.next()? {
                    b'\\' => b'\\',
                    b'"' => b'"',
                    b'a' => 0x07,
                    b'b' => 0x08,
                    b't' => b'\t',
                    b'n' => b'\n',
                    b'v' => 0x0b,
                    b'f' => 0x0c,
                    b'r' => b'\r',
                    d @ b'0'..=b'7' => {
                        let mut value = u32::from(d - b'0');
                        for _ in 0..2 {
                            let next = bytes.next().filter(|c| (b'0'..=b'7').contains(c))?;
                            value = value * 8 + u32::from(next - b'0');
                        }
                        u8::try_from(value).ok()?
                    }
                    _ => return None,
                };
                out.push(escaped);
            }
            other => out.push(other),
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::FakeGit;

    #[test]
    fn pinning_turns_every_ref_into_its_commit_and_leaves_the_working_tree_alone() {
        let git = FakeGit::from(&[
            (
                "rev-parse --verify HEAD^{commit}",
                Ok("aaa
"),
            ),
            (
                "rev-parse --verify main^{commit}",
                Ok("bbb
"),
            ),
            (
                "rev-parse --verify x^{commit}",
                Ok("ccc
"),
            ),
        ]);
        let pinned = |i: ReviewInput| pin_refs(&i, &git).expect("pins");
        assert_eq!(
            pinned(ReviewInput::Commit(None)),
            ReviewInput::Commit(Some("aaa".into()))
        );
        assert_eq!(
            pinned(ReviewInput::Range {
                from: "main".into(),
                to: "x".into()
            }),
            ReviewInput::Range {
                from: "bbb".into(),
                to: "ccc".into()
            }
        );
        assert_eq!(
            pinned(ReviewInput::Branches {
                base: "main".into(),
                head: "x".into()
            }),
            ReviewInput::Branches {
                base: "bbb".into(),
                head: "ccc".into()
            }
        );
        assert_eq!(pinned(ReviewInput::Uncommitted), ReviewInput::Uncommitted);
        assert_eq!(
            pinned(ReviewInput::File("src".into())),
            ReviewInput::File("src".into())
        );
        assert!(matches!(
            pin_refs(&ReviewInput::Commit(Some("--output=x".into())), &git),
            Err(ReviewError::InvalidRef(_))
        ));
        assert!(matches!(
            pin_refs(&ReviewInput::Commit(Some("gone".into())), &git),
            Err(ReviewError::RefNotFound(_))
        ));
    }

    fn allow_all(p: &Path) -> Result<PathBuf, ReviewError> {
        Ok(p.to_path_buf())
    }

    #[test]
    fn a_ref_starting_with_a_dash_is_rejected_before_git_runs() {
        for bad in ["--output=x", "-p", "", "a b", "a\0b", "a..b"] {
            assert!(
                matches!(validate_ref(bad), Err(ReviewError::InvalidRef(_))),
                "{bad:?}"
            );
        }
        assert!(validate_ref("HEAD~1").is_ok());
        assert!(validate_ref("feature/x").is_ok());
    }

    #[test]
    fn no_git_argument_defaults_to_head() {
        let git = FakeGit(
            [
                (
                    "rev-parse --verify HEAD^{commit}".to_string(),
                    Ok("abc\n".to_string()),
                ),
                (
                    "rev-parse --verify HEAD^^{commit}".to_string(),
                    Ok("p\n".to_string()),
                ),
                (
                    "diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative abc"
                        .to_string(),
                    Ok(
                        "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-x\n+y\n"
                            .to_string(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        );
        let r = resolve_input(
            &ReviewInput::Commit(None),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert_eq!(r.files[0].path, "a.rs");
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "abc" && old_rev.as_deref() == Some("p"))
        ); // [CP2 MS2 S1 loop 7] pinned SHAs
    }

    #[test]
    fn a_root_commit_is_reviewed_against_the_empty_tree() {
        let git = FakeGit::from(&[
            ("rev-parse --verify HEAD^{commit}", Ok("r\n")),
            ("rev-parse --verify HEAD^^{commit}", Err("fatal: no parent")),
            ("diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative r",
             Ok("diff --git a/a.rs b/a.rs\nnew file mode 100644\n--- /dev/null\n+++ b/a.rs\n@@ -0,0 +1 @@\n+fn f() {}\n")),
        ]);
        let r = resolve_input(
            &ReviewInput::Commit(None),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("a root commit is reviewable");
        assert_eq!(r.files.len(), 1);
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref old_rev, .. } if old_rev.is_none())
        );
    }

    #[test]
    fn an_unknown_ref_is_a_typed_error() {
        let git = FakeGit(
            [(
                "rev-parse --verify nope^{commit}".to_string(),
                Err("fatal".to_string()),
            )]
            .into_iter()
            .collect(),
        );
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(Some("nope".into())),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::RefNotFound(_))
        ));
    }

    #[test]
    fn a_path_outside_the_workspace_is_rejected_by_the_guard() {
        let deny = |p: &Path| -> Result<PathBuf, ReviewError> {
            Err(ReviewError::OutsideWorkspace(p.display().to_string()))
        };
        let git = FakeGit(Default::default());
        assert!(matches!(
            resolve_input(
                &ReviewInput::File("../fuera".into()),
                Path::new("/w"),
                &git,
                &deny
            ),
            Err(ReviewError::OutsideWorkspace(_))
        ));
    }

    // [CP2 MS2 S1 loop 1] the other RV01 kinds, and the diff-to-file split.
    const ONE_HUNK: &str =
        "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-x\n+y\n";

    #[test]
    fn a_range_is_reviewed_from_its_right_end_against_its_left_end() {
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(ONE_HUNK),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert!(r.files[0].diff.is_some());
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "b" && old_rev.as_deref() == Some("a"))
        );
    }

    #[test]
    fn branches_are_reviewed_from_their_merge_base() {
        let git = FakeGit::from(&[
            ("rev-parse --verify main^{commit}", Ok("a\n")),
            ("rev-parse --verify feature/x^{commit}", Ok("b\n")),
            ("merge-base a b", Ok("mb1\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative mb1..b",
                Ok(ONE_HUNK),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Branches {
                base: "main".into(),
                head: "feature/x".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "b" && old_rev.as_deref() == Some("mb1"))
        );
    }

    #[test]
    fn branches_without_a_common_ancestor_are_a_typed_error_that_says_what_to_do() {
        // [CP2 MS2 S1a loop 11] unrelated histories: `merge-base` exits 1 with no output, never an empty message.
        let git = FakeGit::from(&[
            ("rev-parse --verify main^{commit}", Ok("a\n")),
            ("rev-parse --verify orphan^{commit}", Ok("b\n")),
            ("merge-base a b", Err("")),
        ]);
        match resolve_input(
            &ReviewInput::Branches {
                base: "main".into(),
                head: "orphan".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        ) {
            Err(ReviewError::Git(msg)) => assert!(
                msg.contains("main") && msg.contains("orphan") && msg.contains("--from"),
                "{msg}"
            ),
            // [RED-DRYRUN 2026-10-19] ResolvedInput is not Debug: only the error side is formatted.
            Err(other) => panic!("expected a Git error, got {other:?}"),
            Ok(_) => panic!("expected a Git error, got a resolved input"),
        }
    }

    #[test]
    fn an_unknown_range_end_is_a_typed_error() {
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify nope^{commit}", Err("fatal")),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Range {
                    from: "v1".into(),
                    to: "nope".into()
                },
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::RefNotFound(_))
        ));
    }

    #[test]
    fn uncommitted_changes_are_the_diff_against_head_plus_untracked_files_whole() {
        let git = FakeGit::from(&[
            ("rev-parse --verify HEAD^{commit}", Ok("abc\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(ONE_HUNK),
            ),
            (
                "ls-files --others --exclude-standard -z",
                Ok("new.rs\0dir/other.rs\0"),
            ),
        ]);
        let r = resolve_input(&ReviewInput::Uncommitted, Path::new("/w"), &git, &allow_all)
            .expect("resolves");
        let got: Vec<(&str, bool)> = r
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.diff.is_some()))
            .collect();
        assert_eq!(
            got,
            vec![("a.rs", true), ("new.rs", false), ("dir/other.rs", false)]
        );
        // [CP2 MS2 S1 loop 2] the old side of an uncommitted diff is HEAD.
        assert!(
            matches!(r.source_kind, SourceKind::Worktree { ref old_rev } if old_rev.as_deref() == Some("abc"))
        );
    }

    #[test]
    fn uncommitted_changes_in_a_repo_without_commits_have_no_old_side() {
        let git = FakeGit::from(&[
            (
                "rev-parse --verify HEAD^{commit}",
                Err("fatal: needed a single revision"),
            ),
            // [CP2 MS2 S1 loop 3] staged files (`git add`, no commit yet) are --cached, and must not be lost.
            (
                "ls-files --cached --others --exclude-standard -z",
                Ok("staged.rs\0new.rs\0"),
            ),
        ]);
        let r = resolve_input(&ReviewInput::Uncommitted, Path::new("/w"), &git, &allow_all)
            .expect("resolves");
        let got: Vec<(&str, bool)> = r
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.diff.is_none()))
            .collect();
        assert_eq!(got, vec![("staged.rs", true), ("new.rs", true)]);
        assert!(matches!(
            r.source_kind,
            SourceKind::Worktree { old_rev: None }
        ));
    }

    #[test]
    fn a_hostile_ref_is_rejected_by_resolve_input_before_any_git_call() {
        // [CP2 MS2 S1 loop 2] FakeGit has no entries: any git call is a `Git` error, so only validate_ref yields InvalidRef.
        let git = FakeGit(Default::default());
        for input in [
            ReviewInput::Commit(Some("--output=x".into())),
            ReviewInput::Range {
                from: "v1".into(),
                to: "-p".into(),
            },
            ReviewInput::Branches {
                base: "--upload-pack=x".into(),
                head: "main".into(),
            },
        ] {
            assert!(
                matches!(
                    resolve_input(&input, Path::new("/w"), &git, &allow_all),
                    Err(ReviewError::InvalidRef(_))
                ),
                "{input:?}"
            );
        }
    }

    #[test]
    fn a_diff_section_that_is_not_utf8_is_set_aside_and_the_rest_is_reviewed() {
        struct BytesGit(Vec<u8>);
        impl GitCommand for BytesGit {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "rev-parse --verify v1^{commit}" | "rev-parse --verify v2^{commit}" => {
                        Ok(b"x\n".to_vec())
                    }
                    "diff --no-color --no-ext-diff --no-textconv --relative x..x" => {
                        Ok(self.0.clone())
                    }
                    other => Err(ReviewError::Git(format!("unexpected: {other}"))),
                }
            }
        }
        let mut diff = b"diff --git a/latin1.rs b/latin1.rs\n--- a/latin1.rs\n+++ b/latin1.rs\n@@ -1 +1 @@\n-x\n+caf\xe9\n".to_vec();
        diff.extend_from_slice(ONE_HUNK.as_bytes());
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &BytesGit(diff),
            &allow_all,
        )
        .expect("one bad section does not sink the review");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert_eq!(r.not_utf8, vec!["latin1.rs".to_string()]);
    }

    #[test]
    fn a_merge_commit_with_an_empty_diff_is_nothing_to_review() {
        let git = FakeGit::from(&[
            ("rev-parse --verify M^{commit}", Ok("m\n")),
            ("rev-parse --verify M^^{commit}", Ok("p\n")),
            (
                "diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative m",
                Ok(""),
            ),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(Some("M".into())),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::NothingToReview)
        ));
    }

    #[test]
    fn a_file_input_that_names_a_directory_is_not_reviewable() {
        // [CP2 MS2 S1a loop 11] the T2.3/T2.4 seam: a directory sent as a file is typed, not an opaque I/O error.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        let git = FakeGit(Default::default());
        assert!(matches!(
            resolve_input(&ReviewInput::File("src".into()), d.path(), &git, &allow_all),
            Err(ReviewError::NotReviewable {
                cause: "directory",
                ..
            })
        ));
    }

    #[test]
    fn a_file_input_is_reviewed_whole_from_the_worktree() {
        let git = FakeGit(Default::default());
        let r = resolve_input(
            &ReviewInput::File("src/a.rs".into()),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert_eq!(
            (r.files[0].path.as_str(), r.files[0].diff.is_none()),
            ("src/a.rs", true)
        );
        assert!(matches!(
            r.source_kind,
            SourceKind::Worktree { old_rev: None }
        ));
    }

    #[test]
    fn a_directory_in_a_repo_lists_what_git_does_not_ignore_and_that_exists() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        std::fs::write(d.path().join("src/b.rs"), "fn b() {}\n").expect("write");
        std::fs::create_dir_all(d.path().join("src/vendored")).expect("mkdir"); // [CP2 MS2 S1 loop 8] a submodule
                                                                                // `src/gone.rs` is tracked but deleted from the tree; `.env` is ignored, so ls-files never names it.
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- src",
                Ok("src/a.rs\0src/b.rs\0src/gone.rs\0src/vendored\0"),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs", "src/b.rs"]
        );
        assert!(r.files.iter().all(|f| f.diff.is_none()));
    }

    #[test]
    fn the_workspace_itself_as_a_directory_uses_the_dot_pathspec() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- .",
                Ok("a.rs\0"),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory(".".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
    }

    #[test]
    fn a_directory_outside_a_repo_is_walked_in_order() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src/sub")).expect("mkdir");
        std::fs::write(d.path().join("src/b.rs"), "b\n").expect("write");
        std::fs::write(d.path().join("src/sub/a.rs"), "a\n").expect("write");
        let git = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        // [CP2 MS2 S1b loop 13] a `.git` above TMPDIR would make the production rule (correctly) fail closed.
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        let r = resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/b.rs", "src/sub/a.rs"]
        );
    }

    #[test]
    fn a_git_failure_inside_a_repository_fails_closed_instead_of_walking() {
        // [CP2 MS2 S1a loop 11] a broken git (missing binary, dubious ownership) must not drop .gitignore in silence.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join(".git")).expect("mkdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        let git = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: detected dubious ownership"),
        )]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Directory("src".into()),
                d.path(),
                &git,
                &allow_all
            ),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_multi_file_diff_is_split_by_its_real_paths() {
        let diff = concat!(
            "diff --git a/src/añadir módulo.rs b/src/añadir módulo.rs\n",
            "--- a/src/añadir módulo.rs\t\n+++ b/src/añadir módulo.rs\t\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/old_name.rs b/new_name.rs\nsimilarity index 90%\nrename from old_name.rs\nrename to new_name.rs\n",
            "--- a/old_name.rs\n+++ b/new_name.rs\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/gone.rs b/gone.rs\ndeleted file mode 100644\n--- a/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            "diff --git a/logo.png b/logo.png\nBinary files a/logo.png and b/logo.png differ\n",
            "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n",
            "diff --git \"a/a\\tb.rs\" \"b/a\\tb.rs\"\n--- \"a/a\\tb.rs\"\n+++ \"b/a\\tb.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
            // [CP2 MS2 S1 loop 8] a submodule bump is not reviewed; a named escape (`\v`) is decoded.
            "diff --git a/sub b/sub\nindex 1111111..2222222 160000\n--- a/sub\n+++ b/sub\n@@ -1 +1 @@\n",
            "-Subproject commit 1111111\n+Subproject commit 2222222\n",
            "diff --git \"a/b\\vc.rs\" \"b/b\\vc.rs\"\n--- \"a/b\\vc.rs\"\n+++ \"b/b\\vc.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
        );
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(diff),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec![
                "src/añadir módulo.rs",
                "new_name.rs",
                "gone.rs",
                "logo.png",
                "a\tb.rs",
                "b\u{b}c.rs"
            ]
        );
        // Each file carries only its own section.
        assert!(r.files[2]
            .diff
            .as_deref()
            .is_some_and(|t| t.contains("-x") && !t.contains("+y")));
    }

    #[test]
    fn no_prefix_of_a_hostile_diff_panics_the_splitter() {
        // [CP2 MS2 S1a loop 9] release is panic=abort: the diff split must survive any truncation of hostile input.
        let hostile = concat!(
            "diff --git \"a/x\\",
            "\ndiff --git a/ b/\n+++ \n--- \"\n",
            "diff --git a/s b/s\nindex 1..2 160000\n",
            "diff --git a/b.bin b/b.bin\nGIT binary patch\nliteral 12\nzcm\n",
            "diff --git \"a/\\9\" b/q\nrename to \"\\9\n",
            "+++ b/\t\n",
            "@@ -1,99999999999 +1 @@\r\n+x\r\n"
        );
        for end in hostile
            .char_indices()
            .map(|(i, _)| i)
            .chain([hostile.len()])
        {
            let git = FakeGit::from(&[
                ("rev-parse --verify v1^{commit}", Ok("a\n")),
                ("rev-parse --verify v2^{commit}", Ok("b\n")),
                (
                    "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                    Ok(&hostile[..end]),
                ),
            ]);
            let _ = resolve_input(
                &ReviewInput::Range {
                    from: "v1".into(),
                    to: "v2".into(),
                },
                Path::new("/w"),
                &git,
                &allow_all,
            );
        }
    }

    #[test]
    fn a_header_only_section_takes_its_path_from_the_symmetric_split() {
        // [CP2 MS2 S1a loop 10, headers as git 2.56 writes them] binary sections carry only `diff --git`; the halves are equal.
        let diff = concat!(
            "diff --git a/lib b/util.so b/lib b/util.so\nindex 1..2 100644\nBinary files a/lib b/util.so and b/lib b/util.so differ\n",
            "diff --git a/my file.bin b/my file.bin\nindex 1..2 100644\nBinary files a/my file.bin and b/my file.bin differ\n",
            "diff --git \"a/q\\tz.bin\" \"b/q\\tz.bin\"\nindex 1..2 100644\nBinary files \"a/q\\tz.bin\" and \"b/q\\tz.bin\" differ\n",
            "diff --git a/old name.rs b/new name.rs\nsimilarity index 90%\nrename from old name.rs\nrename to new name.rs\n",
            "--- a/old name.rs\t\n+++ b/new name.rs\t\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/x b/y\nBinary files a/x and b/y differ\n", // asymmetric, no rename: git never writes it
        );
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(diff),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["lib b/util.so", "my file.bin", "q\tz.bin", "new name.rs"]
        );
    }
}
