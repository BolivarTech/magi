// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What a review reviews: its input kinds and how they resolve to files and a content source.
//!
//! Every ref from the caller is validated ([`validate_ref`]) and pinned to the SHA `rev-parse`
//! returns before any other git call, so a branch that moves during the review cannot mix two
//! commits. Diffs are split per file on bytes, so one section that is not UTF-8 is set aside
//! without sinking the rest.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use super::error::ReviewError;
use super::source::{is_never_reviewed, GitCommand};

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
    /// Listed names that are links whose target is gone. They read as files that do not exist,
    /// so they are not reviewed (like a tracked file deleted from the tree); [`resolve_input`]
    /// names them as `not a regular file` when nothing else is left to review.
    pub dangling_links: Vec<String>,
}

/// The diff flags every diff of a review carries: no colour, no external driver, no textconv,
/// paths relative to the workspace (and only what lies under it).
const DIFF_FLAGS: &[&str] = &["--no-color", "--no-ext-diff", "--no-textconv", "--relative"];
/// The git setting that reads `.gitattributes` from a tree instead of the working tree (git
/// 2.46 or later; an older git ignores an unknown setting and keeps the working tree's).
const ATTR_TREE_SETTING: &str = "attr.tree";
/// What `ls-files` lists: tracked and untracked files git does not ignore, NUL-separated.
const LS_FILES_ALL: &[&str] = &[
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
    "-z",
];
/// What `ls-files` names of what `.gitignore` excludes, an ignored directory once with a
/// trailing `/`, NUL-separated.
const LS_FILES_IGNORED_DIRECTORIES: &[&str] = &[
    "ls-files",
    "--others",
    "--ignored",
    "--exclude-standard",
    "--directory",
    "-z",
];
/// The opening of git's header line for a binary section (`Binary files … differ`).
const BINARY_MARKER: &str = "Binary files ";
/// git's header line for a binary section of a `--binary` diff.
const BINARY_PATCH_MARKER: &str = "GIT binary patch";
/// The mode of a gitlink (a submodule): its "content" is a commit id, never reviewed.
const GITLINK_MODE: &str = "160000";
/// The extended header lines whose last word is a file mode (`index` ends in one when the mode
/// did not change); a path-bearing line (`---`, `+++`, `rename to`, …) never counts.
const MODE_HEADER_PREFIXES: &[&str] = &[
    "index ",
    "old mode ",
    "new mode ",
    "new file mode ",
    "deleted file mode ",
];

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
/// for a listed name that climbs out of the workspace or is absolute, or
/// from the guard, [`ReviewError::NotReviewable`] for a `File` that is a directory or a
/// `Directory` that does not exist (`missing`), [`ReviewError::Git`] when git fails (inside a repository, a failing git never falls back to a
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
            // `<sha>^^{commit}` is the first parent, peeled to a commit (`^` then `^{commit}`), of
            // the pinned SHA; git's "no such commit" there is a root commit or a shallow edge, which
            // `ensure_root_commit` tells apart.
            let old_rev = git
                .query(&[
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    &format!("{sha}^^{{commit}}"),
                ])?
                .map(|parent| parent.trim().to_string());
            // The diff is taken against that pinned parent, so it matches `old_rev`: `diff-tree
            // <sha>` prints nothing for a merge commit. Only a root commit needs `--root`.
            let text = match &old_rev {
                Some(parent) => diff(git, parent, &sha)?,
                None => {
                    ensure_root_commit(git, &sha)?;
                    let attributes = attributes_of(&sha);
                    let mut args = vec!["-c", &attributes, "diff-tree", "-p", "-r", "--root"];
                    args.extend_from_slice(DIFF_FLAGS);
                    args.push(&sha);
                    git.run_bytes(&args)?
                }
            };
            from_diff(&text, sha.clone(), old_rev)
        }
        ReviewInput::Range { from, to } => {
            validate_ref(from)?;
            validate_ref(to)?;
            let (old, new) = (verify(git, from)?, verify(git, to)?);
            from_diff(&diff(git, &old, &new)?, new, Some(old))
        }
        ReviewInput::Branches { base, head } => {
            validate_ref(base)?;
            validate_ref(head)?;
            let (base_sha, head_sha) = (verify(git, base)?, verify(git, head)?);
            let merge_base = merge_base(git, base, head, &base_sha, &head_sha)?;
            let text = diff(git, &merge_base, &head_sha)?;
            from_diff(&text, head_sha, Some(merge_base))
        }
        ReviewInput::Uncommitted => resolve_uncommitted(git, workspace)?,
    };
    // MAGI FIX (S6 loop 6): one choke point for every listing. git scopes `ls-files` and
    // `diff --relative` to the workspace, but a name that climbs out or is absolute is refused
    // here, before any read, whatever produced it.
    if let Some(escaping) = resolved.files.iter().find(|f| !is_confined(&f.path)) {
        return Err(ReviewError::OutsideWorkspace(escaping.path.clone()));
    }
    // MAGI FIX (S6 loop 10): A1-m2, the old side of a rename or copy is read at `renamed_from`,
    // so that path goes through the same check (git's `--relative` never emits one that climbs
    // out, pinned by `a_rename_from_outside_a_sub_workspace_is_reviewed_as_an_addition`).
    if let Some(escaping) = resolved
        .files
        .iter()
        .filter_map(|f| f.diff.as_deref().and_then(renamed_from))
        .find(|old| !is_confined(old))
    {
        return Err(ReviewError::OutsideWorkspace(escaping));
    }
    // MAGI FIX (S6 loop 6): with only non-UTF-8 names left, the input went on with no file and
    // the omissions never reached the `nothing to review` error (RV20).
    // MAGI FIX (S6 loop 8): the dangling links every listing leaves out are named too.
    if resolved.files.is_empty() {
        use super::filter::{SkipCause, Skipped};
        let named = |paths: &[String], cause| {
            paths
                .iter()
                .map(move |path| Skipped {
                    path: super::secrets::display_path(path),
                    cause,
                })
                .collect::<Vec<_>>()
        };
        let mut skipped = named(&resolved.not_utf8, SkipCause::NotUtf8);
        skipped.extend(named(&resolved.dangling_links, SkipCause::NotRegularFile));
        return Err(ReviewError::NothingToReview { skipped });
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
/// linked worktree), or one that cannot be read ([`is_git_evidence`]). Decides "no repository" by the disk, never by the text of a git error.
fn inside_a_repository(dir: &Path) -> bool {
    dir.ancestors()
        .any(|a| is_git_evidence(&std::fs::symlink_metadata(a.join(".git")).map(|_| ())))
}

/// Whether a `.git` probe is evidence of a repository: anything but "not found". An entry that
/// exists but cannot be read is a repository, so a failing git there fails closed instead of
/// falling back to a walk that would lose the repository's exclude rules.
// MAGI FIX (S6 loop 6): a permission error on `.git` read as "no repository".
fn is_git_evidence(probe: &std::io::Result<()>) -> bool {
    !matches!(probe, Err(e) if e.kind() == std::io::ErrorKind::NotFound)
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
///
/// # Not a defect: `false` is not "no repository"
///
/// Outside any repository git does not answer `false`: it exits 128 ("not a git repository"),
/// which is the `Err` arm, so the caller reaches its walk. git answers `false` with exit 0
/// only inside a git directory (a bare repository, or a `.git` itself); walking there would
/// read the repository's internals, so that answer fails closed. Measured with git 2.56 and
/// pinned with the real git by `a_directory_outside_any_repository_is_walked_through_the_real_git`
/// and `git_answers_false_only_inside_a_git_directory_where_the_review_fails_closed` (mutating
/// this arm to `Ok(false)` walks the bare repository). S6 loop 7, A1-c1.
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

/// `rev-parse --verify --quiet <r>^{commit}`, trimmed: the SHA every later call uses.
///
/// Only git's "no such commit" answer is [`ReviewError::RefNotFound`]; any other failure (a
/// timeout, a broken repository) stays the error git gave.
fn verify(git: &dyn GitCommand, r: &str) -> Result<String, ReviewError> {
    match git.query(&[
        "rev-parse",
        "--verify",
        "--quiet",
        &format!("{r}^{{commit}}"),
    ])? {
        Some(sha) if !sha.trim().is_empty() => Ok(sha.trim().to_string()),
        _ => Err(ReviewError::RefNotFound(r.to_string())),
    }
}

/// The header prefix of a parent line in a raw commit object.
const PARENT_HEADER: &str = "parent ";

/// Checks that `sha` is a true root commit before it is reviewed against the empty tree.
///
/// git's "no such commit" for `<sha>^` also covers a shallow edge and a parent object missing
/// from the repository; the raw object (`cat-file commit`, whose headers end at the first empty
/// line) says whether the commit names a parent at all. O(n) in the object's headers.
///
/// # Errors
///
/// [`ReviewError::Git`] when the commit names a parent the repository cannot provide (a shallow
/// clone, a missing object), and whatever reading the object fails with.
// MAGI FIX (S6 loop 5): a shallow tip was reviewed with `--root`, its whole tree as added.
fn ensure_root_commit(git: &dyn GitCommand, sha: &str) -> Result<(), ReviewError> {
    let object = git.run_bytes(&["cat-file", "commit", sha])?;
    let has_parent = object
        .split(|b| *b == b'\n')
        .take_while(|line| !line.is_empty())
        .any(|line| line.starts_with(PARENT_HEADER.as_bytes()));
    if has_parent {
        return Err(ReviewError::Git(format!(
            "the parent of commit {sha} is not available (a shallow clone or a missing object); \
             fetch it, or review a range whose commits are all present"
        )));
    }
    Ok(())
}

/// `-c attr.tree=<new> diff <DIFF_FLAGS> <old>..<new>`: the diff between two pinned SHAs, with
/// `.gitattributes` read from the reviewed revision.
///
/// A revision diff otherwise reads the attributes of the working tree, so a later commit (or a
/// local edit) marking a path `-diff` or `binary` turned the reviewed change into "Binary files
/// differ" and left the file out, and the same pinned SHAs could plan and run differently.
/// Only `.gitattributes` moves to the tree: `$GIT_DIR/info/attributes` and `core.attributesFile`
/// are the operator's own settings and still apply. Uncommitted reviews keep the working tree's.
///
/// Declared, not isolated further (S6 loop 9, A1-c3): a git older than 2.46 ignores
/// `attr.tree` and reads the working tree's `.gitattributes`, and every git reads
/// `$GIT_DIR/info/attributes`. Neither changes which revision's bytes are diffed (RV21 holds):
/// an attribute only decides whether a section shows as binary, so the worst case is the file
/// left out as `binary`, which the report names. Refusing revision reviews below 2.46 would cut
/// supported gits for no gain in what is sent. Pinned by
/// `a_non_revision_attribute_only_hides_a_revision_section_never_changes_its_bytes`.
///
/// # Errors
///
/// Any error of `git`.
// MAGI FIX (S6 loop 7): the attribute source was the working tree, not the reviewed revision.
fn diff(git: &dyn GitCommand, old: &str, new: &str) -> Result<Vec<u8>, ReviewError> {
    let attributes = attributes_of(new);
    let spec = format!("{old}..{new}");
    let mut args = vec!["-c", &attributes, "diff"];
    args.extend_from_slice(DIFF_FLAGS);
    args.push(&spec);
    git.run_bytes(&args)
}

/// `diff <DIFF_FLAGS> <head>`: the working tree against the pinned `head`, with the working
/// tree's `.gitattributes` (the content reviewed is the working tree's, so are its attributes).
///
/// An unmerged path is reviewed too: a diff against a commit compares that tree with the
/// working tree and prints a plain `diff --git` section with the conflict markers (measured
/// with git 2.56); only the index-vs-worktree form, never used here, prints a combined
/// `diff --cc` that [`split_diff`] would drop. A review of S6 loop 9 (A1-c2) read it otherwise;
/// pinned by `an_uncommitted_review_keeps_an_unmerged_path_with_its_conflict_markers`.
///
/// # Errors
///
/// Any error of `git`.
fn worktree_diff(git: &dyn GitCommand, head: &str) -> Result<Vec<u8>, ReviewError> {
    let mut args = vec!["diff"];
    args.extend_from_slice(DIFF_FLAGS);
    args.push(head);
    git.run_bytes(&args)
}

/// The `-c` value that reads `.gitattributes` from the tree of commit `sha`.
fn attributes_of(sha: &str) -> String {
    format!("{ATTR_TREE_SETTING}={sha}")
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
///
/// The shape is read through links, whatever `guard` resolved: a link to a directory is a
/// `directory` and a link whose target is gone is `missing`. The binary's guard
/// (`PathGuard::validate`, wrapped by `review_path_guard` in the binary) canonicalizes, so
/// there a link to a directory arrives as the directory; a dangling link arrives as itself under
/// its guarded parent (S6 loop 9, A1-c5) and is `missing` here.
///
/// The path is resolved, not read: the one read happens in the content source, which checks it
/// again at that moment, so a swap between this resolution and the read is refused there (a link
/// swapped in, a FIFO before or after the open, a hard link out, a protected target:
/// `a_file_swapped_between_the_check_and_the_open_is_refused`,
/// `a_file_swapped_for_a_fifo_before_the_open_is_refused_without_blocking`,
/// `a_file_swapped_for_a_fifo_after_the_open_is_refused_without_blocking`,
/// `a_hard_link_to_a_file_outside_the_workspace_is_refused`, `src/review/source.rs`). An explicit
/// FIFO, socket or device reaches that reader and is refused without blocking
/// (`an_explicit_fifo_or_missing_file_is_a_typed_error_without_blocking`,
/// `src/system/review_git.rs`).
// MAGI FIX (S6 loop 10): A1-c2/A1-c3 refuted; the read-time checks they ask for are pinned.
fn resolve_file(
    p: &Path,
    workspace: &Path,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let guarded = guard(&workspace.join(p))?;
    // MAGI FIX (S6 loop 8): `symlink_metadata` saw the link, not its target, so a link to a
    // directory and a dangling link came back as files to read.
    let not_reviewable = |cause| ReviewError::NotReviewable {
        path: p.display().to_string(),
        cause,
    };
    if is_dangling_link(&guarded) {
        return Err(not_reviewable(super::filter::MISSING));
    }
    if std::fs::metadata(&guarded).is_ok_and(|m| m.is_dir()) {
        return Err(not_reviewable("directory"));
    }
    let path = relative(workspace, &guarded)?;
    Ok(ResolvedInput {
        files: vec![ReviewFile { path, diff: None }],
        source_kind: SourceKind::Worktree { old_rev: None },
        not_utf8: Vec::new(),
        dangling_links: Vec::new(),
    })
}

/// A `Directory` input: what git lists (tracked and untracked, not ignored) and still exists
/// as a non-directory, or, with no repository anywhere above it, an ordered walk of the disk
/// that leaves out what `.gitignore` excludes (git names it through an empty repository of the
/// review's own, so the rules apply as they would inside one: RV22).
fn resolve_directory(
    d: &Path,
    workspace: &Path,
    git: &dyn GitCommand,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let dir = guard(&workspace.join(d))?;
    // MAGI FIX (S6 loop 4): a link whose target is gone is missing too, not an I/O error.
    if std::fs::symlink_metadata(&dir).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        || is_dangling_link(&dir)
    {
        return Err(ReviewError::NotReviewable {
            path: d.display().to_string(),
            cause: super::filter::MISSING,
        });
    }
    // MAGI FIX (S6 loop 3): in a repository `ls-files -- <file>` lists the file itself, so a
    // regular file passed as a directory was reviewed; the shape is checked before git runs.
    match std::fs::metadata(&dir) {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            return Err(ReviewError::NotReviewable {
                path: d.display().to_string(),
                cause: NOT_A_DIRECTORY,
            })
        }
        Err(e) => return Err(ReviewError::Io(format!("{}: {e}", d.display()))),
    }
    let rel = relative(workspace, &dir)?;
    let mut resolved = ResolvedInput {
        files: Vec::new(),
        source_kind: SourceKind::Worktree { old_rev: None },
        not_utf8: Vec::new(),
        dangling_links: Vec::new(),
    };
    let pathspec = if rel.is_empty() { "." } else { rel.as_str() };
    if in_work_tree(git, workspace, &dir)? {
        let mut args = LS_FILES_ALL.to_vec();
        args.extend_from_slice(&["--", pathspec]);
        let (names, not_utf8) = split_nul(&git.run_bytes(&args)?);
        resolved.not_utf8 = not_utf8;
        // `.git`, `.magi` and the memory database are not pruned here, unlike the walk: `ls-files`
        // never enters `.git`, and the filter refuses every never-reviewed path from any listing
        // before a read (one enforcement point, RV22 clause 1; a review of S6 loop 8 asked for a
        // second one here). Pinned, with a tracked memory database and an unignored `.magi/` file, by
        // `a_repository_listing_never_keeps_a_never_reviewed_path`.
        // A tracked file deleted from the tree is not read, and a link whose target is gone
        // (which reads the same) is set aside.
        // MAGI FIX (S6 loop 8): any other metadata error dropped the name in silence; it is kept,
        // and the reader fails naming the path and the error. A submodule (a directory) is kept
        // too: the reader refuses it and the filter lists it as `not a regular file`.
        let present = names
            .into_iter()
            .filter(|name| {
                std::fs::symlink_metadata(workspace.join(name))
                    .map_or_else(|e| e.kind() != std::io::ErrorKind::NotFound, |_| true)
            })
            .collect();
        resolved.files = without_dangling(present, workspace, &mut resolved.dangling_links);
    } else {
        let mut args = LS_FILES_IGNORED_DIRECTORIES.to_vec();
        args.extend_from_slice(&["--", pathspec]);
        let (ignored, _not_utf8) = split_nul(&git.run_bytes_without_repository(&args)?);
        let ignored: HashSet<String> = ignored.into_iter().collect();
        let mut found = Vec::new();
        let mut budget = MAX_WALK_LISTING_BYTES;
        walk(
            &dir,
            &rel,
            &ignored,
            &mut found,
            &mut resolved.not_utf8,
            &mut budget,
        )?;
        found.sort();
        resolved.files = without_dangling(found, workspace, &mut resolved.dangling_links);
    }
    Ok(resolved)
}

/// `names` as files to read, with every link whose target is gone moved to `dangling` instead
/// (relative to `workspace`), in order. O(names).
fn without_dangling(
    names: Vec<String>,
    workspace: &Path,
    dangling: &mut Vec<String>,
) -> Vec<ReviewFile> {
    let (gone, present): (Vec<String>, Vec<String>) = names
        .into_iter()
        .partition(|name| is_dangling_link(&workspace.join(name)));
    dangling.extend(gone);
    present
        .into_iter()
        .map(|path| ReviewFile { path, diff: None })
        .collect()
}

/// The bytes of names a walk outside a repository may list (each name plus one, as `ls-files
/// -z` counts its NUL): the cap a repository's listing has as git output.
const MAX_WALK_LISTING_BYTES: u64 = super::source::MAX_GIT_OUTPUT_BYTES;

/// The cause of a `Directory` input that names something other than a directory.
pub const NOT_A_DIRECTORY: &str = "not a directory";

/// Walks `dir` (whose workspace-relative path is `rel`) without following symlinks, collecting
/// every non-directory entry (a link whose target is gone too, which the caller sets aside with
/// [`without_dangling`]) except a never-reviewed path
/// ([`is_never_reviewed`], never entered) and what `ignored` names (a file by its path, a
/// directory by its path and a trailing `/`, never entered). O(entries).
///
/// Every name listed (a non-UTF-8 one too) and every directory entered is paid from `budget`
/// ([`spend`]), so a deep or wide tree, or a directory
/// of millions of files stops at the cap a repository's listing has instead of being held
/// whole before any content cap applies.
///
/// A link to a directory, a FIFO, a socket or a device file is collected on purpose (reviews
/// of S6 loop 6 asked to drop them here): the reader refuses each one and the filter lists it
/// as `not a regular file`, so the report names it instead of passing over it in silence.
/// Pinned by `an_enumerated_link_to_a_directory_or_fifo_is_skipped_as_not_a_regular_file`.
fn walk(
    dir: &Path,
    rel: &str,
    ignored: &HashSet<String>,
    found: &mut Vec<String>,
    not_utf8: &mut Vec<String>,
    budget: &mut u64,
) -> Result<(), ReviewError> {
    let io = |e: std::io::Error| ReviewError::Io(format!("{}: {e}", dir.display()));
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let name = entry.file_name();
        let joined = |n: &str| {
            if rel.is_empty() {
                n.to_string()
            } else {
                format!("{rel}/{n}")
            }
        };
        let Some(path) = name.to_str().map(joined) else {
            let lossy = joined(&name.to_string_lossy());
            spend(budget, &lossy)?;
            not_utf8.push(lossy);
            continue;
        };
        // MAGI FIX (S6 loop 3): `.git`, `.magi` and the memory database are pruned here, before
        // the walk recurses, so a large nested `.git` is never enumerated.
        if is_never_reviewed(&path)
            || ignored.contains(&path)
            || ignored.contains(&format!("{path}/"))
        {
            continue;
        }
        if entry.file_type().map_err(io)?.is_dir() {
            // MAGI FIX (S6 loop 9): an empty directory cost the walk without paying the cap.
            spend(budget, &path)?;
            walk(&entry.path(), &path, ignored, found, not_utf8, budget)?;
        } else {
            spend(budget, &path)?;
            found.push(path);
        }
    }
    Ok(())
}

/// Takes the cost of listing `path` (its bytes plus one) from `budget`, the bytes a walk may
/// still list, which starts at [`MAX_WALK_LISTING_BYTES`].
///
/// # Errors
///
/// [`ReviewError::InputTooLarge`] naming `path` when its cost passes what is left.
// MAGI FIX (S6 loop 9): the walk listed every name with no bound (A1-c4).
fn spend(budget: &mut u64, path: &str) -> Result<(), ReviewError> {
    let cost = u64::try_from(path.len())
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    *budget = budget
        .checked_sub(cost)
        .ok_or_else(|| ReviewError::InputTooLarge {
            limit: MAX_WALK_LISTING_BYTES,
            held: MAX_WALK_LISTING_BYTES.saturating_sub(*budget),
            path: path.to_string(),
        })?;
    Ok(())
}

/// Uncommitted changes: the diff against the SHA of `HEAD` plus untracked files whole, or, in a
/// repository without commits, every tracked and untracked file whole.
///
/// A listed link whose target is gone is set aside ([`without_dangling`]): it reads as a file
/// that does not exist (`workspace` is where the names resolve).
///
/// git must place the workspace inside its work tree first: a repository's `core.worktree` can
/// move the work tree out of the workspace, and a diff against it would read files there.
fn resolve_uncommitted(
    git: &dyn GitCommand,
    workspace: &Path,
) -> Result<ResolvedInput, ReviewError> {
    if !in_work_tree(git, workspace, workspace)? {
        return Err(ReviewError::Git(format!(
            "{} is not inside a git work tree; uncommitted changes need one",
            workspace.display()
        )));
    }
    match verify(git, "HEAD") {
        Ok(head) => {
            let mut resolved = from_diff(&worktree_diff(git, &head)?, String::new(), None);
            resolved.source_kind = SourceKind::Worktree {
                old_rev: Some(head),
            };
            // `ls-files` lists only what lies under its cwd (the workspace, `-C`), so an untracked
            // file above a sub-workspace is never named here, with or without a pathspec (a
            // review of S6 loop 8 read it as `../…`). Measured with git 2.56 and pinned by
            // `an_uncommitted_review_in_a_sub_workspace_lists_nothing_outside_it`; whatever a
            // listing names, `resolve_input` refuses a name that climbs out.
            let (names, not_utf8) =
                split_nul(&git.run_bytes(&["ls-files", "--others", "--exclude-standard", "-z"])?);
            resolved.not_utf8.extend(not_utf8);
            let untracked = without_dangling(names, workspace, &mut resolved.dangling_links);
            resolved.files.extend(untracked);
            Ok(resolved)
        }
        Err(ReviewError::RefNotFound(_)) => {
            let (names, not_utf8) = split_nul(&git.run_bytes(LS_FILES_ALL)?);
            let mut dangling_links = Vec::new();
            Ok(ResolvedInput {
                files: without_dangling(names, workspace, &mut dangling_links),
                source_kind: SourceKind::Worktree { old_rev: None },
                not_utf8,
                dangling_links,
            })
        }
        Err(other) => Err(other),
    }
}

/// Whether `path` is a link (or a junction) whose target does not exist. Such a link reads as a
/// file that does not exist, so an enumeration leaves it out the way it leaves out a tracked
/// file deleted from the tree. A path that does not exist at all is not a dangling link.
/// Only a missing target counts: a loop or an inaccessible target is not dangling, so it reaches
/// the reader, which reports it.
fn is_dangling_link(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
        && std::fs::metadata(path).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
}

/// Whether `path` is a workspace-relative path made only of plain components: no `..`, no root,
/// no drive prefix, not empty. O(path length).
fn is_confined(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
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
        dangling_links: Vec::new(),
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
                if let Some(path) = reviewable_path(text, unquote) {
                    files.push(ReviewFile {
                        path,
                        diff: Some(text.to_string()),
                    });
                } else if let Some(path) = reviewable_path(text, unquote_lossy) {
                    // MAGI FIX (S6 loop 4): a quoted path whose escapes decode to non-UTF-8
                    // bytes was dropped from both lists.
                    not_utf8.push(path);
                }
            }
            Err(_) => {
                if let Some(path) =
                    reviewable_path(&String::from_utf8_lossy(section), unquote_lossy)
                {
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
        .any(is_binary_marker_line)
}

/// Whether one header line of a diff section is git's binary marker (`Binary files … differ`
/// or `GIT binary patch`). O(line).
pub(crate) fn is_binary_marker_line(line: &str) -> bool {
    line.starts_with(BINARY_MARKER) || line == BINARY_PATCH_MARKER
}

/// The path of a section worth reviewing (it has hunks, or it is binary and gets listed), or
/// `None` for a gitlink, a header-only section with nothing to review, or an unparseable header.
///
/// `decode` undoes git's quoting of a path ([`unquote`], or [`unquote_lossy`] for the lists of
/// paths that are not UTF-8).
///
/// A gitlink test on any mode line drops no regular content: git prints a type change between a
/// gitlink and a regular file as two sections (a deletion, then an addition), so the gitlink's
/// section holds only `Subproject commit` and the regular file's section carries no `160000`
/// (measured with git 2.56; pinned by
/// `a_gitlink_replaced_by_a_file_or_the_reverse_keeps_the_regular_side`, `src/system/review_git.rs`).
// MAGI FIX (S6 loop 10): A1-m1/A1-b1 refuted with real git; both transitions are pinned.
fn reviewable_path(section: &str, decode: Decode) -> Option<String> {
    let mut lines = section.lines();
    let first = lines.next()?;
    let header: Vec<&str> = lines.take_while(|l| !l.starts_with("@@")).collect();
    let has_hunk = section.lines().any(|l| l.starts_with("@@"));
    // MAGI FIX (S6 loop 4): anchored to the mode lines; `--- a/mode 160000` read as a gitlink.
    let gitlink = header.iter().any(|l| {
        MODE_HEADER_PREFIXES.iter().any(|p| l.starts_with(p))
            && l.split(' ').next_back() == Some(GITLINK_MODE)
    });
    if gitlink || !(has_hunk || is_binary_section(section)) {
        return None;
    }
    section_path(first, &header, decode).filter(|p| !p.is_empty())
}

/// The old path of a renamed or copied file's section (its `rename from` or `copy from` header
/// line), or `None` when the section is neither or the path does not decode.
// MAGI FIX (S6 loop 3): `diff.renames=copies` prints `copy from`; without it the old side was read
// at the destination path.
pub(crate) fn renamed_from(section: &str) -> Option<String> {
    section
        .lines()
        .skip(1)
        .take_while(|l| !l.starts_with("@@"))
        .find_map(|l| {
            l.strip_prefix("rename from ")
                .or_else(|| l.strip_prefix("copy from "))
        })
        .and_then(unquote)
}

/// The path of a section: `rename to` (or `copy to`), then `+++`, then `---` for a deletion, then the
/// symmetric split of the `diff --git` header.
fn section_path(first: &str, header: &[&str], decode: Decode) -> Option<String> {
    let after = |prefix: &str| header.iter().find_map(|l| l.strip_prefix(prefix));
    if let Some(p) = after("rename to ")
        .or_else(|| after("copy to "))
        .and_then(decode)
    {
        return Some(p);
    }
    let side = |line: Option<&str>| -> Option<String> {
        let raw = line?;
        let raw = raw.strip_suffix('\t').unwrap_or(raw);
        if raw == "/dev/null" {
            return None;
        }
        let path = decode(raw)?;
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
    from_sides.or_else(|| header_path(first, decode))
}

/// The path of a `diff --git <a> <b>` header whose two halves are the same path: the text has
/// odd length `2k + 1`, byte `k` is a space, and the halves differ only in `a/` against `b/`
/// (quoted or not). Never cut at the first or last ` b/`: a path may contain one.
fn header_path(first: &str, decode: Decode) -> Option<String> {
    let rest = first.strip_prefix("diff --git ")?;
    let k = rest.len() / 2;
    if rest.len() % 2 == 0 || rest.as_bytes().get(k) != Some(&b' ') {
        return None;
    }
    let (left, right) = (rest.get(..k)?, rest.get(k + 1..)?);
    for (a, b) in [("a/", "b/"), ("\"a/", "\"b/")] {
        if let (Some(l), Some(r)) = (left.strip_prefix(a), right.strip_prefix(b)) {
            if l == r {
                return decode(left)?.strip_prefix("a/").map(str::to_string);
            }
        }
    }
    None
}

/// How a diff path is decoded from git's quoting: [`unquote`] or [`unquote_lossy`].
type Decode = fn(&str) -> Option<String>;

/// Undoes git's C-style quoting (`"..."` with `\\ \" \a \b \t \n \v \f \r` and octal `\ooo`
/// bytes); an unquoted text comes back as is. `None` for a malformed quote or escape, or bytes
/// that are not UTF-8.
fn unquote(text: &str) -> Option<String> {
    String::from_utf8(unquote_bytes(text)?).ok()
}

/// [`unquote`], with bytes that are not UTF-8 decoded lossily: the name of a path set aside as
/// not UTF-8. `None` only for a malformed quote or escape.
fn unquote_lossy(text: &str) -> Option<String> {
    Some(String::from_utf8_lossy(&unquote_bytes(text)?).into_owned())
}

/// The bytes git's C-style quoting stands for (see [`unquote`]); `None` for a malformed quote
/// or escape.
///
/// Every escape consumes its backslash and the byte after it (three more for an octal), so an
/// escaped backslash followed by an escaped quote (`\\\"`) decodes to `\"`: a review of S6 loop 8
/// read it as a raw quote. Pinned by `an_escaped_backslash_followed_by_an_escaped_quote_decodes_both`.
fn unquote_bytes(text: &str) -> Option<Vec<u8>> {
    let Some(inner) = text.strip_prefix('"') else {
        return Some(text.as_bytes().to_vec());
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
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::source::{ContentSource, WorktreeSource};
    use crate::review::test_support::FakeGit;

    #[test]
    fn a_copied_file_reads_its_old_side_under_the_copy_source() {
        // [MAGI S6 loop 3, A1] `diff.renames=copies` makes git print `copy from`/`copy to`: the old
        // side lives at the source path, never at the destination.
        let section =
            "diff --git a/src/a.rs b/src/b.rs\nsimilarity index 90%\ncopy from src/a.rs\n\
                       copy to src/b.rs\n--- a/src/a.rs\n+++ b/src/b.rs\n@@ -1 +1 @@\n-x\n+y\n";
        assert_eq!(renamed_from(section).as_deref(), Some("src/a.rs"));
        let renamed = "diff --git a/o.rs b/n.rs\nrename from o.rs\nrename to n.rs\n";
        assert_eq!(renamed_from(renamed).as_deref(), Some("o.rs"));
    }

    #[test]
    fn pinning_turns_every_ref_into_its_commit_and_leaves_the_working_tree_alone() {
        let git = FakeGit::from(&[
            (
                "rev-parse --verify --quiet HEAD^{commit}",
                Ok("aaa
"),
            ),
            (
                "rev-parse --verify --quiet main^{commit}",
                Ok("bbb
"),
            ),
            (
                "rev-parse --verify --quiet x^{commit}",
                Ok("ccc
"),
            ),
            ("rev-parse --verify --quiet gone^{commit}", Err("")),
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
                    "rev-parse --verify --quiet HEAD^{commit}".to_string(),
                    Ok("abc\n".to_string()),
                ),
                (
                    "rev-parse --verify --quiet abc^^{commit}".to_string(),
                    Ok("p\n".to_string()),
                ),
                (
                    "-c attr.tree=abc diff --no-color --no-ext-diff --no-textconv --relative p..abc".to_string(),
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
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("r\n")),
            ("rev-parse --verify --quiet r^^{commit}", Err("")),
            ("cat-file commit r", Ok("tree t\nauthor a <a@x> 0 +0000\n\nparent in a message\n")),
            ("-c attr.tree=r diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative r",
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
                "rev-parse --verify --quiet nope^{commit}".to_string(),
                Err(String::new()),
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
    fn a_git_that_fails_while_resolving_a_ref_is_a_git_error_not_a_missing_ref() {
        // [MAGI §6 MS2 A loop 1] only git's "no such commit" answer (exit 1, nothing on stderr) is a missing ref;
        // a timeout or a corrupt repository is reported as what it is.
        let git = FakeGit::from(&[(
            "rev-parse --verify --quiet main^{commit}",
            Err("fatal: unable to read refs/heads/main"),
        )]);
        for input in [
            ReviewInput::Commit(Some("main".into())),
            ReviewInput::Range {
                from: "main".into(),
                to: "main".into(),
            },
        ] {
            assert!(
                matches!(
                    resolve_input(&input, Path::new("/w"), &git, &allow_all),
                    Err(ReviewError::Git(_))
                ),
                "{input:?}"
            );
        }
        assert!(matches!(
            pin_refs(&ReviewInput::Commit(Some("main".into())), &git),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_commit_whose_parent_is_unavailable_is_an_error_not_a_root_commit() {
        // [MAGI S6 loop 5, A1-m2] a shallow edge (or a missing parent object) also answers "no such
        // commit" for `<sha>^`; reviewing it with `--root` would send the whole tree as added.
        let git = FakeGit::from(&[
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
            ("rev-parse --verify --quiet abc^^{commit}", Err("")),
            (
                "cat-file commit abc",
                Ok("tree t\nparent p\nauthor a <a@x> 0 +0000\n\nm\n"),
            ),
            (
                "-c attr.tree=abc diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(ONE_HUNK),
            ),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(None),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_failing_parent_lookup_is_an_error_not_a_root_commit() {
        // A root commit is git's "no such commit" for `<sha>^`; anything else must not drop the old side.
        let git = FakeGit::from(&[
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
            (
                "rev-parse --verify --quiet abc^^{commit}",
                Err("fatal: git rev-parse did not finish in 300 s"),
            ),
            (
                "-c attr.tree=abc diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(ONE_HUNK),
            ),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(None),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_git_failure_on_head_never_falls_back_to_reviewing_every_file_whole() {
        // [MAGI §6 MS2 A loop 1] the unborn-repository fallback is for git's "no such commit" answer only.
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "rev-parse --verify --quiet HEAD^{commit}",
                Err("fatal: git rev-parse did not finish in 300 s"),
            ),
            (
                "ls-files --cached --others --exclude-standard -z",
                Ok("a.rs\0b.rs\0"),
            ),
        ]);
        assert!(matches!(
            resolve_input(&ReviewInput::Uncommitted, Path::new("/w"), &git, &allow_all),
            Err(ReviewError::Git(_))
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
            ("rev-parse --verify --quiet v1^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet v2^{commit}", Ok("b\n")),
            (
                "-c attr.tree=b diff --no-color --no-ext-diff --no-textconv --relative a..b",
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
            ("rev-parse --verify --quiet main^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet feature/x^{commit}", Ok("b\n")),
            ("merge-base a b", Ok("mb1\n")),
            (
                "-c attr.tree=b diff --no-color --no-ext-diff --no-textconv --relative mb1..b",
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
            ("rev-parse --verify --quiet main^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet orphan^{commit}", Ok("b\n")),
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
            ("rev-parse --verify --quiet v1^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet nope^{commit}", Err("")),
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
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
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
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            ("rev-parse --verify --quiet HEAD^{commit}", Err("")),
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
    fn a_listed_name_that_climbs_out_of_the_workspace_is_refused_in_every_listing() {
        // [MAGI S6 loop 6, A1-c1] git scopes `ls-files` to its cwd (pinned against the real git
        // by `an_uncommitted_review_in_a_sub_workspace_lists_nothing_outside_it`), but a name
        // that climbs out (`../secret.txt`) or is absolute must never become a file to read.
        let d = tempfile::tempdir().expect("tempdir");
        let ws = d.path().join("ws");
        std::fs::create_dir_all(&ws).expect("mkdir");
        std::fs::write(d.path().join("secret.txt"), "s\n").expect("write");
        std::fs::write(ws.join("a.rs"), "fn a() {}\n").expect("write");
        let cases = [
            (
                ReviewInput::Uncommitted,
                FakeGit::from(&[
                    ("rev-parse --is-inside-work-tree", Ok("true\n")),
                    ("rev-parse --verify --quiet HEAD^{commit}", Ok("abc\n")),
                    (
                        "diff --no-color --no-ext-diff --no-textconv --relative abc",
                        Ok(""),
                    ),
                    (
                        "ls-files --others --exclude-standard -z",
                        Ok("a.rs\0../secret.txt\0"),
                    ),
                ]),
            ),
            (
                ReviewInput::Uncommitted,
                FakeGit::from(&[
                    ("rev-parse --is-inside-work-tree", Ok("true\n")),
                    ("rev-parse --verify --quiet HEAD^{commit}", Err("")),
                    (
                        "ls-files --cached --others --exclude-standard -z",
                        Ok("a.rs\0../secret.txt\0"),
                    ),
                ]),
            ),
            (
                ReviewInput::Directory(".".into()),
                FakeGit::from(&[
                    ("rev-parse --is-inside-work-tree", Ok("true\n")),
                    (
                        "ls-files --cached --others --exclude-standard -z -- .",
                        Ok("a.rs\0../secret.txt\0"),
                    ),
                ]),
            ),
        ];
        for (input, git) in cases {
            let r = resolve_input(&input, &ws, &git, &allow_all);
            assert!(
                matches!(r, Err(ReviewError::OutsideWorkspace(ref p)) if p == "../secret.txt"),
                "{input:?}: {:?}",
                r.as_ref().err()
            );
        }
    }

    #[test]
    fn a_git_entry_that_cannot_be_read_counts_as_a_repository() {
        // [MAGI S6 loop 6, A1-c4] only "not found" means no `.git`: a permission error on an
        // existing entry must not route the review to the no-repository walk.
        use std::io::{Error, ErrorKind};
        assert!(!is_git_evidence(&Err(Error::from(ErrorKind::NotFound))));
        assert!(is_git_evidence(&Err(Error::from(
            ErrorKind::PermissionDenied
        ))));
        assert!(is_git_evidence(&Ok(())));
    }

    #[test]
    fn an_input_whose_every_name_is_not_utf8_is_nothing_to_review_and_names_them() {
        // [MAGI S6 loop 6, A1-c3] with only non-UTF-8 names left, resolve_input returned Ok with
        // no file and the omissions were lost from the later `nothing to review` error.
        use crate::review::filter::{SkipCause, Skipped};
        struct NotUtf8Git;
        impl GitCommand for NotUtf8Git {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "rev-parse --is-inside-work-tree" => Ok(b"true\n".to_vec()),
                    "rev-parse --verify --quiet HEAD^{commit}" => {
                        Err(ReviewError::Git(String::new()))
                    }
                    "ls-files --cached --others --exclude-standard -z" => {
                        Ok(b"caf\xe9.rs\0".to_vec())
                    }
                    other => Err(ReviewError::Git(other.to_string())),
                }
            }
        }
        let r = resolve_input(
            &ReviewInput::Uncommitted,
            Path::new("/w"),
            &NotUtf8Git,
            &allow_all,
        );
        let Err(ReviewError::NothingToReview { skipped }) = r else {
            panic!("expected NothingToReview, got {:?}", r.err());
        };
        assert_eq!(
            skipped,
            vec![Skipped {
                path: "caf\u{fffd}.rs".into(),
                cause: SkipCause::NotUtf8
            }]
        );
    }

    #[cfg(unix)] // a symlink needs Developer Mode on Windows, a FIFO has no Windows form
    #[test]
    fn an_enumerated_link_to_a_directory_or_fifo_is_skipped_as_not_a_regular_file() {
        // [MAGI S6 loop 6, A1-b1/A1-b2, by design] the walk lists them so the report names them;
        // the reader refuses them, and the filter lists them as `not a regular file` (README).
        use crate::review::filter::{filter_files, SkipCause, Skipped};
        use crate::review::languages::LanguageRegistry;
        use crate::review::source::WorktreeSource;
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        std::fs::create_dir_all(d.path().join("sub")).expect("mkdir");
        std::fs::write(d.path().join("sub/a.rs"), "fn a() {}\n").expect("write");
        std::os::unix::fs::symlink(d.path().join("sub"), d.path().join("link")).expect("link");
        let made = std::process::Command::new("mkfifo")
            .arg(d.path().join("pipe"))
            .status()
            .expect("mkfifo runs");
        assert!(made.success());
        let git = FakeGit::from(&[
            (
                "rev-parse --is-inside-work-tree",
                Err("fatal: not a git repository"),
            ),
            (
                "ls-files --others --ignored --exclude-standard --directory -z -- .",
                Ok(""),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory(".".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        let source = WorktreeSource::new(d.path().to_path_buf());
        let f = filter_files(r.files, &source, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("sub/a.rs is kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["sub/a.rs"]
        );
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "link".into(),
                    cause: SkipCause::NotRegularFile
                },
                Skipped {
                    path: "pipe".into(),
                    cause: SkipCause::NotRegularFile
                },
            ]
        );
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
                    "rev-parse --verify --quiet v1^{commit}"
                    | "rev-parse --verify --quiet v2^{commit}" => Ok(b"x\n".to_vec()),
                    "-c attr.tree=x diff --no-color --no-ext-diff --no-textconv --relative x..x" => {
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
    fn a_rename_whose_source_climbs_out_of_the_workspace_is_refused() {
        // [MAGI S6 loop 10, A1-m2] `renamed_from` names the path the old side is read at; it
        // goes through the same confinement check as `files[].path`, before any read.
        let section = "diff --git a/../outside.rs b/a.rs\nsimilarity index 75%\nrename from ../outside.rs\nrename to a.rs\n--- a/../outside.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-x\n+y\n";
        let git = FakeGit::from(&[
            ("rev-parse --verify --quiet v1^{commit}", Ok("x\n")),
            ("rev-parse --verify --quiet v2^{commit}", Ok("x\n")),
            (
                "-c attr.tree=x diff --no-color --no-ext-diff --no-textconv --relative x..x",
                Ok(section),
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
        );
        assert!(
            matches!(r, Err(ReviewError::OutsideWorkspace(ref p)) if p == "../outside.rs"),
            "{:?}",
            r.as_ref().map(|r| &r.files)
        );
    }

    #[test]
    fn a_commit_with_a_parent_is_diffed_against_that_pinned_parent() {
        // [MAGI §6 MS2 A loop 2] `diff-tree <merge>` prints nothing for a merge commit; the diff
        // against the pinned first parent is what the review reads (`old_rev`), merge or not.
        let git = FakeGit::from(&[
            ("rev-parse --verify --quiet M^{commit}", Ok("m\n")),
            ("rev-parse --verify --quiet m^^{commit}", Ok("p\n")),
            (
                "-c attr.tree=m diff --no-color --no-ext-diff --no-textconv --relative p..m",
                Ok(ONE_HUNK),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Commit(Some("M".into())),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
    }

    #[test]
    fn a_directory_that_does_not_exist_is_missing_not_nothing_to_review() {
        // [MAGI §6 MS2 A loop 2] a typo in a directory name says what is wrong, like a file's does.
        let d = tempfile::tempdir().expect("tempdir");
        let git = FakeGit(Default::default());
        assert!(matches!(
            resolve_input(
                &ReviewInput::Directory("nope".into()),
                d.path(),
                &git,
                &allow_all
            ),
            Err(ReviewError::NotReviewable {
                cause: "missing",
                ..
            })
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
        // [MAGI S6 loop 8, A1-b4 Balthasar] the submodule is kept: the reader refuses it and the
        // filter lists it as `not a regular file`, instead of a silent drop.
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs", "src/b.rs", "src/vendored"]
        );
        assert!(r.files.iter().all(|f| f.diff.is_none()));
    }

    #[test]
    fn a_listed_name_whose_metadata_cannot_be_read_is_kept_for_the_reader_to_report() {
        // [MAGI S6 loop 8, A1-c2 Caspar] any metadata error but "not found" dropped the name in
        // silence; it is kept, and the reader fails naming it.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        // A component that is a file (Unix: not a directory) or a name Windows refuses.
        let bad = if cfg!(windows) {
            "bad<name.rs"
        } else {
            "a.rs/x.rs"
        };
        assert!(
            std::fs::symlink_metadata(d.path().join(bad))
                .is_err_and(|e| e.kind() != std::io::ErrorKind::NotFound),
            "the fixture must fail with something other than not found"
        );
        let listing = format!("a.rs\0{bad}\0");
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- .",
                Ok(listing.as_str()),
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
            vec!["a.rs", bad]
        );
        let read = WorktreeSource::new(d.path().to_path_buf()).read(bad);
        assert!(matches!(read, Err(ReviewError::Io(_))), "{read:?}");
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
    fn a_directory_input_that_names_a_regular_file_is_not_reviewable() {
        // [MAGI S6 loop 3, A3] in a repository `ls-files -- <file>` lists the file itself, so the
        // shape is checked before git runs; no git call is expected (FakeGit has none).
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        let git = FakeGit::from(&[]);
        match resolve_input(
            &ReviewInput::Directory("a.rs".into()),
            d.path(),
            &git,
            &allow_all,
        ) {
            Err(ReviewError::NotReviewable { path, cause }) => {
                assert_eq!(path, "a.rs");
                assert_eq!(cause, NOT_A_DIRECTORY);
            }
            Err(other) => panic!("expected NotReviewable, got {other:?}"),
            Ok(_) => panic!("a regular file was accepted as a directory"),
        }
    }

    #[test]
    fn the_walk_outside_a_repo_never_enters_a_never_reviewed_path() {
        // [MAGI S6 loop 3, A4] a nested `.git`, `.magi` or the memory database is pruned before
        // the walk recurses: a large `.git` is never enumerated during the preparation.
        let d = tempfile::tempdir().expect("tempdir");
        for dir in ["src/sub/.git/objects", "src/.magi/logs"] {
            std::fs::create_dir_all(d.path().join(dir)).expect("mkdir");
        }
        for file in [
            "src/a.rs",
            "src/sub/.git/config",
            "src/sub/.git/objects/x",
            "src/.magi/magi.toml",
            "src/.magi-rs-memory.db-wal",
        ] {
            std::fs::write(d.path().join(file), "x\n").expect("write");
        }
        let git = FakeGit::from(&[
            (
                "rev-parse --is-inside-work-tree",
                Err("fatal: not a git repository"),
            ),
            (
                "ls-files --others --ignored --exclude-standard --directory -z -- src",
                Ok(""),
            ),
        ]);
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
            vec!["src/a.rs"]
        );
    }

    #[test]
    fn a_directory_outside_a_repo_is_walked_in_order_without_what_gitignore_excludes() {
        // [MAGI §6 MS2 A loop 1] outside a repository git still names what `.gitignore` excludes, and the walk leaves it out.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src/sub")).expect("mkdir");
        std::fs::write(d.path().join("src/b.rs"), "b\n").expect("write");
        std::fs::write(d.path().join("src/sub/a.rs"), "a\n").expect("write");
        std::fs::write(d.path().join("src/c.env"), "c\n").expect("write");
        let git = FakeGit::from(&[
            (
                "rev-parse --is-inside-work-tree",
                Err("fatal: not a git repository"),
            ),
            (
                "ls-files --others --ignored --exclude-standard --directory -z -- src",
                Ok("src/c.env\0"),
            ),
        ]);
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
            ("rev-parse --verify --quiet v1^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet v2^{commit}", Ok("b\n")),
            (
                "-c attr.tree=b diff --no-color --no-ext-diff --no-textconv --relative a..b",
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
                ("rev-parse --verify --quiet v1^{commit}", Ok("a\n")),
                ("rev-parse --verify --quiet v2^{commit}", Ok("b\n")),
                (
                    "-c attr.tree=b diff --no-color --no-ext-diff --no-textconv --relative a..b",
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
    fn a_section_whose_path_is_not_utf8_is_listed_in_every_shape_git_writes() {
        // [MAGI §6 MS2 A loop 1, bytes as git 2.47 writes them with core.quotePath=false] a name byte that is not UTF-8
        // is raw, never an octal escape, in a plain, a quoted (it holds a tab) and a header-only binary section.
        let mut diff = b"diff --git a/caf\xe9.rs b/caf\xe9.rs\n--- a/caf\xe9.rs\n+++ b/caf\xe9.rs\n@@ -1 +1 @@\n-x\n+y\n".to_vec();
        diff.extend_from_slice(
            b"diff --git \"a/t\\tb\xe9.rs\" \"b/t\\tb\xe9.rs\"\n--- \"a/t\\tb\xe9.rs\"\n+++ \"b/t\\tb\xe9.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
        );
        diff.extend_from_slice(
            b"diff --git a/bin\xe9.so b/bin\xe9.so\nindex 1..2 100644\nBinary files a/bin\xe9.so and b/bin\xe9.so differ\n",
        );
        let (files, not_utf8) = split_diff(&diff);
        assert!(files.is_empty());
        assert_eq!(
            not_utf8,
            vec![
                "caf\u{fffd}.rs".to_string(),
                "t\tb\u{fffd}.rs".to_string(),
                "bin\u{fffd}.so".to_string()
            ]
        );
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
            ("rev-parse --verify --quiet v1^{commit}", Ok("a\n")),
            ("rev-parse --verify --quiet v2^{commit}", Ok("b\n")),
            (
                "-c attr.tree=b diff --no-color --no-ext-diff --no-textconv --relative a..b",
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

    #[cfg(unix)] // Windows cannot name a file with bytes that are not UTF-8.
    #[test]
    fn a_top_level_name_that_is_not_utf8_is_set_aside_without_a_leading_slash() {
        // [MAGI §6 MS2 A loop 1] the walk joined an empty parent with `/`: `/caf�.rs`, where a UTF-8 name reads `a.rs`.
        use std::os::unix::ffi::OsStrExt;
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        let name = std::ffi::OsStr::from_bytes(b"caf\xe9.rs");
        std::fs::write(d.path().join(name), "x\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "x\n").expect("write");
        let git = FakeGit::from(&[
            (
                "rev-parse --is-inside-work-tree",
                Err("fatal: not a git repository"),
            ),
            (
                "ls-files --others --ignored --exclude-standard --directory -z -- .",
                Ok(""),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory(".".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.not_utf8, vec!["caf\u{fffd}.rs".to_string()]);
    }

    #[cfg(unix)]
    #[test]
    fn only_a_link_whose_target_is_gone_is_dangling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("real.rs"), "fn f() {}").expect("real");
        std::os::unix::fs::symlink("real.rs", dir.join("ok")).expect("ok");
        std::os::unix::fs::symlink("gone.rs", dir.join("dangling")).expect("dangling");
        std::os::unix::fs::symlink("b", dir.join("a")).expect("a");
        std::os::unix::fs::symlink("a", dir.join("b")).expect("b");
        assert!(is_dangling_link(&dir.join("dangling")));
        assert!(!is_dangling_link(&dir.join("ok")));
        assert!(!is_dangling_link(&dir.join("real.rs")));
        assert!(!is_dangling_link(&dir.join("missing")));
        assert!(!is_dangling_link(&dir.join("a")), "a loop is not dangling");
    }
    #[test]
    fn a_quoted_path_that_decodes_to_non_utf8_is_set_aside_not_dropped() {
        // [MAGI S6 MS2 loop 4, A1 Melchior] with octal escapes (core.quotePath=true) the section
        // is UTF-8 text but its path is not: it fell through both lists.
        let diff = concat!(
            "diff --git \"a/caf\\351.rs\" \"b/caf\\351.rs\"\n",
            "--- \"a/caf\\351.rs\"\n+++ \"b/caf\\351.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/ok.rs b/ok.rs\n--- a/ok.rs\n+++ b/ok.rs\n@@ -1 +1 @@\n-x\n+y\n",
        );
        let (files, not_utf8) = split_diff(diff.as_bytes());
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["ok.rs"]
        );
        assert_eq!(not_utf8, vec!["caf\u{fffd}.rs".to_string()]);
    }
    #[test]
    fn an_escaped_backslash_followed_by_an_escaped_quote_decodes_both() {
        // [MAGI S6 loop 8, A1-b3 Balthasar] the review read `\\\"` as an escaped backslash and a
        // raw quote, dropping the section. Each escape consumes exactly two bytes, so the next
        // `\"` starts a fresh escape: the name `a\"b.rs` survives, in the header and in a section.
        assert_eq!(unquote(r#""a/a\\\"b.rs""#).as_deref(), Some(r#"a/a\"b.rs"#));
        assert_eq!(unquote(r#""a\\\\\"""#).as_deref(), Some(r#"a\\""#));
        let diff = concat!(
            "diff --git \"a/a\\\\\\\"b.rs\" \"b/a\\\\\\\"b.rs\"\n",
            "--- \"a/a\\\\\\\"b.rs\"\n+++ \"b/a\\\\\\\"b.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
        );
        let (files, not_utf8) = split_diff(diff.as_bytes());
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec![r#"a\"b.rs"#]
        );
        assert!(not_utf8.is_empty(), "{not_utf8:?}");
    }
    #[test]
    fn a_path_shaped_like_a_mode_line_is_not_mistaken_for_a_submodule() {
        // [MAGI S6 MS2 loop 4, A1 Melchior] `--- a/mode 160000` holds `mode ` and ends in the
        // gitlink mode, so the file was dropped from both lists as a submodule.
        let diff = concat!(
            "diff --git a/mode 160000 b/mode 160000\nindex 1..2 100644\n",
            "--- a/mode 160000\n+++ b/mode 160000\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/sub b/sub\nindex 1..2 160000\n--- a/sub\n+++ b/sub\n@@ -1 +1 @@\n",
            "-Subproject commit 1\n+Subproject commit 2\n",
            "diff --git a/new b/new\nnew file mode 160000\nindex 0..1\n--- /dev/null\n+++ b/new\n",
            "@@ -0,0 +1 @@\n+Subproject commit 1\n",
        );
        let (files, _) = split_diff(diff.as_bytes());
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["mode 160000"]
        );
    }
    #[cfg(unix)]
    // Creating a symlink needs Developer Mode on Windows; the rule is the same there.
    #[test]
    fn an_input_left_with_only_dangling_links_is_nothing_to_review_and_names_them() {
        // [MAGI S6 loop 8, A1-b4 Balthasar] every listing leaves a dangling link out (it reads as
        // a file that does not exist), and with nothing else left the error named nothing.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::os::unix::fs::symlink(d.path().join("gone.rs"), d.path().join("src/dangling.rs"))
            .expect("link");
        let in_repo = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- src",
                Ok("src/dangling.rs\0"),
            ),
            ("rev-parse --verify --quiet HEAD^{commit}", Err("")),
            (
                "ls-files --cached --others --exclude-standard -z",
                Ok("src/dangling.rs\0"),
            ),
        ]);
        let outside = FakeGit::from(&[
            (
                "rev-parse --is-inside-work-tree",
                Err("fatal: not a git repository"),
            ),
            (
                "ls-files --others --ignored --exclude-standard --directory -z -- src",
                Ok(""),
            ),
        ]);
        let cases: [(ReviewInput, &FakeGit); 3] = [
            (ReviewInput::Directory("src".into()), &in_repo),
            (ReviewInput::Uncommitted, &in_repo),
            (ReviewInput::Directory("src".into()), &outside),
        ];
        for (input, git) in cases {
            if std::ptr::eq(git, &outside) && d.path().ancestors().any(|a| a.join(".git").exists())
            {
                continue; // a `.git` above TMPDIR makes the walk (correctly) fail closed
            }
            let r = resolve_input(&input, d.path(), git, &allow_all);
            let Err(ReviewError::NothingToReview { skipped }) = r else {
                panic!(
                    "{input:?}: expected NothingToReview, got {:?}",
                    r.map(|r| r.files)
                );
            };
            assert_eq!(
                skipped,
                vec![super::super::filter::Skipped {
                    path: "src/dangling.rs".into(),
                    cause: super::super::filter::SkipCause::NotRegularFile,
                }],
                "{input:?}"
            );
        }
    }
    #[cfg(unix)]
    // Creating a symlink needs Developer Mode on Windows; the rule is the same there.
    #[test]
    fn a_file_input_naming_a_link_to_a_directory_or_a_dangling_link_is_not_reviewable() {
        // [MAGI S6 loop 8, A1-b1/A1-b2 Balthasar] `resolve_file` looked at the link itself, so a
        // link to a directory and a link whose target is gone came back as files to read, with
        // whatever guard the caller gave.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("sub")).expect("mkdir");
        std::os::unix::fs::symlink(d.path().join("sub"), d.path().join("to_dir")).expect("link");
        std::os::unix::fs::symlink(d.path().join("gone.rs"), d.path().join("dangling.rs"))
            .expect("link");
        let git = FakeGit(Default::default());
        for (name, expected) in [("to_dir", "directory"), ("dangling.rs", "missing")] {
            let r = resolve_input(&ReviewInput::File(name.into()), d.path(), &git, &allow_all);
            assert!(
                matches!(&r, Err(ReviewError::NotReviewable { cause, .. }) if *cause == expected),
                "{name}: {:?}",
                r.err()
            );
        }
    }
    #[cfg(unix)]
    // Creating a symlink needs Developer Mode on Windows; the rule is the same there.
    #[test]
    fn a_directory_input_naming_a_dangling_link_is_missing() {
        // [MAGI S6 MS2 loop 4, A1 Melchior] a link whose target is gone reads as a path that
        // does not exist, as everywhere else, not as an I/O error.
        let d = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(d.path().join("gone"), d.path().join("dir")).expect("symlink");
        let git = FakeGit(Default::default());
        let r = resolve_input(
            &ReviewInput::Directory("dir".into()),
            d.path(),
            &git,
            &allow_all,
        );
        assert!(
            matches!(
                r,
                Err(ReviewError::NotReviewable {
                    cause: "missing",
                    ..
                })
            ),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn a_walk_outside_a_repository_stops_at_its_listing_cap() {
        // [MAGI S6 loop 9, A1-c4] the walk collected every name with no bound, so a directory
        // of millions of files held them all before any content cap applied. It now stops at
        // the cap a repository's listing has, with a typed error naming where it stopped.
        let d = tempfile::tempdir().expect("tempdir");
        for name in ["a.rs", "b.rs", "c.rs"] {
            std::fs::write(
                d.path().join(name),
                "x
",
            )
            .expect("write");
        }
        let walk_with = |cap: u64| {
            let mut found = Vec::new();
            let mut not_utf8 = Vec::new();
            let mut budget = cap;
            walk(
                d.path(),
                "",
                &HashSet::new(),
                &mut found,
                &mut not_utf8,
                &mut budget,
            )
            .map(|()| found.len())
        };
        // Each name costs its bytes plus one: three names of four bytes take 15.
        assert_eq!(walk_with(15).expect("fits"), 3);
        assert!(matches!(
            walk_with(14),
            Err(ReviewError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn a_walk_pays_for_every_directory_it_enters_and_every_name_it_sets_aside() {
        // [MAGI S6 loop 9, commit review of A1-c4, security] a tree of deep or wide empty
        // directories cost the walk time and recursion without paying the cap, and a name that
        // is not UTF-8 must pay it too. Each is refused with a typed error naming the path.
        let walk_in = |root: &Path, cap: u64| {
            let mut found = Vec::new();
            let mut not_utf8 = Vec::new();
            let mut budget = cap;
            walk(
                root,
                "",
                &HashSet::new(),
                &mut found,
                &mut not_utf8,
                &mut budget,
            )
        };
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("d1/d2/d3/d4")).expect("mkdir");
        // `d1` (3) + `d1/d2` (6) + `d1/d2/d3` (9) fit 18; `d1/d2/d3/d4` (12) does not.
        match walk_in(d.path(), 18) {
            Err(ReviewError::InputTooLarge { path, .. }) => assert_eq!(path, "d1/d2/d3/d4"),
            other => panic!("{other:?}"),
        }
        assert!(walk_in(d.path(), 30).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let u = tempfile::tempdir().expect("tempdir");
            std::fs::write(u.path().join(std::ffi::OsStr::from_bytes(b"x\xff")), "x\n")
                .expect("write");
            // Listed lossily as `x\u{fffd}`: four bytes, five with its separator.
            assert!(matches!(
                walk_in(u.path(), 4),
                Err(ReviewError::InputTooLarge { .. })
            ));
            assert!(walk_in(u.path(), 5).is_ok());
        }
    }

    #[test]
    fn a_walk_of_a_deep_tree_does_not_grow_the_call_stack() {
        // [MAGI S6 loop 33, A1] the walk recursed once per directory level with no depth cap,
        // so a deep enough tree overflowed the stack and aborted the process. It runs here on
        // a thread with a deliberately small stack, so a depth the recursion could not survive
        // stays small enough for any file system.
        const DEPTH: usize = 300;
        const SMALL_STACK_BYTES: usize = 64 * 1024;
        let d = tempfile::tempdir().expect("tempdir");
        let mut deepest = d.path().to_path_buf();
        for _ in 0..DEPTH {
            deepest.push("d");
        }
        std::fs::create_dir_all(&deepest).expect("mkdir");
        std::fs::write(deepest.join("leaf.rs"), "x\n").expect("write");
        let root = d.path().to_path_buf();
        let found = std::thread::Builder::new()
            .stack_size(SMALL_STACK_BYTES)
            .spawn(move || {
                let mut found = Vec::new();
                let mut not_utf8 = Vec::new();
                let mut budget = u64::MAX;
                walk(
                    &root,
                    "",
                    &HashSet::new(),
                    &mut found,
                    &mut not_utf8,
                    &mut budget,
                )
                .map(|()| found)
            })
            .expect("spawn")
            .join()
            .expect("join")
            .expect("walk");
        let leaf = format!("{}leaf.rs", "d/".repeat(DEPTH));
        assert_eq!(found, vec![leaf]);
    }
}
