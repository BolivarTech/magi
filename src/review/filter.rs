// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What never leaves the process and what is not code: the per-file filter of a review.
//!
//! Every file of a review is classified here before any byte of it is batched: a path a review
//! never reads or `.gitignore` excludes, a path or content that holds a secret, a file over the
//! byte cap, a binary and a text that is not UTF-8 are left out with their cause; everything
//! else is kept, and a kept file with no language extractor is declared.

use std::collections::HashSet;
use std::path::Path;

use super::error::ReviewError;
use super::input::{
    in_work_tree, is_binary_section, renamed_from, split_nul, ReviewFile, ReviewInput,
};
use super::languages::LanguageRegistry;
use super::secrets::{display_path, find_secret};
use super::source::{is_never_reviewed, ContentSource, GitCommand};

/// The cause of [`ReviewError::NotReviewable`] for a file requested by name that does not exist.
pub(crate) const MISSING: &str = "missing";
/// The tracing target of the filter's run-log entries.
const LOG_TARGET: &str = "magi_rs::review";
/// `ls-files` for a file named by the caller: tracked or untracked, when a rule excludes it.
const LS_FILES_IGNORED_ANY: &[&str] = &[
    "ls-files",
    "--cached",
    "--others",
    "--ignored",
    "--exclude-standard",
    "-z",
];
/// `ls-files` for a directory or uncommitted changes: the tracked files a rule excludes (the
/// enumeration lists them; untracked ignored files are never enumerated).
const LS_FILES_IGNORED_TRACKED: &[&str] = &[
    "ls-files",
    "--cached",
    "--ignored",
    "--exclude-standard",
    "-z",
];

/// A file left out of a review, with why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Skipped {
    /// The path, as it may leave the process ([`display_path`]).
    pub path: String,
    /// Why it was left out.
    pub cause: SkipCause,
}

/// Why a file was left out of a review.
///
/// Serialized as the text the report shows (`"binary"`, `"not utf-8"`, `"secret"`, `"ignored"`,
/// `"outside workspace"`, `"too large"`, `"not a regular file"`), the same text [`ReviewError::NotReviewable`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipCause {
    /// A binary file.
    Binary,
    /// Text that is not valid UTF-8.
    NotUtf8,
    /// A file whose path or content holds a secret.
    Secret,
    /// A file `.gitignore` excludes, or one a review never reads.
    Ignored,
    /// A file that resolves outside the workspace.
    OutsideWorkspace,
    /// A file over the per-file byte cap.
    TooLarge,
    /// A path whose target is not a regular file: a link to a directory inside the workspace, a
    /// FIFO, a socket or a device file.
    NotRegularFile,
}

impl serde::Serialize for SkipCause {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl std::fmt::Display for Skipped {
    /// `<path> (<cause>)`, the shown path and the cause's text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.path, self.cause.as_str())
    }
}

impl SkipCause {
    /// The cause's text: the one place it is spelled.
    fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::NotUtf8 => "not utf-8",
            Self::Secret => "secret",
            Self::Ignored => "ignored",
            Self::OutsideWorkspace => "outside workspace",
            Self::TooLarge => "too large",
            Self::NotRegularFile => "not a regular file",
        }
    }
}

/// The outcome of filtering.
#[derive(Debug)]
pub struct Filtered {
    /// The files that are reviewed, in input order.
    pub kept: Vec<ReviewFile>,
    /// The files left out, with their cause, in input order.
    pub skipped: Vec<Skipped>,
    /// Kept files with no language extractor, as [`display_path`] shows them.
    pub no_extractor: Vec<String>,
}

/// What classifying one file decided.
enum Verdict {
    /// Reviewed.
    Keep,
    /// Left out, with its cause.
    Skip(SkipCause),
}

/// Splits the files of a review into what is reviewed and what is left out.
///
/// For each file, in this order, the first cause that applies wins: a path a review never
/// reads (a `.git` or `.magi` component, or a name that starts with `.magi-rs-memory.db`,
/// whatever `is_ignored` says) or one `is_ignored` names, as the new path or as a rename's old
/// path → `ignored`; a path that holds a
/// secret → `secret`, without reading it; a file that does not exist → an error, never a skip;
/// over the byte cap → `too large`; a NUL anywhere in a side → `binary`; not UTF-8 →
/// `not utf-8`; a secret anywhere in the new side, the old side or the diff text → `secret`.
/// A diff section whose header lines (before the first `@@`) carry git's binary marker is
/// `binary` without reading anything ([`is_binary_section`]).
/// The binary and UTF-8 checks run on both sides that exist (binary first). The
/// secret scan covers both whole sides, not only what is sent: the safe direction. A renamed
/// file's old side is read under its old path (`rename from`).
///
/// # Arguments
///
/// * `files` - The files the input resolved to.
/// * `source` - The review's single content source.
/// * `registry` - The language extractors, to declare kept files with none.
/// * `explicit` - Whether the file was requested by name: then a skip is an error.
/// * `is_ignored` - A positive set: whether `.gitignore` excludes a path (built by the caller
///   from [`ignored_paths`], once per review).
///
/// # Returns
///
/// The kept files, the skipped ones and the kept ones with no extractor. Every path that
/// leaves the process goes through [`display_path`]. Each `secret` skip writes one INFO entry
/// with the shown path and the pattern's name, never the value.
///
/// # Errors
///
/// [`ReviewError::NotReviewable`] when `explicit` and the file is skipped or missing,
/// [`ReviewError::Io`] when an enumerated file disappeared, [`ReviewError::NothingToReview`]
/// when no file is kept, and any other error of `source`.
///
/// O(total bytes read) time.
pub fn filter_files(
    files: Vec<ReviewFile>,
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    explicit: bool,
    is_ignored: &dyn Fn(&str) -> bool,
) -> Result<Filtered, ReviewError> {
    let mut filtered = Filtered {
        kept: Vec::new(),
        skipped: Vec::new(),
        no_extractor: Vec::new(),
    };
    for file in files {
        match classify(&file, source, is_ignored, explicit)? {
            Verdict::Keep => {
                if registry.get(&file.path).is_none() {
                    filtered.no_extractor.push(display_path(&file.path));
                }
                filtered.kept.push(file);
            }
            Verdict::Skip(cause) if explicit => {
                return Err(ReviewError::NotReviewable {
                    path: display_path(&file.path),
                    cause: cause.as_str(),
                })
            }
            Verdict::Skip(cause) => filtered.skipped.push(Skipped {
                path: display_path(&file.path),
                cause,
            }),
        }
    }
    if filtered.kept.is_empty() {
        // MAGI FIX (S6 loop 3): the omissions and their causes travel with the error (RV20/RV22).
        return Err(ReviewError::NothingToReview {
            skipped: filtered.skipped,
        });
    }
    Ok(filtered)
}

/// Classifies one file (see [`filter_files`] for the order).
fn classify(
    file: &ReviewFile,
    source: &dyn ContentSource,
    is_ignored: &dyn Fn(&str) -> bool,
    explicit: bool,
) -> Result<Verdict, ReviewError> {
    let path = file.path.as_str();
    // A renamed or copied file's old side lives under its old path.
    let old_path = file.diff.as_deref().and_then(renamed_from);
    // MAGI FIX (S6 loop 7): the old path meets the same name rule as the new one: a crafted
    // commit carried a blob stored under `.git/` (or an ignored name) in as a rename's old side.
    if [Some(path), old_path.as_deref()]
        .into_iter()
        .flatten()
        .any(|p| is_never_reviewed(p) || is_ignored(p))
    {
        return Ok(Verdict::Skip(SkipCause::Ignored));
    }
    if let Some(name) = find_secret(path) {
        return Ok(secret(path, name));
    }
    if file.diff.as_deref().is_some_and(is_binary_section) {
        return Ok(Verdict::Skip(SkipCause::Binary));
    }
    let new = match read_side(source.read(path))? {
        Side::Content(bytes) => bytes,
        Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
    };
    let old = match &file.diff {
        Some(_) => match read_side(source.read_old(old_path.as_deref().unwrap_or(path)))? {
            Side::Content(bytes) => bytes,
            Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
        },
        None => None,
    };
    if new.is_none() && file.diff.is_none() {
        return Err(if explicit {
            ReviewError::NotReviewable {
                path: display_path(path),
                cause: MISSING,
            }
        } else {
            ReviewError::Io(format!(
                "{} disappeared during the review",
                display_path(path)
            ))
        });
    }
    // MAGI FIX (S6 loop 4): both sides, not the new one alone: a text file whose old side was
    // binary or Latin-1 carried old-side bytes into the review.
    // MAGI FIX (S6 loop 7): the whole side, not git's first 8 000 bytes: a text prefix hid a
    // binary payload, which is valid UTF-8 (NUL is a code point). One pass over bytes already
    // read and capped (MAX_REVIEW_FILE_BYTES), so the cost stays O(file).
    let sides = [new.as_deref(), old.as_deref()];
    if sides.iter().flatten().any(|side| side.contains(&0)) {
        return Ok(Verdict::Skip(SkipCause::Binary));
    }
    if sides
        .iter()
        .flatten()
        .any(|side| std::str::from_utf8(side).is_err())
    {
        return Ok(Verdict::Skip(SkipCause::NotUtf8));
    }
    let faces = [
        new.as_deref(),
        old.as_deref(),
        file.diff.as_deref().map(str::as_bytes),
    ];
    for face in faces.into_iter().flatten() {
        if let Some(name) = find_secret(&String::from_utf8_lossy(face)) {
            return Ok(secret(path, name));
        }
    }
    Ok(Verdict::Keep)
}

/// A `secret` skip, logged with the shown path and the pattern's name (never the value).
fn secret(path: &str, pattern: &'static str) -> Verdict {
    let shown = display_path(path);
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::INFO,
        path = shown.as_str(),
        pattern = pattern,
        "a file was left out of the review because it holds a secret"
    );
    Verdict::Skip(SkipCause::Secret)
}

/// What reading one side gave.
enum Side {
    /// The content, or `None` when the side has no such file.
    Content(Option<Vec<u8>>),
    /// A refusal of the source that is a skip.
    Skip(SkipCause),
}

/// Maps a read to its content or to the skip its refusal stands for.
fn read_side(read: Result<Option<Vec<u8>>, ReviewError>) -> Result<Side, ReviewError> {
    match read {
        Ok(bytes) => Ok(Side::Content(bytes)),
        Err(ReviewError::OutsideWorkspace(_)) => Ok(Side::Skip(SkipCause::OutsideWorkspace)),
        Err(ReviewError::TooLarge { .. }) => Ok(Side::Skip(SkipCause::TooLarge)),
        Err(ReviewError::Protected(_)) => Ok(Side::Skip(SkipCause::Ignored)),
        Err(ReviewError::NotRegularFile(_)) => Ok(Side::Skip(SkipCause::NotRegularFile)),
        Err(other) => Err(other),
    }
}

/// The paths `.gitignore` excludes that a review can still reach.
///
/// Paths are relative to `workspace` with `/` separators, the same space as
/// [`ReviewFile::path`]: the production git runs with `-C <workspace>`. History inputs
/// (`Commit`, `Range`, `Branches`) ignore nothing and call no git: `.gitignore` is a worktree
/// rule. Otherwise one `rev-parse --is-inside-work-tree`, then one `ls-files` call: for `File`,
/// tracked and untracked files a rule excludes (a file that does not exist is not listed, so it
/// stays `missing`); for `Directory` and `Uncommitted` (`pathspec` `.`), the tracked ones a rule
/// excludes, which the enumeration lists (untracked ignored files are never enumerated).
/// Outside any repository the set is empty.
///
/// `pathspec` is a literal path, never a glob: the production git sets
/// `GIT_LITERAL_PATHSPECS=1` on every call, so a name holding `*`, `?` or `[` matches only
/// itself. Measured with git 2.56 (reviews of S6 loops 5 and 6 raised it twice): a tracked or
/// untracked ignored file named `a[1].rs` is listed with or without the variable, because git
/// tries the exact name before the glob; without it the glob also names `a1.rs`. Pinned by
/// `a_tracked_ignored_file_named_with_glob_characters_keeps_its_ignore_verdict` and
/// `an_ignored_file_with_glob_characters_in_its_name_keeps_its_ignore_verdict`
/// (`src/system/review_git.rs`).
///
/// # Arguments
///
/// * `input` - What the caller asked to review.
/// * `pathspec` - The workspace-relative path the query is limited to.
/// * `workspace` - Where the `.git` search starts when `rev-parse` fails.
/// * `git` - The review's git.
///
/// # Errors
///
/// [`ReviewError::Git`] when git fails while a `.git` exists in `workspace` or an ancestor (an
/// empty set there would send an ignored file), and any error of `git`.
///
/// O(output bytes) time.
pub fn ignored_paths(
    input: &ReviewInput,
    pathspec: &str,
    workspace: &Path,
    git: &dyn GitCommand,
) -> Result<HashSet<String>, ReviewError> {
    let query = match input {
        ReviewInput::Commit(_) | ReviewInput::Range { .. } | ReviewInput::Branches { .. } => {
            return Ok(HashSet::new())
        }
        ReviewInput::File(_) => LS_FILES_IGNORED_ANY,
        ReviewInput::Directory(_) | ReviewInput::Uncommitted => LS_FILES_IGNORED_TRACKED,
    };
    if !in_work_tree(git, workspace, workspace)? {
        return Ok(HashSet::new());
    }
    let mut args = query.to_vec();
    args.extend_from_slice(&["--", pathspec]);
    // A name that is not UTF-8 is never enumerated as a file to review (the input sets it
    // aside), so it needs no entry here.
    let (names, _not_utf8) = split_nul(&git.run_bytes(&args)?);
    Ok(names.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::MapSource;

    #[test]
    fn a_secret_in_a_whole_file_skips_it_and_an_explicit_one_is_an_error() {
        let text = "const K: &str = \"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let src = MapSource::new(&[("config.rs", text), ("a.rs", "fn f() {}\n")], &[]);
        let files = vec![
            ReviewFile {
                path: "config.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "config.rs".into(),
                cause: SkipCause::Secret
            }]
        );
        let one = vec![ReviewFile {
            path: "config.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "secret",
                ..
            })
        ));
    }

    #[test]
    fn a_path_shaped_like_a_key_is_never_sent() {
        // [CP2 MS2 S1b loop 13] RV22: every batch header carries the path, so the path is scanned like the content.
        // [CP2 MS2 S1b1 loop 16] and the key never leaves through the report, the log or an error either, whatever the cause.
        let name = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", ".rs");
        let shown = "<path redacted: github-token>";
        let src = MapSource::new(&[(name, "fn f() {}\n"), ("a.rs", "fn f() {}\n")], &[]);
        let both = || {
            vec![
                ReviewFile {
                    path: name.into(),
                    diff: None,
                },
                ReviewFile {
                    path: "a.rs".into(),
                    diff: None,
                },
            ]
        };
        let log = crate::logging::testutil::capture(|| {
            let f = filter_files(both(), &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: shown.into(),
                    cause: SkipCause::Secret
                }]
            );
        });
        assert!(!log.contains("AAAAAAAAAAAAAAAA"), "{log}");
        let f = filter_files(both(), &src, &LanguageRegistry::new(), false, &|p| {
            p == name
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: shown.into(),
                cause: SkipCause::Ignored
            }]
        );
        let one = vec![ReviewFile {
            path: name.into(),
            diff: None,
        }];
        match filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false) {
            Err(ReviewError::NotReviewable {
                path,
                cause: "secret",
            }) => assert_eq!(path, shown),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_secret_only_in_the_old_side_skips_the_file() {
        // The diff and the new side are clean; only the revision before the change holds the key.
        // [CP2 MS2 S1b loop 15] a kept sibling makes the skip and its `secret` cause observable (RV-S30, second clause).
        let diff = "@@ -10,1 +10,1 @@\n-fn g() {}\n+fn h() {}\n";
        let old = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";\nfn g() {}\n"; // allow-secret-scan
        let src = MapSource::new(
            &[("a.rs", "fn h() {}\n"), ("b.rs", "fn b() {}\n")],
            &[("a.rs", old)],
        );
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.into()),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("b.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::Secret
            }]
        );
    }

    #[test]
    fn a_secret_on_the_old_side_of_a_renamed_file_skips_it() {
        // [MAGI §6 MS2 A loop 1] a rename's old side lives under its old path: read by the new path it was empty,
        // and a key on a removed line or outside the hunk reached a model.
        let removed = concat!("-", "sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let moved = format!(
            "diff --git a/old.rs b/new.rs\nsimilarity index 90%\nrename from old.rs\nrename to new.rs\n\
             --- a/old.rs\n+++ b/new.rs\n@@ -1,2 +1,1 @@\n fn f() {{}}\n{removed}\n"
        );
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let old_key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\nBBBB\n-----END PRIVATE KEY-----\n");
        let new_key = old_key.replace("\nBBBB\n", "\nAAAA\n");
        let body_only = "diff --git a/k_old.rs b/k_new.rs\nsimilarity index 90%\nrename from k_old.rs\n\
                         rename to k_new.rs\n--- a/k_old.rs\n+++ b/k_new.rs\n@@ -3,1 +3,1 @@\n-BBBB\n+AAAA\n";
        // The new side of k_new.rs has no header either: only its old side holds the `BEGIN` line.
        let new_body = new_key.replace(begin, "");
        let src = MapSource::new(
            &[
                ("new.rs", "fn f() {}\n"),
                ("k_new.rs", new_body.as_str()),
                ("a.rs", "fn a() {}\n"),
            ],
            &[
                ("old.rs", "fn f() {}\nfn g() {}\n"),
                ("k_old.rs", old_key.as_str()),
            ],
        );
        let files = vec![
            ReviewFile {
                path: "new.rs".into(),
                diff: Some(moved),
            },
            ReviewFile {
                path: "k_new.rs".into(),
                diff: Some(body_only.into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "new.rs".into(),
                    cause: SkipCause::Secret
                },
                Skipped {
                    path: "k_new.rs".into(),
                    cause: SkipCause::Secret
                }
            ]
        );
    }

    #[test]
    fn skip_causes_serialize_with_the_spec_texts() {
        let got: Vec<String> = [
            SkipCause::Binary,
            SkipCause::NotUtf8,
            SkipCause::Secret,
            SkipCause::Ignored,
            SkipCause::OutsideWorkspace,
            SkipCause::TooLarge,
            SkipCause::NotRegularFile,
        ]
        .iter()
        .map(|c| serde_json::to_string(c).expect("serializes"))
        .collect();
        assert_eq!(
            got,
            vec![
                "\"binary\"",
                "\"not utf-8\"",
                "\"secret\"",
                "\"ignored\"",
                "\"outside workspace\"",
                "\"too large\"",
                "\"not a regular file\""
            ]
        );
    }

    #[test]
    fn a_pem_key_whose_header_is_outside_the_hunk_still_skips_the_file() {
        // [CP2 MS2 S1b loop 11] the hunk touches only the key body; both whole sides are scanned, so `BEGIN` is seen.
        // Split so tests/no_hardcoded_secrets.rs (which flags the literal `-----BEGIN`) does not read a header here.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let new = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\nAAAA\n-----END PRIVATE KEY-----\n");
        let old = new.replace("\nAAAA\n", "\nBBBB\n");
        let files = vec![
            ReviewFile {
                path: "key.rs".into(),
                diff: Some("@@ -3,1 +3,1 @@\n-BBBB\n+AAAA\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(
            &[("key.rs", new.as_str()), ("a.rs", "fn f() {}\n")],
            &[("key.rs", old.as_str())],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "key.rs".into(),
                cause: SkipCause::Secret
            }]
        );
    }

    #[test]
    fn a_file_over_the_size_cap_is_skipped_and_an_explicit_one_is_an_error() {
        // [CP2 MS2 S1a loop 9] the content source refuses it (T2.3); the filter turns that into `too large`.
        struct Capped;
        impl ContentSource for Capped {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if path == "huge.rs" {
                    Err(ReviewError::TooLarge {
                        path: path.into(),
                        limit: 1,
                    })
                }
                // [S1a loop 10] typed
                else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "huge.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Capped, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "huge.rs".into(),
                cause: SkipCause::TooLarge
            }]
        );
        let one = vec![ReviewFile {
            path: "huge.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Capped, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "too large",
                ..
            })
        ));
    }

    #[test]
    fn an_armored_pgp_private_key_skips_the_file_and_logs_only_its_pattern_name() {
        // Split so tests/no_hardcoded_secrets.rs (which flags the literal `-----BEGIN`) reads no
        // header here; the body is obviously fake.
        let begin = concat!("-----", "BEGIN PGP PRIVATE KEY BLOCK-----");
        let end = concat!("-----", "END PGP PRIVATE KEY BLOCK-----");
        let body = "lQOYBGZ0FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE";
        let text = format!("const K: &str = r\"\n{begin}\n\n{body}\n=FAKE\n{end}\n\";\n");
        let public = concat!("-----", "BEGIN PGP PUBLIC KEY BLOCK-----");
        let pubtext = format!("{public}\n\n{body}\n");
        let src = MapSource::new(
            &[
                ("key.rs", text.as_str()),
                ("pub.asc", pubtext.as_str()),
                ("a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let files = vec![
            ReviewFile {
                path: "key.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "pub.asc".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let line = crate::logging::testutil::capture(|| {
            let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: "key.rs".into(),
                    cause: SkipCause::Secret
                }]
            );
            // A public key is not a secret.
            assert!(f.kept.iter().any(|k| k.path == "pub.asc"), "{f:?}");
        });
        assert!(
            line.contains("key.rs") && line.contains("pgp-private-key"),
            "{line}"
        );
        assert!(!line.contains(body), "{line}");
    }

    #[test]
    fn a_secret_omission_logs_the_path_and_the_pattern_name_never_the_value() {
        // [CP2 MS2 S1b loop 9] the consumer of SecretPattern.name (loop 8), pinned.
        let key = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let text = format!("const K: &str = \"{key}\";\n");
        let src = MapSource::new(
            &[("config.rs", text.as_str()), ("a.rs", "fn f() {}\n")],
            &[],
        );
        let files = vec![
            ReviewFile {
                path: "config.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let line = crate::logging::testutil::capture(|| {
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
        });
        assert!(
            line.contains("config.rs") && line.contains("github-token"),
            "{line}"
        );
        assert!(!line.contains(key), "{line}");
    }

    #[test]
    fn a_file_that_resolves_outside_the_workspace_is_skipped_and_an_explicit_one_is_an_error() {
        struct Escaping;
        impl ContentSource for Escaping {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if path == "link.rs" {
                    Err(ReviewError::OutsideWorkspace(path.into()))
                } else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "link.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Escaping, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "link.rs".into(),
                cause: SkipCause::OutsideWorkspace
            }]
        );
        let one = vec![ReviewFile {
            path: "link.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Escaping, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "outside workspace",
                ..
            })
        ));
    }

    #[test]
    fn a_secret_in_a_removed_line_skips_the_file() {
        // [CP2 MS2 S1b loop 15] with a kept sibling, so the skip and its `secret` cause are asserted, not inferred.
        let diff =
            "@@ -1,2 +1,1 @@\n-const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";\n fn f() {}\n"; // allow-secret-scan
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.into()),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn f() {}\n"), ("b.rs", "fn b() {}\n")], &[]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("b.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::Secret
            }]
        );
    }

    #[test]
    fn binaries_and_non_utf8_are_skipped_with_their_cause_and_unknown_text_is_kept() {
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "logo.png".into(),
                diff: None,
            },
            ReviewFile {
                path: "conf.toml".into(),
                diff: None,
            },
            ReviewFile {
                path: "latin1.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "notes.txt".into(),
                diff: None,
            },
        ];
        let src = MapSource::with_bytes(&[
            ("a.rs", b"fn f() {}\n"),
            ("logo.png", b"\x89PNG\r\n\x1a\n\0\0"),
            ("conf.toml", b"a = 1\n"),
            ("latin1.rs", b"// caf\xe9\n"),
            ("notes.txt", b"plain\n"),
        ]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("some kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs", "conf.toml", "notes.txt"]
        );
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "logo.png".into(),
                    cause: SkipCause::Binary
                },
                Skipped {
                    path: "latin1.rs".into(),
                    cause: SkipCause::NotUtf8
                }
            ]
        );
        // [CP2 MS2 S1 loop 3] config_files.rs is not ported (RV20(3), RV-S14): toml has no extractor either.
        assert_eq!(
            f.no_extractor,
            vec!["conf.toml".to_string(), "notes.txt".to_string()]
        );
    }

    #[test]
    fn an_explicit_binary_file_is_an_error_not_a_skip() {
        let files = vec![ReviewFile {
            path: "logo.png".into(),
            diff: None,
        }];
        let src = MapSource::with_bytes(&[("logo.png", b"\0\0\0")]);
        assert!(matches!(
            filter_files(files, &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "binary",
                ..
            })
        ));
    }

    #[test]
    fn ignored_files_and_never_reviewed_paths_are_left_out() {
        let files = vec![
            ReviewFile {
                path: ".env".into(),
                diff: None,
            },
            ReviewFile {
                path: ".magi/magi.toml".into(),
                diff: None,
            },
            ReviewFile {
                path: "sub/.git".into(),
                diff: None,
            }, // [CP2 MS2 S1b loop 14] a worktree/submodule pointer file
            ReviewFile {
                path: ".magi-rs-memory.db-wal".into(),
                diff: None,
            }, // [S1b1 loop 16] the prefix rule
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        // [CP2 MS2 S1b loop 13] `.env` holds a key: ignored wins over secret (the order above), so its cause is `ignored`.
        let env_text = "K=AKIAABCDEFGHIJKLMNOP\n"; // allow-secret-scan
        let src = MapSource::new(
            &[
                (".env", env_text),
                (".magi/magi.toml", "a=1\n"),
                ("sub/.git", "gitdir: ../.git/modules/sub\n"),
                (".magi-rs-memory.db-wal", "w\n"),
                ("a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == ".env"
        })
        .expect("a.rs kept");
        assert_eq!(f.kept.len(), 1);
        assert!(f.skipped.iter().all(|s| s.cause == SkipCause::Ignored));
    }

    #[test]
    fn the_first_applicable_cause_wins_binary_then_not_utf8_then_secret() {
        let key = b"AKIAABCDEFGHIJKLMNOP"; // allow-secret-scan
        let bin: Vec<u8> = [b"\0\0".as_slice(), key].concat();
        let latin: Vec<u8> = [b"caf\xe9 ".as_slice(), key].concat();
        let files = vec![
            ReviewFile {
                path: "k.bin".into(),
                diff: None,
            },
            ReviewFile {
                path: "k.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src =
            MapSource::with_bytes(&[("k.bin", &bin), ("k.rs", &latin), ("a.rs", b"fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped.iter().map(|s| s.cause).collect::<Vec<_>>(),
            vec![SkipCause::Binary, SkipCause::NotUtf8]
        );
    }

    #[test]
    fn a_binary_section_as_git_prints_it_is_binary_without_reading_the_file() {
        // The section comes from real `git diff` output through the input's own splitter, so it
        // carries its `diff --git` and `index` header lines. `.gitattributes` makes git call the
        // file binary although its bytes are valid UTF-8 with no NUL: only the marker can say so.
        fn git(dir: &Path, args: &[&str]) -> Vec<u8> {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .expect("git runs");
            assert!(out.status.success(), "git {args:?}");
            out.stdout
        }
        let d = tempfile::tempdir().expect("tempdir");
        git(d.path(), &["init", "-q", "-b", "main"]);
        for (k, v) in [
            ("user.email", "t@example.invalid"),
            ("user.name", "t"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ] {
            git(d.path(), &["config", k, v]);
        }
        std::fs::write(d.path().join(".gitattributes"), "*.dat binary\n").expect("write");
        std::fs::write(d.path().join("a.dat"), "hello\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.dat"), "world\n").expect("write");
        let diff = git(
            d.path(),
            &["diff", "--no-color", "--no-ext-diff", "--no-textconv"],
        );
        let (sections, _) = crate::review::input::split_diff(&diff);
        let section = sections
            .into_iter()
            .find(|f| f.path == "a.dat")
            .expect("git lists the binary file");
        assert!(
            section
                .diff
                .as_deref()
                .is_some_and(|s| s.starts_with("diff --git ")),
            "{section:?}"
        );
        let both = || {
            vec![
                ReviewFile {
                    path: section.path.clone(),
                    diff: section.diff.clone(),
                },
                ReviewFile {
                    path: "a.rs".into(),
                    diff: None,
                },
            ]
        };
        let binary = vec![Skipped {
            path: "a.dat".into(),
            cause: SkipCause::Binary,
        }];
        // Never read: a read of a.dat fails the whole review.
        struct Unreadable(SkipOrFail);
        #[derive(Clone, Copy)]
        enum SkipOrFail {
            Fail,
            TooLarge,
        }
        impl ContentSource for Unreadable {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                match (path, self.0) {
                    ("a.dat", SkipOrFail::Fail) => Err(ReviewError::Io("a.dat was read".into())),
                    ("a.dat", SkipOrFail::TooLarge) => Err(ReviewError::TooLarge {
                        path: path.into(),
                        limit: 1,
                    }),
                    _ => Ok(Some(b"fn f() {}\n".to_vec())),
                }
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.read(path)
            }
        }
        let reg = LanguageRegistry::new();
        let f = filter_files(both(), &Unreadable(SkipOrFail::Fail), &reg, false, &|_| {
            false
        })
        .expect("a.dat is not read");
        assert_eq!(f.skipped, binary);
        // Over the cap, it is still `binary`, not `too large`.
        let f = filter_files(
            both(),
            &Unreadable(SkipOrFail::TooLarge),
            &reg,
            false,
            &|_| false,
        )
        .expect("a.rs kept");
        assert_eq!(f.skipped, binary);
        // Valid UTF-8 with no NUL: git's marker still decides.
        let src = MapSource::new(
            &[("a.dat", "world\n"), ("a.rs", "fn f() {}\n")],
            &[("a.dat", "hello\n")],
        );
        let f = filter_files(both(), &src, &reg, false, &|_| false).expect("a.rs kept");
        assert_eq!(f.skipped, binary);
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
    }

    #[test]
    fn a_binary_section_of_a_diff_is_skipped_as_binary() {
        let files = vec![
            ReviewFile {
                path: "logo.png".into(),
                diff: Some("Binary files a/logo.png and b/logo.png differ\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn f() {}\n")], &[]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "logo.png".into(),
                cause: SkipCause::Binary
            }]
        );
    }

    #[test]
    fn an_explicit_ignored_file_is_an_error() {
        let files = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        let src = MapSource::new(&[(".env", "X=1\n")], &[]);
        assert!(matches!(
            filter_files(files, &src, &LanguageRegistry::new(), true, &|p| p
                == ".env"),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn a_tracked_file_that_gitignore_matches_is_left_out_of_a_directory_review() {
        // [CP2 MS2 S1b loop 10] `ls-files --cached` lists tracked files whatever .gitignore says (git 2.56, measured).
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --ignored --exclude-standard -z -- src",
                Ok("src/tracked.env\0"),
            ),
        ]);
        let ignored = ignored_paths(
            &ReviewInput::Directory("src".into()),
            "src",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let files = vec![
            ReviewFile {
                path: "src/tracked.env".into(),
                diff: None,
            },
            ReviewFile {
                path: "src/a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(
            &[
                ("src/tracked.env", "HOST=internal\n"),
                ("src/a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            ignored.contains(p)
        })
        .expect("a.rs kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs"]
        );
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "src/tracked.env".into(),
                cause: SkipCause::Ignored
            }]
        );
    }

    #[test]
    fn a_missing_file_requested_by_name_is_missing_not_ignored() {
        // [CP2 MS2 S1b loop 10] a nonexistent file is not in the ignored set, so `missing` wins.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- gone.rs",
                Ok(""),
            ),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- .env",
                Ok(".env\0"),
            ),
        ]);
        let gone = ignored_paths(
            &ReviewInput::File("gone.rs".into()),
            "gone.rs",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let one = vec![ReviewFile {
            path: "gone.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(
                one,
                &MapSource::new(&[], &[]),
                &LanguageRegistry::new(),
                true,
                &|p| gone.contains(p)
            ),
            Err(ReviewError::NotReviewable {
                cause: "missing",
                ..
            })
        ));
        let env = ignored_paths(
            &ReviewInput::File(".env".into()),
            ".env",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let one = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        let src = MapSource::new(&[(".env", "X=1\n")], &[]);
        assert!(matches!(
            filter_files(one, &src, &LanguageRegistry::new(), true, &|p| env
                .contains(p)),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn history_inputs_and_non_repos_ignore_nothing() {
        // [CP2 MS2 S1b loop 10] .gitignore is a worktree rule; an empty FakeGit fails any call it receives.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let none = FakeGit::from(&[]);
        for input in [
            ReviewInput::Commit(None),
            ReviewInput::Range {
                from: "a".into(),
                to: "b".into(),
            },
            ReviewInput::Branches {
                base: "a".into(),
                head: "b".into(),
            },
        ] {
            assert!(ignored_paths(&input, ".", Path::new("/w"), &none)
                .expect("no git call")
                .is_empty());
        }
        let outside = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        let no_repo = tempfile::tempdir().expect("tempdir"); // no .git here or above
                                                             // [CP2 MS2 S1b loop 13] the assumption, made visible: a `.git` above TMPDIR makes the rule (correctly) fail closed.
        assert!(
            no_repo.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        assert!(ignored_paths(
            &ReviewInput::Directory(".".into()),
            ".",
            no_repo.path(),
            &outside
        )
        .expect("not a repo")
        .is_empty());
    }

    #[test]
    fn history_inputs_ignore_nothing_even_inside_a_repository_whose_rules_match() {
        // The test above cannot see a git call for a history input: its empty FakeGit fails the
        // call, and outside a repository that failure reads as "ignore nothing". Here git
        // answers, and would name an ignored file, so only skipping git keeps the set empty.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let answering = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --ignored --exclude-standard -z -- .",
                Ok(".env\0"),
            ),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- .",
                Ok(".env\0"),
            ),
        ]);
        for input in [
            ReviewInput::Commit(None),
            ReviewInput::Range {
                from: "a".into(),
                to: "b".into(),
            },
            ReviewInput::Branches {
                base: "a".into(),
                head: "b".into(),
            },
        ] {
            let ignored = ignored_paths(&input, ".", Path::new("/w"), &answering).expect("no git");
            assert!(ignored.is_empty(), "{input:?}: {ignored:?}");
        }
    }

    #[test]
    fn a_git_failure_inside_a_repository_is_an_error_not_an_empty_ignore_set() {
        // [CP2 MS2 S1a loop 11] an empty set here would review an explicitly requested ignored `.env`.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join(".git"), "gitdir: elsewhere\n")
            .expect("a worktree's .git file");
        let broken = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        assert!(matches!(
            ignored_paths(&ReviewInput::File(".env".into()), ".env", d.path(), &broken),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_diff_file_without_an_extractor_is_declared_too() {
        // [CP2 MS2 S1b loop 10] RV20(3) covers files reviewed by diff, not only whole files.
        let files = vec![ReviewFile {
            path: "conf.toml".into(),
            diff: Some("@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n".into()),
        }];
        let src = MapSource::new(&[("conf.toml", "a = 2\n")], &[("conf.toml", "a = 1\n")]);
        let f =
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert_eq!(
            (f.kept.len(), f.no_extractor.clone()),
            (1, vec!["conf.toml".to_string()])
        );
    }

    #[test]
    fn a_missing_whole_file_is_never_skipped_silently() {
        let src = MapSource::new(&[], &[]);
        let one = || {
            vec![ReviewFile {
                path: "gone.rs".into(),
                diff: None,
            }]
        };
        assert!(matches!(
            filter_files(one(), &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "missing",
                ..
            })
        ));
        assert!(matches!(
            filter_files(one(), &src, &LanguageRegistry::new(), false, &|_| false),
            Err(ReviewError::Io(_))
        ));
        // With a diff, an absent new side is a deletion and is kept.
        let del = vec![ReviewFile {
            path: "gone.rs".into(),
            diff: Some("@@ -1,1 +0,0 @@\n-x\n".into()),
        }];
        let src_old = MapSource::new(&[], &[("gone.rs", "x\n")]);
        assert_eq!(
            filter_files(del, &src_old, &LanguageRegistry::new(), false, &|_| false)
                .expect("kept")
                .kept
                .len(),
            1
        );
    }

    #[test]
    fn an_empty_directory_is_nothing_to_review() {
        let src = MapSource::new(&[], &[]);
        assert!(matches!(
            filter_files(Vec::new(), &src, &LanguageRegistry::new(), false, &|_| {
                false
            }),
            Err(ReviewError::NothingToReview { .. })
        ));
    }

    #[test]
    fn a_non_empty_input_where_every_file_is_omitted_is_nothing_to_review() {
        // [CP2 MS2 S1b1 loop 16] RV20: the emptiness that counts is the kept list's, after every file was classified.
        let key = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";\n"; // allow-secret-scan
        let src = MapSource::new(&[(".env", "X=1\n"), ("a.rs", key), ("b.rs", key)], &[]);
        let ignored = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(ignored, &src, &LanguageRegistry::new(), false, &|p| p
                == ".env"),
            Err(ReviewError::NothingToReview { .. })
        ));
        let secrets = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        assert!(matches!(
            filter_files(secrets, &src, &LanguageRegistry::new(), false, &|_| false),
            Err(ReviewError::NothingToReview { .. })
        ));
    }

    #[test]
    fn an_all_omitted_input_keeps_every_omission_and_its_cause_in_the_error() {
        // [MAGI S6 loop 3, A2] RV20/RV22: with nothing kept, the omissions are still reported.
        let key = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";
"; // allow-secret-scan
        let src = MapSource::new(
            &[
                (
                    ".env", "X=1
",
                ),
                ("a.rs", key),
                ("b.bin", "a b"),
            ],
            &[],
        );
        let files = [".env", "a.rs", "b.bin"]
            .iter()
            .map(|p| ReviewFile {
                path: (*p).into(),
                diff: None,
            })
            .collect();
        let e = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == ".env"
        })
        .expect_err("nothing kept");
        let ReviewError::NothingToReview { skipped } = &e else {
            panic!("expected NothingToReview, got {e:?}");
        };
        let causes: Vec<(&str, SkipCause)> =
            skipped.iter().map(|s| (s.path.as_str(), s.cause)).collect();
        assert_eq!(
            causes,
            vec![
                (".env", SkipCause::Ignored),
                ("a.rs", SkipCause::Secret),
                ("b.bin", SkipCause::Binary)
            ]
        );
        let shown = e.to_string();
        for part in [".env (ignored)", "a.rs (secret)", "b.bin (binary)"] {
            assert!(shown.contains(part), "{shown}");
        }
    }

    #[test]
    fn a_file_whose_target_is_protected_is_ignored_and_an_explicit_one_is_an_error() {
        // [CP2 MS2 S1b1 loop 16] T2.3's Protected: a symlink alias of a never-reviewed path.
        struct Aliased;
        impl ContentSource for Aliased {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if p == "notes.rs" {
                    Err(ReviewError::Protected(p.into()))
                } else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "notes.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Aliased, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "notes.rs".into(),
                cause: SkipCause::Ignored
            }]
        );
        let one = vec![ReviewFile {
            path: "notes.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Aliased, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn a_text_diff_that_quotes_the_binary_marker_is_still_text() {
        // [CP2 MS2 S1b1 loop 16] the marker is the whole section, never a line inside a hunk.
        let diff = "@@ -1,1 +1,2 @@\n x\n+Binary files a/logo.png and b/logo.png differ\n";
        let files = vec![ReviewFile {
            path: "notes.md".into(),
            diff: Some(diff.into()),
        }];
        let src = MapSource::new(
            &[(
                "notes.md",
                "x\nBinary files a/logo.png and b/logo.png differ\n",
            )],
            &[("notes.md", "x\n")],
        );
        let f =
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert!(f.skipped.is_empty() && f.kept.len() == 1, "{f:?}");
    }
    #[test]
    fn an_old_side_that_is_binary_or_not_utf8_skips_the_file_even_if_the_new_one_is_text() {
        // [MAGI S6 MS2 loop 4, A2 Caspar] the checks ran on the new side only, so a text file
        // whose previous revision was binary or Latin-1 carried old-side bytes into the review.
        struct OldBytes;
        impl ContentSource for OldBytes {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(b"fn h() {}\n".to_vec()))
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(match path {
                    "bin.rs" => b"fn g() {}\0\n".to_vec(),
                    "latin1.rs" => b"// caf\xe9\nfn g() {}\n".to_vec(),
                    _ => b"fn g() {}\n".to_vec(),
                }))
            }
        }
        let diff = "@@ -2,1 +1,1 @@\n-fn g() {}\n+fn h() {}\n";
        let files = ["bin.rs", "latin1.rs", "a.rs"]
            .into_iter()
            .map(|p| ReviewFile {
                path: p.into(),
                diff: Some(diff.into()),
            })
            .collect();
        let f = filter_files(files, &OldBytes, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "bin.rs".into(),
                    cause: SkipCause::Binary
                },
                Skipped {
                    path: "latin1.rs".into(),
                    cause: SkipCause::NotUtf8
                }
            ]
        );
    }

    #[test]
    fn a_diff_file_with_neither_side_is_skipped_as_not_a_regular_file() {
        // [MAGI S6 loop 8, A2-c1 Caspar] a submodule's gitlink reads as absent on both sides
        // (`RevisionSource::holds`), so a diff section that reached the filter in that shape was
        // kept with no content and no cause. `split_diff` drops gitlink sections first; this pins
        // the filter's own answer for whatever brings that shape.
        let files = vec![
            ReviewFile {
                path: "vendored".into(),
                diff: Some("@@ -1 +1 @@\n-Subproject commit 1\n+Subproject commit 2\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: Some("@@ -1 +1 @@\n-fn f() {}\n+fn g() {}\n".into()),
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn g() {}\n")], &[("a.rs", "fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "vendored".into(),
                cause: SkipCause::NotRegularFile
            }]
        );
    }

    #[test]
    fn a_nul_anywhere_in_a_file_makes_it_binary() {
        // [MAGI S6 loop 7, A2-c2] only the first 8 000 bytes were sniffed: a text prefix hid a
        // binary payload, which is valid UTF-8 and was batched (RV20).
        let mut late: Vec<u8> = vec![b'a'; 9_000];
        late.extend_from_slice(b"\0payload\n");
        let files = vec![
            ReviewFile {
                path: "late.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::with_bytes(&[("late.rs", &late), ("a.rs", b"fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "late.rs".into(),
                cause: SkipCause::Binary,
            }]
        );
    }

    #[test]
    fn a_rename_from_a_never_reviewed_or_ignored_path_is_ignored() {
        // [MAGI S6 loop 7, A2-m2] only the new path met the name rule: a crafted commit carried a
        // blob stored under `.git/` or an ignored name into the review as a rename's old side.
        let rename = |from: &str, to: &str| {
            format!(
                "diff --git a/{from} b/{to}\nsimilarity index 90%\nrename from {from}\n\
                 rename to {to}\n--- a/{from}\n+++ b/{to}\n@@ -1,1 +1,1 @@\n-fn g() {{}}\n+fn h() {{}}\n"
            )
        };
        let src = MapSource::new(
            &[
                ("x.rs", "fn h() {}\n"),
                ("y.rs", "fn h() {}\n"),
                ("a.rs", "fn a() {}\n"),
            ],
            &[
                (".git/hooks/x.rs", "fn g() {}\n"),
                ("local.env", "fn g() {}\n"),
            ],
        );
        let files = vec![
            ReviewFile {
                path: "x.rs".into(),
                diff: Some(rename(".git/hooks/x.rs", "x.rs")),
            },
            ReviewFile {
                path: "y.rs".into(),
                diff: Some(rename("local.env", "y.rs")),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == "local.env"
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped
                .iter()
                .map(|s| (s.path.as_str(), s.cause))
                .collect::<Vec<_>>(),
            vec![("x.rs", SkipCause::Ignored), ("y.rs", SkipCause::Ignored)]
        );
    }
}
