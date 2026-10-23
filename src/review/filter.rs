// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What never leaves the process and what is not code: the per-file filter of a review.

use std::collections::HashSet;
use std::path::Path;

use super::error::ReviewError;
use super::input::{ReviewFile, ReviewInput};
use super::languages::LanguageRegistry;
use super::source::{ContentSource, GitCommand};

/// A file left out of a review, with why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Skipped {
    /// The path, as it may leave the process.
    pub path: String,
    /// Why it was left out.
    pub cause: SkipCause,
}

/// Why a file was left out of a review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SkipCause {
    /// A binary file.
    #[serde(rename = "binary")]
    Binary,
    /// Text that is not valid UTF-8.
    #[serde(rename = "not utf-8")]
    NotUtf8,
    /// A file that holds a secret.
    #[serde(rename = "secret")]
    Secret,
    /// A file `.gitignore` excludes, or one a review never reads.
    #[serde(rename = "ignored")]
    Ignored,
    /// A file that resolves outside the workspace.
    #[serde(rename = "outside workspace")]
    OutsideWorkspace,
    /// A file over the per-file byte cap.
    #[serde(rename = "too large")]
    TooLarge,
}

/// The outcome of filtering.
#[derive(Debug)]
pub struct Filtered {
    /// The files that are reviewed.
    pub kept: Vec<ReviewFile>,
    /// The files left out, with their cause.
    pub skipped: Vec<Skipped>,
    /// Kept files with no language extractor.
    pub no_extractor: Vec<String>,
}

/// Splits the files of a review into what is reviewed and what is left out.
pub fn filter_files(
    files: Vec<ReviewFile>,
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    explicit: bool,
    is_ignored: &dyn Fn(&str) -> bool,
) -> Result<Filtered, ReviewError> {
    let _ = (files, source, registry, explicit, is_ignored);
    todo!()
}

/// The paths `.gitignore` excludes that a review can still reach.
pub fn ignored_paths(
    input: &ReviewInput,
    pathspec: &str,
    workspace: &Path,
    git: &dyn GitCommand,
) -> Result<HashSet<String>, ReviewError> {
    let _ = (input, pathspec, workspace, git);
    todo!()
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
    fn skip_causes_serialize_with_the_spec_texts() {
        let got: Vec<String> = [
            SkipCause::Binary,
            SkipCause::NotUtf8,
            SkipCause::Secret,
            SkipCause::Ignored,
            SkipCause::OutsideWorkspace,
            SkipCause::TooLarge,
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
                "\"too large\""
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
            Err(ReviewError::NothingToReview)
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
            Err(ReviewError::NothingToReview)
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
            Err(ReviewError::NothingToReview)
        ));
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
}
