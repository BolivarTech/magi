// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Packs the reviewed content into batches that fit the pool window.

use crate::review::error::ReviewError;
use crate::review::input::ReviewFile;
use crate::review::languages::LanguageRegistry;
use crate::review::source::ContentSource;

/// One batch: what one `consult code-review` receives.
#[derive(Debug)]
pub struct ReviewBatch {
    /// Position of the batch in the review, from 0.
    pub index: usize,
    /// What the batch reviews, in input order.
    pub items: Vec<BatchItem>,
    /// Estimated tokens of the items plus one header per distinct file.
    pub tokens: usize,
}

/// One piece of reviewed content.
#[derive(Debug)]
pub enum BatchItem {
    /// One diff hunk with its file header.
    Hunk {
        /// File path.
        file: String,
        /// The hunk text.
        text: String,
        /// First line on the new side.
        first_line: u32,
        /// Last line on the new side.
        last_line: u32,
        /// First line on the old side.
        old_first_line: u32,
        /// Last line on the old side.
        old_last_line: u32,
    },
    /// One unit of a whole file that did not fit.
    Unit {
        /// File path.
        file: String,
        /// Unit name.
        name: String,
        /// Unit text.
        text: String,
        /// First line.
        first_line: u32,
        /// Last line.
        last_line: u32,
    },
    /// A whole file.
    WholeFile {
        /// File path.
        file: String,
        /// File text.
        text: String,
        /// Number of lines.
        lines: u32,
    },
}

/// What an item drags into its batch, as `(key, tokens)`.
pub type Companion = (String, usize);

/// The per-file header every batch prompt carries.
pub fn file_header(path: &str) -> String {
    let _ = path;
    todo!()
}

/// Packs `files` into batches of at most `limit_tokens`.
///
/// # Errors
///
/// See [`plan_batches_with`].
pub fn plan_batches(
    files: &[ReviewFile],
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    limit_tokens: usize,
) -> Result<Vec<ReviewBatch>, ReviewError> {
    let _ = (files, source, registry, limit_tokens);
    todo!()
}

/// Packs `files` into batches, charging each item's companion once per batch.
///
/// # Errors
///
/// Stub.
pub fn plan_batches_with(
    files: &[ReviewFile],
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    limit_tokens: usize,
    companion: &dyn Fn(&BatchItem) -> Result<Option<Companion>, ReviewError>,
) -> Result<Vec<ReviewBatch>, ReviewError> {
    let _ = (files, source, registry, limit_tokens, companion);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::window::estimate_tokens;

    use crate::review::test_support::MapSource;

    #[test]
    fn hunks_are_packed_greedily_and_a_new_batch_starts_when_the_next_does_not_fit() {
        // Each hunk is ~170 bytes (~43 tokens): two fit in 100 tokens, three do not.
        let hunk = || format!("@@ -1,1 +1,1 @@\n+{}\n", "x".repeat(150));
        let files: Vec<ReviewFile> = ["a.rs", "b.rs", "c.rs"]
            .iter()
            .map(|p| ReviewFile {
                path: p.to_string(),
                diff: Some(hunk()),
            })
            .collect();
        let src = MapSource::new(&[("a.rs", "x\n"), ("b.rs", "x\n"), ("c.rs", "x\n")], &[]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 100).expect("plans");
        assert_eq!(
            b.iter().map(|x| x.items.len()).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(
            b.iter().all(|x| x.tokens <= 100),
            "{:?}",
            b.iter().map(|x| x.tokens).collect::<Vec<_>>()
        );
        assert_eq!(b.iter().map(|x| x.index).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn a_hunk_carries_both_sides_and_a_deletion_has_an_empty_new_side() {
        // [CP2 MS2 S1 loop 2] a finding on a removed line uses the old numbering.
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some("@@ -10,3 +12,2 @@\n x\n-y\n z\n".into()),
            },
            ReviewFile {
                path: "gone.rs".into(),
                diff: Some("@@ -1,4 +0,0 @@\n-a\n-b\n-c\n-d\n".into()),
            },
        ];
        let src = MapSource::new(&[("a.rs", "x\nz\n")], &[("gone.rs", "a\nb\nc\nd\n")]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 1_000).expect("plans");
        let sides: Vec<(u32, u32, u32, u32)> = b[0]
            .items
            .iter()
            .map(|i| match i {
                BatchItem::Hunk {
                    first_line,
                    last_line,
                    old_first_line,
                    old_last_line,
                    ..
                } => (*first_line, *last_line, *old_first_line, *old_last_line),
                other => panic!("expected hunks, got {other:?}"),
            })
            .collect();
        assert_eq!(sides, vec![(12, 13, 10, 12), (1, 0, 1, 4)]);
    }

    #[test]
    fn a_batch_counts_the_header_of_each_of_its_files_once() {
        // [CP2 MS2 S1 loop 3] RV04 bounds what the trio receives, header included.
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some("@@ -1,1 +1,1 @@\n+x\n@@ -5,1 +5,1 @@\n+y\n".into()),
        }];
        let src = MapSource::new(&[("a.rs", "x\n")], &[]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 1_000).expect("plans");
        let items: usize = b[0]
            .items
            .iter()
            .map(|i| match i {
                BatchItem::Hunk { text, .. }
                | BatchItem::Unit { text, .. }
                | BatchItem::WholeFile { text, .. } => estimate_tokens(text),
            })
            .sum();
        assert_eq!(b[0].tokens, items + estimate_tokens(&file_header("a.rs")));
        assert_eq!(file_header("src/a.rs"), "### src/a.rs (rust)\n");
        assert_eq!(file_header("notes.txt"), "### notes.txt\n"); // [CP2 MS2 S1 loop 5] no language claimed
    }

    #[test]
    fn the_fit_test_and_the_too_large_threshold_count_the_incoming_files_header() {
        // [CP2 MS2 S1b loop 9] with a limit of two items plus ONE header, three files do not pair up: each pair needs two
        // headers. A header-blind fit packs two and goes over the limit.
        let hunk = || format!("@@ -1,1 +1,1 @@\n+{}\n", "x".repeat(150));
        let files: Vec<ReviewFile> = ["a.rs", "b.rs", "c.rs"]
            .iter()
            .map(|p| ReviewFile {
                path: p.to_string(),
                diff: Some(hunk()),
            })
            .collect();
        let src = MapSource::new(&[("a.rs", "x\n"), ("b.rs", "x\n"), ("c.rs", "x\n")], &[]);
        let reg = LanguageRegistry::new();
        let hdr = estimate_tokens(&file_header("a.rs"));
        let item = plan_batches(&files[..1], &src, &reg, 10_000).expect("plans")[0].tokens - hdr;
        let limit = 2 * item + hdr;
        let b = plan_batches(&files, &src, &reg, limit).expect("plans");
        assert_eq!(
            b.iter().map(|x| x.items.len()).collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        assert!(b.iter().all(|x| x.tokens <= limit));
        match plan_batches(&files[..1], &src, &reg, item + hdr - 1) {
            Err(ReviewError::TooLargeForBatch { tokens, limit, .. }) => {
                assert_eq!((tokens, limit), (item + hdr, item + hdr - 1))
            }
            other => panic!("expected TooLargeForBatch, got {other:?}"),
        }
    }

    #[test]
    fn a_whole_file_that_vanished_before_batching_is_an_io_error() {
        // [CP2 MS2 S1b loop 9] T2.4's rule, repeated at batching time: never a silent drop.
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "gone.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn f() {}\n")], &[]);
        assert!(matches!(
            plan_batches(&files, &src, &LanguageRegistry::new(), 1_000),
            Err(ReviewError::Io(_))
        ));
    }

    #[test]
    fn a_file_that_holds_a_secret_at_batching_time_is_never_batched() {
        // [CP2 MS2 S1b loop 10] RV22 at the egress point: T2.4 scanned an earlier read; these are the bytes that would go out.
        let key = "const K: &str = \"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("a.rs", key)], &[]);
        match plan_batches(&files, &src, &LanguageRegistry::new(), 1_000) {
            Err(ReviewError::EgressSecret { path }) => assert_eq!(path, "a.rs"), // [S1b2 loop 16] typed
            other => panic!("expected EgressSecret, got {other:?}"),
        }
    }

    #[test]
    fn the_egress_scan_covers_the_file_header_too() {
        // [CP2 MS2 S1b loop 13] the header carries the path; T2.4 skips such a path, and this is the net at egress.
        let name = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", ".rs");
        let files = vec![ReviewFile {
            path: name.into(),
            diff: None,
        }];
        let src = MapSource::new(&[(name, "fn f() {}\n")], &[]);
        assert!(
            matches!(plan_batches(&files, &src, &LanguageRegistry::new(), 1_000),
                     Err(ReviewError::EgressSecret { ref path }) if path == "<path redacted: github-token>")
        );
    }

    #[test]
    fn the_egress_scan_covers_hunk_and_unit_items_too() {
        // [CP2 MS2 S1b loop 12] RV22 at egress for every item shape, not only whole files.
        let key = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let hunk = format!("@@ -1,1 +1,1 @@\n-x\n+let k = \"{key}\";\n");
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(hunk),
        }];
        let src = MapSource::new(&[("a.rs", "x\n")], &[("a.rs", "x\n")]);
        assert!(
            matches!(plan_batches(&files, &src, &LanguageRegistry::new(), 1_000),
                     Err(ReviewError::EgressSecret { ref path }) if path == "a.rs")
        );
        let body = |v: &str| format!("    let _{v} = 1;\n").repeat(14);
        let text = format!(
            "fn f() {{\n{}}}\n\nfn g() {{\n    let k = \"{key}\";\n{}}}\n",
            body("a"),
            body("b")
        );
        let files = vec![ReviewFile {
            path: "b.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("b.rs", &text)], &[]);
        assert!(
            matches!(plan_batches(&files, &src, &LanguageRegistry::new(), 100),
                     Err(ReviewError::EgressSecret { ref path }) if path == "b.rs")
        );
    }

    #[test]
    fn a_whole_file_without_a_trailing_newline_counts_its_last_line() {
        // [CP2 MS2 S1b loop 12] a '\n' count would give 1 here, and line 2 would fall out of Reviewed (U24).
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("a.rs", "fn f() {}\nfn g() {}")], &[]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 1_000).expect("plans");
        assert!(
            matches!(
                b[0].items.as_slice(),
                [BatchItem::WholeFile { lines: 2, .. }]
            ),
            "{:?}",
            b[0].items
        );
    }

    #[test]
    fn only_empty_files_are_nothing_to_review() {
        let files = vec![ReviewFile {
            path: "empty.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("empty.rs", "")], &[]);
        assert!(matches!(
            plan_batches(&files, &src, &LanguageRegistry::new(), 100),
            Err(ReviewError::NothingToReview)
        ));
    }

    #[test]
    fn a_whole_file_that_does_not_fit_is_split_by_units() {
        // Two functions of ~60 tokens each: the whole file (~120) does not fit in 100, each unit does.
        let body = |v: &str| format!("    let _{v} = 1;\n").repeat(14);
        let text = format!("fn f() {{\n{}}}\n\nfn g() {{\n{}}}\n", body("a"), body("b"));
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("a.rs", &text)], &[]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 100).expect("plans");
        let names: Vec<&str> = b
            .iter()
            .flat_map(|x| x.items.iter())
            .map(|i| match i {
                BatchItem::Unit { name, .. } => name.as_str(),
                other => panic!("expected units, got {other:?}"),
            })
            .collect();
        assert_eq!(names, vec!["f", "g"]);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn a_single_unit_larger_than_the_limit_is_a_typed_error() {
        let text = format!("fn big() {{\n{}}}\n", "    let _x = 1;\n".repeat(50)); // ~200 tokens
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("a.rs", &text)], &[]);
        match plan_batches(&files, &src, &LanguageRegistry::new(), 100) {
            Err(ReviewError::TooLargeForBatch {
                tokens,
                limit,
                what,
            }) => {
                assert_eq!(limit, 100);
                assert!(tokens > limit, "{tokens}");
                assert!(what.contains("big"), "{what}");
            }
            other => panic!("expected TooLargeForBatch, got {other:?}"),
        }
    }

    #[test]
    fn a_single_hunk_larger_than_the_limit_is_a_typed_error_naming_its_lines() {
        let diff = format!("@@ -1,1 +7,1 @@\n+{}\n", "x".repeat(800)); // ~200 tokens
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(diff),
        }];
        let src = MapSource::new(&[("a.rs", "x\n")], &[]);
        match plan_batches(&files, &src, &LanguageRegistry::new(), 100) {
            Err(ReviewError::TooLargeForBatch { what, limit, .. }) => {
                assert_eq!(limit, 100);
                assert!(what.starts_with("a.rs:7"), "{what}");
            }
            other => panic!("expected TooLargeForBatch, got {other:?}"),
        }
    }

    #[test]
    fn a_whole_file_without_an_extractor_that_does_not_fit_is_a_typed_error() {
        let text = "plain line of prose\n".repeat(40); // ~200 tokens, no extractor for .txt
        let files = vec![ReviewFile {
            path: "notes.txt".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("notes.txt", &text)], &[]);
        assert!(matches!(
            plan_batches(&files, &src, &LanguageRegistry::new(), 100),
            Err(ReviewError::TooLargeForBatch { ref what, .. }) if what == "notes.txt"
        ));
    }

    #[test]
    fn a_companion_is_charged_once_per_batch_against_the_full_limit() {
        // [enmienda autorizada por el usuario 2026-10-20 (CP2 MS3 loop 12, opción 1)] Two hunks of one unit share their
        // companion; a hunk of another unit opens a batch; a hunk that cannot carry its companion alone is too large.
        let hunk = |n: u32| format!("@@ -{n},1 +{n},1 @@\n+{}\n", "x".repeat(150)); // ~43 tokens each
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(format!("{}{}{}", hunk(1), hunk(2), hunk(30))),
        }];
        let one = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(hunk(1)),
        }];
        let src = MapSource::new(&[("a.rs", "x\n")], &[]);
        let reg = LanguageRegistry::new();
        let hdr = estimate_tokens(&file_header("a.rs"));
        let item = plan_batches(&one, &src, &reg, 10_000).expect("plans")[0].tokens - hdr;
        let unit = |i: &BatchItem| -> Result<Option<Companion>, ReviewError> {
            Ok(match i {
                BatchItem::Hunk { first_line, .. } if *first_line < 10 => {
                    Some(("a.rs:1-9".to_string(), 50))
                }
                BatchItem::Hunk { .. } => Some(("a.rs:30-39".to_string(), 50)),
                _ => None,
            })
        };
        let limit = hdr + 2 * item + 50; // two hunks and ONE copy of their shared companion
        let b = plan_batches_with(&files, &src, &reg, limit, &unit).expect("plans");
        assert_eq!(
            b.iter().map(|x| x.items.len()).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(
            b.iter().all(|x| x.tokens + 50 <= limit),
            "{:?}",
            b.iter().map(|x| x.tokens).collect::<Vec<_>>()
        );
        assert_eq!(
            plan_batches(&files, &src, &reg, limit)
                .expect("plans")
                .len(),
            1,
            "no companion: as before"
        );
        match plan_batches_with(&one, &src, &reg, hdr + item + 49, &unit) {
            Err(ReviewError::TooLargeForBatch { tokens, .. }) => {
                assert_eq!(tokens, hdr + item + 50)
            }
            other => panic!("expected TooLargeForBatch, got {other:?}"),
        }
    }
}
