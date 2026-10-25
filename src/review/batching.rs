// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Packs the reviewed content into batches that fit the pool window.
//!
//! Each batch is what one `consult code-review` receives: hunks of a diff, whole files, or the
//! units of a whole file too large for one batch, each file preceded by its [`file_header`].
//! Packing is greedy in input order. This is also the egress point of the review: every byte a
//! batch carries passes the secret detector here, whatever an earlier filter decided.

use std::collections::HashSet;

use crate::review::context::detect_language;
use crate::review::error::ReviewError;
use crate::review::hunk::parse_hunks;
use crate::review::input::ReviewFile;
use crate::review::languages::{LanguageExtractor, LanguageRegistry};
use crate::review::secrets::{display_path, find_secret};
use crate::review::source::ContentSource;
use crate::review::window::estimate_tokens;

/// The language name [`detect_language`] gives a file it does not recognise.
const UNKNOWN_LANGUAGE: &str = "unknown";

/// The marker that opens a hunk header line (`@@ -a,b +c,d @@`).
const HUNK_MARKER: &str = "@@";

/// One batch: what one `consult code-review` receives.
#[derive(Debug)]
pub struct ReviewBatch {
    /// Position of the batch in the review, from 0.
    pub index: usize,
    /// What the batch reviews, in input order.
    pub items: Vec<BatchItem>,
    /// Estimated tokens of the items plus one [`file_header`] per distinct file of the batch.
    /// Companions ([`plan_batches_with`]) are not included: whoever adds them charges them.
    pub tokens: usize,
}

/// One piece of reviewed content.
#[derive(Debug)]
pub enum BatchItem {
    /// One diff hunk, with the diff's file header lines before it.
    Hunk {
        /// File path, relative to the workspace.
        file: String,
        /// The hunk text as it is sent.
        text: String,
        /// First line on the new side (`c` of `+c,d`).
        first_line: u32,
        /// Last line on the new side (`c + d − 1`); below `first_line` when the side is empty.
        last_line: u32,
        /// First line on the old side (`a` of `-a,b`).
        old_first_line: u32,
        /// Last line on the old side (`a + b − 1`); below `old_first_line` when the side is
        /// empty.
        old_last_line: u32,
    },
    /// One unit (function, struct, class, …) of a whole file that did not fit in one batch.
    Unit {
        /// File path, relative to the workspace.
        file: String,
        /// The unit's name, as the language extractor reports it.
        name: String,
        /// The unit text.
        text: String,
        /// First line of the unit (1-based, inclusive).
        first_line: u32,
        /// Last line of the unit (1-based, inclusive).
        last_line: u32,
    },
    /// A whole file.
    WholeFile {
        /// File path, relative to the workspace.
        file: String,
        /// The file text.
        text: String,
        /// Its line count, `text.lines().count()`: a last line without `'\n'` counts.
        lines: u32,
    },
}

impl BatchItem {
    /// The file the item belongs to.
    fn file(&self) -> &str {
        match self {
            Self::Hunk { file, .. } | Self::Unit { file, .. } | Self::WholeFile { file, .. } => {
                file
            }
        }
    }

    /// The text the item sends.
    fn text(&self) -> &str {
        match self {
            Self::Hunk { text, .. } | Self::Unit { text, .. } | Self::WholeFile { text, .. } => {
                text
            }
        }
    }

    /// How a [`ReviewError::TooLargeForBatch`] names the item: the path for a whole file,
    /// `<path>:<first>-<last>` for a hunk (new side), and the same plus the name for a unit.
    fn describe(&self) -> String {
        match self {
            Self::Hunk {
                file,
                first_line,
                last_line,
                ..
            } => format!("{}:{first_line}-{last_line}", display_path(file)),
            Self::Unit {
                file,
                name,
                first_line,
                last_line,
                ..
            } => format!("{}:{first_line}-{last_line} ({name})", display_path(file)),
            Self::WholeFile { file, .. } => display_path(file),
        }
    }
}

/// What an item drags into its batch, as `(key, tokens)`: charged once per batch per key.
pub type Companion = (String, usize);

/// The per-file header every batch prompt carries: `### <path> (<language>)\n`, or
/// `### <path>\n` when [`detect_language`] does not know the language.
///
/// # Arguments
///
/// * `path` - The file path, relative to the workspace.
///
/// # Returns
///
/// The header line, newline included.
///
/// # Examples
///
/// ```
/// use magi_rs::review::batching::file_header;
///
/// assert_eq!(file_header("src/a.rs"), "### src/a.rs (rust)\n");
/// assert_eq!(file_header("notes.txt"), "### notes.txt\n");
/// ```
pub fn file_header(path: &str) -> String {
    let language = detect_language(path);
    if language == UNKNOWN_LANGUAGE {
        format!("### {path}\n")
    } else {
        format!("### {path} ({language})\n")
    }
}

/// Packs `files` into batches of at most `limit_tokens`, with no companion.
///
/// Equivalent to [`plan_batches_with`] with a companion that is always `None`.
///
/// # Arguments
///
/// * `files` - The files to review, in review order.
/// * `source` - The single content source of the review (the one the filters read).
/// * `registry` - The language extractors, to split a whole file that does not fit.
/// * `limit_tokens` - The per-batch limit in tokens.
///
/// # Returns
///
/// The batches, numbered from 0, never empty.
///
/// # Errors
///
/// As [`plan_batches_with`].
///
/// # Examples
///
/// ```
/// use magi_rs::review::batching::plan_batches;
/// use magi_rs::review::error::ReviewError;
/// use magi_rs::review::input::ReviewFile;
/// use magi_rs::review::languages::LanguageRegistry;
/// use magi_rs::review::source::ContentSource;
///
/// struct OneFile;
/// impl ContentSource for OneFile {
///     fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
///         Ok(Some(b"fn f() {}\n".to_vec()))
///     }
///     fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
///         Ok(None)
///     }
/// }
///
/// let files = vec![ReviewFile { path: "a.rs".into(), diff: None }];
/// let batches = plan_batches(&files, &OneFile, &LanguageRegistry::new(), 1_000)?;
/// assert_eq!((batches.len(), batches[0].items.len()), (1, 1));
/// # Ok::<(), ReviewError>(())
/// ```
pub fn plan_batches(
    files: &[ReviewFile],
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    limit_tokens: usize,
) -> Result<Vec<ReviewBatch>, ReviewError> {
    plan_batches_with(files, source, registry, limit_tokens, &|_| Ok(None))
}

/// Packs `files` into batches of at most `limit_tokens`, charging each item's companion once
/// per batch.
///
/// Items are packed greedily in input order: a diff gives one item per hunk; a file reviewed
/// whole gives one item, or, when it does not fit in an empty batch, one item per non-blank
/// unit its language extractor finds. An item fits when the batch's tokens, plus the companions
/// already charged, plus the item, plus its file's header (if the file is new to the batch),
/// plus its companion (if its key is new to the batch) stay within `limit_tokens`; otherwise a
/// new batch starts. The invariant is `tokens + Σ distinct companions ≤ limit_tokens`.
///
/// Every item text and every file header is passed through the secret detector before it joins
/// a batch. That scan repeats the filter's on purpose: it is the egress net of the review, it
/// costs one more O(bytes) pass, and it must not be removed to save it.
///
/// O(n) in items (plus O(bytes) for the scan), with one key set per batch.
///
/// # Arguments
///
/// * `files` - The files to review, in review order.
/// * `source` - The single content source of the review (the one the filters read).
/// * `registry` - The language extractors, to split a whole file that does not fit.
/// * `limit_tokens` - The per-batch limit in tokens.
/// * `companion` - What an item drags into its batch, as `(key, tokens)`, or `None`.
///
/// # Returns
///
/// The batches, numbered from 0, never empty.
///
/// # Errors
///
/// * [`ReviewError::EgressSecret`] when an item or a header holds a secret.
/// * [`ReviewError::TooLargeForBatch`] when an item, its header and its companion do not fit
///   in an empty batch, or a whole file without an extractor (or without units) does not.
/// * [`ReviewError::Io`] when a file reviewed whole has disappeared from the source.
/// * [`ReviewError::NotReviewable`] when a file reviewed whole is no longer UTF-8.
/// * [`ReviewError::Git`] when a hunk header cannot be parsed.
/// * [`ReviewError::NothingToReview`] when no file produces an item.
/// * Any error of `source` or `companion`, unchanged.
///
/// See [`plan_batches`] for an example; it is this function with no companion.
pub fn plan_batches_with(
    files: &[ReviewFile],
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    limit_tokens: usize,
    companion: &dyn Fn(&BatchItem) -> Result<Option<Companion>, ReviewError>,
) -> Result<Vec<ReviewBatch>, ReviewError> {
    let mut packer = Packer::new(limit_tokens, companion);
    for file in files {
        let header = file_header(&file.path);
        let header_tokens = estimate_tokens(&header);
        let items = match &file.diff {
            Some(diff) => hunk_items(&file.path, diff)?,
            None => whole_file_items(&file.path, source, registry, limit_tokens, header_tokens)?,
        };
        if items.is_empty() {
            continue;
        }
        scan(&file.path, &header)?;
        for item in items {
            packer.add(item, header_tokens)?;
        }
    }
    packer.finish()
}

/// Fails with [`ReviewError::EgressSecret`] when `text`, about to leave in a batch, holds a
/// secret. O(n) in the text.
fn scan(path: &str, text: &str) -> Result<(), ReviewError> {
    match find_secret(text) {
        Some(_) => Err(ReviewError::EgressSecret {
            path: display_path(path),
        }),
        None => Ok(()),
    }
}

/// One item per hunk of `diff`; a diff without hunks gives none. The new side is not read: a
/// deletion has an empty new side and its hunks are built all the same.
fn hunk_items(path: &str, diff: &str) -> Result<Vec<BatchItem>, ReviewError> {
    let mut items = Vec::new();
    for hunk in parse_hunks(path, diff) {
        let Some(marker) = hunk.content.lines().find(|l| l.starts_with(HUNK_MARKER)) else {
            continue;
        };
        let ((old_first_line, old_last_line), (first_line, last_line)) = hunk_ranges(marker)
            .ok_or_else(|| {
                ReviewError::Git(format!("malformed hunk header in {}", display_path(path)))
            })?;
        items.push(BatchItem::Hunk {
            file: path.to_string(),
            text: hunk.content,
            first_line,
            last_line,
            old_first_line,
            old_last_line,
        });
    }
    Ok(items)
}

/// The `(first, last)` ranges of the old and new sides of a `@@ -a,b +c,d @@` line.
///
/// A side `s,n` gives `(s, s + n − 1)`; an omitted count is 1. An empty side (`n = 0`, where git
/// writes the line BEFORE the change) gives `(s + 1, s)`. `None` on a malformed line or an
/// overflow.
fn hunk_ranges(line: &str) -> Option<((u32, u32), (u32, u32))> {
    let mut parts = line.strip_prefix(HUNK_MARKER)?.split_whitespace();
    let old = side_range(parts.next()?.strip_prefix('-')?)?;
    let new = side_range(parts.next()?.strip_prefix('+')?)?;
    Some((old, new))
}

/// The `(first, last)` range of one side, `start[,count]`, of a hunk header.
fn side_range(side: &str) -> Option<(u32, u32)> {
    let (start, count) = match side.split_once(',') {
        Some((s, n)) => (s.parse::<u32>().ok()?, n.parse::<u32>().ok()?),
        None => (side.parse::<u32>().ok()?, 1),
    };
    if count == 0 {
        Some((start.checked_add(1)?, start))
    } else {
        Some((start, start.checked_add(count - 1)?))
    }
}

/// The items of a file reviewed whole: none when empty, the whole file when it fits in an empty
/// batch with its header (`header_tokens`), otherwise its non-blank units.
fn whole_file_items(
    path: &str,
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    limit_tokens: usize,
    header_tokens: usize,
) -> Result<Vec<BatchItem>, ReviewError> {
    let bytes = source.read(path)?.ok_or_else(|| {
        ReviewError::Io(format!(
            "{} disappeared during the review",
            display_path(path)
        ))
    })?;
    let text = String::from_utf8(bytes).map_err(|_| ReviewError::NotReviewable {
        path: display_path(path),
        cause: "not utf-8",
    })?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let lines = u32::try_from(text.lines().count()).unwrap_or(u32::MAX);
    let whole = BatchItem::WholeFile {
        file: path.to_string(),
        text,
        lines,
    };
    let alone = estimate_tokens(whole.text()).saturating_add(header_tokens);
    if alone <= limit_tokens {
        return Ok(vec![whole]);
    }
    let too_large = || ReviewError::TooLargeForBatch {
        what: whole.describe(),
        tokens: alone,
        limit: limit_tokens,
    };
    let extractor = registry.get(path).ok_or_else(too_large)?;
    let every_line: HashSet<u32> = (1..=lines).collect();
    let units: Vec<BatchItem> = extractor
        .extract_units(whole.text(), path, &every_line)
        .into_iter()
        // The extractor also emits an empty `TopLevel` unit for each gap between constructs.
        .filter(|u| !u.content.trim().is_empty())
        .map(|u| BatchItem::Unit {
            file: path.to_string(),
            name: u.name,
            text: u.content,
            first_line: u.start_line,
            last_line: u.end_line,
        })
        .collect();
    if units.is_empty() {
        return Err(too_large());
    }
    Ok(units)
}

/// The greedy packer: the open batch, the finished ones, and what the open batch already holds.
struct Packer<'c> {
    limit: usize,
    companion: &'c dyn Fn(&BatchItem) -> Result<Option<Companion>, ReviewError>,
    done: Vec<ReviewBatch>,
    items: Vec<BatchItem>,
    tokens: usize,
    companion_tokens: usize,
    files: HashSet<String>,
    keys: HashSet<String>,
}

impl<'c> Packer<'c> {
    /// An empty packer for batches of at most `limit` tokens.
    fn new(
        limit: usize,
        companion: &'c dyn Fn(&BatchItem) -> Result<Option<Companion>, ReviewError>,
    ) -> Self {
        Self {
            limit,
            companion,
            done: Vec::new(),
            items: Vec::new(),
            tokens: 0,
            companion_tokens: 0,
            files: HashSet::new(),
            keys: HashSet::new(),
        }
    }

    /// Scans `item` and adds it to the open batch, or to a new one when it does not fit.
    ///
    /// # Errors
    ///
    /// [`ReviewError::EgressSecret`], [`ReviewError::TooLargeForBatch`], or the companion's.
    fn add(&mut self, item: BatchItem, header_tokens: usize) -> Result<(), ReviewError> {
        scan(item.file(), item.text())?;
        let item_tokens = estimate_tokens(item.text());
        let companion = (self.companion)(&item)?;
        let companion_tokens = companion.as_ref().map_or(0, |(_, t)| *t);
        let alone = item_tokens
            .saturating_add(header_tokens)
            .saturating_add(companion_tokens);
        if alone > self.limit {
            return Err(ReviewError::TooLargeForBatch {
                what: item.describe(),
                tokens: alone,
                limit: self.limit,
            });
        }
        let new_file = !self.files.contains(item.file());
        let new_key = companion
            .as_ref()
            .is_some_and(|(k, _)| !self.keys.contains(k));
        let need = item_tokens
            .saturating_add(if new_file { header_tokens } else { 0 })
            .saturating_add(if new_key { companion_tokens } else { 0 });
        let used = self.tokens.saturating_add(self.companion_tokens);
        if used.saturating_add(need) > self.limit {
            self.close();
        }
        if self.files.insert(item.file().to_string()) {
            self.tokens = self.tokens.saturating_add(header_tokens);
        }
        if let Some((key, tokens)) = companion {
            if self.keys.insert(key) {
                self.companion_tokens = self.companion_tokens.saturating_add(tokens);
            }
        }
        self.tokens = self.tokens.saturating_add(item_tokens);
        self.items.push(item);
        Ok(())
    }

    /// Closes the open batch, if it holds anything, and opens an empty one.
    fn close(&mut self) {
        if self.items.is_empty() {
            return;
        }
        self.done.push(ReviewBatch {
            index: self.done.len(),
            items: std::mem::take(&mut self.items),
            tokens: self.tokens,
        });
        self.tokens = 0;
        self.companion_tokens = 0;
        self.files.clear();
        self.keys.clear();
    }

    /// The finished batches.
    ///
    /// # Errors
    ///
    /// [`ReviewError::NothingToReview`] when no item was added.
    fn finish(mut self) -> Result<Vec<ReviewBatch>, ReviewError> {
        self.close();
        if self.done.is_empty() {
            return Err(ReviewError::NothingToReview);
        }
        Ok(self.done)
    }
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
    fn a_realistically_formatted_file_that_does_not_fit_is_split_by_units() {
        // Imports, comments and blank lines between functions: every gap is its own unit, so
        // no unit holds the functions around it and each fits.
        let body = |v: &str| format!("    let _{v} = 1;\n").repeat(14);
        let text = format!(
            "use a::b;\n\n// first\nfn f() {{\n{}}}\n\n// second\nfn g() {{\n{}}}\n\nfn h() {{\n{}}}\n",
            body("a"),
            body("b"),
            body("c")
        );
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        let src = MapSource::new(&[("a.rs", &text)], &[]);
        let b = plan_batches(&files, &src, &LanguageRegistry::new(), 100).expect("plans");
        let names: Vec<&str> = b
            .iter()
            .flat_map(|x| x.items.iter())
            .filter_map(|i| match i {
                BatchItem::Unit { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        for f in ["f", "g", "h"] {
            assert!(names.contains(&f), "{names:?}");
        }
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
