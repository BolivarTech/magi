// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-15

//! Project documentation as chunks for the `code-review` context.
//!
//! Three kinds of source feed the documentation index: Markdown (one chunk per ATX heading; a
//! setext heading is text, `setext_headings_stay_one_chunk`), Rust and Python doc comments (one
//! chunk per documented item), nothing else. Every chunk carries a SHA-256 over its path, section
//! and text, so it is stored once however many revisions hold it. A [`DocSource`] (path plus the
//! key of its content) lets the index serve each revision exactly its own documentation.
//!
//! # Limits of line-based extraction
//!
//! Doc comments are found line by line, without a lexer: a `///` line inside a `/* */` comment or
//! a string is taken as documentation, a `def` line inside a Python triple-quoted string can yield
//! a chunk, and a Python default holding an odd number of quotes inside one hides the docstrings
//! after it. Each changes only what is retrieved: every chunk is still masked and netted, and
//! malformed input never panics.
//!
//! # No panics
//!
//! The module inherits the lints of [`crate::review`]: every read goes through `.get()` and
//! `split_at_checked`.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::encoding::lower_hex;
use crate::review::error::ReviewError;
use crate::review::languages::LanguageRegistry;
use crate::review::secrets::{display_path, holds_key_block, leaks_secret, mask_secrets};
use crate::review::source::GitCommand;

/// One documentation chunk: a Markdown section or the doc comment of one item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocChunk {
    /// Path of the source file, as the revision lists it.
    pub path: String,
    /// Heading path (`A > B`), item name, or `""` for a preamble or a module-level comment.
    pub section: String,
    /// The text embedded and shown to the reviewers.
    pub text: String,
    /// SHA-256 hex of `path + "\n" + section + "\n" + text`; independent of any revision.
    pub hash: String,
}

/// One documentation source of the reviewed revision.
///
/// `key` is the git blob id in a revision, or [`worktree_key`] of the bytes in the working tree;
/// the two forms never coincide, but their chunks (same hash) are shared.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocSource {
    /// Path of the file, as the revision lists it.
    pub path: String,
    /// Key of the file's content.
    pub key: String,
}

/// Documentation sources: `README.md` and `CLAUDE.md` only at the workspace root (exact path),
/// `docs/` and `adr/` as prefixes from the root. A nested `sub/CLAUDE.md` is not a source.
pub const DOC_SOURCES: &[&str] = &["README.md", "CLAUDE.md", "docs/", "adr/"];

/// Bound on one chunk's text: 8 KiB, about 2 048 tokens at four bytes per token, the `num_ctx`
/// Ollama gives an embedding model by default. A margin, not a guarantee: four bytes per token
/// is the unmeasured heuristic of the batch estimator (PM-S-13, A6), and at three bytes per
/// token a maximal chunk is about 2 700 tokens, past that window, where the backend truncates
/// the tail of the text without a signal. The cost is retrieval quality on the tail of a very
/// long section, never a leak or a wrong chunk served.
pub const MAX_DOC_CHUNK_BYTES: usize = 8_192;

/// Bound on one chunk's section, its part number included. Every part of a split section carries
/// the section again, so an unbounded one made the chunks (and their stored, encrypted rows)
/// grow with the square of a long heading; bounded, they grow linearly with the text.
pub const MAX_DOC_SECTION_BYTES: usize = 512;

/// Room kept under [`MAX_DOC_SECTION_BYTES`] for a part number (` (n)`, up to 13 digits).
const SECTION_PART_ROOM: usize = 16;

// `bounded_section` cuts at `MAX_DOC_SECTION_BYTES - SECTION_PART_ROOM - len(SECTION_CUT_MARK)`:
// the bound must leave a positive cut.
const _: () = assert!(MAX_DOC_SECTION_BYTES > SECTION_PART_ROOM + SECTION_CUT_MARK.len());

/// What ends a section cut to its bound.
const SECTION_CUT_MARK: &str = "…";

/// Bound on the chunks one review indexes (its missing sources): indexing runs before the clock,
/// so it must be bounded itself.
///
/// About four times this repository (5 179 chunks measured on 2026-10-17: 172 Markdown sections
/// and 5 007 doc-comment blocks), so a project of this size is indexed whole and only a much
/// larger one is cut, with the cut reported.
pub const MAX_DOC_CHUNKS: usize = 20_000;

/// Bound on the bytes (text and section) of the chunks one review indexes, next to
/// [`MAX_DOC_CHUNKS`]: 32 MiB, so the plaintext a review holds stays bounded however big its
/// chunks are; past it a source is cut and declared like a chunk-count cut.
pub const MAX_DOC_INDEX_BYTES: usize = 32 * 1024 * 1024;

/// Version of the chunking rules: bump it whenever [`netted_chunks`] would cut a source
/// differently, so every stored chunk is indexed again instead of mixing two chunkers.
///
/// A change to the secret patterns (`SECRET_PATTERNS` in [`crate::review::secrets`]) counts:
/// chunks are masked before they are hashed and embedded, and only this bump makes every source
/// be chunked and masked again with the new table; egress masking covers what the trio is sent,
/// never the embedder path. Pinned by
/// `a_change_to_the_secret_patterns_comes_with_a_doc_index_version_bump`.
///
/// A change to the masking *implementation* counts too, not only to the pattern table: a chunk
/// stored under the old algorithm would otherwise be served again to every later review.
///
/// 2 since sections are cut to [`MAX_DOC_SECTION_BYTES`] (MS3 §6 loop 5); 3 since
/// [`chunk_hash`] length-prefixes its input (MS3 §6 loop 7); 4 since a chunk whose secret hides
/// behind a control or format character is withheld with its source (MS3 §6 loop 16); 5 since the
/// lookup columns hold HMAC tags and the path is sealed (S6 loop 17: v0.25.0 databases have this
/// schema from `magi init`, no migration).
pub const DOC_INDEX_VERSION: u32 = 5;

/// Bound on the local references (branches and tags) [`live_doc_sources`] reads; with more, the
/// clean-up is skipped for that review (RV19: "more than `MAX_GC_REFS` references").
///
/// It counts references, as specified, so 65 refs at one commit still skip
/// (`past_the_ref_cap_the_clean_up_is_skipped_even_when_every_ref_shares_one_commit`); under it
/// the tips are deduplicated into commits before listing, a cost optimization only (at most
/// `MAX_GC_REFS + 1` `ls-tree` runs).
pub const MAX_GC_REFS: usize = 64;

/// How far below a doc block the item line is looked for (attributes and plain comments between).
const MAX_ITEM_LOOKAHEAD_LINES: usize = 16;

/// Bytes of a block doc comment opener (`/**` or `/*!`).
const BLOCK_DOC_OPENER_LEN: usize = 3;

/// Opener of a block comment.
const BLOCK_COMMENT_OPENER: &str = "/*";

/// Closer of a block comment.
const BLOCK_COMMENT_CLOSER: &str = "*/";

/// Keywords whose following identifier names the documented item.
const NAMED_ITEM_KEYWORDS: &[&str] = &[
    "fn", "struct", "enum", "trait", "mod", "const", "static", "type",
];

/// Modifiers that may precede the item keyword and are skipped.
const ITEM_MODIFIERS: &[&str] = &["pub", "async", "unsafe", "default", "extern"];

/// Tokens after `const` that make it a modifier (`const fn`, `const unsafe fn`...).
const CONST_MODIFIER_FOLLOWERS: &[&str] = &["fn", "unsafe", "async", "extern"];

/// Separator of the nested heading names in a section path.
const SECTION_SEPARATOR: &str = " > ";

/// Prefix of a working-tree content key; git blob ids are plain hex, so the two never collide.
pub const WORKTREE_KEY_PREFIX: &str = "wt:";

/// Deepest ATX heading level (`######`).
const MAX_HEADING_LEVEL: usize = 6;

/// Leading spaces an ATX heading or a fence may carry.
const MAX_BLOCK_INDENT: usize = 3;

/// Shortest fence run (`` ``` `` or `~~~`).
const MIN_FENCE_LEN: usize = 3;

/// The `git ls-tree` mode of a symbolic link: its blob holds the target path, not a file.
pub const SYMLINK_MODE: &str = "120000";

/// The three quote characters of a Python docstring opener.
const PY_TRIPLE_QUOTES: [&str; 2] = ["\"\"\"", "'''"];

/// The Markdown file extension, compared in any ASCII case.
const MARKDOWN_EXTENSION: &str = ".md";

/// Whether `path` names a Markdown file: the `.md` extension in any ASCII case (`README.MD`).
/// The one extension test shared by [`is_doc_source`], [`yields_doc_chunks`] and [`chunk_source`].
// MAGI FIX (S6 loop 18, R1): the chunk predicates compared `.md` exactly.
fn is_markdown_path(path: &str) -> bool {
    path.len() >= MARKDOWN_EXTENSION.len()
        && path
            .get(path.len() - MARKDOWN_EXTENSION.len()..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(MARKDOWN_EXTENSION))
}

/// Whether `path` is a Markdown documentation source by path: `README.md` or `CLAUDE.md` at the
/// root, or a `.md` under `docs/` or `adr/`. Other files there (`.rst`, `.txt`) are not indexed,
/// a declared limit (`docs_txt_is_not_a_doc_source`).
///
/// `path` is as git prints it (`/` separators on every platform); a backslash is part of a file
/// name, never a separator (`a_backslash_path_is_not_a_doc_source_on_any_platform`).
///
/// O(`DOC_SOURCES`) per call.
///
/// # Examples
///
/// ```
/// use magi_rs::review::docs::is_doc_source;
///
/// assert!(is_doc_source("docs/guide.md"));
/// assert!(!is_doc_source("sub/CLAUDE.md"));
/// ```
// MAGI FIX (MS3 §6 loop10): `docs/api.txt` was a source that never yields chunks.
pub fn is_doc_source(path: &str) -> bool {
    // MAGI FIX (S6 loop 17, R1): compared in ASCII lower case, as Windows and macOS file systems
    // are (`Readme.md`); the stored path keeps its own case.
    let lower = path.to_ascii_lowercase();
    is_markdown_path(&lower)
        && DOC_SOURCES.iter().any(|s| {
            let s = s.to_ascii_lowercase();
            if s.ends_with('/') {
                lower.starts_with(&s)
            } else {
                lower == s
            }
        })
}

/// Builds the chunk (or the bounded parts of it) for one section.
///
/// A text longer than [`MAX_DOC_CHUNK_BYTES`] is split at line boundaries (a single longer line
/// at a character boundary) into parts named `<section> (1)`, `<section> (2)`... Blank text gives
/// no chunk.
///
/// Every one-line secret of the section and the text is masked ([`mask_secrets`]) before the
/// text is split, so no part holds a byte of one and the hash is over what is sent (RV22
/// amended). A section or text holding a private-key block is neither masked nor split: it
/// stays one chunk for [`without_secrets`] to drop (RV22's exception), by its header, which
/// `find_secret` matches; the store refuses such a chunk too. O(n) in the text.
// MAGI FIX (MS3 §6 loop1): a chunk with a one-line secret was kept whole so that
// `without_secrets` could drop it; RV22 amended masks it instead.
fn make_chunks(path: &str, section: &str, text: &str) -> Vec<DocChunk> {
    let text = text.trim_end();
    if text.trim().is_empty() {
        return Vec::new();
    }
    if holds_key_block(section) || holds_key_block(text) {
        return vec![chunk_of(path, section.to_string(), text.to_string())];
    }
    // MAGI FIX (MS3 §6 loop5): a section is cut to its bound after masking (a cut before it could
    // leave half a secret no pattern matches); it was copied whole into every part.
    let section = bounded_section(&mask_secrets(section));
    let parts = split_bounded(&mask_secrets(text));
    let numbered = parts.len() > 1;
    parts
        .into_iter()
        .enumerate()
        .map(|(i, part)| {
            let section = if numbered {
                format!("{section} ({})", i + 1).trim_start().to_string()
            } else {
                section.clone()
            };
            chunk_of(path, section, part)
        })
        .collect()
}

/// `section` cut at a character boundary so that, with a part number ([`SECTION_PART_ROOM`]),
/// it fits [`MAX_DOC_SECTION_BYTES`]; a cut one ends with [`SECTION_CUT_MARK`]. The heading
/// line itself is part of the chunk's text, so nothing of it is lost to the index. O(bound).
fn bounded_section(section: &str) -> String {
    let limit = MAX_DOC_SECTION_BYTES - SECTION_PART_ROOM;
    if section.len() <= limit {
        return section.to_string();
    }
    let mut cut = limit - SECTION_CUT_MARK.len();
    while !section.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}{SECTION_CUT_MARK}",
        section.get(..cut).unwrap_or_default()
    )
}

/// The hash a [`DocChunk`] carries: SHA-256 hex of `len(path):path len(section):section text`,
/// length-prefixed so no two chunks share one (a git path may hold a newline;
/// `chunks_that_join_to_the_same_text_never_share_a_hash`).
///
/// The one definition of the chunk hash, so the chunkers and the store that checks a chunk
/// before reusing a stored vector by it cannot drift apart. O(n) in the three.
///
/// # Examples
///
/// ```
/// use magi_rs::review::docs::{chunk_hash, netted_chunks};
/// use magi_rs::review::languages::LanguageRegistry;
///
/// let c = &netted_chunks("README.md", "# A\nx\n", &LanguageRegistry::new()).0[0];
/// assert_eq!(c.hash, chunk_hash(&c.path, &c.section, &c.text));
/// ```
// MAGI FIX (MS3 §6 loop7): the input was `path\nsection\ntext`, ambiguous for a path with a newline.
pub fn chunk_hash(path: &str, section: &str, text: &str) -> String {
    lower_hex(&Sha256::digest(
        format!("{}:{path}{}:{section}{text}", path.len(), section.len()).as_bytes(),
    ))
}

/// The chunk of `path` with `section` and `text`, hashed over the three. O(n) in the text.
fn chunk_of(path: &str, section: String, text: String) -> DocChunk {
    let hash = chunk_hash(path, &section, &text);
    DocChunk {
        path: path.to_string(),
        section,
        text,
        hash,
    }
}

/// Splits `text` into parts of at most [`MAX_DOC_CHUNK_BYTES`], at line boundaries.
fn split_bounded(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        for piece in split_long_line(line) {
            if !current.is_empty() && current.len() + piece.len() > MAX_DOC_CHUNK_BYTES {
                parts.push(std::mem::take(&mut current));
            }
            current.push_str(piece);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// The longest UTF-8 encoding of one character, in bytes.
const MAX_UTF8_CHAR_BYTES: usize = 4;

// MAGI FIX (MS3 §6 loop6): `split_long_line` steps back to a character boundary from the bound;
// a bound under one character's length would cut at 0 and never advance. Checked at build time.
const _: () = assert!(MAX_DOC_CHUNK_BYTES >= MAX_UTF8_CHAR_BYTES);

/// Cuts a line longer than the bound at character boundaries; shorter lines come back whole.
///
/// Every cut advances by at least one character because [`MAX_DOC_CHUNK_BYTES`] is at least
/// [`MAX_UTF8_CHAR_BYTES`] (a build-time assertion), so the loop always ends.
fn split_long_line(line: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = line;
    while rest.len() > MAX_DOC_CHUNK_BYTES {
        let mut cut = MAX_DOC_CHUNK_BYTES;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        match rest.split_at_checked(cut) {
            Some((head, tail)) => {
                pieces.push(head);
                rest = tail;
            }
            None => break,
        }
    }
    pieces.push(rest);
    pieces
}

/// The ATX heading level and title of `line`, if it is one (`#`..`######`, up to three leading
/// spaces, a space or the end after the hashes).
fn atx_heading(line: &str) -> Option<(usize, &str)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > MAX_BLOCK_INDENT {
        return None;
    }
    let body = line.get(indent..)?;
    let level = body.len() - body.trim_start_matches('#').len();
    if level == 0 || level > MAX_HEADING_LEVEL {
        return None;
    }
    let rest = body.get(level..)?;
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None;
    }
    let title = rest.trim();
    // CommonMark: a closing run of `#` counts only when a space precedes it (`# C#` keeps its `#`).
    let unclosed = title.trim_end_matches('#');
    let title = if unclosed.is_empty() || unclosed.ends_with([' ', '\t']) {
        unclosed.trim_end()
    } else {
        title
    };
    Some((level, title))
}

/// The fence opened or closed by `line` as (character, run length), if it is one.
fn fence_of(line: &str) -> Option<(char, usize)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > MAX_BLOCK_INDENT {
        return None;
    }
    let body = line.get(indent..)?;
    let ch = body.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let run = body.chars().take_while(|c| *c == ch).count();
    (run >= MIN_FENCE_LEN).then_some((ch, run))
}

/// Chunks a Markdown file by ATX heading; the section is the heading path (`A > B`).
///
/// Text before the first heading is a chunk with section `""`. A `#` line inside a `` ``` `` or
/// `~~~` fence is text, not a heading; a fence left open runs to the end of the file (a malformed
/// document indexes worse, it never fails). A section longer than [`MAX_DOC_CHUNK_BYTES`] is
/// split into numbered parts. O(n) in the text.
///
/// # Arguments
///
/// * `path` - Path recorded in each chunk.
/// * `text` - The file's content.
///
/// # Returns
///
/// The chunks in document order; empty for blank text.
///
/// # Examples
///
/// Crate-private: the public entry is [`netted_chunks`] (RV22).
pub(crate) fn chunk_markdown(path: &str, text: &str) -> Vec<DocChunk> {
    let mut chunks = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut section = String::new();
    let mut body = String::new();
    let mut fence: Option<(char, usize)> = None;
    for line in text.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\n', '\r']);
        if let Some((ch, run)) = fence {
            // MAGI FIX (MS3 §6 loop2): a closing fence carries no info string (CommonMark), so
            // "```rust" inside a "```" fence is text, not its end.
            if fence_of(bare).is_some_and(|(c, r)| c == ch && r >= run)
                && bare.trim().chars().all(|c| c == ch)
            {
                fence = None;
            }
        } else if let Some(opened) = fence_of(bare) {
            fence = Some(opened);
        } else if let Some((level, title)) = atx_heading(bare) {
            chunks.extend(make_chunks(path, &section, &body));
            body.clear();
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, title.to_string()));
            section = stack
                .iter()
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join(SECTION_SEPARATOR);
        }
        body.push_str(line);
    }
    chunks.extend(make_chunks(path, &section, &body));
    chunks
}

/// Which side of an item a doc comment documents.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DocKind {
    /// `///` or `/** */`: documents the item that follows.
    Outer,
    /// `//!` or `/*! */`: documents the enclosing module.
    Inner,
}

/// The kind of a line doc comment (`///`, not `////`; `//!`).
fn line_doc_kind(trimmed: &str) -> Option<DocKind> {
    if trimmed.starts_with("////") {
        None
    } else if trimmed.starts_with("///") {
        Some(DocKind::Outer)
    } else if trimmed.starts_with("//!") {
        Some(DocKind::Inner)
    } else {
        None
    }
}

/// The kind and opener length of a block doc comment start (`/** `, `/*! `), not `/**/` or `/***`.
fn block_doc_kind(trimmed: &str) -> Option<DocKind> {
    if trimmed.starts_with("/**/") || trimmed.starts_with("/***") {
        None
    } else if trimmed.starts_with("/**") {
        Some(DocKind::Outer)
    } else if trimmed.starts_with("/*!") {
        Some(DocKind::Inner)
    } else {
        None
    }
}

/// Leading identifier characters of `s`.
fn ident_prefix(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// The self type named by the text after `impl`: the type after `for` when present, else the
/// first one (`impl<T> Display for Config<T>` gives `Config`).
fn impl_self_type(after_impl: &str) -> String {
    let mut s = after_impl.trim_start();
    if s.starts_with('<') {
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in s.char_indices() {
            match c {
                '<' => depth += 1,
                '>' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = Some(i + c.len_utf8());
                        break;
                    }
                }
                _ => {}
            }
        }
        s = end.and_then(|e| s.get(e..)).unwrap_or("");
    }
    let head = s.split('{').next().unwrap_or("");
    let head = head.split(" where ").next().unwrap_or("");
    let ty = head.split_once(" for ").map_or(head, |(_, after)| after);
    let ty = ty
        .trim()
        .trim_start_matches('&')
        .split_whitespace()
        .find(|t| !t.starts_with('\'') && *t != "dyn" && *t != "mut")
        .unwrap_or("");
    let ty: String = ty
        .chars()
        .take_while(|c| *c != '<' && *c != '(' && !c.is_whitespace())
        .collect();
    ident_prefix(ty.rsplit("::").next().unwrap_or(""))
}

/// The name of the item declared on `line`, or `""` when it has no named keyword.
///
/// Skips `pub`, `pub(..)`, `async`, `unsafe`, `extern "..."` and a `const` that modifies `fn`.
fn rust_item_name(line: &str) -> String {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while let Some(&tok) = tokens.get(i) {
        // MAGI FIX (MS3 §6 loop2): `pub (crate)`, with a space, is a visibility too.
        let spaced = tok == "pub" && tokens.get(i + 1).is_some_and(|n| n.starts_with('('));
        if tok.starts_with("pub(") || spaced {
            if spaced {
                i += 1;
            }
            while tokens.get(i).is_some_and(|t| !t.contains(')')) {
                i += 1;
            }
        } else if ITEM_MODIFIERS.contains(&tok) || tok.starts_with('"') {
            // modifier or the ABI string after `extern`
        } else if tok == "const"
            && tokens
                .get(i + 1)
                .is_some_and(|n| CONST_MODIFIER_FOLLOWERS.contains(n))
        {
            // `const fn`: the keyword after `const` names the item
        } else {
            break;
        }
        i += 1;
    }
    let Some(&keyword) = tokens.get(i) else {
        return String::new();
    };
    if keyword == "impl" || keyword.starts_with("impl<") {
        return line
            .split_once("impl")
            .map_or_else(String::new, |(_, rest)| impl_self_type(rest));
    }
    let mut name_at = i + 1;
    if keyword == "static" && tokens.get(name_at) == Some(&"mut") {
        name_at += 1;
    }
    if NAMED_ITEM_KEYWORDS.contains(&keyword) || keyword == "macro_rules!" {
        return tokens
            .get(name_at)
            .map_or_else(String::new, |t| ident_prefix(t));
    }
    String::new()
}

/// The name of the item whose doc block ended just before line `from`: attributes, blank lines
/// and plain comments (`//` and `/* */`, on one line or several) are skipped; `""` when there is
/// no named item.
// MAGI FIX (MS3 §6 loop1): a plain `/* */` comment before the item was read as the item line.
// MAGI FIX (MS3 §6 loop3): an attribute written over several lines is skipped whole, its
// brackets counted, so its continuation lines are never read as the item line.
fn item_name_after(lines: &[&str], from: usize) -> String {
    let mut in_comment = false;
    let mut attr_depth = 0usize;
    for k in from..from.saturating_add(MAX_ITEM_LOOKAHEAD_LINES) {
        let Some(line) = lines.get(k) else {
            break;
        };
        let t = line.trim();
        if in_comment {
            in_comment = !t.contains(BLOCK_COMMENT_CLOSER);
            continue;
        }
        if attr_depth > 0 || t.starts_with("#[") || t.starts_with("#![") {
            // MAGI FIX (MS3 §6 loop12): an item written after its attribute on one line is named.
            if attr_depth == 0 {
                // MAGI FIX (MS3 §6 loop20, R1): stacked attributes `#[a] #[b] fn x()` are all skipped.
                let mut rest = after_attribute(t).filter(|r| !r.is_empty());
                while let Some(r) = rest.filter(|r| r.starts_with("#[")) {
                    match after_attribute(r) {
                        Some(next) => rest = Some(next).filter(|n| !n.is_empty()),
                        None => {
                            rest = None;
                            break;
                        }
                    }
                }
                if let Some(rest) = rest {
                    return rust_item_name(rest);
                }
            }
            attr_depth = bracket_depth(attr_depth, t);
            continue;
        }
        if let Some(rest) = t.strip_prefix(BLOCK_COMMENT_OPENER) {
            match rest.split_once(BLOCK_COMMENT_CLOSER) {
                Some((_, after)) if !after.trim().is_empty() => {
                    return rust_item_name(after.trim())
                }
                Some(_) => {}
                None => in_comment = true,
            }
            continue;
        }
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        return rust_item_name(t);
    }
    String::new()
}

/// What follows the closing `]` of the attribute that opens `line` (`#[..]` or `#![..]`), when
/// it closes on this line: the brackets are counted as [`bracket_depth`] counts them (a bracket
/// inside a `"…"` string is text), the remainder trimmed. `None` when the attribute stays open
/// past the line. O(len of `line`).
fn after_attribute(line: &str) -> Option<&str> {
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for (i, c) in line.char_indices() {
        if quoted {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => quoted = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            '[' => depth = depth.saturating_add(1),
            ']' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    return line.get(i + 1..).map(str::trim);
                }
            }
            _ => {}
        }
    }
    None
}

/// `depth` after the square brackets of `line`: each `[` opens, each `]` closes, saturating at
/// zero; a bracket inside a `"…"` string (with `\` escapes, on this line) is text. O(len of
/// `line`).
// MAGI FIX (MS3 §6 loop10): `#[cfg(x = "[")]` left the depth open.
fn bracket_depth(depth: usize, line: &str) -> usize {
    let (mut d, mut quoted, mut escaped) = (depth, false, false);
    for c in line.chars() {
        if quoted {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => quoted = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            '[' => d = d.saturating_add(1),
            ']' => d = d.saturating_sub(1),
            _ => {}
        }
    }
    d
}

/// Strips the `///` or `//!` opener and one following space.
fn strip_line_doc(trimmed: &str) -> &str {
    let rest = trimmed.get(3..).unwrap_or("");
    rest.strip_prefix(' ').unwrap_or(rest)
}

/// Collects the block doc comment starting at line `start` (which holds the opener): the text,
/// the index of the line after the comment, and what follows the closer on its own line (`""`
/// when nothing does).
///
/// The closer `*/` is looked for before a continuation line loses its leading `*`, so the usual
/// ` */` line closes the comment.
// MAGI FIX (MS3 §6 loop1): the leading `*` was stripped first, so a ` */` line closed nothing and
// the code below was swallowed into the chunk.
// MAGI FIX (MS3 §6 loop3): the text after the closer is returned, so an item declared on the
// closing line names its chunk instead of the next item.
fn collect_block_doc<'a>(lines: &[&'a str], start: usize) -> (String, usize, &'a str) {
    let mut body = String::new();
    let mut j = start;
    let mut after = "";
    while let Some(&line) = lines.get(j) {
        let mut t = line.trim_start();
        if j == start {
            t = t.get(BLOCK_DOC_OPENER_LEN..).unwrap_or("");
        }
        let (content, closed) = match t.find(BLOCK_COMMENT_CLOSER) {
            Some(e) => {
                after = t.get(e + BLOCK_COMMENT_CLOSER.len()..).unwrap_or("");
                (t.get(..e).unwrap_or(t), true)
            }
            None => (t, false),
        };
        let content = if j == start {
            content
        } else {
            let c = content.strip_prefix('*').unwrap_or(content);
            c.strip_prefix(' ').unwrap_or(c)
        };
        body.push_str(content.trim_end());
        body.push('\n');
        j += 1;
        if closed {
            break;
        }
    }
    (body, j, after)
}

/// Rust doc comments of `lines`, in file order.
fn rust_doc_chunks(path: &str, lines: &[&str]) -> Vec<DocChunk> {
    let mut chunks = Vec::new();
    let mut i = 0;
    while let Some(line) = lines.get(i) {
        let t = line.trim_start();
        if let Some(kind) = line_doc_kind(t) {
            let mut body = String::new();
            let mut j = i;
            while let Some(l) = lines.get(j) {
                let lt = l.trim_start();
                if line_doc_kind(lt) != Some(kind) {
                    break;
                }
                body.push_str(strip_line_doc(lt));
                body.push('\n');
                j += 1;
            }
            let name = if kind == DocKind::Outer {
                item_name_after(lines, j)
            } else {
                String::new()
            };
            chunks.extend(make_chunks(path, &name, &body));
            i = j;
        } else if let Some(kind) = block_doc_kind(t) {
            let (body, j, after) = collect_block_doc(lines, i);
            let name = if kind == DocKind::Outer {
                // MAGI FIX (MS3 §6 loop20, R1): an empty `after` is no line, so the block form
                // looks as far as the line form does.
                let tail: Vec<&str> = Some(after)
                    .filter(|a| !a.is_empty())
                    .into_iter()
                    .chain(lines.iter().skip(j).copied())
                    .take(MAX_ITEM_LOOKAHEAD_LINES)
                    .collect();
                item_name_after(&tail, 0)
            } else {
                String::new()
            };
            chunks.extend(make_chunks(path, &name, &body));
            i = j;
        } else {
            i += 1;
        }
    }
    chunks
}

/// Index of the first line at or after `from` that is neither blank nor a `#` comment.
fn py_next_statement(lines: &[&str], from: usize) -> Option<usize> {
    (from..lines.len()).find(|&k| {
        lines
            .get(k)
            .is_some_and(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
    })
}

/// Whether what follows a closing triple quote ends the statement: nothing, or a `#` comment.
fn py_statement_ends(after: &str) -> bool {
    let after = after.trim_start();
    after.is_empty() || after.starts_with('#')
}

/// The docstring starting on line `at` (`"""` or `'''`, with an optional `r`/`u` prefix): its text
/// and the index of the line after it.
///
/// One `r`/`R`/`u`/`U` prefix is accepted. A bytes prefix (`b`, `rb`, `br`) or an f-string
/// (`f`, `fr`, `rf`), in any case, is no docstring: Python keeps only a plain `str` literal as
/// `__doc__`. The string must be a statement of its own (`"""a""" + x` is data), and one that
/// never closes is no docstring (it never swallows the rest of the file).
// MAGI FIX (MS3 §6 loop2): an expression that opened with a string, or an unterminated string,
// was chunked as a docstring.
fn py_docstring_at(lines: &[&str], at: usize) -> Option<(String, usize)> {
    let t = lines.get(at)?.trim_start();
    let t = t.strip_prefix(['r', 'R', 'u', 'U']).unwrap_or(t);
    let quote = PY_TRIPLE_QUOTES.iter().find(|q| t.starts_with(**q))?;
    let rest = t.get(quote.len()..)?;
    if let Some(end) = rest.find(quote) {
        let after = rest.get(end + quote.len()..)?;
        return py_statement_ends(after)
            .then(|| (rest.get(..end).unwrap_or("").trim().to_string(), at + 1));
    }
    let mut body = vec![rest.trim().to_string()];
    let mut j = at + 1;
    while let Some(line) = lines.get(j) {
        j += 1;
        if let Some(end) = line.find(quote) {
            if !py_statement_ends(line.get(end + quote.len()..).unwrap_or("")) {
                return None;
            }
            body.push(line.get(..end).unwrap_or("").trim().to_string());
            return Some((body.join("\n").trim().to_string(), j));
        }
        body.push(line.trim().to_string());
    }
    None
}

/// The name of the `def`/`class` on `line`, if it is one.
fn py_def_name(line: &str) -> Option<String> {
    let t = line.trim_start();
    let t = t.strip_prefix("async ").unwrap_or(t).trim_start();
    let rest = t
        .strip_prefix("def ")
        .or_else(|| t.strip_prefix("class "))?;
    Some(ident_prefix(rest.trim_start()))
}

/// The code of a Python line, without its `#` comment, and the change in bracket depth it makes.
/// A `#` or a bracket inside a one-line string literal (`'…'`, `"…"`, with `\` escapes) is text.
/// O(line).
///
/// Declared (MS3 §6 loop12): the quote state resets at each line, so a string that spans lines
/// inside a `def` header (a triple-quoted default) can leave the depth unbalanced; the header then
/// ends at the last line the lookahead reads and the docstrings after it are not indexed. It is
/// this module's line-based extraction limit (no parse of the source): the effect is a missing
/// chunk, never a wrong one, and what is indexed is masked and secret-checked like every chunk.
fn py_code(line: &str) -> (&str, i64) {
    let mut depth = 0i64;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => quote = Some(c),
            '#' => return (line.get(..i).unwrap_or(line), depth),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    (line, depth)
}

/// Index of the last line of the header that starts at `at` (the first line where the
/// parentheses balance).
///
/// Brackets inside a one-line string literal (`'…'`, `"…"`, with `\` escapes) and after a `#`
/// comment do not count.
// MAGI FIX (MS3 §6 loop2): `def f(p="("):` or a `# (todo` comment left the depth open, the
// header ran to the end of the file and every later docstring was lost.
fn py_header_end(lines: &[&str], at: usize) -> usize {
    let mut depth = 0i64;
    for k in at..lines.len() {
        let Some(line) = lines.get(k) else { break };
        depth += py_code(line).1;
        if depth <= 0 {
            return k;
        }
    }
    lines.len().saturating_sub(1)
}

/// Python docstrings of `lines`: the module's, and the first statement of each `def`/`class`.
fn py_doc_chunks(path: &str, lines: &[&str]) -> Vec<DocChunk> {
    let mut chunks = Vec::new();
    let mut i = 0;
    if let Some(first) = py_next_statement(lines, 0) {
        if let Some((text, next)) = py_docstring_at(lines, first) {
            chunks.extend(make_chunks(path, "", &text));
            i = next;
        }
    }
    while let Some(line) = lines.get(i) {
        let Some(name) = py_def_name(line) else {
            i += 1;
            continue;
        };
        let end = py_header_end(lines, i);
        i = end + 1;
        if !lines
            .get(end)
            .is_some_and(|l| py_code(l).0.trim_end().ends_with(':'))
        {
            continue;
        }
        if let Some(body) = py_next_statement(lines, end + 1) {
            if let Some((text, next)) = py_docstring_at(lines, body) {
                chunks.extend(make_chunks(path, &name, &text));
                i = next;
            }
        }
    }
    chunks
}

/// Chunks the doc comments of a Rust or Python file; code is never a chunk.
///
/// Rust: `///`, `//!`, `/** */` and `/*! */`; `//` comments are never docs. The section is the
/// item's name: attribute lines are skipped, then `pub`, `pub(..)`, `async`, `unsafe`, `extern
/// ".."`; `const fn x` names `x`; `impl` names its self type (the one after `for` when present);
/// `macro_rules! m` names `m`; no named keyword, and every inner comment, gives `""`. Python: a
/// string is a docstring only as the first statement of a module, `def` or `class` body;
/// `x = """..."""` is data. Another language with an extractor gives no chunk. O(n) in the text.
///
/// # Arguments
/// * `path` - Path recorded in each chunk; its extension picks the language.
/// * `text` - The file's content.
/// * `registry` - Languages the review has extractors for; a file outside it gives nothing.
///
/// # Returns
/// The chunks in file order.
pub(crate) fn chunk_doc_comments(
    path: &str,
    text: &str,
    registry: &LanguageRegistry,
) -> Vec<DocChunk> {
    let Some(language) = doc_comment_language(path, registry) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    match language {
        DocCommentLanguage::Rust => rust_doc_chunks(path, &lines),
        DocCommentLanguage::Python => py_doc_chunks(path, &lines),
    }
}

/// A language whose doc comments [`chunk_doc_comments`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocCommentLanguage {
    /// `///`, `//!` and `/** */` comments of a `.rs` file.
    Rust,
    /// Docstrings of a `.py` file.
    Python,
}

/// The doc-comment language of `path`: the one place that names the languages whose doc
/// comments are chunked, so the chunker and [`yields_doc_chunks`] cannot drift. `None` for a
/// file without an extractor in `registry` or of another language. O(len of the path).
fn doc_comment_language(path: &str, registry: &LanguageRegistry) -> Option<DocCommentLanguage> {
    registry.get(path)?;
    // MAGI FIX (MS3 §6 loop20, R1): the registry matches the extension in any ASCII case.
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)?;
    if ext.eq_ignore_ascii_case("rs") {
        Some(DocCommentLanguage::Rust)
    } else if ext.eq_ignore_ascii_case("py") {
        Some(DocCommentLanguage::Python)
    } else {
        None
    }
}

/// Whether [`netted_chunks`] can give `path` any chunk: a Markdown file that [`is_doc_source`]
/// accepts, or a file whose doc comments are chunked. A file for which this is
/// `false` never yields documentation, so a review neither reads nor keys it for the index.
///
/// # Arguments
///
/// * `path` - Path of the file.
/// * `registry` - Languages the review has extractors for.
///
/// # Returns
///
/// `true` when the file may yield chunks. O(len of the path).
///
/// # Examples
///
/// ```
/// use magi_rs::review::docs::yields_doc_chunks;
/// use magi_rs::review::languages::LanguageRegistry;
///
/// let reg = LanguageRegistry::new();
/// assert!(yields_doc_chunks("README.md", &reg));
/// assert!(yields_doc_chunks("src/a.rs", &reg));
/// assert!(!yields_doc_chunks("src/B.java", &reg));
/// ```
// MAGI FIX (Loop1 r2, Minor 1): every file with an extractor was read to key the index,
// though only Rust and Python give doc-comment chunks.
pub fn yields_doc_chunks(path: &str, registry: &LanguageRegistry) -> bool {
    if is_markdown_path(path) {
        is_doc_source(path)
    } else {
        doc_comment_language(path, registry).is_some()
    }
}

/// The chunks of one file, by kind: the extension decides before the path.
///
/// A `.md` that [`is_doc_source`] accepts goes to [`chunk_markdown`]; a file with a language
/// extractor goes to [`chunk_doc_comments`] (its doc comments, never its code, even under
/// `docs/`); anything else gives nothing.
///
/// # Arguments
///
/// * `path` - Path of the file in the revision.
/// * `text` - Its content.
/// * `registry` - Languages the review has extractors for.
///
/// # Returns
///
/// The chunks, possibly none.
pub(crate) fn chunk_source(path: &str, text: &str, registry: &LanguageRegistry) -> Vec<DocChunk> {
    if is_markdown_path(path) {
        if is_doc_source(path) {
            chunk_markdown(path, text)
        } else {
            Vec::new()
        }
    } else {
        chunk_doc_comments(path, text, registry)
    }
}

/// The chunks of one documentation source, netted (RV22): `chunk_source`'s, through
/// [`without_secrets`], and none when the file holds a private-key block anywhere. The one
/// public way to chunk a source, so no caller sends key-block bytes by forgetting the net
/// (`chunking_is_netted_by_default`). O(n) in the text.
///
/// # Arguments
/// * `path` - Path of the file in the revision.
/// * `text` - Its content.
/// * `registry` - Languages the review has extractors for.
///
/// # Returns
/// The chunks safe to send, and the source's [`display_path`] once per dropped chunk; a
/// non-empty second part means: withhold the source.
///
/// # Examples
///
/// ```
/// use magi_rs::review::docs::netted_chunks;
/// use magi_rs::review::languages::LanguageRegistry;
///
/// let (c, dropped) = netted_chunks("README.md", "# A\nx\n## B\ny\n", &LanguageRegistry::new());
/// assert_eq!((c[1].section.as_str(), dropped.len()), ("A > B", 0));
/// ```
// MAGI FIX (MS3 §6 loop8): the public chunkers returned key-block chunks unnetted.
pub fn netted_chunks(
    path: &str,
    text: &str,
    registry: &LanguageRegistry,
) -> (Vec<DocChunk>, Vec<String>) {
    if holds_key_block(text) {
        return (Vec::new(), vec![display_path(path)]);
    }
    without_secrets(chunk_source(path, text, registry))
}

/// Splits `chunks` into those safe to send and the sources of those holding a secret.
///
/// The chunkers mask every one-line secret, so a chunk they built is dropped only for a
/// private-key block (RV22's exception); a chunk whose path, section or text still matches
/// [`leaks_secret`] (built by hand, or cut into a new match) is dropped too, before any
/// embedding. O(total text) plus O(chunks log sources).
///
/// **Source-scoped** (RV22 omits the *file*): once any chunk of a path is dropped, every chunk
/// of that path is, since a key body line copied elsewhere matches no pattern. A block in the
/// file's code is in no chunk, which is why [`netted_chunks`] also checks the whole file.
///
/// # Returns
///
/// The kept chunks, and the [`display_path`] of each dropped chunk's source, one per chunk in
/// order (its length is the count).
// MAGI FIX (MS3 §6 loop6): the drop was per chunk, so a sibling holding a copied key line was
// kept; it is now per source.
pub fn without_secrets(chunks: Vec<DocChunk>) -> (Vec<DocChunk>, Vec<String>) {
    // MAGI FIX (MS3 §6 loop3): the path travels with the chunk; RV22's tie-break omits it.
    let holds =
        |c: &DocChunk| leaks_secret(&c.path) || leaks_secret(&c.section) || leaks_secret(&c.text);
    let tainted: BTreeSet<String> = chunks
        .iter()
        .filter(|c| holds(c))
        .map(|c| c.path.clone())
        .collect();
    let mut kept = Vec::with_capacity(chunks.len());
    let mut dropped = Vec::new();
    for chunk in chunks {
        if tainted.contains(&chunk.path) {
            dropped.push(display_path(&chunk.path));
        } else {
            kept.push(chunk);
        }
    }
    (kept, dropped)
}

/// Whole sources in path order while their chunks fit in [`MAX_DOC_CHUNKS`] and their bytes
/// (text and section) in [`MAX_DOC_INDEX_BYTES`] (`bytes_bound_the_index_too`).
///
/// A source that does not fit is skipped and the next ones still enter (one big source never
/// starves the rest); a source with more chunks than the bound is never indexed. A source left
/// out is not indexed, so a later review indexes it. O(n log n) in the sources (sorted by path).
///
/// # Returns
///
/// The kept sources, and whether any was left out (to be reported).
pub fn bounded(
    sources: Vec<(DocSource, Vec<DocChunk>)>,
) -> (Vec<(DocSource, Vec<DocChunk>)>, bool) {
    let (kept, caps) = bounded_capped(sources);
    (kept, caps.chunks || caps.bytes)
}

/// Which of the index's two bounds left a source out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapsHit {
    /// A source was left out for [`MAX_DOC_CHUNKS`].
    pub chunks: bool,
    /// A source was left out for [`MAX_DOC_INDEX_BYTES`].
    pub bytes: bool,
}

/// [`bounded`], naming the bound that cut: the report says which cap a review hit.
///
/// # Returns
///
/// The kept sources, and which bounds left a source out.
pub fn bounded_capped(
    mut sources: Vec<(DocSource, Vec<DocChunk>)>,
) -> (Vec<(DocSource, Vec<DocChunk>)>, CapsHit) {
    sources.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut total, mut bytes) = (0usize, 0usize);
    let mut truncated = CapsHit::default();
    let mut kept = Vec::with_capacity(sources.len());
    for (source, chunks) in sources {
        // MAGI FIX (MS3 §6 loop8): only the chunk count was bounded, not the bytes held.
        let size = chunks.iter().fold(0usize, |n, c| {
            n.saturating_add(c.text.len())
                .saturating_add(c.section.len())
        });
        // MAGI FIX (MS3 §6 loop15): a source with no chunk still costs a registry row, so it
        // counts as one against the bound (R1 Caspar).
        let cost = chunks.len().max(1);
        let over_chunks = total.saturating_add(cost) > MAX_DOC_CHUNKS;
        let over_bytes = bytes.saturating_add(size) > MAX_DOC_INDEX_BYTES;
        if over_chunks || over_bytes {
            truncated.chunks |= over_chunks;
            truncated.bytes |= over_bytes;
        } else {
            total += cost;
            bytes += size;
            kept.push((source, chunks));
        }
    }
    (kept, truncated)
}

/// The content key of a working-tree file: `wt:` plus the SHA-256 hex of its bytes.
///
/// Content-addressed, and never equal to a git blob id (hex without a prefix). O(n).
pub fn worktree_key(bytes: &[u8]) -> String {
    format!("{WORKTREE_KEY_PREFIX}{}", lower_hex(&Sha256::digest(bytes)))
}

/// `<model_id>#<dim>#<DOC_INDEX_VERSION>`: the identity of the vectors an index holds.
///
/// The dimension is part of it, so a dimension change reads exactly like a model change.
pub fn doc_index_id(model_id: &str, dim: usize) -> String {
    format!("{model_id}#{dim}#{DOC_INDEX_VERSION}")
}

/// The blobs of `git ls-tree -r -z` output (`<mode> <type> <id>\t<path>\0`) whose path `keep`
/// accepts, in listing order.
///
/// A submodule (`commit`) is skipped, so is a symlink (a blob of mode [`SYMLINK_MODE`], whose
/// content is its target path), and so is a path that is not UTF-8 (it cannot be a
/// documentation source). Any other entry that does not parse is [`ReviewError::Git`], never
/// skipped in silence. O(bytes).
///
/// # Errors
///
/// [`ReviewError::Git`] for a malformed entry or an object type other than `blob`/`commit`.
pub fn parse_ls_tree(
    out: &[u8],
    keep: &dyn Fn(&str) -> bool,
) -> Result<Vec<DocSource>, ReviewError> {
    let malformed = || ReviewError::Git("malformed `git ls-tree` entry".to_string());
    let mut sources = Vec::new();
    for entry in out.split(|b| *b == 0).filter(|e| !e.is_empty()) {
        let tab = entry
            .iter()
            .position(|b| *b == b'\t')
            .ok_or_else(malformed)?;
        let (meta, path) = entry.split_at_checked(tab).ok_or_else(malformed)?;
        let meta = std::str::from_utf8(meta).map_err(|_| malformed())?;
        let mut fields = meta.split_whitespace();
        let (Some(mode), Some(kind), Some(id), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(malformed());
        };
        match kind {
            // MAGI FIX (MS3 §6 loop6): a symlink's blob is its target path, no documentation.
            "blob" if mode == SYMLINK_MODE => continue,
            "blob" => {}
            "commit" => continue,
            _ => return Err(malformed()),
        }
        let Ok(path) = std::str::from_utf8(path.get(1..).unwrap_or_default()) else {
            continue;
        };
        if keep(path) {
            sources.push(DocSource {
                path: path.to_string(),
                key: id.to_string(),
            });
        }
    }
    Ok(sources)
}

/// `git ls-tree -r -z <rev>` through `git`, parsed by [`parse_ls_tree`].
///
/// # Errors
///
/// [`ReviewError::Git`] when `rev` starts with `-` (it would be read as an option), when git
/// fails, or when an entry does not parse.
pub fn revision_doc_sources(
    git: &dyn GitCommand,
    rev: &str,
    keep: &dyn Fn(&str) -> bool,
) -> Result<Vec<DocSource>, ReviewError> {
    if rev.starts_with('-') {
        return Err(ReviewError::Git(
            "revision looks like an option".to_string(),
        ));
    }
    parse_ls_tree(&git.run_bytes(&["ls-tree", "-r", "-z", rev])?, keep)
}

/// The sources every local ref still reaches: `HEAD` and each branch and tag tip.
///
/// "Reaches" is the tree at each ref's tip, not its history: a version only in a branch's
/// history is deleted and indexed again if needed (a cost, never a loss). An annotated tag
/// peels to its tree. `rev-parse` failing (an unborn HEAD included: it reads as a broken git),
/// or `for-each-ref` failing, gives `Ok(None)`: a git that does not answer deletes nothing.
/// O(refs x tree entries) local git calls.
///
/// # Returns
///
/// `Ok(None)` when the clean-up must be skipped (more than [`MAX_GC_REFS`] references, or
/// a git that does not answer); otherwise the union of every ref's sources.
///
/// # Errors
///
/// [`ReviewError::Git`] when a listed ref cannot be listed or parsed (a tag naming a blob, say).
/// It deletes nothing either: the caller (`collect_garbage`) returns it, and the runner skips
/// the clean-up and logs it, never failing the review.
pub fn live_doc_sources(
    git: &dyn GitCommand,
    keep: &dyn Fn(&str) -> bool,
) -> Result<Option<BTreeSet<DocSource>>, ReviewError> {
    let head = git.run_bytes(&["rev-parse", "--verify", "HEAD^{commit}"]);
    // MAGI FIX (MS3 §6 loop9): an annotated tag's own id listed its commit's tree twice; each
    // line is `<id> <peeled id>`, the peeled one empty unless the ref is an annotated tag.
    let Ok(refs) = git.run_bytes(&[
        "for-each-ref",
        "--format=%(objectname) %(*objectname)",
        "refs/heads",
        "refs/tags",
    ]) else {
        return Ok(None);
    };
    let to_ids = |bytes: &[u8]| -> Result<Vec<String>, ReviewError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| ReviewError::Git("git printed a non-UTF-8 object id".to_string()))?;
        Ok(text
            .lines()
            .filter_map(|l| l.split_whitespace().last())
            .map(str::to_string)
            .collect())
    };
    let ref_ids = to_ids(&refs)?;
    // MAGI FIX (MS3 §6 loop6): RV19 bounds the references; the bound counted distinct commits.
    if ref_ids.len() > MAX_GC_REFS {
        return Ok(None);
    }
    let mut commits: BTreeSet<String> = ref_ids.into_iter().collect();
    // MAGI FIX (MS3 §6 loop13): a HEAD that does not resolve skips the clean-up whether or not
    // refs exist: an unborn HEAD and a broken git fail alike, and an empty live set deletes all.
    match head {
        Ok(bytes) => commits.extend(to_ids(&bytes)?),
        Err(_) => return Ok(None),
    }
    let mut live = BTreeSet::new();
    for commit in &commits {
        live.extend(revision_doc_sources(git, commit, keep)?);
    }
    Ok(Some(live))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::secrets::find_secret;

    #[test]
    fn markdown_is_chunked_by_heading_with_its_section_path() {
        let c = chunk_markdown("README.md", "# A\nintro\n## B\nbody b\n## C\nbody c\n");
        assert_eq!(
            c.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["A", "A > B", "A > C"]
        );
        assert!(c[1].text.contains("body b"));
    }

    #[test]
    fn setext_headings_stay_one_chunk() {
        // [MAGI S6 MS3 loop9, R1 Melchior] Declared limitation: only ATX headings split, so an
        // underlined (setext) heading is text of one unnamed section, never a lost line.
        let c = chunk_markdown("README.md", "Title\n=====\nintro\nSub\n---\nbody\n");
        assert_eq!(c.len(), 1);
        assert!(c[0].section.is_empty() && c[0].text.contains("Sub\n---\nbody"));
    }

    #[test]
    fn a_heading_inside_a_code_fence_does_not_start_a_section() {
        // [CP2 MS3 loop 2] A shell comment in a fenced block is not an ATX heading.
        let c = chunk_markdown("README.md", "# Usage\n```sh\n# install\nmake\n```\nmore\n");
        assert_eq!(c.len(), 1);
        assert!(c[0].text.contains("# install"));
        // [CP2 MS3 loop 5] Tilde fences too.
        let t = chunk_markdown("README.md", "# Usage\n~~~\n# install\n~~~\nmore\n");
        assert_eq!(t.len(), 1, "{t:?}");
        // [CP2 MS3 loop 7] An unclosed fence runs to the end of the file (declared).
        let u = chunk_markdown("README.md", "# A\n```\n# B\n");
        assert_eq!(u.len(), 1, "{u:?}");
    }

    #[test]
    fn a_fence_line_with_an_info_string_never_closes_a_fence() {
        // [MAGI S6 MS3 loop2, R1 Caspar] CommonMark: a closing fence carries no info string, so
        // "```rust" inside a "```" fence is text; it closed the fence and "# x" became a heading.
        let c = chunk_markdown("README.md", "# A\n```\n```rust\n# x\n```\n# B\nb\n");
        assert_eq!(
            c.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["A", "B"],
            "{c:?}"
        );
    }

    #[test]
    fn a_markdown_file_without_headings_is_one_chunk() {
        let c = chunk_markdown("README.md", "just text\nmore\n");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].section, "");
        assert!(
            chunk_markdown("README.md", "").is_empty(),
            "[CP2 MS3 loop 13] an empty file has nothing to index"
        );
    }

    #[test]
    fn the_hash_changes_with_the_text_and_not_otherwise() {
        let a = chunk_markdown("d.md", "# A\nx\n");
        let b = chunk_markdown("d.md", "# A\nx\n");
        let c = chunk_markdown("d.md", "# A\ny\n");
        assert_eq!(a[0].hash, b[0].hash);
        assert_ne!(a[0].hash, c[0].hash);
        assert_eq!(a[0].hash.len(), 64);
        // [CP2 MS3 loop 14 (pasada final)] Each hash input on its own: the path and the section count too. The hash has no
        // revision input, so the same section in two revisions (a and b above, read twice) keeps its hash: RV-S37's reuse.
        assert_ne!(
            a[0].hash,
            chunk_markdown("e.md", "# A\nx\n")[0].hash,
            "path"
        );
        assert_ne!(
            a[0].hash,
            chunk_markdown("d.md", "# B\nx\n")[0].hash,
            "section"
        );
    }

    #[test]
    fn a_long_section_is_split_into_bounded_parts() {
        // [CP2 MS3 loop 14 (pasada final)] One request never exceeds an embedder's input: parts at line boundaries.
        let body = "x\n".repeat(MAX_DOC_CHUNK_BYTES); // twice the bound
        let c = chunk_markdown("README.md", &format!("# A\n{body}"));
        assert!(c.len() >= 2, "{}", c.len());
        assert!(
            c.iter().all(|x| x.text.len() <= MAX_DOC_CHUNK_BYTES),
            "every part within the bound"
        );
        assert_eq!(c[0].section, "A (1)");
        assert_eq!(c[1].section, "A (2)");
    }

    #[test]
    fn a_huge_heading_gives_every_part_a_bounded_section() {
        // [MAGI S6 MS3 loop5, R1 Caspar] The section was copied whole into every part of its
        // split text: a 1 MiB heading made ~128 parts carrying ~128 MiB of section (an 8 MiB
        // file, gigabytes), each encrypted and stored again. The heading line stays whole in
        // the body; the section that names the part is cut to a bound.
        let heading = "h".repeat(1 << 20);
        let c = chunk_markdown("README.md", &format!("# {heading}\nbody\n"));
        assert!(c.len() > 100, "{}", c.len());
        let section_bytes: usize = c.iter().map(|x| x.section.len()).sum();
        assert!(
            c.iter().all(|x| x.section.len() <= MAX_DOC_SECTION_BYTES),
            "a section past the bound"
        );
        assert!(section_bytes <= c.len() * MAX_DOC_SECTION_BYTES);
        assert!(c[0].section.ends_with(" (1)"), "parts stay numbered");
        let text: String = c.iter().map(|x| x.text.as_str()).collect();
        assert!(
            text.contains(&heading),
            "the heading itself is still indexed"
        );
    }

    #[test]
    fn rust_doc_comments_become_chunks_named_after_their_item() {
        let src =
            "/// Adds two numbers.\n/// Panics never.\nfn add(a: u32, b: u32) -> u32 { a + b }\n";
        let c = chunk_doc_comments("src/m.rs", src, &LanguageRegistry::new());
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].section, "add");
        assert!(c[0].text.contains("Adds two numbers."));
        // [CP2 MS3 loop 13] Every item keyword names its section, not only `fn`.
        let s = chunk_doc_comments(
            "src/m.rs",
            "/// A config.\npub struct Config { x: u8 }\n",
            &LanguageRegistry::new(),
        );
        assert_eq!(
            s.iter().map(|c| c.section.as_str()).collect::<Vec<_>>(),
            vec!["Config"]
        );
    }

    #[test]
    fn every_item_shape_names_its_section() {
        // [CP2 MS3 loop 14 (pasada final)] The precise naming rule, Rust and Python.
        let reg = LanguageRegistry::new();
        let name = |p: &str, src: &str| {
            chunk_doc_comments(p, src, &reg)
                .iter()
                .map(|c| c.section.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            name("src/m.rs", "/// D.\n#[derive(Debug)]\npub struct Config;\n"),
            vec!["Config"],
            "attribute skipped"
        );
        assert_eq!(
            name("src/m.rs", "/// S.\npub const fn size() -> usize { 1 }\n"),
            vec!["size"],
            "const fn"
        );
        assert_eq!(
            name("src/m.rs", "/// I.\nimpl Display for Config {}\n"),
            vec!["Config"],
            "impl"
        );
        assert_eq!(
            name("src/m.rs", "/// M.\nmacro_rules! twice { () => {} }\n"),
            vec!["twice"],
            "macro_rules!"
        );
        assert_eq!(
            name(
                "m.py",
                "def f():\n    \"\"\"Explains f.\"\"\"\n    return 1\n"
            ),
            vec!["f"],
            "def"
        );
        assert_eq!(
            name("m.py", "@dataclass\nclass C:\n    \"\"\"A C.\"\"\"\n"),
            vec!["C"],
            "decorated class"
        );
        assert_eq!(
            name("m.py", "\"\"\"Module docs.\"\"\"\nx = 1\n"),
            vec![""],
            "module docstring"
        );
        assert!(
            name("m.py", "def f():\n    x = 1\n    \"\"\"not docs\"\"\"\n").is_empty(),
            "a later string is data"
        );
    }

    #[test]
    fn a_visibility_written_with_a_space_still_names_the_item() {
        // [MAGI S6 MS3 loop2, R1 Caspar] `pub (crate)` is valid Rust; the `(crate)` token was
        // read as the item keyword and the section came out empty.
        let reg = LanguageRegistry::new();
        let c = chunk_doc_comments("src/m.rs", "/// S.\npub (crate) fn size() {}\n", &reg);
        assert_eq!(c.first().map(|x| x.section.as_str()), Some("size"), "{c:?}");
        let d = chunk_doc_comments("src/m.rs", "/// T.\npub ( in crate::a ) struct T;\n", &reg);
        assert_eq!(d.first().map(|x| x.section.as_str()), Some("T"), "{d:?}");
    }

    #[test]
    fn a_python_string_is_a_docstring_only_as_a_whole_statement() {
        // [MAGI S6 MS3 loop2, R1 Caspar] A string that opens an expression is data, and a
        // string that never closes is no docstring: it swallowed the rest of the file.
        let reg = LanguageRegistry::new();
        let expr = chunk_doc_comments(
            "m.py",
            "def f():\n    \"\"\"part\"\"\" + x\n    return 1\n",
            &reg,
        );
        assert!(expr.is_empty(), "{expr:?}");
        let open = chunk_doc_comments(
            "m.py",
            "def f():\n    \"\"\"never closed\n    x = 1\n",
            &reg,
        );
        assert!(open.is_empty(), "{open:?}");
        let commented =
            chunk_doc_comments("m.py", "def f():\n    \"\"\"Doc f.\"\"\"  # ok\n", &reg);
        assert!(
            commented.iter().any(|c| c.text.contains("Doc f.")),
            "{commented:?}"
        );
    }

    #[test]
    fn brackets_in_a_header_string_or_comment_never_hide_the_docstrings_below() {
        // [MAGI S6 MS3 loop2, R1 Melchior] The header's brackets were counted in strings and
        // comments, so the header never ended and every later docstring of the file was lost.
        let reg = LanguageRegistry::new();
        let src = "def f(p=\"(\"):\n    \"\"\"Doc f.\"\"\"\n\ndef g():  # (todo\n    \"\"\"Doc g.\"\"\"\n\ndef h():\n    \"\"\"Doc h.\"\"\"\n";
        let c = chunk_doc_comments("m.py", src, &reg);
        assert_eq!(
            c.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["f", "g", "h"],
            "{c:?}"
        );
    }

    #[test]
    fn plain_comments_and_string_data_are_not_documentation() {
        // [CP2 MS3 loop 13] The false-positive side of the doc-comment rules.
        let reg = LanguageRegistry::new();
        assert!(chunk_doc_comments("src/m.rs", "// just a note\nfn a() {}\n", &reg).is_empty());
        assert!(chunk_doc_comments("m.py", "PATTERN = \"\"\"not docs\"\"\"\n", &reg).is_empty());
        assert!(
            chunk_doc_comments("m.py", "# a comment\ndef f():\n    return 1\n", &reg).is_empty()
        );
    }

    #[test]
    fn inner_rust_and_python_doc_comments_become_chunks_too() {
        // [CP2 MS3 loop 4] The three documented forms, not only `///`.
        let r = chunk_doc_comments(
            "src/lib.rs",
            "//! Crate overview.\nfn a() {}\n",
            &LanguageRegistry::new(),
        );
        assert!(
            r.iter().any(|c| c.text.contains("Crate overview.")),
            "{r:?}"
        );
        let p = chunk_doc_comments(
            "m.py",
            "def f():\n    \"\"\"Explains f.\"\"\"\n    return 1\n",
            &LanguageRegistry::new(),
        );
        assert!(p.iter().any(|c| c.text.contains("Explains f.")), "{p:?}");
        // [CP2 MS3 loop 5] Block doc comments are rustdoc too.
        let b = chunk_doc_comments(
            "src/lib.rs",
            "/** Block docs. */\nfn b() {}\n",
            &LanguageRegistry::new(),
        );
        assert!(b.iter().any(|c| c.text.contains("Block docs.")), "{b:?}");
        // [CP2 MS3 loop 7] Python's other docstring quote.
        let q = chunk_doc_comments(
            "m.py",
            "def g():\n    '''Explains g.'''\n    return 1\n",
            &LanguageRegistry::new(),
        );
        assert!(q.iter().any(|c| c.text.contains("Explains g.")), "{q:?}");
    }

    #[test]
    fn a_block_doc_closed_on_its_own_line_stops_before_the_code() {
        // [MAGI S6 MS3 loop1, R1 Melchior] The usual ` */` closer line closed nothing: the
        // continuation `*` was stripped first, so the code below was swallowed into the chunk.
        let reg = LanguageRegistry::new();
        let c = chunk_doc_comments("src/m.rs", "/**\n * x\n */\nfn f() {}\n", &reg);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].section, "f", "{c:?}");
        assert!(!c[0].text.contains("fn f"), "{c:?}");
        let flush = chunk_doc_comments("src/m.rs", "/*!\n * Crate.\n*/\nfn g() {}\n", &reg);
        assert!(
            flush.len() == 1 && !flush[0].text.contains("fn g"),
            "{flush:?}"
        );
        // A plain block comment between the doc block and its item is skipped too.
        let skipped = chunk_doc_comments(
            "src/m.rs",
            "/// Doc.\n/* plain\n   comment */\nfn h() {}\n",
            &reg,
        );
        assert_eq!(skipped[0].section, "h", "{skipped:?}");
    }

    #[test]
    fn each_source_file_is_chunked_by_its_kind() {
        // [CP2 MS3 loop 7] RV07(3): a source file gives its doc comments, never its code.
        let reg = LanguageRegistry::new();
        let rs = chunk_source(
            "src/m.rs",
            "/// Adds.\nfn add() { let body_marker = 1; }\n",
            &reg,
        );
        assert_eq!(rs.len(), 1, "{rs:?}");
        assert!(
            rs[0].text.contains("Adds.") && !rs[0].text.contains("body_marker"),
            "{rs:?}"
        );
        assert_eq!(
            chunk_source("README.md", "# A\nx\n## B\ny\n", &reg).len(),
            2
        );
        assert!(chunk_source("notes.txt", "# A\nx\n", &reg).is_empty());
        // [CP2 MS3 loop 14 (pasada final)] Under docs/ the extension still decides: code is never sent as documentation.
        let ex = chunk_source(
            "docs/example.rs",
            "/// Shows use.\nfn demo() { let body_marker = 1; }\n",
            &reg,
        );
        assert!(
            ex.iter().all(|c| !c.text.contains("body_marker"))
                && ex.iter().any(|c| c.text.contains("Shows use.")),
            "{ex:?}"
        );
    }

    #[test]
    fn a_chunk_with_a_one_line_secret_is_kept_with_the_secret_masked() {
        // [MAGI S6 MS3 loop1, R1 all three] RV22 amended: mask, do not omit, the documentation
        // chunks included. The hash is over what is sent, never over the secret.
        // [RED-DRYRUN 2026-10-17, E3] One-line let: rustfmt splits the call and moves the allow marker off the
        // literal's line, and tests/no_hardcoded_secrets.rs goes red (measured).
        let text = "# Setup\nexport KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\n# Other\nfine\n"; // allow-secret-scan
        let (kept, dropped) = without_secrets(chunk_markdown("README.md", text));
        assert!(dropped.is_empty(), "{dropped:?}");
        assert_eq!(
            kept.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["Setup", "Other"]
        );
        assert!(
            kept[0].text.contains("<redacted: anthropic-key>") && !kept[0].text.contains("sk-ant-"),
            "{kept:?}"
        );
        // [MAGI S6 MS3 loop7] Through `chunk_hash`, whose input is now length-prefixed.
        let expected = chunk_hash("README.md", "Setup", &kept[0].text);
        assert_eq!(kept[0].hash, expected, "hashed over the masked text");
        // [CP2 MS3 loop 13] A secret in a heading is a secret too (the section is sent with the chunk).
        let heading = "# sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\nfine\n"; // allow-secret-scan
        let (kept, dropped) = without_secrets(chunk_markdown("README.md", heading));
        assert!(dropped.is_empty(), "{dropped:?}");
        assert_eq!(kept[0].section, "<redacted: anthropic-key>", "{kept:?}");
        assert!(!kept[0].text.contains("sk-ant-"), "{kept:?}");
    }

    #[test]
    fn a_chunk_holding_a_private_key_block_is_left_out() {
        // [MAGI S6 MS3 loop1] RV22's exception: a key block is never masked, its chunk is dropped.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let text =
            format!("# Keys\n{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n# Other\nfine\n");
        let (kept, dropped) = without_secrets(chunk_markdown("README.md", &text));
        // [MAGI S6 MS3 loop6, R1 Caspar] The exception omits the file: the clean sibling goes too.
        assert_eq!(dropped, vec!["README.md".to_string(); 2]);
        assert!(kept.is_empty(), "{kept:?}");
    }

    #[test]
    fn a_private_key_block_in_one_chunk_withholds_every_chunk_of_its_source() {
        // [MAGI S6 MS3 loop6, R1 Caspar] RV22's exception omits the whole file: a body line
        // copied into another section has no pattern of its own, so only omitting every chunk of
        // the source keeps it from the embedder. Another source's chunks are unaffected.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let body = "MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w";
        let text = format!("# Keys\n{begin}\n{body}\n# Copied\nsee {body}\n# Other\nfine\n");
        let mut chunks = chunk_markdown("README.md", &text);
        assert_eq!(chunks.len(), 3, "precondition: three sections");
        assert!(
            find_secret(&chunks[1].text).is_none(),
            "precondition: the copied line matches no pattern"
        );
        chunks.extend(chunk_markdown("docs/b.md", "# B\nfine\n"));
        let (kept, dropped) = without_secrets(chunks);
        assert_eq!(dropped, vec!["README.md".to_string(); 3]);
        assert_eq!(
            kept.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(),
            vec!["docs/b.md"]
        );
        assert!(kept.iter().all(|c| !c.text.contains(body)), "{kept:?}");
    }

    #[test]
    fn a_secret_only_in_the_section_still_drops_the_chunk() {
        // The section travels with the chunk: a clean text does not save a secret name. A
        // doc-comment chunk is the case where the section is not also part of the text.
        let section = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA".to_string(); // allow-secret-scan
        let chunk = DocChunk {
            path: "src/m.rs".into(),
            section,
            text: "fine".into(),
            hash: String::new(),
        };
        let (kept, dropped) = without_secrets(vec![chunk]);
        assert!(kept.is_empty());
        assert_eq!(dropped, vec!["src/m.rs".to_string()]);
    }

    #[test]
    fn an_item_on_the_closing_line_of_its_block_doc_names_the_chunk() {
        // [MAGI S6 MS3 loop3, R1 Melchior] The line after the closer was read, so a one-line
        // `/** */` followed on the same line by its item took the next item's name.
        let c = rust_doc_chunks("a.rs", &["/** Loads it. */ fn load() {}", "fn other() {}"]);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].section, "load");
        let attr = rust_doc_chunks("a.rs", &["/** Gated. */ #[inline]", "fn gated() {}"]);
        assert_eq!(attr[0].section, "gated");
    }

    #[test]
    fn an_item_on_its_attribute_line_is_named() {
        // MAGI FIX (MS3 §6 loop12): `#[derive(Debug)] pub fn foo()` was consumed whole as an
        // attribute, so its chunk had no section name.
        for item in [
            "#[inline] pub fn foo() {}",
            "#[cfg(any(a, b))] pub fn foo() {}",
            "#[doc = \"]\"] pub fn foo() {}",
        ] {
            let lines = ["/// Docs.", item];
            let c = rust_doc_chunks("a.rs", &lines);
            assert_eq!(c.len(), 1, "{item}: {c:?}");
            assert!(c[0].section.contains("foo"), "{item}: {c:?}");
        }
    }

    #[test]
    fn an_item_on_the_closing_line_of_a_multi_line_attribute_is_named() {
        // MAGI FIX (MS3 §6 loop21, R1 Caspar): the rest of a line was read only when it opened
        // the attribute, so `)] fn foo() {}` closing a multi-line one was skipped whole and the
        // chunk took the next item's name (or none).
        let lines = [
            "/// Doc.",
            "#[cfg(",
            "    feature = \"x\"",
            ")] fn foo() {}",
            "fn bar() {}",
        ];
        let c = rust_doc_chunks("a.rs", &lines);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].section, "foo");
    }

    #[test]
    fn a_multi_line_attribute_before_the_item_is_skipped_whole() {
        // [MAGI S6 MS3 loop3, R1 Melchior] The attribute's continuation lines were read as the
        // item line, leaving the chunk without a section.
        let lines = [
            "/// Gated.",
            "#[cfg(any(",
            "    feature = \"a\",",
            "    feature = \"b\"",
            "))]",
            "fn gated() {}",
        ];
        let c = rust_doc_chunks("a.rs", &lines);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].section, "gated");
    }

    #[test]
    fn quoted_bracket_in_attr() {
        // [MAGI S6 MS3 loop10, R1 Melchior] A `[` inside an attribute's string left the depth
        // open and swallowed the item line.
        let lines = ["/// Gated.", "#[cfg(x = \"[\")]", "fn gated() {}"];
        let c = rust_doc_chunks("a.rs", &lines);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].section, "gated");
    }

    #[test]
    fn a_secret_only_in_the_path_drops_the_chunk() {
        // [MAGI S6 MS3 loop3, R1 Caspar] The path travels with the chunk (context frame, clear
        // column) and a path cannot be masked and still name its file: RV22's tie-break omits it.
        let path = "docs/sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.md"; // allow-secret-scan
        let chunks = chunk_markdown(path, "# Usage\nfine\n");
        assert_eq!(chunks.len(), 1, "precondition: the text alone is clean");
        let (kept, dropped) = without_secrets(chunks);
        assert!(kept.is_empty(), "{kept:?}");
        assert_eq!(dropped, vec![display_path(path)]);
        assert!(
            dropped.iter().all(|p| !p.contains("sk-ant-")),
            "{dropped:?}"
        );
    }

    #[test]
    fn a_secret_straddling_the_part_cut_never_reaches_the_embedder() {
        // One line longer than the bound with a secret across the cut: the text is masked before
        // it is split, so no part holds a byte of the secret, and every part stays bounded.
        // [MAGI S6 MS3 loop1] Masked, not dropped (RV22 amended).
        let secret = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let line = format!(
            "{}{secret}{}",
            "x".repeat(MAX_DOC_CHUNK_BYTES - 10),
            "y".repeat(100)
        );
        let chunks = chunk_markdown("README.md", &format!("# A\n{line}\n# B\nfine\n"));
        let (kept, dropped) = without_secrets(chunks);
        assert!(dropped.is_empty(), "{dropped:?}");
        assert!(kept.iter().all(|c| !c.text.contains("sk-ant-")
            && !c.text.contains("AAAAAAAAAAAAAAAAAAAA")
            && c.text.len() <= MAX_DOC_CHUNK_BYTES));
        let joined: String = kept.iter().map(|c| c.text.as_str()).collect();
        assert!(
            joined.contains("<redacted: anthropic-key>"),
            "the mask may span the cut"
        );
        assert_eq!(kept.last().map(|c| c.section.as_str()), Some("B"));
    }

    #[test]
    fn a_closing_hash_run_is_stripped_only_after_a_space() {
        let section = |t: &str| chunk_markdown("d.md", t).first().map(|c| c.section.clone());
        assert_eq!(section("# C#\nx\n"), Some("C#".to_string()));
        assert_eq!(section("# A ##\nx\n"), Some("A".to_string()));
    }

    #[test]
    fn a_revision_that_looks_like_an_option_is_refused() {
        let git = TableGit(vec![("ls-tree -r -z --output=x", Vec::new())]);
        assert!(matches!(
            revision_doc_sources(&git, "--output=x", &|_: &str| true),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn ls_tree_entries_with_the_wrong_shape_are_errors() {
        let keep = |_: &str| true;
        // Too few fields, too many fields, and an object type that is neither blob nor commit.
        for bad in [
            &b"100644 blob\tREADME.md\0"[..],
            &b"100644 blob aaaa extra\tREADME.md\0"[..],
            &b"040000 tree aaaa\tdocs\0"[..],
        ] {
            assert!(
                matches!(parse_ls_tree(bad, &keep), Err(ReviewError::Git(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_item_line_is_looked_for_within_a_bounded_lookahead() {
        let reg = LanguageRegistry::new();
        let name = |blanks: usize| {
            let src = format!("/// D.\n{}fn f() {{}}\n", "\n".repeat(blanks));
            chunk_doc_comments("src/m.rs", &src, &reg)
                .first()
                .map(|c| c.section.clone())
        };
        assert_eq!(name(MAX_ITEM_LOOKAHEAD_LINES - 1), Some("f".to_string()));
        assert_eq!(name(MAX_ITEM_LOOKAHEAD_LINES), Some(String::new()));
    }

    #[test]
    fn doc_comments_of_an_upper_case_extension_are_chunked() {
        // [MAGI S6 MS3 loop20, R1 Caspar] the registry matches `.RS` in lower case, but the
        // language match was exact, so the file had an extractor and never yielded a chunk.
        let reg = LanguageRegistry::new();
        for (path, src) in [
            ("src/FOO.RS", "/// Doc of f.\nfn f() {}\n"),
            (
                "src/BAR.PY",
                "def g():\n    \"\"\"Doc of g.\"\"\"\n    pass\n",
            ),
        ] {
            assert!(yields_doc_chunks(path, &reg), "{path}");
            assert!(!chunk_source(path, src, &reg).is_empty(), "{path}");
        }
    }

    #[test]
    fn a_block_doc_names_an_item_sixteen_lines_below_like_a_line_doc() {
        // [MAGI S6 MS3 loop20, R1 Melchior] the block form chained a leading `after` and looked
        // at one line fewer than the line form.
        let reg = LanguageRegistry::new();
        let name = |blanks: usize| {
            let src = format!("/** D. */\n{}fn f() {{}}\n", "\n".repeat(blanks));
            chunk_doc_comments("src/m.rs", &src, &reg)
                .first()
                .map(|c| c.section.clone())
        };
        assert_eq!(name(MAX_ITEM_LOOKAHEAD_LINES - 1), Some("f".to_string()));
        assert_eq!(name(MAX_ITEM_LOOKAHEAD_LINES), Some(String::new()));
    }

    #[test]
    fn stacked_attributes_on_the_item_line_keep_its_name() {
        // [MAGI S6 MS3 loop20, R1 Melchior] `#[a] #[b] fn x()` read `#[b]` as the keyword.
        let reg = LanguageRegistry::new();
        let src = "/// D.\n#[a] #[b] fn x() {}\n";
        let chunks = chunk_doc_comments("src/m.rs", src, &reg);
        assert_eq!(chunks.first().map(|c| c.section.as_str()), Some("x"));
    }

    #[test]
    fn indexing_is_bounded_and_the_truncation_is_reported() {
        // [CP2 MS3 loop 6] Reverse order, so keeping the first by path needs the sort, not only the cut.
        // [RV21 D+B, 2026-10-20] One chunk per source here; whole sources are kept.
        let many: Vec<(DocSource, Vec<DocChunk>)> = (0..MAX_DOC_CHUNKS + 1)
            .rev()
            .map(|i| {
                let path = format!("docs/{i:05}.md");
                (
                    DocSource {
                        path: path.clone(),
                        key: format!("k{i}"),
                    },
                    vec![DocChunk {
                        path,
                        section: String::new(),
                        text: "x".into(),
                        hash: format!("{i}"),
                    }],
                )
            })
            .collect();
        let (kept, truncated) = bounded(many);
        assert_eq!(kept.len(), MAX_DOC_CHUNKS);
        assert!(truncated);
        assert_eq!(
            kept.first().map(|(s, _)| s.path.as_str()),
            Some("docs/00000.md")
        );
    }

    #[test]
    fn sources_without_chunks_count_against_the_bound() {
        // [MAGI S6 MS3 loop15, R1 Caspar] a source with no chunk (an empty README, a file with no
        // doc comment) cost nothing, so a tree of them was registered without limit.
        let empty: Vec<(DocSource, Vec<DocChunk>)> = (0..MAX_DOC_CHUNKS + 1)
            .map(|i| {
                (
                    DocSource {
                        path: format!("docs/{i:05}.md"),
                        key: format!("k{i}"),
                    },
                    Vec::new(),
                )
            })
            .collect();
        let (kept, truncated) = bounded(empty);
        assert_eq!(kept.len(), MAX_DOC_CHUNKS);
        assert!(truncated);
    }

    #[test]
    fn a_source_is_indexed_whole_or_left_for_a_later_review() {
        // [RV21 D+B, 2026-10-20] A half-indexed source would be remembered as indexed and never completed.
        let c = |p: &str, n: usize| -> (DocSource, Vec<DocChunk>) {
            (
                DocSource {
                    path: p.into(),
                    key: format!("k-{p}"),
                },
                (0..n)
                    .map(|i| DocChunk {
                        path: p.into(),
                        section: format!("{i}"),
                        text: "x".into(),
                        hash: format!("{p}{i}"),
                    })
                    .collect(),
            )
        };
        let (kept, truncated) = bounded(vec![
            c("a.md", MAX_DOC_CHUNKS - 1),
            c("b.md", 2),
            c("c.md", 1),
        ]);
        assert!(truncated);
        // [CP2 MS3 loop 13] b.md does not fit and is skipped; c.md still does.
        assert_eq!(
            kept.iter()
                .map(|(s, v)| (s.path.as_str(), v.len()))
                .collect::<Vec<_>>(),
            vec![("a.md", MAX_DOC_CHUNKS - 1), ("c.md", 1)]
        );
        // [CP2 MS3 loop 13] An oversized first source never starves the rest.
        let (kept, truncated) = bounded(vec![c("a.md", MAX_DOC_CHUNKS + 1), c("b.md", 1)]);
        assert!(truncated);
        assert_eq!(
            kept.iter()
                .map(|(s, _)| s.path.as_str())
                .collect::<Vec<_>>(),
            vec!["b.md"]
        );
    }

    #[test]
    fn chunking_is_netted_by_default() {
        // [MAGI S6 MS3 loop8, R1 Caspar] The public chunkers handed a private-key block to any
        // caller that forgot `without_secrets`; the public entry nets the source itself.
        let reg = LanguageRegistry::new();
        let body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC";
        let block = format!(
            "{}\n{body}\n",
            concat!("-----", "BEGIN ", "PRIVATE KEY-----")
        );
        let md = format!("# Setup\n{block}# Usage\nrun it\n");
        let rs = format!("/// Adds.\nfn add() {{}}\nconst K: &str = \"{block}\";\n");
        for (path, text) in [("README.md", md.as_str()), ("src/a.rs", rs.as_str())] {
            let (kept, dropped) = netted_chunks(path, text, &reg);
            assert!(kept.is_empty() && !dropped.is_empty(), "{path}: {kept:?}");
        }
        let (kept, dropped) = netted_chunks("README.md", "# A\nplain\n", &reg);
        assert_eq!((kept.len(), dropped.len()), (1, 0));
    }

    #[test]
    fn bytes_bound_the_index_too() {
        // [MAGI S6 MS3 loop8, R1 Caspar] A few huge sources under MAX_DOC_CHUNKS were held whole
        // in memory; the bytes are bounded too, and a source past them is cut, declared.
        let c = |p: &str, bytes: usize| -> (DocSource, Vec<DocChunk>) {
            let chunk = DocChunk {
                path: p.into(),
                section: String::new(),
                text: "x".repeat(bytes),
                hash: p.into(),
            };
            let key = format!("k-{p}");
            (
                DocSource {
                    path: p.into(),
                    key,
                },
                vec![chunk],
            )
        };
        let half = MAX_DOC_INDEX_BYTES / 2;
        let (kept, truncated) = bounded(vec![c("a.md", half), c("b.md", half + 1), c("c.md", 1)]);
        assert!(truncated);
        let names: Vec<&str> = kept.iter().map(|(s, _)| s.path.as_str()).collect();
        assert_eq!(names, vec!["a.md", "c.md"]);
    }

    #[test]
    fn worktree_keys_are_content_addressed_and_never_look_like_a_git_id() {
        // [RV21 D+B, 2026-10-20]
        assert_eq!(worktree_key(b"x"), worktree_key(b"x"));
        assert_ne!(worktree_key(b"x"), worktree_key(b"y"));
        assert!(worktree_key(b"x").starts_with("wt:"));
        assert_eq!(worktree_key(b"x").len(), 3 + 64);
    }

    #[test]
    fn ls_tree_blobs_become_the_doc_sources_of_that_revision() {
        // [RV21 D+B, 2026-10-20] -z keeps spaces in paths; a submodule is not a blob; `keep` filters.
        // [RED-DRYRUN 2026-10-20, RV21] `\x00`, not `\0`: before a digit `\0` is clippy::octal_escapes.
        let out = b"100644 blob aaaa\tREADME.md\x00100644 blob bbbb\tsrc/a b.rs\x00160000 commit cccc\tvendor/sub\x00100644 blob dddd\tnotes.txt\x00";
        let s = parse_ls_tree(out, &|p: &str| p != "notes.txt").expect("parses");
        assert_eq!(
            s,
            vec![
                DocSource {
                    path: "README.md".into(),
                    key: "aaaa".into()
                },
                DocSource {
                    path: "src/a b.rs".into(),
                    key: "bbbb".into()
                }
            ]
        );
    }

    #[test]
    fn a_symlink_blob_is_no_documentation_source() {
        // [MAGI S6 MS3 loop6, R1 Melchior] git stores a symlink (mode 120000) as a blob whose
        // content is the target path: a junk chunk, never documentation.
        let out = b"120000 blob aaaa\tdocs/link.md\x00100644 blob bbbb\tdocs/real.md\0";
        let sources = parse_ls_tree(out, &|_: &str| true).expect("parses");
        assert_eq!(
            sources.iter().map(|s| s.path.as_str()).collect::<Vec<_>>(),
            vec!["docs/real.md"]
        );
    }

    #[test]
    fn a_bytes_literal_is_no_python_docstring_whatever_its_prefix() {
        // [MAGI S6 MS3 loop6, R1 Melchior] Python keeps only a `str` literal as `__doc__`: a bytes
        // literal (`b`, `rb`, `br`, any case) in docstring position is data, so it is no chunk,
        // while the `r` and `u` string prefixes still make a docstring.
        for prefix in ["b", "B", "rb", "br", "Rb", "bR"] {
            let text = format!("{prefix}\"\"\"Bytes.\"\"\"\n");
            let c = py_doc_chunks("m.py", &text.lines().collect::<Vec<_>>());
            assert!(c.is_empty(), "{prefix}: {c:?}");
        }
        for prefix in ["", "r", "R", "u", "U"] {
            let text = format!("{prefix}\"\"\"Module doc.\"\"\"\n");
            let c = py_doc_chunks("m.py", &text.lines().collect::<Vec<_>>());
            assert_eq!(c.len(), 1, "{prefix}: {c:?}");
            assert_eq!(c[0].text, "Module doc.", "{prefix}");
        }
    }

    #[test]
    fn an_f_string_is_no_python_docstring_whatever_its_prefix() {
        // [MAGI S6 MS3 loop7, R1 Balthasar] An f-string compiles to an expression, never a
        // constant, so Python leaves `__doc__` as `None`: it is data, like a bytes literal.
        for prefix in ["f", "F", "fr", "rf", "Rf", "fR"] {
            let text = format!("{prefix}\"\"\"Not a doc.\"\"\"\n");
            let c = py_doc_chunks("m.py", &text.lines().collect::<Vec<_>>());
            assert!(c.is_empty(), "{prefix}: {c:?}");
        }
    }

    #[test]
    fn an_ls_tree_entry_that_does_not_parse_is_an_error_never_skipped() {
        // [RV21 D+B, 2026-10-20] A silent skip would leave a source out of the reviewed revision's set.
        assert!(matches!(
            parse_ls_tree(b"garbage\0", &|_: &str| true),
            Err(ReviewError::Git(_))
        ));
    }

    /// Answers each argv (joined by spaces) from a fixed table; anything else is a git error. [RV21 D+B, 2026-10-20]
    struct TableGit(Vec<(&'static str, Vec<u8>)>);
    impl GitCommand for TableGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            let k = args.join(" ");
            self.0
                .iter()
                .find(|(a, _)| *a == k)
                .map(|(_, out)| out.clone())
                .ok_or(ReviewError::Git(k))
        }
    }

    #[test]
    fn a_revision_lists_its_sources_with_ls_tree() {
        let git = TableGit(vec![(
            "ls-tree -r -z abc",
            b"100644 blob k1\tREADME.md\0".to_vec(),
        )]);
        assert_eq!(
            revision_doc_sources(&git, "abc", &|_: &str| true).expect("lists"),
            vec![DocSource {
                path: "README.md".into(),
                key: "k1".into()
            }]
        );
    }

    #[test]
    fn live_doc_sources_are_the_union_of_every_local_ref() {
        // [RV21 D+B, 2026-10-20] RV-S39: what a branch, a tag or HEAD still reaches stays indexed.
        let git = TableGit(vec![
            ("rev-parse --verify HEAD^{commit}", b"h1\n".to_vec()),
            (
                "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
                b"h1\nb2\n".to_vec(),
            ),
            ("ls-tree -r -z h1", b"100644 blob k1\tREADME.md\0".to_vec()),
            ("ls-tree -r -z b2", b"100644 blob k2\tREADME.md\0".to_vec()),
        ]);
        let live = live_doc_sources(&git, &|_: &str| true)
            .expect("ok")
            .expect("under the cap");
        assert_eq!(
            live.iter().map(|s| s.key.as_str()).collect::<Vec<_>>(),
            vec!["k1", "k2"]
        );
    }

    #[test]
    fn annotated_tag_peels_to_commit() {
        // [MAGI S6 MS3 loop9, R1 Caspar] An annotated tag's own object id listed its tree a
        // second time; it is peeled to the commit it names, so that commit is listed once.
        let git = TableGit(vec![
            ("rev-parse --verify HEAD^{commit}", b"h1\n".to_vec()),
            (
                "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
                b"h1 \nt1 h1\n".to_vec(),
            ),
            ("ls-tree -r -z h1", b"100644 blob k1\tREADME.md\0".to_vec()),
        ]);
        let live = live_doc_sources(&git, &|_: &str| true).expect("ok");
        assert_eq!(live.map(|s| s.len()), Some(1));
    }

    #[test]
    fn past_the_ref_cap_the_clean_up_is_skipped_never_partial() {
        // [RV21 D+B, 2026-10-20] A partial live set would delete what an unwalked ref still reaches.
        let refs: String = (0..=MAX_GC_REFS).map(|i| format!("r{i}\n")).collect();
        let git = TableGit(vec![
            ("rev-parse --verify HEAD^{commit}", b"r0\n".to_vec()),
            (
                "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
                refs.into_bytes(),
            ),
        ]);
        assert_eq!(live_doc_sources(&git, &|_: &str| true).expect("ok"), None);
    }

    #[test]
    fn past_the_ref_cap_the_clean_up_is_skipped_even_when_every_ref_shares_one_commit() {
        // [MAGI S6 MS3 loop6, R1 Caspar] RV19 bounds the references, not the distinct commits:
        // sixty-five refs at one commit cost one listing, and the clean-up is still skipped.
        let refs = "h1\n".repeat(MAX_GC_REFS + 1);
        let at_cap = "h1\n".repeat(MAX_GC_REFS);
        let git = |refs: &str| {
            TableGit(vec![
                ("rev-parse --verify HEAD^{commit}", b"h1\n".to_vec()),
                (
                    "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
                    refs.as_bytes().to_vec(),
                ),
                ("ls-tree -r -z h1", b"100644 blob k1\tREADME.md\0".to_vec()),
            ])
        };
        assert_eq!(
            live_doc_sources(&git(&refs), &|_: &str| true).expect("ok"),
            None
        );
        let live = live_doc_sources(&git(&at_cap), &|_: &str| true)
            .expect("ok")
            .expect("at the cap the clean-up runs");
        assert_eq!(live.len(), 1, "one listing for the shared commit");
    }

    #[test]
    fn an_unborn_head_contributes_nothing_and_is_not_an_error() {
        // [CP2 MS3 loop 13] A repository with no commit yet: rev-parse fails, no refs.
        // MAGI FIX (MS3 §6 loop13, R1 Caspar): that failure reads the same as a git that is
        // broken, so it no longer yields an empty live set (which deletes every source): the
        // clean-up is skipped, never an error. Cost: a repository with no commit never cleans.
        let git = TableGit(vec![(
            "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
            Vec::new(),
        )]);
        assert_eq!(live_doc_sources(&git, &|_: &str| true).expect("ok"), None);
    }

    #[test]
    fn a_git_that_does_not_answer_skips_the_clean_up() {
        // [CP2 MS3 loop 14 (pasada final)] HEAD unreadable while refs exist is not «unborn»: never delete on a broken git.
        let refs_but_no_head = TableGit(vec![
            (
                "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags",
                b"b2\n".to_vec(),
            ),
            ("ls-tree -r -z b2", b"100644 blob k2\tREADME.md\0".to_vec()),
        ]);
        assert_eq!(
            live_doc_sources(&refs_but_no_head, &|_: &str| true).expect("ok"),
            None
        );
        let nothing_answers = TableGit(vec![]);
        assert_eq!(
            live_doc_sources(&nothing_answers, &|_: &str| true).expect("ok"),
            None
        );
    }

    #[test]
    fn the_index_id_names_the_model_and_the_chunker_version() {
        // [CP2 MS3 loop 10] Same dimension, other model or other chunker: never the same index.
        // [CP2 MS3 loop 11] The dimension too: every change of the vectors' identity is another index.
        assert_eq!(
            doc_index_id("nomic-embed-text", 768),
            format!("nomic-embed-text#768#{DOC_INDEX_VERSION}")
        );
        assert_ne!(doc_index_id("a", 3), doc_index_id("b", 3));
        assert_ne!(doc_index_id("a", 3), doc_index_id("a", 5));
    }

    #[test]
    fn doc_sources_are_recognized_by_path() {
        for p in [
            "README.md",
            "CLAUDE.md",
            "docs/guide.md",
            "docs/adr/0001.md",
            "adr/0001.md",
        ] {
            assert!(is_doc_source(p), "{p}");
        }
        assert!(!is_doc_source("src/main.rs"));
        assert!(
            !is_doc_source("sub/CLAUDE.md"),
            "[CP2 MS3 loop 13] root files by exact path"
        );
    }
    #[test]
    fn documentation_discovery_ignores_the_case_of_the_name() {
        // [MAGI S6 MS3 loop17, R1] `Readme.md` and `Docs/Guide.MD` are documentation on the
        // case-insensitive file systems of Windows and macOS.
        for p in [
            "Readme.md",
            "readme.md",
            "CLAUDE.MD",
            "Docs/Guide.MD",
            "ADR/0001.md",
        ] {
            assert!(is_doc_source(p), "{p}");
        }
        for p in ["sub/Readme.md", "Docs/guide.txt"] {
            assert!(!is_doc_source(p), "{p}");
        }
    }

    #[test]
    fn an_upper_case_md_extension_yields_chunks() {
        // [MAGI S6 MS3 loop18, R1] `is_doc_source` accepted `README.MD`, but the chunk
        // predicates compared `.md` exactly, so the source was listed and gave nothing.
        let reg = LanguageRegistry::new();
        for p in ["README.MD", "docs/FOO.MD"] {
            assert!(is_doc_source(p), "{p}");
            assert!(yields_doc_chunks(p, &reg), "{p}");
            assert!(!chunk_source(p, "# A\nx\n", &reg).is_empty(), "{p}");
        }
    }

    #[test]
    fn docs_txt_is_not_a_doc_source() {
        // [MAGI S6 MS3 loop10, R1 Caspar] The two predicates disagreed on `docs/api.txt`.
        let reg = LanguageRegistry::new();
        for p in ["docs/api.txt", "docs/guide.rst", "adr/1.adoc"] {
            assert!(!is_doc_source(p), "{p}");
            assert!(!yields_doc_chunks(p, &reg), "{p}");
        }
    }

    #[test]
    fn chunks_that_join_to_the_same_text_never_share_a_hash() {
        // [MAGI S6 MS3 loop7, R1 Melchior] A newline in a git path once let two chunks alias.
        assert_ne!(
            chunk_hash("docs/x.md\nfoo.md", "S", "T"),
            chunk_hash("docs/x.md", "foo.md", "S\nT")
        );
        assert_ne!(chunk_hash("a", "b\nc", "d"), chunk_hash("a", "b", "c\nd"));
    }

    /// The secret patterns, by name and in table order, that `DOC_INDEX_VERSION` was last
    /// bumped for.
    const PATTERNS_OF_THIS_DOC_INDEX_VERSION: &[&str] = &[
        "anthropic-key",
        "openai-key",
        "github-token",
        "github-token",
        "aws-access-key",
        "pem-private-key",
        "pem-private-key",
        "pem-private-key",
        "pgp-private-key",
        "stripe-key",
        "google-api-key",
        "slack-token",
        "gitlab-token",
        "aws-secret-key",
        "url-credential",
    ];

    #[test]
    fn a_change_to_the_secret_patterns_comes_with_a_doc_index_version_bump() {
        // [MAGI S6 MS3 loop4, R1 Melchior] Stored chunks are masked and hashed with the patterns
        // of their time; a source already indexed is never re-chunked until DOC_INDEX_VERSION
        // changes. A pattern added, removed or reordered fails here: bump DOC_INDEX_VERSION (and
        // its value below) and update the list in the same change.
        let names: Vec<&str> = crate::review::secrets::SECRET_PATTERNS
            .iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(
            (DOC_INDEX_VERSION, names.as_slice()),
            (5, PATTERNS_OF_THIS_DOC_INDEX_VERSION),
            "the secret patterns changed: bump DOC_INDEX_VERSION and update this list"
        );
    }

    #[test]
    fn a_backslash_path_is_not_a_doc_source_on_any_platform() {
        // [MAGI S6 MS3 loop16, R1 Caspar] Doc paths come from git (`ls-files`, `ls-tree`), which
        // prints `/` on every platform, so a working-tree path never carries a backslash; one
        // that did would be a file NAME on Unix and must not pass as `docs/`. Pinned so that
        // Windows-only CI and a Linux build agree.
        assert!(!is_doc_source("docs\\guide.md"));
        assert!(!is_doc_source("adr\\0001.md"));
        assert!(is_doc_source("docs/guide.md"));
    }

    #[test]
    fn a_secret_split_by_a_hidden_character_is_netted_with_its_source() {
        // [MAGI S6 MS3 loop16, R1 Caspar] The documentation egress ran `mask_secrets` alone, so
        // a token split by a zero-width space reached the embedder whole: the batch and gate
        // paths strip hidden characters before the second look, and so must this one.
        let split = "token = ghp_AAAAAAAAAAAAAAAAAA\u{200b}AAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        assert!(
            find_secret(split).is_none(),
            "the contiguous pass misses it"
        );
        let hidden = chunk_of("docs/a.md", "Setup".into(), split.into());
        let clean = chunk_of("docs/b.md", "Setup".into(), "plain prose".into());
        let (kept, dropped) = without_secrets(vec![hidden, clean]);
        assert_eq!(dropped, ["docs/a.md"]);
        assert!(kept.iter().all(|c| c.path == "docs/b.md"), "{kept:?}");
    }
}
