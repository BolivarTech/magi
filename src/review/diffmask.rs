// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! The diff of a masked file, rebuilt or masked by line so that no byte of a masked secret leaves.
//!
//! **Egress invariant.** Every diff text this module returns is built from the (masked) sides of
//! the file or masked line by line, and its caller scans it with `find_secret` before it is
//! served; a line that no masked side shows verbatim never travels. The two paths, [`rebuilt_diff`]
//! (lines taken from the sides at the numbers the hunk headers give) and [`fallback_diff`] (the
//! raw diff masked by line), are checked against each other by `no_byte_of_a_secret_leaves`.

use std::collections::HashSet;

use super::secrets::{find_secret, holds_key_block, mask_secrets, REDACTED_OPEN};

/// The marker that opens and closes a hunk header line (`@@ -a,b +c,d @@`).
pub(super) const HUNK_START: &str = "@@";

/// The name a redacted line carries when it holds no secret of its own: it was left out because
/// no masked side shows it (drift between the diff and the sides, or one half of a secret that
/// spans lines), not because it is a credential, and the model is not told it is one.
pub(super) const DRIFT_PATTERN: &str = "diff-drift";

/// The prefix of the line that opens each file's section of a git diff and so ends the previous
/// section's last hunk.
const SECTION_START: &str = "diff ";

/// [`masked_raw_diff`], scanned: `None` when its own output still holds a secret, so the caller
/// leaves the file out as a `secret` skip instead of sending it. A body line that a clean side
/// shows verbatim is kept as it is, so a secret that exists only across adjacent diff lines
/// (a name and its quoted value on neighbouring context lines) survives the by-line masking; the
/// re-scan is what keeps every egress text validated, as the rebuilt path already is. O(diff + sides).
// MAGI FIX (MS3 §6 loop13): the fallback was the one egress text never re-scanned.
pub(super) fn fallback_diff(diff: &str, new: Option<&[u8]>, old: Option<&[u8]>) -> Option<String> {
    let masked = masked_raw_diff(diff, new, old);
    find_secret(&masked).is_none().then_some(masked)
}
/// `diff` masked by line, when [`rebuilt_diff`] cannot rebuild it: a header line outside a hunk
/// is masked with [`mask_secrets`]; an `@@` line keeps its ranges and its function context only
/// when a masked side shows it ([`shown_by_a_side`]); a body line is kept only when a masked
/// side shows its text verbatim (context and added lines on `new`, context and deleted on `old`),
/// else its text becomes `<redacted: pattern>` after its marker. [`mask_secrets`] alone on the
/// diff is not enough: a hunk can hold a secret's value without the key line that makes it one
/// (`diff_alone_is_unsafe_to_mask`), and every line
/// holding a byte of a masked secret differs from its masked side. So masking, never omission, and
/// no byte of the secret leaves (RV22's tie-break) (`unrebuilt_diff_is_masked`). O(diff + sides).
pub(super) fn masked_raw_diff(diff: &str, new: Option<&[u8]>, old: Option<&[u8]>) -> String {
    let (new_vec, old_vec) = (
        side_lines(new).unwrap_or_default(),
        side_lines(old).unwrap_or_default(),
    );
    let new_lines: HashSet<&str> = new_vec.iter().copied().collect();
    let old_lines: HashSet<&str> = old_vec.iter().copied().collect();
    let redacted = |text: &str| {
        let name = find_secret(text).unwrap_or(DRIFT_PATTERN);
        format!("{REDACTED_OPEN}{name}>")
    };
    let mut index: Option<SideIndex<'_>> = None;
    let mut in_hunk = false;
    let mut out = String::with_capacity(diff.len());
    for line in diff.split_inclusive('\n') {
        let (body, end) = line.strip_suffix('\n').map_or((line, ""), |b| (b, "\n"));
        if body.starts_with("@@") {
            in_hunk = true;
            // MAGI FIX (MS3 §6 loop12): the function context is a line of the file, so, as in
            // `rebuilt_diff`, it stays only when a masked side shows it; masking the line alone
            // would leave a value whose key sits on another line. A header that does not parse
            // is redacted whole.
            let kept = match hunk_header(body) {
                Some((header, ..)) => {
                    let context = body.get(header.len()..).unwrap_or_default();
                    let idx = index.get_or_insert_with(|| SideIndex::new(&new_vec, &old_vec));
                    if shown_by_a_side(context, idx) {
                        body.to_string()
                    } else {
                        header.to_string()
                    }
                }
                None => format!("@@ {}", redacted(body)),
            };
            out.push_str(&kept);
            out.push_str(end);
            continue;
        }
        // MAGI FIX (MS3 §6 loop14): a new file's section ends the hunk, so its header lines are
        // masked as headers, not redacted as body lines.
        if body.starts_with(SECTION_START) {
            in_hunk = false;
        }
        let marker = body.chars().next();
        let text = body.get(1..).unwrap_or_default();
        let kept = match (in_hunk, marker) {
            (false, _) | (true, Some('\\')) => mask_secrets(body),
            (true, Some(' ')) if new_lines.contains(text) && old_lines.contains(text) => {
                body.into()
            }
            (true, Some('+')) if new_lines.contains(text) => body.into(),
            (true, Some('-')) if old_lines.contains(text) => body.into(),
            (true, Some(m @ (' ' | '+' | '-'))) => format!("{m}{}", redacted(text)),
            (true, _) => redacted(body),
        };
        out.push_str(&kept);
        out.push_str(end);
    }
    out
}

/// `diff` with every content line taken from the (masked) side it shows: a context or added line
/// from `new` (a context line compared with `old` too, never written from it), a deleted line
/// from `old`, at the numbers its hunk header gives. Header lines and `\ No newline` markers are
/// kept; a hunk header keeps its ranges, and its function context only when a masked side line
/// shows it as it is ([`shown_by_a_side`]). `None` when a line names a side line that does not
/// exist, a header does not parse or a body line has no marker. A body line that differs from
/// its side's is replaced by the side's text, and counted unless masking the diff's line gives
/// the side's; the caller logs the count (RV21) and checks the result for secrets again. Header
/// lines are kept as they are; one that holds a secret (a secret-shaped path) fails the caller's
/// re-scan, and the file is then masked by line ([`fallback_diff`], headers through
/// [`mask_secrets`]), never kept with it.
///
/// Subset of git's unified diff read: lines outside hunks verbatim; a hunk opens at
/// `@@ -a[,b] +c[,d] @@` (`@@@` gives `None`); body lines start with ` `, `+`, `-` or `\`,
/// anything else gives `None`, and the caller masks the file by line ([`fallback_diff`]) and
/// leaves it out (`secret`) only when that fallback is refused. `in_hunk` is not reset at a
/// section boundary here: a section's own header lines end the hunk, as in the raw masker's
/// parse, which reads one section.
///
/// O(diff + sides) plus O(n log n) to index the n side lines once and O(log n × 80) per hunk for
/// the function context (masked files only), plus O(diff × s) for the divergence test, which
/// runs [`mask_secrets`] (a full pattern scan and one `String`) on every body line, s being the
/// scan's cost per line.
///
/// # Returns
///
/// The rebuilt diff and the number of lines that diverged beyond a mask, or `None`.
// MAGI FIX (MS3 §6 loop5): the divergence was replaced silently, and the parsed subset unstated.
// MAGI FIX (MS3 §6 loop7): the bound left the per-hunk context scan out (R6a Melchior).
pub(super) fn rebuilt_diff(
    diff: &str,
    new: Option<&[u8]>,
    old: Option<&[u8]>,
) -> Option<(String, usize)> {
    let (new_lines, old_lines) = (side_lines(new)?, side_lines(old)?);
    let (mut next_new, mut next_old) = (0usize, 0usize);
    let mut in_hunk = false;
    let mut diverged = 0usize;
    let mut index: Option<SideIndex<'_>> = None;
    let mut out = String::with_capacity(diff.len());
    for raw in diff.split_inclusive('\n') {
        let (line, eol) = match raw.strip_suffix('\n') {
            Some(line) => (line, "\n"),
            None => (raw, ""),
        };
        if line.starts_with(HUNK_START) {
            let (header, old_start, new_start) = hunk_header(line)?;
            out.push_str(header);
            // MAGI FIX (MS3 §6 loop6): the function context is a line of the file; it was
            // dropped (R6a Melchior). It is kept when a masked side line starts with it as it
            // is, so it carries what the whole-file mask left there and nothing more; masking
            // the context alone would be the diff-alone case (a YAML value with no key line).
            let context = line.get(header.len()..).unwrap_or_default();
            let index = index.get_or_insert_with(|| SideIndex::new(&new_lines, &old_lines));
            if shown_by_a_side(context, index) {
                out.push_str(context);
            }
            next_old = old_start.saturating_sub(1);
            next_new = new_start.saturating_sub(1);
            in_hunk = true;
        } else if !in_hunk || line.starts_with('\\') {
            out.push_str(line);
        } else {
            // MAGI FIX (MS3 §6 loop7): a context line is compared with the old side too (read,
            // never written), so a stale old side counts (R6a Caspar).
            let (marker, text, old_context) = match line.chars().next()? {
                ' ' => {
                    let old_text = take(&old_lines, &mut next_old)?;
                    (' ', take(&new_lines, &mut next_new)?, Some(old_text))
                }
                '+' => ('+', take(&new_lines, &mut next_new)?, None),
                '-' => ('-', take(&old_lines, &mut next_old)?, None),
                _ => return None,
            };
            // MAGI FIX (MS3 §6 loop6): a difference counts unless masking the diff's own line
            // gives the side's; a mask on the side's line no longer excused any difference.
            let shown = line.get(1..).unwrap_or_default();
            // Declared (MS3 §6 loop12): the side was masked by `mask_secrets_keeping_lines`, this
            // one line by `mask_secrets`. For a secret whose span crosses a line break the two
            // differ (the side's later lines are blanked, the diff's line alone shows no key), so
            // the count can be HIGH, never low. A counted line is only logged by the caller (a
            // WARN, RV21): the rebuilt text is the sides' own masked lines either way, so a high
            // count costs a log entry and never a leak (`raw_diff_masks_hunk_context`,
            // `diff_alone_is_unsafe_to_mask` pin the by-line path a failed re-scan takes).
            let differs = |side: &str| shown != side && mask_secrets(shown) != side;
            if differs(text) || old_context.is_some_and(differs) {
                diverged = diverged.saturating_add(1);
            }
            out.push(marker);
            out.push_str(text);
        }
        out.push_str(eol);
    }
    Some((out, diverged))
}

/// Every line of both masked sides, sorted once per file, for [`shown_by_a_side`]: the lines
/// that start with a text are contiguous there, right at the first line not less than it.
struct SideIndex<'a>(Vec<&'a str>);

impl<'a> SideIndex<'a> {
    /// The sorted lines of `new` and `old`. O(n log n) for n lines.
    fn new(new: &[&'a str], old: &[&'a str]) -> Self {
        let mut lines: Vec<&str> = new.iter().chain(old).copied().collect();
        visited(lines.len());
        lines.sort_unstable();
        Self(lines)
    }
}

/// Whether a hunk header's function context (` <text>` after the closing `@@`, git cuts it to
/// 80 bytes) is shown as it is by some line of the masked sides: the line starts with `<text>`.
/// An empty context is shown. A context that carries a secret byte never matches, since the
/// sides carry its mask there instead, and neither does one that opens a private-key block
/// (such sides never reach here). O(log lines · context) on `index`: a line starting with
/// `text` sorts at or after it and before any other line that does not
/// (`many_hunks_stay_near_linear`). Any line counts, not the one git took it from: a context
/// shared by two lines is kept either way (cosmetic; the text is the masked side's, declared).
// MAGI FIX (MS3 §6 loop10): each hunk scanned both whole sides, quadratic in the hunks.
fn shown_by_a_side(context: &str, index: &SideIndex<'_>) -> bool {
    let Some(text) = context.strip_prefix(' ').filter(|t| !t.is_empty()) else {
        return context.is_empty();
    };
    let at = index.0.partition_point(|line| *line < text);
    visited(index.0.len().max(2).ilog2() as usize);
    !holds_key_block(text) && index.0.get(at).is_some_and(|line| line.starts_with(text))
}

// The side-line visit counter of the function-context lookup: a no-op outside test builds, so
// no test concern sits in the hot loop (counter).
use counter::visited;
#[cfg(test)]
pub(super) use counter::CONTEXT_VISITS;

/// The release build's [`visited`]: nothing.
#[cfg(not(test))]
mod counter {
    /// Counts nothing.
    pub(super) fn visited(_n: usize) {}
}

/// The lines of a side, each without its `\n` (a `\r` stays, as git shows it); none for an
/// absent side; `None` when the side is not UTF-8. O(side).
fn side_lines(side: Option<&[u8]>) -> Option<Vec<&str>> {
    match side {
        Some(bytes) => Some(std::str::from_utf8(bytes).ok()?.split('\n').collect()),
        None => Some(Vec::new()),
    }
}

/// The line at `*next` of `lines`, advancing `*next`. O(1).
fn take<'a>(lines: &[&'a str], next: &mut usize) -> Option<&'a str> {
    let line = lines.get(*next).copied()?;
    *next = next.saturating_add(1);
    Some(line)
}

/// A hunk header `@@ -a[,b] +c[,d] @@[ context]`: the header up to its closing `@@`, and the
/// first old and new line numbers. `None` when it does not parse. O(line).
fn hunk_header(line: &str) -> Option<(&str, usize, usize)> {
    let rest = line.get(HUNK_START.len()..)?;
    let close = rest.find(HUNK_START)?;
    let header = line.get(..HUNK_START.len() + close + HUNK_START.len())?;
    let mut parts = rest.get(..close)?.split_whitespace();
    let start = |part: Option<&str>, sign: char| -> Option<usize> {
        let side = part?.strip_prefix(sign)?;
        side.split(',').next()?.parse().ok()
    };
    let old = start(parts.next(), '-')?;
    let new = start(parts.next(), '+')?;
    Some((header, old, new))
}

/// The test build's counter, read by `many_hunks_stay_near_linear`.
#[cfg(test)]
mod counter {
    thread_local! {
        /// Side lines the function-context lookup examined on this thread.
        pub(in crate::review) static CONTEXT_VISITS: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    /// Counts `n` side-line visits of the function-context lookup.
    pub(super) fn visited(n: usize) {
        CONTEXT_VISITS.with(|c| c.set(c.get().saturating_add(n)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::secrets::mask_secrets_keeping_lines;

    /// Deterministic generator (an LCG): the cases are fixed, and no dependency is added.
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    /// A key-shaped token of the given kind (`n` even: AWS access key, odd: a `ghp_` token), built
    /// at run time so no literal in the source looks like a secret.
    fn secret_token(state: &mut u64, n: u64) -> String {
        let (prefix, len) = if n.is_multiple_of(2) {
            (["AK", "IA"].concat(), 16)
        } else {
            (["gh", "p_"].concat(), 36)
        };
        let alphabet: Vec<char> = ('A'..='Z').chain('0'..='9').collect();
        let body: String = (0..len)
            .filter_map(|_| alphabet.get(next(state) as usize % alphabet.len()))
            .collect();
        format!("{prefix}{body}")
    }

    /// A git-shaped one-hunk diff over `common` lines with `added` inserted at `p` and `removed`
    /// deleted at `q`, plus the new and old sides it describes.
    fn case(
        common: &[String],
        p: usize,
        q: usize,
        added: &str,
        removed: &str,
    ) -> (String, String, String) {
        let (mut new, mut old, mut body) = (String::new(), String::new(), String::new());
        for (i, line) in common.iter().enumerate() {
            if i == p {
                new.push_str(&format!("{added}\n"));
                body.push_str(&format!("+{added}\n"));
            }
            if i == q {
                old.push_str(&format!("{removed}\n"));
                body.push_str(&format!("-{removed}\n"));
            }
            new.push_str(&format!("{line}\n"));
            old.push_str(&format!("{line}\n"));
            body.push_str(&format!(" {line}\n"));
        }
        let n = common.len() + 1;
        (
            format!("@@ -1,{n} +1,{n} @@ {}\n{body}", "fn f() {"),
            new,
            old,
        )
    }

    #[test]
    fn no_byte_of_a_secret_leaves() {
        // Both masking paths, over generated cases (an added, a removed and a context line each
        // holding a secret, at varying positions): no 8-byte window of any secret may appear in a
        // produced diff, and the rebuilt path must actually produce one in most cases.
        let mut state = 7u64;
        let mut rebuilt_count = 0;
        for n in 0..32u64 {
            let (s_add, s_del, s_ctx) = (
                secret_token(&mut state, n),
                secret_token(&mut state, n + 1),
                secret_token(&mut state, n + 2),
            );
            let size = 6 + (next(&mut state) as usize % 5);
            let ctx_at = next(&mut state) as usize % size;
            let common: Vec<String> = (0..size)
                .map(|i| {
                    if i == ctx_at {
                        format!("let c = \"{s_ctx}\";")
                    } else {
                        format!("let v{i} = {i};")
                    }
                })
                .collect();
            let (p, q) = (
                next(&mut state) as usize % size,
                next(&mut state) as usize % size,
            );
            let (diff, new, old) = case(
                &common,
                p,
                q,
                &format!("let a = \"{s_add}\";"),
                &format!("let d = \"{s_del}\";"),
            );
            let mask =
                |s: &str| mask_secrets_keeping_lines(s).map_or_else(|| s.to_string(), |m| m.text);
            let (new_m, old_m) = (mask(&new), mask(&old));
            let rebuilt =
                rebuilt_diff(&diff, Some(new_m.as_bytes()), Some(old_m.as_bytes())).map(|(d, _)| d);
            rebuilt_count += usize::from(rebuilt.is_some());
            let fallback = masked_raw_diff(&diff, Some(new_m.as_bytes()), Some(old_m.as_bytes()));
            for out in rebuilt.iter().chain(std::iter::once(&fallback)) {
                for s in [&s_add, &s_del, &s_ctx] {
                    for window in s.as_bytes().windows(8) {
                        let w = String::from_utf8_lossy(window);
                        assert!(!out.contains(w.as_ref()), "case {n}: {w} leaked in {out}");
                    }
                }
            }
        }
        assert!(
            rebuilt_count > 16,
            "the rebuilt path ran in only {rebuilt_count} of 32 cases"
        );
    }

    #[test]
    fn fallback_marks_each_span_by_its_own_pattern() {
        // [MAGI S6 MS3 loop14, R6a1 Caspar] Every redacted line carried the file's first
        // pattern, so a clean line that merely drifted from its side was shown to the model as
        // a credential. A line holding a secret names its own pattern; any other redacted
        // line is declared as drift.
        let token = ["gh", "p_", &"A".repeat(36)].concat(); // allow-secret-scan
        let named = find_secret(&token).expect("premise: a token is a secret");
        let diff =
            format!("@@ -1,1 +1,3 @@\n fn f() {{}}\n+let t = \"{token}\";\n+let drifted = 1;\n");
        let out = masked_raw_diff(&diff, Some(b"fn f() {}\n"), Some(b"fn f() {}\n"));
        assert!(!out.contains(&token), "{out}");
        assert!(out.contains(&format!("+{REDACTED_OPEN}{named}>")), "{out}");
        assert!(
            out.contains(&format!("+{REDACTED_OPEN}diff-drift>")),
            "{out}"
        );
        assert!(!out.contains("aws-secret-key"), "{out}");
    }

    #[test]
    fn fallback_refusal_is_reachable() {
        // [MAGI S6 MS3 loop14, R6a1 Caspar] The `None` arm was pinned by a test whose input never
        // reached it. A two-line secret whose lines both sit verbatim in a clean old and new
        // side survives by-line masking as context, and the re-scan refuses it.
        let value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"; // allow-secret-scan
        let side = format!("aws_secret_key =\nfn f() {{}}\n\"{value}\"\n");
        let diff =
            "@@ -1,5 +1,5 @@\n aws_secret_key =\n \"VALUE\"\n fn f() {}\n fn f() {}\n fn f() {}\n"
                .replace("VALUE", value);
        assert_eq!(find_secret(&side), None, "premise: each side is clean");
        assert!(find_secret(&diff).is_some(), "premise: the diff is not");
        assert_eq!(
            fallback_diff(&diff, Some(side.as_bytes()), Some(side.as_bytes())),
            None
        );
    }

    #[test]
    fn crlf_side_lines_match_a_crlf_diff_without_a_divergence() {
        // [MAGI S6 MS3 loop15, R6a1] a side line keeps its `\r` as git shows it, so a CRLF file's
        // diff lines (which carry the same `\r`) match their side lines: no divergence is
        // counted and the rebuilt diff is the diff.
        let new = b"a\r\nc\r\n";
        let old = b"a\r\nb\r\n";
        let diff = "@@ -1,2 +1,2 @@\n a\r\n-b\r\n+c\r\n";
        assert_eq!(
            rebuilt_diff(diff, Some(new), Some(old)),
            Some((diff.to_string(), 0))
        );
    }

    #[test]
    fn a_new_file_header_ends_the_hunk() {
        // [MAGI S6 MS3 loop14, R6a1 Balthasar] `in_hunk` was never reset: a header line after a
        // hunk was read as a body line and redacted whole instead of masked as a header.
        let diff = "@@ -1,1 +1,1 @@\n fn f() {}\ndiff --git a/b.rs b/b.rs\n--- a/b.rs\n";
        let out = masked_raw_diff(diff, Some(b"fn f() {}\n"), Some(b"fn f() {}\n"));
        assert!(
            out.contains("diff --git a/b.rs b/b.rs\n--- a/b.rs\n"),
            "{out}"
        );
    }
}
