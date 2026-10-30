// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The product's secret detector: what a review never sends out of the process.
//!
//! Every byte a review sends to a model passes through [`find_secret`] first. The patterns are
//! matched without dependencies (a prefix, a character class and a minimum length), in time
//! linear in the text. Not covered, by design: JWTs (any base64url JSON looks like one), Azure
//! keys, high-entropy strings with no prefix, and the base64 body of a PEM or armored OpenPGP
//! private key without its `BEGIN` line. The minimum lengths can miss a shorter real key; that is the trade-off with
//! false positives on ordinary code. So are the left boundaries: a short prefix (`sk-`, `ghp_`, `AKIA`, …) glued to a
//! word character is not a token, because base64 runs and kebab-case names hold them by chance; the distinctive
//! `sk-ant-` and `github_pat_` match wherever they occur. A PEM or OpenPGP header must be in
//! upper case as every standard tool writes it: a hand-made `-----begin private key-----` is
//! not detected.

use std::ops::Range;

/// What may follow a token prefix.
#[derive(Clone, Copy)]
enum Tail {
    /// `[A-Za-z0-9_-]`.
    Word,
    /// `[A-Z0-9]`.
    UpperDigit,
    /// `[A-Za-z0-9]`.
    Alnum,
    /// `[A-Za-z0-9/+]`.
    Base64,
}

impl Tail {
    /// Whether `b` belongs to the class.
    fn admits(self, b: u8) -> bool {
        match self {
            Self::Word => is_word(b),
            Self::UpperDigit => b.is_ascii_uppercase() || b.is_ascii_digit(),
            Self::Alnum => b.is_ascii_alphanumeric(),
            Self::Base64 => b.is_ascii_alphanumeric() || b == b'/' || b == b'+',
        }
    }
}

/// How a pattern matches.
enum Matcher {
    /// A prefix with a left boundary, then a tail of at least `min` characters of a class.
    Token {
        /// The prefixes, any of which opens the token.
        prefixes: &'static [&'static str],
        /// The tail's character class.
        tail: Tail,
        /// The tail's minimum length.
        min: usize,
        /// What may not follow the prefix (the Anthropic prefix inside the OpenAI one).
        not_followed_by: Option<&'static str>,
        /// Whether the prefix needs no left boundary: it is distinctive enough (two parts and a
        /// separator) that no base64 run or kebab-case name holds it by chance.
        glued: bool,
    },
    /// A PEM private-key header.
    PemPrivateKey,
    /// A fixed header, found anywhere in the text (an armored OpenPGP private key).
    Header(&'static str),
    /// The AWS secret access key, recognized by its name and an assignment.
    AwsSecretKey,
}

/// One named secret pattern.
pub(crate) struct SecretPattern {
    /// The pattern's name, as the run log and a redacted path show it.
    pub name: &'static str,
    /// How it matches.
    matcher: Matcher,
}

/// The opening of a PEM block (split so the source holds no PEM header).
const PEM_BEGIN: &str = concat!("-----", "BEGIN ");
/// The longest type word a PEM private-key header may name (`RSA`, `EC`, `ECDSA`, `OPENSSH`,
/// `ENCRYPTED`, …), so a header is checked in constant time.
const PEM_MAX_TYPE_WORD: usize = 16;
/// The most type words a PEM private-key header may name before `PRIVATE KEY` (`SSH2
/// ENCRYPTED`, `RSA ENCRYPTED`), so a header is checked in constant time.
const PEM_MAX_TYPE_WORDS: usize = 4;
/// The header of an SSH2 private key (RFC 4716 armor: four dashes and spaces; split so the
/// source holds no such header). Its `END` line has the same armor, so its mask runs to the end
/// of the text: the safe direction.
const SSH2_PRIVATE_KEY: &str = concat!("---- BEGIN SSH2 ", "ENCRYPTED PRIVATE KEY ----");
/// The end of a PEM private-key header.
const PEM_PRIVATE_KEY: &str = "PRIVATE KEY-----";
/// The header of an armored OpenPGP private key (split so the source holds no such header).
const PGP_PRIVATE_KEY: &str = concat!("-----", "BEGIN PGP PRIVATE KEY BLOCK-----");
/// The opening of the `END` line of an armored block (split so the source holds no PEM line).
const ARMOR_END: &str = concat!("-----", "END ");
/// The dashes that close an armor line.
const ARMOR_DASHES: &str = "-----";
/// What opens the mark that replaces a masked secret: `<redacted: <pattern name>>`.
const REDACTED_OPEN: &str = "<redacted: ";
/// The words of the AWS secret access key's name, in order, each with whether it may be left
/// out: `AWS_SECRET_KEY` and the STS field `SecretAccessKey` name the same key. A name must keep
/// `aws` or `access` (see [`aws_name_end`]), so a generic `secret_key` is not taken for it.
const AWS_SECRET_WORDS: &[(&str, bool)] = &[
    ("aws", true),
    ("secret", false),
    ("access", true),
    ("key", false),
];
/// The minimum length of the AWS secret access key.
const AWS_SECRET_MIN: usize = 40;
/// The blanks allowed around the `=` or `:` of an AWS secret key assignment.
const BLANKS: &[u8] = b" \t";
/// The quotes allowed after an AWS secret key's name and before the key.
const QUOTES: &[u8] = b"\"'";

/// The patterns, in the order that decides the name of a match: `anthropic-key` comes before
/// `openai-key` because an Anthropic key also starts with `sk-`. A Stripe key needs no such
/// care: `sk_live_` opens with an underscore, never with `sk-`, so it is always `stripe-key`
/// whatever the order (pinned by `common_provider_keys_and_the_aws_secret_key_are_secrets`).
pub(crate) const SECRET_PATTERNS: &[SecretPattern] = &[
    SecretPattern {
        name: "anthropic-key",
        matcher: Matcher::Token {
            prefixes: &["sk-ant-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: None,
            glued: true,
        },
    },
    SecretPattern {
        name: "openai-key",
        matcher: Matcher::Token {
            prefixes: &["sk-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: Some("ant-"),
            glued: false,
        },
    },
    SecretPattern {
        name: "github-token",
        matcher: Matcher::Token {
            prefixes: &["ghp_", "gho_", "ghu_", "ghs_", "ghr_"],
            tail: Tail::Word,
            min: 36,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "github-token",
        matcher: Matcher::Token {
            prefixes: &["github_pat_"],
            tail: Tail::Word,
            min: 22,
            not_followed_by: None,
            glued: true,
        },
    },
    SecretPattern {
        name: "aws-access-key",
        matcher: Matcher::Token {
            prefixes: &["AKIA", "ASIA"],
            tail: Tail::UpperDigit,
            min: 16,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "pem-private-key",
        matcher: Matcher::PemPrivateKey,
    },
    SecretPattern {
        name: "pem-private-key",
        matcher: Matcher::Header(SSH2_PRIVATE_KEY),
    },
    SecretPattern {
        name: "pgp-private-key",
        matcher: Matcher::Header(PGP_PRIVATE_KEY),
    },
    SecretPattern {
        name: "stripe-key",
        matcher: Matcher::Token {
            prefixes: &["sk_live_", "rk_live_"],
            tail: Tail::Alnum,
            min: 24,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "google-api-key",
        matcher: Matcher::Token {
            prefixes: &["AIza"],
            tail: Tail::Word,
            min: 35,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "slack-token",
        matcher: Matcher::Token {
            prefixes: &["xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"],
            tail: Tail::Word,
            min: 10,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "gitlab-token",
        matcher: Matcher::Token {
            prefixes: &["glpat-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: None,
            glued: false,
        },
    },
    SecretPattern {
        name: "aws-secret-key",
        matcher: Matcher::AwsSecretKey,
    },
];

/// `[A-Za-z0-9_-]`: the characters of a token, and the ones a left boundary may not be.
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Whether `at` follows a run of `-` that no word character precedes: a removed-line marker (a
/// `-` that opens a line), a `--` comment (`---` on a removed line, `+--` on an added one) or an
/// option-like `--`. [`is_word`] counts `-` as part of a token, so without this every token on a
/// removed line, or after a `--` comment, would fail its left boundary. A run glued to a word
/// (`task--`) is still part of that word. O(length of the run).
// MAGI FIX (S6 loop 5): only one `-` after a newline was a boundary, so `---ghp_…` was sent.
fn after_diff_marker(bytes: &[u8], at: usize) -> bool {
    let run = bytes
        .get(..at)
        .unwrap_or_default()
        .iter()
        .rev()
        .take_while(|b| **b == b'-')
        .count();
    run > 0
        && at
            .checked_sub(run + 1)
            .and_then(|before| bytes.get(before))
            .is_none_or(|b| !is_word(*b))
}

/// Whether at least `min` bytes of class `tail` start at `at`. O(min).
fn tail_at_least(bytes: &[u8], at: usize, tail: Tail, min: usize) -> bool {
    bytes
        .get(at..)
        .unwrap_or_default()
        .iter()
        .take(min)
        .take_while(|b| tail.admits(**b))
        .count()
        >= min
}

/// Whether `bytes` holds `word` at `at`, ignoring ASCII case.
fn word_at(bytes: &[u8], at: usize, word: &str) -> bool {
    bytes
        .get(at..at.saturating_add(word.len()))
        .is_some_and(|s| s.eq_ignore_ascii_case(word.as_bytes()))
}

/// The first index at or after `at` whose byte is not in `skip`.
fn skip_bytes(bytes: &[u8], mut at: usize, skip: &[u8]) -> usize {
    while bytes.get(at).is_some_and(|b| skip.contains(b)) {
        at += 1;
    }
    at
}

impl Matcher {
    /// Whether the pattern occurs anywhere in `text`.
    ///
    /// Linear in the text: each occurrence of a prefix reads at most its minimum tail length (an
    /// occurrence inside another token fails its left boundary first, unless the prefix is
    /// glued), and only the first match reads its whole tail.
    fn matches(&self, text: &str) -> bool {
        self.first_span(text).is_some()
    }

    /// The byte range of the first occurrence of the pattern in `text`, if any.
    ///
    /// A token's range runs to the end of its tail; the AWS secret key's from its name to the end
    /// of the key; a PEM or OpenPGP private key's from its header through the end of its `END`
    /// line, or to the end of the text when there is none (the base64 body matches no pattern,
    /// so it is masked with its header).
    // MAGI FIX (S6 loop 3): the outward scrubber masks the span; `matches` is its `is_some`.
    fn first_span(&self, text: &str) -> Option<Range<usize>> {
        let bytes = text.as_bytes();
        match self {
            Self::Token { prefixes, .. } => (0..prefixes.len())
                .filter_map(|lane| self.token_span(lane, text))
                .min_by_key(|r| r.start),
            Self::PemPrivateKey => text.match_indices(PEM_BEGIN).find_map(|(at, _)| {
                let mut rest = at + PEM_BEGIN.len();
                // MAGI FIX (S6 loop 4): any one type word, not a closed list (`ECDSA` was sent).
                // MAGI FIX (S6 loop 7): up to PEM_MAX_TYPE_WORDS of them (`SSH2 ENCRYPTED`).
                for _ in 0..PEM_MAX_TYPE_WORDS {
                    if word_at(bytes, rest, PEM_PRIVATE_KEY) {
                        break;
                    }
                    match pem_type_word(bytes, rest) {
                        Some(word) => rest += word + 1,
                        None => break,
                    }
                }
                bytes
                    .get(rest..)
                    .is_some_and(|r| r.starts_with(PEM_PRIVATE_KEY.as_bytes()))
                    .then(|| at..armor_end(text, rest + PEM_PRIVATE_KEY.len()))
            }),
            Self::Header(header) => text
                .find(header)
                .map(|at| at..armor_end(text, at + header.len())),
            Self::AwsSecretKey => {
                (0..bytes.len()).find_map(|at| aws_secret_end(bytes, at).map(|end| at..end))
            }
        }
    }
}

/// The length of the PEM type word that starts at `at` (1 to [`PEM_MAX_TYPE_WORD`] characters of
/// `[A-Z0-9]` followed by one space), or `None` when none does. O(1).
fn pem_type_word(bytes: &[u8], at: usize) -> Option<usize> {
    let word = bytes
        .get(at..)
        .unwrap_or_default()
        .iter()
        .take(PEM_MAX_TYPE_WORD + 1)
        .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        .count();
    (1..=PEM_MAX_TYPE_WORD)
        .contains(&word)
        .then_some(word)
        .filter(|w| bytes.get(at + w) == Some(&b' '))
}

impl Matcher {
    /// How many independent searches the pattern runs: one per prefix of a token, one otherwise.
    fn lanes(&self) -> usize {
        match self {
            Self::Token { prefixes, .. } => prefixes.len(),
            _ => 1,
        }
    }

    /// The first span of search `lane` in `text`: of a token's `lane`-th prefix alone, or
    /// [`Matcher::first_span`] for any other pattern. O(n).
    fn lane_span(&self, lane: usize, text: &str) -> Option<Range<usize>> {
        match self {
            Self::Token { .. } => self.token_span(lane, text),
            _ => self.first_span(text),
        }
    }

    /// The first token of the `lane`-th prefix in `text` that has a left boundary (or needs
    /// none: `glued`); `None` for any other pattern. O(n).
    fn token_span(&self, lane: usize, text: &str) -> Option<Range<usize>> {
        let Self::Token {
            prefixes, glued, ..
        } = self
        else {
            return None;
        };
        let prefix = prefixes.get(lane)?;
        let bytes = text.as_bytes();
        text.match_indices(prefix).find_map(|(at, _)| {
            // MAGI FIX (S6 loop 4): a distinctive prefix glued to a word was sent.
            let bounded = *glued
                || at
                    .checked_sub(1)
                    .and_then(|before| bytes.get(before))
                    .is_none_or(|b| !is_word(*b))
                || after_diff_marker(bytes, at);
            bounded.then(|| self.token_at(lane, bytes, at)).flatten()
        })
    }

    /// The token of the `lane`-th prefix that opens at `at`, its left boundary aside; `None`
    /// when none opens there or the pattern is not a token. O(min) unless it matches.
    fn token_at(&self, lane: usize, bytes: &[u8], at: usize) -> Option<Range<usize>> {
        let Self::Token {
            prefixes,
            tail,
            min,
            not_followed_by,
            ..
        } = self
        else {
            return None;
        };
        let prefix = prefixes.get(lane)?;
        let end = at + prefix.len();
        let opens = bytes.get(at..end) == Some(prefix.as_bytes());
        let excluded = not_followed_by.is_some_and(|n| word_at(bytes, end, n));
        (opens && !excluded && tail_at_least(bytes, end, *tail, *min))
            .then(|| at..tail_end(bytes, end, *tail))
    }
}

/// The index past the run of class `tail` that starts at `at`.
fn tail_end(bytes: &[u8], at: usize, tail: Tail) -> usize {
    at + bytes
        .get(at..)
        .unwrap_or_default()
        .iter()
        .take_while(|b| tail.admits(**b))
        .count()
}

/// The end of an armored block whose header ends at `from`: past the `-----` that closes its
/// `END` line, or the end of the text when the block has no `END` line.
fn armor_end(text: &str, from: usize) -> usize {
    text.get(from..)
        .and_then(|rest| {
            let end = rest.find(ARMOR_END)? + ARMOR_END.len();
            let close = rest.get(end..)?.find(ARMOR_DASHES)? + ARMOR_DASHES.len();
            Some(from + end + close)
        })
        .unwrap_or(text.len())
}

/// Where the name of an AWS secret access key that opens at `at` ends: the words of
/// [`AWS_SECRET_WORDS`] in order and in any case, with at most one `_` or `-` between two of
/// them, the optional ones present or not, and at least one optional word present. `None` when
/// no such name opens there. O(1).
// MAGI FIX (S6 loop 7): `AWS_SECRET_KEY` and `SecretAccessKey` were not names of the key.
fn aws_name_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut pos = at;
    let mut started = false;
    let mut kept_optional = false;
    for (word, optional) in AWS_SECRET_WORDS {
        let separator =
            usize::from(started && bytes.get(pos).is_some_and(|b| *b == b'_' || *b == b'-'));
        if word_at(bytes, pos + separator, word) {
            pos += separator + word.len();
            started = true;
            kept_optional |= *optional;
        } else if !optional {
            return None;
        }
    }
    kept_optional.then_some(pos)
}

/// Where the AWS secret access key that opens at `at` ends: its name ([`aws_name_end`]), an
/// optional quote that closes it (a JSON property, a quoted hash key), blanks, `=`, `=>` or `:`,
/// blanks, an optional quote, and a key of at least 40 characters. `None` when none opens there.
// MAGI FIX (S6 loop 7): `"aws_secret_access_key": "…"` and `=>` were not assignments.
fn aws_secret_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut pos = aws_name_end(bytes, at)?;
    if bytes.get(pos).is_some_and(|b| QUOTES.contains(b)) {
        pos += 1;
    }
    pos = skip_bytes(bytes, pos, BLANKS);
    if !bytes.get(pos).is_some_and(|b| *b == b'=' || *b == b':') {
        return None;
    }
    pos += 1;
    if bytes.get(pos - 1) == Some(&b'=') && bytes.get(pos) == Some(&b'>') {
        pos += 1;
    }
    pos = skip_bytes(bytes, pos, BLANKS);
    if bytes.get(pos).is_some_and(|b| QUOTES.contains(b)) {
        pos += 1;
    }
    tail_at_least(bytes, pos, Tail::Base64, AWS_SECRET_MIN)
        .then(|| tail_end(bytes, pos, Tail::Base64))
}

/// The name of the first secret pattern (in table order) that matches `text`, if any.
///
/// Patterns are tried in table order, so the name is the first pattern's, not the first
/// position's. O(n · p) for a text of n bytes and p prefixes.
///
/// # Examples
///
/// ```
/// use magi_rs::review::secrets::find_secret;
///
/// assert_eq!(find_secret("let x = 1;"), None);
/// ```
pub fn find_secret(text: &str) -> Option<&'static str> {
    SECRET_PATTERNS
        .iter()
        .find(|p| p.matcher.matches(text))
        .map(|p| p.name)
}

/// The path as it may leave the process.
///
/// A path in which [`find_secret`] finds a pattern becomes `<path redacted: <pattern name>>`;
/// any other path keeps its text with its control characters escaped
/// ([`char::escape_default`]), so a file name never replays a terminal escape.
///
/// # Examples
///
/// ```
/// use magi_rs::review::secrets::display_path;
///
/// assert_eq!(display_path("src/a.rs"), "src/a.rs");
/// assert_eq!(display_path("a\u{7}.rs"), "a\\u{7}.rs");
/// ```
pub fn display_path(path: &str) -> String {
    if let Some(name) = find_secret(path) {
        return format!("<path redacted: {name}>");
    }
    escape_controls(path)
}

/// `text` with every secret [`find_secret`] would name replaced by `<redacted: <pattern name>>`.
///
/// Each turn masks the earliest secret of what follows the last mask (ties go to table order),
/// and the end of a mask counts as a left boundary.
///
/// # Arguments
/// * `text` - Any text about to leave the process.
///
/// # Returns
/// The masked text; `text` unchanged when it holds no secret.
///
/// O(n · s) for a text of n bytes and s searches (one per token prefix, one per other pattern):
/// each search keeps its next span and runs again, from the mask's end, only when a mask passed
/// its start; the others check only whether a token opens at the mask's end.
///
/// Maintenance: the cached spans are correct only while every boundary a search over the whole
/// text can miss after a mask is the one `token_after_mask` checks. A change to a `Matcher`
/// or to a left-boundary rule needs the same change there; the property test against the
/// rescanning loop (`one_pass_masking_equals_the_rescanning_loop_…`) is what catches a mismatch.
// MAGI FIX (S6 loop 4): each turn rescanned the rest for every pattern, O(k · n · p).
pub fn mask_secrets(text: &str) -> String {
    let at = |offset: usize, r: Range<usize>| offset + r.start..offset + r.end;
    let lanes: Vec<(&SecretPattern, usize)> = SECRET_PATTERNS
        .iter()
        .flat_map(|p| (0..p.matcher.lanes()).map(move |lane| (p, lane)))
        .collect();
    let mut next: Vec<Option<Range<usize>>> = lanes
        .iter()
        .map(|(p, lane)| p.matcher.lane_span(*lane, text))
        .collect();
    let mut out = String::with_capacity(text.len());
    let mut done = 0;
    while let Some((span, name)) = next
        .iter()
        .zip(&lanes)
        .filter_map(|(span, (p, _))| span.clone().map(|s| (s, p.name)))
        .min_by_key(|(s, _)| s.start)
    {
        out.push_str(text.get(done..span.start).unwrap_or_default());
        out.push_str(REDACTED_OPEN);
        out.push_str(name);
        out.push('>');
        done = span.end;
        let rest = text.get(done..).unwrap_or_default();
        for (slot, (p, lane)) in next.iter_mut().zip(&lanes) {
            if slot.as_ref().is_some_and(|s| s.start < done) {
                *slot = p.matcher.lane_span(*lane, rest).map(|r| at(done, r));
            } else if let Some(r) = token_after_mask(p, *lane, rest) {
                *slot = Some(at(done, r));
            }
        }
    }
    out.push_str(text.get(done..).unwrap_or_default());
    out
}

/// The token of `pattern`'s search `lane` whose left boundary is the end of a mask: the only
/// tokens of `rest` (what follows the mask) that a search over the whole text can miss. One
/// opens at the start of `rest`, which is a boundary; or right behind a run of `-` that opens
/// `rest`, as [`after_diff_marker`] reads a run that opens a line. Every other match is decided
/// by bytes inside `rest`, so the cached span already holds it. A change to the boundary rules
/// must be mirrored here: the property test against the rescanning loop is what catches a
/// mismatch. O(min + length of the run).
// MAGI FIX (S6 loop 4): a token right after a mask and a `-` was left in the clear.
// MAGI FIX (S6 loop 5): a run of hyphens after a mask is a boundary too, as after a newline.
fn token_after_mask(pattern: &SecretPattern, lane: usize, rest: &str) -> Option<Range<usize>> {
    let bytes = rest.as_bytes();
    let run = bytes.iter().take_while(|b| **b == b'-').count();
    pattern.matcher.token_at(lane, bytes, 0).or_else(|| {
        (run > 0)
            .then(|| pattern.matcher.token_at(lane, bytes, run))
            .flatten()
    })
}

/// `text` as it may leave the process: [`mask_secrets`], then every control character escaped.
///
/// Single-line by contract: a newline or a tab becomes its two-character escape, which is what a
/// one-line error message needs (its callers are [`super::error::ReviewError`]'s messages).
/// Multi-line text, such as a model's reply, goes through [`mask_secrets`] alone.
///
/// # Arguments
/// * `text` - A foreign text bound for one line (git stderr, a path, a ref, an I/O message).
///
/// # Returns
/// The masked, escaped text.
pub fn outward_text(text: &str) -> String {
    escape_controls(&mask_secrets(text))
}

/// The characters that can break, reorder or hide text on a terminal or a log viewer: every
/// format character (Unicode category Cf, as of Unicode 16; the unassigned code points of the
/// tag block are included) and the line and paragraph separators (Zl, Zp, U+2028..U+2029).
/// The table is hand-kept to avoid a Unicode-data dependency; its test walks every range.
// MAGI FIX (S6 loop 5): the first list missed the separators, the interlinear annotations, the
// tags and the Cf characters outside the BMP.
const HIDDEN_FORMAT_CHARS: &[std::ops::RangeInclusive<char>] = &[
    '\u{00ad}'..='\u{00ad}',
    '\u{0600}'..='\u{0605}',
    '\u{061c}'..='\u{061c}',
    '\u{06dd}'..='\u{06dd}',
    '\u{070f}'..='\u{070f}',
    '\u{0890}'..='\u{0891}',
    '\u{08e2}'..='\u{08e2}',
    '\u{180e}'..='\u{180e}',
    '\u{200b}'..='\u{200f}',
    '\u{2028}'..='\u{202e}',
    '\u{2060}'..='\u{2064}',
    '\u{2066}'..='\u{206f}',
    '\u{feff}'..='\u{feff}',
    '\u{fff9}'..='\u{fffb}',
    '\u{110bd}'..='\u{110bd}',
    '\u{110cd}'..='\u{110cd}',
    '\u{13430}'..='\u{1343f}',
    '\u{1bca0}'..='\u{1bca3}',
    '\u{1d173}'..='\u{1d17a}',
    // MAGI FIX (S6 loop 7): the whole tag block, U+E0000 and U+E0002..U+E001F included.
    '\u{e0000}'..='\u{e007f}',
];

/// `text` with every control character escaped ([`char::escape_default`]) and every
/// [`HIDDEN_FORMAT_CHARS`] character escaped as `\u{…}`. O(n · r) for r ranges, a constant.
// MAGI FIX (S6 loop 5): a right-to-left override in a file name could reorder a report line.
fn escape_controls(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_control() {
                c.escape_default().to_string()
            } else if HIDDEN_FORMAT_CHARS.iter().any(|r| r.contains(&c)) {
                c.escape_unicode().to_string()
            } else {
                c.to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_named_secret_pattern_is_found_and_ordinary_code_is_not() {
        // [RED-DRYRUN 2026-10-17, D11] One-line lets: rustfmt splits an assert_eq! and moves the allow marker
        // off the literal's line, and tests/no_hardcoded_secrets.rs then fails (the PEM line had no marker).
        let anthropic = "let k = \"sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\";"; // allow-secret-scan
        let github = "token = ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let pem = "-----BEGIN RSA PRIVATE KEY-----"; // allow-secret-scan
        let aws = "AKIAABCDEFGHIJKLMNOP"; // allow-secret-scan
        assert_eq!(find_secret(anthropic), Some("anthropic-key"));
        assert_eq!(find_secret(github), Some("github-token"));
        assert_eq!(find_secret(pem), Some("pem-private-key"));
        assert_eq!(find_secret(aws), Some("aws-access-key"));
        assert_eq!(find_secret("fn sk_ant() {} // see sk- docs"), None);
    }

    #[test]
    fn an_openai_key_is_named_as_such_and_an_anthropic_key_is_not_mistaken_for_one() {
        // [CP2 MS2 S1 loop 1] the anthropic prefix is also an openai prefix: pattern order decides.
        let openai = "OPENAI=sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let anthropic = "K=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        assert_eq!(find_secret(openai), Some("openai-key"));
        assert_eq!(find_secret(anthropic), Some("anthropic-key"));
    }

    #[test]
    fn every_pem_private_key_header_and_temporary_aws_keys_are_secrets() {
        // [CP2 MS2 S1 loop 2] the spec says PEM private keys and AWS keys in general.
        let ec = "-----BEGIN EC PRIVATE KEY-----"; // allow-secret-scan
        let pkcs8 = "-----BEGIN PRIVATE KEY-----"; // allow-secret-scan
        let openssh = "-----BEGIN OPENSSH PRIVATE KEY-----"; // allow-secret-scan
        let encrypted = "-----BEGIN ENCRYPTED PRIVATE KEY-----"; // allow-secret-scan
        let dsa = "-----BEGIN DSA PRIVATE KEY-----"; // allow-secret-scan  [CP2 MS2 S1b loop 13]
        let asia = "ASIAABCDEFGHIJKLMNOP"; // allow-secret-scan
        let fine_grained = "github_pat_AAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        for (text, name) in [
            (ec, "pem-private-key"),
            (pkcs8, "pem-private-key"),
            (openssh, "pem-private-key"),
            (encrypted, "pem-private-key"),
            (dsa, "pem-private-key"),
            (asia, "aws-access-key"),
            (fine_grained, "github-token"),
        ] {
            assert_eq!(find_secret(text), Some(name), "{text}");
        }
    }

    #[test]
    fn near_misses_at_each_pattern_boundary_are_not_secrets() {
        let short_openai = "sk-AAAAAAAAAAAAAAAAAAA"; // 19 after the prefix
        let short_github = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 35 after the prefix
        let short_aws = "AKIAABCDEFGHIJKLMNO"; // 15 after the prefix
        let short_anthropic = "sk-ant-AAAAAAAAAAAAAAAAAAA"; // allow-secret-scan (19 after sk-ant-: neither name, S1b loop 11)
                                                            // [RED-DRYRUN 2026-10-18, D11 class] rustfmt splits the array and tests/no_hardcoded_secrets.rs flags the two PEM
                                                            // headers ([PEM block]); one-line lets keep the marker on the literal's line.
        let public = "-----BEGIN PUBLIC KEY-----"; // allow-secret-scan
        let certificate = "-----BEGIN CERTIFICATE-----"; // allow-secret-scan
        for text in [
            short_openai,
            short_github,
            short_aws,
            short_anthropic,
            public,
            certificate,
        ] {
            assert_eq!(find_secret(text), None, "{text}");
        }
    }

    #[test]
    fn a_secret_prefix_inside_a_longer_token_is_not_a_secret() {
        // [CP2 MS2 S1b loop 9] left boundary: kebab-case and base64 runs that contain `sk-` or `AKIA` are ordinary text.
        let kebab = "let class = \"task-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";"; // allow-secret-scan
        let blob = "data=QUJDAKIAABCDEFGHIJKLMNOP"; // allow-secret-scan
        let start = "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        assert_eq!(find_secret(kebab), None);
        assert_eq!(find_secret(blob), None);
        assert_eq!(find_secret(start), Some("openai-key")); // a prefix that opens the text still counts
    }

    #[test]
    fn common_provider_keys_and_the_aws_secret_key_are_secrets() {
        // [CP2 MS2 S1a loop 10] prefixes split with concat! so neither tests/no_hardcoded_secrets.rs nor a host's push
        // protection reads a whole key-shaped literal here.
        let cases = [
            (
                concat!("sk_", "live_", "AAAAAAAAAAAAAAAAAAAAAAAA"),
                "stripe-key",
            ),
            (
                concat!("rk_", "live_", "AAAAAAAAAAAAAAAAAAAAAAAA"),
                "stripe-key",
            ),
            (
                concat!("AI", "za", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                "google-api-key",
            ),
            (concat!("xo", "xb-", "1234567890-AAAA"), "slack-token"),
            (
                concat!("gl", "pat-", "AAAAAAAAAAAAAAAAAAAA"),
                "gitlab-token",
            ),
            (
                concat!(
                    "AWS_SECRET",
                    "_ACCESS_KEY = \"",
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "\""
                ),
                "aws-secret-key",
            ),
            (
                concat!(
                    "aws_secret",
                    "_access_key: ",
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                ),
                "aws-secret-key",
            ),
        ];
        for (text, name) in cases {
            assert_eq!(find_secret(text), Some(name), "{text}");
        }
        for text in [
            concat!("sk_", "live_", "AAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("AI", "za", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("gl", "pat-", "AAAAAAAAAAAAAAAAAAA"),
            concat!(
                "aws_secret",
                "_access_key = ",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            ),
        ] {
            assert_eq!(find_secret(text), None, "{text}");
        }
    }

    #[test]
    fn every_token_shaped_prefix_needs_a_left_boundary_and_the_aws_key_name_any_casing() {
        // [CP2 MS2 S1b loop 12] AIza, sk_live_ and the other S1a loop 10 prefixes inside a base64 run are not keys; the AWS
        // secret key is found under its camelCase and kebab-case names. Split with concat! for tests/no_hardcoded_secrets.rs.
        for blob in [
            concat!(
                "data=QUJD",
                "AI",
                "za",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            ),
            concat!("data=QUJD", "sk_", "live_", "AAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("data=QUJD", "rk_", "live_", "AAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("data=QUJD", "xo", "xb-", "1234567890-AAAA"),
            concat!("data=QUJD", "gl", "pat-", "AAAAAAAAAAAAAAAAAAAA"),
        ] {
            assert_eq!(find_secret(blob), None, "{blob}");
        }
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        for name in [
            concat!("aws", "SecretAccessKey"),
            concat!("aws-secret", "-access-key"),
            concat!("Aws", "SecretAccessKey"),
        ] {
            assert_eq!(
                find_secret(&format!("{name} = '{tail}'")),
                Some("aws-secret-key"),
                "{name}"
            );
        }
        let env = format!(
            "process.env.{} = \"{tail}\"",
            concat!("AWS_SECRET", "_ACCESS_KEY")
        );
        assert_eq!(find_secret(&env), Some("aws-secret-key"));
    }

    #[test]
    fn a_token_after_a_diff_line_marker_is_a_secret() {
        // [MAGI §6 MS2 A loop 1] a removed diff line starts with `-`, a word character: without this the left boundary
        // rejected every `sk-`, `ghp_` and `AKIA` token on a deleted line, and the diff text reached a model.
        let removed = [
            concat!("-", "sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("-", "gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("-", "AK", "IA", "ABCDEFGHIJKLMNOP"),
        ];
        for line in removed {
            let hunk = format!("@@ -1,2 +1,1 @@\n context\n{line}\n");
            assert!(find_secret(line).is_some(), "{line}");
            assert!(find_secret(&hunk).is_some(), "{hunk}");
        }
        // Inside a line a `-` is still part of the word: kebab-case runs stay ordinary text.
        let kebab = concat!("x\n", "-task", "-sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        assert_eq!(find_secret(kebab), None);
    }

    #[test]
    fn a_token_after_a_run_of_hyphens_that_opens_a_line_is_a_secret() {
        // [MAGI S6 loop 5, A2-c1] a `--` comment (Haskell, SQL, Lua) before a token becomes
        // `---` on a removed line and `+--` on an added one; the single-marker rule missed both.
        let lines = [
            concat!("---", "gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!("+--", "sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA"),
            concat!(" --", "AK", "IA", "ABCDEFGHIJKLMNOP"),
            concat!("--", "gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        ];
        for line in lines {
            let hunk = format!("@@ -1,2 +1,1 @@\n context\n{line}\n");
            assert!(find_secret(line).is_some(), "{line}");
            assert!(find_secret(&hunk).is_some(), "{hunk}");
        }
        // A run of hyphens glued to a word is still part of that word.
        let glued = concat!(
            "x\n",
            "-task--",
            "gh",
            "p_",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        );
        assert_eq!(find_secret(glued), None);
    }

    #[test]
    fn a_shown_path_escapes_format_characters_that_reorder_or_hide_text() {
        // [MAGI S6 loop 5, A2-m3] a right-to-left override or a zero-width space in a file name
        // could visually reorder a report or run-log line; they are escaped like controls.
        for c in ['\u{202e}', '\u{200b}', '\u{2066}', '\u{feff}', '\u{061c}'] {
            let shown = display_path(&format!("a{c}b.rs"));
            assert!(!shown.contains(c), "{c:?} kept in {shown:?}");
            assert!(shown.contains(&c.escape_unicode().to_string()), "{shown:?}");
        }
        assert_eq!(display_path("año/módulo.rs"), "año/módulo.rs");
    }

    #[test]
    fn a_shown_path_escapes_every_format_and_separator_character() {
        // [MAGI S6 loop 5, review of A2-m3] the whole Cf set plus the line and paragraph
        // separators (Zl, Zp): each can break, reorder or hide a report or run-log line.
        let expected: &[(char, char)] = &[
            ('\u{00ad}', '\u{00ad}'),
            ('\u{0600}', '\u{0605}'),
            ('\u{061c}', '\u{061c}'),
            ('\u{06dd}', '\u{06dd}'),
            ('\u{070f}', '\u{070f}'),
            ('\u{0890}', '\u{0891}'),
            ('\u{08e2}', '\u{08e2}'),
            ('\u{180e}', '\u{180e}'),
            ('\u{200b}', '\u{200f}'),
            ('\u{2028}', '\u{202e}'),
            ('\u{2060}', '\u{2064}'),
            ('\u{2066}', '\u{206f}'),
            ('\u{feff}', '\u{feff}'),
            ('\u{fff9}', '\u{fffb}'),
            ('\u{110bd}', '\u{110bd}'),
            ('\u{110cd}', '\u{110cd}'),
            ('\u{13430}', '\u{1343f}'),
            ('\u{1bca0}', '\u{1bca3}'),
            ('\u{1d173}', '\u{1d17a}'),
            ('\u{e0001}', '\u{e0001}'),
            ('\u{e0020}', '\u{e007f}'),
        ];
        for &(first, last) in expected {
            for c in (first..=last).filter(|c| !c.is_control()) {
                let shown = display_path(&format!("a{c}b"));
                assert!(
                    !shown.contains(c),
                    "{} kept in {shown:?}",
                    c.escape_unicode()
                );
            }
        }
    }

    #[test]
    fn a_shown_path_escapes_control_characters() {
        // [CP2 MS2 S1b1 loop 16] a file name can carry a terminal escape; the report never replays it.
        assert_eq!(display_path("a\u{1b}[31m.rs"), "a\\u{1b}[31m.rs");
        assert_eq!(display_path("src/a.rs"), "src/a.rs");
    }

    #[test]
    fn mask_secrets_replaces_only_the_secret_span_and_keeps_the_rest() {
        // [MAGI S6 loop 3, S1] the outward scrubber masks the span, not the whole text.
        let line = "token sk-abcdefghijklmnopqrstuvwxyz0123 here"; // allow-secret-scan
        assert_eq!(mask_secrets(line), "token <redacted: openai-key> here");
        let two = "a=AKIAABCDEFGHIJKLMNOP b=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA!"; // allow-secret-scan
        let want = "a=<redacted: aws-access-key> b=<redacted: github-token>!";
        assert_eq!(mask_secrets(two), want);
        assert_eq!(mask_secrets("let x = 1;"), "let x = 1;");
    }

    #[test]
    fn mask_secrets_hides_a_whole_pem_block_and_an_aws_secret_assignment() {
        // [MAGI S6 loop 3, S1] the PEM body matches no pattern, so the block is masked to its END line.
        let end = concat!("-----", "END RSA PRIVATE KEY-----");
        let pem = format!("x {PEM_BEGIN}RSA {PEM_PRIVATE_KEY}\nMIIBOgIBAAJBAK\n{end} y");
        assert_eq!(mask_secrets(&pem), "x <redacted: pem-private-key> y");
        let open = format!("x {PGP_PRIVATE_KEY}\nbody without end");
        assert_eq!(mask_secrets(&open), "x <redacted: pgp-private-key>");
        let aws = "aws_secret_access_key = AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA tail"; // allow-secret-scan
        assert_eq!(mask_secrets(aws), "<redacted: aws-secret-key> tail");
    }

    #[test]
    fn outward_text_masks_secrets_and_escapes_control_characters() {
        let raw = "fatal: \u{1b}[31mbad sk-abcdefghijklmnopqrstuvwxyz0123\n"; // allow-secret-scan
        let out = outward_text(raw);
        assert_eq!(out, "fatal: \\u{1b}[31mbad <redacted: openai-key>\\n");
        assert_eq!(find_secret(&out), None);
    }

    #[test]
    fn a_pem_private_key_of_any_single_type_word_is_a_secret() {
        // [MAGI S6 MS2 loop 4, A2 Melchior] the type word is not a closed list: `ECDSA` and
        // `ED25519` keys were kept and sent. Public keys and certificates stay ordinary text.
        for kind in ["ECDSA ", "ED25519 ", "X448 ", ""] {
            let header = format!("{PEM_BEGIN}{kind}{PEM_PRIVATE_KEY}");
            assert_eq!(find_secret(&header), Some("pem-private-key"), "{header}");
        }
        for kind in [
            "ECDSA PUBLIC KEY-----",
            "CERTIFICATE-----",
            "ECDSA  PRIVATE KEY-----",
        ] {
            let header = format!("{PEM_BEGIN}{kind}");
            assert_eq!(find_secret(&header), None, "{header}");
        }
    }

    #[test]
    fn a_distinctive_prefix_glued_to_a_word_is_still_a_secret() {
        // [MAGI S6 MS2 loop 4, A2 Caspar] `sk-ant-` and `github_pat_` cannot occur by chance in a
        // base64 run or a kebab-case name, so they need no left boundary: glued to an identifier
        // they were sent. The short prefixes keep theirs (a_secret_prefix_inside_a_longer_token).
        let glued = [
            (
                concat!("abc", "sk-", "ant-", "api03-AAAAAAAAAAAAAAAAAAAA"),
                "anthropic-key",
            ),
            (
                concat!("KEY_", "sk-", "ant-", "api03-AAAAAAAAAAAAAAAAAAAA"),
                "anthropic-key",
            ),
            (
                concat!("x_", "github_", "pat_", "AAAAAAAAAAAAAAAAAAAAAA"),
                "github-token",
            ),
        ];
        for (text, name) in glued {
            assert_eq!(find_secret(text), Some(name), "{text}");
            assert!(
                !mask_secrets(text).contains("AAAAAAAAAAAAAAAAAAAA"),
                "{text}"
            );
        }
    }

    /// A failure deadline far above a linear pass (milliseconds) and below the quadratic one.
    const MASK_DEADLINE_SECS: u64 = 10;

    #[test]
    fn mask_secrets_masks_many_secrets_in_one_pass_and_one_right_after_a_masked_block() {
        // [MAGI S6 MS2 loop 4, A2 Caspar/Balthasar] each turn rescanned the rest for every
        // pattern: with a pattern whose only match is at the end, 4 000 secrets took ~28 s in a
        // debug build. Masking keeps a next span per pattern and must mask what the rescanning
        // loop masked, a token that opens right where a masked block ends included.
        let key = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let aws = concat!("AK", "IA", "ABCDEFGHIJKLMNOP");
        let mut text = String::new();
        let mut want = String::new();
        for i in 0..2_000 {
            text.push_str(&format!("x{i} {key} y {aws};\n"));
            want.push_str(&format!(
                "x{i} <redacted: github-token> y <redacted: aws-access-key>;\n"
            ));
        }
        text.push_str(concat!("z sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA"));
        want.push_str("z <redacted: openai-key>");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(mask_secrets(&text));
        });
        let masked = rx
            .recv_timeout(std::time::Duration::from_secs(MASK_DEADLINE_SECS))
            .expect("masking 4 000 secrets is linear, not quadratic");
        assert_eq!(masked, want);
        let end = concat!("-----", "END RSA PRIVATE KEY-----");
        let pem = format!("{PEM_BEGIN}RSA {PEM_PRIVATE_KEY}\nMIIB\n{end}{key} tail");
        assert_eq!(
            mask_secrets(&pem),
            "<redacted: pem-private-key><redacted: github-token> tail"
        );
    }
    /// The masking loop before S6 loop 4: each turn rescans what follows the last mask for every
    /// pattern. O(k · n · p), kept as the reference the one-pass masking must equal.
    fn mask_secrets_by_rescanning(text: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        while let Some((span, name)) = SECRET_PATTERNS
            .iter()
            .filter_map(|p| p.matcher.first_span(rest).map(|r| (r, p.name)))
            .min_by_key(|(r, _)| r.start)
        {
            out.push_str(rest.get(..span.start).unwrap_or_default());
            out.push_str(REDACTED_OPEN);
            out.push_str(name);
            out.push('>');
            rest = rest.get(span.end..).unwrap_or_default();
        }
        out.push_str(rest);
        out
    }

    #[test]
    fn one_pass_masking_equals_the_rescanning_loop_on_adjacent_and_fuzzed_secrets() {
        // [MAGI S6 MS2 loop 4, review of the one-pass masking, security] a token that opens at
        // the end of a mask, or one byte after it behind a `-` (read as a diff marker by the
        // rescanning loop), was left in the clear.
        let sk = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let ant = concat!("sk-", "ant-", "api03-AAAAAAAAAAAAAAAAAAAA");
        let ghp = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let aws = concat!("AK", "IA", "ABCDEFGHIJKLMNOP");
        let aws_secret = concat!(
            "aws_secret",
            "_access_key=",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        );
        let pem = format!(
            "{PEM_BEGIN}EC {PEM_PRIVATE_KEY}\nMIIB\n{}",
            concat!("-----", "END EC PRIVATE KEY-----")
        );
        let fragments: Vec<&str> = vec![
            sk, ant, ghp, aws, aws_secret, &pem, "-", "\n", "_", "a", " ", "x-", "-\n", ".",
        ];
        let adjacent = [
            format!("{aws}-{sk}"),
            format!("{sk}-{sk}"),
            format!("{pem}{ghp}"),
            format!("{pem}-{sk}"),
            format!("{aws}{ghp}"),
            format!("{aws_secret}-{aws}"),
            format!("x{ant}-{ghp}"),
        ];
        for text in &adjacent {
            assert_eq!(
                mask_secrets(text),
                mask_secrets_by_rescanning(text),
                "{text}"
            );
        }
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..3_000 {
            let mut text = String::new();
            for _ in 0..8 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let pick = usize::try_from(state >> 33).unwrap_or(0) % fragments.len();
                text.push_str(fragments.get(pick).copied().unwrap_or_default());
            }
            assert_eq!(
                mask_secrets(&text),
                mask_secrets_by_rescanning(&text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn the_aws_secret_key_is_found_under_its_common_aliases() {
        // [MAGI S6 loop 7, A2-c3] `AWS_SECRET_KEY` and the STS `SecretAccessKey` field name the
        // same key and were sent. A name with neither `aws` nor `access` stays ordinary text.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        for name in [
            concat!("AWS_SECRET", "_KEY"),
            concat!("aws-secret", "-key"),
            concat!("Secret", "AccessKey"),
            concat!("secret_", "access_key"),
        ] {
            assert_eq!(
                find_secret(&format!("{name}={tail}")),
                Some("aws-secret-key"),
                "{name}"
            );
        }
        assert_eq!(
            find_secret(&format!("{}={tail}", concat!("secret", "_key"))),
            None
        );
    }

    #[test]
    fn an_aws_secret_key_under_a_quoted_name_is_a_secret_and_masked_whole() {
        // [MAGI S6 loop 7, A2-b1] the closing quote of a JSON property, and a Ruby or PHP `=>`,
        // stood between the name and its separator, so the key was sent.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let name = concat!("aws_secret", "_access_key");
        for text in [
            format!("{{\"{name}\": \"{tail}\"}}"),
            format!("{{\"{name}\":\"{tail}\"}}"),
            format!("'{name}' => '{tail}'"),
            format!("{name} => \"{tail}\""),
        ] {
            assert_eq!(find_secret(&text), Some("aws-secret-key"), "{text}");
            assert!(!mask_secrets(&text).contains(tail), "{text}");
        }
    }

    #[test]
    fn a_pem_private_key_header_of_several_type_words_is_a_secret() {
        // [MAGI S6 loop 7, A2-b4] `SSH2 ENCRYPTED` and `RSA ENCRYPTED` headers name a private key
        // with two type words, and the SSH2 (RFC 4716 style) armor uses four dashes and spaces:
        // all were sent. Public keys stay ordinary text.
        for kind in ["SSH2 ENCRYPTED ", "RSA ENCRYPTED ", "A B C D "] {
            let header = format!("{PEM_BEGIN}{kind}{PEM_PRIVATE_KEY}");
            assert_eq!(find_secret(&header), Some("pem-private-key"), "{header}");
        }
        let ssh2 = concat!("---- BEGIN SSH2 ", "ENCRYPTED PRIVATE KEY ----");
        assert_eq!(find_secret(ssh2), Some("pem-private-key"));
        for kind in ["SSH2 PUBLIC KEY-----", "A B C D E PRIVATE KEY-----"] {
            let header = format!("{PEM_BEGIN}{kind}");
            assert_eq!(find_secret(&header), None, "{header}");
        }
        assert_eq!(
            find_secret(concat!("---- BEGIN SSH2 ", "PUBLIC KEY ----")),
            None
        );
    }

    #[test]
    fn a_shown_path_escapes_every_code_point_of_the_tag_block() {
        // [MAGI S6 loop 7, A2-c4] the table said the tag block's unassigned code points were
        // included, but U+E0000 and U+E0002..U+E001F were left out and reached the output as is.
        for c in ('\u{e0000}'..='\u{e007f}').filter(|c| !c.is_control()) {
            let shown = display_path(&format!("a{c}b"));
            assert!(!shown.contains(c), "{} kept", c.escape_unicode());
        }
    }
}
