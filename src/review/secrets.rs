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
//! false positives on ordinary code.

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
/// The words of the AWS secret access key's name, in order.
const AWS_SECRET_WORDS: &[&str] = &["aws", "secret", "access", "key"];
/// The minimum length of the AWS secret access key.
const AWS_SECRET_MIN: usize = 40;
/// The blanks allowed around the `=` or `:` of an AWS secret key assignment.
const BLANKS: &[u8] = b" \t";
/// The quotes allowed before an AWS secret key.
const QUOTES: &[u8] = b"\"'";

/// The patterns, in the order that decides the name of a match: `anthropic-key` comes before
/// `openai-key` because an Anthropic key also starts with `sk-`.
pub(crate) const SECRET_PATTERNS: &[SecretPattern] = &[
    SecretPattern {
        name: "anthropic-key",
        matcher: Matcher::Token {
            prefixes: &["sk-ant-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "openai-key",
        matcher: Matcher::Token {
            prefixes: &["sk-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: Some("ant-"),
        },
    },
    SecretPattern {
        name: "github-token",
        matcher: Matcher::Token {
            prefixes: &["ghp_", "gho_", "ghu_", "ghs_", "ghr_"],
            tail: Tail::Word,
            min: 36,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "github-token",
        matcher: Matcher::Token {
            prefixes: &["github_pat_"],
            tail: Tail::Word,
            min: 22,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "aws-access-key",
        matcher: Matcher::Token {
            prefixes: &["AKIA", "ASIA"],
            tail: Tail::UpperDigit,
            min: 16,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "pem-private-key",
        matcher: Matcher::PemPrivateKey,
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
        },
    },
    SecretPattern {
        name: "google-api-key",
        matcher: Matcher::Token {
            prefixes: &["AIza"],
            tail: Tail::Word,
            min: 35,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "slack-token",
        matcher: Matcher::Token {
            prefixes: &["xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"],
            tail: Tail::Word,
            min: 10,
            not_followed_by: None,
        },
    },
    SecretPattern {
        name: "gitlab-token",
        matcher: Matcher::Token {
            prefixes: &["glpat-"],
            tail: Tail::Word,
            min: 20,
            not_followed_by: None,
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

/// Whether the byte before `at` is a removed-line marker: a `-` that opens a line. A diff marks
/// a removed line with `-`, which [`is_word`] counts as part of a token, so without this every
/// token on a removed line would fail its left boundary.
fn after_diff_marker(bytes: &[u8], at: usize) -> bool {
    let Some(marker) = at.checked_sub(1) else {
        return false;
    };
    bytes.get(marker) == Some(&b'-')
        && marker
            .checked_sub(1)
            .is_none_or(|before| bytes.get(before) == Some(&b'\n'))
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
    /// Linear in the text: an occurrence of a token prefix inside another token fails its left
    /// boundary before its tail is read, and a tail is read for at most its minimum length.
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
            Self::Token {
                prefixes,
                tail,
                min,
                not_followed_by,
            } => prefixes
                .iter()
                .filter_map(|prefix| {
                    text.match_indices(prefix).find_map(|(at, _)| {
                        let bounded = at
                            .checked_sub(1)
                            .and_then(|before| bytes.get(before))
                            .is_none_or(|b| !is_word(*b))
                            || after_diff_marker(bytes, at);
                        let end = at + prefix.len();
                        let excluded = not_followed_by.is_some_and(|n| word_at(bytes, end, n));
                        (bounded && !excluded && tail_at_least(bytes, end, *tail, *min))
                            .then(|| at..tail_end(bytes, end, *tail))
                    })
                })
                .min_by_key(|r| r.start),
            Self::PemPrivateKey => text.match_indices(PEM_BEGIN).find_map(|(at, _)| {
                let mut rest = at + PEM_BEGIN.len();
                // MAGI FIX (S6 loop 4): any one type word, not a closed list (`ECDSA` was sent).
                if !word_at(bytes, rest, PEM_PRIVATE_KEY) {
                    rest += pem_type_word(bytes, rest).map_or(0, |word| word + 1);
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

/// Where the AWS secret access key that opens at `at` ends: its name (the four words in any case
/// with at most one `_` or `-` between them), blanks, `=` or `:`, blanks, an optional quote, and
/// a key of at least 40 characters. `None` when none opens there.
fn aws_secret_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut pos = at;
    for (i, word) in AWS_SECRET_WORDS.iter().enumerate() {
        if i > 0 && bytes.get(pos).is_some_and(|b| *b == b'_' || *b == b'-') {
            pos += 1;
        }
        if !word_at(bytes, pos, word) {
            return None;
        }
        pos += word.len();
    }
    pos = skip_bytes(bytes, pos, BLANKS);
    if !bytes.get(pos).is_some_and(|b| *b == b'=' || *b == b':') {
        return None;
    }
    pos = skip_bytes(bytes, pos + 1, BLANKS);
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
/// # Arguments
/// * `text` - Any text about to leave the process.
///
/// # Returns
/// The masked text; `text` unchanged when it holds no secret.
pub fn mask_secrets(text: &str) -> String {
    if find_secret(text).is_none() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    // Each turn masks the earliest secret of the rest (ties go to table order), so a text with
    // k secrets is scanned k + 1 times: O((k + 1) · n · p).
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

/// `text` as it may leave the process: [`mask_secrets`], then every control character escaped.
///
/// # Arguments
/// * `text` - Any foreign text (git stderr, a path, a ref, a model message).
///
/// # Returns
/// The masked, escaped text.
pub fn outward_text(text: &str) -> String {
    escape_controls(&mask_secrets(text))
}

/// `text` with every control character escaped ([`char::escape_default`]).
fn escape_controls(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_control() {
                c.escape_default().to_string()
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
}
