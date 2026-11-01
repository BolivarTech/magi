// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The product's secret detector: what a review never sends out of the process.
//!
//! Every byte a review sends to a model passes through [`find_secret`] first. The patterns are
//! matched without dependencies (a prefix, a character class and a minimum length), in time
//! linear in the text. Not covered, by design: JWTs (any base64url JSON looks like one), Azure
//! keys, high-entropy strings with no prefix, the base64 body of a PEM or armored OpenPGP
//! private key without its `BEGIN` line, and an AWS secret key without its name: the key is found
//! after its name through an assignment or a comparison, or quoted next to it as an argument or
//! a collection item (`f(aws_secret_key, "…")`), never as a bare 40-character value.
//! The minimum lengths can miss a shorter real key; that is the trade-off with false positives on
//! ordinary code. So are the left boundaries: a short prefix (`sk-`, `ghp_`, `AKIA`, …) glued to a
//! word character is not a token, because base64 runs and kebab-case names hold them by chance; the distinctive
//! `sk-ant-` and `github_pat_` match wherever they occur. A token prefix and its exclusion match
//! in any ASCII case, so `SK-ANT-…` is a key, and so is an upper-case name that opens like one
//! (`GITHUB_PAT_…`) with a long enough tail: the safe direction. A PEM or OpenPGP header must be in
//! upper case as every standard tool writes it: a hand-made `-----begin private key-----` is
//! not detected. A password in a URL (`scheme://user:password@host`) is a secret unless it is a
//! placeholder (`{password}`, `$PASSWORD`, `<password>`, `***`); a URL with a user and no
//! password is not.

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
        /// The prefixes, any of which opens the token, in any ASCII case.
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
    /// The AWS secret access key, recognized by its name and an assignment, a comparison, or a
    /// quoted key passed next to it.
    AwsSecretKey,
    /// A password in a URL's authority: `scheme://user:password@host`.
    UrlCredential,
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
/// The header of an encrypted SSH2 private key (RFC 4716 armor: four dashes and spaces; split
/// so the source holds no such header). Its `END` line has the same armor ([`SSH2_ARMOR_END`]);
/// with none, its mask runs to the end of the text: the safe direction.
const SSH2_PRIVATE_KEY: &str = concat!("---- BEGIN SSH2 ", "ENCRYPTED PRIVATE KEY ----");
/// The header of an unencrypted SSH2 private key (the ssh.com/Tectia armor of
/// [`SSH2_PRIVATE_KEY`] without `ENCRYPTED`), masked the same way.
// MAGI FIX (S6 loop 12, A2-m1): the unencrypted armor matched no pattern and its file was sent.
const SSH2_PLAIN_PRIVATE_KEY: &str = concat!("---- BEGIN SSH2 ", "PRIVATE KEY ----");
/// The end of a PEM private-key header.
const PEM_PRIVATE_KEY: &str = "PRIVATE KEY-----";
/// The header of an armored OpenPGP private key (split so the source holds no such header).
const PGP_PRIVATE_KEY: &str = concat!("-----", "BEGIN PGP PRIVATE KEY BLOCK-----");
/// The opening of the `END` line of an armored block (split so the source holds no PEM line).
const ARMOR_END: &str = concat!("-----", "END ");
/// The dashes that close an armor line.
const ARMOR_DASHES: &str = "-----";
/// The opening of the `END` line of an SSH2 (RFC 4716) block: four dashes and a space.
const SSH2_ARMOR_END: &str = "---- END ";
/// The dashes that close an SSH2 armor line.
const SSH2_ARMOR_DASHES: &str = "----";
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
/// The operators that may stand between an AWS secret key's name and the key, longest first so
/// a prefix (`=` of `==`) never wins over the whole operator: assignments (`=`, `:`, `:=`, `=>`)
/// and comparisons (`==`, `!=`, JavaScript's `===` and `!==`).
// MAGI FIX (S6 loop 11): a key compared against its name (`==`, `!=`) was sent.
const AWS_OPERATORS: &[&[u8]] = &[
    b"===",
    b"!==",
    b"==",
    b"!=",
    b"=>",
    b":=",
    b"=",
    COLON_OPERATOR,
];
/// The operator of a YAML mapping (and a JSON property): the one whose value may start on the
/// next line with no other mark ([`skip_continuation`]).
const COLON_OPERATOR: &[u8] = b":";
/// The chomping (`-`, `+`) and indentation (`1`-`9`) indicators a YAML block scalar header may
/// carry after its `|` or `>`.
const BLOCK_SCALAR_MODIFIERS: &[u8] = b"+-123456789";
/// The most indicators a YAML block scalar header carries: one chomping and one indentation.
const MAX_BLOCK_SCALAR_MODIFIERS: usize = 2;
/// The bytes that may stand between an AWS secret key's name and a quoted key passed next to it
/// as an argument or a collection item: blanks and line breaks, quotes, the separators of an
/// argument list or a collection, and the bytes of [`AWS_OPERATORS`]. `<` is left out, so a
/// `<=` stays no assignment.
// MAGI FIX (S6 loop 15, A2-1, security): `configure(aws_secret_access_key, "…")` was sent.
const AWS_ARGUMENT_SEPARATORS: &[u8] = b" \t\r\n\"'`,()[]{}:=>!";
/// The quotes that may open a key passed next to its name: a string literal of most languages.
const AWS_ARGUMENT_QUOTES: &[u8] = b"\"'`";
/// The most separator bytes read after an AWS secret key's name before a quoted key, so each
/// name reads a bounded run.
const AWS_ARGUMENT_WINDOW: usize = 16;
/// What separates a URL's scheme from its authority.
const URL_AUTHORITY_START: &str = "://";
/// The longest userinfo (`user:password`) a URL credential may have, so each `://` reads a
/// bounded run and the search stays linear.
const MAX_URL_USERINFO: usize = 256;
/// How far past [`MAX_URL_USERINFO`] the search for the authority's last `@` reads, so a host
/// after a long userinfo is still seen and the read stays bounded.
const MAX_URL_HOST_PEEK: usize = 256;
/// The bytes that end a URL's userinfo without an `@`: blanks, the authority's other
/// delimiters, and the quotes and brackets RFC 3986 does not allow unencoded in a userinfo
/// (`"`, `` ` ``, `<`, `>`). The sub-delimiters `'`, `(` and `)` are allowed there, so they stay
/// inside the run: a URL in prose that a `)` or `'` closes is still ended by the blank or `/`
/// that follows it.
// MAGI FIX (S6 loop 14, A2-1, security): `'`, `(` and `)` ended the userinfo, so a password
// holding one (`pa(ss`) found no `@` and its file was sent.
const URL_USERINFO_STOP: &[u8] = b" \t\r\n/?#\"`<>";
/// The bytes a placeholder's name may hold (besides ASCII letters and digits): `db_password`,
/// `.Values.password`, `db-pass`.
const URL_PLACEHOLDER_NAME_BYTES: &[u8] = b"_.-";

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
        name: "pem-private-key",
        matcher: Matcher::Header(SSH2_PLAIN_PRIVATE_KEY),
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
    // MAGI FIX (S6 loop 13, A2-c2): a database URL's password in a reviewed file was sent.
    SecretPattern {
        name: "url-credential",
        matcher: Matcher::UrlCredential,
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
                .map(|at| at..header_block_end(text, at + header.len(), header)),
            Self::AwsSecretKey => {
                (0..bytes.len()).find_map(|at| aws_secret_end(bytes, at).map(|end| at..end))
            }
            Self::UrlCredential => text.match_indices(URL_AUTHORITY_START).find_map(|(at, _)| {
                let from = at + URL_AUTHORITY_START.len();
                url_userinfo_end(bytes, from).map(|end| from..end)
            }),
        }
    }
}

/// Where the userinfo of a URL credential that opens at `from` (just past `://`) ends: past the
/// last `@` within [`MAX_URL_USERINFO`] bytes that ends a credential (a password may hold an
/// unencoded `@`): one with a `:` before it, a password between them that is not a placeholder
/// ([`is_url_placeholder`]), and a host after it (an ASCII letter or digit, or `[` for an IPv6
/// literal). Every such `@` is tried, so a later `@` that ends nothing never hides an earlier
/// credential. The user may be empty (`redis://:password@host`). The authority ends at a
/// [`URL_USERINFO_STOP`] byte or a control character; `None` when no `@` ends a credential.
/// O([`MAX_URL_USERINFO`]²), a constant, per `://`: one placeholder check per `@`.
fn url_userinfo_end(bytes: &[u8], from: usize) -> Option<usize> {
    let rest = bytes.get(from..).unwrap_or_default();
    let window = rest
        .get(..rest.len().min(MAX_URL_USERINFO + 1 + MAX_URL_HOST_PEEK))
        .unwrap_or_default();
    let stop = window
        .iter()
        .position(|b| URL_USERINFO_STOP.contains(b) || b.is_ascii_control())
        .unwrap_or(window.len());
    let run = window.get(..stop)?;
    let colon = run.iter().position(|b| *b == b':')?;
    let last = run.len().saturating_sub(1).min(MAX_URL_USERINFO);
    // The last `@` that ends a credential, as `redact_url` reads the authority's last `@`: a
    // password may hold an unencoded `@`, so the mask runs to the furthest one that qualifies.
    // MAGI FIX (S6 loop 14, A2-1, security review): every `@` within the cap is tried, not the
    // run's last one alone: a later `@` with no host after it (`f('…u:pw@db','@')`) or past the
    // cap (a minified line) hid the credential before it.
    (colon + 1..=last).rev().find_map(|at_sign| {
        if run.get(at_sign) != Some(&b'@') {
            return None;
        }
        let password = run.get(colon + 1..at_sign)?;
        // MAGI FIX (S6 loop 14, A2-1, security): the whole password must be a placeholder
        // form; its first byte alone let `$ecret` or `{actual` through as a placeholder.
        let real = !password.is_empty() && !is_url_placeholder(password);
        let host = run
            .get(at_sign + 1)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'[');
        (real && host).then_some(from + at_sign + 1)
    })
}

/// Whether a URL password is, as a whole, a placeholder rather than a credential: a template
/// slot (`{}`, `{0}`, `{password}`, `{{.Values.password}}`), a variable (`${db.password}`,
/// `$PASSWORD`, upper case only, as an environment variable is written), a documentation slot
/// (`[password]`), a format directive (`%s`, one letter) or a mask already applied (`*`, `***`).
/// A name holds ASCII letters, digits and [`URL_PLACEHOLDER_NAME_BYTES`]. Anything else, even
/// one that opens like a placeholder (`$ecret`, `{actual`, `*pw`, `%41bc`), is a credential.
/// `<password>` needs no form: `<` ends the userinfo ([`URL_USERINFO_STOP`]) before its `@`.
/// Limit: a real password that is itself exactly one of these forms (`{actual}`) reads as a
/// placeholder; the forms are what templates and documentation write. O(password length).
fn is_url_placeholder(password: &[u8]) -> bool {
    let name = |b: &[u8]| {
        b.iter()
            .all(|c| c.is_ascii_alphanumeric() || URL_PLACEHOLDER_NAME_BYTES.contains(c))
    };
    let inside = |open: &[u8], close: &[u8]| {
        password
            .strip_prefix(open)
            .and_then(|rest| rest.strip_suffix(close))
    };
    let env_name = |b: &[u8]| {
        b.first()
            .is_some_and(|c| c.is_ascii_uppercase() || *c == b'_')
            && b.iter()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_')
    };
    if let Some(body) = inside(b"${", b"}") {
        return !body.is_empty() && name(body);
    }
    if let Some(body) = inside(b"{{", b"}}").or_else(|| inside(b"{", b"}")) {
        return name(body);
    }
    if let Some(body) = inside(b"[", b"]") {
        return !body.is_empty() && name(body);
    }
    match password {
        [b'$', rest @ ..] => env_name(rest),
        [b'%', letter] => letter.is_ascii_alphabetic(),
        stars => stars.iter().all(|c| *c == b'*'),
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
        // MAGI FIX (S6 loop 15, A2-2, security): the prefix was found case-sensitively while the
        // `ant-` exclusion was not, so `sk-ANT-…` matched neither key and its file was sent.
        prefix_indices(bytes, prefix).find_map(|at| {
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
        let opens = word_at(bytes, at, prefix);
        let excluded = not_followed_by.is_some_and(|n| word_at(bytes, end, n));
        (opens && !excluded && tail_at_least(bytes, end, *tail, *min))
            .then(|| at..tail_end(bytes, end, *tail))
    }
}

/// The indices at which `prefix` occurs in `bytes`, ignoring ASCII case as [`word_at`] does, so
/// a prefix and its exclusion ([`Matcher::Token`]'s `not_followed_by`) read the same case.
/// Every prefix is ASCII, so each index is a character boundary. O(n · prefix length).
fn prefix_indices<'a>(bytes: &'a [u8], prefix: &'a str) -> impl Iterator<Item = usize> + 'a {
    (0..bytes.len()).filter(move |at| word_at(bytes, *at, prefix))
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
    armor_end_with(text, from, ARMOR_END, ARMOR_DASHES)
}

/// The end of the block a [`Matcher::Header`] opens, its header ending at `from`: an OpenPGP
/// block (five dashes) ends as [`armor_end`] reads it, an SSH2 block (four dashes and a space,
/// RFC 4716) at its own `---- END … ----` line. O(n).
// MAGI FIX (S6 loop 14, A2-3): an SSH2 block was searched for a five-dash end it never has, so
// its mask ran to the end of the text and the content after the key was lost.
fn header_block_end(text: &str, from: usize, header: &str) -> usize {
    if header.starts_with(ARMOR_DASHES) {
        armor_end(text, from)
    } else {
        armor_end_with(text, from, SSH2_ARMOR_END, SSH2_ARMOR_DASHES)
    }
}

/// The end of an armored block whose header ends at `from`: past the `close` dashes that end
/// its line opening with `end`, or the end of the text when there is no such line. O(n).
fn armor_end_with(text: &str, from: usize, end: &str, close: &str) -> usize {
    text.get(from..)
        .and_then(|rest| {
            let end_at = rest.find(end)? + end.len();
            let close_at = rest.get(end_at..)?.find(close)? + close.len();
            Some(from + end_at + close_at)
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

/// Where the AWS secret access key that opens at `at` ends: its name ([`aws_name_end`]), then
/// the key after an assignment or a comparison ([`aws_assignment_end`]) or quoted next to it
/// ([`aws_argument_end`]). `None` when none opens there. O(1) when no name opens at `at`.
fn aws_secret_end(bytes: &[u8], at: usize) -> Option<usize> {
    let name_end = aws_name_end(bytes, at)?;
    aws_assignment_end(bytes, name_end).or_else(|| aws_argument_end(bytes, name_end))
}

/// Where the key ends after an AWS secret key's name that ends at `name_end`: an optional quote
/// that closes the name (a JSON property, a quoted hash key), blanks, an operator of
/// [`AWS_OPERATORS`] (an assignment or a comparison), blanks, a continuation to the next line if
/// any ([`skip_continuation`]), an optional quote, and a key of at
/// least 40 characters. `None` when there is none.
// MAGI FIX (S6 loop 7): `"aws_secret_access_key": "…"` and `=>` were not assignments.
fn aws_assignment_end(bytes: &[u8], name_end: usize) -> Option<usize> {
    let mut pos = name_end;
    if bytes.get(pos).is_some_and(|b| QUOTES.contains(b)) {
        pos += 1;
    }
    pos = skip_bytes(bytes, pos, BLANKS);
    // MAGI FIX (S6 loop 9): Go's and Make's `:=` left its `=` before the key, which was sent.
    // MAGI FIX (S6 loop 11): `==` and `!=` were not accepted; the operator is now one of a table.
    let operator = AWS_OPERATORS
        .iter()
        .find(|op| bytes.get(pos..).is_some_and(|rest| rest.starts_with(op)))?;
    pos += operator.len();
    pos = skip_bytes(bytes, pos, BLANKS);
    // MAGI FIX (S6 loop 16, A2-3, security): a YAML block scalar, a YAML value on the next line
    // and a shell line continued with `\` put a line break before the key, which was sent.
    pos = skip_continuation(bytes, pos, operator == &COLON_OPERATOR)?;
    if bytes.get(pos).is_some_and(|b| QUOTES.contains(b)) {
        pos += 1;
    }
    tail_at_least(bytes, pos, Tail::Base64, AWS_SECRET_MIN)
        .then(|| tail_end(bytes, pos, Tail::Base64))
}

/// Where the value starts after the line an assignment is continued from, at `at` (just past
/// the operator and its blanks): unchanged when no continuation opens there; past a YAML block
/// scalar header (`|` or `>`, at most [`MAX_BLOCK_SCALAR_MODIFIERS`] chomping or indentation
/// indicators, blanks), a shell `\`, or nothing at all after a `:` (a YAML value on the next
/// line), then one line break and the next line's indentation. A `=` with a bare line break
/// after it continues nothing, so code that breaks a line after `=` is not read as a key.
///
/// # Arguments
///
/// * `bytes` - The text.
/// * `at` - Just past the operator and the blanks after it.
/// * `after_colon` - Whether the operator was `:` ([`COLON_OPERATOR`]).
///
/// # Returns
///
/// The position of the value, or `None` when a header or a `\` opens with no line break after
/// it (no key follows on this line, and none on the next). O(blanks).
fn skip_continuation(bytes: &[u8], at: usize, after_colon: bool) -> Option<usize> {
    let header = match bytes.get(at) {
        Some(b'|' | b'>') => {
            let modifiers = bytes
                .get(at + 1..)
                .unwrap_or_default()
                .iter()
                .take(MAX_BLOCK_SCALAR_MODIFIERS)
                .take_while(|b| BLOCK_SCALAR_MODIFIERS.contains(b))
                .count();
            Some(skip_bytes(bytes, at + 1 + modifiers, BLANKS))
        }
        Some(b'\\') => Some(at + 1),
        _ => None,
    };
    let from = header.unwrap_or(at);
    let after_break = [&b"\r\n"[..], b"\n"]
        .iter()
        .find(|nl| bytes.get(from..).is_some_and(|rest| rest.starts_with(nl)))
        .map(|nl| from + nl.len());
    match (header, after_break) {
        (_, Some(next)) if header.is_some() || after_colon => Some(skip_bytes(bytes, next, BLANKS)),
        (Some(_), _) => None,
        (None, _) => Some(at),
    }
}

/// Where a quoted key passed next to an AWS secret key's name that ends at `name_end` ends: at
/// most [`AWS_ARGUMENT_WINDOW`] bytes of [`AWS_ARGUMENT_SEPARATORS`], the last of them a quote of
/// [`AWS_ARGUMENT_QUOTES`], then a key of at least 40 characters. The quote keeps an unquoted
/// identifier next to the name (`f(aws_secret_key, someLongVariableName)`) from being taken for
/// the key. `None` when there is none. O([`AWS_ARGUMENT_WINDOW`] + key length).
// MAGI FIX (S6 loop 15, A2-1, security): a key passed as an argument or a collection item was
// sent; only an assignment or a comparison after the name was read.
fn aws_argument_end(bytes: &[u8], name_end: usize) -> Option<usize> {
    let run = bytes
        .get(name_end..)
        .unwrap_or_default()
        .iter()
        .take(AWS_ARGUMENT_WINDOW)
        .take_while(|b| AWS_ARGUMENT_SEPARATORS.contains(b))
        .count();
    let pos = name_end + run;
    let quoted = run
        .checked_sub(1)
        .and_then(|last| bytes.get(name_end + last))
        .is_some_and(|b| AWS_ARGUMENT_QUOTES.contains(b));
    (quoted && tail_at_least(bytes, pos, Tail::Base64, AWS_SECRET_MIN))
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
/// grown over every secret that overlaps it (`merged_end`), and the end of a mask counts as a
/// left boundary.
///
/// # Arguments
/// * `text` - Any text about to leave the process.
///
/// # Returns
/// The masked text; `text` unchanged when it holds no secret.
///
/// O(n · s · (k + 1)) for a text of n bytes, s searches (one per token prefix, one per other
/// pattern) and k masks: each search keeps its next span and runs again, from the mask's end,
/// only when a mask passed its start, so a search whose spans are masked one by one rescans what
/// follows each of them; the others check only whether a token opens at the mask's end. The
/// texts it masks are small (paths, one-line errors, report fields), where k stays small.
// MAGI FIX (S6 loop 13, A2-b6): the bound omitted the k factor of a search rerun after each mask.
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
        done = merged_end(span.end, next.iter().flatten());
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

/// The end of a mask that ends at `end`, grown over every span of `spans` that opens inside it
/// and ends past it, until none does. A span that overlaps the mask is a secret the mask cut
/// (a token glued to a PEM header eats its `-----BEGIN`); searching what follows the mask
/// would no longer find it, so the mask covers it whole. O(s²) for s spans, a constant.
///
/// # Arguments
/// * `end` - Where the chosen span ends.
/// * `spans` - The first span of each search, from where the last mask ended.
///
/// # Returns
/// The end of the merged mask, at least `end`.
// MAGI FIX (S6 loop 9): an overlapped PEM span was dropped and its body was sent in the clear.
fn merged_end<'a>(mut end: usize, spans: impl Iterator<Item = &'a Range<usize>> + Clone) -> usize {
    while let Some(grown) = spans
        .clone()
        .filter(|s| s.start < end && s.end > end)
        .map(|s| s.end)
        .max()
    {
        end = grown;
    }
    end
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
///
/// The bidirectional embeddings and overrides U+202A..U+202E (U+202E, RIGHT-TO-LEFT OVERRIDE,
/// included) sit in the `U+2028..=U+202E` range together with the separators, and the isolates
/// U+2066..U+2069 in `U+2066..=U+206F`: a review that reads the table as missing them (S6
/// loop 7, A2-b2) is a false positive, pinned by
/// `a_shown_path_escapes_format_characters_that_reorder_or_hide_text` with a literal U+202E.
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
// MAGI FIX (S6 loop 9, C2-m1): shared with the text report, which writes model titles with it.
pub(crate) fn escape_controls(text: &str) -> String {
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

    #[test]
    fn a_token_prefix_in_any_case_is_a_secret_and_keeps_its_boundaries() {
        // [MAGI S6 loop 15, A2-2, security] prefixes were compared case-sensitively while the
        // `ant-` exclusion was not, so `sk-ANT-…` was neither an OpenAI nor an Anthropic key and
        // its file was sent. Prefixes and exclusions now ignore ASCII case alike.
        let cases = [
            (
                concat!("sk-", "ANT-", "api03-AAAAAAAAAAAAAAAAAAAA"),
                "anthropic-key",
            ),
            (
                concat!("SK-", "ANT-", "api03-AAAAAAAAAAAAAAAAAAAA"),
                "anthropic-key",
            ),
            (
                concat!("Sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA"),
                "openai-key",
            ),
            (
                concat!("GH", "P_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                "github-token",
            ),
            (
                concat!("GITHUB_", "PAT_", "AAAAAAAAAAAAAAAAAAAAAA"),
                "github-token",
            ),
            (
                concat!("Gl", "PAT-", "AAAAAAAAAAAAAAAAAAAA"),
                "gitlab-token",
            ),
        ];
        for (text, name) in cases {
            assert_eq!(find_secret(text), Some(name), "{text}");
            assert!(
                !mask_secrets(text).contains("AAAAAAAAAAAAAAAAAAAA"),
                "{text}"
            );
            assert_eq!(
                mask_secrets(text),
                mask_secrets_by_rescanning(text),
                "{text}"
            );
        }
        // The left boundary still applies whatever the case.
        let glued = concat!("task", "-SK-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        assert_eq!(find_secret(glued), None);
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
    /// pattern. O(k · n · p), kept as the reference the one-pass masking must equal. Since S6
    /// loop 9 a mask grows over every first span that overlaps it, as `mask_secrets` does.
    fn mask_secrets_by_rescanning(text: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        loop {
            let spans: Vec<(Range<usize>, &str)> = SECRET_PATTERNS
                .iter()
                .filter_map(|p| p.matcher.first_span(rest).map(|r| (r, p.name)))
                .collect();
            let Some((span, name)) = spans.iter().min_by_key(|(r, _)| r.start).cloned() else {
                break;
            };
            out.push_str(rest.get(..span.start).unwrap_or_default());
            out.push_str(REDACTED_OPEN);
            out.push_str(name);
            out.push('>');
            let end = merged_end(span.end, spans.iter().map(|(r, _)| r));
            rest = rest.get(end..).unwrap_or_default();
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
        // [MAGI S6 loop 13, A2-c2] the URL credential, whose match is decided before its span.
        let url = concat!("db://", "u:", "pw@", "h");
        let fragments: Vec<&str> = vec![
            sk, ant, ghp, aws, aws_secret, &pem, url, "-", "\n", "_", "a", " ", "x-", "-\n", ".",
            "/", "@", ":",
        ];
        let adjacent = [
            format!("{aws}-{sk}"),
            format!("{sk}-{sk}"),
            format!("{pem}{ghp}"),
            format!("{pem}-{sk}"),
            format!("{aws}{ghp}"),
            format!("{aws_secret}-{aws}"),
            format!("x{ant}-{ghp}"),
            format!("{sk}{pem}"),
            format!("{aws}{pem}{sk}"),
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
    fn an_unencrypted_ssh2_private_key_is_a_secret_and_is_masked() {
        // [MAGI S6 loop 12, A2-m1, security] only the `ENCRYPTED` RFC 4716 header was listed; the
        // unencrypted ssh.com/Tectia armor matched no pattern and its file was sent.
        let header = concat!("---- BEGIN SSH2 ", "PRIVATE KEY ----");
        let file = format!("x = 1\n{header}\nAAAAB3NzaC1yc2EAAAADAQAB\n");
        assert_eq!(find_secret(&file), Some("pem-private-key"));
        let masked = mask_secrets(&file);
        assert!(!masked.contains("BEGIN SSH2"), "{masked}");
        assert!(!masked.contains("AAAAB3NzaC1yc2E"), "{masked}");
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

    #[test]
    fn an_aws_secret_key_under_a_colon_equals_assignment_is_a_secret_and_masked_whole() {
        // [MAGI S6 loop 9, A2-m1] Go's `:=` and Make's `:=` left the `=` after the `:`, so the
        // base64 check failed at it and the key was sent.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        for text in [
            format!("{} := \"{tail}\"", concat!("awsSecret", "AccessKey")),
            format!("{} := {tail}", concat!("AWS_SECRET", "_ACCESS_KEY")),
            format!("{}:={tail}", concat!("aws_secret", "_access_key")),
        ] {
            assert_eq!(find_secret(&text), Some("aws-secret-key"), "{text}");
            assert!(!mask_secrets(&text).contains(tail), "{text}");
        }
    }

    #[test]
    fn an_aws_secret_key_in_a_comparison_is_a_secret_and_masked_whole() {
        // [MAGI S6 loop 11, A2-m1, security] `==`, `!=` and JavaScript's `===`/`!==` failed the
        // operator check (or left an `=` before the key), so a literal key compared against the
        // name was sent while the same key after `=` was masked.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let name = concat!("aws_secret", "_access_key");
        for text in [
            format!("if {name} == \"{tail}\""),
            format!("if {name}!={tail}"),
            format!("if ({name} === '{tail}')"),
            format!("if ({name} !== \"{tail}\")"),
        ] {
            assert_eq!(find_secret(&text), Some("aws-secret-key"), "{text}");
            assert!(!mask_secrets(&text).contains(tail), "{text}");
        }
        // An operator that is none of the accepted ones still names no key.
        assert_eq!(find_secret(&format!("{name} <= {tail}")), None);
    }

    #[test]
    fn an_aws_secret_key_passed_with_its_name_as_an_argument_or_a_collection_item_is_a_secret() {
        // [MAGI S6 loop 15, A2-1, security] only an assignment or a comparison after the name
        // was read, so a quoted key passed next to it (`configure(name, "…")`, a tuple, a list,
        // a dict value in a tuple, a call wrapped over two lines) was sent.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let name = concat!("aws_secret", "_access_key");
        let texts = [
            format!("configure({name}, \"{tail}\")"),
            format!("(\"{name}\", \"{tail}\")"),
            format!("[\"{name}\", '{tail}']"),
            format!("set({name},\n    \"{tail}\")"),
            format!("{{'{name}': (\"{tail}\",)}}"),
            format!("f({name}, `{tail}`)"),
        ];
        for text in &texts {
            assert_eq!(find_secret(text), Some("aws-secret-key"), "{text}");
            let masked = mask_secrets(text);
            assert!(!masked.contains(tail), "{masked}");
            assert_eq!(masked, mask_secrets_by_rescanning(text), "{text}");
        }
        // An unquoted identifier next to the name is code, not a key; a value past the short
        // window, or after a byte that is no separator, is not taken for the key either.
        let ident = "AbcdefghijAbcdefghijAbcdefghijAbcdefghij";
        for plain in [
            format!("f({name}, {ident})"),
            format!("f({name},{}\"{tail}\")", " ".repeat(32)),
            format!("f({name}; \"{tail}\")"),
        ] {
            assert_eq!(find_secret(&plain), None, "{plain}");
        }
    }

    #[test]
    fn an_aws_secret_key_continued_on_the_next_line_is_a_secret_and_masked_whole() {
        // [MAGI S6 loop 16, A2-3, security] only blanks were skipped after the operator, so a
        // YAML block scalar (`|`, `>`, with chomping and indentation indicators), a YAML value on
        // the next indented line, and a shell line continued with `\` sent the key.
        let tail = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let name = concat!("aws_secret", "_access_key");
        let upper = concat!("AWS_SECRET", "_ACCESS_KEY");
        let texts = [
            format!("{name}: |\n  {tail}\n"),
            format!("{name}: >-\n    {tail}\n"),
            format!("{name}: |2+ \r\n  {tail}\r\n"),
            format!("{name}:\n  {tail}\n"),
            format!("export {upper}=\\\n  {tail}\n"),
            format!("{upper}= \\\n\t\"{tail}\"\n"),
        ];
        for text in &texts {
            assert_eq!(find_secret(text), Some("aws-secret-key"), "{text:?}");
            let masked = mask_secrets(text);
            assert!(!masked.contains(tail), "{masked:?}");
            assert_eq!(masked, mask_secrets_by_rescanning(text), "{text:?}");
        }
        // An indicator or a `\` with no line break after it, and a bare line break after `=`,
        // continue nothing.
        for plain in [
            format!("{name}: | {tail}"),
            format!("{upper}=\\ {tail}"),
            format!("{upper}=\n  {tail}"),
        ] {
            assert_eq!(find_secret(&plain), None, "{plain:?}");
        }
    }

    #[test]
    fn a_pem_header_is_a_secret_only_in_upper_case() {
        // [MAGI S6 loop 16, A2-6, false positive] the review read the case-insensitive probe that
        // ends the type words as the closing check: `BEGIN`, every type word and `PRIVATE KEY-----`
        // must all be upper case, as the module doc says.
        let body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7";
        for header in [
            concat!("-----", "BEGIN private key-----"),
            concat!("-----", "BEGIN rsa PRIVATE KEY-----"),
            concat!("-----", "begin PRIVATE KEY-----"),
            concat!("-----", "BEGIN RSA Private Key-----"),
        ] {
            assert_eq!(
                find_secret(&format!("{header}\n{body}\n")),
                None,
                "{header}"
            );
        }
        let upper = format!("{PEM_BEGIN}RSA {PEM_PRIVATE_KEY}\n{body}\n");
        assert_eq!(find_secret(&upper), Some("pem-private-key"));
    }

    #[test]
    fn a_token_that_runs_into_a_pem_header_masks_the_whole_pem_block() {
        // [MAGI S6 loop 9, A2-b1, security] an OpenAI-shaped token glued to a PEM header ate
        // its `-----BEGIN`, so the PEM search over what followed the mask found no header and
        // the key body was sent in the clear.
        let sk = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7";
        let end = concat!("-----", "END PRIVATE KEY-----");
        let text = format!("a {sk}{PEM_BEGIN}{PEM_PRIVATE_KEY}\n{body}\n{end} b");
        let masked = mask_secrets(&text);
        assert!(!masked.contains(PEM_PRIVATE_KEY), "{masked}");
        assert!(!masked.contains(body), "{masked}");
        assert!(!masked.contains("AAAAAAAA"), "{masked}");
        assert!(masked.starts_with("a <redacted: "), "{masked}");
        assert!(masked.ends_with("> b"), "{masked}");
    }

    #[test]
    fn a_password_in_a_url_is_a_secret_and_a_url_without_one_is_not() {
        // [MAGI S6 loop 13, A2-c2, security] no pattern matched `scheme://user:password@host`, so
        // a reviewed `docker-compose.yml` or `settings.py` holding a database URL was batched
        // whole and its password left the process, although the product already treats this
        // very shape as a credential (REQ-A16c, `redact_url`).
        let user = "svc";
        let at = |scheme: &str, pass: &str, host: &str| format!("{scheme}://{user}:{pass}@{host}");
        for secret in [
            at("postgres", "S3cr3t", "db.internal/app"),
            format!("DATABASE_URL=\"{}\"", at("mysql", "p", "10.0.0.1:3306")),
            at("https", "x", "[::1]:8080/"),
            format!("redis://:{}@cache:6379/0", "S3cr3t"),
            // An unencoded `@` in the password: the authority's last `@` ends the mask.
            at("postgres", "S3cr3t@S3cr3t", "db"),
        ] {
            assert_eq!(find_secret(&secret), Some("url-credential"), "{secret}");
            let masked = mask_secrets(&secret);
            assert!(masked.contains("<redacted: url-credential>"), "{masked}");
            assert!(
                !masked.contains("S3cr3t") && !masked.contains(":p@"),
                "{masked}"
            );
        }
        for plain in [
            "https://example.com/a@b".to_string(),
            format!("ssh://{user}@github.com:22/x"),
            "http://host:8080/path".to_string(),
            "let u = format!(\"{scheme}://{user}:{password}@{host}\");".to_string(),
            format!("postgres://{user}:$PASSWORD@db"),
            format!("postgres://{user}:<password>@db"),
            "https://***@example.invalid/x".to_string(),
            format!("https://{user}@example.invalid:443/a:b@c"),
            "mailto:a@b.c".to_string(),
        ] {
            assert_eq!(find_secret(&plain), None, "{plain}");
        }
    }

    #[test]
    fn a_url_password_is_a_placeholder_only_in_an_exact_placeholder_form() {
        // [MAGI S6 loop 14, A2-1, security] only the password's first byte was checked, so a
        // real password that opens with `$`, `{`, `[`, `%` or `*` (`$ecret`, `{actual`, `*pw`)
        // read as a placeholder and its file was sent.
        let user = "svc";
        let url = |pass: &str| format!("postgres://{user}:{pass}@db.internal/app");
        for real in [
            "$ecret", "$3cr3t", "{actual", "{a}b", "[pw]x", "%41bc", "*pw", "**x**", "$", "${}",
            "${PASS",
        ] {
            let secret = url(real);
            assert_eq!(find_secret(&secret), Some("url-credential"), "{secret}");
        }
        for placeholder in [
            "{password}",
            "{}",
            "{0}",
            "{{ db_password }}",
            "{{.Values.password}}",
            "${DB_PASSWORD}",
            "${db.password}",
            "$PASSWORD",
            "$DB_PASS_2",
            "[password]",
            "%s",
            "***",
        ] {
            let plain = url(placeholder);
            assert_eq!(find_secret(&plain), None, "{plain}");
        }
    }

    #[test]
    fn a_url_password_may_hold_the_sub_delimiters_a_userinfo_allows() {
        // [MAGI S6 loop 14, A2-1, security] `'`, `(` and `)` ended the userinfo, although RFC
        // 3986 allows them unencoded in it: `svc:pa(ss@db` found no `@` and its file was sent.
        for pass in ["pa(ss", "pa)ss", "pa'ss", "(pass)"] {
            let secret = format!("postgres://svc:{pass}@db.internal/app");
            assert_eq!(find_secret(&secret), Some("url-credential"), "{secret}");
            let masked = mask_secrets(&secret);
            assert!(!masked.contains("ss@"), "{masked}");
        }
        for plain in [
            "(see https://example.com/a) and (b:c@d)",
            "'https://example.com' 'a:b@c'",
        ] {
            assert_eq!(find_secret(plain), None, "{plain}");
        }
    }

    #[test]
    fn a_url_credential_is_found_whatever_follows_its_at_sign() {
        // [MAGI S6 loop 14, A2-1, security review] only the run's last `@` was tried, so a later
        // `@` with no host after it (`'@'`) or past the userinfo cap (a minified line) hid a real
        // credential before it, and its file was sent.
        let far = "x".repeat(MAX_URL_USERINFO + MAX_URL_HOST_PEEK);
        for secret in [
            "f('postgres://u:pw@db','@')".to_string(),
            "f('postgres://u:pw@db','a@')".to_string(),
            format!("var a='postgres://u:pw@db',b='{far}@x';"),
            format!("postgres://u:pw@db,{far}"),
        ] {
            assert_eq!(find_secret(&secret), Some("url-credential"), "{secret}");
            let masked = mask_secrets(&secret);
            assert!(!masked.contains("u:pw@"), "{masked}");
        }
        // An unencoded `@` in the password still masks the whole userinfo.
        let masked = mask_secrets("postgres://u:pw@pw@db");
        assert_eq!(masked, "postgres://<redacted: url-credential>db");
    }

    #[test]
    fn an_ssh2_private_key_is_masked_to_its_own_end_line() {
        // [MAGI S6 loop 14, A2-3] the end of a block was searched with five dashes only, so an
        // SSH2 key (four-dash RFC 4716 armor) was masked to the end of the text and everything
        // after it was lost from the report, the error or the model text that carried it.
        for kind in ["", "ENCRYPTED "] {
            let begin = format!("{}{kind}PRIVATE KEY ----", concat!("---- BEGIN ", "SSH2 "));
            let end = format!("{}{kind}PRIVATE KEY ----", concat!("---- END ", "SSH2 "));
            let text = format!("a {begin}\nAAAAB3NzaC1yc2EAAAADAQAB\n{end} b\nkept");
            assert_eq!(
                mask_secrets(&text),
                "a <redacted: pem-private-key> b\nkept",
                "{text}"
            );
        }
        // With no end line, the mask still runs to the end of the text: the safe direction.
        let open = format!("a {SSH2_PLAIN_PRIVATE_KEY}\nAAAAB3NzaC1yc2EAAAADAQAB");
        assert_eq!(mask_secrets(&open), "a <redacted: pem-private-key>");
    }
}
