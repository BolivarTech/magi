// Author: Julian Bolivar
// Version: 2.1.0
// Date: 2026-08-07
//! Fails if the repo tree contains key-like material, private IPs, or hardcoded absolute
//! user paths.
//!
//! Covers THREE surfaces with different rules — the list is explicit on purpose (loop 1 fix
//! round CE, F13): the third was added because a file new to this milestone,
//! `tests/support/mod.rs`, fell outside both of the other two — clean by accident of what
//! someone happened to write, not by scan coverage. A new `tests/*.rs` added tomorrow must
//! fall under the third without anyone having to remember to extend it:
//!
//! - **Sources** (`src/**/*.rs`): key-like patterns only, ignoring comment lines. Historical
//!   behavior, unchanged.
//! - **Documentation** (`.md`/`.toml`/`.yml`/`.yaml`/`.example` at the root, `docs/` and
//!   `.github/`): keys plus private IPs, internal hostnames, and absolute user paths.
//!   Documentation is the surface that most easily leaks infrastructure, and until now nobody
//!   was watching it.
//! - **Test helpers** (`tests/**/*.rs`): the same non-strict treatment as sources
//!   (`skip_line_comments = true`, `strict = false`) — a test double earns the same tolerance
//!   as production code, because `strict` mode's IP/path checks are aimed at documentation
//!   prose, not code.
//!
//! A line can be exempted with the [`ALLOW_MARKER`] marker. It's deliberately
//! explicit: it exempts **one** line, stays visible in the file, and forces the exemption to
//! be someone's decision rather than a side effect of where the text happened to land. This
//! very file needs the marker on several lines: it's where the patterns and the adversarial
//! fixtures that test them live, so with the third surface covering itself, those lines would
//! detect themselves if they weren't exempted.

use std::fs;
use std::path::Path;

/// Marker that exempts its line from the scan.
///
/// Meant for prose that *describes* the patterns (the gate table in
/// `docs/CODE-STANDARDS-CHECKLIST.md` names them literally) or for synthetic fixtures.
/// Equivalent to `# noqa` / `gitleaks:allow`.
const ALLOW_MARKER: &str = "allow-secret-scan";

/// Extensions treated as documentation or configuration.
const DOC_EXTENSIONS: [&str; 5] = ["md", "toml", "yml", "yaml", "example"];

/// Root files that are part of what ships.
///
/// Explicit list instead of walking `.`: the root mixes tracked files with gitignored local
/// ones that legitimately contain internal infrastructure.
const ROOT_DOCS: [&str; 5] = [
    "README.md",
    "CHANGELOG.md",
    "Cargo.toml",
    "deny.toml",
    "rustfmt.toml",
];

/// Directories that are never walked: build artifacts, git metadata, and generated output.
const SKIP_DIRS: [&str; 4] = ["target", ".git", "graphify-out", "node_modules"];

/// A finding: path, line (1-based), and the category that triggered it.
type Hit = (String, usize, &'static str);

/// `true` if `tok` is an IPv4 in a private range (RFC 1918).
///
/// Implemented by hand instead of with `regex` to avoid adding a dependency (§0.2/B14: prefer
/// the standard library) for a pattern this simple.
fn is_private_ipv4(tok: &str) -> bool {
    let parts: Vec<&str> = tok.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let mut octets = [0u16; 4];
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() || part.len() > 3 || !part.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        match part.parse::<u16>() {
            Ok(v) if v <= 255 => octets[i] = v,
            _ => return false,
        }
    }
    octets[0] == 10
        || (octets[0] == 192 && octets[1] == 168)
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
}

/// `true` if the line contains a private IPv4.
///
/// Splits on any character that isn't a digit or a dot, so that
/// `http://192.168.0.30:8080` produces the token `192.168.0.30`.
fn has_private_ipv4(line: &str) -> bool {
    line.split(|c: char| !c.is_ascii_digit() && c != '.')
        .any(is_private_ipv4)
}

/// `true` if the line contains an absolute path to a specific user's home directory.
///
/// Requires an alphanumeric character after the prefix so a generic mention of `/home/` or
/// `C:\Users\` with no name after it doesn't get flagged.
fn has_absolute_home_path(line: &str) -> bool {
    const PREFIXES: [&str; 4] = ["/home/", "/Users/", "C:\\Users\\", "C:/Users/"];
    PREFIXES.iter().any(|p| {
        line.match_indices(p).any(|(idx, _)| {
            line[idx + p.len()..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric())
        })
    })
}

/// `true` if the line contains an `sk-…` token with real key material behind it.
///
/// The 16-character threshold distinguishes an actual key from a mention in prose
/// (`sk-ant-api...`, `sk-<your-key>`), which is the frequent false positive.
fn has_key_like_token(line: &str) -> bool {
    line.match_indices("sk-").any(|(idx, _)| {
        line[idx + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .count()
            >= 16
    })
}

/// Classifies a line, returning the category of the first pattern that fires.
///
/// `strict` turns on the checks that only apply to documentation (private IP, absolute path).
/// Returns `None` if the line carries [`ALLOW_MARKER`].
fn classify(line: &str, strict: bool) -> Option<&'static str> {
    if line.contains(ALLOW_MARKER) {
        return None;
    }
    // The two `if` lines below are the PATTERN definitions themselves, not leaked values — with
    // the F13 extension of this scan to tests/**/*.rs, this file's own detection logic would
    // otherwise flag itself; each carries its own trailing marker because the scanner
    // classifies line by line, so a marker on this comment would not reach them.
    if line.contains("sk-ant-api" /* allow-secret-scan */) {
        return Some("Anthropic key");
    }
    if line.contains("-----BEGIN" /* allow-secret-scan */) {
        return Some("PEM block");
    }
    // Documentation only: `src/` is full of legitimate synthetic fixtures
    // (`sk-super-secret-PROBE-value`, `sk-proj-OPENAISECRET`) that exist to prove redaction
    // works. Flagging them one by one would be noise, and widening the source scan is not
    // this gate's scope.
    if strict && has_key_like_token(line) {
        return Some("key-like token");
    }
    if strict && has_private_ipv4(line) {
        return Some("private IP");
    }
    if strict && has_absolute_home_path(line) {
        return Some("absolute user path");
    }
    None
}

/// Walks `dir`, accumulating findings over files whose extension is in `extensions`.
///
/// `skip_line_comments` ignores lines starting with `//`, needed in Rust sources where
/// internal documentation names the patterns. `strict` is passed through unchanged to
/// [`classify`].
fn scan(
    dir: &Path,
    extensions: &[&str],
    skip_line_comments: bool,
    strict: bool,
    hits: &mut Vec<Hit>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !SKIP_DIRS.contains(&name) {
                scan(&path, extensions, skip_line_comments, strict, hits);
            }
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !extensions.contains(&ext) {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for (i, line) in content.lines().enumerate() {
            if skip_line_comments && line.trim_start().starts_with("//") {
                continue;
            }
            if let Some(kind) = classify(line, strict) {
                hits.push((path.display().to_string(), i + 1, kind));
            }
        }
    }
}

/// Formats the findings for the failure message, one line per finding.
fn render(hits: &[Hit]) -> String {
    hits.iter()
        .map(|(f, l, k)| format!("\n  {f}:{l}  [{k}]"))
        .collect()
}

/// No `.rs` under `src/` carries key-like material. Historical behavior.
#[test]
fn test_no_hardcoded_secrets_in_source_tree() {
    let mut hits = Vec::new();
    scan(Path::new("src"), &["rs"], true, false, &mut hits);
    assert!(
        hits.is_empty(),
        "possible hardcoded secret in:{}",
        render(&hits)
    );
}

/// No doc or config file carries keys, private IPs, or absolute user paths.
///
/// The README and `docs/` are the surface that most easily leaks internal infrastructure, and
/// they used to be completely outside the scan.
#[test]
fn test_no_hardcoded_secrets_in_documentation() {
    let mut hits = Vec::new();
    // `tests/fixtures` holds committed sample `magi.toml` files (the v0.11.0
    // migration fixtures, REQ-A21c). Fixtures are exactly the kind of file that
    // can accidentally carry a real credential someone pasted while generating
    // them, so this surface gets the same strict scan as `docs`/`.github` —
    // the one deliberate exception (a synthetic credential in
    // `with-credentials.toml`, needed to exercise the migration-error
    // redaction path) is exempted line-by-line via `ALLOW_MARKER`, not by
    // excluding the directory.
    for dir in ["docs", ".github", "tests/fixtures"] {
        scan(Path::new(dir), &DOC_EXTENSIONS, false, true, &mut hits);
    }
    // The root goes by explicit list, NOT by walking the directory: tracked files coexist
    // there with gitignored local ones (`CLAUDE.local.md`, `magi.toml`) that legitimately
    // contain internal IPs and paths and are never published. Enumerating what actually ships
    // avoids depending on git from the test and keeps the scope visible in the code.
    for name in ROOT_DOCS {
        let path = Path::new(name);
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        for (i, line) in content.lines().enumerate() {
            if let Some(kind) = classify(line, true) {
                hits.push((name.to_string(), i + 1, kind));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "possible leak in documentation:{}",
        render(&hits)
    );
}

/// The detector is NOT blind: every known leak shape triggers its category.
///
/// Without this test, the two tests above could pass by finding nothing to look at, which is
/// indistinguishable from passing because things are actually clean.
#[test]
fn test_detector_catches_known_leak_shapes() {
    // Fixtures below are deliberately shaped like each leak category, not real secrets — F13
    // extended this scan to tests/**/*.rs, so the two entries the scan itself would trip on
    // carry the same allowance `with-credentials.toml` already uses (trailing per line: the
    // scanner classifies line by line).
    let cases: [(&str, &str); 6] = [
        (
            "api_key = \"sk-ant-api03-REDACTED\"", /* allow-secret-scan */
            "Anthropic key",
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----", /* allow-secret-scan */
            "PEM block",
        ),
        ("token: sk-abcdefghijklmnopqrstuvwxyz", "key-like token"),
        ("host = \"192.168.0.30\"", "private IP"),
        ("remote: 10.1.2.3:22", "private IP"),
        ("path = \"/home/jdoe/.ssh/id_rsa\"", "absolute user path"),
    ];
    for (line, expected) in cases {
        assert_eq!(
            classify(line, true),
            Some(expected),
            "did not detect the leak in: {line}"
        );
    }
}

/// Legitimate prose and placeholders don't trigger, or the gate becomes unusable.
#[test]
fn test_detector_ignores_prose_and_placeholders() {
    let benign = [
        "set ANTHROPIC_API_KEY or store it in the vault",
        "base_url = \"http://localhost:11434/v1\"",
        "el prefijo sk-ant se menciona sin material de clave detrás",
        "clone it to /home/ and run the build",
        "version 2.35 or newer, see 1.2.3.4 in the RFC",
        "a public address like 8.8.8.8 is not private",
    ];
    for line in benign {
        assert_eq!(classify(line, true), None, "false positive in: {line}");
    }
}

/// REQ-A00b (loop 1 fix round CE, F13): source-tree-style scanning now also reaches
/// `tests/**/*.rs`. Before this test, `tests/support/mod.rs` — new in this milestone — fell
/// outside BOTH existing scans: the source-tree one above only walks `src`, and the
/// documentation one only reads doc extensions (`.md`/`.toml`/`.yml`/`.yaml`/`.example`), never
/// `.rs`. It was clean by content, never by coverage — a future edit to a test double had no
/// safety net the way `src/**/*.rs` and `tests/fixtures` already do.
///
/// Same treatment as the source-tree scan (`skip_line_comments = true`, `strict = false`): a
/// test double is allowed the same non-strict pass source gets, since `strict` mode's IP/path
/// checks are aimed at documentation prose, not code.
#[test]
fn test_no_hardcoded_secrets_in_test_helpers() {
    let mut hits = Vec::new();
    scan(Path::new("tests"), &["rs"], true, false, &mut hits);
    assert!(
        hits.is_empty(),
        "possible hardcoded secret in tests/**/*.rs:{}",
        render(&hits)
    );
}

/// The scan above passing is not, by itself, proof that it is watching anything — the same
/// "clean is not the same as covered" gap this whole finding is about. This drives `scan` over
/// a tempdir shaped like the coverage it protects (a `.rs` file, non-strict) and plants a real
/// leak shape in it, so the assertion above is backed by something that can be shown to fail.
#[test]
fn test_the_rs_scan_catches_a_planted_leak_like_test_helpers_would() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("planted.rs"),
        // Matches the unconditional "Anthropic key" pattern, not the `strict`-gated
        // key-like-token one: the real coverage this proves reuses `strict = false` (same
        // non-strict pass `src/**/*.rs` gets), so a `strict`-only pattern would prove nothing.
        "let leaked = \"sk-ant-api-planted-for-this-test\";\n", // allow-secret-scan: fixture
    )
    .expect("write planted fixture");
    let mut hits = Vec::new();
    scan(dir.path(), &["rs"], true, false, &mut hits);
    assert!(
        !hits.is_empty(),
        "the scan did not detect the leak planted under a .rs tree"
    );
}

/// The marker exempts exactly its own line, and only its own.
#[test]
fn test_allow_marker_exempts_only_its_own_line() {
    let leak = "key = \"sk-ant-api03-REDACTED\""; // allow-secret-scan: fixture, not a real key
    assert!(
        classify(leak, true).is_some(),
        "the line with no marker must trigger"
    );
    let marked = format!("{leak} // {ALLOW_MARKER}");
    assert_eq!(classify(&marked, true), None, "the marker must exempt it");
}
