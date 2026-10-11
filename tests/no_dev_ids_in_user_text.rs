// Author: Julian Bolivar
// Version: 0.23.0
// Date: 2026-10-11

//! Guards the text a user can read against development identifiers (`REQ-…`, `SC-…`, `D-…`,
//! `PM-…`, `R-…`, `RF-…`, `F-…`).
//!
//! Three surfaces are covered here: every string literal of production code under `src/`
//! (the maintainer benches in `src/bin/` included), the published documentation (`README.md`
//! and `docs/*.md`), and the unreleased section of `CHANGELOG.md`. The `--help` screens are
//! guarded by the binary's own unit tests, which can see the clap tree.
//!
//! The development-id matcher below is a copy of the one in `src/main.rs`: an integration test
//! cannot import from the binary.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn no_production_string_literal_carries_a_development_id() {
    let mut hits = Vec::new();
    for path in rust_sources_under(Path::new("src")) {
        let source = fs::read_to_string(&path)
            .expect("readable")
            .replace('\r', "");
        for (line, literal) in production_string_literals(&source) {
            if let Some(id) = dev_id_in(&literal) {
                hits.push(format!("{}:{line}: {id} in {literal:?}", path.display()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "user-facing text carries development ids:\n{}",
        hits.join("\n")
    );
}

#[test]
fn the_published_docs_carry_no_development_id() {
    let mut docs = vec![PathBuf::from("README.md")];
    docs.extend(markdown_files_directly_under(Path::new("docs")));
    let mut hits = Vec::new();
    for path in docs {
        let text = fs::read_to_string(&path)
            .expect("readable")
            .replace('\r', "");
        for (n, line) in text.lines().enumerate() {
            if let Some(id) = dev_id_in(line) {
                hits.push(format!("{}:{}: {id}", path.display(), n + 1));
            }
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn the_unreleased_changelog_section_carries_no_development_id() {
    let text = fs::read_to_string("CHANGELOG.md")
        .expect("readable")
        .replace('\r', "");
    let section = unreleased_section(&text).expect("an [Unreleased] or [0.23.0] section exists");
    assert_eq!(dev_id_in(section), None);
}

#[test]
fn the_dev_id_matcher_follows_its_table() {
    let table: [(&str, Option<&str>); 9] = [
        ("global (REQ-V04).", Some("REQ-V04")),
        ("see D-V07", Some("D-V07")),
        ("(REQ-H02/H21)", Some("REQ-H02")),
        ("D-PM-1 says", Some("D-PM-1")),
        ("SC-H24, PM-C1, R-V08, RF-8, F-3", Some("SC-H24")),
        ("UTF-8 and SHA-256 and AES-256-GCM-SIV", None),
        ("D-Bus, R-tree, a PM-ready build", None),
        ("REQUEST-1", None),
        ("", None),
    ];
    for (text, expected) in table {
        assert_eq!(dev_id_in(text), expected, "{text:?}");
    }
}

#[test]
fn the_literal_extractor_follows_its_fixtures() {
    let source = r##"
/// Doc comment (REQ-A1) is not a literal.
fn shown() -> &'static str { "visible (REQ-V04)" }
// a line comment "REQ-X9"
/* a block "REQ-Y9" comment */
fn raw() -> &'static str { r#"raw "quoted" (D-V07)"# }
fn escaped() -> &'static str { "say \"hi\" { not a brace" }
fn chars() -> char { '{' }
fn lifetime<'a>(x: &'a str) -> &'a str { x }
#[cfg(test)]
mod tests {
    #[test] fn t() { assert_eq!("hidden (REQ-T1)", "x"); }
}
fn after_tests() -> &'static str { "after (SC-H24)" }
"##;
    let found: Vec<String> = production_string_literals(source)
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    assert_eq!(
        found,
        vec![
            "visible (REQ-V04)".to_string(),
            r#"raw "quoted" (D-V07)"#.to_string(),
            r#"say \"hi\" { not a brace"#.to_string(),
            "after (SC-H24)".to_string(),
        ]
    );
}

/// Attribute that marks a test-only item, whose literals are not production text.
const CFG_TEST_ATTRIBUTE: &str = "#[cfg(test)]";

/// Prefixes of the development identifiers that must never reach a user (requirements,
/// scenarios, decisions, premortem gaps, review findings).
const DEV_ID_PREFIXES: [&str; 7] = ["REQ", "SC", "D", "PM", "R", "RF", "F"];

/// First development identifier in `text` (for example `REQ-V04`, `D-PM-1`, `RF-8`), if any.
///
/// An identifier is one of [`DEV_ID_PREFIXES`] as a whole token, then `-`, an optional run of
/// capitals, optionally another `-` and run of capitals, then at least one digit. Walks the
/// characters, never byte offsets, so non-ASCII text is safe. Copy of the matcher in
/// `src/main.rs`: an integration test cannot import from the binary.
///
/// Complexity: `O(n * p)` for `n` characters and `p` prefixes.
fn dev_id_in(text: &str) -> Option<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    for start in 0..chars.len() {
        if start > 0 && at(start - 1).is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        for prefix in DEV_ID_PREFIXES {
            let mut i = start;
            if !prefix.chars().all(|p| {
                let ok = at(i) == Some(p);
                i += 1;
                ok
            }) || at(i) != Some('-')
            {
                continue;
            }
            i += 1;
            while at(i).is_some_and(|c| c.is_ascii_uppercase()) {
                i += 1;
            }
            if at(i) == Some('-') {
                i += 1;
                while at(i).is_some_and(|c| c.is_ascii_uppercase()) {
                    i += 1;
                }
            }
            if !at(i).is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            while at(i).is_some_and(|c| c.is_ascii_digit()) {
                i += 1;
            }
            let end = chars.get(i).map_or(text.len(), |&(b, _)| b);
            return text.get(chars[start].0..end);
        }
    }
    None
}

/// String literals of production code in `source`, as `(1-based line, raw content)`.
///
/// Raw content means escapes are not resolved. Line, block and doc comments are skipped, and so
/// is every item preceded by `#[cfg(test)]`, up to its closing brace or semicolon. Raw strings,
/// byte strings and character literals are told apart from lifetimes.
///
/// Complexity: `O(n)` in the length of `source`.
fn production_string_literals(source: &str) -> Vec<(usize, String)> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;
    // `skipping` is set after a `#[cfg(test)]` attribute until the item it decorates ends.
    let mut skipping = false;
    let mut depth = 0usize;
    let mut entered = false;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut nesting = 1usize;
            i += 2;
            while i < chars.len() && nesting > 0 {
                match (chars[i], chars.get(i + 1).copied()) {
                    ('/', Some('*')) => {
                        nesting += 1;
                        i += 2;
                    }
                    ('*', Some('/')) => {
                        nesting -= 1;
                        i += 2;
                    }
                    ('\n', _) => {
                        line += 1;
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
        } else if c == '#' && starts_with_at(&chars, i, CFG_TEST_ATTRIBUTE) {
            skipping = true;
            depth = 0;
            entered = false;
            i += CFG_TEST_ATTRIBUTE.len();
        } else if let Some((hashes, open_len)) = raw_string_opening(&chars, i) {
            let start_line = line;
            let body_start = i + open_len;
            let mut j = body_start;
            while j < chars.len() && !raw_string_closes(&chars, j, hashes) {
                if chars[j] == '\n' {
                    line += 1;
                }
                j += 1;
            }
            if !skipping {
                out.push((
                    start_line,
                    chars[body_start..j.min(chars.len())].iter().collect(),
                ));
            }
            i = j + 1 + hashes;
        } else if c == '"' {
            let start_line = line;
            let mut j = i + 1;
            while j < chars.len() && chars[j] != '"' {
                if chars[j] == '\\' {
                    j += 1;
                }
                if chars.get(j) == Some(&'\n') {
                    line += 1;
                }
                j += 1;
            }
            if !skipping {
                out.push((
                    start_line,
                    chars[i + 1..j.min(chars.len())].iter().collect(),
                ));
            }
            i = j + 1;
        } else if c == '\'' {
            i += char_literal_len(&chars, i);
        } else {
            if skipping {
                match c {
                    '{' => {
                        depth += 1;
                        entered = true;
                    }
                    '}' => {
                        depth = depth.saturating_sub(1);
                        if entered && depth == 0 {
                            skipping = false;
                        }
                    }
                    ';' if depth == 0 && !entered => skipping = false,
                    _ => {}
                }
            }
            i += 1;
        }
    }
    out
}

/// Whether `chars[at..]` starts with `needle`.
fn starts_with_at(chars: &[char], at: usize, needle: &str) -> bool {
    needle
        .chars()
        .enumerate()
        .all(|(k, n)| chars.get(at + k) == Some(&n))
}

/// If a raw string literal (`r"`, `r#"`, `br#"`) opens at `at`, its hash count and the length of
/// the opening delimiter. The preceding character must not continue an identifier.
fn raw_string_opening(chars: &[char], at: usize) -> Option<(usize, usize)> {
    if at > 0 && (chars[at - 1].is_alphanumeric() || chars[at - 1] == '_') {
        return None;
    }
    let mut j = at;
    if chars.get(j) == Some(&'b') {
        j += 1;
    }
    if chars.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while chars.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    (chars.get(j) == Some(&'"')).then_some((hashes, j + 1 - at))
}

/// Whether a raw string with `hashes` hashes closes at `at`.
fn raw_string_closes(chars: &[char], at: usize, hashes: usize) -> bool {
    chars.get(at) == Some(&'"') && (1..=hashes).all(|k| chars.get(at + k) == Some(&'#'))
}

/// Length of what starts at the `'` in `chars[at]`: a whole character literal, or just the quote
/// when it begins a lifetime.
fn char_literal_len(chars: &[char], at: usize) -> usize {
    match (chars.get(at + 1), chars.get(at + 2)) {
        (Some('\\'), _) => {
            let mut j = at + 3;
            while j < chars.len() && chars[j] != '\'' {
                j += 1;
            }
            j + 1 - at
        }
        (Some(_), Some('\'')) => 3,
        _ => 1,
    }
}

/// Every `.rs` file under `dir`, recursively, in a stable order.
fn rust_sources_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .expect("readable directory")
        .map(|e| e.expect("readable entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            found.extend(rust_sources_under(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            found.push(path);
        }
    }
    found
}

/// The `.md` files directly inside `dir`, not its subdirectories (`docs/test/` holds smoke
/// evidence, which is not published documentation).
fn markdown_files_directly_under(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .expect("readable directory")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "md"))
        .collect();
    found.sort();
    found
}

/// The first changelog section: from its `## [` heading up to the next one.
///
/// Complexity: `O(n)`.
fn unreleased_section(changelog: &str) -> Option<&str> {
    let start = changelog.find("## [")?;
    let rest = &changelog[start + 1..];
    let end = rest
        .find("\n## [")
        .map_or(changelog.len(), |e| start + 1 + e);
    changelog.get(start..end)
}
