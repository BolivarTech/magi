// Author: Julian Bolivar
// Version: 0.24.0
// Date: 2026-10-13

//! REQ-K03: one origin comparison, one key selection and one env > vault precedence in `src/`.

use std::path::{Path, PathBuf};

/// Where a file's production code ends: the convention of `FIRST_TEST_MODULE` in `src/main.rs`.
const FIRST_TEST_MODULE: &str = "\n#[cfg(test)]\nmod ";

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable source dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `(path relative to src/ with '/', production text without '\r')` for every `.rs` under `src/`.
fn production_sources() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p)
                .expect("utf-8 source")
                .replace('\r', "");
            let prod = text
                .split(FIRST_TEST_MODULE)
                .next()
                .unwrap_or("")
                .to_string();
            let rel = p
                .strip_prefix(&root)
                .expect("under src")
                .to_string_lossy()
                .replace('\\', "/");
            (rel, prod)
        })
        .collect()
}

/// Files holding a `fn <name>(` definition on a non-comment line.
fn definitions(name: &str) -> Vec<String> {
    let needle = format!("fn {name}(");
    production_sources()
        .into_iter()
        .filter(|(_, text)| {
            text.lines()
                .any(|l| !l.trim_start().starts_with("//") && l.contains(&needle))
        })
        .map(|(path, _)| path)
        .collect()
}

#[test]
fn origin_comparison_and_key_selection_are_defined_once_in_the_neutral_module() {
    assert_eq!(
        definitions("same_origin"),
        vec!["endpoint_key.rs".to_string()]
    );
    assert_eq!(
        definitions("select_api_key"),
        vec!["endpoint_key.rs".to_string()]
    );
}

/// The body of `fn <name>(` in `text`, up to the next top-level item.
fn fn_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let start = text.find(&format!("fn {name}("))?;
    let rest = &text[start..];
    let end = [
        "\nfn ",
        "\nasync fn ",
        "\npub fn ",
        "\npub(crate) fn ",
        "\nstruct ",
        "\nimpl ",
        "\nconst ",
    ]
    .iter()
    .filter_map(|n| rest[1..].find(n).map(|i| i + 1))
    .min()
    .unwrap_or(rest.len());
    Some(&rest[..end])
}

#[test]
fn the_env_then_vault_precedence_is_defined_once_and_its_former_copies_delegate() {
    assert_eq!(
        definitions("env_or_vault"),
        vec!["section_keys.rs".to_string()]
    );
    for (path, text) in production_sources() {
        for former in ["resolve_openai_key", "discover_config", "decision_api_key"] {
            if let Some(body) = fn_body(&text, former) {
                assert!(
                    !body.contains(".get("),
                    "{path}: {former} reads the vault itself instead of calling env_or_vault"
                );
            }
        }
    }
}

/// Names of the functions, in production code, that call `fn <callee>(` (definitions and comments excluded).
/// Every caller this returns must have a non-empty name (asserted by the guard): a declaration shape the
/// tokenizer misses yields an empty name and turns the guard red instead of passing silently.
fn callers(callee: &str) -> Vec<(String, String)> {
    let call = format!("{callee}(");
    let def = format!("fn {callee}(");
    let mut out = Vec::new();
    for (path, text) in production_sources() {
        let mut current = String::new();
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            if let Some(i) = t.find("fn ") {
                if let Some(name) = t[i + 3..].split(['(', '<']).next() {
                    if t[..i].split_whitespace().all(|w| {
                        matches!(
                            w,
                            "pub"
                                | "pub(crate)"
                                | "pub(super)"
                                | "async"
                                | "const"
                                | "unsafe"
                                | "extern"
                                | "\"C\""
                                | "default"
                        )
                    }) {
                        current = name.trim().to_string();
                    }
                }
            }
            if t.contains(&call) && !t.contains(&def) {
                out.push((path.clone(), current.clone()));
            }
        }
    }
    out
}

#[test]
fn same_origin_is_called_only_by_select_api_key_and_that_only_by_section_keys_resolve() {
    assert!(
        callers("same_origin")
            .iter()
            .all(|(p, f)| p == "endpoint_key.rs" && f == "select_api_key"),
        "{:?}",
        callers("same_origin")
    );
    assert_eq!(
        callers("select_api_key"),
        vec![("section_keys.rs".to_string(), "resolve".to_string())]
    );
}
