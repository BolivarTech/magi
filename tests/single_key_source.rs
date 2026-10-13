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
