// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
//! Falla si el árbol de fuentes contiene material tipo-clave hardcodeado.
use std::fs;
use std::path::Path;

fn scan(dir: &Path, hits: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !matches!(name, "target" | ".git" | "graphify-out") {
                scan(&path, hits);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&path) {
                for (i, line) in content.lines().enumerate() {
                    // sk-ant / sk- / private key headers; excluye comentarios de ejemplo.
                    let l = line.trim_start();
                    if l.starts_with("//") {
                        continue;
                    }
                    if line.contains("sk-ant-api") || line.contains("-----BEGIN") {
                        hits.push(format!("{}:{}", path.display(), i + 1));
                    }
                }
            }
        }
    }
}

#[test]
fn test_no_hardcoded_secrets_in_source_tree() {
    let mut hits = Vec::new();
    scan(Path::new("src"), &mut hits);
    assert!(hits.is_empty(), "posible secreto hardcodeado en: {hits:?}");
}
