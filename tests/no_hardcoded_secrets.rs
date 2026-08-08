// Author: Julian Bolivar
// Version: 2.1.0
// Date: 2026-08-07
//! Falla si el árbol del repo contiene material tipo-clave, IPs privadas o rutas
//! absolutas de usuario hardcodeadas.
//!
//! Cubre TRES superficies con reglas distintas — la lista es explícita a propósito (loop 1 fix
//! round CE, F13): la tercera se agregó porque un archivo nuevo de este milestone,
//! `tests/support/mod.rs`, no caía en ninguna de las otras dos, limpio por accidente de lo que
//! alguien escribió y no por cobertura del escaneo. Un `tests/*.rs` nuevo que se sume mañana
//! debe caer bajo la tercera sin que nadie tenga que acordarse de extenderla:
//!
//! - **Fuentes** (`src/**/*.rs`): patrones tipo-clave únicamente, ignorando líneas
//!   de comentario. Comportamiento histórico, sin cambios.
//! - **Documentación** (`.md`/`.toml`/`.yml`/`.yaml`/`.example` en la raíz, `docs/`
//!   y `.github/`): además de las claves, IPs privadas, hostnames internos y rutas
//!   absolutas de usuario. La documentación es la superficie que más fácil filtra
//!   infraestructura, y hasta ahora no la miraba nadie.
//! - **Ayudantes de test** (`tests/**/*.rs`): el mismo trato no-estricto que las fuentes
//!   (`skip_line_comments = true`, `strict = false`) — un doble de test se gana la misma
//!   tolerancia que el código de producción, porque los chequeos de IP/ruta de `strict` apuntan
//!   a prosa de documentación, no a código.
//!
//! Una línea puede eximirse con el marcador [`ALLOW_MARKER`]. Es deliberadamente
//! explícito: exime **una** línea, queda visible en el archivo, y obliga a que la
//! exención sea una decisión de alguien y no un efecto lateral de dónde cayó el texto. Este
//! propio archivo necesita el marcador en varias líneas: es donde viven los patrones y los
//! fixtures adversariales que los prueban, así que con la tercera superficie cubriéndose a sí
//! mismo, esas líneas se detectan a sí mismas si no se eximen.

use std::fs;
use std::path::Path;

/// Marcador que exime a su línea del escaneo.
///
/// Pensado para prosa que *describe* los patrones (la tabla de gates en
/// `docs/CODE-STANDARDS-CHECKLIST.md` los nombra literalmente) o para fixtures
/// sintéticos. Equivale a `# noqa` / `gitleaks:allow`.
const ALLOW_MARKER: &str = "allow-secret-scan";

/// Extensiones tratadas como documentación o configuración.
const DOC_EXTENSIONS: [&str; 5] = ["md", "toml", "yml", "yaml", "example"];

/// Archivos de la raíz que forman parte de lo publicado.
///
/// Lista explícita en vez de recorrer `.`: la raíz mezcla archivos trackeados con
/// locales gitignored que sí contienen infraestructura interna.
const ROOT_DOCS: [&str; 5] = [
    "README.md",
    "CHANGELOG.md",
    "Cargo.toml",
    "deny.toml",
    "rustfmt.toml",
];

/// Directorios que nunca se recorren: artefactos de build, metadatos de git y
/// salidas generadas.
const SKIP_DIRS: [&str; 4] = ["target", ".git", "graphify-out", "node_modules"];

/// Un hallazgo: ruta, línea (1-based) y la categoría que disparó.
type Hit = (String, usize, &'static str);

/// `true` si `tok` es una IPv4 en rango privado (RFC 1918).
///
/// Se implementa a mano en vez de con `regex` para no agregar una dependencia
/// (§0.2/B14: preferir la librería estándar) por un patrón de esta simplicidad.
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

/// `true` si la línea contiene una IPv4 privada.
///
/// Trocea por cualquier carácter que no sea dígito o punto, de modo que
/// `http://192.168.0.30:8080` produzca el token `192.168.0.30`.
fn has_private_ipv4(line: &str) -> bool {
    line.split(|c: char| !c.is_ascii_digit() && c != '.')
        .any(is_private_ipv4)
}

/// `true` si la línea contiene una ruta absoluta al home de un usuario concreto.
///
/// Exige un carácter alfanumérico después del prefijo para no marcar la mención
/// genérica de `/home/` o `C:\Users\` sin nombre detrás.
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

/// `true` si la línea contiene un token `sk-…` con material de clave real detrás.
///
/// El umbral de 16 caracteres distingue una clave de una mención en prosa
/// (`sk-ant-api...`, `sk-<tu-clave>`), que es el falso positivo frecuente.
fn has_key_like_token(line: &str) -> bool {
    line.match_indices("sk-").any(|(idx, _)| {
        line[idx + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .count()
            >= 16
    })
}

/// Clasifica una línea, devolviendo la categoría del primer patrón que dispare.
///
/// `strict` activa los controles que solo aplican a documentación (IP privada,
/// ruta absoluta). Devuelve `None` si la línea lleva [`ALLOW_MARKER`].
fn classify(line: &str, strict: bool) -> Option<&'static str> {
    if line.contains(ALLOW_MARKER) {
        return None;
    }
    // The two `if` lines below are the PATTERN definitions themselves, not leaked values — with
    // the F13 extension of this scan to tests/**/*.rs, this file's own detection logic would
    // otherwise flag itself; each carries its own trailing marker because the scanner
    // classifies line by line, so a marker on this comment would not reach them.
    if line.contains("sk-ant-api" /* allow-secret-scan */) {
        return Some("clave Anthropic");
    }
    if line.contains("-----BEGIN" /* allow-secret-scan */) {
        return Some("bloque PEM");
    }
    // Solo en documentación: `src/` está lleno de fixtures sintéticos legítimos
    // (`sk-super-secret-PROBE-value`, `sk-proj-OPENAISECRET`) que existen para
    // probar que la redacción funciona. Marcarlos uno por uno sería ruido, y
    // ampliar el escaneo de fuentes no es el alcance de este gate.
    if strict && has_key_like_token(line) {
        return Some("token tipo-clave");
    }
    if strict && has_private_ipv4(line) {
        return Some("IP privada");
    }
    if strict && has_absolute_home_path(line) {
        return Some("ruta absoluta de usuario");
    }
    None
}

/// Recorre `dir` acumulando hallazgos sobre los archivos cuya extensión esté en
/// `extensions`.
///
/// `skip_line_comments` ignora las líneas que empiezan con `//`, necesario en
/// fuentes Rust donde la documentación interna nombra los patrones. `strict`
/// se pasa tal cual a [`classify`].
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

/// Formatea los hallazgos para el mensaje de fallo, una línea por hallazgo.
fn render(hits: &[Hit]) -> String {
    hits.iter()
        .map(|(f, l, k)| format!("\n  {f}:{l}  [{k}]"))
        .collect()
}

/// Ningún `.rs` bajo `src/` lleva material tipo-clave. Comportamiento histórico.
#[test]
fn test_no_hardcoded_secrets_in_source_tree() {
    let mut hits = Vec::new();
    scan(Path::new("src"), &["rs"], true, false, &mut hits);
    assert!(
        hits.is_empty(),
        "posible secreto hardcodeado en:{}",
        render(&hits)
    );
}

/// Ninguna doc o config lleva claves, IPs privadas ni rutas absolutas de usuario.
///
/// El README y `docs/` son la superficie que más fácil filtra infraestructura
/// interna, y quedaban completamente fuera del escaneo.
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
    // La raíz va por lista explícita, NO recorriendo el directorio: ahí conviven
    // archivos trackeados con locales gitignored (`CLAUDE.local.md`, `magi.toml`)
    // que legítimamente contienen IPs y rutas internas y que nunca se publican.
    // Enumerar lo que sí ships evita depender de git desde el test y deja el
    // alcance visible en el código.
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
        "posible fuga en documentación:{}",
        render(&hits)
    );
}

/// El detector NO es ciego: cada forma de fuga conocida dispara su categoría.
///
/// Sin esta prueba, los dos tests de arriba podrían pasar por no encontrar nada
/// que mirar, que es indistinguible de pasar por estar limpio.
#[test]
fn test_detector_catches_known_leak_shapes() {
    // Fixtures below are deliberately shaped like each leak category, not real secrets — F13
    // extended this scan to tests/**/*.rs, so the two entries the scan itself would trip on
    // carry the same allowance `with-credentials.toml` already uses (trailing per line: the
    // scanner classifies line by line).
    let cases: [(&str, &str); 6] = [
        (
            "api_key = \"sk-ant-api03-REDACTED\"", /* allow-secret-scan */
            "clave Anthropic",
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----", /* allow-secret-scan */
            "bloque PEM",
        ),
        ("token: sk-abcdefghijklmnopqrstuvwxyz", "token tipo-clave"),
        ("host = \"192.168.0.30\"", "IP privada"),
        ("remote: 10.1.2.3:22", "IP privada"),
        (
            "path = \"/home/jdoe/.ssh/id_rsa\"",
            "ruta absoluta de usuario",
        ),
    ];
    for (line, expected) in cases {
        assert_eq!(
            classify(line, true),
            Some(expected),
            "no detectó la fuga en: {line}"
        );
    }
}

/// Prosa legítima y placeholders no disparan, o el gate se vuelve inusable.
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
        assert_eq!(classify(line, true), None, "falso positivo en: {line}");
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
        "posible secreto hardcodeado en tests/**/*.rs:{}",
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
        // Matches the unconditional "clave Anthropic" pattern, not the `strict`-gated
        // key-like-token one: the real coverage this proves reuses `strict = false` (same
        // non-strict pass `src/**/*.rs` gets), so a `strict`-only pattern would prove nothing.
        "let leaked = \"sk-ant-api-planted-for-this-test\";\n", // allow-secret-scan: fixture
    )
    .expect("write planted fixture");
    let mut hits = Vec::new();
    scan(dir.path(), &["rs"], true, false, &mut hits);
    assert!(
        !hits.is_empty(),
        "el escaneo no detectó el leak plantado bajo un árbol .rs"
    );
}

/// El marcador exime exactamente su línea, y solo la suya.
#[test]
fn test_allow_marker_exempts_only_its_own_line() {
    let leak = "key = \"sk-ant-api03-REDACTED\""; // allow-secret-scan: fixture, not a real key
    assert!(
        classify(leak, true).is_some(),
        "la línea sin marcador debe disparar"
    );
    let marked = format!("{leak} // {ALLOW_MARKER}");
    assert_eq!(classify(&marked, true), None, "el marcador debe eximirla");
}
