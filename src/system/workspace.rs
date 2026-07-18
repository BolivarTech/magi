// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Lints de panic/bounds-safety: SOLO en producción. Los tests usan
// `unwrap`/`expect`/indexing idiomáticamente (un fallo en un test ES el test
// fallando, que es el comportamiento correcto).
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]
// Scaffolding intencional del plan (MS1 T1): [`Workspace`]/[`discover`]/
// [`detect_legacy_files`] son la API pública de descubrimiento que consume el
// wiring de `magi init` en MS2/T11 (una tarea posterior del mismo plan). Los
// tests de este módulo ya ejercitan cada ítem; el warning solo aparece en
// builds sin `cfg(test)`. No es un símbolo huérfano fabricado para silenciar al
// linter — mismo patrón sancionado en `src/headless/types.rs` (T0).
#![allow(dead_code)]

//! Descubrimiento del directorio de estado unificado `.magi/` (REQ-H16/H30/H31).
//!
//! El estado del proyecto (config, DB cifrada, logs) vive bajo un único
//! directorio `.magi/`, descubierto por **walk-up** desde el working directory
//! al estilo de `.git`. Este módulo aporta:
//!
//! - [`Workspace`]: el `.magi/` descubierto y las rutas de sus artefactos.
//! - [`discover`]: el walk-up endurecido (rechazo de symlinks, límite de fs).
//! - [`detect_legacy_files`]: la **primitiva** de detección de archivos legacy
//!   sueltos en el cwd (la **emisión** del warning es MS2).

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use magi_rs::headless::HeadlessError;

/// Nombre del directorio de estado unificado que se busca en el walk-up.
const MAGI_DIR_NAME: &str = ".magi";

/// Nombre del archivo de base de datos cifrada dentro de `.magi/`.
const DB_FILE_NAME: &str = ".magi-rs-memory.db";

/// Nombre del archivo de configuración dentro de `.magi/`.
const CONFIG_FILE_NAME: &str = "magi.toml";

/// Nombre del subdirectorio de logs dentro de `.magi/`.
const LOGS_DIR_NAME: &str = "logs";

/// Mensaje de error cuando un componente de la ruta descubierta es un symlink.
const SYMLINK_COMPONENT_MSG: &str = "symlinked path component in .magi discovery";

/// El directorio de estado `.magi/` descubierto y las rutas de sus artefactos.
///
/// Se construye exclusivamente por [`discover`], que garantiza que ningún
/// componente de la ruta es un symlink y que `magi_dir` es un directorio real.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    /// Directorio que **contiene** el `.magi/` (su ancestro directo).
    pub root: PathBuf,
    /// Ruta absoluta y validada del directorio `.magi/`.
    pub magi_dir: PathBuf,
}

impl Workspace {
    /// Ruta de la base de datos cifrada (`.magi/.magi-rs-memory.db`).
    #[must_use]
    pub fn db_path(&self) -> PathBuf {
        self.magi_dir.join(DB_FILE_NAME)
    }

    /// Ruta del archivo de configuración (`.magi/magi.toml`).
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.magi_dir.join(CONFIG_FILE_NAME)
    }

    /// Ruta del subdirectorio de logs (`.magi/logs`).
    #[must_use]
    pub fn logs_dir(&self) -> PathBuf {
        self.magi_dir.join(LOGS_DIR_NAME)
    }
}

/// Descubre el `.magi/` del ancestro más cercano a `start`, con un walk-up
/// endurecido de un **solo** mecanismo (sin canonicalizar el `start`).
///
/// El walk es **crudo** (no resuelve symlinks del `start`, para poder
/// rechazarlos en vez de seguirlos): (1) `start` se vuelve absoluto con
/// [`std::path::absolute`] y se normaliza **léxicamente** (sin tocar el fs);
/// (2) se valida que **ningún** componente de esa ruta sea un symlink —
/// defensa real contra config-injection en el mismo filesystem; (3) se sube
/// componente a componente, deteniéndose en el **límite de sistema de
/// archivos**, buscando un `.magi` que sea un directorio (y no un symlink).
/// El resultado es la ruta absoluta ya validada, **sin re-canonicalizar**
/// (anti-TOCTOU: una segunda resolución del fs reabriría la ventana
/// check→use).
///
/// # Complejidad
/// `O(d)` accesos al fs, con `d` = profundidad de `start` (una llamada
/// `symlink_metadata` por componente y por candidato).
///
/// # Residual de plataforma (macOS)
/// La política estricta rechaza **cualquier** componente ancestro que sea un
/// symlink. En macOS `/tmp`→`/private/tmp`, `/var`, `/etc` son symlinks del
/// sistema operativo, por lo que descubrir directamente bajo `/tmp` falla; el
/// uso real (proyectos bajo `/Users/...`, que no es symlink) no se ve
/// afectado. Es un trade-off aceptado a favor de la seguridad, no se resuelve
/// relajando el rechazo.
///
/// # Errors
/// Devuelve [`HeadlessError::InputInvalid`] si algún componente de la ruta
/// (incluido el propio `.magi`) es un symlink, o [`HeadlessError::Io`] ante un
/// error de E/S al hacer absoluta la ruta o leer metadatos.
pub fn discover(start: &Path) -> Result<Option<Workspace>, HeadlessError> {
    let absolute = std::path::absolute(start).map_err(|e| HeadlessError::Io(e.to_string()))?;
    let start_norm = lexical_normalize(&absolute);
    ensure_ancestor_chain_symlink_free(&start_norm)?;

    for dir in collect_search_dirs(&start_norm)? {
        let candidate = dir.join(MAGI_DIR_NAME);
        match classify_magi_candidate(&candidate)? {
            MagiCandidate::Directory => {
                return Ok(Some(Workspace {
                    root: dir,
                    magi_dir: candidate,
                }));
            }
            MagiCandidate::Absent => {}
        }
    }
    Ok(None)
}

/// Detecta la presencia de archivos legacy sueltos en `cwd` — la **primitiva**
/// de REQ-H31 (la emisión del warning al usuario es responsabilidad de MS2).
///
/// Devuelve `true` sii **no** existe un `.magi/` en `cwd` **y** existe al menos
/// uno de los archivos del layout legacy (`.magi-rs-memory.db` o `magi.toml`)
/// suelto en `cwd`. Con un `.magi/` presente el layout ya está migrado y no hay
/// nada que advertir.
#[must_use]
pub fn detect_legacy_files(cwd: &Path) -> bool {
    !cwd.join(MAGI_DIR_NAME).is_dir()
        && (cwd.join(DB_FILE_NAME).exists() || cwd.join(CONFIG_FILE_NAME).exists())
}

/// Clasificación de un candidato `.magi` sin seguir su enlace final.
enum MagiCandidate {
    /// El candidato existe y es un directorio real (workspace válido).
    Directory,
    /// El candidato no existe o no es un directorio (se continúa el walk-up).
    Absent,
}

/// Clasifica un candidato `.magi` inspeccionando su metadato **sin seguir** el
/// enlace del componente final.
///
/// # Errors
/// Devuelve [`HeadlessError::InputInvalid`] si el candidato es un symlink, o
/// [`HeadlessError::Io`] ante un error de E/S distinto de "no existe".
fn classify_magi_candidate(candidate: &Path) -> Result<MagiCandidate, HeadlessError> {
    match fs::symlink_metadata(candidate) {
        Ok(md) => {
            let file_type = md.file_type();
            if file_type.is_symlink() {
                Err(HeadlessError::InputInvalid(
                    SYMLINK_COMPONENT_MSG.to_owned(),
                ))
            } else if file_type.is_dir() {
                Ok(MagiCandidate::Directory)
            } else {
                Ok(MagiCandidate::Absent)
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(MagiCandidate::Absent),
        Err(e) => Err(HeadlessError::Io(e.to_string())),
    }
}

/// Valida que **ningún** componente de `path` (de la raíz al final) sea un
/// symlink, recorriendo los prefijos de más corto a más largo.
///
/// El orden raíz→hoja es esencial: al llegar a cada prefijo, todos los
/// prefijos más cortos ya se confirmaron no-symlink, así que
/// `symlink_metadata` sigue sus componentes intermedios con seguridad (son
/// reales) y solo el **último** componente del prefijo queda bajo prueba —
/// exactamente el que aún no se validó.
///
/// # Complejidad
/// `O(d)` con `d` = número de componentes de `path`.
///
/// # Errors
/// Devuelve [`HeadlessError::InputInvalid`] si algún componente es un symlink,
/// o [`HeadlessError::Io`] ante un error de E/S distinto de "no existe".
fn ensure_ancestor_chain_symlink_free(path: &Path) -> Result<(), HeadlessError> {
    let mut prefixes: Vec<&Path> = path.ancestors().collect();
    prefixes.reverse();
    for prefix in prefixes {
        match fs::symlink_metadata(prefix) {
            Ok(md) if md.file_type().is_symlink() => {
                return Err(HeadlessError::InputInvalid(
                    SYMLINK_COMPONENT_MSG.to_owned(),
                ));
            }
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(HeadlessError::Io(e.to_string())),
        }
    }
    Ok(())
}

/// Reúne los directorios candidatos desde `start` hacia la raíz (más cercano
/// primero), deteniéndose en el **límite de sistema de archivos**.
///
/// # Complejidad
/// `O(d)` con `d` = profundidad de `start` (dos `symlink_metadata` por nivel
/// para el chequeo de límite de fs).
///
/// # Errors
/// Devuelve [`HeadlessError::Io`] si no puede leer los metadatos necesarios
/// para el chequeo de límite de sistema de archivos.
fn collect_search_dirs(start: &Path) -> Result<Vec<PathBuf>, HeadlessError> {
    let mut dirs = Vec::new();
    let mut current = start.to_path_buf();
    loop {
        dirs.push(current.clone());
        match current.parent() {
            Some(parent) => {
                let parent = parent.to_path_buf();
                if is_fs_boundary(&current, &parent)? {
                    break;
                }
                current = parent;
            }
            None => break,
        }
    }
    Ok(dirs)
}

/// Normaliza `path` **léxicamente** (sin tocar el fs): descarta los componentes
/// `.` y resuelve `..` puramente sobre la cadena de componentes.
///
/// Resolver `..` de forma léxica es seguro aquí porque [`discover`] rechaza
/// después cualquier componente symlink de la cadena resultante.
///
/// # Complejidad
/// `O(d)` con `d` = número de componentes de `path`.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Indica si `dir` es la raíz de su sistema de archivos respecto a `parent`
/// (POSIX): compara el número de dispositivo de ambos.
///
/// # Errors
/// Devuelve [`HeadlessError::Io`] si no puede leer los metadatos de `dir` o
/// `parent`.
#[cfg(unix)]
fn is_fs_boundary(dir: &Path, parent: &Path) -> Result<bool, HeadlessError> {
    use std::os::unix::fs::MetadataExt;

    let dir_dev = fs::symlink_metadata(dir)
        .map_err(|e| HeadlessError::Io(e.to_string()))?
        .dev();
    let parent_dev = fs::symlink_metadata(parent)
        .map_err(|e| HeadlessError::Io(e.to_string()))?
        .dev();
    Ok(dir_dev != parent_dev)
}

/// Indica si `dir` cruza un límite de volumen respecto a `parent` (Windows):
/// compara la raíz de volumen **léxicamente** vía [`Component::Prefix`], sin
/// syscall crudo (respeta `#![forbid(unsafe_code)]`).
///
/// # Errors
/// Nunca falla; la firma `Result` unifica con la variante POSIX.
#[cfg(windows)]
fn is_fs_boundary(dir: &Path, parent: &Path) -> Result<bool, HeadlessError> {
    Ok(volume_prefix(dir) != volume_prefix(parent))
}

/// Extrae la raíz de volumen léxica de `path` (drive `C:` o UNC
/// `\\server\share`), o `None` si la ruta no tiene prefijo.
#[cfg(windows)]
fn volume_prefix(path: &Path) -> Option<std::ffi::OsString> {
    path.components().find_map(|component| match component {
        Component::Prefix(prefix) => Some(prefix.as_os_str().to_os_string()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::{detect_legacy_files, discover, Workspace};
    // Solo los tests de rechazo de symlink (cfg(unix)) inspeccionan la variante.
    #[cfg(unix)]
    use magi_rs::headless::HeadlessError;
    use std::path::PathBuf;

    #[test]
    fn test_discover_finds_nearest_ancestor_magi_dir() {
        // Raíz canónica UNA vez (resuelve /tmp→/private/tmp en macOS ANTES del
        // walk); discover NO canonicaliza. El guard `tmp` se mantiene vivo.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("a/b/.magi")).unwrap();
        let sub = root.join("a/b/c/d");
        std::fs::create_dir_all(&sub).unwrap();

        let ws = discover(&sub).unwrap().expect("found");

        assert_eq!(ws.magi_dir, root.join("a/b/.magi"));
        assert_eq!(ws.root, root.join("a/b"));
    }

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_symlinked_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("elsewhere");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join(".magi")).unwrap();

        assert!(matches!(
            discover(&root),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_symlinked_ancestor_component() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("real");
        std::fs::create_dir_all(real.join(".magi")).unwrap();
        let link = root.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let sub = link.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        // El ancestro `link` es un symlink ⇒ rechazo estricto.
        assert!(matches!(
            discover(&sub),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    #[test]
    fn test_discover_returns_none_when_no_magi_dir_in_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let sub = root.join("x/y/z");
        std::fs::create_dir_all(&sub).unwrap();

        assert_eq!(discover(&sub).unwrap(), None);
    }

    #[test]
    fn test_workspace_path_helpers_are_under_magi_dir() {
        let magi_dir = PathBuf::from("/proj/.magi");
        let ws = Workspace {
            root: PathBuf::from("/proj"),
            magi_dir: magi_dir.clone(),
        };

        assert_eq!(ws.db_path(), magi_dir.join(".magi-rs-memory.db"));
        assert_eq!(ws.config_path(), magi_dir.join("magi.toml"));
        assert_eq!(ws.logs_dir(), magi_dir.join("logs"));
    }

    #[test]
    fn test_detect_legacy_files_true_for_loose_db_without_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".magi-rs-memory.db"), b"x").unwrap();

        assert!(detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_true_for_loose_config_without_magi_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("magi.toml"), b"x").unwrap();

        assert!(detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_false_when_magi_dir_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".magi")).unwrap();
        // Un archivo legacy suelto se ignora si ya existe `.magi/`.
        std::fs::write(tmp.path().join(".magi-rs-memory.db"), b"x").unwrap();

        assert!(!detect_legacy_files(tmp.path()));
    }

    #[test]
    fn test_detect_legacy_files_false_when_directory_is_clean() {
        let tmp = tempfile::tempdir().unwrap();

        assert!(!detect_legacy_files(tmp.path()));
    }
}
