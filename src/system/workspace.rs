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
use std::io::Write;
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

/// Prefijo del directorio temporal hermano del `init` atómico en Linux
/// (`.magi.tmp.<rand>`), renombrado no-reemplazante sobre el `.magi/` final.
// Narrow allow: only the Linux `renameat2` path (`place_magi_dir`) consumes this;
// on other platforms the mkdir-gate is used and the constant is unreferenced.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const TMP_DIR_PREFIX: &str = ".magi.tmp.";

/// Modo restrictivo de directorio (`0700`): rwx solo para el dueño (REQ-H38, unix).
#[cfg(unix)]
const RESTRICTIVE_DIR_MODE: u32 = 0o700;

/// Modo restrictivo de archivo (`0600`): rw solo para el dueño (REQ-H38, unix).
#[cfg(unix)]
const RESTRICTIVE_FILE_MODE: u32 = 0o600;

/// Máscara de acceso `GENERIC_ALL` de Windows — control total para el usuario
/// al que se restringe la ACL (REQ-H38, Windows).
#[cfg(windows)]
const WINDOWS_FULL_CONTROL_MASK: u32 = 0x1000_0000;

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
    // Narrow allow: consumed by the MS2 headless wiring (config load); `db_path`
    // is already used by `magi init` (T11) but this accessor is not yet.
    #[allow(dead_code)]
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.magi_dir.join(CONFIG_FILE_NAME)
    }

    /// Ruta del subdirectorio de logs (`.magi/logs`).
    // Narrow allow: consumed by the MS2 headless log writer; not yet used in
    // production (T11 only needs `db_path`).
    #[allow(dead_code)]
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
/// [`std::path::absolute`] **sin** resolver `..`; (2) se valida que **ningún**
/// componente `Normal` de esa ruta **cruda** sea un symlink, recorriéndola de
/// izquierda a derecha y resolviendo `..` léxicamente por *pop* del prefijo —
/// así un componente symlink se rechaza **antes** de que un `..` posterior lo
/// borre léxicamente (`<root>/link/../sub` normalizaría a `<root>/sub`,
/// ocultando el `link` symlink si el chequeo corriera tras normalizar); (3)
/// recién entonces se normaliza **léxicamente** (`lexical_normalize`, ya
/// garantizada libre de symlinks) y se sube componente a componente,
/// deteniéndose en el **límite de sistema de archivos**, buscando un `.magi`
/// que sea un directorio (y no un symlink). El resultado es la ruta absoluta ya
/// validada, **sin re-canonicalizar** (anti-TOCTOU: una segunda resolución del
/// fs reabriría la ventana check→use).
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
    // Symlink check on the RAW absolute path BEFORE lexical `..` resolution, so a
    // symlinked component is caught at its own depth even when a later `..` would
    // lexically erase it (`<root>/link/../sub`).
    ensure_raw_chain_symlink_free(&absolute)?;
    let start_norm = lexical_normalize(&absolute);

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
// Narrow allow: `init`/`discover` are now used by `magi init` (main.rs, MS1 T11),
// but this legacy-file primitive's only consumer is MS2 T7 (startup warning
// emission) — kept as a scoped allow rather than a module-wide one until then.
#[allow(dead_code)]
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

/// Valida que **ningún** componente `Normal` de la ruta **cruda** `absolute`
/// sea un symlink, recorriéndola de izquierda a derecha y resolviendo `..`
/// léxicamente por *pop* del prefijo acumulado.
///
/// Correr sobre la ruta **cruda** (antes de [`lexical_normalize`]) es lo que
/// cierra el bypass `..`-a-través-de-symlink: cada componente `Normal` se
/// `symlink_metadata`-prueba en el instante en que se *empuja* al prefijo —a su
/// propia profundidad—, así un symlink se rechaza **antes** de que un `..`
/// posterior lo borre léxicamente (`<root>/link/../sub`). Un
/// [`Component::ParentDir`] hace *pop* de un componente que ya se validó al
/// empujarlo, de modo que un `..` legítimo en la ruta del operador sigue
/// funcionando; [`Component::Prefix`]/[`Component::RootDir`] anclan el recorrido
/// y [`Component::CurDir`] se ignora.
///
/// # Complejidad
/// `O(d)` con `d` = número de componentes de `absolute` (un `symlink_metadata`
/// por componente `Normal`).
///
/// # Errors
/// Devuelve [`HeadlessError::InputInvalid`] si algún componente `Normal` es un
/// symlink, o [`HeadlessError::Io`] ante un error de E/S distinto de "no existe".
fn ensure_raw_chain_symlink_free(absolute: &Path) -> Result<(), HeadlessError> {
    let mut prefix = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // The popped component was already symlink-checked when pushed.
                prefix.pop();
            }
            Component::Prefix(_) | Component::RootDir => {
                prefix.push(component.as_os_str());
            }
            Component::Normal(name) => {
                prefix.push(name);
                match fs::symlink_metadata(&prefix) {
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

/// Scaffolds a fresh `.magi/` state directory under `cwd` and returns the
/// resulting [`Workspace`] (REQ-H01/H38/H41).
///
/// Creates `cwd/.magi/` holding `magi.toml` (rendered defaults), an empty
/// `logs/` subdirectory, and the encrypted-store database with **all five
/// tables** created empty and **no envelope** (the first real open bootstraps
/// it, MS1 Task 3). The directory is placed **atomically and no-replace**: on
/// Linux via `renameat2(RENAME_NOREPLACE)` of a sibling temp dir, elsewhere via
/// a `create_dir` mkdir-gate — both refuse (never overwrite) if `.magi/`
/// already exists. Every created object is restricted to the current user
/// (`0700`/`0600` on unix, an ACL restricted to the current user on Windows).
///
/// # Errors
/// - [`HeadlessError::Aborted`] if `cwd/.magi/` already exists.
/// - [`HeadlessError::Io`] on a filesystem or ACL error (bad parent, rename).
/// - [`HeadlessError::Storage`] if the database schema cannot be created.
pub fn init(cwd: &Path) -> Result<Workspace, HeadlessError> {
    let absolute = std::path::absolute(cwd).map_err(|e| HeadlessError::Io(e.to_string()))?;
    let root = lexical_normalize(&absolute);
    let magi_dir = root.join(MAGI_DIR_NAME);
    place_magi_dir(&magi_dir)?;
    Ok(Workspace { root, magi_dir })
}

/// Places a populated `.magi/` at `magi_dir` atomically and no-replace via
/// `renameat2(RENAME_NOREPLACE)` of a sibling temp directory (Linux).
///
/// Builds the whole tree in `.magi.tmp.<rand>` (never a half-populated `.magi/`
/// visible to a reader) and renames it into place; on a populate error removes
/// only its own freshly-created scaffold.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `magi_dir` exists; [`HeadlessError::Io`] /
/// [`HeadlessError::Storage`] on a filesystem or schema error.
#[cfg(target_os = "linux")]
fn place_magi_dir(magi_dir: &Path) -> Result<(), HeadlessError> {
    let parent = magi_dir
        .parent()
        .ok_or_else(|| HeadlessError::Io("target .magi has no parent directory".to_owned()))?;
    let tmp = parent.join(format!("{TMP_DIR_PREFIX}{:016x}", rand::random::<u64>()));
    create_gate_dir(&tmp)?;
    populate_or_cleanup(&tmp)?;
    rename_no_replace(&tmp, magi_dir)
}

/// Places a populated `.magi/` at `magi_dir` via a `create_dir` mkdir-gate
/// (macOS, other unix, Windows) — `create_dir` is itself atomic no-replace.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `magi_dir` exists; [`HeadlessError::Io`] /
/// [`HeadlessError::Storage`] on a filesystem or schema error.
#[cfg(not(target_os = "linux"))]
fn place_magi_dir(magi_dir: &Path) -> Result<(), HeadlessError> {
    place_via_mkdir_gate(magi_dir)
}

/// Creates `magi_dir` in place as the atomic no-replace gate and populates it;
/// on a population error removes only the just-created scaffold (no user data).
///
/// # Errors
/// [`HeadlessError::Aborted`] if `magi_dir` exists; [`HeadlessError::Io`] /
/// [`HeadlessError::Storage`] on a filesystem or schema error.
fn place_via_mkdir_gate(magi_dir: &Path) -> Result<(), HeadlessError> {
    create_gate_dir(magi_dir)?;
    populate_or_cleanup(magi_dir)
}

/// Populates the freshly-created, restricted `scaffold` directory and, on any
/// populate error, best-effort removes the scaffold `init` itself just created,
/// returning the **original** error (never the cleanup error).
///
/// Removing `init`'s own half-built scaffold does **not** violate never-delete
/// (REQ-H20/H41): the scaffold holds no user data yet — only `logs/`, a
/// defaults `magi.toml`, and an empty envelope-less DB. Leaving it orphaned
/// would make a later `init` refuse (the no-replace gate), so the cleanup keeps
/// a crashed/failed `init` retryable.
///
/// # Errors
/// The original [`HeadlessError`] from [`populate_in_place`] (I/O, storage, or a
/// pre-existing child), unchanged; the cleanup outcome is intentionally ignored.
fn populate_or_cleanup(scaffold: &Path) -> Result<(), HeadlessError> {
    populate_or_cleanup_with(scaffold, populate_in_place)
}

/// [`populate_or_cleanup`] with an injectable populate step, so the
/// failure-cleanup path is unit-testable without forcing a real populate error.
///
/// # Errors
/// The original error returned by `populate`, unchanged (cleanup errors are
/// swallowed).
fn populate_or_cleanup_with<F>(scaffold: &Path, populate: F) -> Result<(), HeadlessError>
where
    F: FnOnce(&Path) -> Result<(), HeadlessError>,
{
    match populate(scaffold) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best-effort: the scaffold contains NO user data (never-delete safe);
            // return the ORIGINAL error, not the cleanup outcome.
            let _ = fs::remove_dir_all(scaffold);
            Err(e)
        }
    }
}

/// Renames `tmp` onto `final_dir` atomically without replacing an existing
/// target (`renameat2(RENAME_NOREPLACE)`), falling back to the portable
/// mkdir-gate if the kernel/filesystem does not support the flag.
///
/// # Errors
/// [`HeadlessError::Aborted`] if `final_dir` already exists; [`HeadlessError::Io`]
/// / [`HeadlessError::Storage`] on any other error.
#[cfg(target_os = "linux")]
fn rename_no_replace(tmp: &Path, final_dir: &Path) -> Result<(), HeadlessError> {
    use rustix::fs::{renameat_with, RenameFlags, CWD};
    use rustix::io::Errno;

    match renameat_with(CWD, tmp, CWD, final_dir, RenameFlags::NOREPLACE) {
        Ok(()) => Ok(()),
        Err(errno) => {
            let _ = fs::remove_dir_all(tmp);
            if errno == Errno::EXIST {
                Err(HeadlessError::Aborted)
            } else if errno == Errno::NOSYS || errno == Errno::INVAL || errno == Errno::OPNOTSUPP {
                // RENAME_NOREPLACE unsupported here → portable mkdir-gate fallback.
                place_via_mkdir_gate(final_dir)
            } else {
                Err(HeadlessError::Io(format!("rename failed: {errno:?}")))
            }
        }
    }
}

/// Creates a directory as an atomic no-replace gate with restrictive permissions
/// from creation (`0700` on unix, a current-user ACL on Windows).
///
/// # Errors
/// [`HeadlessError::Aborted`] if `path` already exists; [`HeadlessError::Io`] on
/// any other filesystem or ACL error.
fn create_gate_dir(path: &Path) -> Result<(), HeadlessError> {
    create_restricted_dir_impl(path).map_err(map_create_err)?;
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Maps a directory/file creation [`io::Error`] to a [`HeadlessError`], turning
/// an `AlreadyExists` into [`HeadlessError::Aborted`] (the no-replace refusal).
fn map_create_err(e: io::Error) -> HeadlessError {
    if e.kind() == io::ErrorKind::AlreadyExists {
        HeadlessError::Aborted
    } else {
        HeadlessError::Io(e.to_string())
    }
}

/// Creates a single directory restricted to the owner (`0700`) from creation.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(unix)]
fn create_restricted_dir_impl(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new()
        .mode(RESTRICTIVE_DIR_MODE)
        .create(path)
}

/// Creates a single directory (non-recursive, no-replace); permissions are
/// tightened separately per platform.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(not(unix))]
fn create_restricted_dir_impl(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

/// Populates an already-created, restricted `.magi/` directory with `logs/`,
/// `magi.toml` (rendered defaults) and the empty five-table database.
///
/// # Errors
/// [`HeadlessError::Io`] on a filesystem/ACL error, [`HeadlessError::Storage`] if
/// the schema cannot be created, or [`HeadlessError::Aborted`] on an unexpected
/// pre-existing child (should not occur in a fresh directory).
fn populate_in_place(dir: &Path) -> Result<(), HeadlessError> {
    create_gate_dir(&dir.join(LOGS_DIR_NAME))?;
    write_restricted_file(
        &dir.join(CONFIG_FILE_NAME),
        crate::defaults::render_default_magi_toml().as_bytes(),
    )?;
    create_db(&dir.join(DB_FILE_NAME))?;
    Ok(())
}

/// Writes `contents` to a new owner-restricted file (`0600` on unix, current-user
/// ACL on Windows), refusing to overwrite an existing file.
///
/// # Errors
/// [`HeadlessError::Aborted`] if the file already exists; [`HeadlessError::Io`] on
/// any other filesystem or ACL error.
fn write_restricted_file(path: &Path, contents: &[u8]) -> Result<(), HeadlessError> {
    let mut file = open_new_restricted(path).map_err(map_create_err)?;
    file.write_all(contents)
        .map_err(|e| HeadlessError::Io(e.to_string()))?;
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Creates and opens a new file restricted to the owner (`0600`) from creation,
/// failing if it already exists.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(unix)]
fn open_new_restricted(path: &Path) -> io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(RESTRICTIVE_FILE_MODE)
        .open(path)
}

/// Creates and opens a new file (no-replace); permissions are tightened
/// separately per platform.
///
/// # Errors
/// Propagates the underlying [`io::Error`] (incl. `AlreadyExists`).
#[cfg(not(unix))]
fn open_new_restricted(path: &Path) -> io::Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Creates the state database at `path` with all five schema tables empty (no
/// envelope), restricted to the owner.
///
/// Pre-creates the file with restrictive permissions so `rusqlite` never
/// materializes it world-readable under the umask; SQLite then treats the empty
/// file as a fresh database. No `PRAGMA` is set here — the first real open
/// (MS1 Task 3) configures WAL and bootstraps the envelope.
///
/// # Errors
/// [`HeadlessError::Aborted`] if the file already exists; [`HeadlessError::Io`]
/// on an ACL error; [`HeadlessError::Storage`] if the schema cannot be created.
fn create_db(path: &Path) -> Result<(), HeadlessError> {
    open_new_restricted(path).map_err(map_create_err)?;
    {
        let conn =
            rusqlite::Connection::open(path).map_err(|e| HeadlessError::Storage(e.to_string()))?;
        crate::system::database::init_schema(&conn)
            .map_err(|e| HeadlessError::Storage(e.to_string()))?;
    }
    #[cfg(windows)]
    restrict_to_current_user(path)?;
    Ok(())
}

/// Restricts `path`'s DACL to the current user only — the Windows equivalent of
/// unix `0700`/`0600` (REQ-H38) — using the safe `windows-acl` crate.
///
/// Grants the current user full control (which writes a PROTECTED DACL, severing
/// inheritance) then removes every other ACE, leaving exactly one allow entry.
///
/// # Errors
/// [`HeadlessError::Io`] if the path is not valid UTF-8 or any Win32 ACL call
/// fails (the numeric error code is included, never a secret).
#[cfg(windows)]
fn restrict_to_current_user(path: &Path) -> Result<(), HeadlessError> {
    use windows_acl::acl::{AceType, ACL};
    use windows_acl::helper::{current_user, name_to_sid, sid_to_string, string_to_sid};

    let path_str = path.to_str().ok_or_else(|| {
        HeadlessError::Io("path is not valid UTF-8 for ACL application".to_owned())
    })?;
    let user = current_user()
        .ok_or_else(|| HeadlessError::Io("cannot resolve current Windows user".to_owned()))?;
    let user_sid = name_to_sid(&user, None)
        .map_err(|code| HeadlessError::Io(format!("name_to_sid failed (code {code})")))?;
    let user_string = sid_to_string(user_sid.as_ptr() as _)
        .map_err(|code| HeadlessError::Io(format!("sid_to_string failed (code {code})")))?;

    let mut acl = ACL::from_file_path(path_str, false)
        .map_err(|code| HeadlessError::Io(format!("read ACL failed (code {code})")))?;

    // Grant the current user full control; windows-acl writes a PROTECTED DACL,
    // severing inheritance so no parent ACE leaks in.
    acl.add_entry(
        user_sid.as_ptr() as _,
        AceType::AccessAllow,
        0,
        WINDOWS_FULL_CONTROL_MASK,
    )
    .map_err(|code| HeadlessError::Io(format!("grant user ACE failed (code {code})")))?;

    // Remove every ACE that is not the current user's, restricting access to
    // exactly this user (drops inherited SYSTEM/Administrators/Users entries).
    let entries = acl
        .all()
        .map_err(|code| HeadlessError::Io(format!("enumerate ACL failed (code {code})")))?;
    for entry in entries {
        if entry.string_sid == user_string {
            continue;
        }
        let sid = string_to_sid(&entry.string_sid)
            .map_err(|code| HeadlessError::Io(format!("string_to_sid failed (code {code})")))?;
        acl.remove(sid.as_ptr() as _, None, None)
            .map_err(|code| HeadlessError::Io(format!("remove ACE failed (code {code})")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        create_gate_dir, detect_legacy_files, discover, init, populate_or_cleanup_with, Workspace,
    };
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
    fn test_init_creates_structure_and_refuses_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).expect("init");
        assert!(ws.magi_dir.join("magi.toml").exists());
        assert!(ws.magi_dir.join("logs").is_dir());
        assert!(ws.db_path().exists());
        // A second init must refuse (never overwrite) — atomic no-replace gate.
        assert!(matches!(init(tmp.path()), Err(HeadlessError::Aborted)));
    }

    #[test]
    #[cfg(unix)]
    fn test_init_sets_restrictive_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();
        let dir_mode = std::fs::metadata(&ws.magi_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let db_mode = std::fs::metadata(ws.db_path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
        assert_eq!(db_mode, 0o600);
    }

    #[test]
    #[cfg(windows)]
    fn test_init_restricts_acl_to_current_user() {
        use windows_acl::acl::ACL;
        use windows_acl::helper::{current_user, name_to_sid, sid_to_string};

        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();

        let user = current_user().unwrap();
        let user_sid = name_to_sid(&user, None).unwrap();
        let user_string = sid_to_string(user_sid.as_ptr() as _).unwrap();

        let acl = ACL::from_file_path(ws.magi_dir.to_str().unwrap(), false).unwrap();
        let entries = acl.all().unwrap();
        assert!(!entries.is_empty(), "the DACL must contain the user's ACE");
        for entry in entries {
            assert_eq!(
                entry.string_sid, user_string,
                "only the current user may hold an ACE on .magi/"
            );
        }
        // Owner access is retained: the DB the process just wrote is still there.
        assert!(ws.db_path().exists());
    }

    #[test]
    fn test_orphan_tmp_dir_does_not_break_a_later_init() {
        let tmp = tempfile::tempdir().unwrap();
        // Simulate a crashed prior run that left a stray sibling temp dir behind.
        std::fs::create_dir(tmp.path().join(".magi.tmp.deadbeef")).unwrap();

        let ws = init(tmp.path()).expect("init succeeds despite the orphan tmp");
        assert!(ws.magi_dir.is_dir());
        assert!(ws.db_path().exists());
        // The `.magi/` is complete, not half-populated.
        assert!(ws.magi_dir.join("magi.toml").exists());
    }

    #[test]
    fn test_populate_failure_cleans_up_scaffold_and_allows_retry() {
        // REQ-H41 / Fix: a populate error AFTER the scaffold dir is created must
        // remove only that just-built scaffold (no user data yet), return the
        // ORIGINAL error, and leave no orphan that a retry would refuse.
        let tmp = tempfile::tempdir().unwrap();
        let scaffold = tmp.path().join(".magi");
        create_gate_dir(&scaffold).expect("gate dir created");
        assert!(
            scaffold.is_dir(),
            "the scaffold exists before populate runs"
        );

        let err = populate_or_cleanup_with(&scaffold, |_| {
            Err(HeadlessError::Storage(
                "injected populate failure".to_owned(),
            ))
        })
        .expect_err("the injected populate failure must propagate");
        assert!(
            matches!(err, HeadlessError::Storage(_)),
            "the ORIGINAL populate error is returned, not the cleanup outcome"
        );
        assert!(
            !scaffold.exists(),
            "the half-built scaffold must be removed on populate failure"
        );

        // The cleaned-up failure does not block a subsequent real init.
        let ws = init(tmp.path()).expect("init proceeds after the cleaned-up failure");
        assert!(ws.db_path().exists());
        assert!(ws.magi_dir.join("magi.toml").exists());
    }

    // T2↔T3 lock-in (MS1 Task 2 Step 4c-bis / Task 3 Step 9b): a freshly-`init`ed
    // DB has exactly the five empty tables and NO envelope row — the precondition
    // under which Task 3's never-delete state machine bootstraps cleanly (never
    // `DbCorrupt`). Now that `open_with_state_machine` exists (T3), the final
    // assertion drives it directly, closing the T2↔T3 coupling executably.
    #[test]
    fn test_fresh_init_db_bootstraps_cleanly_under_state_machine() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = init(tmp.path()).unwrap();
        // Structural precondition: five empty tables, no envelope.
        let conn = rusqlite::Connection::open(ws.db_path()).unwrap();
        for table in [
            "sessions",
            "messages",
            "knowledge",
            "memories",
            "vault_meta",
        ] {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                .unwrap_or_else(|_| panic!("table `{table}` must exist on fresh init"));
            assert_eq!(count, 0, "table `{table}` must be empty on fresh init");
        }
        let has_envelope: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vault_meta WHERE key = 'wrapped_dek'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_envelope, 0, "a fresh init has no envelope yet");
        drop(conn);

        // T3 lock-in: the state machine opens the fresh DB cleanly (bootstraps the
        // envelope), NEVER `DbCorrupt`.
        match crate::system::database::EncryptedSqliteMemory::open_with_state_machine(
            ws.db_path(),
            zeroize::Zeroizing::new("fresh-init-state-machine-master".to_string()),
        ) {
            Ok(_) => {}
            Err(e) => panic!("fresh init must bootstrap cleanly under the state machine: {e:?}"),
        }
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

    #[test]
    #[cfg(unix)]
    fn test_discover_rejects_parentdir_through_symlink_component() {
        // The `..`-through-symlink bypass: a start path that traverses a symlink
        // and then `..` back out. Lexical normalization would rewrite
        // `<root>/link/../real/sub` to `<root>/real/sub`, erasing the symlinked
        // `link` before it is ever checked. The raw-component check catches `link`
        // at its own depth, BEFORE the `..` pops it.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let real = root.join("real");
        std::fs::create_dir_all(real.join(".magi")).unwrap();
        std::fs::create_dir_all(real.join("sub")).unwrap();
        std::os::unix::fs::symlink(&real, root.join("link")).unwrap();

        let start = root.join("link").join("..").join("real").join("sub");
        assert!(
            matches!(discover(&start), Err(HeadlessError::InputInvalid(_))),
            "a `..` that first traverses a symlinked component must be rejected"
        );
    }

    #[test]
    fn test_discover_allows_parentdir_on_non_symlinked_path() {
        // A legitimate `..` on a fully non-symlinked path still resolves and finds
        // the ancestor `.magi/`: `<root>/a/b/../c` normalizes to `<root>/a/c`, and
        // the walk-up discovers `<root>/a/.magi`.
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("a").join(".magi")).unwrap();
        std::fs::create_dir_all(root.join("a").join("b")).unwrap();
        std::fs::create_dir_all(root.join("a").join("c")).unwrap();

        let start = root.join("a").join("b").join("..").join("c");
        let ws = discover(&start).unwrap().expect("found");
        assert_eq!(ws.magi_dir, root.join("a").join(".magi"));
        assert_eq!(ws.root, root.join("a"));
    }
}
