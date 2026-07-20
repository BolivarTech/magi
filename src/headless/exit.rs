// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Mapeo de un desenlace headless a un exit code accionable (REQ-H23/H23b).
//!
//! [`exit_code`] es una función **pura y total**: no hay rama que no produzca
//! un código, y el `match` interno sobre [`HeadlessError`] es **exhaustivo sin
//! comodín `_`** — una variante nueva de la fuente rompe el build en lugar de
//! degradar en silencio a un exit code por default (mismo patrón que
//! `HeadlessError::from(VaultError)`, T0).
//!
//! **Precedencia** (REQ-H23, "cuando co-ocurren varias condiciones"): un error
//! tipado presente domina siempre — `InputInvalid`/`InputTooLarge` ⇒
//! `EXIT_MISUSE`, cualquier otra variante ⇒ `EXIT_RUNTIME`. Solo en
//! **ausencia** de error se evalúa el criterio determinístico de exit 3
//! (REQ-H23b): el tier denegó al menos un tool **y** el turno final del
//! agente no produjo respuesta (`response_empty`) **y** el `stop_reason`
//! resultante es [`StopReason::Denied`] — las tres señales coinciden por
//! construcción (`Denied` se asigna precisamente bajo esa condición), pero se
//! verifican las tres para no depender de un único canal. Cualquier otro caso
//! es éxito (`EXIT_OK`).
//!
//! `exit_code` es `pub`: el runner de MS2 vive en el crate del binario y
//! solo puede alcanzar API `pub` de la lib.

use super::types::StopReason;
use super::HeadlessError;

/// Exit code de una corrida exitosa (REQ-H23).
const EXIT_OK: i32 = 0;

/// Exit code de un error de runtime/agente: DB corrupta, passphrase
/// incorrecta o no disponible, I/O, storage, cancelación, timeout (REQ-H23).
const EXIT_RUNTIME: i32 = 1;

/// Exit code de mal uso de CLI o input inválido: envelope sin `prompt`,
/// input que excede el cap de tamaño, formato inválido (REQ-H23).
const EXIT_MISUSE: i32 = 2;

/// Exit code cuando el tier de autorización bloqueó la tarea: al menos un
/// tool esencial fue denegado y el agente no produjo respuesta (REQ-H23b).
const EXIT_TIER_DENIED: i32 = 3;

/// Calcula el exit code de una corrida headless según su desenlace.
///
/// # Parámetros
/// - `err`: error tipado de la corrida, si la hubo; su presencia domina la
///   precedencia (ver el rustdoc del módulo).
/// - `stop_reason`: motivo con el que terminó el loop del agente.
/// - `response_empty`: si el turno final del agente no produjo ningún bloque
///   de texto (REQ-H23b — "vacío" es *cero* `TextDelta`, no solo whitespace;
///   esa distinción la resuelve el llamador antes de invocar esta función).
/// - `tier_denied`: si al menos un tool fue denegado por el tier durante la
///   corrida (independientemente de si el agente logró sortear la denegación).
///
/// Devuelve uno de `EXIT_OK`, `EXIT_RUNTIME`, `EXIT_MISUSE` o
/// `EXIT_TIER_DENIED`.
pub fn exit_code(
    err: Option<&HeadlessError>,
    stop_reason: StopReason,
    response_empty: bool,
    tier_denied: bool,
) -> i32 {
    if let Some(e) = err {
        return exit_code_for_error(e);
    }
    if tier_denied && response_empty && stop_reason == StopReason::Denied {
        return EXIT_TIER_DENIED;
    }
    EXIT_OK
}

/// Traduce un [`HeadlessError`] concreto a su clase de exit code.
///
/// El `match` es **exhaustivo sin comodín `_`**: agregar una variante a
/// [`HeadlessError`] fuerza decidir aquí su mapeo en vez de heredar
/// silenciosamente [`EXIT_RUNTIME`].
fn exit_code_for_error(err: &HeadlessError) -> i32 {
    match err {
        HeadlessError::InputInvalid(_) | HeadlessError::InputTooLarge(_) => EXIT_MISUSE,
        HeadlessError::Io(_)
        | HeadlessError::Storage(_)
        | HeadlessError::Aborted
        | HeadlessError::PassphraseUnavailable
        | HeadlessError::Db(_) => EXIT_RUNTIME,
    }
}

#[cfg(test)]
mod tests {
    use super::{exit_code, EXIT_MISUSE, EXIT_OK, EXIT_RUNTIME, EXIT_TIER_DENIED};
    use crate::headless::types::StopReason;
    use crate::headless::HeadlessError;
    use crate::vault::VaultError;

    /// Éxito sin error, sin denegación ⇒ 0.
    #[test]
    fn test_exit_codes_taxonomy() {
        assert_eq!(exit_code(None, StopReason::Done, false, false), EXIT_OK);
        assert_eq!(
            exit_code(
                Some(&HeadlessError::InputInvalid("x".into())),
                StopReason::Error,
                true,
                false
            ),
            EXIT_MISUSE
        );
        // tier bloqueó la tarea: denegado + sin respuesta + stop_reason=denied.
        assert_eq!(
            exit_code(None, StopReason::Denied, true, true),
            EXIT_TIER_DENIED
        );
        // el agente sorteó la denegación: produjo respuesta ⇒ éxito.
        assert_eq!(exit_code(None, StopReason::Done, false, true), EXIT_OK);
    }

    /// `InputTooLarge` es mal-uso de CLI (input que excede el cap) ⇒ 2.
    #[test]
    fn test_input_too_large_maps_to_misuse() {
        let err = HeadlessError::InputTooLarge(10 * 1024 * 1024);
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_MISUSE
        );
    }

    /// Passphrase no disponible (fail-closed sin TTY) ⇒ error de runtime.
    #[test]
    fn test_passphrase_unavailable_maps_to_runtime_error() {
        let err = HeadlessError::PassphraseUnavailable;
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// DB corrupta (envuelta desde `VaultError`) ⇒ error de runtime.
    #[test]
    fn test_db_corrupt_maps_to_runtime_error() {
        let err: HeadlessError = VaultError::DbCorrupt {
            db_path: std::path::PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".into(),
        }
        .into();
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Wrong passphrase (envuelta desde `VaultError`) ⇒ error de runtime.
    #[test]
    fn test_wrong_passphrase_maps_to_runtime_error() {
        let err: HeadlessError = VaultError::WrongPassphrase.into();
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// I/O y storage son errores de runtime, no de mal-uso.
    #[test]
    fn test_io_and_storage_map_to_runtime_error() {
        let io = HeadlessError::Io("disk full".into());
        let storage = HeadlessError::Storage("locked".into());
        assert_eq!(
            exit_code(Some(&io), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
        assert_eq!(
            exit_code(Some(&storage), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Cancelación explícita (`Aborted`) es un error de runtime, no éxito.
    #[test]
    fn test_aborted_maps_to_runtime_error() {
        let err = HeadlessError::Aborted;
        assert_eq!(
            exit_code(Some(&err), StopReason::Error, true, false),
            EXIT_RUNTIME
        );
    }

    /// Un error tipado presente domina sobre la señal de tier-denied: incluso
    /// con `tier_denied=true` y `response_empty=true`, el error gana (2, no 3).
    #[test]
    fn test_error_precedence_wins_over_tier_denied_signal() {
        let err = HeadlessError::InputInvalid("bad envelope".into());
        assert_eq!(
            exit_code(Some(&err), StopReason::Denied, true, true),
            EXIT_MISUSE
        );
    }

    /// `tier_denied` sin `response_empty` nunca produce exit 3, aunque el
    /// `stop_reason` reportado sea `Denied` (inconsistencia defensiva).
    #[test]
    fn test_tier_denied_without_empty_response_is_success() {
        assert_eq!(exit_code(None, StopReason::Denied, false, true), EXIT_OK);
    }
}
