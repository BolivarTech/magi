// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Lectura acotada de la entrada headless (`-i <file>` / stdin, REQ-H03/H29).
//!
//! [`read_input_bounded`] es el único punto de entrada de bytes no confiables
//! hacia el resto del subsistema `headless`: nunca bufferiza una fuente hostil
//! sin límite (un pipe/`-i` de tamaño ilimitado), por lo que el parser del
//! envelope (tarea posterior) siempre recibe, como mucho, `MAX_INPUT_BYTES + 1`
//! bytes en memoria.

use std::io::Read;

use super::limits::MAX_INPUT_BYTES;
use super::HeadlessError;

/// Lee `reader` hasta EOF, acotado a `MAX_INPUT_BYTES` (REQ-H29, anti-DoS).
///
/// Usa `reader.take(MAX_INPUT_BYTES as u64 + 1)`: el `+1` es lo que permite
/// distinguir "la fuente tenía exactamente el cap" de "la fuente excedía el
/// cap" sin necesidad de leer más allá — una fuente hostil e ilimitada (p.ej.
/// `std::io::repeat`) nunca se bufferiza por completo, porque `take` corta la
/// lectura en `cap + 1` bytes pase lo que pase aguas arriba.
///
/// Complejidad `O(n)` en el tamaño de la entrada, acotada por `cap + 1`
/// (`MAX_INPUT_BYTES + 1`) sin importar cuánto produzca `reader`.
///
/// # Errors
///
/// Devuelve [`HeadlessError::InputTooLarge`] con el límite configurado si el
/// contenido leído excede `MAX_INPUT_BYTES`. Devuelve [`HeadlessError::Io`] si
/// el `reader` subyacente falla durante la lectura (p.ej. un error real de E/S
/// de stdin o de un archivo); ese caso se propaga tal cual, sin exponer nunca
/// contenido de la entrada en el mensaje de error.
pub fn read_input_bounded(reader: impl Read) -> Result<Vec<u8>, HeadlessError> {
    let mut buf = Vec::new();
    reader
        .take(MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| HeadlessError::Io(e.to_string()))?;

    if buf.len() > MAX_INPUT_BYTES {
        return Err(HeadlessError::InputTooLarge(MAX_INPUT_BYTES));
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// Una fuente ilimitada (`io::repeat`) nunca se bufferiza por completo:
    /// `take(cap+1)` la corta y el resultado es `InputTooLarge`, no un hang/OOM.
    #[test]
    fn test_read_input_rejects_oversized_without_buffering_all() {
        let r = std::io::repeat(b'a');
        assert!(matches!(
            read_input_bounded(r),
            Err(HeadlessError::InputTooLarge(_))
        ));
    }

    /// Entrada vacía es válida a este nivel: el parser de envelope (tarea
    /// posterior) es quien decide si un prompt vacío es un error de input.
    #[test]
    fn test_read_input_empty_reader_returns_empty_vec() {
        let r = Cursor::new(Vec::new());
        assert_eq!(read_input_bounded(r).unwrap(), Vec::<u8>::new());
    }

    /// Caso borde exacto: `MAX_INPUT_BYTES` bytes caben justo bajo el cap.
    #[test]
    fn test_read_input_accepts_exactly_max_input_bytes() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64);
        let out = read_input_bounded(r).expect("exactly the cap must be accepted");
        assert_eq!(out.len(), MAX_INPUT_BYTES);
    }

    /// Caso borde exacto: `MAX_INPUT_BYTES + 1` bytes excede el cap por uno.
    #[test]
    fn test_read_input_rejects_max_input_bytes_plus_one() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64 + 1);
        assert!(matches!(
            read_input_bounded(r),
            Err(HeadlessError::InputTooLarge(limit)) if limit == MAX_INPUT_BYTES
        ));
    }
}
