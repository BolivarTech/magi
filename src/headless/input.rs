// Author: Julian Bolivar
// Version: 1.1.0
// Date: 2026-07-18

//! Lectura y parseo de la entrada headless (`-i <file>` / stdin, REQ-H03/H10/H11/H29).
//!
//! Dos responsabilidades ortogonales:
//! - [`read_input_bounded`] acota los bytes no confiables a `MAX_INPUT_BYTES`
//!   (nunca bufferiza una fuente hostil ilimitada).
//! - [`parse_input`] auto-detecta texto-plano vs. envelope JSON y, para el
//!   envelope, aplica un **único** parser endurecido (un solo recorrido del
//!   JSON) que rechaza claves duplicadas, campos desconocidos junto a `prompt`,
//!   `prompt` no-string y anidamiento patológico (`> MAX_JSON_DEPTH`), incluso
//!   dentro de los valores de campos desconocidos.

use std::cell::Cell;
use std::collections::HashSet;
use std::fmt;
use std::io::Read;

use serde::de::{
    self, DeserializeSeed, Deserializer, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor,
};

use super::limits::{MAX_INPUT_BYTES, MAX_JSON_DEPTH};
use super::HeadlessError;

/// Mensaje de error cuando el anidamiento JSON supera [`MAX_JSON_DEPTH`].
///
/// Se comparte entre el visitor del envelope y [`DepthLimitedIgnoredAny`], por
/// lo que vive como constante (DRY, aparece en los tres puntos de guardia).
const DEPTH_EXCEEDED: &str = "JSON nesting too deep";

/// Mensaje de error cuando aparece una clave top-level duplicada (REQ-H11: el
/// last-wins silencioso de `serde_json` se rechaza explícitamente).
const DUPLICATE_KEY: &str = "duplicate top-level key";

/// Mensaje de error cuando hay un campo desconocido junto a un `prompt`
/// presente (deny-unknown se aplica sólo entonces, no antes).
const UNKNOWN_FIELD: &str = "unknown field alongside prompt";

/// Formato de la entrada headless, forzable por `--input-format` (REQ-H04).
///
/// Declarado como enum público porque el target de fuzz (T10) y el dispatch de
/// CLI lo referencian; el auto-detect (`None`) lo infiere del primer byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// La entrada es texto plano: todo el contenido es el `prompt` verbatim.
    Text,
    /// La entrada es un envelope JSON (objeto con al menos un `prompt`).
    Json,
}

/// Envelope de entrada resuelto (REQ-H11): el `prompt` obligatorio más los
/// campos opcionales de parametrización por-request.
///
/// Todo campo ausente en el JSON queda `None`; la resolución de defaults
/// (`magi.toml` / flags / agente proactivo) es responsabilidad de una tarea
/// posterior, no de este parser.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Prompt del usuario (obligatorio). En modo texto es la entrada completa
    /// verbatim; en modo envelope, el valor string del campo `prompt`.
    pub prompt: String,
    /// System-prompt propuesto por el caller (política de seguridad; el
    /// operador decide si se honra — REQ-H12b).
    pub system: Option<String>,
    /// Modelo LLM propuesto por el caller.
    pub model: Option<String>,
    /// Proveedor LLM propuesto por el caller.
    pub provider: Option<String>,
    /// Tope de llamadas a tools propuesto por el caller (clampeado al techo del
    /// operador en una tarea posterior — REQ-H12b).
    pub max_tool_calls: Option<u32>,
    /// Si forzar una pasada MAGI multiperspectiva (REQ-H22).
    pub consult: Option<bool>,
}

/// Resultado del recorrido del mapa top-level antes de la decisión final.
///
/// El visitor no puede decidir texto-vs-envelope hasta el **final** del mapa
/// (SC-H36: un objeto sin `prompt` es texto verbatim, sin importar sus otros
/// campos), por lo que separa "es un envelope válido" de "no había `prompt`".
enum MapOutcome {
    /// El objeto tenía `prompt` y sólo campos conocidos: es un envelope.
    Envelope(Envelope),
    /// El objeto NO tenía `prompt`: no es un envelope (cae a texto verbatim,
    /// salvo `--input-format json` que lo convierte en error de input).
    NoPrompt,
}

/// Construye un [`Envelope`] de sólo-texto: todo el input es el `prompt`.
///
/// Complejidad `O(n)` por la copia del texto; el resto de campos quedan `None`.
fn text_envelope(text: &str) -> Envelope {
    Envelope {
        prompt: text.to_string(),
        system: None,
        model: None,
        provider: None,
        max_tool_calls: None,
        consult: None,
    }
}

/// `true` si `byte` es whitespace JSON (space, tab, LF, CR).
///
/// Se usa para hallar el primer byte significativo del auto-detect sin
/// depender de `char::is_whitespace` (que aceptaría Unicode fuera del grammar
/// JSON).
fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

/// Entra a un contenedor JSON (map/seq) incrementando `depth` y falla si supera
/// [`MAX_JSON_DEPTH`].
///
/// Compartida por [`EnvelopeVisitor`] y [`DepthLimitedIgnoredAny`] para que el
/// tope de profundidad sea **global** (64 vale para todo valor, conocido o no —
/// cierra el bypass del `IgnoredAny` plano, que recursaría bajo el límite
/// interno de `serde_json`, 128).
///
/// # Errors
///
/// Devuelve `E::custom(DEPTH_EXCEEDED)` si la profundidad tras incrementar
/// excede [`MAX_JSON_DEPTH`].
fn enter_depth<E: de::Error>(depth: &Cell<u32>) -> Result<(), E> {
    let next = depth.get().saturating_add(1);
    if next > MAX_JSON_DEPTH {
        return Err(E::custom(DEPTH_EXCEEDED));
    }
    depth.set(next);
    Ok(())
}

/// Sale de un contenedor JSON decrementando `depth` (saturante por robustez).
fn leave_depth(depth: &Cell<u32>) {
    depth.set(depth.get().saturating_sub(1));
}

/// Seed que ignora el contenido de un valor JSON pero **cuenta su profundidad**
/// contra [`MAX_JSON_DEPTH`], compartiendo el contador con el visitor padre.
///
/// Reemplaza a `serde::de::IgnoredAny` para los valores de campos desconocidos:
/// el `IgnoredAny` plano recursa bajo el límite interno de `serde_json` (128),
/// no bajo el nuestro (64), dejando pasar un valor profundo-en-desconocido de
/// profundidad ∈ (64, 128]. Este seed cierra ese bypass.
struct DepthLimitedIgnoredAny<'a> {
    /// Contador de profundidad compartido con el visitor del envelope.
    depth: &'a Cell<u32>,
}

impl<'de, 'a> DeserializeSeed<'de> for DepthLimitedIgnoredAny<'a> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DepthLimitedVisitor { depth: self.depth })
    }
}

/// Visitor que descarta cualquier valor JSON pero contabiliza su anidamiento
/// (delegado por [`DepthLimitedIgnoredAny`]).
struct DepthLimitedVisitor<'a> {
    /// Contador de profundidad compartido.
    depth: &'a Cell<u32>,
}

impl<'de, 'a> Visitor<'de> for DepthLimitedVisitor<'a> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value within the depth limit")
    }

    fn visit_bool<E>(self, _v: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _v: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _v: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _v: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _v: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        enter_depth::<A::Error>(self.depth)?;
        while seq
            .next_element_seed(DepthLimitedIgnoredAny { depth: self.depth })?
            .is_some()
        {}
        leave_depth(self.depth);
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        enter_depth::<A::Error>(self.depth)?;
        // Las claves JSON son siempre strings (profundidad 1): `IgnoredAny`
        // plano es seguro para ellas. Sólo los VALORES pueden anidar, y esos
        // llevan el seed que cuenta profundidad.
        while map.next_key::<IgnoredAny>()?.is_some() {
            map.next_value_seed(DepthLimitedIgnoredAny { depth: self.depth })?;
        }
        leave_depth(self.depth);
        Ok(())
    }
}

/// Visitor del envelope: un **único** recorrido del objeto top-level que aplica
/// todas las guardias y recolecta las claves antes de la decisión final.
struct EnvelopeVisitor {
    /// Contador de profundidad; el objeto top-level cuenta como nivel 1.
    depth: Cell<u32>,
}

impl<'de> Visitor<'de> for EnvelopeVisitor {
    type Value = MapOutcome;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON envelope object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<MapOutcome, A::Error>
    where
        A: MapAccess<'de>,
    {
        enter_depth::<A::Error>(&self.depth)?;

        let mut seen: HashSet<String> = HashSet::new();
        let mut prompt: Option<String> = None;
        let mut system: Option<String> = None;
        let mut model: Option<String> = None;
        let mut provider: Option<String> = None;
        let mut max_tool_calls: Option<u32> = None;
        let mut consult: Option<bool> = None;
        let mut unknown_seen = false;

        while let Some(key) = map.next_key::<String>()? {
            // Dup-key aplica SIEMPRE (aborta antes de decidir) — REQ-H11.
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom(DUPLICATE_KEY));
            }
            match key.as_str() {
                // `prompt` no-string ⇒ error de tipo inmediato (sin recursión):
                // el deserializador de String falla en el primer token no-string.
                "prompt" => prompt = Some(map.next_value::<String>()?),
                "system" => system = map.next_value::<Option<String>>()?,
                "model" => model = map.next_value::<Option<String>>()?,
                "provider" => provider = map.next_value::<Option<String>>()?,
                "max_tool_calls" => max_tool_calls = map.next_value::<Option<u32>>()?,
                "consult" => consult = map.next_value::<Option<bool>>()?,
                // Campo desconocido: su valor se consume con el seed que cuenta
                // profundidad (NUNCA `IgnoredAny` plano — cerraría bajo 128).
                _ => {
                    unknown_seen = true;
                    map.next_value_seed(DepthLimitedIgnoredAny { depth: &self.depth })?;
                }
            }
        }

        leave_depth(&self.depth);

        // Decisión AL FINAL del mapa, en orden (REQ-H10):
        //   1. sin `prompt` ⇒ NoPrompt (texto verbatim; SC-H36 gana sobre deny-unknown).
        //   2. con `prompt` + campo desconocido ⇒ error (deny-unknown recién acá).
        //   3. con `prompt` + sólo conocidos ⇒ Envelope.
        match prompt {
            None => Ok(MapOutcome::NoPrompt),
            Some(prompt) => {
                if unknown_seen {
                    Err(A::Error::custom(UNKNOWN_FIELD))
                } else {
                    Ok(MapOutcome::Envelope(Envelope {
                        prompt,
                        system,
                        model,
                        provider,
                        max_tool_calls,
                        consult,
                    }))
                }
            }
        }
    }
}

/// Lee `reader` hasta EOF, acotado a `max_input_bytes` (REQ-H29, anti-DoS).
///
/// `max_input_bytes` es el cap EFECTIVO de esta corrida — el operador puede
/// bajarlo (nunca subirlo) vía `[headless] max_input_bytes` en `magi.toml`
/// (spec §11); [`MAX_INPUT_BYTES`] es solo el valor por-default que
/// `HeadlessLimits::default()` usa cuando el operador no lo fija.
///
/// Usa `reader.take(max_input_bytes as u64 + 1)`: el `+1` es lo que permite
/// distinguir "la fuente tenía exactamente el cap" de "la fuente excedía el
/// cap" sin necesidad de leer más allá — una fuente hostil e ilimitada (p.ej.
/// `std::io::repeat`) nunca se bufferiza por completo, porque `take` corta la
/// lectura en `cap + 1` bytes pase lo que pase aguas arriba.
///
/// Complejidad `O(n)` en el tamaño de la entrada, acotada por `cap + 1`
/// sin importar cuánto produzca `reader`.
///
/// # Errors
///
/// Devuelve [`HeadlessError::InputTooLarge`] con el límite configurado si el
/// contenido leído excede `max_input_bytes`. Devuelve [`HeadlessError::Io`] si
/// el `reader` subyacente falla durante la lectura (p.ej. un error real de E/S
/// de stdin o de un archivo); ese caso se propaga tal cual, sin exponer nunca
/// contenido de la entrada en el mensaje de error.
pub fn read_input_bounded(
    reader: impl Read,
    max_input_bytes: usize,
) -> Result<Vec<u8>, HeadlessError> {
    // STUB (TDD Red): ignores the effective cap and always bounds against the
    // module constant. The Green commit wires `max_input_bytes` through.
    let _ = max_input_bytes;
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

/// Parsea `bytes` a un [`Envelope`], auto-detectando texto-plano vs. envelope
/// JSON (o forzando el formato con `forced_fmt`) — REQ-H10/H11.
///
/// Semántica (un solo parser, sin doble-parse):
/// 1. UTF-8 estricto: bytes no-UTF8 ⇒ [`HeadlessError::InputInvalid`].
/// 2. `forced_fmt == Some(Text)` ⇒ nunca parsea: todo el input es el `prompt`.
/// 3. Auto-detect por el primer byte no-blanco: si no es `{`, la entrada no es
///    un envelope ⇒ prompt verbatim (o `InputInvalid` si `forced_fmt == Json`).
/// 4. Si es `{`, un único recorrido (`EnvelopeVisitor`) aplica dup-key,
///    profundidad (`> MAX_JSON_DEPTH`, incluso dentro de campos desconocidos) y
///    la decisión de fin-de-mapa (sin `prompt` ⇒ texto; con `prompt` + campo
///    desconocido ⇒ error; `prompt` no-string ⇒ error).
///
/// La guardia de profundidad **gana** sobre "texto verbatim": un `{`-input sin
/// `prompt` pero patológicamente anidado se **rechaza** por DoS, no se acepta
/// como prompt gigante.
///
/// Complejidad `O(n)` en el tamaño de la entrada; la recursión del recorrido
/// está acotada por [`MAX_JSON_DEPTH`], por lo que no puede desbordar la pila.
///
/// # Errors
///
/// Devuelve [`HeadlessError::InputInvalid`] si: los bytes no son UTF-8; se
/// fuerza `Json` pero la entrada no es un objeto (o no tiene `prompt`); el JSON
/// es malformado; hay una clave duplicada; hay un campo desconocido junto a
/// `prompt`; `prompt` no es un string; o el anidamiento supera
/// [`MAX_JSON_DEPTH`]. El mensaje **jamás** incluye el contenido crudo de la
/// entrada.
pub fn parse_input(
    bytes: &[u8],
    forced_fmt: Option<InputFormat>,
) -> Result<Envelope, HeadlessError> {
    // 1. UTF-8 estricto.
    let text = std::str::from_utf8(bytes)
        .map_err(|_| HeadlessError::InputInvalid("input is not valid UTF-8".to_string()))?;

    // 2. Formato forzado a texto: nunca se parsea como JSON.
    if forced_fmt == Some(InputFormat::Text) {
        return Ok(text_envelope(text));
    }

    // 3. Auto-detect barato: primer byte no-blanco.
    let looks_like_object = text.bytes().find(|&b| !is_json_whitespace(b)) == Some(b'{');

    if !looks_like_object {
        if forced_fmt == Some(InputFormat::Json) {
            return Err(HeadlessError::InputInvalid(
                "expected a JSON object under --input-format json".to_string(),
            ));
        }
        return Ok(text_envelope(text));
    }

    // 4. Único parser endurecido del objeto.
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let outcome = (&mut de)
        .deserialize_map(EnvelopeVisitor {
            depth: Cell::new(0),
        })
        .map_err(|_| HeadlessError::InputInvalid("malformed JSON envelope".to_string()))?;
    // Rechazar datos basura tras el objeto (sin silent-accept).
    de.end().map_err(|_| {
        HeadlessError::InputInvalid("trailing data after JSON envelope".to_string())
    })?;

    match outcome {
        MapOutcome::Envelope(envelope) => Ok(envelope),
        MapOutcome::NoPrompt => {
            if forced_fmt == Some(InputFormat::Json) {
                Err(HeadlessError::InputInvalid(
                    "JSON object has no `prompt` field under --input-format json".to_string(),
                ))
            } else {
                Ok(text_envelope(text))
            }
        }
    }
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
            read_input_bounded(r, MAX_INPUT_BYTES),
            Err(HeadlessError::InputTooLarge(_))
        ));
    }

    /// Entrada vacía es válida a este nivel: el parser de envelope (tarea
    /// posterior) es quien decide si un prompt vacío es un error de input.
    #[test]
    fn test_read_input_empty_reader_returns_empty_vec() {
        let r = Cursor::new(Vec::new());
        assert_eq!(
            read_input_bounded(r, MAX_INPUT_BYTES).unwrap(),
            Vec::<u8>::new()
        );
    }

    /// Caso borde exacto: `MAX_INPUT_BYTES` bytes caben justo bajo el cap.
    #[test]
    fn test_read_input_accepts_exactly_max_input_bytes() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64);
        let out = read_input_bounded(r, MAX_INPUT_BYTES).expect("exactly the cap must be accepted");
        assert_eq!(out.len(), MAX_INPUT_BYTES);
    }

    /// Caso borde exacto: `MAX_INPUT_BYTES + 1` bytes excede el cap por uno.
    #[test]
    fn test_read_input_rejects_max_input_bytes_plus_one() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64 + 1);
        assert!(matches!(
            read_input_bounded(r, MAX_INPUT_BYTES),
            Err(HeadlessError::InputTooLarge(limit)) if limit == MAX_INPUT_BYTES
        ));
    }

    /// REQ-H29/spec §11: el cap EFECTIVO (`[headless] max_input_bytes`) debe
    /// gobernar la lectura, no el `MAX_INPUT_BYTES` constante — un operador que
    /// baja el cap a 10 bytes debe ver una entrada de 11 bytes rechazada aunque
    /// esté muy por debajo del default de 10 MiB.
    #[test]
    fn test_read_input_bounded_respects_custom_effective_cap() {
        let small_cap = 10usize;
        let r = Cursor::new(vec![b'x'; small_cap + 1]);
        assert!(
            matches!(
                read_input_bounded(r, small_cap),
                Err(HeadlessError::InputTooLarge(limit)) if limit == small_cap
            ),
            "a custom (smaller) effective cap must be enforced, not the module constant"
        );

        let r_ok = Cursor::new(vec![b'x'; small_cap]);
        let out =
            read_input_bounded(r_ok, small_cap).expect("exactly the custom cap must be accepted");
        assert_eq!(out.len(), small_cap);
    }

    // ---- parse_input --------------------------------------------------------

    /// Auto-detect: objeto con `prompt` ⇒ envelope; texto ⇒ prompt verbatim;
    /// objeto SIN `prompt` ⇒ texto verbatim (SC-H36).
    #[test]
    fn test_parse_input_autodetect() {
        let e = parse_input(br#"{"prompt":"hi","consult":true}"#, None).unwrap();
        assert_eq!(e.prompt, "hi");
        assert_eq!(e.consult, Some(true));

        let t = parse_input(b"just text", None).unwrap();
        assert_eq!(t.prompt, "just text");

        let j = parse_input(br#"{"foo":1}"#, None).unwrap(); // objeto sin prompt => texto
        assert_eq!(j.prompt, r#"{"foo":1}"#);
    }

    /// La prioridad de deny-unknown depende de la presencia de `prompt`
    /// (resuelve la contradicción `{"foo":1}`): sin `prompt`, un campo
    /// desconocido NO invalida (es texto); con `prompt`, sí.
    #[test]
    fn test_parse_input_unknown_field_priority_depends_on_prompt_presence() {
        let t = parse_input(br#"{"foo":1}"#, None).unwrap();
        assert_eq!(t.prompt, r#"{"foo":1}"#);

        assert!(matches!(
            parse_input(br#"{"foo":1,"prompt":"x"}"#, None),
            Err(HeadlessError::InputInvalid(_))
        ));

        assert_eq!(
            parse_input(br#"{"prompt":"x","consult":true}"#, None)
                .unwrap()
                .prompt,
            "x"
        );
    }

    /// `prompt` no-string ⇒ InputInvalid; clave duplicada ⇒ InputInvalid;
    /// anidamiento profundo (forzado Json, no-objeto) ⇒ InputInvalid.
    #[test]
    fn test_parse_input_rejects_nonstring_prompt_dupkey_and_deep() {
        assert!(matches!(
            parse_input(br#"{"prompt":123}"#, Some(InputFormat::Json)),
            Err(HeadlessError::InputInvalid(_))
        ));
        assert!(matches!(
            parse_input(br#"{"prompt":"a","prompt":"b"}"#, None),
            Err(HeadlessError::InputInvalid(_))
        ));
        let deep = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        assert!(matches!(
            parse_input(deep.as_bytes(), Some(InputFormat::Json)),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// Construye `levels` objetos `{"a": ...}` anidados, valor interno `1`.
    ///
    /// La profundidad de contenedores resultante es exactamente `levels`.
    fn nested_object(levels: u32) -> String {
        let mut s = String::from("1");
        for _ in 0..levels {
            s = format!(r#"{{"a":{s}}}"#);
        }
        s
    }

    /// Frontera de profundidad: 64 niveles OK (cae a texto verbatim), 65
    /// niveles ⇒ InputInvalid por la guardia de profundidad.
    #[test]
    fn test_parse_input_depth_boundary_64_ok_65_rejected() {
        // 64 contenedores (== MAX_JSON_DEPTH): NO error. Sin `prompt` ⇒ texto.
        assert!(parse_input(nested_object(64).as_bytes(), None).is_ok());
        // 65 contenedores (> MAX_JSON_DEPTH): rechazado por profundidad.
        assert!(matches!(
            parse_input(nested_object(65).as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// Un valor ~100-profundo DENTRO de un campo desconocido, con `prompt`
    /// presente, ⇒ InputInvalid: prueba que el bypass del `IgnoredAny` plano
    /// (que recursaría bajo el límite interno 128) está cerrado.
    #[test]
    fn test_parse_input_depth_inside_unknown_field_is_bounded() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"prompt":"x","foo":{deep_value}}}"#);
        assert!(matches!(
            parse_input(input.as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// La guardia de profundidad GANA sobre "texto verbatim": un `{`-input SIN
    /// `prompt` pero patológicamente anidado se rechaza (DoS), no se acepta
    /// como prompt gigante.
    #[test]
    fn test_parse_input_deep_object_without_prompt_rejected_by_depth() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"foo":{deep_value}}}"#);
        assert!(matches!(
            parse_input(input.as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// `forced_fmt = Text` con un `{`-input profundo ⇒ texto verbatim: nunca se
    /// parsea, la profundidad no aplica (sólo el cap de bytes).
    #[test]
    fn test_parse_input_forced_text_never_parses_deep_object() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"foo":{deep_value}}}"#);
        let e = parse_input(input.as_bytes(), Some(InputFormat::Text)).unwrap();
        assert_eq!(e.prompt, input);
    }

    /// Format forcing: `Json` + texto no-objeto ⇒ InputInvalid; `Text` +
    /// `{"prompt":"x"}` ⇒ el prompt es el string JSON verbatim (no se parsea).
    #[test]
    fn test_parse_input_format_forcing() {
        assert!(matches!(
            parse_input(b"just text", Some(InputFormat::Json)),
            Err(HeadlessError::InputInvalid(_))
        ));

        let e = parse_input(br#"{"prompt":"x"}"#, Some(InputFormat::Text)).unwrap();
        assert_eq!(e.prompt, r#"{"prompt":"x"}"#);
    }

    /// Bytes no-UTF8 ⇒ InputInvalid (nunca panic).
    #[test]
    fn test_parse_input_rejects_non_utf8() {
        assert!(matches!(
            parse_input(&[0xff, 0xfe, 0x00], None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// Unit-smoke del target de fuzz `fuzz_headless_input` (REQ-H35): entradas
    /// degeneradas (vacía, no-UTF8, JSON patológicamente anidado, clave
    /// duplicada, `prompt` no-string, strings con `{`/`[`/claves embebidas)
    /// nunca panican y siempre devuelven un `Result` tipado — ni OOM (lectura
    /// acotada) ni stack overflow (profundidad acotada). Corre en cada §0.1,
    /// complementando la corrida coverage-guided de CI.
    #[test]
    fn test_parse_input_smoke_never_panics_on_degenerate_bytes() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            vec![0xff, 0xfe, 0x00, 0x80],
            deep.into_bytes(),
            br#"{"prompt":"a","prompt":"b"}"#.to_vec(),
            br#"{"prompt":123}"#.to_vec(),
            b"{[not valid json".to_vec(),
            br#"["array","not","object"]"#.to_vec(),
            b"{".to_vec(),
            b"plain text with { and [ chars".to_vec(),
            br#"{"prompt":"x","unknown":{"nested":[1,2,3]}}"#.to_vec(),
        ];
        for bytes in &cases {
            for fmt in [None, Some(InputFormat::Json), Some(InputFormat::Text)] {
                // Nunca panic; el resultado tipado se descarta (sólo robustez).
                let _ = parse_input(bytes, fmt);
            }
            // La lectura acotada del mismo input tampoco panica.
            let _ = read_input_bounded(Cursor::new(bytes.clone()), MAX_INPUT_BYTES);
        }
    }
}
