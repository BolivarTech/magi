// Author: Julian Bolivar Version: 1.1.0 Date: 2026-07-18

//! Reading and parsing of headless input (`-i <file>` / stdin, REQ-H03/H10/H11/H29).
//!
//! Two orthogonal responsibilities:
//! - [`read_input_bounded`] bounds untrusted bytes to `MAX_INPUT_BYTES`
//! (never buffers an unlimited hostile source).
//! - [`parse_input`] auto-detects plain text vs. JSON envelope and, for the
//! envelope, applies a **single** hardened parser (one pass over the JSON) that rejects
//! duplicate keys, unknown fields alongside `prompt`, non-string `prompt`, and pathological
//! nesting (`> MAX_JSON_DEPTH`), even inside the values of unknown fields.

use std::cell::Cell;
use std::collections::HashSet;
use std::fmt;
use std::io::Read;

use magi_core::schema::Mode;
use serde::de::{
    self, DeserializeSeed, Deserializer, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor,
};

use super::limits::MAX_JSON_DEPTH;
use super::HeadlessError;
use crate::magi::mode::{ModeExt, ModeParseError};

/// Error message when JSON nesting exceeds [`MAX_JSON_DEPTH`].
///
/// It is shared between the envelope visitor and [`DepthLimitedIgnoredAny`], so it lives as a
/// constant (DRY, it appears in the three guard points).
const DEPTH_EXCEEDED: &str = "JSON nesting too deep";

/// Error message when a duplicate top-level key appears (REQ-H11: `serde_json`'s silent last-
/// wins is explicitly rejected).
const DUPLICATE_KEY: &str = "duplicate top-level key";

/// Error message when there is an unknown field alongside a present `prompt` (deny-unknown
/// applies only then, not before).
const UNKNOWN_FIELD: &str = "unknown field alongside prompt";

/// Format of the headless input, forceable via `--input-format` (REQ-H04).
///
/// Declared as a public enum because the fuzz target (T10) and CLI dispatch reference it; auto-
/// detect (`None`) infers it from the first byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// The input is plain text: all content is the `prompt` verbatim.
    Text,
    /// The input is a JSON envelope (object with at least one `prompt`).
    Json,
}

/// Resolved input envelope (REQ-H11): the mandatory `prompt` plus the optional per-request
/// parameterization fields.
///
/// Any field absent in the JSON becomes `None`; default resolution (`magi.toml` / flags /
/// proactive agent) is the responsibility of a later task, not this parser.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// User prompt (mandatory). In text mode it is the entire input verbatim; in envelope mode,
    /// the string value of the `prompt` field.
    pub prompt: String,
    /// System prompt proposed by the caller (security policy; the operator decides whether to
    /// honor it — REQ-H12b).
    pub system: Option<String>,
    /// LLM model proposed by the caller.
    pub model: Option<String>,
    /// LLM provider proposed by the caller.
    pub provider: Option<String>,
    /// Tool call cap proposed by the caller (clamped to the operator's ceiling in a later task
    /// — REQ-H12b).
    pub max_tool_calls: Option<u32>,
    /// Whether to force a MAGI multi-perspective pass (REQ-H22).
    pub consult: Option<bool>,
    /// Explicit mode proposed by the caller (REQ-A07/A07c). `None` if the field is absent; the
    /// content (absent/blank vs. present-and-unrecognized) is validated in
    /// [`Envelope::resolved_mode`], not here — this parser only collects the raw string, just
    /// as it does for `system`/`model`/`provider`.
    pub mode: Option<String>,
    /// Declares that the `prompt` under analysis is NOT trusted (REQ-A07d): with this active,
    /// omitting the mode becomes an error instead of inference. `None` if the field is absent —
    /// the surface that actually needs this flag is precisely this one (the envelope is the
    /// consumer of an automated gate).
    pub untrusted_content: Option<bool>,
}

impl Envelope {
    /// Resolves the envelope's `mode` field to a [`Mode`] (REQ-A07c).
    ///
    /// Same treatment as a configuration value ([`ModeExt::parse_config_value`]): absent or
    /// blank ⇒ `Ok(None)`; present and unrecognized ⇒ `Err`. The field was declared by a human
    /// or an integrator system writing the envelope, not a model — therefore it goes through
    /// configuration validation and not through `magi_rs::magi::mode::normalize_label`, which
    /// is the open-format normalization intended for LLM output text.
    ///
    /// # Errors
    /// [`ModeParseError::Unknown`] if `mode` carries content and does not name any of the three
    /// valid modes. Narrow allow: consumed by the real mode resolution in Task 2.3/2.4 — this
    /// task only adds the field and its parsing, it does not connect it to dispatch. Covered by
    /// `every_surface_accepts_an_explicit_mode`.
    #[allow(dead_code)]
    pub fn resolved_mode(&self) -> Result<Option<Mode>, ModeParseError> {
        <Mode as ModeExt>::parse_config_value(self.mode.as_deref().unwrap_or_default())
    }
}

/// Result of the top-level map traversal before the final decision.
///
/// The visitor cannot decide text-vs-envelope until the **end** of the map (SC-H36: an object
/// without `prompt` is verbatim text, regardless of its other fields), so it separates "is a
/// valid envelope" from "there was no `prompt`".
enum MapOutcome {
    /// The object had `prompt` and only known fields: it is an envelope.
    Envelope(Envelope),
    /// The object did NOT have `prompt`: it is not an envelope (falls back to verbatim text,
    /// except `--input-format json` turns it into an input error).
    NoPrompt,
}

/// Builds a text-only [`Envelope`]: all input is the `prompt`.
///
/// Complexity `O(n)` due to copying the text; the rest of the fields become `None`.
fn text_envelope(text: &str) -> Envelope {
    Envelope {
        prompt: text.to_string(),
        system: None,
        model: None,
        provider: None,
        max_tool_calls: None,
        consult: None,
        mode: None,
        untrusted_content: None,
    }
}

/// `true` if `byte` is JSON whitespace (space, tab, LF, CR).
///
/// Used to find the first significant byte of auto-detect without relying on
/// `char::is_whitespace` (which would accept Unicode outside the JSON grammar).
fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

/// Enters a JSON container (map/seq) incrementing `depth` and fails if it exceeds
/// [`MAX_JSON_DEPTH`].
///
/// Shared by [`EnvelopeVisitor`] and [`DepthLimitedIgnoredAny`] so the depth cap is **global**
/// (64 applies to every value, known or not — it closes the bypass of plain `IgnoredAny`, which
/// would recurse under `serde_json`'s internal limit, 128).
///
/// # Errors
///
/// Returns `E::custom(DEPTH_EXCEEDED)` if the depth after incrementing exceeds
/// [`MAX_JSON_DEPTH`].
fn enter_depth<E: de::Error>(depth: &Cell<u32>) -> Result<(), E> {
    let next = depth.get().saturating_add(1);
    if next > MAX_JSON_DEPTH {
        return Err(E::custom(DEPTH_EXCEEDED));
    }
    depth.set(next);
    Ok(())
}

/// Leaves a JSON container decrementing `depth` (saturating for robustness).
fn leave_depth(depth: &Cell<u32>) {
    depth.set(depth.get().saturating_sub(1));
}

/// Seed that ignores the content of a JSON value but **counts its depth** against
/// [`MAX_JSON_DEPTH`], sharing the counter with the parent visitor.
///
/// Replaces `serde::de::IgnoredAny` for the values of unknown fields: plain `IgnoredAny`
/// recurses under `serde_json`'s internal limit (128), not under ours (64), allowing a deep-in-
/// unknown value of depth ∈ (64, 128]. This seed closes that bypass.
struct DepthLimitedIgnoredAny<'a> {
    /// Depth counter shared with the envelope visitor.
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

/// Visitor that discards any JSON value but accounts for its nesting (delegated by
/// [`DepthLimitedIgnoredAny`]).
struct DepthLimitedVisitor<'a> {
    /// Shared depth counter.
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
        // JSON keys are always strings (depth 1): plain `IgnoredAny` is safe for them. Only
        // VALUES can nest, and those carry the depth-counting seed.
        while map.next_key::<IgnoredAny>()?.is_some() {
            map.next_value_seed(DepthLimitedIgnoredAny { depth: self.depth })?;
        }
        leave_depth(self.depth);
        Ok(())
    }
}

/// Envelope visitor: a **single** pass over the top-level object that applies all guards and
/// collects the keys before the final decision.
struct EnvelopeVisitor {
    /// Depth counter; the top-level object counts as level 1.
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
        let mut mode: Option<String> = None;
        let mut untrusted_content: Option<bool> = None;
        let mut unknown_seen = false;

        while let Some(key) = map.next_key::<String>()? {
            // Dup-key applies ALWAYS (aborts before deciding) — REQ-H11.
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom(DUPLICATE_KEY));
            }
            match key.as_str() {
                // Non-string `prompt` ⇒ immediate type error (no recursion): the String
                // deserializer fails on the first non-string token.
                "prompt" => prompt = Some(map.next_value::<String>()?),
                "system" => system = map.next_value::<Option<String>>()?,
                "model" => model = map.next_value::<Option<String>>()?,
                "provider" => provider = map.next_value::<Option<String>>()?,
                "max_tool_calls" => max_tool_calls = map.next_value::<Option<u32>>()?,
                "consult" => consult = map.next_value::<Option<bool>>()?,
                "mode" => mode = map.next_value::<Option<String>>()?,
                "untrusted_content" => untrusted_content = map.next_value::<Option<bool>>()?,
                // Unknown field: its value is consumed with the depth-counting seed (NEVER
                // plain `IgnoredAny` — it would fall under 128).
                _ => {
                    unknown_seen = true;
                    map.next_value_seed(DepthLimitedIgnoredAny { depth: &self.depth })?;
                }
            }
        }

        leave_depth(&self.depth);

        // Decision AT THE END of the map, in order (REQ-H10):
        // 1. without `prompt` ⇒ NoPrompt (verbatim text; SC-H36 wins over deny-unknown).
        // 2. with `prompt` + unknown field ⇒ error (deny-unknown only here).
        // 3. with `prompt` + only known fields ⇒ Envelope.
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
                        mode,
                        untrusted_content,
                    }))
                }
            }
        }
    }
}

/// Reads `reader` until EOF, bounded to `max_input_bytes` (REQ-H29, anti-DoS).
///
/// `max_input_bytes` is the EFFECTIVE cap for this run — the operator can lower it (never raise
/// it) via `[headless] max_input_bytes` in `magi.toml` (spec §11);
/// [`MAX_INPUT_BYTES`](super::limits::MAX_INPUT_BYTES) is only the default value that
/// `HeadlessLimits::default()` uses when the operator does not set it.
///
/// Uses `reader.take(max_input_bytes as u64 + 1)`: the `+1` is what makes it possible to
/// distinguish "the source had exactly the cap" from "the source exceeded the cap" without
/// reading further — a hostile and unlimited source (e.g., `std::io::repeat`) is never fully
/// buffered, because `take` cuts off reading at `cap + 1` bytes no matter what happens
/// upstream.
///
/// Complexity `O(n)` in the size of the input, bounded by `cap + 1` regardless of how much
/// `reader` produces.
///
/// # Errors
///
/// Returns [`HeadlessError::InputTooLarge`] with the configured limit if the read content
/// exceeds `max_input_bytes`. Returns [`HeadlessError::Io`] if the underlying `reader` fails
/// during reading (e.g., a real I/O error from stdin or a file); that case is propagated as-is,
/// never exposing input content in the error message.
pub fn read_input_bounded(
    reader: impl Read,
    max_input_bytes: usize,
) -> Result<Vec<u8>, HeadlessError> {
    let mut buf = Vec::new();
    reader
        .take(max_input_bytes as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| HeadlessError::Io(e.to_string()))?;

    if buf.len() > max_input_bytes {
        return Err(HeadlessError::InputTooLarge(max_input_bytes));
    }

    Ok(buf)
}

/// Parses `bytes` into an [`Envelope`], auto-detecting plain text vs. JSON envelope (or forcing
/// the format with `forced_fmt`) — REQ-H10/H11.
///
/// Semantics (a single parser, no double-parse):
/// 1. Strict UTF-8: non-UTF8 bytes ⇒ [`HeadlessError::InputInvalid`].
/// 2. `forced_fmt == Some(Text)` ⇒ never parses: all input is the `prompt`.
/// 3. Auto-detect by the first non-blank byte: if it is not `{`, the input is not
/// an envelope ⇒ prompt verbatim (or `InputInvalid` if `forced_fmt == Json`).
/// 4. If it is `{`, a single pass (`EnvelopeVisitor`) applies dup-key,
/// depth (`> MAX_JSON_DEPTH`, even inside unknown fields) and the end-of-map decision (without
/// `prompt` ⇒ text; with `prompt` + unknown field ⇒ error; non-string `prompt` ⇒ error).
///
/// The depth guard **wins** over "verbatim text": a `{`-input without `prompt` but
/// pathologically nested is **rejected** for DoS, not accepted as a giant prompt.
///
/// Complexity `O(n)` in the size of the input; the traversal recursion is bounded by
/// [`MAX_JSON_DEPTH`], so it cannot overflow the stack.
///
/// # Errors
///
/// Returns [`HeadlessError::InputInvalid`] if: the bytes are not UTF-8; `Json` is forced but
/// the input is not an object (or has no `prompt`); the JSON is malformed; there is a duplicate
/// key; there is an unknown field alongside `prompt`; `prompt` is not a string; or the nesting
/// exceeds [`MAX_JSON_DEPTH`]. The message **never** includes the raw content of the input.
pub fn parse_input(
    bytes: &[u8],
    forced_fmt: Option<InputFormat>,
) -> Result<Envelope, HeadlessError> {
    // 1. Strict UTF-8.
    let text = std::str::from_utf8(bytes)
        .map_err(|_| HeadlessError::InputInvalid("input is not valid UTF-8".to_string()))?;

    // 2. Format forced to text: it is never parsed as JSON.
    if forced_fmt == Some(InputFormat::Text) {
        return Ok(text_envelope(text));
    }

    // 3. Cheap auto-detect: first non-blank byte.
    let looks_like_object = text.bytes().find(|&b| !is_json_whitespace(b)) == Some(b'{');

    if !looks_like_object {
        if forced_fmt == Some(InputFormat::Json) {
            return Err(HeadlessError::InputInvalid(
                "expected a JSON object under --input-format json".to_string(),
            ));
        }
        return Ok(text_envelope(text));
    }

    // 4. Single hardened parser of the object.
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let outcome = (&mut de)
        .deserialize_map(EnvelopeVisitor {
            depth: Cell::new(0),
        })
        .map_err(|_| HeadlessError::InputInvalid("malformed JSON envelope".to_string()))?;
    // Reject garbage data after the object (no silent-accept).
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

    use super::super::limits::MAX_INPUT_BYTES;
    use super::*;

    /// An unlimited source (`io::repeat`) is never fully buffered: `take(cap+1)` cuts it and
    /// the result is `InputTooLarge`, not a hang/OOM.
    #[test]
    fn test_read_input_rejects_oversized_without_buffering_all() {
        let r = std::io::repeat(b'a');
        assert!(matches!(
            read_input_bounded(r, MAX_INPUT_BYTES),
            Err(HeadlessError::InputTooLarge(_))
        ));
    }

    /// Empty input is valid at this level: the envelope parser (later task) is the one that
    /// decides whether an empty prompt is an input error.
    #[test]
    fn test_read_input_empty_reader_returns_empty_vec() {
        let r = Cursor::new(Vec::new());
        assert_eq!(
            read_input_bounded(r, MAX_INPUT_BYTES).unwrap(),
            Vec::<u8>::new()
        );
    }

    /// Exact edge case: `MAX_INPUT_BYTES` bytes fit just under the cap.
    #[test]
    fn test_read_input_accepts_exactly_max_input_bytes() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64);
        let out = read_input_bounded(r, MAX_INPUT_BYTES).expect("exactly the cap must be accepted");
        assert_eq!(out.len(), MAX_INPUT_BYTES);
    }

    /// Exact edge case: `MAX_INPUT_BYTES + 1` bytes exceeds the cap by one.
    #[test]
    fn test_read_input_rejects_max_input_bytes_plus_one() {
        let r = std::io::repeat(b'x').take(MAX_INPUT_BYTES as u64 + 1);
        assert!(matches!(
            read_input_bounded(r, MAX_INPUT_BYTES),
            Err(HeadlessError::InputTooLarge(limit)) if limit == MAX_INPUT_BYTES
        ));
    }

    /// REQ-H29/spec §11: the EFFECTIVE cap (`[headless] max_input_bytes`) must govern reading,
    /// not the constant `MAX_INPUT_BYTES` — an operator who lowers the cap to 10 bytes must see
    /// an 11-byte input rejected even if it is far below the 10 MiB default.
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

    /// Auto-detect: object with `prompt` ⇒ envelope; text ⇒ prompt verbatim; object WITHOUT
    /// `prompt` ⇒ verbatim text (SC-H36).
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

    /// The envelope collects `mode`/`untrusted_content` just like its other optional fields,
    /// and `resolved_mode` validates the raw string (REQ-A07/A07c/A07d).
    #[test]
    fn test_parse_input_carries_mode_and_untrusted_content() {
        let e = parse_input(
            br#"{"prompt":"x","mode":"design","untrusted_content":true}"#,
            None,
        )
        .unwrap();
        assert_eq!(e.mode.as_deref(), Some("design"));
        assert_eq!(e.resolved_mode().unwrap(), Some(Mode::Design));
        assert_eq!(e.untrusted_content, Some(true));

        // Absent ⇒ `None`, without inventing a default.
        let bare = parse_input(br#"{"prompt":"x"}"#, None).unwrap();
        assert_eq!(bare.mode, None);
        assert_eq!(bare.resolved_mode().unwrap(), None);
        assert_eq!(bare.untrusted_content, None);
    }

    /// A present and unrecognized `mode` is a typed error, not a silent `None` — same rule as a
    /// configuration value (REQ-A07c).
    #[test]
    fn test_envelope_resolved_mode_rejects_an_unknown_label() {
        let e = parse_input(br#"{"prompt":"x","mode":"banana"}"#, None).unwrap();
        assert!(matches!(
            e.resolved_mode(),
            Err(ModeParseError::Unknown { .. })
        ));
    }

    /// The priority of deny-unknown depends on the presence of `prompt` (resolves the
    /// contradiction `{"foo":1}`): without `prompt`, an unknown field does NOT invalidate (it
    /// is text); with `prompt`, it does.
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

    /// Non-string `prompt` ⇒ InputInvalid; duplicate key ⇒ InputInvalid; deep nesting (forced
    /// Json, non-object) ⇒ InputInvalid.
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

    /// Builds `levels` nested `{"a": ...}` objects, inner value `1`.
    ///
    /// The resulting container depth is exactly `levels`.
    fn nested_object(levels: u32) -> String {
        let mut s = String::from("1");
        for _ in 0..levels {
            s = format!(r#"{{"a":{s}}}"#);
        }
        s
    }

    /// Depth boundary: 64 levels OK (falls back to verbatim text), 65 levels ⇒ InputInvalid due
    /// to the depth guard.
    #[test]
    fn test_parse_input_depth_boundary_64_ok_65_rejected() {
        // 64 containers (== MAX_JSON_DEPTH): NO error. Without `prompt` ⇒ text.
        assert!(parse_input(nested_object(64).as_bytes(), None).is_ok());
        // 65 containers (> MAX_JSON_DEPTH): rejected due to depth.
        assert!(matches!(
            parse_input(nested_object(65).as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// A ~100-deep value INSIDE an unknown field, with `prompt` present, ⇒ InputInvalid: proves
    /// that the bypass of plain `IgnoredAny` (which would recurse under the internal limit 128)
    /// is closed.
    #[test]
    fn test_parse_input_depth_inside_unknown_field_is_bounded() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"prompt":"x","foo":{deep_value}}}"#);
        assert!(matches!(
            parse_input(input.as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// The depth guard WINS over "verbatim text": a `{`-input WITHOUT `prompt` but
    /// pathologically nested is rejected (DoS), not accepted as a giant prompt.
    #[test]
    fn test_parse_input_deep_object_without_prompt_rejected_by_depth() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"foo":{deep_value}}}"#);
        assert!(matches!(
            parse_input(input.as_bytes(), None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// `forced_fmt = Text` with a deep `{`-input ⇒ verbatim text: it is never parsed, depth
    /// does not apply (only the byte cap).
    #[test]
    fn test_parse_input_forced_text_never_parses_deep_object() {
        let deep_value = format!("{}{}{}", "[".repeat(100), "1", "]".repeat(100));
        let input = format!(r#"{{"foo":{deep_value}}}"#);
        let e = parse_input(input.as_bytes(), Some(InputFormat::Text)).unwrap();
        assert_eq!(e.prompt, input);
    }

    /// Format forcing: `Json` + non-object text ⇒ InputInvalid; `Text` + `{"prompt":"x"}` ⇒ the
    /// prompt is the JSON string verbatim (not parsed).
    #[test]
    fn test_parse_input_format_forcing() {
        assert!(matches!(
            parse_input(b"just text", Some(InputFormat::Json)),
            Err(HeadlessError::InputInvalid(_))
        ));

        let e = parse_input(br#"{"prompt":"x"}"#, Some(InputFormat::Text)).unwrap();
        assert_eq!(e.prompt, r#"{"prompt":"x"}"#);
    }

    /// Non-UTF8 bytes ⇒ InputInvalid (never panic).
    #[test]
    fn test_parse_input_rejects_non_utf8() {
        assert!(matches!(
            parse_input(&[0xff, 0xfe, 0x00], None),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// Unit-smoke of the fuzz target `fuzz_headless_input` (REQ-H35): degenerate inputs (empty,
    /// non-UTF8, pathologically nested JSON, duplicate key, non-string `prompt`, strings with
    /// `{`/`[`/embedded keys) never panic and always return a typed `Result` — neither OOM
    /// (bounded reading) nor stack overflow (bounded depth). Runs at every §0.1, complementing
    /// the CI coverage-guided run.
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
                // Never panic; the typed result is discarded (only robustness).
                let _ = parse_input(bytes, fmt);
            }
            // The bounded reading of the same input also does not panic.
            let _ = read_input_bounded(Cursor::new(bytes.clone()), MAX_INPUT_BYTES);
        }
    }
}
