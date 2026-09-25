// Author: Julian Bolivar
// Version: 0.19.1
// Date: 2026-09-24

//! Cut-attempt log and bounded reasoning trace (REQ-EE-2, REQ-EE-4).

use std::borrow::Cow;
use std::collections::BTreeMap;

use magi_core::reporting::{CompletionRecord, MagiReport};
use magi_core::rotation::AgentRotation;
use magi_core::schema::AgentName;

/// Tracing target of the cut-attempt WARN line (REQ-EE-2).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CUT_ATTEMPT_TARGET: &str = "magi_rs::consult::cut_attempt";
/// Tracing target of the reasoning-trace INFO lines (REQ-EE-4). Its own target so an operator
/// can raise or silence the trace per REQ-L30/L31 without touching the WARN line.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const REASONING_TRACE_TARGET: &str = "magi_rs::consult::reasoning_trace";
/// Characters kept at EACH end of a long trace (spec §0 named constants; OQ-4, user decision
/// 2026-09-24). A trace of at most twice this length is written whole.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const TRACE_EXCERPT_CHARS: usize = 4096;

/// The cut attempts of one consult, ready to log (REQ-EE-2/-4).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct CutAttemptLog<'a> {
    /// The cut attempts, in seat order then attempt order.
    cuts: Vec<magi_rs::magi::completion_report::CutAttempt<'a>>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl<'a> CutAttemptLog<'a> {
    /// Locates the cut attempts through [`magi_rs::magi::completion_report::cut_attempts`].
    #[must_use]
    pub(crate) fn from_parts(
        completions: &'a BTreeMap<AgentName, Vec<CompletionRecord>>,
        rotations: &BTreeMap<AgentName, AgentRotation>,
        failed_agents: &BTreeMap<AgentName, String>,
    ) -> Self {
        let _ = (completions, rotations, failed_agents);
        Self { cuts: Vec::new() }
    }

    /// `from_parts` over the report's three maps — the form the three consult sites call.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn from_report(report: &'a MagiReport) -> Self {
        Self::from_parts(
            &report.completions,
            &report.rotations,
            &report.failed_agents,
        )
    }

    /// The WARN texts, one per cut attempt, in order.
    #[must_use]
    pub(crate) fn lines(&self) -> Vec<String> {
        let _ = &self.cuts;
        Vec::new()
    }

    /// Emits the WARN lines and, for cut attempts carrying a trace, the INFO trace lines.
    pub(crate) fn emit(&self) {}
}

/// Makes foreign text safe inside ONE log line.
#[allow(dead_code)]
#[must_use]
pub(crate) fn line_safe(text: &str) -> Cow<'_, str> {
    Cow::Borrowed(text)
}

/// What of a trace reaches the log (REQ-EE-4).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TraceExcerpt {
    /// The masked trace fits in `2 * TRACE_EXCERPT_CHARS` characters.
    Whole(String),
    /// The first and last `TRACE_EXCERPT_CHARS` characters of the masked trace.
    HeadTail {
        /// The first `TRACE_EXCERPT_CHARS` characters.
        head: String,
        /// The last `TRACE_EXCERPT_CHARS` characters.
        tail: String,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
impl TraceExcerpt {
    /// Masks the FULL trace, then slices it on character boundaries.
    #[must_use]
    pub(crate) fn from_trace(raw: &str) -> Self {
        let _ = raw;
        Self::Whole(String::new())
    }
}

/// Startup notice: `reasoning_trace = true` has no effect when the file filter excludes INFO
/// for [`REASONING_TRACE_TARGET`] (REQ-EE-4).
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn reasoning_trace_filtered_notice(
    file_filter: &magi_rs::logging::filter::Filter,
    reasoning_trace: bool,
) -> Option<magi_rs::notices::Notice> {
    let _ = (file_filter, reasoning_trace);
    None
}

/// Unit tests for the cut-attempt log, the trace excerpt and the startup notice.
#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::schema::AgentName;
    use serde_json::json;
    use std::collections::BTreeMap;

    /// One event as the file branch would render it.
    #[derive(Debug)]
    struct Captured {
        /// The event's level.
        level: tracing::Level,
        /// The event's target.
        target: String,
        /// The event rendered by the production renderer.
        line: String,
    }

    /// Runs `emit` under a scoped subscriber (`tracing::subscriber::with_default`) whose only
    /// layer renders every event through `magi_rs::logging::render::render_event` and keeps level
    /// and target from the event's metadata. Returns every event, in emission order.
    fn capture_all(emit: impl FnOnce()) -> Vec<Captured> {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt as _;

        struct CaptureLayer(Arc<Mutex<Vec<Captured>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                let meta = event.metadata();
                if let Ok(mut g) = self.0.lock() {
                    g.push(Captured {
                        level: *meta.level(),
                        target: meta.target().to_string(),
                        line: magi_rs::logging::render::render_event(event),
                    });
                }
            }
        }
        let buf: Arc<Mutex<Vec<Captured>>> = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&buf)));
        tracing::subscriber::with_default(subscriber, emit);
        let mut guard = buf.lock().expect("never poisoned");
        std::mem::take(&mut *guard)
    }

    /// `record` over magi-core's `Deserialize`, as in `completion_report::tests`.
    fn record(json: serde_json::Value) -> CompletionRecord {
        serde_json::from_value(json).expect("the fixture must match magi-core's shape")
    }

    /// Exactly `chars` characters of plain prose ("the model weighs the diff. " cycled): no run of
    /// 32 `[A-Za-z0-9+/=_-]`, no URL, no control character, so no masking pass touches it.
    fn prose(chars: usize) -> String {
        "the model weighs the diff. "
            .chars()
            .cycle()
            .take(chars)
            .collect()
    }

    /// `prose(total)` with `secret` written over the characters `[at, at + secret.chars().count())`.
    fn prose_with(total: usize, at: usize, secret: &str) -> String {
        let secret: Vec<char> = secret.chars().collect();
        prose(total)
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i >= at && i < at + secret.len() {
                    secret[i - at]
                } else {
                    c
                }
            })
            .collect()
    }

    /// A cut attempt: `finish: length`, the budget spent, reasoning measured.
    fn cut_record(model: &str) -> CompletionRecord {
        record(json!({
            "model": model, "cap": 16_384, "finish": "length",
            "completion_tokens": 16_384, "prompt_tokens": 14_000,
            "reasoning": { "Measured": { "chars": 81_920, "text": null } },
            "control": "default",
        }))
    }

    /// S-2, log half: the exact line, with the same numbers the JSON carries. Nothing derived
    /// by division, no estimate: counts and states only.
    #[test]
    fn a_cut_attempt_line_names_seat_model_cap_counts_state_and_control() {
        let mut completions = BTreeMap::new();
        completions.insert(AgentName::Caspar, vec![cut_record("glm-5.2:cloud")]);

        let log = CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(
            log.lines(),
            vec![
                "cut attempt: seat=caspar attempt=1 model=glm-5.2:cloud cause=length cap=16384 \
                 completion_tokens=16384 prompt_tokens=14000 reasoning=Measured \
                 reasoning_chars=81920 control=default"
                    .to_string()
            ]
        );
    }

    /// The model name is foreign text: a newline or an escape sequence in it must not split the
    /// WARN line or forge a second one (CP2 seg2 loop 1, Caspar).
    #[test]
    fn a_model_name_with_control_characters_stays_on_one_line() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![cut_record(
                "glm\ncut attempt: seat=forged\u{2028}x\u{1b}[31m",
            )],
        );
        let lines =
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).lines();
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].contains('\n'), "one line: {:?}", lines[0]);
        assert!(
            !lines[0].contains('\u{1b}'),
            "no escape reaches the log: {:?}",
            lines[0]
        );
        assert!(
            !lines[0].contains('\u{2028}'),
            "no Unicode line separator: {:?}",
            lines[0]
        );
    }

    /// Absent counts read `unreported`, never `0`; a state with no count reads the same.
    #[test]
    fn an_unreported_count_reads_unreported_and_never_zero() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Melchior,
            vec![record(json!({
                "model": "m", "cap": 16_384, "finish": "length",
                "completion_tokens": null, "prompt_tokens": null,
                "reasoning": "NotMeasured", "control": "disabled",
            }))],
        );

        let lines =
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).lines();
        assert_eq!(
            lines,
            vec![
                "cut attempt: seat=melchior attempt=1 model=m cause=length cap=16384 \
                 completion_tokens=unreported prompt_tokens=unreported reasoning=NotMeasured \
                 reasoning_chars=unreported control=disabled"
                    .to_string()
            ]
        );
    }

    /// An ignored switch logs as `Unsupported` with the count that came back anyway.
    #[test]
    fn an_unsupported_state_logs_its_tag_and_the_count_that_came_back() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Balthasar,
            vec![record(json!({
                "model": "gpt-oss:120b-cloud", "cap": 16_384, "finish": "length",
                "completion_tokens": 16_384, "prompt_tokens": 9_000,
                "reasoning": { "Unsupported": { "backend": "ollama", "chars": 40_000, "text": null } },
                "control": "disabled",
            }))],
        );

        let line = CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new())
            .lines()
            .remove(0);
        assert!(
            line.contains(" reasoning=Unsupported reasoning_chars=40000 "),
            "{line}"
        );
    }

    /// The model is a foreign string: redacted, and the tag survives.
    #[test]
    fn the_model_in_a_cut_line_is_redacted_without_losing_the_tag() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![cut_record(&format!(
                "glm-5.2:cloud via http://alice:{CANARY}@h:11434"
            ))],
        );

        let line = CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new())
            .lines()
            .remove(0);
        assert!(!line.contains(CANARY), "{line}");
        assert!(line.contains("model=glm-5.2:cloud"), "{line}");
    }

    /// S-2 log half, through the real render path: exactly one WARN event per cut attempt, on
    /// its own target, under this process's run id — and nothing for a clean attempt.
    #[test]
    fn a_cut_attempt_emits_one_warn_line_under_the_run_id_and_a_clean_one_emits_none() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![
                cut_record("glm-5.2:cloud"),
                record(json!({
                    "model": "glm-5.2:cloud", "cap": 16_384, "finish": "stop",
                    "completion_tokens": 900, "prompt_tokens": 14_000,
                    "reasoning": { "Measured": { "chars": 3_000, "text": null } },
                })),
            ],
        );

        let events = capture_all(|| {
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).emit()
        });
        let warns: Vec<&Captured> = events
            .iter()
            .filter(|e| e.target == CUT_ATTEMPT_TARGET)
            .collect();
        assert_eq!(warns.len(), 1, "one line per CUT attempt: {events:?}");
        assert_eq!(warns[0].level, tracing::Level::WARN);
        assert!(
            warns[0]
                .line
                .contains(&format!("run={}", magi_rs::logging::run_id())),
            "{}",
            warns[0].line
        );
        assert!(warns[0].line.contains("cap=16384"), "{}", warns[0].line);
        assert!(
            warns[0].line.contains("reasoning_chars=81920"),
            "{}",
            warns[0].line
        );
    }

    /// S-8: with a captured trace, a cut attempt logs its excerpt at INFO on the trace target —
    /// never inside the WARN line, which reaches the screen (REQ-L19).
    #[test]
    fn a_cut_attempt_with_a_trace_logs_it_at_info_and_never_in_the_warn_line() {
        let trace = prose(100);
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![record(json!({
                "model": "glm-5.2:cloud", "cap": 16_384, "finish": "length",
                "completion_tokens": 16_384, "prompt_tokens": 14_000,
                "reasoning": { "Measured": { "chars": 100, "text": trace } },
            }))],
        );

        let events = capture_all(|| {
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).emit()
        });
        let traces: Vec<&Captured> = events
            .iter()
            .filter(|e| e.target == REASONING_TRACE_TARGET)
            .collect();
        assert_eq!(
            traces.len(),
            1,
            "short trace ⇒ one `whole` part: {events:?}"
        );
        assert_eq!(traces[0].level, tracing::Level::INFO);
        assert!(traces[0].line.contains("part=whole"), "{}", traces[0].line);
        assert!(traces[0].line.contains(&trace), "{}", traces[0].line);
        let warn = events
            .iter()
            .find(|e| e.target == CUT_ATTEMPT_TARGET)
            .expect("the cut is logged");
        assert!(
            !warn.line.contains(&trace),
            "trace text in the WARN line: {}",
            warn.line
        );
    }

    /// A trace line never spans lines: a newline inside the model's text is escaped, not
    /// written (CP2 seg2 loop 1, Caspar — log forging by foreign text).
    #[test]
    fn a_trace_with_newlines_is_logged_on_one_line_per_part() {
        let trace = format!("{}\ncut attempt: seat=forged{}", prose(40), prose(40));
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![record(json!({
                "model": "glm-5.2:cloud", "cap": 16_384, "finish": "length",
                "completion_tokens": 16_384, "prompt_tokens": 14_000,
                "reasoning": { "Measured": { "chars": 100, "text": trace } },
            }))],
        );
        let events = capture_all(|| {
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).emit()
        });
        let part = events
            .iter()
            .find(|e| e.target == REASONING_TRACE_TARGET)
            .expect("the trace is logged");
        assert!(!part.line.contains('\n'), "one line: {:?}", part.line);
        assert!(
            part.line.contains("\\ncut attempt: seat=forged"),
            "escaped: {}",
            part.line
        );
    }

    /// S-8: a non-cut attempt logs no trace text at all, even when magi-core captured one.
    #[test]
    fn a_clean_attempt_logs_no_trace_even_when_one_was_captured() {
        let trace = prose(100);
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Melchior,
            vec![record(json!({
                "model": "m", "cap": 16_384, "finish": "stop",
                "completion_tokens": 900, "prompt_tokens": 14_000,
                "reasoning": { "Measured": { "chars": 100, "text": trace } },
            }))],
        );

        let events = capture_all(|| {
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).emit()
        });
        assert!(
            events.iter().all(|e| !e.line.contains(&trace)),
            "a clean attempt leaked its trace: {events:?}"
        );
    }

    /// A long trace logs two parts, head then tail.
    #[test]
    fn a_long_trace_logs_a_head_part_and_a_tail_part() {
        let mut completions = BTreeMap::new();
        completions.insert(
            AgentName::Caspar,
            vec![record(json!({
                "model": "m", "cap": 16_384, "finish": "length",
                "completion_tokens": 16_384, "prompt_tokens": 1,
                "reasoning": { "Measured": { "chars": 10_000, "text": prose(10_000) } },
            }))],
        );

        let events = capture_all(|| {
            CutAttemptLog::from_parts(&completions, &BTreeMap::new(), &BTreeMap::new()).emit()
        });
        let parts: Vec<bool> = events
            .iter()
            .filter(|e| e.target == REASONING_TRACE_TARGET)
            .map(|e| e.line.contains("part=head"))
            .collect();
        assert_eq!(
            parts,
            vec![true, false],
            "head first, then tail: {events:?}"
        );
    }

    /// Boundary: a masked trace of at most 2 × 4096 characters is written whole.
    #[test]
    fn a_trace_up_to_twice_the_bound_is_written_whole() {
        for len in [0, 100, TRACE_EXCERPT_CHARS, 2 * TRACE_EXCERPT_CHARS] {
            let raw = prose(len);
            assert_eq!(
                TraceExcerpt::from_trace(&raw),
                TraceExcerpt::Whole(raw.clone()),
                "len {len}"
            );
        }
    }

    /// Boundary: one character over and the trace becomes a head and a tail of 4096 each.
    #[test]
    fn a_trace_one_char_over_twice_the_bound_is_cut_into_a_head_and_a_tail() {
        let raw = prose(2 * TRACE_EXCERPT_CHARS + 1);
        let TraceExcerpt::HeadTail { head, tail } = TraceExcerpt::from_trace(&raw) else {
            panic!("8193 characters must be cut");
        };
        assert_eq!(head.chars().count(), TRACE_EXCERPT_CHARS);
        assert_eq!(tail.chars().count(), TRACE_EXCERPT_CHARS);
        assert!(raw.starts_with(&head));
        assert!(raw.ends_with(&tail));
    }

    /// MAGI loop 3: a 33-character secret straddling offset 4096 is masked WHOLE, because the
    /// masking runs over the full trace before the cut. Cut first, and each fragment is shorter
    /// than the 32-character matcher and ships unmasked.
    ///
    /// MUTATION (required): slice before masking and this goes red on the head fragment.
    #[test]
    fn a_secret_straddling_the_head_cut_is_masked_whole() {
        const SECRET: &str = "Zx9Q2mP7vL4kR8sT1wY6nB3cD5fG0hJ2a"; // 33 chars, base64-shaped
        let raw = prose_with(10_000, TRACE_EXCERPT_CHARS - 16, SECRET);
        let TraceExcerpt::HeadTail { head, tail } = TraceExcerpt::from_trace(&raw) else {
            panic!("10 000 characters must be cut");
        };
        let first_half: String = SECRET.chars().take(16).collect();
        let second_half: String = SECRET.chars().skip(16).collect();
        for part in [&head, &tail] {
            assert!(
                !part.contains(&first_half),
                "the head fragment survived: {part}"
            );
            assert!(
                !part.contains(&second_half),
                "the tail fragment survived: {part}"
            );
        }
        assert!(
            head.contains("***"),
            "the secret was masked, not dropped silently"
        );
    }

    /// A REGISTERED secret (≥ 8 bytes, not key-shaped) straddling the cut is masked whole too:
    /// the exact pass has to see the whole value, which it only does before the cut.
    #[test]
    fn a_registered_secret_straddling_the_cut_is_masked_whole() {
        const VALUE: &str = "plain-words-passphrase";
        magi_rs::logging::register_process_secrets(&[(
            magi_rs::logging::auditor::SecretName::new("A_TRACE_STRADDLE_GUARD_SECRET"),
            VALUE,
        )]);
        let raw = prose_with(10_000, TRACE_EXCERPT_CHARS - 10, VALUE);
        let TraceExcerpt::HeadTail { head, .. } = TraceExcerpt::from_trace(&raw) else {
            panic!("10 000 characters must be cut");
        };
        assert!(!head.contains("plain-word"), "{head}");
    }

    /// A credential inside a URL straddling the cut is removed by `redact_foreign_text`.
    #[test]
    fn a_url_credential_straddling_the_cut_is_masked_whole() {
        const CANARY: &str = "c4n4ry-s3cr3t-c4n4ry";
        let url = format!("http://alice:{CANARY}@host:11434/v1");
        let raw = prose_with(10_000, TRACE_EXCERPT_CHARS - 20, &url);
        let excerpt = TraceExcerpt::from_trace(&raw);
        let text = format!("{excerpt:?}");
        assert!(!text.contains("c4n4ry"), "{text}");
    }

    /// Multi-byte text at the cut: counted and sliced in characters, never bytes.
    #[test]
    fn a_multibyte_trace_is_cut_on_character_boundaries() {
        let raw: String = "日本語の推論 ".chars().cycle().take(9_000).collect();
        let TraceExcerpt::HeadTail { head, tail } = TraceExcerpt::from_trace(&raw) else {
            panic!("9 000 characters must be cut");
        };
        assert_eq!(head.chars().count(), TRACE_EXCERPT_CHARS);
        assert_eq!(tail.chars().count(), TRACE_EXCERPT_CHARS);
        assert!(raw.starts_with(&head) && raw.ends_with(&tail));
    }

    /// `sanitize_text` runs first: escape sequences and control characters never reach the log.
    #[test]
    fn terminal_escapes_are_stripped_from_the_trace() {
        let excerpt = TraceExcerpt::from_trace("red \u{1b}[31mALERT\u{1b}[0m bell\u{7} done");
        assert_eq!(
            excerpt,
            TraceExcerpt::Whole("red ALERT bell done".to_string())
        );
    }

    /// REQ-EE-4: a file filter above INFO for the trace target makes the flag a no-op, and
    /// that is said at startup. The flag off, or INFO admitted, says nothing.
    #[test]
    fn a_filter_that_drops_the_trace_is_announced_only_when_the_flag_is_on() {
        use magi_rs::logging::filter::Filter;
        let warn = Filter::parse("warn").expect("valid");
        let info = Filter::parse("info").expect("valid");
        let targeted = Filter::parse("warn,magi_rs::consult::reasoning_trace=info").expect("valid");

        let notice = reasoning_trace_filtered_notice(&warn, true).expect("announced");
        assert_eq!(
            notice.level,
            tracing::Level::WARN,
            "it must reach the screen"
        );
        assert!(notice.text.contains("reasoning_trace"), "{}", notice.text);
        assert!(
            notice.text.contains(REASONING_TRACE_TARGET),
            "{}",
            notice.text
        );
        assert!(reasoning_trace_filtered_notice(&warn, false).is_none());
        assert!(reasoning_trace_filtered_notice(&info, true).is_none());
        assert!(
            reasoning_trace_filtered_notice(&targeted, true).is_none(),
            "a per-target directive admitting INFO is honoured (longest match)"
        );
    }

    /// Wiring guard: every surface where a consult report arrives builds the cut-attempt log.
    /// Source-grep because the three arms sit behind real network calls; the needle is split so
    /// this file never satisfies it (CRLF-safe, escaped needle — CLAUDE.md v0.18.0).
    #[test]
    fn every_consult_surface_logs_its_cut_attempts() {
        let needle = format!("{}::from_report(", "CutAttemptLog");
        for (file, source) in [
            ("src/tools/consult.rs", include_str!("tools/consult.rs")),
            ("src/headless_runner.rs", include_str!("headless_runner.rs")),
            ("src/tui/mod.rs", include_str!("tui/mod.rs")),
        ] {
            let source = source.replace('\r', "");
            assert_eq!(
                source.matches(needle.as_str()).count(),
                1,
                "{file} must log the cut attempts exactly once, where its report arrives"
            );
        }
    }

    /// Wiring guard: the startup notice is collected on both surfaces (TUI and headless).
    #[test]
    fn both_surfaces_announce_a_filtered_trace() {
        let needle = format!("{}(", "reasoning_trace_filtered_notice");
        let main = include_str!("main.rs").replace('\r', "");
        assert_eq!(
            main.matches(needle.as_str()).count(),
            2,
            "TUI and headless startup"
        );
    }
}
