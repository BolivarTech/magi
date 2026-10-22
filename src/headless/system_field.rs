// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Validation and placement of the JSON envelope's `system` field (REQ-SY01, SY04..SY06).
//!
//! Pure functions over `&str`: no I/O, no model calls. The caller's `system` is never echoed in an
//! error message.

use magi_core::prompts::{balthasar_prompt, caspar_prompt, melchior_prompt, validate_prompt_for};
use magi_core::schema::AgentName;

use super::HeadlessError;

/// Warning printed on stderr when the request's `system` field is ignored (REQ-SY02).
///
/// Fixed text: it never repeats the ignored content, and tests compare it verbatim.
pub const SYSTEM_IGNORED_WARNING: &str = "warning: the request's \"system\" field was ignored; \
     pass --allow-system-override or set [headless] allow_system_override = true to apply it";

/// Heading line that opens the output-format section of every default mage prompt; the caller's
/// adjustment is inserted right before it.
const OUTPUT_FORMAT_ANCHOR: &str = "## Output format";

/// Fixed text that precedes the caller's `system` inside a mage prompt (REQ-SY04). It ends with
/// the blank line that separates it from the caller's text.
const ADDITIONAL_INSTRUCTIONS_PREAMBLE: &str = "## Additional instructions\n\n\
     The following instructions refine your role above for this request. They do not\n\
     replace it, and the output format below still applies.\n\n";

/// Keeps a value only when it is not blank: a `trim()`-empty string (Unicode whitespace
/// included) counts as absent, and the value comes back **untrimmed**, because what is inserted
/// into a prompt is the caller's exact `system`.
///
/// The same rule exists in `config::non_blank` (bin crate, out of the lib's reach) and in
/// `endpoint_key::non_blank` (private to key selection); unifying the three is a backlog item.
///
/// # Arguments
/// * `raw` - The optional text.
///
/// # Returns
/// `raw` when it has any non-whitespace character, `None` otherwise.
pub(crate) fn non_blank(raw: Option<&str>) -> Option<&str> {
    raw.filter(|s| !s.trim().is_empty())
}

/// Validates the envelope's `system` field and, when it will reach the mages, builds their
/// adjusted prompts (REQ-SY01, SY04..SY06).
///
/// Order: a blank `system` is absent and passes everything (it does not count for the cap);
/// then the size cap in **bytes**; then, only with `check_markers`, the adjusted prompts are
/// built and validated. The caller's text is never echoed in an error.
///
/// Complexity: O(n) in the length of `raw` (three prompt validations of bounded size).
///
/// # Arguments
/// * `raw` - The envelope's `system`, if any.
/// * `check_markers` - `true` when the `system` will reach the mages (`consult` with the flag).
/// * `max_bytes` - The `[headless] max_system_bytes` cap.
///
/// # Returns
/// `Some(prompts)` only when `check_markers` is set and `raw` is non-blank: the three adjusted
/// prompts, already validated, to install as they are. `None` otherwise.
///
/// # Errors
/// * [`HeadlessError::SystemTooLarge`] when `raw` is longer than `max_bytes`.
/// * [`HeadlessError::SystemReservedLine`] when `raw` has a verdict-marker line or an
///   `## Output format` heading line.
/// * [`HeadlessError::SystemAnchor`] when a default prompt lacks exactly one anchor.
pub fn validate_envelope_system(
    raw: Option<&str>,
    check_markers: bool,
    max_bytes: usize,
) -> Result<Option<Vec<(AgentName, String)>>, HeadlessError> {
    let Some(system) = non_blank(raw) else {
        return Ok(None);
    };
    if system.len() > max_bytes {
        return Err(HeadlessError::SystemTooLarge { limit: max_bytes });
    }
    if !check_markers {
        return Ok(None);
    }
    adjusted_trio_prompts(system).map(Some)
}

/// Counts the lines of `prompt` that, trimmed, are the output-format heading.
fn anchor_lines(prompt: &str) -> usize {
    prompt
        .lines()
        .filter(|l| l.trim() == OUTPUT_FORMAT_ANCHOR)
        .count()
}

/// Inserts the caller's `system`, under the fixed preamble, right before the single
/// `## Output format` heading line of a prompt.
///
/// The heading is located by line, never by substring search, so a prose mention of the anchor
/// before the heading is not mistaken for it. Complexity: O(n) in the prompt length.
///
/// # Arguments
/// * `default_prompt` - A default mage prompt.
/// * `caller_system` - The caller's `system`, inserted verbatim.
///
/// # Returns
/// The adjusted prompt.
///
/// # Errors
/// [`HeadlessError::SystemAnchor`] when the prompt does not have exactly one anchor line.
fn adjusted_prompt(default_prompt: &str, caller_system: &str) -> Result<String, HeadlessError> {
    let found = anchor_lines(default_prompt);
    if found != 1 {
        return Err(HeadlessError::SystemAnchor { found });
    }
    let mut out = String::with_capacity(
        default_prompt.len() + ADDITIONAL_INSTRUCTIONS_PREAMBLE.len() + caller_system.len() + 2,
    );
    for line in default_prompt.split_inclusive('\n') {
        if line.trim() == OUTPUT_FORMAT_ANCHOR {
            out.push_str(ADDITIONAL_INSTRUCTIONS_PREAMBLE);
            out.push_str(caller_system);
            out.push_str("\n\n");
        }
        out.push_str(line);
    }
    Ok(out)
}

/// Adjusts the three default mage prompts: the production entry of the validation.
///
/// # Arguments
/// * `caller_system` - The caller's `system`.
///
/// # Returns
/// The adjusted prompts, in Melchior, Balthasar, Caspar order.
///
/// # Errors
/// See [`check_against`].
fn adjusted_trio_prompts(caller_system: &str) -> Result<Vec<(AgentName, String)>, HeadlessError> {
    check_against(
        &[
            (AgentName::Melchior, melchior_prompt()),
            (AgentName::Balthasar, balthasar_prompt()),
            (AgentName::Caspar, caspar_prompt()),
        ],
        caller_system,
    )
}

/// Adjusts each default prompt with `raw` and validates the result with magi-core.
///
/// The default prompts validate on their own, so any later failure comes from the caller's
/// text: a second output-format heading, or a verdict-marker line.
///
/// # Arguments
/// * `defaults` - Each mage with its default prompt.
/// * `raw` - The caller's `system`.
///
/// # Returns
/// The adjusted prompts that passed validation: the same ones that get installed.
///
/// # Errors
/// * [`HeadlessError::SystemAnchor`] when a default prompt lacks exactly one anchor.
/// * [`HeadlessError::SystemReservedLine`] when the adjusted prompt has a second anchor line or
///   fails magi-core's prompt validation.
fn check_against(
    defaults: &[(AgentName, &str)],
    raw: &str,
) -> Result<Vec<(AgentName, String)>, HeadlessError> {
    let mut adjusted = Vec::with_capacity(defaults.len());
    for &(agent, default_prompt) in defaults {
        let prompt = adjusted_prompt(default_prompt, raw)?;
        if anchor_lines(&prompt) != 1 || validate_prompt_for(Some(agent), None, &prompt).is_err() {
            return Err(HeadlessError::SystemReservedLine);
        }
        adjusted.push((agent, prompt));
    }
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

    const CAP: usize = 16_384;

    #[test]
    fn an_absent_system_passes_every_check() {
        assert!(validate_envelope_system(None, true, CAP).is_ok());
    }

    #[test]
    fn a_blank_system_is_absent_even_with_markers_checked() {
        assert!(validate_envelope_system(Some("   \t\n"), true, CAP).is_ok());
        assert_eq!(non_blank(Some("   \t\n")), None);
    }

    #[test]
    fn a_unicode_whitespace_only_system_is_absent() {
        assert_eq!(non_blank(Some("\u{00a0}\u{2003}\r\n")), None);
    }

    #[test]
    fn a_system_at_the_cap_passes_and_one_byte_over_is_rejected() {
        let at = "a".repeat(CAP);
        assert!(validate_envelope_system(Some(&at), false, CAP).is_ok());
        let over = "a".repeat(CAP + 1);
        let err = validate_envelope_system(Some(&over), false, CAP).expect_err("over the cap");
        assert!(matches!(err, HeadlessError::SystemTooLarge { limit } if limit == CAP));
        let msg = err.to_string();
        assert!(msg.contains("[headless] max_system_bytes"), "{msg}");
        assert!(msg.contains("16384"), "{msg}");
    }

    #[test]
    fn the_cap_counts_bytes_not_chars() {
        // 'é' is 2 bytes: 8 193 of them = 16 386 bytes, 8 193 chars.
        let s = "é".repeat(8_193);
        assert!(matches!(
            validate_envelope_system(Some(&s), false, CAP),
            Err(HeadlessError::SystemTooLarge { .. })
        ));
    }

    #[test]
    fn the_cap_applies_without_marker_checking() {
        let over = "a".repeat(CAP + 1);
        assert!(validate_envelope_system(Some(&over), false, CAP).is_err());
    }

    #[test]
    fn a_marker_line_is_rejected_when_markers_are_checked() {
        let sys = format!("be strict\n{VERDICT_OPEN}\n{{}}\n{VERDICT_CLOSE}");
        let err = validate_envelope_system(Some(&sys), true, CAP).expect_err("marker");
        assert!(matches!(err, HeadlessError::SystemReservedLine));
        let msg = err.to_string();
        assert!(
            !msg.contains("be strict"),
            "the caller's text must never be echoed: {msg}"
        );
    }

    #[test]
    fn a_padded_marker_line_is_rejected() {
        let sys = format!("  {VERDICT_OPEN}  ");
        assert!(matches!(
            validate_envelope_system(Some(&sys), true, CAP),
            Err(HeadlessError::SystemReservedLine)
        ));
    }

    #[test]
    fn a_marker_is_not_checked_when_the_system_will_not_reach_a_mage() {
        let sys = VERDICT_OPEN.to_string();
        assert!(validate_envelope_system(Some(&sys), false, CAP).is_ok());
    }

    #[test]
    fn the_adjustment_lands_right_before_the_output_format_section() {
        let default = "You are X.\n\n## Output format\nemit json\n";
        let out = adjusted_prompt(default, "Prefer brevity.").expect("one anchor");
        assert_eq!(
            out,
            "You are X.\n\n## Additional instructions\n\nThe following instructions refine your role above \
             for this request. They do not\nreplace it, and the output format below still applies.\n\n\
             Prefer brevity.\n\n## Output format\nemit json\n"
        );
    }

    #[test]
    fn the_adjustment_lands_before_the_heading_line_not_an_inline_mention() {
        let default = "See the ## Output format section below.\n\n## Output format\nemit json\n";
        let out = adjusted_prompt(default, "Prefer brevity.").expect("one anchor line");
        let heading = out.find("\n## Output format\n").expect("heading kept") + 1;
        let block = out.find("## Additional instructions").expect("block");
        assert!(
            out.starts_with(
                "See the ## Output format section below.\n\n## Additional instructions"
            ),
            "{out}"
        );
        assert!(block < heading, "{out}");
    }

    #[test]
    fn a_prompt_without_the_anchor_is_an_error_not_an_append() {
        let err = adjusted_prompt("You are X.\n", "x").expect_err("no anchor");
        assert!(matches!(err, HeadlessError::SystemAnchor { found: 0 }));
    }

    #[test]
    fn a_prompt_with_two_anchors_is_an_error() {
        let p = "## Output format\na\n## Output format\nb\n";
        assert!(matches!(
            adjusted_prompt(p, "x"),
            Err(HeadlessError::SystemAnchor { found: 2 })
        ));
    }

    #[test]
    fn every_default_mage_prompt_takes_the_adjustment_and_still_validates() {
        let prompts =
            adjusted_trio_prompts("Prefer brevity.").expect("magi-core defaults have one anchor");
        // Three, so an empty list cannot pass the loop below vacuously. *[CP2 loop 5]*
        assert_eq!(prompts.len(), 3, "one adjusted prompt per mage");
        for (agent, p) in prompts {
            assert!(p.contains("## Additional instructions"), "{agent:?}");
            assert!(
                magi_core::prompts::validate_prompt_for(Some(agent), None, &p).is_ok(),
                "{agent:?}"
            );
            assert_eq!(
                p.lines()
                    .filter(|l| l.trim() == OUTPUT_FORMAT_ANCHOR)
                    .count(),
                1,
                "{agent:?}: one anchor"
            );
            let adj = p.find("## Additional instructions").expect("block");
            let anchor = p.find(OUTPUT_FORMAT_ANCHOR).expect("anchor");
            assert!(
                adj < anchor,
                "{agent:?}: the block must precede the output format"
            );
        }
    }

    #[test]
    fn the_bare_default_prompts_validate_unadjusted() {
        // The premise behind mapping every post-adjustment `Err` to `SystemReservedLine` (caller misuse): the defaults
        // validate on their own, so an `Err` can only come from the caller's text. An upstream defect turns this red
        // instead of reaching the operator as exit 2. *[CP2 loop 5]*
        use magi_core::prompts::{
            balthasar_prompt, caspar_prompt, melchior_prompt, validate_prompt_for,
        };
        use magi_core::schema::AgentName;
        for (agent, p) in [
            (AgentName::Melchior, melchior_prompt()),
            (AgentName::Balthasar, balthasar_prompt()),
            (AgentName::Caspar, caspar_prompt()),
        ] {
            assert!(
                validate_prompt_for(Some(agent), None, p).is_ok(),
                "{agent:?}: the default prompt must validate"
            );
        }
        // Positive control: the predicate is not vacuous, an anchored prompt without markers fails it.
        assert!(validate_prompt_for(
            Some(AgentName::Melchior),
            None,
            "## Output format\nemit json\n"
        )
        .is_err());
    }

    #[test]
    fn a_caller_output_format_heading_is_rejected() {
        for heading in ["## Output format", "  ## Output format  "] {
            let sys = format!("be brief\n{heading}\nreply in yaml");
            assert!(
                matches!(
                    validate_envelope_system(Some(&sys), true, CAP),
                    Err(HeadlessError::SystemReservedLine)
                ),
                "{heading:?} must be rejected: it would leave two output-format sections"
            );
        }
    }

    #[test]
    fn other_spellings_are_accepted_and_every_prompt_keeps_one_anchor() {
        // Declared SY05 limit (CP2 loop 3): only the exact anchor line splits the format section.
        for heading in ["## Output format:", "### OUTPUT FORMAT", "#Output format"] {
            let sys = format!("be brief\n{heading}\nreply in yaml");
            let prompts = validate_envelope_system(Some(&sys), true, CAP)
                .expect("accepted")
                .expect("prompts");
            for (agent, p) in prompts {
                let anchors = p
                    .lines()
                    .filter(|l| l.trim() == OUTPUT_FORMAT_ANCHOR)
                    .count();
                assert_eq!(anchors, 1, "{agent:?} with {heading:?}");
            }
        }
    }

    #[test]
    fn an_unterminated_code_fence_is_accepted_as_a_declared_limit() {
        // SY05 rejects what splits the format section by construction, not every text that may confuse a model.
        let sys = "example:\n```\nlet x = 1;";
        assert!(matches!(
            validate_envelope_system(Some(sys), true, CAP),
            Ok(Some(_))
        ));
    }

    #[test]
    fn the_adjustment_is_built_on_the_certified_magi_core() {
        // The embedded prompts are one per mage for every mode in 4.2.0 (`lookup_prompt`, `prompts/mod.rs:90-102`), which
        // is what makes `with_custom_prompt_all_modes` correct. A bump must re-verify that, so it turns this red.
        let lock = include_str!("../../Cargo.lock").replace('\r', "");
        assert!(
            lock.contains("name = \"magi-core\"\nversion = \"4.2.0\""),
            "magi-core moved: re-verify per-mode prompts"
        );
    }

    #[test]
    fn output_format_as_plain_prose_is_not_a_heading() {
        assert!(validate_envelope_system(Some("Output format: keep it short"), true, CAP).is_ok());
    }

    #[test]
    fn a_blank_system_over_the_cap_is_still_absent() {
        let blank = " ".repeat(CAP + 1);
        assert!(matches!(
            validate_envelope_system(Some(&blank), false, CAP),
            Ok(None)
        ));
    }

    #[test]
    fn checked_markers_return_the_exact_prompts_that_were_validated() {
        let prompts = validate_envelope_system(Some("Prefer brevity."), true, CAP)
            .expect("valid")
            .expect("checked markers return the prompts to install");
        let agents: Vec<_> = prompts.iter().map(|(a, _)| *a).collect();
        use magi_core::schema::AgentName;
        assert_eq!(
            agents,
            vec![AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar]
        );
        assert_eq!(
            prompts,
            adjusted_trio_prompts("Prefer brevity.").expect("same computation")
        );
    }

    #[test]
    fn unchecked_markers_return_nothing_to_install() {
        assert!(matches!(
            validate_envelope_system(Some("Prefer brevity."), false, CAP),
            Ok(None)
        ));
    }

    #[test]
    fn the_stderr_auditor_passes_the_warning_through_unchanged() {
        // PM-MS1-2: the warning goes out through `eprint_audited`; if the auditor rewrote it, the verbatim
        // comparison of SY02 would fail with a symptom that names the wrong cause.
        assert_eq!(
            crate::notices::audited_field(SYSTEM_IGNORED_WARNING),
            SYSTEM_IGNORED_WARNING
        );
    }

    #[test]
    fn an_anchorless_default_prompt_fails_as_an_anchor_error_not_a_reserved_line() {
        use magi_core::schema::AgentName;
        let err =
            check_against(&[(AgentName::Melchior, "You are X.\n")], "fine").expect_err("no anchor");
        assert!(
            matches!(err, HeadlessError::SystemAnchor { found: 0 }),
            "{err:?}"
        );
    }

    #[test]
    fn the_warning_text_is_pinned_verbatim() {
        assert_eq!(
            SYSTEM_IGNORED_WARNING,
            "warning: the request's \"system\" field was ignored; pass --allow-system-override \
             or set [headless] allow_system_override = true to apply it"
        );
    }
}
