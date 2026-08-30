// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Which events a branch writes (REQ-L29, REQ-L30, REQ-L31).
//!
//! # Why this is parsed here and not by `EnvFilter`
//!
//! `tracing_subscriber::EnvFilter` speaks a superset of this syntax, and using
//! it would mean enabling the `env-filter` feature, which pulls `matchers` and
//! `regex` in behind it. The grammar REQ-L30 actually asks for is a
//! comma-separated list of `target=level` with an optional bare level as the
//! default — small enough to parse exactly, with no dependency and no syntax
//! the operator can write that this file does not define. A directive language
//! wider than the one documented is a promise the docs do not make.

use tracing::Level;

use crate::logging::LoggingError;

/// Separator between directives in one specification.
const DIRECTIVE_SEPARATOR: char = ',';
/// Separator between a target and its level.
const TARGET_LEVEL_SEPARATOR: char = '=';
/// Character a crate name uses that a module path does not.
const DASH: char = '-';
/// What a dash normalises to (REQ-L31).
const UNDERSCORE: char = '_';

/// One `target=level` directive, with the prefix its children must match.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Override {
    /// The target as written, normalised.
    name: String,
    /// `name::`, so a child match is a `starts_with` and not a `format!`.
    prefix: String,
    /// The level it grants.
    level: Level,
}

/// A parsed filter: one default level plus any per-target overrides.
///
/// # Examples
///
/// ```
/// use magi_rs::logging::filter::Filter;
/// let f = Filter::parse("magi_rs=debug,warn").unwrap();
/// assert_eq!(f.level_for("magi_rs::agent"), tracing::Level::DEBUG);
/// assert_eq!(f.level_for("magi_core::http"), tracing::Level::WARN);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// Applied to any target no override matches.
    default: Level,
    /// Overrides, in declaration order.
    ///
    /// Each carries its name AND the `name::` prefix a child target must start
    /// with. **Precomputed**, because `level_for` runs on every event and
    /// building that prefix there allocated once per override per event, on the
    /// hot path, to produce the same string every time.
    targets: Vec<Override>,
}

impl Filter {
    /// Parses a specification.
    ///
    /// # Parameters
    ///
    /// * `spec` — e.g. `"info"` or `"magi_rs=debug,magi_core=warn,error"`.
    ///
    /// # Returns
    ///
    /// The filter, with every target name normalised `-` to `_`.
    ///
    /// # Errors
    ///
    /// [`LoggingError::FilterInvalid`] for an unknown level word, an empty
    /// target, or a directive with more than one `=`. **Never a silent
    /// fallback** (REQ-L31): a typo that quietly became `info` is a filter the
    /// operator believes is in effect and is not.
    ///
    /// # Complexity
    ///
    /// `O(n)` in the specification's length.
    pub fn parse(spec: &str) -> Result<Self, LoggingError> {
        let mut default = None;
        let mut targets = Vec::new();
        for raw in spec.split(DIRECTIVE_SEPARATOR) {
            let piece = raw.trim();
            if piece.is_empty() {
                continue;
            }
            let mut halves = piece.split(TARGET_LEVEL_SEPARATOR);
            let first = halves.next().unwrap_or_default().trim();
            match halves.next() {
                None => default = Some(parse_level(first, spec)?),
                Some(level) => {
                    if halves.next().is_some() {
                        return Err(invalid(spec, "a directive has more than one '='"));
                    }
                    if first.is_empty() {
                        return Err(invalid(spec, "a directive has no target before its '='"));
                    }
                    let name = first.replace(DASH, &UNDERSCORE.to_string());
                    targets.push(Override {
                        prefix: format!("{name}::"),
                        name,
                        level: parse_level(level.trim(), spec)?,
                    });
                }
            }
        }
        Ok(Self {
            default: default.unwrap_or(Level::INFO),
            targets,
        })
    }

    /// The level in force for `target`.
    ///
    /// **Longest match wins**, which is what makes a general rule and a specific
    /// one composable: `magi_rs=warn,magi_rs::agent=debug` has to give the agent
    /// `debug`, and declaration order is not a reliable way to say so.
    ///
    /// # Complexity
    ///
    /// `O(k)` with `k` the number of overrides.
    #[must_use]
    pub fn level_for(&self, target: &str) -> Level {
        let mut best: Option<(usize, Level)> = None;
        for o in &self.targets {
            let matches = target == o.name || target.starts_with(&o.prefix);
            if matches && best.is_none_or(|(len, _)| o.name.len() > len) {
                best = Some((o.name.len(), o.level));
            }
        }
        best.map_or(self.default, |(_, level)| level)
    }

    /// The most verbose level any target can reach.
    ///
    /// Used for `max_level_hint`, which is a global cut: a hint stricter than
    /// the filter silences events the filter would have passed.
    ///
    /// # Complexity
    ///
    /// `O(k)`.
    #[must_use]
    pub fn max_level(&self) -> Level {
        self.targets
            .iter()
            .fold(self.default, |acc, o| acc.max(o.level))
    }
}

/// Builds this module's error, naming the whole specification and the cause.
///
/// The WHOLE specification, not the offending piece: an operator reading
/// "invalid directive `verbose`" against a four-directive line still has to
/// find which line, and the value is theirs rather than a credential.
fn invalid(spec: &str, reason: &str) -> LoggingError {
    LoggingError::FilterInvalid {
        directive: spec.to_string(),
        reason: reason.to_string(),
    }
}

/// Maps a level word to its level.
///
/// # Errors
///
/// [`LoggingError::FilterInvalid`] for anything else.
fn parse_level(word: &str, spec: &str) -> Result<Level, LoggingError> {
    match word.to_ascii_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        other => Err(invalid(
            spec,
            &format!("{other:?} is not one of trace, debug, info, warn, error"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_level_applies_to_everything() {
        let f = Filter::parse("debug").expect("valid");
        assert_eq!(f.level_for("magi_rs::agent"), Level::DEBUG);
        assert_eq!(f.level_for("anything::at::all"), Level::DEBUG);
    }

    #[test]
    fn a_per_target_directive_overrides_the_default() {
        // REQ-L30's own example.
        let f = Filter::parse("magi_rs=debug,magi_core=warn,error").expect("valid");
        assert_eq!(f.level_for("magi_rs::agent"), Level::DEBUG);
        assert_eq!(f.level_for("magi_core::http"), Level::WARN);
        assert_eq!(f.level_for("some_other_crate"), Level::ERROR);
    }

    #[test]
    fn the_longest_matching_target_wins_not_the_last_declared() {
        // Without this, a general rule and a specific one cannot be combined:
        // whichever was written second would swallow the other.
        let f = Filter::parse("magi_rs=warn,magi_rs::agent=trace").expect("valid");
        assert_eq!(f.level_for("magi_rs::agent"), Level::TRACE);
        assert_eq!(f.level_for("magi_rs::tools"), Level::WARN);
    }

    #[test]
    fn a_target_written_with_dashes_matches_the_module_path() {
        // REQ-L31. An operator copies the crate name off Cargo.toml, where it
        // has dashes, and the module path it must match has underscores.
        let f = Filter::parse("magi-rs=trace").expect("valid");
        assert_eq!(f.level_for("magi_rs::agent"), Level::TRACE);
    }

    #[test]
    fn an_unknown_level_is_a_load_error_and_never_a_silent_info() {
        // REQ-L31's sharp end: a typo that quietly became `info` is a filter the
        // operator believes is in effect and is not.
        assert!(Filter::parse("verbose").is_err());
        assert!(Filter::parse("magi_rs=verbose").is_err());
        assert!(Filter::parse("magi_rs=debug,notalevel").is_err());
    }

    #[test]
    fn a_malformed_directive_is_a_load_error() {
        assert!(Filter::parse("=debug").is_err(), "an empty target");
        assert!(Filter::parse("a=b=c").is_err(), "two separators");
    }

    #[test]
    fn the_hint_is_the_most_verbose_level_any_target_can_reach() {
        // A hint stricter than the filter is a global cut that silences events
        // the filter would have passed, so it has to be the maximum.
        let f = Filter::parse("error,magi_rs=trace").expect("valid");
        assert_eq!(f.max_level(), Level::TRACE);
    }

    #[test]
    fn an_empty_specification_is_info_rather_than_an_error() {
        // A blank value is ABSENT, never invalid — the same rule the whole
        // configuration layer applies to an exported-but-unfilled variable.
        assert_eq!(
            Filter::parse("").expect("valid").level_for("x"),
            Level::INFO
        );
    }
}
