// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! The chokepoint: nothing reaches an output without passing through here.
//!
//! # Why this sits in front of the appender
//!
//! The vault derives with Argon2id, encrypts with AES-256-GCM-SIV, corrects with
//! FEC, keeps the data key masked in RAM, pins it with `mlock` and suppresses
//! core dumps. **One credential in the clear in a log file voids all of it.**
//! The auditor therefore runs before the writer, so no window exists in which
//! something raw is already on disk.
//!
//! # The guarantee is by type, not by convention
//!
//! [`Audited`] has private fields and exactly one constructor —
//! [`Auditor::audit`] — and every output accepts only an `Audited`. Handing a
//! raw `String` to a sink does not compile. [`AuditExempt`] is a **disjoint**
//! type rather than another `Audited`, so "exempt from the audit" shows up in a
//! diff instead of being an untyped convention.

use std::fmt;

/// A secret value that refuses to print itself.
///
/// `Debug` and `Display` both render `***`. Reading the value takes an explicit,
/// named call, so it cannot happen by accident.
///
/// # Why not `Zeroizing`
///
/// `Zeroizing<String>` forwards `Debug` to the inner value, so a `{:?}` on a
/// `tracing` field would print the secret — which REQ-L51 calls the single most
/// likely accident of all. Zeroizing solves a different problem (the bytes in
/// RAM); it does not solve this one.
#[derive(Clone)]
pub struct Secret(String);

/// What both `Debug` and `Display` render for a [`Secret`], and what the auditor
/// substitutes for a redacted range.
pub const REDACTED: &str = "***";

impl Secret {
    /// Wraps a value so it stops printing itself.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The value in the clear.
    ///
    /// **The name is the guard.** Every call site reads as an explicit decision
    /// to expose the value, which is exactly what a `{}` or a `{:?}` would not.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// The name of a registered secret — never its value.
///
/// A newtype over `&'static str`, and the lifetime is the point: a `'static`
/// string **cannot be built accidentally from a runtime value**, so a name can
/// never come from the thing it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SecretName(&'static str);

impl SecretName {
    /// Names a secret from a program constant.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The name as written.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for SecretName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Which subsystem is affected, and why.
///
/// The health tracker of MS2 keeps state **per subsystem**; the cause says the
/// reason currently in force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CauseKey {
    subsystem: &'static str,
    cause: &'static str,
}

impl CauseKey {
    /// Every declared cause.
    ///
    /// Exists so a test in MS2 can enumerate them against the message table;
    /// without it that guard cannot be written at all. Empty in MS1, populated
    /// by MS2's task 3.3.
    pub const ALL: &'static [CauseKey] = &[];

    /// The affected subsystem.
    #[must_use]
    pub const fn subsystem(&self) -> &'static str {
        self.subsystem
    }

    /// The reason in force.
    #[must_use]
    pub const fn cause(&self) -> &'static str {
        self.cause
    }
}

/// A line that has been through the auditor. **The only thing an output takes.**
///
/// Cloneable on purpose: the fan-out hands the same event to both branches, so
/// one of them needs its own copy. Wrapping in an `Arc` would save the `String`
/// copy at the cost of an atomic counter on the hot path, to avoid a memcpy of
/// at most 4 KiB. It clones.
///
/// **No `PartialEq`, no `Hash`, and that is deliberate**: comparing or indexing
/// redacted text invites using it as a map key, which is the path by which an
/// audited value ends up inside a structure whose `Debug` prints the whole
/// thing.
#[derive(Debug, Clone)]
pub struct Audited {
    line: String,
    cause: Option<CauseKey>,
    reserved_len: usize,
}

impl Audited {
    /// The redacted text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.line
    }

    /// The cause this line belongs to, when the emitter declared one.
    #[must_use]
    pub fn cause(&self) -> Option<CauseKey> {
        self.cause
    }

    /// The byte count reserved for this line **before** it was audited.
    ///
    /// The writer releases exactly this, never the length it is holding.
    /// Redaction *shortens* the line — each secret becomes a fixed-width
    /// `***` — so releasing the post-audit length would leave the channel's
    /// counter climbing monotonically on every line containing a secret or a
    /// redacted URL, until it refused new events with capacity to spare.
    #[must_use]
    pub fn reserved_len(&self) -> usize {
        self.reserved_len
    }
}

/// The auditor's own finding: a line it had to redact.
///
/// A **disjoint** type from [`Audited`], with its own constructor, so that
/// "exempt from the audit" appears in the diff. Returning the alarm as an
/// `Audited` would leave this type dead and reopen the constructor-reuse hole.
///
/// **It carries no line.** The alarm never quotes the secret or the offending
/// text (REQ-L50); it carries the secret's NAME — which is not its value — and
/// the target where it was emitted, which is all anyone needs to go find the
/// site.
#[derive(Debug, Clone)]
pub struct AuditExempt {
    secret: SecretName,
    target: &'static str,
}

impl AuditExempt {
    /// The name of the secret that was found.
    #[must_use]
    pub fn secret(&self) -> SecretName {
        self.secret
    }

    /// The target that emitted the offending line.
    #[must_use]
    pub fn target(&self) -> &'static str {
        self.target
    }
}

/// What travels on the appender's channel.
///
/// Lives here rather than in the appender: both variants are the auditor's
/// output types, and the appender only consumes them. Declaring it on the
/// consumer side would make this module depend on that one to name its own
/// results.
#[derive(Debug, Clone)]
pub enum Queued {
    /// The ordinary path: an event that went through the auditor.
    Line(Audited),
    /// The auditor's own finding.
    Alarm(AuditExempt),
}

/// Redacts every line that leaves the process.
#[derive(Default)]
pub struct Auditor {}

impl Auditor {
    /// Builds an auditor with nothing registered yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a secret by name, from variants the caller already derived.
    pub fn register_secret(&self, _name: SecretName, _variants: &[&str]) {
        // Red-phase stub
    }

    /// Audits one line. **The only constructor of [`Audited`].**
    #[must_use]
    pub fn audit(
        &self,
        line: &str,
        target: &'static str,
        cause: Option<CauseKey>,
        reserved_len: usize,
    ) -> (Audited, Option<AuditExempt>) {
        // Red-phase stub: passes the line through untouched.
        let _ = target;
        (
            Audited {
                line: line.to_string(),
                cause,
                reserved_len,
            },
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This module's own source, for the structural guards below.
    const SOURCE: &str = include_str!("auditor.rs");

    #[test]
    fn a_secret_renders_as_stars_in_both_debug_and_display() {
        let s = Secret::new("hunter2-and-then-some");
        assert_eq!(format!("{s}"), REDACTED);
        assert_eq!(format!("{s:?}"), REDACTED);
        assert!(!format!("{s:?}").contains("hunter2"));
        // And the value is still reachable, but only by name.
        assert_eq!(s.expose(), "hunter2-and-then-some");
    }

    #[test]
    fn a_secret_inside_a_struct_still_refuses_to_print() {
        // The realistic accident is not `{:?}` on the secret itself but `{:?}`
        // on something that contains it.
        #[derive(Debug)]
        struct Config {
            key: Secret,
        }
        let c = Config {
            key: Secret::new("sk-ant-do-not-print-me"),
        };
        assert!(!format!("{c:?}").contains("sk-ant"));
        // Reading the field keeps `dead_code` quiet AND states the point: the
        // value is still there, it just does not print itself.
        assert_eq!(c.key.expose(), "sk-ant-do-not-print-me");
    }

    #[test]
    fn an_audited_carries_the_reserved_length_it_was_given_not_its_own() {
        let auditor = Auditor::new();
        let reserved = 4096;
        let (audited, _) = auditor.audit("short line", "magi_rs::agent", None, reserved);
        assert_eq!(
            audited.reserved_len(),
            reserved,
            "the writer releases what was reserved, not what it holds"
        );
        assert_ne!(
            audited.reserved_len(),
            audited.as_str().len(),
            "if these coincided the test would pass for the wrong reason"
        );
    }

    #[test]
    fn an_audited_carries_its_cause_key_unchanged() {
        let auditor = Auditor::new();
        let (none, _) = auditor.audit("x", "t", None, 0);
        assert!(none.cause().is_none(), "no cause means no cause");
    }

    #[test]
    fn audited_has_no_public_field_and_no_conversion_into_it() {
        // `Audited` guarantees by TYPE that nothing reaches an output unaudited.
        // The compiler enforces it through the private fields; this test guards
        // the shape against a later edit that would quietly reopen the hole.
        let opener = format!("pub struct {} {{", "Audited");
        let decl = SOURCE
            .split(&opener)
            .nth(1)
            .expect("the struct is declared here");
        let decl = decl.split('}').next().expect("its body ends somewhere");
        assert!(
            !decl.contains("pub "),
            "a public field would let anything build an Audited: {decl}"
        );
        // The needles are BUILT rather than written, because a source check
        // whose assertions contain their own patterns finds itself and fails.
        // That is what happened on the first run of this test.
        for from in ["String", "&str", "&String", "Box<str>"] {
            let needle = format!("impl From<{from}> for Audited");
            assert!(
                !SOURCE.contains(&needle),
                "{needle} would be a second constructor wearing a different name"
            );
        }
    }

    #[test]
    fn audit_exempt_is_a_disjoint_type_and_carries_no_line() {
        let opener = format!("pub struct {} {{", "AuditExempt");
        let decl = SOURCE
            .split(&opener)
            .nth(1)
            .expect("the struct is declared here");
        let decl = decl.split('}').next().expect("its body ends somewhere");
        assert!(
            !decl.contains("line"),
            "the alarm must never carry the offending text: {decl}"
        );
        assert!(
            decl.contains("secret") && decl.contains("target"),
            "it carries the name and the site, which is all the operator needs"
        );
    }
}
