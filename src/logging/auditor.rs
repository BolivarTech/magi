// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

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

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Mutex;

use crate::headless::output::secret_pattern_ranges;
use crate::redact::{locate_userinfo, UserinfoLocation};

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

/// The embedder's subsystem half, as [`CauseKey::ALL`] declares it.
const EMBEDDER: &str = "embedder";

/// The cause an endpoint that could not be reached at all is keyed by.
const UNREACHABLE: &str = "unreachable";

/// The cause an endpoint that answered — badly — is keyed by.
///
/// **A separate cause from [`UNREACHABLE`], because R-L13b keys on the error
/// VARIANT.** Collapsing a subsystem to one cause makes SC-L16 — the screen
/// shows a second notice when the embedder goes from an HTTP error to a
/// refused connection — unreachable in production, and the health tracker's
/// entire cause-change branch dead with it.
///
/// The message table says the same thing in its own shape: the two embedder
/// rows carry **different** degradation strings and an **identical** recovery
/// string, because degradation is per variant while recovery is per subsystem.
/// That asymmetry is also why one success event can answer both causes — the
/// tracker keys its state on the subsystem, so a success sets it healthy
/// whichever of its causes it names.
const HTTP_ERROR: &str = "http_error";

impl CauseKey {
    /// Every declared cause.
    ///
    /// Exists so a test in MS2 can enumerate them against the message table;
    /// without it that guard cannot be written at all. Empty in MS1, populated
    /// by MS2's task 3.3.
    ///
    /// **One entry per cause an emitting site actually declares**, and each
    /// owes a row in `logging::health`'s message table — a key with no row
    /// renders as a defect report rather than a message, which is what
    /// `every_declared_cause_key_has_a_screen_message` holds down. The
    /// emitting site keeps its own constants (`tracing` fields take literals,
    /// not values); a site whose spelling drifts from this list stops
    /// resolving here, and its own tests say so.
    pub const ALL: &'static [CauseKey] = &[
        CauseKey::new(EMBEDDER, UNREACHABLE),
        CauseKey::new(EMBEDDER, HTTP_ERROR),
    ];

    /// Builds a cause key from its two program-constant halves.
    ///
    /// **The only constructor.** Both fields stay private so a `CauseKey`
    /// cannot be assembled from anything but a call site's own literals —
    /// mirroring [`SecretName::new`], which exists for the identical reason:
    /// the value must come from a constant the emitter wrote, never from the
    /// runtime text it is describing (R-L13).
    #[must_use]
    pub const fn new(subsystem: &'static str, cause: &'static str) -> Self {
        Self { subsystem, cause }
    }

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

    /// Rewrites the text, keeping everything else.
    ///
    /// **A transformation, never a second constructor.** It takes `self` by
    /// value, so it can only be applied to something that already went through
    /// [`Auditor::audit`] — which remains the only path from an unaudited
    /// `&str`. Stage 3's escaping uses this, and if it produced a `String` the
    /// escaped text would live outside the type that proves it was audited.
    ///
    /// # Complexity
    ///
    /// `O(n)` plus whatever `f` costs.
    #[must_use]
    pub fn map_line(mut self, f: impl FnOnce(&str) -> String) -> Self {
        self.line = f(&self.line);
        // **The measure follows the line.** `reserved_len` is what the queue
        // holds and what the writer gives back, so a transformation that
        // changes the text and leaves the number behind makes the emitter
        // reserve one amount and the writer release another. Escaping is
        // exactly such a transformation: it can grow a line severalfold.
        self.reserved_len = self.line.len();
        self
    }

    /// Cuts the text to `max` bytes for display, on a character boundary.
    ///
    /// The marker names the original length, so a reader can tell a truncated
    /// line from a short one.
    ///
    /// # Complexity
    ///
    /// `O(max)`.
    #[must_use]
    pub fn truncate_for_display(self, max: usize) -> Self {
        if self.line.len() <= max {
            return self;
        }
        let original = self.line.len();
        self.map_line(|line| {
            let mut cut = max;
            while cut > 0 && !line.is_char_boundary(cut) {
                cut -= 1;
            }
            let head = line.get(..cut).unwrap_or("");
            format!("{head}… [truncated, {original} bytes]")
        })
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

/// Base of the rolling hash. A prime above 256, so every byte gets its own
/// digit; a composite base collapses whole classes of window to one value.
const ROLLING_BASE: u64 = 1_000_003;

/// Shortest value the exact pass will scan for.
///
/// Below this, ordinary text collides constantly and the log would come back
/// half redacted. A shorter secret is still registered — pass 1 covers it — and
/// the caller is told, which beats rejecting it and leaving it uncovered AND
/// unannounced.
pub const MIN_SECRET_BYTES: usize = 8;

/// The scheme separator that opens a URL authority.
const SCHEME_SEP: &str = "://";

/// What the auditor keeps for one registered variant: never the value.
///
/// `(len, hash, pow)` and nothing else. `pow` is `base^(len-1)`, precomputed
/// once so the roll is O(1) per byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Digest {
    len: usize,
    hash: u64,
    pow: u64,
}

impl Digest {
    /// Digests a variant, or `None` when it is too short to scan for.
    ///
    /// # Complexity
    ///
    /// `O(k)` over the variant.
    fn of(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < MIN_SECRET_BYTES {
            return None;
        }
        let mut pow = 1u64;
        for _ in 1..bytes.len() {
            pow = pow.wrapping_mul(ROLLING_BASE);
        }
        Some(Self {
            len: bytes.len(),
            hash: window_hash(bytes),
            pow,
        })
    }
}

/// The hash of a whole window: `sum bytes[i] * base^(len-1-i)`, wrapping at 2^64.
///
/// # Complexity
///
/// `O(k)`.
fn window_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0u64;
    for &b in bytes {
        hash = hash.wrapping_mul(ROLLING_BASE).wrapping_add(u64::from(b));
    }
    hash
}

/// One registered secret: its name, and the digests of its variants.
#[derive(Debug, Clone)]
struct Registered {
    name: SecretName,
    digests: Vec<Digest>,
}

/// Redacts every line that leaves the process.
///
/// # What it never holds
///
/// Registered secrets are kept as `(length, hash, pow)`. The value is hashed on
/// the way in and dropped; there is no masked copy to unmask, because there is
/// no copy. That is the whole reason the comparison is a rolling hash and not a
/// literal search: a `memmem` over the plaintext would need the plaintext.
#[derive(Default)]
pub struct Auditor {
    registered: Mutex<Vec<Registered>>,
    /// Alarms already raised, keyed by `(secret, target)`.
    alarmed: Mutex<BTreeSet<(SecretName, &'static str)>>,
}

impl Auditor {
    /// Builds an auditor with nothing registered yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a secret by name, from variants the caller already derived.
    ///
    /// # Parameters
    ///
    /// * `name` — a program constant. Never the value.
    /// * `variants` — the forms the value can take on a line: raw, escaped,
    ///   percent-encoded. **The composer derives them**, not the auditor: the
    ///   encoder that produced the value inside a URL is the one that has to
    ///   produce the variant, and it lives on the resolving side.
    ///
    /// # Returns
    ///
    /// `false` when at least one variant was shorter than [`MIN_SECRET_BYTES`],
    /// so the caller can warn. The name is registered either way.
    ///
    /// # Complexity
    ///
    /// `O(total variant bytes)`.
    pub fn register_secret(&self, name: SecretName, variants: &[&str]) -> bool {
        let mut all_long = true;
        let mut digests = Vec::with_capacity(variants.len());
        for v in variants {
            match Digest::of(v.as_bytes()) {
                Some(d) => digests.push(d),
                None => all_long = false,
            }
        }
        if let Ok(mut reg) = self.registered.lock() {
            reg.push(Registered { name, digests });
        }
        all_long
    }

    /// Audits one line. **The only constructor of [`Audited`].**
    ///
    /// # Parameters
    ///
    /// * `line` — the rendered, not-yet-escaped text.
    /// * `target` — where it was emitted; the alarm carries it so an operator
    ///   knows where to go look.
    /// * `cause` — the emitter's declared cause, if any.
    /// * `reserved_len` — bytes reserved for this line **before** auditing.
    ///
    /// # Returns
    ///
    /// The redacted line, and an alarm when a registered secret was found.
    ///
    /// # The two passes do not chain, and that is the requirement
    ///
    /// Both run over the ORIGINAL text and contribute ranges; the ranges are
    /// unioned and the substitution happens once. Chained, pass 1 could mutilate
    /// part of a live secret inside a URL authority — the `user:pass@` where
    /// `pass` is also a registered value — leaving a residue that no longer
    /// equals the registered value. Pass 2 would then look for a literal pass 1
    /// had just broken, fail to find it, and **the residue would ship**: a leak
    /// created by the ORDER of two defences that each work alone.
    ///
    /// # Complexity
    ///
    /// `O(k*n)` with `k` registered variants and `n` the line length, plus
    /// `O(m log m)` to order the ranges.
    #[must_use]
    pub fn audit(
        &self,
        line: &str,
        target: &'static str,
        cause: Option<CauseKey>,
        reserved_len: usize,
    ) -> (Audited, Option<AuditExempt>) {
        let mut ranges = pattern_pass(line);
        let (exact, found) = self.exact_pass(line);
        ranges.extend(exact);

        let alarm = found.and_then(|secret| self.alarm(secret, target));

        (
            Audited {
                line: redact_ranges(line, ranges),
                cause,
                reserved_len,
            },
            alarm,
        )
    }

    /// Pass 2: the ranges where a registered variant appears, and which secret.
    ///
    /// # Complexity
    ///
    /// `O(k*n)`, one rolling sweep per registered variant.
    fn exact_pass(&self, line: &str) -> (Vec<Range<usize>>, Option<SecretName>) {
        let Ok(reg) = self.registered.lock() else {
            return (Vec::new(), None);
        };
        let bytes = line.as_bytes();
        let mut out = Vec::new();
        let mut found = None;
        for entry in reg.iter() {
            for d in &entry.digests {
                if bytes.len() < d.len {
                    continue;
                }
                let Some(first) = bytes.get(..d.len) else {
                    continue;
                };
                let mut hash = window_hash(first);
                if hash == d.hash {
                    out.push(0..d.len);
                    found.get_or_insert(entry.name);
                }
                for start in 1..=(bytes.len() - d.len) {
                    let outgoing = bytes.get(start - 1).copied().unwrap_or(0);
                    let incoming = bytes.get(start + d.len - 1).copied().unwrap_or(0);
                    hash = hash
                        .wrapping_sub(u64::from(outgoing).wrapping_mul(d.pow))
                        .wrapping_mul(ROLLING_BASE)
                        .wrapping_add(u64::from(incoming));
                    if hash == d.hash {
                        out.push(start..start + d.len);
                        found.get_or_insert(entry.name);
                    }
                }
            }
        }
        (out, found)
    }

    /// Raises an alarm the first time a secret is seen at a target.
    ///
    /// Deduplicated by `(secret, target)` inside the auditor, which is why the
    /// sink needs a non-deduplicating delivery for it: a `&'static str` key
    /// cannot express the pair.
    ///
    /// # Complexity
    ///
    /// `O(log n)` over the alarms already raised.
    fn alarm(&self, secret: SecretName, target: &'static str) -> Option<AuditExempt> {
        let mut seen = self.alarmed.lock().ok()?;
        if !seen.insert((secret, target)) {
            return None;
        }
        Some(AuditExempt { secret, target })
    }

    /// Whether a name has been registered at all.
    #[cfg(test)]
    pub(crate) fn is_registered(&self, name: SecretName) -> bool {
        self.registered
            .lock()
            .map(|r| r.iter().any(|e| e.name == name))
            .unwrap_or(false)
    }

    /// Whether a name participates in the exact (second) pass.
    #[cfg(test)]
    pub(crate) fn in_exact_pass(&self, name: SecretName) -> bool {
        self.registered
            .lock()
            .map(|r| r.iter().any(|e| e.name == name && !e.digests.is_empty()))
            .unwrap_or(false)
    }

    /// Renders the auditor's retained state, for the no-plaintext guard.
    ///
    /// `cfg(test)` only: in production this would be surface that PRINTS the
    /// auditor's state, which is the last thing anyone should be able to reach.
    #[cfg(test)]
    pub(crate) fn debug_dump_state(&self) -> String {
        self.registered
            .lock()
            .map(|r| format!("{r:?}"))
            .unwrap_or_default()
    }
}

/// The operator-facing text of an alarm.
///
/// **It never quotes the secret or the offending line** (REQ-L50). Naming the
/// value inside the leak detector would be a new leak channel built inside the
/// thing that exists to find them. The name and the target are all anyone needs
/// to go to the site.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn render_alarm(alarm: &AuditExempt) -> String {
    format!(
        "SECURITY: the value registered as {} reached a log line emitted by {}.          It was masked, and this is the alarm that says masking happened - both,          never one. Go look at that target.",
        alarm.secret(),
        alarm.target()
    )
}

/// Pass 1: the ranges the pattern matchers and the URL authority rule claim.
///
/// Both halves already exist in this crate and are reused rather than rewritten:
/// [`secret_pattern_ranges`] carries the key shapes, [`locate_userinfo`] carries
/// the RFC 3986 authority rule. Writing either traversal a second time is how
/// two copies drift apart, and here drifting means one of them stops seeing a
/// credential.
///
/// # Complexity
///
/// `O(n)`.
fn pattern_pass(line: &str) -> Vec<Range<usize>> {
    let mut out = secret_pattern_ranges(line);
    out.extend(userinfo_ranges(line));
    out
}

/// The byte ranges of every URL `userinfo` on the line.
///
/// **By position, never by content** (REQ-L46): the authority's last `@` closes
/// the userinfo and everything before it goes. Locating it by what it looks like
/// loses to double percent-encoding.
///
/// # Complexity
///
/// `O(n)`.
fn userinfo_ranges(line: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = line.get(from..).and_then(|s| s.find(SCHEME_SEP)) {
        let start = from + rel;
        let mut end = start;
        while end < bytes.len() && !is_url_terminator(bytes.get(end).copied().unwrap_or(0)) {
            end += 1;
        }
        if let Some(url) = line.get(start..end) {
            if let UserinfoLocation::Found { start: a, end: b } = locate_userinfo(url) {
                out.push(start + a..start + b);
            }
        }
        from = end.max(start + SCHEME_SEP.len());
        if from >= line.len() {
            break;
        }
    }
    out
}

/// Whether a byte ends a URL when a log line embeds one in prose.
///
/// # Complexity
///
/// `O(1)`.
fn is_url_terminator(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'"' || b == QUOTE || b == b'<' || b == b'>'
}

/// The single-quote byte, named so the match above reads without an escape.
const QUOTE: u8 = 39;

/// Applies every range once, merged, replacing each with [`REDACTED`].
///
/// # Complexity
///
/// `O(m log m)` to order, then `O(n)` to rebuild.
fn redact_ranges(line: &str, mut ranges: Vec<Range<usize>>) -> String {
    if ranges.is_empty() {
        return line.to_string();
    }
    ranges.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    let mut out = String::with_capacity(line.len());
    let mut cursor = 0usize;
    for r in merged {
        if let Some(seg) = line.get(cursor..r.start) {
            out.push_str(seg);
        }
        out.push_str(REDACTED);
        cursor = r.end;
    }
    if let Some(tail) = line.get(cursor..) {
        out.push_str(tail);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This module's own source, for the structural guards below.
    const SOURCE: &str = include_str!("auditor.rs");

    #[test]
    fn transforming_the_line_moves_the_measure_with_it() {
        // `reserved_len` is what the emitter reserves and what the writer gives
        // back. A transformation that changes the text and leaves the number
        // behind makes those two different amounts, and the counter drifts
        // until the channel refuses everything -- the shape of the defect that
        // made the byte budget a lifetime quota.
        //
        // Escaping is exactly such a transformation, and this was nearly
        // reintroduced while fixing the measure: the layer was about to reserve
        // the escaped length against a field still holding the rendered one.
        let auditor = Auditor::new();
        let (audited, _) = auditor.audit("abc", "magi_rs::tests", None, 3);
        assert_eq!(audited.reserved_len(), 3);

        let grown = audited.map_line(|l| format!("{l}{l}{l}"));
        assert_eq!(grown.as_str().len(), 9, "the fixture must have grown it");
        assert_eq!(
            grown.reserved_len(),
            9,
            "the measure stayed on the old line, so the writer would release \
             less than the emitter reserved"
        );
    }

    #[test]
    fn a_percent_encoded_variant_is_found_when_only_it_appears() {
        let auditor = Auditor::new();
        let raw = "p4ss@word/with?reserved#chars";
        let encoded = crate::encoding::percent_encode(raw);
        assert!(auditor.register_secret(SecretName::new("K"), &[raw, encoded.as_str()]));
        let line = format!("credential={encoded}");
        let (audited, alarm) = auditor.audit(&line, "magi_rs::tests", None, 0);
        assert!(
            !audited.as_str().contains(encoded.as_str()),
            "the encoded form survived: {}",
            audited.as_str()
        );
        assert!(alarm.is_some());
    }

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
    fn the_two_passes_do_not_chain_so_an_overlapping_secret_still_matches() {
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("BASE_URL_PASSWORD"), &["hunter2-longer"]);
        // Pass 1 (URL redaction) would mutilate the password inside the
        // authority; pass 2 must still see the ORIGINAL line.
        let (audited, alarm) = auditor.audit(
            "GET https://bob:hunter2-longer@example.com/v1",
            "magi_rs::logging",
            None,
            0,
        );
        assert!(
            !audited.as_str().contains("hunter2-longer"),
            "residue leaked: {}",
            audited.as_str()
        );
        assert!(alarm.is_some(), "an exact live-secret match must alarm");
    }

    #[test]
    fn the_auditor_never_materialises_a_secret_in_the_clear() {
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["s3cret-value-long-enough"]);
        let dump = auditor.debug_dump_state();
        // Without this the assertion below is vacuously true on an empty dump:
        // a state that shows NOTHING trivially fails to show the secret.
        assert!(
            dump.contains("K"),
            "the dump must actually show the registered state: {dump}"
        );
        assert!(
            !dump.contains("s3cret-value-long-enough"),
            "and it must not show the value: {dump}"
        );
    }

    #[test]
    fn a_short_secret_is_registered_excluded_from_pass_two_and_warned() {
        let auditor = Auditor::new();
        let all_long = auditor.register_secret(SecretName::new("SHORT"), &["abc"]);
        assert!(!all_long, "a secret under 8 bytes must warn");
        assert!(
            auditor.is_registered(SecretName::new("SHORT")),
            "it is still registered for pass 1"
        );
        assert!(
            !auditor.in_exact_pass(SecretName::new("SHORT")),
            "and excluded from pass 2"
        );
    }

    #[test]
    fn redaction_happens_before_chunking_over_the_whole_line() {
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["a-secret-that-straddles"]);
        let line = format!(
            "{}{}{}",
            "x".repeat(4090),
            "a-secret-that-straddles",
            "y".repeat(10)
        );
        let (audited, _) = auditor.audit(&line, "magi_rs::logging", None, 0);
        assert!(
            !audited.as_str().contains("a-secret-that-straddles"),
            "a secret past the chunk boundary would not match"
        );
    }

    #[test]
    fn a_foreign_string_is_redacted_like_a_native_one() {
        let auditor = Auditor::new();
        let (audited, _) = auditor.audit(
            "magi-core: POST https://u:p@host/v1 failed",
            "magi_core::http",
            None,
            0,
        );
        assert!(
            !audited.as_str().contains("u:p@"),
            "foreign events get the same treatment"
        );
    }

    #[test]
    fn a_line_with_nothing_to_hide_comes_back_unchanged_and_without_an_alarm() {
        // Without this the tests above pass just as well against an auditor
        // that redacts everything.
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["never-appears-here"]);
        let clean = "2026-08-14T00:00:00Z INFO magi_rs::agent: ordinary line";
        let (audited, alarm) = auditor.audit(clean, "magi_rs::agent", None, 0);
        assert_eq!(audited.as_str(), clean);
        assert!(alarm.is_none());
    }

    #[test]
    fn the_alarm_names_the_secret_and_the_target_and_quotes_neither_line_nor_value() {
        let auditor = Auditor::new();
        let name = SecretName::new("BASE_URL_PASSWORD");
        auditor.register_secret(name, &["hunter2-longer"]);
        let (audited, alarm) = auditor.audit(
            "GET https://bob:hunter2-longer@example.com/v1",
            "magi_rs::agent",
            None,
            0,
        );
        let alarm = alarm.expect("a live secret must alarm");
        let text = render_alarm(&alarm);

        assert!(
            text.contains("BASE_URL_PASSWORD"),
            "it names the secret: {text}"
        );
        assert!(text.contains("magi_rs::agent"), "and the site: {text}");
        assert!(
            !text.contains("hunter2-longer"),
            "it must NEVER quote the value: {text}"
        );
        assert!(
            !text.contains(audited.as_str()),
            "nor the offending line: {text}"
        );
    }

    #[test]
    fn the_same_secret_at_the_same_target_alarms_once_but_masks_every_time() {
        // Both, never one (REQ-L50): masking alone leaves the leak in place
        // forever; alarming alone would emit the secret anyway. The DEDUP is on
        // the alarm, not on the masking.
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["repeated-secret-value"]);
        let line = "url=https://x/repeated-secret-value";

        let (first, alarm_one) = auditor.audit(line, "magi_rs::agent", None, 0);
        let (second, alarm_two) = auditor.audit(line, "magi_rs::agent", None, 0);

        assert!(alarm_one.is_some(), "the first sighting alarms");
        assert!(alarm_two.is_none(), "the second does not repeat the alarm");
        assert!(!first.as_str().contains("repeated-secret-value"));
        assert!(
            !second.as_str().contains("repeated-secret-value"),
            "masking never stops, even once the alarm has been raised"
        );
    }

    #[test]
    fn the_same_secret_at_another_target_alarms_again() {
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["repeated-secret-value"]);
        let line = "url=https://x/repeated-secret-value";
        assert!(auditor.audit(line, "magi_rs::agent", None, 0).1.is_some());
        assert!(
            auditor.audit(line, "magi_core::http", None, 0).1.is_some(),
            "a second site is a second thing to go fix"
        );
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
