// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! An auditor belongs to whoever built it, and a finding stays inside its test.
//!
//! # Why this is an integration binary and not a unit test
//!
//! `cargo nextest` runs **each test in its own process**, so a global subscriber
//! installed by one of these is that process's alone and a panic aborts only it.
//! The property being asserted here — isolation — is therefore partly a property
//! of the runner: plain `cargo test` shares a process, and so do doctests.
//!
//! # What this file does NOT claim
//!
//! The full canary — run a session, grep the whole log directory including the
//! decompressed `.xz`, and the TUI's message buffer — is task 6.5, and it belongs
//! there. In phase 3 the appender does not exist and nothing is written to disk,
//! so a canary here would pass **because there is no pipeline yet**: exactly the
//! guardian that does not exercise what it claims to guard.

use magi_rs::logging::auditor::{render_alarm, Auditor, SecretName};

/// A value long enough for the exact pass to scan for it.
const LIVE_SECRET: &str = "hunter2-and-then-some";
/// The name it is registered under. A program constant, never a runtime value.
const SECRET_NAME: SecretName = SecretName::new("BASE_URL_PASSWORD");

#[test]
fn one_auditors_registration_does_not_reach_another() {
    let registered = Auditor::new();
    assert!(registered.register_secret(SECRET_NAME, &[LIVE_SECRET]));

    let untouched = Auditor::new();

    let line = format!("GET https://svc/{LIVE_SECRET}");
    let (from_registered, alarm) = registered.audit(&line, "magi_rs::tests", None, 0);
    let (from_untouched, no_alarm) = untouched.audit(&line, "magi_rs::tests", None, 0);

    assert!(
        !from_registered.as_str().contains(LIVE_SECRET),
        "the auditor that knows the secret must hide it"
    );
    assert!(alarm.is_some(), "and say so");

    assert!(
        no_alarm.is_none(),
        "a second auditor never learned about it, so it has nothing to alarm about"
    );
    assert!(
        from_untouched.as_str().contains(LIVE_SECRET),
        "and — the point of this test — it does NOT hide it either. If it did, \
         registration would be leaking between instances through something global, \
         and one test's finding could abort another."
    );
}

#[test]
fn redaction_works_over_a_hand_built_line_with_both_shapes_on_it() {
    let auditor = Auditor::new();
    assert!(auditor.register_secret(SECRET_NAME, &[LIVE_SECRET]));

    // A URL credential (pass 1, by position) and the live value (pass 2, by
    // rolling hash) on the same line, in that order.
    let line = format!("POST https://bob:p4ssw0rd@api.example.com/v1 token={LIVE_SECRET}");
    let (audited, alarm) = auditor.audit(&line, "magi_rs::tests", None, 0);

    assert!(
        !audited.as_str().contains("bob:p4ssw0rd@"),
        "the URL credential survived: {}",
        audited.as_str()
    );
    assert!(
        !audited.as_str().contains(LIVE_SECRET),
        "the live secret survived: {}",
        audited.as_str()
    );
    assert!(
        audited.as_str().contains("api.example.com"),
        "and the parts that are not secret are still readable: {}",
        audited.as_str()
    );

    let text = render_alarm(&alarm.expect("a live secret alarms"));
    assert!(
        !text.contains(LIVE_SECRET),
        "the alarm never quotes the value"
    );
}

#[test]
fn an_alarm_in_one_test_does_not_follow_into_the_next() {
    // The dedup set lives in the auditor instance, so a fresh one alarms again
    // for the same pair. If the set were global, this test would see the alarm
    // already raised by the test above and come back empty — which is the
    // failure mode this file exists to rule out.
    let auditor = Auditor::new();
    assert!(auditor.register_secret(SECRET_NAME, &[LIVE_SECRET]));
    let line = format!("token={LIVE_SECRET}");
    let (_, alarm) = auditor.audit(&line, "magi_rs::tests", None, 0);
    assert!(
        alarm.is_some(),
        "a fresh auditor has raised nothing yet, whatever other tests did"
    );
}
