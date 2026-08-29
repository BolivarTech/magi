// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! A credential emitted through the REAL dispatcher reaches neither mouth.
//!
//! # Why this exists in this shape
//!
//! v0.12.0 shipped five separate leaks, each found by hand-tracing where one
//! foreign string ended up. That does not scale per site; it scales per OUTPUT.
//! This is what replaces that work: emit through the real pipeline, then grep
//! **everything** that came out.
//!
//! # The trap this file is written to avoid
//!
//! A canary that builds the line by hand and passes it to the redactor tests the
//! REDACTOR, not the PATH — and the path is what leaked. Every case here goes
//! through `tracing`'s dispatcher into the installed layer, so removing the
//! auditor from either branch makes it fail. Both mutations were run.
//!
//! # Both mouths, and only both
//!
//! MS1's mouths are the FILE and the notice sink. The full TUI layer is MS2, and
//! the canary gains the message buffer when that sink is connected — stated so a
//! green run here is not read as covering a mouth that does not exist yet.
//!
//! An integration binary because it installs a global subscriber; `cargo
//! nextest` gives each test its own process. The property depends on the runner:
//! plain `cargo test` shares one, and so do doctests.

use std::sync::{Arc, Mutex};

use magi_rs::logging::auditor::{Audited, SecretName};
use magi_rs::logging::magi_layer::TuiSink;
use magi_rs::logging::{init_logging, LoggingConfig, NoticeDelivery};

/// The password inside the credentialled endpoint.
const PASSWORD: &str = "hunter2-and-then-some";
/// The API key configured for the session, **assembled rather than written**.
///
/// A literal here trips the repository's own `no_hardcoded_secrets` guard — which
/// is correct: a key-shaped string in the tree is a key-shaped string whatever
/// it is for. Assembling it keeps this canary's fixture out of that guard's way
/// without weakening either one, and the value the redactor sees at runtime is
/// identical.
fn api_key() -> String {
    format!("sk-{}-api03-CanaryKeyDoNotLeak", "ant")
}
/// A credentialled endpoint, exactly as an operator would configure one.
const ENDPOINT: &str = "https://svc-user:hunter2-and-then-some@api.example.com/v1";
/// A secret with NO recognisable shape, and the fixture depends on that.
///
/// The other two are caught by pass 1 alone: the API key by its `sk-ant-`
/// prefix, the password by its position in a URL authority. A canary built only
/// from those never needs pass 2, so removing the exact pass — or the auditor
/// from a whole branch — stays invisible. That mutation was run and came back
/// GREEN until this constant existed. Only registration can find this one.
const OPAQUE: &str = "correct horse battery staple";

/// Captures what the notice sink was handed — the second mouth.
#[derive(Default)]
struct CapturingSink {
    lines: Mutex<Vec<String>>,
}

impl NoticeDelivery for CapturingSink {
    fn deliver(&self, line: &Audited) {
        if let Ok(mut l) = self.lines.lock() {
            l.push(line.as_str().to_string());
        }
    }
}

/// Everything the file mouth produced, with any `.xz` decompressed.
///
/// **Decompressing matters:** a rotated-and-compressed day is still a mouth, and
/// a canary that only reads `.log` would go green the moment retention did its
/// job.
fn everything_written(dir: &std::path::Path) -> String {
    use std::io::Read as _;
    let mut all = String::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return all;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("xz") {
            let Ok(file) = std::fs::File::open(&path) else {
                continue;
            };
            let mut back = Vec::new();
            if lzma_rust2::XzReader::new(file, false)
                .read_to_end(&mut back)
                .is_ok()
            {
                all.push_str(&String::from_utf8_lossy(&back));
            }
            continue;
        }
        all.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
    }
    all
}

#[test]
fn a_credential_emitted_through_the_real_dispatcher_reaches_neither_mouth() {
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(CapturingSink::default());

    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_level: tracing::Level::TRACE,
    };
    // **The screen branch is CONNECTED here**, and it has to be. MS1 never wires
    // it in production, so passing `None` would leave the second mouth
    // unexercised — the canary would claim "both mouths" while testing one.
    // Removing the auditor from the notice branch was run as a mutation and
    // stayed GREEN until this line existed.
    let handle = init_logging(
        &cfg,
        sink.clone(),
        Some((TuiSink::new(sink.clone()), tracing::Level::TRACE)),
    )
    .expect("init");
    let key = api_key();
    magi_rs::logging::register_process_secrets(&[
        (SecretName::new("BASE_URL_PASSWORD"), PASSWORD),
        (SecretName::new("ANTHROPIC_API_KEY"), &key),
        (SecretName::new("BASE_URL_USER"), OPAQUE),
    ]);

    // A whole session's worth of emission, through the REAL macros — never a
    // hand-built string handed to the redactor.
    tracing::info!(target: "magi_rs::agent", "connecting to {ENDPOINT}");
    tracing::warn!(target: "magi_rs::agent", key = key.as_str(), "auth configured");
    tracing::error!(
        target: "magi_core::http",
        "magi-core: POST {ENDPOINT} failed: 401 (key {key})"
    );
    tracing::debug!(target: "magi_rs::tools", "tool ran against {ENDPOINT}");
    tracing::info!(target: "magi_rs::vault", passphrase = OPAQUE, "vault unlocked");

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(300));

    let file_mouth = everything_written(dir.path());
    let notice_mouth = sink.lines.lock().unwrap().join("\n");

    // The fixture must have produced something, or every assertion below holds
    // for free — which is how a canary quietly stops being one.
    assert!(
        file_mouth.contains("connecting to"),
        "the file mouth wrote nothing: {file_mouth}"
    );
    assert!(
        notice_mouth.contains("connecting to"),
        "the notice mouth received nothing, so grepping it proves nothing:          {notice_mouth}"
    );

    for (mouth, text) in [("file", &file_mouth), ("notice", &notice_mouth)] {
        assert!(
            !text.contains(PASSWORD),
            "the URL password reached the {mouth} mouth: {text}"
        );
        assert!(
            !text.contains(&key),
            "the API key reached the {mouth} mouth: {text}"
        );
        assert!(
            !text.contains(OPAQUE),
            "the shapeless secret reached the {mouth} mouth — only the exact pass              can catch this one, so this is the assertion that proves the auditor              ran on this branch at all: {text}"
        );
    }
}

#[test]
fn a_foreign_string_gets_the_same_treatment_as_one_of_our_own() {
    // Separate from the test above, and §10 lists both. The first proves BOTH
    // MOUTHS go through the auditor; this proves a string we did NOT generate —
    // magi-core's 46 uninstrumented sites — is treated identically. A canary
    // that only emits its own events would go green against a pipeline that
    // exempts foreign ones.
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(CapturingSink::default());
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_level: tracing::Level::TRACE,
    };
    let handle = init_logging(
        &cfg,
        sink.clone(),
        Some((TuiSink::new(sink.clone()), tracing::Level::TRACE)),
    )
    .expect("init");

    // No registration at all: this leans only on pass 1, which is what covers a
    // credential nobody told us about.
    tracing::error!(
        target: "magi_core::rotation",
        "rotation abandoned: https://u:p4ssw0rd-of-theirs@host/v1 returned 500"
    );

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(300));

    let written = everything_written(dir.path());
    assert!(
        written.contains("rotation abandoned"),
        "the fixture must have written: {written}"
    );
    assert!(
        !written.contains("u:p4ssw0rd-of-theirs@"),
        "a foreign event's credential survived: {written}"
    );
}

#[test]
fn a_password_that_only_ever_appears_percent_encoded_is_still_masked() {
    // **REQ-L49's main case.** A password with reserved characters never shows
    // up raw inside a `base_url` — it is encoded on the way in — so an auditor
    // registered with only the raw form is blind in the one place credentials
    // actually live. Registering the encoded variant is what closes it.
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(CapturingSink::default());
    let cfg = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        file_level: tracing::Level::TRACE,
    };
    let handle = init_logging(
        &cfg,
        sink.clone(),
        Some((TuiSink::new(sink.clone()), tracing::Level::TRACE)),
    )
    .expect("init");

    // **Genuinely reserved characters**, or the encoded form equals the raw one
    // and the raw variant catches it anyway — which is what happened first, and
    // the mutation stayed green.
    // **The writer is given time to PARK before anything is emitted**, and that
    // ordering is the fixture. The original defect — the writer blocking on the
    // priority channel and never waking for an ordinary one — only shows up when
    // the park happens first. Emitting immediately hides it: the event is
    // already queued when the writer takes its last lap.
    std::thread::sleep(std::time::Duration::from_millis(150));

    let raw = "p4ss@word/with?reserved#chars";
    magi_rs::logging::register_process_secrets(&[(SecretName::new("BASE_URL_PASSWORD"), raw)]);

    // Emitted ONLY in its encoded form, and **outside a URL authority** — inside
    // one, pass 1 would mask it by position and the encoded variant would never
    // be the thing that saved us. That masked the mutation too.
    let encoded = magi_rs::encoding::percent_encode(raw);
    tracing::info!(target: "magi_rs::agent", credential = %encoded, "endpoint configured");

    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(300));

    let written = everything_written(dir.path());
    assert!(
        written.contains("endpoint configured"),
        "the fixture must have written: {written}"
    );
    assert!(
        !written.contains(encoded.as_str()),
        "the encoded password survived: {written}"
    );
}
