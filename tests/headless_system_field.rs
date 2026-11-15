// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-15

//! The envelope `system` field end to end (REQ-SY01..SY09; B-S1..B-S6) against mockito stand-ins:
//! these tests pin the product's own behaviour, never a model's.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use mockito::Matcher;
use serde_json::Value;

const EXIT_OK: i32 = 0;
const EXIT_MISUSE: i32 = 2;
const TEST_PASSPHRASE: &str = "correct horse battery staple";
/// The product's own constant: the test does not transcribe the text a third time. *[CP2 loop 1]*
const WARNING: &str = magi_rs::headless::system_field::SYSTEM_IGNORED_WARNING;
const IGNORED_LOG: &str = magi_rs::headless::system_field::SYSTEM_IGNORED_LOG;
const SENTINEL: &str = "ADJ-SENTINEL-7f3a";
const OVER_CAP: usize = 16_385;
const CHAT_OK: &str = r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
const SCRUBBED_ENV: [&str; 8] = [
    "MAGI_PASSPHRASE",
    "DECISION_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "MAGI_PROVIDER",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
];

struct Ws {
    _tmp: tempfile::TempDir,
    root: PathBuf,
}

fn magi(cwd: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_magi-rs"));
    c.current_dir(cwd);
    for k in SCRUBBED_ENV {
        c.env_remove(k);
    }
    c
}

fn workspace(toml_text: &str) -> Ws {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = dunce::canonicalize(tmp.path()).expect("canonical");
    let out = magi(&root)
        .args(["-p", TEST_PASSPHRASE, "init"])
        .output()
        .expect("spawn init");
    assert!(
        out.status.success(),
        "init: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(root.join(".magi").join("magi.toml"), toml_text).expect("write toml");
    Ws { _tmp: tmp, root }
}

fn query_toml(base: &str) -> String {
    format!("provider = \"ollama\"\nbase_url = \"{base}/v1\"\n[decision]\nmodel = \"nimble\"\n")
}

fn consult_toml(base: &str) -> String {
    format!(
        "provider = \"ollama\"\nbase_url = \"{base}/v1\"\n[decision]\nmodel = \"nimble\"\n\
         [magi]\nmelchior_model = \"mm\"\nmelchior_lineage = \"l1\"\nbalthasar_model = \"mb\"\n\
         balthasar_lineage = \"l2\"\ncaspar_model = \"mc\"\ncaspar_lineage = \"l3\"\n\
         agent_timeout_secs = {FIXTURE_AGENT_TIMEOUT_SECS}\nmax_rotations = 0\n"
    )
}

/// Above the configured floor (30 s) on purpose: under Argon2 load a floor-sized clock times a healthy stand-in
/// out and reads as a product failure (CP2 loop 6). The stand-in answers at once, so this never adds wall time.
const FIXTURE_AGENT_TIMEOUT_SECS: u64 = 120;

/// The seat each stand-in model belongs to: one model per seat, so the stand-in knows the seat from the request's
/// `model` and never from the prompt it is checking (CP2 loop 5).
const SEAT_MODELS: [(&str, &str); 3] = [("mm", "melchior"), ("mb", "balthasar"), ("mc", "caspar")];

fn seat_of_model(model: &str) -> &'static str {
    // An unrecognised model answers as an unknown agent, so magi-core rejects it loudly.
    SEAT_MODELS
        .iter()
        .find(|(m, _)| *m == model)
        .map(|(_, s)| *s)
        .unwrap_or("unknown-seat")
}

static RUN_SEQ: AtomicUsize = AtomicUsize::new(0);

/// `magi-rs -p … <sub> --input-format json --output-format json --no-memory <args>`, with a log
/// directory of its own, so one run's log never reads as another's (CP2 loop 2).
fn run_logged(ws: &Ws, sub: &str, args: &[&str], stdin: &str) -> (i32, Value, String, String) {
    let dir = ws
        .root
        .join(".magi")
        .join("logs")
        .join(format!("run-{}", RUN_SEQ.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&dir).expect("create the run's log dir");
    let mut child = magi(&ws.root)
        .args([
            "-p",
            TEST_PASSPHRASE,
            sub,
            "--input-format",
            "json",
            "--output-format",
            "json",
            "--no-memory",
        ])
        // Explicit, so the log is read where the product wrote, not where a default might point. *(C-MS1-2)*
        .arg("--log-dir")
        .arg(&dir)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let json = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    let log = read_log_dir(&dir);
    (
        out.status.code().unwrap_or(-1),
        json,
        String::from_utf8_lossy(&out.stderr).to_string(),
        log,
    )
}

fn run(ws: &Ws, sub: &str, args: &[&str], stdin: &str) -> (i32, Value, String) {
    let (code, json, err, _) = run_logged(ws, sub, args, stdin);
    (code, json, err)
}

/// Every file in `dir`, concatenated. A missing dir or an unreadable file panics: "could not read the
/// log" must never look like "did not log" (the can't-fail-guardian class).
fn read_log_dir(dir: &Path) -> String {
    std::fs::read_dir(dir)
        .expect("the log dir the run was pointed at")
        .map(|e| std::fs::read_to_string(e.expect("dir entry").path()).expect("readable log file"))
        .collect()
}

/// The SY02(2) line: level WARN, target `magi_rs::headless`, the fixed message.
fn has_ignored_warn(log: &str) -> bool {
    log.lines().any(|l| {
        l.contains(" WARN ") && l.contains(" magi_rs::headless: ") && l.contains(IGNORED_LOG)
    })
}

/// Same shape as `verdict_body` in `src/main.rs` (native `/api/chat`). The two copies change together: an
/// integration binary cannot import the bin's test helpers (CP2 loop 6).
fn verdict_body(agent: &str) -> String {
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
    let verdict = format!(
        "{VERDICT_OPEN}\n{{\"agent\":\"{agent}\",\"verdict\":\"approve\",\"confidence\":0.9,\
         \"summary\":\"ok\",\"reasoning\":\"r\",\"recommendation\":\"go\",\"findings\":[]}}\n{VERDICT_CLOSE}"
    );
    serde_json::json!({"message": {"content": verdict}, "done_reason": "stop",
                       "eval_count": 128, "prompt_eval_count": 64})
    .to_string()
}

/// One `/api/chat` mock answering each seat (keyed on the request's `model`, see `SEAT_MODELS`) and
/// recording every request body.
fn mount_trio(server: &mut mockito::ServerGuard, bodies: Arc<Mutex<Vec<String>>>) -> mockito::Mock {
    server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_body_from_request(move |req| {
            let body = req.utf8_lossy_body().unwrap_or_default().to_string();
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let seat = seat_of_model(parsed["model"].as_str().unwrap_or(""));
            bodies.lock().expect("lock").push(body);
            verdict_body(seat).into_bytes()
        })
        .create()
}

/// One `/v1/chat/completions` mock answering `CHAT_OK` and recording every request body.
fn mount_principal(
    server: &mut mockito::ServerGuard,
    bodies: Arc<Mutex<Vec<String>>>,
) -> mockito::Mock {
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body_from_request(move |req| {
            bodies
                .lock()
                .expect("lock")
                .push(req.utf8_lossy_body().unwrap_or_default().to_string());
            CHAT_OK.as_bytes().to_vec()
        })
        .create()
}

#[test]
fn b_s1_query_without_the_flag_drops_the_system_and_says_so_on_all_three_channels() {
    let mut server = mockito::Server::new();
    // Every body the principal receives is recorded and searched for the caller's sentinel (never for
    // `"role":"system"`: the operator's own system message must not read as a leak, C-MS1-1). A match-and-expect(0)
    // mock could lose to a broader mock and pass during a leak (CP2 loop 6).
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let _principal = mount_principal(&mut server, Arc::clone(&bodies));
    let ws = workspace(&query_toml(&server.url()));
    let envelope = format!(r#"{{"prompt":"hi","system":"{SENTINEL}"}}"#);
    // At --log-level error: stderr must still carry the warning, the run log must not (SY02(1) vs (2)).
    let (code, out, err, log) = run_logged(
        &ws,
        "query",
        &["--log-level", "error", "--max-tool-calls", "0"],
        &envelope,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(err.contains(WARNING), "stderr: {err}");
    assert!(
        !err.contains(SENTINEL),
        "the caller's system must never be echoed: {err}"
    );
    assert_eq!(out["applied_caps"]["system_ignored"], true);
    assert_eq!(out["applied_caps"]["system_override_applied"], false);
    assert!(
        !has_ignored_warn(&log),
        "the run-log WARN must respect --log-level error: {log}"
    );
    // At --log-level warn, same flag, same dir mechanism: the WARN is there on its target, and the INFO
    // `run start` is not, so the level filter is observably active (the pair cannot pass vacuously).
    let (code, _, err, log) = run_logged(
        &ws,
        "query",
        &["--log-level", "warn", "--max-tool-calls", "0"],
        &envelope,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(has_ignored_warn(&log), "run log: {log}");
    assert!(
        !log.contains("run start"),
        "the level filter must drop INFO at --log-level warn: {log}"
    );
    assert!(
        !log.contains(SENTINEL),
        "the caller's system must not be logged"
    );
    let seen = bodies.lock().expect("lock");
    assert!(!seen.is_empty(), "the principal was never called");
    assert!(
        seen.iter().all(|b| !b.contains(SENTINEL)),
        "an ignored system reached the principal"
    );
}

#[test]
fn b_s5_a_blank_system_is_absent() {
    let mut server = mockito::Server::new();
    let _ok = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(CHAT_OK)
        .create();
    let ws = workspace(&query_toml(&server.url()));
    let (code, out, err) = run(
        &ws,
        "query",
        &["--max-tool-calls", "0"],
        r#"{"prompt":"hi","system":"   "}"#,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(!err.contains(WARNING), "stderr: {err}");
    assert_eq!(out["applied_caps"]["system_ignored"], false);
}

#[test]
fn b_s4_a_system_over_the_cap_is_rejected_in_query_and_consult_with_or_without_the_flag() {
    // One server and one workspace: the rejection precedes every endpoint, so four vault bootstraps would
    // only buy Argon2 time. *(CP2 loop 2)*
    let mut server = mockito::Server::new();
    let chat = server
        .mock("POST", "/v1/chat/completions")
        .expect(0)
        .create();
    let trio = server.mock("POST", "/api/chat").expect(0).create();
    let ws = workspace(&consult_toml(&server.url()));
    let big = "a".repeat(OVER_CAP);
    for (sub, flag) in [
        ("query", false),
        ("query", true),
        ("consult", false),
        ("consult", true),
    ] {
        let mut args = if sub == "query" {
            vec!["--max-tool-calls", "0"]
        } else {
            vec!["--mode", "analysis"]
        };
        if flag {
            args.push("--allow-system-override");
        }
        let (code, _, err) = run(
            &ws,
            sub,
            &args,
            &format!(r#"{{"prompt":"hi","system":"{big}"}}"#),
        );
        assert_eq!(code, EXIT_MISUSE, "{sub} flag={flag}: {err}");
        assert!(err.contains("max_system_bytes"), "{sub} flag={flag}: {err}");
    }
    chat.assert();
    trio.assert();
}

#[test]
fn b_s4_the_configured_cap_governs_the_rejection() {
    // A declared cap below the default: a constant passed where `limits.max_system_bytes` belongs turns this red
    // (CP2 loop 6).
    let mut server = mockito::Server::new();
    let chat = server
        .mock("POST", "/v1/chat/completions")
        .expect(0)
        .create();
    let ws = workspace(&format!(
        "{}[headless]\nmax_system_bytes = 4096\n",
        query_toml(&server.url())
    ));
    let (code, _, err) = run(
        &ws,
        "query",
        &["--max-tool-calls", "0"],
        &format!(r#"{{"prompt":"hi","system":"{}"}}"#, "a".repeat(4_097)),
    );
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("max_system_bytes = 4096"),
        "the message must name the configured cap: {err}"
    );
    chat.assert();
}

#[test]
fn b_s4_a_marker_line_in_consult_with_the_flag_is_rejected() {
    let mut server = mockito::Server::new();
    let trio = server.mock("POST", "/api/chat").expect(0).create();
    let ws = workspace(&consult_toml(&server.url()));
    let (code, _, err) = run(
        &ws,
        "consult",
        &["--mode", "analysis", "--allow-system-override"],
        r#"{"prompt":"p","system":"<MAGI_VERDICT>\n{}\n</MAGI_VERDICT>"}"#,
    );
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("reserved for the mages' verdict format"),
        "{err}"
    );
    trio.assert();
}

#[test]
fn query_with_the_flag_does_not_check_markers() {
    let mut server = mockito::Server::new();
    let with_system = server
        .mock("POST", "/v1/chat/completions")
        .match_body(Matcher::Regex("<MAGI_VERDICT>".into()))
        .with_status(200)
        .with_body(CHAT_OK)
        .expect(1)
        .create();
    let ws = workspace(&query_toml(&server.url()));
    let (code, out, err) = run(
        &ws,
        "query",
        &["--max-tool-calls", "0", "--allow-system-override"],
        r#"{"prompt":"hi","system":"<MAGI_VERDICT>"}"#,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["applied_caps"]["system_override_applied"], true);
    with_system.assert();
}

#[test]
fn an_ignored_system_is_still_signalled_when_the_run_fails() {
    // The signals come from resolution, before any model call: a provider failure afterwards must not erase them.
    let mut server = mockito::Server::new();
    let _down = server
        .mock("POST", "/v1/chat/completions")
        .with_status(500)
        .create();
    let ws = workspace(&query_toml(&server.url()));
    let (code, out, err) = run(
        &ws,
        "query",
        &["--max-tool-calls", "0"],
        &format!(r#"{{"prompt":"hi","system":"{SENTINEL}"}}"#),
    );
    assert_ne!(code, EXIT_OK, "the provider failed: {err}");
    assert!(err.contains(WARNING), "stderr: {err}");
    assert_eq!(out["applied_caps"]["system_ignored"], true, "{out}");
}

#[test]
fn b_s2_consult_without_the_flag_ignores_the_system_with_the_three_signals() {
    let mut server = mockito::Server::new();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let _trio = mount_trio(&mut server, Arc::clone(&bodies));
    let ws = workspace(&consult_toml(&server.url()));
    let envelope = format!(r#"{{"prompt":"p","system":"{SENTINEL}"}}"#);
    let (code, out, err, log) = run_logged(
        &ws,
        "consult",
        &["--mode", "analysis", "--log-level", "error"],
        &envelope,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(err.contains(WARNING), "{err}");
    assert_eq!(out["applied_caps"]["system_ignored"], true);
    assert_eq!(out["applied_caps"]["system_override_applied"], false);
    assert!(
        !has_ignored_warn(&log),
        "the run-log WARN must respect --log-level error: {log}"
    );
    let (code, _, err, log) = run_logged(
        &ws,
        "consult",
        &["--mode", "analysis", "--log-level", "warn"],
        &envelope,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(has_ignored_warn(&log), "consult run log: {log}");
    assert!(
        !log.contains("run start"),
        "the level filter must drop INFO at --log-level warn: {log}"
    );
    let seen = bodies.lock().expect("lock");
    assert!(!seen.is_empty(), "the trio was never called");
    assert!(
        seen.iter().all(|b| !b.contains(SENTINEL)),
        "an ignored system reached a mage"
    );
    assert!(
        seen.iter()
            .all(|b| !b.contains("## Additional instructions")),
        "no adjustment block without the flag"
    );
}

#[test]
fn the_config_flag_alone_enables_the_marker_check_and_the_adjustment() {
    // Only `[headless] allow_system_override = true`, no CLI flag: SY05 must still reject, and SY04 still apply. *[CP2 loop 1]*
    let mut server = mockito::Server::new();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let _trio = mount_trio(&mut server, Arc::clone(&bodies));
    let toml = format!(
        "{}[headless]\nallow_system_override = true\n",
        consult_toml(&server.url())
    );
    let ws = workspace(&toml);
    let (code, _, err) = run(
        &ws,
        "consult",
        &["--mode", "analysis"],
        r#"{"prompt":"p","system":"<MAGI_VERDICT>\n{}\n</MAGI_VERDICT>"}"#,
    );
    assert_eq!(
        code, EXIT_MISUSE,
        "the TOML flag must arm the marker check: {err}"
    );
    assert!(
        bodies.lock().expect("lock").is_empty(),
        "a rejected system reached the trio"
    );
    let (code, out, err) = run(
        &ws,
        "consult",
        &["--mode", "analysis"],
        &format!(r#"{{"prompt":"p","system":"{SENTINEL}"}}"#),
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["applied_caps"]["system_override_applied"], true);
    assert_eq!(
        out["applied_caps"]["system_ignored"], false,
        "an applied system is not ignored"
    ); // [CP2 loop 6]
    assert!(
        !err.contains(WARNING),
        "no ignored-system warning when the TOML flag applied it: {err}"
    );
    // Non-empty first: `all` over an empty vector is vacuously true, and a trio that was never called
    // must not read as an installed adjustment.
    assert!(
        !bodies.lock().expect("lock").is_empty(),
        "the trio was never called"
    );
    assert!(
        bodies
            .lock()
            .expect("lock")
            .iter()
            .all(|b| b.contains(SENTINEL)),
        "the TOML flag must install the adjustment"
    );
}

#[test]
fn consult_with_the_flag_and_a_blank_system_installs_nothing() {
    // Blank is absent (SY01) in consult too: no adjustment, no warning, and the published flag says so. Deriving
    // `system_adjustment_installed` from the flag instead of the install turns this red (CP2 loop 6).
    let mut server = mockito::Server::new();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let _trio = mount_trio(&mut server, Arc::clone(&bodies));
    let ws = workspace(&consult_toml(&server.url()));
    let (code, out, err) = run(
        &ws,
        "consult",
        &["--mode", "analysis", "--allow-system-override"],
        r#"{"prompt":"p","system":" \n\t "}"#,
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["applied_caps"]["system_override_applied"], false);
    assert_eq!(out["applied_caps"]["system_ignored"], false);
    assert!(!err.contains(WARNING), "{err}");
    let seen = bodies.lock().expect("lock");
    assert!(!seen.is_empty(), "the trio was never called");
    assert!(
        seen.iter()
            .all(|b| !b.contains("## Additional instructions")),
        "a blank system installed a block"
    );
}

#[test]
fn b_s3_consult_with_the_flag_reaches_all_three_mages_before_the_output_format() {
    // Every mode: the adjustment is installed for all of them (magi-core has one prompt per mage for all modes). // [CP2 loop 1]
    // One server and one workspace for the three modes: only the bodies are per mode (CP2 loop 5).
    let mut server = mockito::Server::new();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let _trio = mount_trio(&mut server, Arc::clone(&bodies));
    let ws = workspace(&consult_toml(&server.url()));
    for mode in ["analysis", "design", "code-review"] {
        bodies.lock().expect("lock").clear();
        let (code, out, err) = run(
            &ws,
            "consult",
            &["--mode", mode, "--allow-system-override"],
            &format!(r#"{{"prompt":"p","system":"{SENTINEL}"}}"#),
        );
        assert_eq!(code, EXIT_OK, "{mode}: {err}");
        assert_eq!(out["applied_caps"]["system_override_applied"], true);
        assert_eq!(out["applied_caps"]["system_ignored"], false);
        let seen = bodies.lock().expect("lock");
        let mut seats = std::collections::BTreeSet::new();
        for body in seen.iter() {
            let parsed: Value = serde_json::from_str(body).expect("json body");
            // The seat comes from the model the request went to, and the prompt must be that seat's own: a prompt
            // installed on the wrong seat turns this red (transposition mutation, Step 5). *(CP2 loop 5)*
            let seat = seat_of_model(parsed["model"].as_str().expect("model"));
            let system = parsed["messages"][0]["content"]
                .as_str()
                .expect("system message");
            let role = system.lines().next().unwrap_or("").to_lowercase();
            assert!(
                role.contains(seat),
                "{mode}: the {seat} model got the prompt of {role:?}"
            );
            let adj = system
                .find("## Additional instructions")
                .expect("block present");
            let sentinel = system.find(SENTINEL).expect("sentinel present");
            let anchor = system.find("## Output format").expect("anchor present");
            assert!(
                adj < sentinel && sentinel < anchor,
                "{seat}: wrong placement"
            );
            seats.insert(seat);
        }
        assert_eq!(
            seats.len(),
            3,
            "{mode}: the adjustment must reach all three mages: {seats:?}"
        );
    }
}

#[test]
fn the_adjustment_survives_a_rotation_to_a_declared_fallback() {
    let mut server = mockito::Server::new();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&bodies);
    let _trio = server
        .mock("POST", "/api/chat")
        .with_body_from_request(move |req| {
            let body = req.utf8_lossy_body().unwrap_or_default().to_string();
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let header = parsed["messages"][0]["content"]
                .as_str()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("")
                .to_lowercase();
            let seat = ["melchior", "balthasar", "caspar"]
                .into_iter()
                .find(|s| header.contains(s))
                .unwrap_or("unknown-seat");
            seen.lock().expect("lock").push(body);
            if seat == "melchior" && parsed["model"] == "mm" {
                // A 200 whose body is not a chat reply: measured to rotate in the Red dry run (2026-10-17). An HTTP
                // 500 was measured NOT to reach the fallback (CP2 loop 4), so it is not used here.
                return br#"{"error":"titular down"}"#.to_vec();
            }
            verdict_body(seat).into_bytes()
        })
        .create();
    let toml = consult_toml(&server.url()).replace("max_rotations = 0", "max_rotations = 1")
        + "[[magi.fallback]]\nmodel = \"f\"\nlineage = \"l4\"\n";
    let ws = workspace(&toml);
    let (code, out, err) = run(
        &ws,
        "consult",
        &["--mode", "analysis", "--allow-system-override"],
        &format!(r#"{{"prompt":"p","system":"{SENTINEL}"}}"#),
    );
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["applied_caps"]["system_override_applied"], true);
    let seen = bodies.lock().expect("lock");
    let fallback: Vec<&String> = seen
        .iter()
        .filter(|b| b.contains(r#""model":"f""#))
        .collect();
    assert!(
        !fallback.is_empty(),
        "the fallback was never called: the rotation did not happen"
    );
    assert!(
        fallback.iter().all(|b| b.contains(SENTINEL)),
        "the rotated seat lost the adjustment"
    );
}
