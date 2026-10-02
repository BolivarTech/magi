// Author: Julian Bolivar
// Version: 0.22.0
// Date: 2026-10-01

//! `magi decide` end to end against a local stand-in for a System One endpoint
//! (REQ-D22; S3, S4, S5, S6, S13, S20, S22, S23). The stand-in is `mockito`: these tests pin the
//! product's own behaviour, never a model's (real backends are SMOKE's job).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use mockito::Matcher;
use serde_json::{json, Value};

const EXIT_OK: i32 = 0;
const EXIT_RUNTIME: i32 = 1;
const EXIT_MISUSE: i32 = 2;

const OK_BODY: &str = r#"{"model":"nimble","answers":{"today":{"type":"noul","noul":0.9}},"usage":{"input_tokens":5,"output_tokens":1}}"#;
const INPUT: &str = r#"{"state":{"ticket":"x"},"questions":{"today":{"type":"noul","instructions":"Is it today?"}}}"#;

/// Variables that would make a result depend on the developer's shell.
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

/// A workspace scaffolded by the binary, whose `magi.toml` is replaced by `toml_text`.
fn workspace(toml_text: &str) -> Ws {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = dunce::canonicalize(tmp.path()).expect("canonical");
    let out = magi(&root).arg("init").output().expect("spawn init");
    assert!(
        out.status.success(),
        "init: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(root.join(".magi").join("magi.toml"), toml_text).expect("write toml");
    Ws { _tmp: tmp, root }
}

/// `provider = "ollama"`, root `base_url = <base>/v1`, `[decision]` with `nimble` and `extra` appended.
fn decision_toml(base: &str, extra: &str) -> String {
    format!(
        "provider = \"ollama\"\nbase_url = \"{base}/v1\"\n[decision]\nmodel = \"nimble\"\n{extra}"
    )
}

/// Runs `magi-rs decide --no-memory <args>` with `stdin`, returning (exit code, stdout, stderr).
fn decide(ws: &Ws, args: &[&str], stdin: &str, env: &[(&str, &str)]) -> (i32, String, String) {
    let mut c = magi(&ws.root);
    for (k, v) in env {
        c.env(k, v);
    }
    let mut child = c
        .arg("decide")
        .arg("--no-memory")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decide");
    if let Err(e) = child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stdin.as_bytes())
    {
        // A child that stops reading early (oversized input) closes its end: that is not a test failure.
        assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe, "write stdin: {e}");
    }
    let out = child.wait_with_output().expect("wait");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn envelope(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not one JSON value ({e}): {stdout}"))
}

fn closed_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

fn run_log(ws: &Ws) -> String {
    fn walk(dir: &Path, out: &mut String) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, out);
            } else if let Ok(bytes) = std::fs::read(&p) {
                out.push_str(&String::from_utf8_lossy(&bytes));
            }
        }
    }
    let mut out = String::new();
    walk(&ws.root.join(".magi").join("logs"), &mut out);
    out
}

#[test]
fn the_help_lists_every_flag_of_req_d22() {
    let out = magi(Path::new("."))
        .args(["decide", "--help"])
        .output()
        .expect("spawn");
    let help = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--input",
        "--output",
        "--questions",
        "--workdir",
        "--no-memory",
        "--timeout",
    ] {
        assert!(help.contains(flag), "{flag} missing:\n{help}");
    }
}

/// S1 shape and S23(e): stdout carries the envelope and nothing else.
#[test]
fn a_decision_from_stdin_prints_only_the_envelope_and_exits_0() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let (code, out, err) = decide(&ws, &[], INPUT, &[]);
    assert_eq!(code, EXIT_OK, "{err}");
    let v = envelope(&out);
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["model"], "nimble");
    assert_eq!(v["answers"]["today"]["type"], "noul");
    assert!(
        !err.contains("schema_version"),
        "the envelope goes to stdout only: {err}"
    );
}

/// S23(c).
#[test]
fn input_and_output_files_replace_stdin_and_stdout() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    std::fs::write(ws.root.join("in.json"), INPUT).expect("write input");
    let out_path = ws.root.join("out.json");
    let (code, out, err) = decide(&ws, &["-i", "in.json", "-o", "out.json"], "", &[]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.trim().is_empty(), "nothing on stdout with -o: {out}");
    let v: Value =
        serde_json::from_slice(&std::fs::read(out_path).expect("out.json")).expect("json");
    assert_eq!(v["model"], "nimble");
}

#[test]
fn an_unwritable_output_path_fails_like_query() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let (code, out, err) = decide(&ws, &["-o", "no/such/dir/out.json"], INPUT, &[]);
    assert_eq!(code, EXIT_MISUSE);
    assert!(out.trim().is_empty(), "{out}");
    // [PM-MS1-C1] exit 2 and an empty stdout are also what clap gives for an unknown subcommand, so
    // without this line the test passes in Red with no `decide` at all.
    assert!(
        err.contains("output parent directory does not exist"),
        "the failure is the output write, not argument parsing: {err}"
    );
}

/// S23(a) and S3: local validation errors never reach the endpoint.
#[test]
fn invalid_input_is_rejected_before_any_request() {
    let mut server = mockito::Server::new();
    let never = server.mock("POST", Matcher::Any).expect(0).create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    std::fs::write(
        ws.root.join("q.json"),
        r#"{"today":{"type":"noul","instructions":"?"}}"#,
    )
    .expect("q");
    let cases: [(&[&str], &str); 3] = [
        (&["-q", "q.json"], INPUT),
        (
            &[],
            r#"{"state":"x","questions":{"triage":{"type":"choice","instructions":"Which?"}}}"#,
        ),
        (&["--timeout", "0"], INPUT),
    ];
    for (args, stdin) in cases {
        let (code, out, err) = decide(&ws, args, stdin, &[]);
        assert_eq!(code, EXIT_MISUSE, "{args:?}: {err}");
        let v = envelope(&out);
        assert_eq!(v["error"]["kind"], "invalid_input", "{args:?}");
        assert_eq!(v["chain"], json!([]), "{args:?}");
    }
    never.assert();
}

/// S23(b): `--timeout` overrides `timeout_secs` per attempt.
#[test]
fn the_timeout_flag_overrides_the_toml_per_attempt() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let hold = std::thread::spawn(move || {
        let accepted = listener.accept();
        std::thread::sleep(std::time::Duration::from_secs(10));
        drop(accepted);
    });
    let ws = workspace(&decision_toml(
        &format!("http://{addr}"),
        "timeout_secs = 120\n",
    ));
    let (code, out, err) = decide(&ws, &["--timeout", "1"], INPUT, &[]);
    assert_eq!(code, EXIT_RUNTIME, "{err}");
    assert_eq!(envelope(&out)["chain"][0]["cause"], "timeout");
    drop(hold);
}

/// S23(d): an explicit model is tried alone.
#[test]
fn an_explicit_model_is_tried_alone() {
    let mut server = mockito::Server::new();
    let never = server
        .mock("POST", "/v1/systemone")
        .match_body(Matcher::PartialJson(json!({"model": "nimble"})))
        .expect(0)
        .create();
    server
        .mock("POST", "/v1/systemone")
        .match_body(Matcher::PartialJson(json!({"model": "tev1"})))
        .with_status(200)
        .with_body(OK_BODY.replace("nimble", "tev1"))
        .create();
    let ws = workspace(&decision_toml(
        &server.url(),
        "[[decision.fallback]]\nmodel = \"tev1\"\n",
    ));
    let input =
        r#"{"model":"tev1","state":"x","questions":{"today":{"type":"noul","instructions":"?"}}}"#;
    let (code, out, err) = decide(&ws, &[], input, &[]);
    assert_eq!(code, EXIT_OK, "{err}");
    let v = envelope(&out);
    assert_eq!(
        v["chain"],
        json!([{"model": "tev1", "outcome": "answered"}])
    );
    never.assert();
}

/// S4 (mutation-verified): the principal's key never reaches another origin, and startup is not blocked.
#[test]
fn the_principal_key_never_reaches_another_origin() {
    let mut principal = mockito::Server::new();
    let untouched = principal.mock("POST", Matcher::Any).expect(0).create();
    let mut other = mockito::Server::new();
    other
        .mock("POST", "/v1/systemone")
        .match_header("authorization", Matcher::Missing)
        .with_status(401)
        .create();
    let ws = workspace(&format!(
        "provider = \"ollama\"\nbase_url = \"{}/v1\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{}\"\n",
        principal.url(),
        other.url()
    ));
    let (code, out, err) = decide(&ws, &[], INPUT, &[("OPENAI_API_KEY", "prin-key")]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    let v = envelope(&out);
    assert_eq!(v["error"]["kind"], "missing_key");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("DECISION_API_KEY"));
    untouched.assert();
}

/// S5: the same origin inherits the principal's key.
#[test]
fn the_same_origin_inherits_the_principal_key() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", "Bearer prin-key")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let (code, _out, err) = decide(&ws, &[], INPUT, &[("OPENAI_API_KEY", "prin-key")]);
    assert_eq!(code, EXIT_OK, "{err}");
    m.assert();
}

#[test]
fn the_decision_key_wins_over_the_principal_key() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", "Bearer dec-key")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let env = [
        ("OPENAI_API_KEY", "prin-key"),
        ("DECISION_API_KEY", "dec-key"),
    ];
    let (code, _out, err) = decide(&ws, &[], INPUT, &env);
    assert_eq!(code, EXIT_OK, "{err}");
    m.assert();
}

#[test]
fn an_anthropic_principal_never_lends_its_key() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", Matcher::Missing)
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&format!(
        "provider = \"anthropic\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{}\"\n",
        server.url()
    ));
    let (code, _out, err) = decide(&ws, &[], INPUT, &[("ANTHROPIC_API_KEY", "ak-planted")]);
    assert_eq!(code, EXIT_OK, "{err}");
    m.assert();
}

/// REQ-D05: placeholders without a vault are `missing_key`, before any request.
#[test]
fn placeholders_without_a_vault_are_missing_key_before_any_request() {
    let mut server = mockito::Server::new();
    let never = server.mock("POST", Matcher::Any).expect(0).create();
    let base = server.url().replace("http://", "http://[user]:[password]@");
    let ws = workspace(&format!(
        "provider = \"ollama\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{base}\"\n"
    ));
    let (code, out, err) = decide(&ws, &[], INPUT, &[]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    let v = envelope(&out);
    assert_eq!(v["error"]["kind"], "missing_key");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("DECISION_BASE_URL_USER"));
    assert_eq!(v["chain"], json!([]));
    never.assert();
}

/// S13: no decision model answers — an error, never an answer.
#[test]
fn a_dead_endpoint_is_an_error_with_the_chain_and_no_answers() {
    let ws = workspace(&decision_toml(
        &format!("http://127.0.0.1:{}", closed_port()),
        "[[decision.fallback]]\nmodel = \"tev1\"\n",
    ));
    let (code, out, err) = decide(&ws, &[], INPUT, &[]);
    assert_eq!(code, EXIT_RUNTIME, "{err}");
    let v = envelope(&out);
    assert_eq!(v["error"]["kind"], "decision_unavailable");
    assert_eq!(v["chain"][0]["cause"], "transport");
    assert_eq!(v["chain"][1]["cause"], "transport");
    assert!(
        v.get("answers").is_none() && !out.contains("probabilities"),
        "{out}"
    );
}

/// S20: a base_url without the provider's API prefix is diagnosed, not corrected.
#[test]
fn a_missing_api_prefix_is_diagnosed_with_the_tried_url() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(404)
        .with_body("404 page not found")
        .create();
    let ws = workspace(&format!(
        "provider = \"ollama\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{}\"\n",
        server.url()
    ));
    let (code, out, err) = decide(&ws, &[], INPUT, &[]);
    assert_eq!(code, EXIT_RUNTIME, "{err}");
    let msg = envelope(&out)["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        msg.contains(&format!("{}/v1/systemone", server.url())) && msg.contains("base_url"),
        "{msg}"
    );
    // CP2 MS1 r2 (Caspar): S20 also puts the URL and the hint in each attempt's detail.
    let detail = envelope(&out)["chain"][0]["detail"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        detail.contains(&format!("{}/v1/systemone", server.url())) && detail.contains("base_url"),
        "{detail}"
    );
}

/// S22 for the CLI (mutation-verified): foreign secrets never reach stdout, stderr or the run log, and the
/// state never reaches the run log.
#[test]
fn foreign_secrets_and_the_state_never_leak() {
    let planted_key = "sk-ant-api03-planted0123456789abcdefghij"; // allow-secret-scan
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(422)
        .with_body(format!("{{\"error\":\"rejected PROVIDER-TEXT-a7c2; Authorization: Bearer {planted_key} at https://alice:s3cr3t@host/v1\"}}"))
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let state_marker = "STATE-MARKER-c41d";
    let input = format!(
        r#"{{"state":{{"note":"{state_marker}"}},"questions":{{"today":{{"type":"noul","instructions":"?"}}}}}}"#
    );
    let (code, out, err) = decide(&ws, &[], &input, &[]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(envelope(&out)["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("rejected"));
    let log = run_log(&ws);
    assert!(
        log.contains("nimble"),
        "the decision is in the run log (REQ-D17):\n{log}"
    );
    for (where_, text) in [("stdout", &out), ("stderr", &err), ("run log", &log)] {
        assert!(!text.contains(planted_key), "{where_} leaked the key");
        assert!(!text.contains("s3cr3t"), "{where_} leaked the credential");
    }
    assert!(
        !log.contains(state_marker),
        "the state never reaches the run log"
    );
    assert!(
        !log.contains("PROVIDER-TEXT-a7c2"),
        "provider text never reaches the run log (REQ-D11)"
    );
}

const PASS: &str = "correct horse battery staple";

/// Stores `value` under `name` in the workspace's vault (the value travels on stdin, never argv).
fn vault_set(ws: &Ws, name: &str, value: &str) {
    let mut child = magi(&ws.root)
        .args(["vault", "set", name, "-p", PASS])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vault set");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(value.as_bytes())
        .expect("write value");
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "vault set {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// CP2 MS1 r1 (Caspar): a DECISION_API_KEY stored in the vault is sent when the env has none.
#[test]
fn a_vault_decision_key_is_sent_when_the_env_has_none() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", "Bearer vault-key")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&format!(
        "provider = \"ollama\"\nbase_url = \"http://127.0.0.1:{}/v1\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{}\"\n",
        closed_port(),
        server.url()
    ));
    vault_set(&ws, "DECISION_API_KEY", "vault-key");
    let (code, _out, err) = decide(&ws, &[], INPUT, &[("MAGI_PASSPHRASE", PASS)]);
    assert_eq!(code, EXIT_OK, "{err}");
    m.assert();
}

/// REQ-D05 (CP2 MS1 r1, Caspar): both vault entries resolve the placeholders; the credential travels as Basic.
#[test]
fn decision_base_url_placeholders_resolve_from_the_vault() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", "Basic YWxpY2U6czNjcjN0")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let templated = server.url().replace("http://", "http://[user]:[password]@");
    let ws = workspace(&format!(
        "provider = \"ollama\"\nbase_url = \"http://127.0.0.1:{}/v1\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{templated}\"\n",
        closed_port()
    ));
    vault_set(&ws, "DECISION_BASE_URL_USER", "alice");
    vault_set(&ws, "DECISION_BASE_URL_PASSWORD", "s3cr3t");
    let (code, out, err) = decide(&ws, &[], INPUT, &[("MAGI_PASSPHRASE", PASS)]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(
        !out.contains("s3cr3t") && !err.contains("s3cr3t"),
        "the credential is never shown"
    );
    m.assert();
}

/// REQ-D05 (CP2 MS1 r1, Caspar): a missing password entry is missing_key naming it, before any request.
#[test]
fn a_missing_password_entry_is_missing_key_naming_it() {
    let mut server = mockito::Server::new();
    let never = server.mock("POST", Matcher::Any).expect(0).create();
    let templated = server.url().replace("http://", "http://[user]:[password]@");
    let ws = workspace(&format!(
        "provider = \"ollama\"\nbase_url = \"http://127.0.0.1:{}/v1\"\n[decision]\nmodel = \"nimble\"\nbase_url = \"{templated}\"\n",
        closed_port()
    ));
    vault_set(&ws, "DECISION_BASE_URL_USER", "alice");
    let (code, out, err) = decide(&ws, &[], INPUT, &[("MAGI_PASSPHRASE", PASS)]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    let v = envelope(&out);
    assert_eq!(v["error"]["kind"], "missing_key");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("DECISION_BASE_URL_PASSWORD"));
    assert_eq!(v["chain"], json!([]));
    never.assert();
}

/// REQ-D22 (CP2 MS1 r1, Caspar): with --no-memory and no `.magi/`, decide runs on the built-in defaults:
/// it reaches input validation (an envelope on stdout) instead of stopping at the workspace.
#[test]
fn without_a_workspace_no_memory_runs_on_the_defaults() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = dunce::canonicalize(tmp.path()).expect("canonical");
    let ws = Ws { _tmp: tmp, root };
    let (code, out, err) = decide(&ws, &[], r#"{"state":"x"}"#, &[]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert_eq!(
        envelope(&out)["error"]["kind"],
        "invalid_input",
        "validation ran, so the defaults loaded"
    );
    assert!(!ws.root.join(".magi").exists(), "nothing is scaffolded");
}

/// S6 through `decide`: an old toml is refused before the vault is touched.
#[test]
fn decide_refuses_an_old_toml() {
    let ws = workspace("provider = \"ollama\"\n");
    let (code, _out, err) = decide(&ws, &[], INPUT, &[]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("[decision]") && err.contains("magi-rs init"),
        "{err}"
    );
}

/// Review Focus 1: an oversized input is cut like `query`'s, before any request.
#[test]
fn an_oversized_input_is_cut_like_query() {
    let mut server = mockito::Server::new();
    let never = server.mock("POST", Matcher::Any).expect(0).create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let big = format!(
        r#"{{"state":"{}","questions":{{"today":{{"type":"noul","instructions":"?"}}}}}}"#,
        "x".repeat(10 * 1024 * 1024 + 1)
    );
    let (code, out, err) = decide(&ws, &[], &big, &[]);
    assert_eq!(code, EXIT_MISUSE);
    assert!(out.trim().is_empty(), "{out}");
    // [PM-MS1-C1] see `an_unwritable_output_path_fails_like_query`: exit 2 alone passes in Red.
    assert!(
        err.contains("input exceeds"),
        "the failure is the input cap, not argument parsing: {err}"
    );
    never.assert();
}

/// Review Focus 4: a blank DECISION_API_KEY is absent, never an empty Bearer.
#[test]
fn a_blank_decision_key_is_absent() {
    let mut server = mockito::Server::new();
    let m = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", "Bearer prin-key")
        .with_status(200)
        .with_body(OK_BODY)
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let env = [("OPENAI_API_KEY", "prin-key"), ("DECISION_API_KEY", "  ")];
    let (code, _out, err) = decide(&ws, &[], INPUT, &env);
    assert_eq!(code, EXIT_OK, "{err}");
    m.assert();
}

/// A registered secret that matches no pattern is masked by the exact pass wherever the provider
/// echoes it: here inside a rejection message that reaches the user envelope.
#[test]
fn a_registered_secret_echoed_by_the_provider_is_masked_in_the_envelope() {
    let toy_secret = "toysecret-9f3a1c7e5b2d4a60";
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/systemone")
        .with_status(422)
        .with_body(format!(
            "{{\"error\":\"rejected, saw {toy_secret} in the request\"}}"
        ))
        .create();
    let ws = workspace(&decision_toml(&server.url(), ""));
    let (code, out, err) = decide(&ws, &[], INPUT, &[("DECISION_API_KEY", toy_secret)]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    let v = envelope(&out);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("rejected"));
    for (where_, text) in [
        ("stdout", &out),
        ("stderr", &err),
        ("run log", &run_log(&ws)),
    ] {
        assert!(
            !text.contains(toy_secret),
            "{where_} leaked the registered secret"
        );
    }
}
