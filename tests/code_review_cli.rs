// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! `magi code-review` end to end (REQ-RV01, RV02, RV13; RV-S1, RV-S10; D24 `--structured-verdicts`) against one
//! mockito stand-in for the trio (`/api/chat`) and the gate (`/v1/systemone`): these tests pin the product's own
//! behaviour, never a model's.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde_json::Value;

const EXIT_OK: i32 = 0;
const EXIT_MISUSE: i32 = 2;
const TEST_PASSPHRASE: &str = "correct horse battery staple";
/// The committed file; its first line is what the stand-in trio points at, and the sentinel proves the
/// commit's content (not a guess) reached the trio and the gate.
const LIB_RS: &str = "pub fn constant() -> i32 {\n    41 // CR-SENTINEL-5d2e\n}\n";
const SENTINEL: &str = "CR-SENTINEL-5d2e";
const FINDING_TITLE: &str = "constant return value";
const GATE_MODEL: &str = "g1";
const SCRUBBED_ENV: [&str; 11] = [
    "MAGI_PASSPHRASE",
    "DECISION_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CODE_REVIEW_API_KEY",
    "MAGI_API_KEY",
    "EMBEDDING_API_KEY",
    "OPENAI_BASE_URL",
    "MAGI_PROVIDER",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
];
/// Above the configured floor (30 s): under Argon2 load a floor-sized clock times a healthy stand-in out (MS1 CP2
/// loop 6). The stand-in answers at once, so this never adds wall time.
const FIXTURE_AGENT_TIMEOUT_SECS: u64 = 120;
const SEAT_MODELS: [(&str, &str); 3] = [("mm", "melchior"), ("mb", "balthasar"), ("mc", "caspar")];

/// `root` is `<tmp>/ws`; `<tmp>/fuera/x.rs` exists beside it, so the hostile path is real and only the sandbox can
/// refuse it.
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

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .current_dir(root)
        // [CP2 MS2 S2b loop 6] core.autocrlf=false, §0.3's rule for git fixtures.
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A scaffolded workspace under `<tmp>/ws` with `toml_text` as its `magi.toml`, a sibling `<tmp>/fuera/x.rs`, and a
/// git repo whose only commit adds `.gitignore` and `src/lib.rs` (`.magi/` ignored: two files, one of them code).
fn workspace(toml_text: &str) -> Ws {
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = dunce::canonicalize(tmp.path()).expect("canonical");
    let root = base.join("ws");
    std::fs::create_dir_all(root.join("src")).expect("ws/src");
    std::fs::create_dir_all(base.join("fuera")).expect("fuera");
    std::fs::write(base.join("fuera").join("x.rs"), "fn outside() {}\n").expect("outside file");
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
    std::fs::write(root.join(".gitignore"), ".magi/\n").expect("gitignore");
    std::fs::write(root.join("src").join("lib.rs"), LIB_RS).expect("lib.rs");
    git(&root, &["init", "-q"]);
    git(&root, &["add", ".gitignore", "src/lib.rs"]);
    git(&root, &["commit", "-q", "-m", "init"]);
    Ws { _tmp: tmp, root }
}

/// Trio on the stand-in's native `/api/chat`, gate on its `/v1/systemone` (the section inherits the root
/// `base_url`, RV14), one gate model and no rotation anywhere, so every request is accounted for. `{base}/v1` is
/// right for both: magi-core's Ollama provider normalises a `/v1` suffix away (`providers/ollama.rs:100-105`, 4.2.0),
/// and `decide` appends `/systemone` to a URL ending in `/v1` (CP2 MS2 S2b loop 7, verified).
fn review_toml(base: &str) -> String {
    format!(
        "provider = \"ollama\"\nbase_url = \"{base}/v1\"\n[decision]\nmodel = \"nimble\"\n\
         [magi]\nmelchior_model = \"mm\"\nmelchior_lineage = \"l1\"\nbalthasar_model = \"mb\"\n\
         balthasar_lineage = \"l2\"\ncaspar_model = \"mc\"\ncaspar_lineage = \"l3\"\n\
         agent_timeout_secs = {FIXTURE_AGENT_TIMEOUT_SECS}\nmax_rotations = 0\n\
         [code_review]\nmodel = \"{GATE_MODEL}\"\nmin_confidence = 0.4\n"
    )
}

/// `magi-rs -p … code-review --no-memory <args>`: (exit code, stdout parsed as JSON or `Null`, stderr).
fn review(ws: &Ws, args: &[&str]) -> (i32, Value, String) {
    let out = magi(&ws.root)
        .args(["-p", TEST_PASSPHRASE, "code-review", "--no-memory"])
        .args(args)
        .output()
        .expect("spawn code-review");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (
        out.status.code().unwrap_or(-1),
        serde_json::from_str(stdout.trim()).unwrap_or(Value::Null),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn seat_of_model(model: &str) -> &'static str {
    SEAT_MODELS
        .iter()
        .find(|(m, _)| *m == model)
        .map(|(_, s)| *s)
        .unwrap_or("unknown-seat")
}

/// Native `/api/chat` reply: an approving verdict carrying ONE finding on `src/lib.rs:2`, the sentinel's line.
/// Third copy of this shape (see the task text); the three change together.
fn verdict_body(agent: &str) -> String {
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
    let verdict = format!(
        "{VERDICT_OPEN}\n{{\"agent\":\"{agent}\",\"verdict\":\"approve\",\"confidence\":0.9,\
         \"summary\":\"ok\",\"reasoning\":\"r\",\"recommendation\":\"go\",\"findings\":[\
         {{\"severity\":\"warning\",\"title\":\"{FINDING_TITLE}\",\"detail\":\"d\",\
         \"file\":\"src/lib.rs\",\"line\":2,\"category\":\"logic\"}}]}}\n{VERDICT_CLOSE}"
    );
    serde_json::json!({"message": {"content": verdict}, "done_reason": "stop",
                       "eval_count": 128, "prompt_eval_count": 64})
    .to_string()
}

/// A gate answer: `real_defect` at 0.95, above `min_confidence` 0.4, so the finding is `kept`.
fn gate_body() -> String {
    serde_json::json!({"model": GATE_MODEL,
        "answers": {"gate": {"type": "choice", "choice": "real_defect",
                             "probabilities": {"real_defect": 0.95, "false_positive": 0.05}}},
        "usage": {"input_tokens": 5, "output_tokens": 1}})
    .to_string()
}

/// Every request body one stand-in received, in arrival order.
type Bodies = Arc<Mutex<Vec<String>>>;

/// Both stand-ins on one server, each recording every request body it receives. A named alias, not the spelled-out
/// tuple: `clippy::type_complexity` rejects the latter under `-D warnings` (RED-DRYRUN 2026-10-17, measured).
fn mount(server: &mut mockito::ServerGuard) -> (Bodies, Bodies, mockito::Mock, mockito::Mock) {
    let trio_bodies = Arc::new(Mutex::new(Vec::new()));
    let gate_bodies = Arc::new(Mutex::new(Vec::new()));
    let (tb, gb) = (Arc::clone(&trio_bodies), Arc::clone(&gate_bodies));
    let trio = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_body_from_request(move |req| {
            let body = req.utf8_lossy_body().unwrap_or_default().to_string();
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let seat = seat_of_model(parsed["model"].as_str().unwrap_or(""));
            tb.lock().expect("lock").push(body);
            verdict_body(seat).into_bytes()
        })
        .create();
    let gate = server
        .mock("POST", "/v1/systemone")
        .with_status(200)
        .with_body_from_request(move |req| {
            gb.lock()
                .expect("lock")
                .push(req.utf8_lossy_body().unwrap_or_default().to_string());
            gate_body().into_bytes()
        })
        .create();
    (trio_bodies, gate_bodies, trio, gate)
}

/// RV-S10: the sandbox and the ref validation run before any model is contacted.
#[test]
fn rv_s10_a_hostile_path_or_ref_fails_with_exit_2_before_any_model_call() {
    let mut server = mockito::Server::new();
    let (trio_bodies, gate_bodies, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, _, err) = review(&ws, &["../fuera/x.rs"]);
    assert_eq!(code, EXIT_MISUSE, "outside path: {err}");
    assert!(
        err.contains("fuera"),
        "stderr must name the refused path: {err}"
    );
    // `=` forces the value: clap would otherwise read `--output=x` as `-o` and never reach the ref check.
    let (code, _, err) = review(&ws, &["--commit=--output=x"]);
    assert_eq!(code, EXIT_MISUSE, "option-shaped ref: {err}");
    assert!(
        err.contains("--output=x"),
        "stderr must name the refused ref: {err}"
    );
    assert!(
        trio_bodies.lock().expect("lock").is_empty(),
        "the trio was contacted"
    );
    assert!(
        gate_bodies.lock().expect("lock").is_empty(),
        "the gate was contacted"
    );
    // [CP2 MS2 S2b loop 7] RV-S7's run-log half at the subcommand wiring: a bare eprintln! would leave this empty.
    let logs: String = std::fs::read_dir(ws.root.join(".magi").join("logs"))
        .expect("run log dir")
        .filter_map(Result::ok)
        .map(|e| std::fs::read_to_string(e.path()).unwrap_or_default())
        .collect();
    assert!(
        logs.contains("fuera") && logs.contains("--output=x"),
        "both refusals must reach the run log: {logs}"
    );
}

/// [MAGI §6 MS2 E loop 1] RV01 first: a refused input never reaches the endpoints, not even the
/// capability probe that sizes the batches.
#[test]
fn a_refused_input_contacts_no_endpoint_not_even_the_probe() {
    let mut server = mockito::Server::new();
    let (trio_bodies, gate_bodies, _t, _g) = mount(&mut server);
    let probes = Arc::new(Mutex::new(0usize));
    let (p1, p2) = (Arc::clone(&probes), Arc::clone(&probes));
    let _show = server
        .mock("POST", "/api/show")
        .with_status(404)
        .with_body_from_request(move |_| {
            *p1.lock().expect("lock") += 1;
            Vec::new()
        })
        .create();
    let _tags = server
        .mock("GET", "/api/tags")
        .with_status(404)
        .with_body_from_request(move |_| {
            *p2.lock().expect("lock") += 1;
            Vec::new()
        })
        .create();
    let ws = workspace(&review_toml(&server.url()));
    for args in [
        &["../fuera/x.rs"][..],
        &["--commit=--output=x"],
        &["--commit", "nope"],
    ] {
        let (code, _, err) = review(&ws, args);
        assert_eq!(code, EXIT_MISUSE, "{args:?}: {err}");
    }
    assert_eq!(*probes.lock().expect("lock"), 0, "the probe was contacted");
    assert!(trio_bodies.lock().expect("lock").is_empty());
    assert!(gate_bodies.lock().expect("lock").is_empty());
}

/// RV-S1 with an explicit commit: one magi-core report, the commit's content sent to the three seats and to the gate,
/// and the `code_review` block (RV13) on the envelope that `RunOutcome.code_review` (T2.9) carries.
#[test]
fn rv_s1_a_committed_change_gets_one_magi_core_report_with_the_gate_states() {
    let mut server = mockito::Server::new();
    let (trio_bodies, gate_bodies, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, out, err) = review(&ws, &["--commit", "HEAD", "--output-format", "json"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(
        out["response"]
            .as_str()
            .is_some_and(|r| r.contains("MAGI SYSTEM -- VERDICT")),
        "the report must be magi-core's: {out}"
    );
    assert!(out["consult"].is_object(), "the consult object: {out}");
    let cr = &out["code_review"];
    assert_eq!(cr["batches_planned"], 1, "{cr}");
    assert_eq!(cr["batches_completed"], 1, "{cr}");
    assert_eq!(
        cr["gate_states"]["kept"], 1,
        "the three seats' finding dedups to one, kept: {cr}"
    );
    assert_eq!(cr["per_model"][GATE_MODEL], 1, "{cr}");
    assert!(
        out["consult"].get("agents").is_none(),
        "agents only with --structured-verdicts: {out}"
    );
    let trio = trio_bodies.lock().expect("lock");
    assert_eq!(trio.len(), 3, "one batch, three seats, no retry");
    assert!(
        trio.iter().all(|b| b.contains(SENTINEL)),
        "the commit's content must reach every seat"
    );
    let gate = gate_bodies.lock().expect("lock");
    assert_eq!(gate.len(), 1, "one consolidated finding, one gate call");
    assert!(
        gate[0].contains(SENTINEL) && gate[0].contains(FINDING_TITLE),
        "state = finding + code: {}",
        gate[0]
    );
    assert!(
        gate[0].contains(&format!("\"model\":\"{GATE_MODEL}\"")),
        "{}",
        gate[0]
    );
}

/// RV-S1 as the spec states it: no git argument reviews HEAD's changes (`ReviewInput::Commit(None)`) end to end.
#[test]
fn rv_s1_with_no_git_argument_reviews_the_head_commit() {
    // [CP2 MS2 S2b loop 6] the default is pinned at clap level elsewhere; this composes it with the pipeline.
    let mut server = mockito::Server::new();
    let (trio_bodies, _gate_bodies, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, out, err) = review(&ws, &["--output-format", "json"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["code_review"]["batches_completed"], 1, "{out}");
    assert!(
        trio_bodies
            .lock()
            .expect("lock")
            .iter()
            .all(|b| b.contains(SENTINEL)),
        "HEAD's content must reach the trio"
    );
}

/// The most common invocation, plain text: magi-core's report on stdout.
#[test]
fn the_default_text_output_is_the_magi_core_report() {
    // [CP2 MS2 S2b loop 7] every other exit-0 test reads JSON.
    let mut server = mockito::Server::new();
    let (_tb, _gb, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let out = magi(&ws.root)
        .args([
            "-p",
            TEST_PASSPHRASE,
            "code-review",
            "--no-memory",
            "--commit",
            "HEAD",
        ])
        .output()
        .expect("spawn code-review");
    assert_eq!(
        out.status.code(),
        Some(EXIT_OK),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("MAGI SYSTEM -- VERDICT"));
    // [CP2 MS2 S2b loop 8] the JSON envelope carries the same banner inside `response`: text is not JSON.
    assert!(
        serde_json::from_slice::<Value>(&out.stdout).is_err(),
        "text output must not be the JSON envelope"
    );
}

/// A positional path that names a directory is reviewed as a directory (`path_input`, the rule the tool shares).
#[test]
fn a_positional_directory_is_reviewed_as_a_directory() {
    // [CP2 MS2 S2b loop 8] the runner's wiring of path_input, end to end.
    let mut server = mockito::Server::new();
    let (trio_bodies, _gb, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, out, err) = review(&ws, &["src", "--output-format", "json"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert_eq!(out["code_review"]["batches_completed"], 1, "{out}");
    assert!(
        trio_bodies
            .lock()
            .expect("lock")
            .iter()
            .all(|b| b.contains(SENTINEL)),
        "src/lib.rs reached the trio"
    );
}

/// D24 + REQ-EA06 as in `consult`: `--structured-verdicts` needs json (exit 2, nothing contacted) and, with it,
/// the consolidated agents and consensus reach the `consult` object.
#[test]
fn structured_verdicts_need_json_and_add_the_consolidated_agents_and_consensus() {
    let mut server = mockito::Server::new();
    let (trio_bodies, _gate_bodies, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, _, err) = review(&ws, &["--commit", "HEAD", "--structured-verdicts"]);
    assert_eq!(code, EXIT_MISUSE, "{err}");
    assert!(
        err.contains("--structured-verdicts needs --output-format json"),
        "{err}"
    );
    assert!(
        trio_bodies.lock().expect("lock").is_empty(),
        "a misuse must cost no model call"
    );
    let (code, out, err) = review(
        &ws,
        &[
            "--commit",
            "HEAD",
            "--output-format",
            "json",
            "--structured-verdicts",
        ],
    );
    assert_eq!(code, EXIT_OK, "{err}");
    let agents = out["consult"]["agents"].as_array().expect("agents array");
    assert_eq!(agents.len(), 3, "one consolidated entry per seat: {out}");
    assert!(
        out["consult"]["consensus"]["consensus_verdict"].is_string(),
        "{out}"
    );
}

/// The envelope's `usage` carries the review's tokens (`code_review.batch_totals`), never zero.
#[test]
fn the_envelope_usage_carries_the_batch_token_totals() {
    let mut server = mockito::Server::new();
    let (_tb, _gb, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, out, err) = review(&ws, &["--commit", "HEAD", "--output-format", "json"]);
    assert_eq!(code, EXIT_OK, "{err}");
    let totals = &out["code_review"]["batch_totals"];
    assert!(
        totals["input_tokens"].as_u64().is_some_and(|n| n > 0),
        "{out}"
    );
    assert_eq!(
        out["usage"]["input_tokens"], totals["input_tokens"],
        "{out}"
    );
    assert_eq!(
        out["usage"]["output_tokens"], totals["output_tokens"],
        "{out}"
    );
}

/// The `[code_review]` key, given only as `CODE_REVIEW_API_KEY`. Not a real key.
const GATE_KEY: &str = "cr-gate-test-key-4f1d7"; // allow-secret-scan

/// The section's own key reaches the gate's client: the gate stand-in answers only a request
/// that carries it, so a client built without the key leaves the finding unjudged.
#[test]
fn the_code_review_key_reaches_the_gate() {
    let mut server = mockito::Server::new();
    let _trio = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_body_from_request(|req| {
            let body = req.utf8_lossy_body().unwrap_or_default().to_string();
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            verdict_body(seat_of_model(parsed["model"].as_str().unwrap_or(""))).into_bytes()
        })
        .create();
    let gate = server
        .mock("POST", "/v1/systemone")
        .match_header("authorization", format!("Bearer {GATE_KEY}").as_str())
        .with_status(200)
        .with_body(gate_body())
        .expect(1)
        .create();
    let ws = workspace(&review_toml(&server.url()));
    let out = magi(&ws.root)
        .env("CODE_REVIEW_API_KEY", GATE_KEY)
        .args([
            "-p",
            TEST_PASSPHRASE,
            "code-review",
            "--no-memory",
            "--commit",
            "HEAD",
        ])
        .args(["--output-format", "json"])
        .output()
        .expect("spawn code-review");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(
        out.status.code(),
        Some(EXIT_OK),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_str(stdout.trim()).unwrap_or(Value::Null);
    assert_eq!(v["code_review"]["gate_states"]["kept"], 1, "{v}");
    gate.assert();
}

/// RV-S7 at the binary (MAGI S6 loop 3, E8): an explicit `--timeout` too short for the planned batches is refused before
/// any model is contacted, with exit 2 and the batches, ceiling, floor and smallest `--timeout` on stderr even under
/// `--log-level error` (RV06: stderr whatever `--log-level` says).
#[test]
fn rv_s7_an_insufficient_timeout_is_refused_on_stderr_with_exit_2_before_any_model_call() {
    let mut server = mockito::Server::new();
    let (trio_bodies, gate_bodies, _t, _g) = mount(&mut server);
    let ws = workspace(&review_toml(&server.url()));
    let (code, _, err) = review(
        &ws,
        &["--commit", "HEAD", "--timeout", "1", "--log-level", "error"],
    );
    assert_eq!(code, EXIT_MISUSE, "{err}");
    for needle in [
        "1 batches",
        "per batch",
        "floor",
        "per-batch ceiling",
        "pass --timeout",
    ] {
        assert!(err.contains(needle), "stderr must carry `{needle}`: {err}");
    }
    assert!(
        trio_bodies.lock().expect("lock").is_empty(),
        "the trio was contacted"
    );
    assert!(
        gate_bodies.lock().expect("lock").is_empty(),
        "the gate was contacted"
    );
}
