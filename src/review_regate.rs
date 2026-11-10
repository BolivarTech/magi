// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! Criterion 2 of the v0.25.0 certification (spec §3): the `verify` half of the gate calibration,
//! re-judged by the review's real gate.
//!
//! Test-only (`#[cfg(test)] mod review_regate;` in `main.rs`). It lives in the bin because
//! `MagiConfig` and `effective_code_review()` are bin items: an integration test sees only the
//! lib. Nothing here adds surface to the product: the measurement is the `#[ignore]` test
//! `regate_verify_set`, run with `cargo nextest run --run-ignored only regate_verify_set`.
//! It sits at the crate root on purpose: the bin has no test tree, and a `#[path]` module in a
//! test directory would add no isolation (`cfg(test)` already keeps it out of every build).

/// Criterion 2's floor on the false positives the gate removes, in percent.
const FP_REMOVED_MIN_PCT: f64 = 40.0;
/// Criterion 2's ceiling on the real defects the gate drops, in percent.
const REAL_DROPPED_MAX_PCT: f64 = 5.0;
/// The calibration class of a finding that is wrong about the code.
const CLASS_FALSE_POSITIVE: &str = "false_positive";
/// The calibration class of a finding that points to a real defect.
const CLASS_REAL_DEFECT: &str = "real_defect";

/// The verdict of criterion 2 over a set of judged items.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Criterion2 {
    /// `false_positive` items that came out `filtered`, in percent of the `false_positive` items.
    fp_removed_pct: f64,
    /// `real_defect` items that came out `filtered`, in percent of the `real_defect` items.
    real_dropped_pct: f64,
    /// `fp_removed_pct >= 40` and `real_dropped_pct <= 5`.
    pass: bool,
}

/// Computes criterion 2 from each item's expected class and gate state.
///
/// Only `filtered` removes a finding: `kept`, `low_confidence` and `unfiltered` stay in the
/// report. Each percentage is `100.0 * n / d`, in that order.
///
/// # Arguments
/// * `rows` - `(class, state)` per item: the calibration class and the gate state.
///
/// # Returns
/// The two percentages and whether both thresholds hold. A class with no item gives a NaN
/// percentage, which never passes. O(rows).
fn criterion_2(rows: &[(&str, &str)]) -> Criterion2 {
    /// The only gate state that removes a finding from the report.
    const STATE_FILTERED: &str = "filtered";
    let filtered_pct = |class: &str| {
        let (total, filtered) = rows
            .iter()
            .filter(|(c, _)| *c == class)
            .fold((0usize, 0usize), |(t, f), (_, s)| {
                (t + 1, f + usize::from(*s == STATE_FILTERED))
            });
        100.0 * filtered as f64 / total as f64
    };
    let fp_removed_pct = filtered_pct(CLASS_FALSE_POSITIVE);
    let real_dropped_pct = filtered_pct(CLASS_REAL_DEFECT);
    Criterion2 {
        fp_removed_pct,
        real_dropped_pct,
        pass: fp_removed_pct >= FP_REMOVED_MIN_PCT && real_dropped_pct <= REAL_DROPPED_MAX_PCT,
    }
}

// The tests and the measurement's adapter live in their own module, after the subject, like
// every other file's: the source guards read a file's production code up to its first
// `#[cfg(test)]` module.
#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use magi_core::schema::Severity;
    use magi_rs::magi::endpoint::Scope;
    use magi_rs::review::consolidate::ConsolidatedFinding;
    use magi_rs::review::gate::Gate;
    use magi_rs::section_keys::{ApiKeys, KeyInputs, SectionKeys};
    use serde_json::{json, Value};

    use super::*;
    use crate::config::{resolve_effective_provider_kind, MagiConfig};
    use crate::{
        code_review_api_key, registered_magi_kind, resolve_endpoints, resolve_section_endpoint,
        section_endpoints,
    };

    /// The whole file a calibration window came from, as far as the window shows it.
    ///
    /// The calibration's `state.code` is the file's lines `line +/- 40`, each written
    /// `{n:5}: text`. `Gate::judge` takes the whole file and numbers its own window, so the
    /// window is put back at its line numbers (the lines before it empty): the gate then shows
    /// exactly those lines, with its own numbering.
    ///
    /// # Arguments
    /// * `window` - The numbered window.
    ///
    /// # Returns
    /// The file text, one `\n`-terminated line per line number up to the window's last.
    ///
    /// # Panics
    /// When a line of the window does not start with its number (from 1).
    fn file_from_window(window: &str) -> String {
        let mut lines: Vec<&str> = Vec::new();
        for numbered in window.lines() {
            let (number, text) = numbered
                .trim_start()
                .split_once(':')
                .expect("a numbered window line");
            let n: usize = number.parse().expect("a line number");
            assert!(n >= 1, "line numbers start at 1");
            if lines.len() < n {
                lines.resize(n, "");
            }
            lines[n - 1] = text.strip_prefix(' ').unwrap_or(text);
        }
        lines.iter().map(|l| format!("{l}\n")).collect()
    }

    /// The `ConsolidatedFinding` of a calibration item (`state.finding`), as MS2 T2.7 builds it.
    ///
    /// # Panics
    /// When the item lacks a field the calibration always writes.
    fn finding_of(state: &Value) -> ConsolidatedFinding {
        let f = &state["finding"];
        let text = |k: &str| {
            f[k].as_str()
                .expect("a calibration finding field")
                .to_string()
        };
        let severity: Severity = serde_json::from_value(f["severity"].clone()).expect("a severity");
        ConsolidatedFinding {
            file: Some(text("file")),
            line: Some(text("line").parse().expect("a line number")),
            severity,
            reported_severity: severity,
            titles: vec![text("title")],
            details: vec![text("detail")],
            categories: vec![],
            sources: vec![],
            marks: vec![],
            batches: vec![0],
        }
    }

    /// The objects of a JSONL file.
    ///
    /// # Panics
    /// When the file cannot be read or a line is not JSON.
    fn read_jsonl(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("a JSON line"))
            .collect()
    }

    /// A path from the environment.
    ///
    /// # Panics
    /// When the variable is not set.
    fn env_path(name: &str) -> PathBuf {
        PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
    }

    #[test]
    fn the_criterion_2_verdict_is_computed_from_the_states() {
        // 10 false positives with 4 filtered (40 %), 20 real defects with 1 filtered (5 %): pass, at both edges.
        let mut rows: Vec<(&str, &str)> = Vec::new();
        // `repeat_n` where the plan wrote `repeat(..).take(n)`: the same rows, which clippy's
        // `manual_repeat_n` (rustc 1.99) rejects under -D warnings.
        rows.extend(std::iter::repeat_n(("false_positive", "filtered"), 4));
        rows.extend(std::iter::repeat_n(("false_positive", "low_confidence"), 6));
        rows.extend(std::iter::repeat_n(("real_defect", "filtered"), 1));
        rows.extend(std::iter::repeat_n(("real_defect", "kept"), 19));
        let c = criterion_2(&rows);
        assert_eq!(
            (c.fp_removed_pct, c.real_dropped_pct, c.pass),
            (40.0, 5.0, true)
        );
        rows[11] = ("real_defect", "filtered"); // 2 of 20 real defects: 10 %
        assert!(!criterion_2(&rows).pass);
    }

    #[test]
    fn a_calibration_window_reaches_the_gate_with_its_lines_at_their_numbers() {
        // The adapter of `regate_verify_set`: the gate re-numbers the code it is given, so the
        // window has to come back at its own line numbers or the gate shows other lines. A
        // calibration window starts at `line - 40`: here lines 10..=90 around line 50, one blank.
        let text = |n: u32| {
            if n == 60 {
                String::new()
            } else {
                format!("l{n}")
            }
        };
        let window: String = (10..=90).map(|n| format!("{n:5}: {}\n", text(n))).collect();
        let shown: String = (10..=90).map(|n| format!("{n}: {}\n", text(n))).collect();
        let finding = ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(50),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![],
            sources: vec![],
            marks: vec![],
            batches: vec![0],
        };
        let state = magi_rs::review::gate::gate_state(&finding, &file_from_window(&window));
        assert_eq!(state["code"], shown.as_str());
    }

    /// Re-judges the calibration's `verify` half with the real gate (spec §3, criterion 2).
    ///
    /// Environment: `MAGI_REGATE_SET` (the `verify` rows of `calibration/set.jsonl`, copied
    /// unchanged), `MAGI_REGATE_CALIBRATION` (`calibration/results.jsonl`, to check the ids),
    /// `MAGI_REGATE_OUT` (the JSONL written, one row per item), and optionally
    /// `MAGI_REGATE_WORKSPACE` (a directory whose `.magi/magi.toml` is loaded; without it, the
    /// built-in defaults). Keys come from the environment only (no vault is opened).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "calls the real gate: cargo nextest run --run-ignored only regate_verify_set"]
    async fn regate_verify_set() {
        let set = read_jsonl(&env_path("MAGI_REGATE_SET"));
        let calibration = read_jsonl(&env_path("MAGI_REGATE_CALIBRATION"));
        let out_path = env_path("MAGI_REGATE_OUT");

        // The set is the calibration's verify half, exactly: 89 distinct ids, 42 + 47, same classes.
        let verify: BTreeMap<String, String> = calibration
            .iter()
            .filter(|r| r["half"] == "verify")
            .map(|r| {
                (
                    r["id"].as_str().expect("an id").to_string(),
                    r["class"].as_str().expect("a class").to_string(),
                )
            })
            .collect();
        let ids: BTreeSet<&str> = set
            .iter()
            .map(|r| r["id"].as_str().expect("an id"))
            .collect();
        let classes = |c: &str| set.iter().filter(|r| r["class"] == c).count();
        assert_eq!(
            (
                set.len(),
                ids.len(),
                classes(CLASS_FALSE_POSITIVE),
                classes(CLASS_REAL_DEFECT)
            ),
            (89, 89, 42, 47),
            "the set is not the verify half"
        );
        assert_eq!(
            ids,
            verify.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            "the set's ids are not the calibration's verify ids"
        );
        assert!(
            set.iter().all(|r| verify
                .get(r["id"].as_str().unwrap_or_default())
                .map(String::as_str)
                == r["class"].as_str()),
            "a class differs from the calibration's"
        );

        // The configuration, the endpoint and the key, resolved as `magi code-review` does.
        let cfg = match std::env::var("MAGI_REGATE_WORKSPACE") {
            Ok(ws) => {
                MagiConfig::load(&PathBuf::from(ws).join(".magi").join("magi.toml"))
                    .expect("the workspace's magi.toml loads")
                    .0
            }
            Err(_) => MagiConfig::default(),
        };
        let section = cfg.effective_code_review();
        let chain = section.chain();
        let u14: Vec<(&str, Option<f64>)> = vec![
            ("clef:cloud", Some(0.40)),
            ("clef-flash", Some(0.32)),
            ("nimble", Some(0.80)),
        ];
        assert_eq!(
            chain
                .iter()
                .map(|m| (m.model.as_str(), m.min_confidence))
                .collect::<Vec<_>>(),
            u14,
            "the gate's chain is not U14's default: this measurement would certify nothing"
        );
        let env = |k: &str| std::env::var(k).ok();
        let api_keys = ApiKeys {
            anthropic_key: env("ANTHROPIC_API_KEY"),
            openai_key: env("OPENAI_API_KEY"),
            decision_key: env("DECISION_API_KEY"),
            magi_key: env("MAGI_API_KEY"),
            embedding_key: env("EMBEDDING_API_KEY"),
            code_review_key: env("CODE_REVIEW_API_KEY"),
        };
        let provider_kind = resolve_effective_provider_kind(&cfg, env("MAGI_PROVIDER").as_deref())
            .expect("the provider kind");
        let endpoints = resolve_endpoints(&cfg, env("OPENAI_BASE_URL").as_deref(), None)
            .expect("the endpoints");
        let (embedding, decision, code_review) = section_endpoints(&cfg, None);
        let keys = SectionKeys::resolve(&KeyInputs {
            env: &api_keys,
            secret_store: None,
            magi_kind: registered_magi_kind(&cfg, provider_kind),
            root_endpoint: endpoints.root.as_str(),
            magi_endpoint: Some(endpoints.magi.as_str()),
            embedding_endpoint: embedding.as_deref(),
            decision_endpoint: decision.as_deref(),
            code_review_endpoint: code_review.as_deref(),
        });
        let template = cfg
            .effective_code_review_base_url()
            .expect("the [code_review] endpoint");
        let base_url =
            resolve_section_endpoint(&template, Scope::CodeReview, None).expect("resolves");
        let call_timeout = Duration::from_secs(
            section
                .timeout_secs
                .unwrap_or(crate::defaults::DEFAULT_CODE_REVIEW_TIMEOUT_SECS),
        );
        let client = crate::code_review_runner::gate_client(
            base_url.as_str(),
            code_review_api_key(&keys),
            &chain,
            call_timeout,
        )
        .expect("the gate's client");

        // One gate for every item, as in a review of 89 findings: the 400 memory is exercised.
        let mut gate = Gate::new(
            &client,
            chain,
            section
                .max_state_bytes
                .unwrap_or(crate::defaults::DEFAULT_CODE_REVIEW_MAX_STATE_BYTES),
        )
        .with_call_timeout(call_timeout);
        let mut out =
            std::io::BufWriter::new(std::fs::File::create(&out_path).expect("the out file"));
        let mut judged: Vec<(String, String)> = Vec::with_capacity(set.len());
        for row in &set {
            let state = &row["state"];
            let code = file_from_window(state["code"].as_str().expect("the code window"));
            let v = gate.judge(&finding_of(state), Some(&code), None).await;
            let verdict = serde_json::to_value(&v).expect("a verdict serializes");
            let class = row["class"].as_str().expect("a class").to_string();
            let state_name = verdict["state"].as_str().expect("a state").to_string();
            writeln!(
                out,
                "{}",
                json!({"id": row["id"], "class": class, "gate": verdict})
            )
            .expect("writes");
            judged.push((class, state_name));
        }
        out.flush().expect("flushes");
        let rows: Vec<(&str, &str)> = judged
            .iter()
            .map(|(c, s)| (c.as_str(), s.as_str()))
            .collect();
        let c = criterion_2(&rows);
        let line = format!(
            "criterion 2: false positives removed {:.1} % (>= {FP_REMOVED_MIN_PCT}), real \
             defects dropped {:.1} % (<= {REAL_DROPPED_MAX_PCT}), per model {:?}, stopped {:?}: {}",
            c.fp_removed_pct,
            c.real_dropped_pct,
            gate.per_model(),
            gate.stopped(),
            if c.pass { "PASS" } else { "FAIL" }
        );
        println!("{line}");
        assert!(c.pass, "{line}");
    }
}
