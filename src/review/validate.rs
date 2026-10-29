// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Deterministic validation of the findings a batch returned (REQ-RV09).

use std::collections::HashMap;
use std::path::Path;

use magi_core::schema::{AgentName, AgentOutput, Finding, Severity};

use super::index::{LexicalIndex, Reviewed, PATH_SEPARATOR};

/// Why a finding left the main list.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum DiscardCause {
    /// The finding's file was never shown to the trio.
    #[serde(rename = "file not reviewed")]
    FileNotReviewed,
    /// The finding's line lies outside what was shown.
    #[serde(rename = "line out of range")]
    LineOutOfRange,
    /// The finding cites a symbol the reviewed code does not contain.
    #[serde(rename = "unknown symbol")]
    UnknownSymbol,
}

/// A finding removed from (or flagged in) the main list, with the reason.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Discarded {
    /// The mage that reported it.
    pub agent: AgentName,
    /// Zero-based batch index.
    pub batch: usize,
    /// The finding as the mage reported it.
    pub finding: Finding,
    /// The failed check.
    pub cause: DiscardCause,
    /// The first cited symbol that is missing, for `UnknownSymbol`.
    pub symbol: Option<String>,
}

/// What `validate_batch` decided for one batch.
#[derive(Debug, Default)]
pub struct Validation {
    /// Findings removed from their agent.
    pub discarded: Vec<Discarded>,
    /// Reported `critical` whose only failed check is the symbol one: kept for the gate (U21).
    pub symbol_unverified: Vec<Discarded>,
}

/// Whether `s` is `[A-Za-z_][A-Za-z0-9_]*`.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Whether `s` is identifiers joined by `::` or `.`.
fn is_citation(s: &str) -> bool {
    s.split(PATH_SEPARATOR)
        .all(|part| part.split('.').all(is_identifier))
}

/// The code spans of `text` that look like a symbol: identifiers joined by `::` or `.`, between a
/// pair of single backticks (`Baz::qux`, `self.helper`). Spans follow CommonMark: a run of
/// backticks opens a span that only the next run of the same length closes, and everything in
/// between is literal. So only a single-backtick span can be a citation (``` ``x`` ``` is not),
/// single backticks nested inside a longer span (``` `` a `b` c `` ```) never pair, and a run
/// that nothing closes is plain text, after which later runs may still open spans.
///
/// Complexity: O(len of `text`).
pub(crate) fn cited_identifiers(text: &str) -> Vec<String> {
    // Runs of consecutive backticks as (start byte, length); a backtick is one byte.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (pos, c) in text.char_indices() {
        if c != '`' {
            continue;
        }
        match runs.last_mut() {
            Some((start, len)) if *start + *len == pos => *len += 1,
            _ => runs.push((pos, 1)),
        }
    }
    // For each run, the index of the next run of the same length: its closer, if any.
    let mut closer = vec![None; runs.len()];
    let mut next_of_len: HashMap<usize, usize> = HashMap::new();
    for (i, &(_, len)) in runs.iter().enumerate().rev() {
        if let Some(slot) = closer.get_mut(i) {
            *slot = next_of_len.insert(len, i);
        }
    }
    let mut cited = Vec::new();
    let mut i = 0;
    while let Some(&(open, open_len)) = runs.get(i) {
        let close = closer.get(i).copied().flatten();
        match close.and_then(|j| runs.get(j).map(|&(start, _)| (j, start))) {
            Some((j, close_start)) => {
                if open_len == 1 {
                    if let Some(span) = text.get(open + 1..close_start).filter(|s| is_citation(s)) {
                        cited.push(span.to_string());
                    }
                }
                i = j + 1;
            }
            None => i += 1,
        }
    }
    cited
}

/// `path` with `\` replaced by `/`.
fn slashed(path: &str) -> String {
    path.replace('\\', "/")
}

/// Whether `head` is `workspace` once a drive letter's case is ignored.
fn same_prefix(head: &str, workspace: &str) -> bool {
    if head == workspace {
        return true;
    }
    let is_drive = workspace.get(1..2) == Some(":");
    is_drive
        && head
            .get(..1)
            .zip(workspace.get(..1))
            .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
        && head.get(1..) == workspace.get(1..)
}

/// Normalizes a reported location to a workspace-relative path with `/` and `line = 0` to `None`.
///
/// Works on the text, not on `Path`, so the result is the same on every platform: `\` becomes `/`
/// in both arguments, the workspace is removed as a prefix only at a component boundary (`/w2/a.rs`
/// does not start with `/w`), the drive letter is compared ignoring case, and a leading `/` left
/// by that removal and any leading `./` are dropped. An absolute path outside the workspace stays
/// absolute. A workspace that is the filesystem root (`/`) holds every absolute path, so the
/// leading `/` alone is removed.
///
/// Complexity: O(len of `file` + len of `workspace`).
pub(crate) fn normalize_location(
    workspace: &Path,
    file: &str,
    line: Option<u32>,
) -> (String, Option<u32>) {
    let root = slashed(&workspace.to_string_lossy());
    let workspace = root.trim_end_matches('/');
    let mut path = slashed(file);
    if workspace.is_empty() && !root.is_empty() {
        // MAGI FIX (S6 loop 3): the workspace is the filesystem root (`/`): every absolute path
        // is under it, so its leading `/` is the removed prefix.
        path = path.trim_start_matches('/').to_string();
    } else if !workspace.is_empty() {
        let relative = path
            .get(..workspace.len())
            .filter(|head| same_prefix(head, workspace))
            .and_then(|_| path.get(workspace.len()..))
            .filter(|rest| rest.is_empty() || rest.starts_with('/'))
            .map(|rest| rest.trim_start_matches('/').to_string());
        if let Some(relative) = relative {
            path = relative;
        }
    }
    while let Some(rest) = path.strip_prefix("./") {
        path = rest.to_string();
    }
    (path, line.filter(|&l| l != 0))
}

/// The first cited symbol of `finding` that the index lacks, reading `title` then `detail`.
/// File names (`consts.rs`) are not symbols and are skipped.
fn first_missing_symbol(finding: &Finding, index: &LexicalIndex) -> Option<String> {
    cited_identifiers(&finding.title)
        .into_iter()
        .chain(cited_identifiers(&finding.detail))
        .find(|c| !index.is_file_name(c) && !index.contains(c))
}

/// The failed check of one finding, if any: file, then range, then symbol (the first to fail
/// names the cause). The symbol is set only for `UnknownSymbol`.
fn check(
    finding: &Finding,
    location: &(String, Option<u32>),
    reviewed: &Reviewed,
    index: &LexicalIndex,
) -> Option<(DiscardCause, Option<String>)> {
    finding.file.as_ref()?;
    let (file, line) = location;
    if !reviewed.has_file(file) {
        return Some((DiscardCause::FileNotReviewed, None));
    }
    if line.is_some_and(|l| !reviewed.contains_line(file, l)) {
        return Some((DiscardCause::LineOutOfRange, None));
    }
    if index.is_complete() && index.has_extractor(file) {
        return first_missing_symbol(finding, index)
            .map(|s| (DiscardCause::UnknownSymbol, Some(s)));
    }
    None
}

/// Removes from `agents` every finding the review cannot support and returns what was removed.
///
/// Runs per batch **after** every completed batch filled `reviewed` (U24: "reviewed" is the union
/// of all batches). Checks are independent and run in this order, the first failure naming the
/// cause: the file was shown, the line lies in a shown range, every symbol cited in backticks in
/// `title` and `detail` occurs in the code (only for files with a language extractor, and only
/// with a complete index). A finding without `file`, or with a blank one, skips the first two and
/// leaves with no `file`; one without `line`, the range check.
///
/// A reported `critical` whose only failure is the symbol check stays in `agents` for the gate
/// (U21) and is recorded in [`Validation::symbol_unverified`]; file and range failures discard it.
///
/// # Side effects
///
/// Discarded findings are removed from `agents[i].findings`. Every finding that stays leaves with
/// the `file`/`line` of `normalize_location`, the one point of normalization, so later stages
/// see the same path as `reviewed`. A discarded finding is returned as the mage reported it.
///
/// Complexity: O(f log r + c) for f findings, r ranges per file and c characters of title and
/// detail.
pub fn validate_batch(
    agents: &mut [AgentOutput],
    batch: usize,
    reviewed: &Reviewed,
    index: &LexicalIndex,
    workspace: &Path,
) -> Validation {
    let mut validation = Validation::default();
    for agent in agents.iter_mut() {
        let name = agent.agent;
        let mut kept = Vec::with_capacity(agent.findings.len());
        for mut finding in std::mem::take(&mut agent.findings) {
            // A blank file is no location, as consolidation and magi-core's key read it.
            if finding.file.as_deref().is_some_and(|f| f.trim().is_empty()) {
                finding.file = None;
            }
            let location = normalize_location(
                workspace,
                finding.file.as_deref().unwrap_or_default(),
                finding.line,
            );
            let verdict = check(&finding, &location, reviewed, index);
            let record = |cause, symbol| Discarded {
                agent: name,
                batch,
                finding: finding.clone(),
                cause,
                symbol,
            };
            match verdict {
                Some((DiscardCause::UnknownSymbol, symbol))
                    if finding.severity == Severity::Critical =>
                {
                    validation
                        .symbol_unverified
                        .push(record(DiscardCause::UnknownSymbol, symbol));
                }
                Some((cause, symbol)) => {
                    validation.discarded.push(record(cause, symbol));
                    continue;
                }
                None => {}
            }
            if finding.file.is_some() {
                finding.file = Some(location.0);
            }
            finding.line = location.1;
            kept.push(finding);
        }
        agent.findings = kept;
    }
    validation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::languages::LanguageRegistry;

    fn agent(findings: serde_json::Value) -> AgentOutput {
        crate::review::test_support::out("melchior", "approve", 0.9, findings)
    }

    #[test]
    fn findings_outside_what_was_reviewed_are_discarded_with_their_cause() {
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("src/a.rs", 10, 20);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"t1","detail":"d","file":"src/ghost.rs","line":3},
            {"severity":"warning","title":"t2","detail":"d","file":"src/a.rs","line":99},
            {"severity":"warning","title":"t3","detail":"d","file":"src/a.rs","line":15}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::FileNotReviewed, DiscardCause::LineOutOfRange]
        );
        assert_eq!(agents[0].findings.len(), 1);
        assert_eq!(agents[0].findings[0].title, "t3");
    }

    #[test]
    fn a_line_less_finding_on_an_unreviewed_file_is_discarded_and_one_without_a_file_is_kept() {
        // [CP2 MS2 S1 loop 2] the file check does not depend on the line.
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("src/a.rs", 1, 5);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"t1","detail":"d","file":"src/ghost.rs"},
            {"severity":"warning","title":"t2","detail":"d"}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::FileNotReviewed]
        );
        assert_eq!(
            agents[0]
                .findings
                .iter()
                .map(|f| f.title.as_str())
                .collect::<Vec<_>>(),
            vec!["t2"]
        );
    }

    #[test]
    fn a_blank_file_is_no_location_like_a_missing_one() {
        // Consolidation reads an empty `file` as unlocated (magi-core's key does too); the
        // validation must agree instead of discarding it as a file that was not reviewed.
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("src/a.rs", 1, 5);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"t1","detail":"d","file":""},
            {"severity":"warning","title":"t2","detail":"d","file":"  ","line":3}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert!(d.is_empty(), "{d:?}");
        assert!(
            agents[0].findings.iter().all(|f| f.file.is_none()),
            "{:?}",
            agents[0].findings
        );
    }

    #[test]
    fn a_file_reviewed_in_another_batch_counts() {
        // [CP2 MS2 S1 loop 3] the context-only half of RV-S29 moves to MS3 T3.6, where RAG context exists.
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("b.rs", 100);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"info","title":"b","detail":"d","file":"b.rs","line":1},
            {"severity":"info","title":"c","detail":"d","file":"c.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].cause, DiscardCause::FileNotReviewed); // c.rs was never reviewed
    }

    #[test]
    fn an_unknown_cited_symbol_is_discarded_only_where_an_extractor_exists() {
        let files = vec![("a.rs".to_string(), "fn real() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("a.rs", 1);
        reviewed.add_whole_file("conf.unknownlang", 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"`ghost` is unused","detail":"d","file":"a.rs","line":1},
            {"severity":"warning","title":"`real` leaks","detail":"d","file":"a.rs","line":1},
            {"severity":"warning","title":"`ghost` again","detail":"d","file":"conf.unknownlang","line":1},
            {"severity":"warning","title":"no code cited","detail":"d","file":"a.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].cause, DiscardCause::UnknownSymbol);
        assert_eq!(d[0].symbol.as_deref(), Some("ghost"));
        assert_eq!(agents[0].findings.len(), 3);
    }

    #[test]
    fn a_critical_whose_only_failure_is_an_unknown_symbol_goes_to_the_gate_marked() {
        // [CP2 MS2 S1b loop 13, user 2026-10-19] RV-S35: U21 reaches it; kept, recorded apart, never discarded.
        let files = vec![("a.rs".to_string(), "fn real() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("a.rs", 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"critical","title":"`ghost` corrupts state","detail":"d","file":"a.rs","line":1}
        ]))];
        let v = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"));
        assert!(v.discarded.is_empty());
        assert_eq!(agents[0].findings.len(), 1);
        assert_eq!(v.symbol_unverified.len(), 1);
        assert_eq!(
            (
                v.symbol_unverified[0].cause.clone(),
                v.symbol_unverified[0].symbol.as_deref()
            ),
            (DiscardCause::UnknownSymbol, Some("ghost"))
        );
    }

    #[test]
    fn a_critical_outside_what_was_reviewed_is_still_discarded() {
        // [CP2 MS2 S1b loop 13] the exemption covers the symbol check only: file and range are objective evidence.
        let files = vec![("a.rs".to_string(), "fn real() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("a.rs", 1, 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"critical","title":"`ghost` corrupts state","detail":"d","file":"a.rs","line":40},
            {"severity":"critical","title":"t","detail":"d","file":"ghost.rs","line":1}
        ]))];
        let v = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"));
        assert_eq!(
            v.discarded
                .iter()
                .map(|x| x.cause.clone())
                .collect::<Vec<_>>(),
            vec![DiscardCause::LineOutOfRange, DiscardCause::FileNotReviewed]
        );
        assert!(v.symbol_unverified.is_empty());
        assert!(agents[0].findings.is_empty());
    }

    #[test]
    fn cited_identifiers_only_take_identifier_shaped_code_spans() {
        assert_eq!(
            cited_identifiers("use `foo_bar`, `Baz::qux` and `self.helper`, not `a b` or `1x`"),
            vec![
                "foo_bar".to_string(),
                "Baz::qux".to_string(),
                "self.helper".to_string()
            ]
        );
    }

    #[test]
    fn a_non_ascii_citation_is_never_checked_so_it_cannot_be_discarded() {
        // The index splits non-ASCII identifiers into fragments; the citation rule skips them,
        // so a finding naming one is kept rather than discarded as an unknown symbol.
        assert!(cited_identifiers("calls `h\u{e9}llo` and `\u{4f60}\u{597d}`").is_empty());
    }

    #[test]
    fn a_double_backtick_citation_is_not_taken_as_an_identifier() {
        // [CP2 MS2 S1b loop 10] the loop-7 rule was declared but no test exercised it.
        assert_eq!(
            cited_identifiers("see ``ghost`` and `real`"),
            vec!["real".to_string()]
        );
    }

    #[test]
    fn single_backticks_nested_in_a_longer_span_are_not_citations() {
        // [MAGI MS2 S6 loop 2, C] a span closes only on a run of its own length, so what lies
        // inside a doubled or tripled span is literal text.
        assert!(cited_identifiers("`` a `b` c ``").is_empty());
        assert!(cited_identifiers("``` x `` y `z` ```").is_empty());
        assert_eq!(
            cited_identifiers("`` a `b` c `` then `d`"),
            vec!["d".to_string()]
        );
    }

    #[test]
    fn an_unclosed_longer_run_is_literal_and_later_singles_still_pair() {
        // No run of the same length closes it, so `` is plain text, as in CommonMark.
        assert_eq!(cited_identifiers("`` a `b` c"), vec!["b".to_string()]);
        assert_eq!(cited_identifiers("``` a `` `b`"), vec!["b".to_string()]);
    }

    #[test]
    fn a_single_span_holding_a_doubled_run_is_one_span() {
        // The doubled run inside does not close it, and its content is no identifier.
        assert!(cited_identifiers("`a ``b`` c` and ok").is_empty());
        assert_eq!(cited_identifiers("`a ``b`` c` `e`"), vec!["e".to_string()]);
    }

    #[test]
    fn a_qualified_citation_is_checked_by_its_last_segment_against_every_identifier_in_the_code() {
        // [CP2 MS2 S1 loop 1] the corpus is every identifier, so a called dependency method is known.
        let code = "fn real() { let v = Vec::new(); self.helper(v); }\nfn helper(_: Vec<u8>) {}\n";
        let index = LexicalIndex::build(
            &[("a.rs".to_string(), code.to_string())],
            &LanguageRegistry::new(),
        );
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("a.rs", 2);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"`Vec::new` allocates","detail":"d","file":"a.rs","line":1},
            {"severity":"warning","title":"`self.helper` leaks","detail":"d","file":"a.rs","line":1},
            {"severity":"warning","title":"`Baz::ghost` is wrong","detail":"d","file":"a.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.symbol.as_deref()).collect::<Vec<_>>(),
            vec![Some("Baz::ghost")]
        );
        assert_eq!(agents[0].findings.len(), 2);
    }

    #[test]
    fn a_symbol_only_on_the_old_side_of_a_diff_is_known() {
        // The change removed `old_fn`; a finding about the removal must not be discarded as a hallucination.
        let files = vec![
            ("a.rs".to_string(), "fn new_fn() {}\n".to_string()), // new side
            ("a.rs".to_string(), "fn old_fn() {}\n".to_string()),
        ]; // old side
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert!(index.contains("old_fn") && index.contains("new_fn"));
        assert!(!index.contains("ghost"));
    }

    #[test]
    fn discard_causes_serialize_with_the_spec_texts() {
        let got: Vec<String> = [
            DiscardCause::FileNotReviewed,
            DiscardCause::LineOutOfRange,
            DiscardCause::UnknownSymbol,
        ]
        .iter()
        .map(|c| serde_json::to_string(c).expect("serializes"))
        .collect();
        assert_eq!(
            got,
            vec![
                "\"file not reviewed\"",
                "\"line out of range\"",
                "\"unknown symbol\""
            ]
        );
    }

    #[test]
    fn an_absolute_path_and_line_zero_are_normalized() {
        let (f, l) = normalize_location(Path::new("/w"), "/w/src/a.rs", Some(0));
        assert_eq!((f.as_str(), l), ("src/a.rs", None));
        // [CP2 MS2 S1 loop 3] a leading `./` must not turn into a false `file not reviewed`.
        let (f, _) = normalize_location(Path::new("/w"), "./src/a.rs", Some(2));
        assert_eq!(f, "src/a.rs");
    }

    #[test]
    fn the_workspace_prefix_is_removed_only_at_a_component_boundary() {
        assert_eq!(
            normalize_location(Path::new("/w"), "/w2/src/a.rs", Some(1)).0,
            "/w2/src/a.rs"
        );
        assert_eq!(
            normalize_location(Path::new("/w"), "/etc/passwd", Some(1)).0,
            "/etc/passwd"
        );
    }

    #[test]
    fn a_filesystem_root_workspace_makes_every_absolute_path_relative() {
        // [MAGI §6 MS2 C loop 3] trimming `/` leaves an empty root: an absolute path under it is
        // still in the workspace and must reach the reviewed set's relative key.
        assert_eq!(
            normalize_location(Path::new("/"), "/src/a.rs", Some(1)),
            ("src/a.rs".to_string(), Some(1))
        );
        assert_eq!(
            normalize_location(Path::new("/"), "src/a.rs", Some(1)).0,
            "src/a.rs"
        );
        assert_eq!(
            normalize_location(Path::new(r"C:\"), r"C:\src\a.rs", None).0,
            "src/a.rs"
        );
    }

    #[test]
    fn the_symbol_check_reads_title_and_detail_and_names_the_first_unknown() {
        let files = vec![("a.rs".to_string(), "fn real() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("a.rs", 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"leak","detail":"see `ghost`","file":"a.rs","line":1},
            {"severity":"warning","title":"`real` and `g1`","detail":"also `g2`","file":"a.rs","line":1},
            {"severity":"warning","title":"`ghost` somewhere","detail":"d"},
            {"severity":"warning","title":"`ghost`","detail":"d","file":"other.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        let got: Vec<(DiscardCause, Option<&str>)> = d
            .iter()
            .map(|x| (x.cause.clone(), x.symbol.as_deref()))
            .collect();
        assert_eq!(
            got,
            vec![
                (DiscardCause::UnknownSymbol, Some("ghost")),
                (DiscardCause::UnknownSymbol, Some("g1")),
                (DiscardCause::FileNotReviewed, None)
            ]
        ); // file is checked before the symbol
        assert_eq!(
            agents[0]
                .findings
                .iter()
                .map(|f| f.title.as_str())
                .collect::<Vec<_>>(),
            vec!["`ghost` somewhere"]
        );
    }

    #[test]
    fn a_windows_absolute_path_is_normalized_on_every_platform() {
        let (f, l) = normalize_location(Path::new(r"C:\w"), r"c:\w\src\a.rs", Some(3));
        assert_eq!((f.as_str(), l), ("src/a.rs", Some(3)));
        let (f, _) = normalize_location(Path::new(r"C:\w"), r"src\b.rs", None);
        assert_eq!(f, "src/b.rs");
    }

    #[test]
    fn an_empty_hunk_side_adds_nothing_and_does_not_hide_the_other_side() {
        // [CP2 MS2 S1 loop 6] a deletion's new side is (1, 0): it must not register lines nor the file on its own.
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("only_empty.rs", 1, 0);
        reviewed.add_lines("gone.rs", 1, 0);
        reviewed.add_lines("gone.rs", 1, 30);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"info","title":"a","detail":"d","file":"gone.rs","line":5},
            {"severity":"info","title":"b","detail":"d","file":"only_empty.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::FileNotReviewed]
        );
        assert_eq!(agents[0].findings.len(), 1);
    }

    #[test]
    fn a_kept_finding_leaves_with_its_normalized_location() {
        // [CP2 MS2 S1b loop 9] the single normalization point: T2.7's dedup and the gate's window read these fields.
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("src/a.rs", 10, 20);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"abs","detail":"d","file":"/w/src/a.rs","line":12},
            {"severity":"warning","title":"dot","detail":"d","file":"./src/a.rs","line":0}
        ]))];
        assert!(
            validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"))
                .discarded
                .is_empty()
        );
        let got: Vec<(Option<&str>, Option<u32>)> = agents[0]
            .findings
            .iter()
            .map(|f| (f.file.as_deref(), f.line))
            .collect();
        assert_eq!(
            got,
            vec![(Some("src/a.rs"), Some(12)), (Some("src/a.rs"), None)]
        );
    }

    #[test]
    fn a_backticked_file_name_is_not_checked_as_a_symbol() {
        // [CP2 MS2 S1b loop 11] `consts.rs` would be looked up as the symbol `rs`; an invented member still is checked.
        let index = LexicalIndex::build(
            &[
                (
                    "src/consts.rs".to_string(),
                    "pub const A: u8 = 1;\n".to_string(),
                ),
                ("conf.toml".to_string(), "a = 1\n".to_string()),
            ],
            &LanguageRegistry::new(),
        );
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("src/consts.rs", 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"critical","title":"`consts.rs` and `conf.toml` disagree","detail":"see `mod.rs`","file":"src/consts.rs","line":1},
            {"severity":"info","title":"`self.ghost` is unused","detail":"d","file":"src/consts.rs","line":1}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::UnknownSymbol]
        );
        assert_eq!(agents[0].findings.len(), 1);
        assert!(
            index.is_file_name("consts.rs")
                && index.is_file_name("mod.rs")
                && !index.is_file_name("self.ghost")
        );
        assert!(
            !index.is_file_name("Foo::bar.rs"),
            "joined by `::`: a symbol path, never a file name (S1b loop 13)"
        );
    }

    #[test]
    fn a_cited_file_whose_extension_nothing_reviewed_has_is_checked_as_a_symbol() {
        // [CP2 MS2 S1b loop 13] the declared reverse case, narrowed in S1b2 loop 16: a known extension (`conf.toml`) is a
        // file name even with no `.toml` reviewed; only an extension outside every list (`notes.zzz`) is looked up as `zzz`.
        let index = LexicalIndex::build(
            &[("src/a.rs".to_string(), "fn a() {}\n".to_string())],
            &LanguageRegistry::new(),
        );
        assert!(index.is_file_name("conf.toml"));
        assert!(!index.is_file_name("notes.zzz"));
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("src/a.rs", 1);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"`conf.toml` is stale","detail":"d","file":"src/a.rs","line":1},
            {"severity":"warning","title":"`notes.zzz` is stale","detail":"d","file":"src/a.rs","line":1}]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::UnknownSymbol]
        );
        assert_eq!(agents[0].findings.len(), 1, "the `conf.toml` finding stays");
    }

    #[test]
    fn a_file_registered_by_units_leaves_the_gaps_between_them_unreviewed() {
        // [CP2 MS2 S1b loop 11] unit level: only add_whole_file covers a whole file; T2.9 decides when to call it.
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("a.rs", 1, 10);
        reviewed.add_lines("a.rs", 20, 30);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"gap","detail":"d","file":"a.rs","line":15},
            {"severity":"warning","title":"unit","detail":"d","file":"a.rs","line":25}
        ]))];
        let d = validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w")).discarded;
        assert_eq!(
            d.iter().map(|x| x.cause.clone()).collect::<Vec<_>>(),
            vec![DiscardCause::LineOutOfRange]
        );
        assert_eq!(agents[0].findings.len(), 1);
    }

    #[test]
    fn an_index_over_its_cap_turns_the_symbol_check_off_instead_of_discarding() {
        // [CP2 MS2 S1b loop 12] an incomplete index cannot say a symbol is absent, so it discards nothing.
        let files = vec![(
            "a.rs".to_string(),
            "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n".to_string(),
        )];
        let index = LexicalIndex::build_with_cap(&files, &LanguageRegistry::new(), 2);
        assert!(!index.is_complete());
        assert!(LexicalIndex::build(&files, &LanguageRegistry::new()).is_complete());
        let mut reviewed = Reviewed::default();
        reviewed.add_whole_file("a.rs", 3);
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"`ghost` is wrong","detail":"d","file":"a.rs","line":1}]))];
        assert!(
            validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"))
                .discarded
                .is_empty()
        );
        assert_eq!(agents[0].findings.len(), 1);
    }

    #[test]
    fn a_deleted_file_counts_as_reviewed_for_its_removed_lines() {
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("gone.rs", 1, 30); // T2.9 registers the old-side lines of a deletion
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(
            serde_json::json!([{"severity":"info","title":"t","detail":"d","file":"gone.rs","line":5}]),
        )];
        assert!(
            validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"))
                .discarded
                .is_empty()
        );
    }

    #[test]
    fn a_kept_finding_whose_line_is_zero_leaves_without_a_line() {
        // A finding built in memory can carry line 0 (deserialization already maps it to None).
        let mut reviewed = Reviewed::default();
        reviewed.add_lines("src/a.rs", 1, 5);
        let index = LexicalIndex::build(&[], &LanguageRegistry::new());
        let mut agents = vec![agent(serde_json::json!([
            {"severity":"warning","title":"t","detail":"d","file":"src/a.rs","line":3}
        ]))];
        agents[0].findings[0].line = Some(0);
        assert!(
            validate_batch(&mut agents, 0, &reviewed, &index, Path::new("/w"))
                .discarded
                .is_empty()
        );
        assert_eq!(agents[0].findings[0].line, None);
    }
}
