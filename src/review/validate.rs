// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Deterministic validation of the findings a batch returned (REQ-RV09).

use std::path::Path;

use magi_core::schema::{AgentName, AgentOutput, Finding};

use super::index::{LexicalIndex, Reviewed};

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

/// Identifier-shaped spans between single backticks.
pub fn cited_identifiers(text: &str) -> Vec<String> {
    todo!("{text}")
}

/// Normalizes a reported location to a workspace-relative path with `/`.
pub fn normalize_location(
    workspace: &Path,
    file: &str,
    line: Option<u32>,
) -> (String, Option<u32>) {
    todo!("{} {file} {line:?}", workspace.display())
}

/// Removes from `agents` every finding the review cannot support.
pub fn validate_batch(
    agents: &mut [AgentOutput],
    batch: usize,
    reviewed: &Reviewed,
    index: &LexicalIndex,
    workspace: &Path,
) -> Validation {
    let _ = (reviewed, index);
    todo!("{} {batch} {}", agents.len(), workspace.display())
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
    fn a_double_backtick_citation_is_not_taken_as_an_identifier() {
        // [CP2 MS2 S1b loop 10] the loop-7 rule was declared but no test exercised it.
        assert_eq!(
            cited_identifiers("see ``ghost`` and `real`"),
            vec!["real".to_string()]
        );
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
}
