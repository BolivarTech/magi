// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Review context assembled by a fixed priority (REQ-RV07, RV03).

use super::batching::BatchItem;
use super::error::ReviewError;
use super::index::LexicalIndex;
use super::languages::LanguageRegistry;
use super::source::ContentSource;

/// What a context item is. The declaration order is the priority order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextKind {
    /// The unit that contains a change.
    Unit,
    /// The definition of a name the batch uses.
    Definition,
    /// A unit that names a changed unit.
    Caller,
    /// A chunk of project documentation.
    Documentation,
    /// A validated historical finding.
    History,
}

/// One piece of retrieved context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextItem {
    /// What the item is.
    pub kind: ContextKind,
    /// The file it comes from.
    pub file: String,
    /// A short name for the item.
    pub label: String,
    /// The text sent to the trio.
    pub text: String,
    /// Whether the item is one of several homonyms.
    pub candidate: bool,
    /// The lines of `file` the text covers, when it is a span of a file.
    pub lines: Option<(u32, u32)>,
}

/// The context that fits a batch, and what was left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembled {
    /// The items that entered, in priority order.
    pub items: Vec<ContextItem>,
    /// The items left out, in priority order.
    pub trimmed: Vec<(ContextKind, String)>,
    /// Whether documentation and history were skipped for lack of an embedder.
    pub without_embedder: bool,
    /// Whether `--no-memory` left out the stores.
    pub without_memory: bool,
    /// History entries and doc chunks the secret detector left out.
    pub withheld: usize,
}

/// Callers kept per name.
pub const MAX_CALLERS_PER_NAME: usize = 10;

/// What indexing left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexDeclarations {
    /// Chunks the secret detector left out.
    pub withheld: usize,
    /// Whether indexing stopped at its cap.
    pub truncated: bool,
}

/// Fixed tokens of one item's frame in the prompt.
pub const CONTEXT_ITEM_FRAME_TOKENS: usize = 18;

/// Longest doc-search query, in bytes.
pub const DOC_QUERY_MAX_BYTES: usize = 8_192;

/// What `assemble` charges for one item.
pub fn item_cost(item: &ContextItem) -> usize {
    // Red stub: removed in Green
    let _ = item;
    0
}

/// Adds the units, definitions, documentation and history that fit.
pub fn assemble(
    batch_tokens: usize,
    limit_tokens: usize,
    units: Vec<ContextItem>,
    definitions: Vec<ContextItem>,
    docs: Option<Vec<ContextItem>>,
    history: Option<Vec<ContextItem>>,
) -> Result<Assembled, ReviewError> {
    // Red stub: removed in Green
    let _ = (
        batch_tokens,
        limit_tokens,
        units,
        definitions,
        docs,
        history,
    );
    Ok(Assembled {
        items: Vec::new(),
        trimmed: Vec::new(),
        without_embedder: false,
        without_memory: false,
        withheld: 0,
    })
}

/// Adds what indexing left out to `a`.
pub fn declare_index(a: &mut Assembled, d: &IndexDeclarations) {
    // Red stub: removed in Green
    let _ = (a, d);
}

/// The definitions of the names the batch code uses.
pub fn definitions_for(
    text: &str,
    index: &LexicalIndex,
    sources: &dyn ContentSource,
) -> Result<Vec<ContextItem>, ReviewError> {
    // Red stub: removed in Green
    let _ = (text, index, sources);
    Ok(Vec::new())
}

/// The units that name any of `names`.
pub fn callers_for(
    names: &[&str],
    index: &LexicalIndex,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
) -> Result<(Vec<ContextItem>, Vec<String>), ReviewError> {
    // Red stub: removed in Green
    let _ = (names, index, sources, registry);
    Ok((Vec::new(), Vec::new()))
}

/// The containing unit of a batch item.
pub fn containing_unit(
    item: &BatchItem,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
) -> Result<Option<ContextItem>, ReviewError> {
    // Red stub: removed in Green
    let _ = (item, sources, registry);
    Ok(None)
}

/// The doc-search query cut from `text`.
pub fn doc_query(text: &str) -> &str {
    // Red stub: removed in Green
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::secrets::display_path;
    use crate::review::window::estimate_tokens;

    fn item(kind: ContextKind, label: &str, tokens: usize) -> ContextItem {
        // [CP2 MS3 loop 11] `tokens` is the item's whole cost: its frame, its path ("a.rs" = 1 token)
        // and its text.
        let text_tokens = tokens
            .checked_sub(CONTEXT_ITEM_FRAME_TOKENS + 1)
            .expect("tokens covers the frame");
        ContextItem {
            kind,
            file: "a.rs".into(),
            label: label.into(),
            text: "x".repeat(text_tokens * 4),
            candidate: false,
            lines: None,
        }
    }

    #[test]
    fn the_frame_of_each_item_counts_in_its_cost() {
        // [CP2 MS3 loop 11] The prompt carries the delimiters and the path too: the budget counts
        // what is sent.
        let def = || ContextItem {
            kind: ContextKind::Definition,
            file: "a.rs".into(),
            label: "d".into(),
            text: "x".repeat(40),
            candidate: false,
            lines: None,
        };
        let cost = 10 + 1 + CONTEXT_ITEM_FRAME_TOKENS; // text, path, frame [CP2 MS3 loop 12: symbolic]
        assert_eq!(item_cost(&def()), cost);
        let fits = assemble(
            100 - cost,
            100,
            vec![],
            vec![def()],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("fits");
        assert_eq!(fits.items.len(), 1, "exactly the limit");
        let over = assemble(
            100 - cost + 1,
            100,
            vec![],
            vec![def()],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("ok");
        assert!(
            over.items.is_empty() && over.trimmed.len() == 1,
            "one token over: trimmed"
        );
    }

    #[test]
    fn a_unit_shared_by_two_hunks_enters_once() {
        // [CP2 MS3 loop 11] Two hunks of one function share their unit: charged once, never a false
        // TooLargeForBatch.
        let unit = || {
            let mut u = item(ContextKind::Unit, "f", 30);
            u.lines = Some((3, 9));
            u
        };
        let a = assemble(60, 100, vec![unit(), unit()], vec![], None, None).expect("one copy fits");
        assert_eq!(a.items.len(), 1);
    }

    #[test]
    fn a_definition_with_the_lines_of_a_unit_is_not_repeated() {
        // [CP2 MS3 loop 12] Dedup by (file, lines) across kinds, units first: never charged twice,
        // never trimmed.
        let unit = {
            let mut u = item(ContextKind::Unit, "f", 30);
            u.lines = Some((3, 9));
            u
        };
        let same = {
            let mut d = item(ContextKind::Definition, "f", 30);
            d.lines = Some((3, 9));
            d
        };
        let a =
            assemble(60, 100, vec![unit], vec![same], Some(vec![]), Some(vec![])).expect("fits");
        assert_eq!(
            a.items.iter().map(|i| i.kind.clone()).collect::<Vec<_>>(),
            vec![ContextKind::Unit]
        );
        assert!(a.trimmed.is_empty(), "{:?}", a.trimmed);
    }

    #[test]
    fn items_enter_by_priority_and_the_tail_is_trimmed() {
        let a = assemble(
            10,
            100,
            vec![item(ContextKind::Unit, "u", 20)],
            vec![item(ContextKind::Definition, "d", 30)],
            Some(vec![item(ContextKind::Documentation, "doc", 30)]),
            Some(vec![item(ContextKind::History, "h", 30)]),
        )
        .expect("fits");
        assert_eq!(
            a.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["u", "d", "doc"]
        );
        assert_eq!(a.trimmed, vec![(ContextKind::History, "h".to_string())]);
        assert!(
            !a.without_embedder,
            "[CP2 MS3 loop 12] Some docs and history: the embedder is there"
        );
    }

    #[test]
    fn with_no_room_left_only_the_containing_unit_is_added() {
        let a = assemble(
            80,
            100,
            vec![item(ContextKind::Unit, "u", 20)],
            vec![item(
                ContextKind::Definition,
                "d",
                CONTEXT_ITEM_FRAME_TOKENS + 1,
            )],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("unit fits exactly");
        assert_eq!(a.items.len(), 1);
        assert_eq!(a.trimmed.len(), 1);
    }

    #[test]
    fn a_containing_unit_that_does_not_fit_is_an_error_never_a_trim() {
        assert!(matches!(
            assemble(
                90,
                100,
                vec![item(ContextKind::Unit, "u", 20)],
                vec![],
                None,
                None
            ),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_second_containing_unit_that_does_not_fit_is_an_error_too() {
        // [CP2 MS3 loop 6] Units are mandatory one by one: never the first kept and the second
        // dropped.
        let units = vec![
            item(ContextKind::Unit, "u1", 20),
            item(ContextKind::Unit, "u2", 40),
        ];
        assert!(matches!(
            assemble(50, 100, units, vec![], None, None),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_batch_over_the_limit_is_an_error_never_an_empty_context() {
        // [CP2 MS3 loop 6, 12] Planning never hands one over (plan_batches_with); a caller that did
        // is a visible defect.
        assert!(matches!(
            assemble(101, 100, vec![], vec![], None, None),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_source_read_error_while_gathering_context_is_returned_never_skipped() {
        // [CP2 MS3 loop 6] No silent failure: a definition or caller that cannot be read fails the
        // preparation, typed.
        struct Broken;
        impl ContentSource for Broken {
            fn read(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io("disk".into()))
            }
            fn read_old(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ("a.rs".to_string(), "fn parse() {}\n".to_string()),
            (
                "b.rs".to_string(),
                "fn use_it() {\n    parse();\n}\n".to_string(),
            ),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert!(matches!(
            definitions_for("parse();", &index, &Broken),
            Err(ReviewError::Io(_))
        ));
        assert!(matches!(
            callers_for(&["parse"], &index, &Broken, &LanguageRegistry::new()),
            Err(ReviewError::Io(_))
        ));
    }

    #[test]
    fn the_containing_unit_of_a_hunk_is_found_on_its_side_and_priced_as_assemble_charges_it() {
        // [CP2 MS3 loop 12, option 1] One definition for planning's companion and for the context:
        // what plan_batches_with charged is what assemble adds.
        use crate::review::test_support::MapSource;
        let new = "fn f() {\n    let x = 1;\n}\n";
        let old = "fn gone() {\n    y();\n}\n";
        let src = MapSource::new(&[("a.rs", new)], &[("a.rs", old)]);
        let reg = LanguageRegistry::new();
        let hunk = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "+    let x = 1;".into(),
            first_line: 2,
            last_line: 2,
            old_first_line: 2,
            old_last_line: 2,
        };
        let u = containing_unit(&hunk, &src, &reg)
            .expect("reads")
            .expect("inside f");
        assert_eq!(
            (u.kind.clone(), u.label.as_str(), u.lines),
            (ContextKind::Unit, "f", Some((1, 3)))
        );
        assert_eq!(
            item_cost(&u),
            estimate_tokens(&u.text)
                + estimate_tokens(&display_path(&u.file))
                + CONTEXT_ITEM_FRAME_TOKENS
        );
        let deleted = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "-    y();".into(),
            first_line: 2,
            last_line: 1,
            old_first_line: 2,
            old_last_line: 2,
        };
        assert_eq!(
            containing_unit(&deleted, &src, &reg)
                .expect("reads")
                .expect("old side")
                .label,
            "gone"
        );
        let whole = BatchItem::WholeFile {
            file: "a.rs".into(),
            text: new.into(),
            lines: 3,
        };
        assert!(
            containing_unit(&whole, &src, &reg)
                .expect("reads")
                .is_none(),
            "a whole file is its own unit"
        );
    }

    #[test]
    fn the_doc_query_is_cut_on_a_char_boundary() {
        // [CP2 MS3 loop 4] A byte cut inside a multi-byte char would panic (§0: no slicing).
        assert_eq!(doc_query("abc"), "abc");
        assert_eq!(
            doc_query(&"é".repeat(DOC_QUERY_MAX_BYTES)).len(),
            DOC_QUERY_MAX_BYTES
        );
        assert_eq!(
            doc_query(&format!("a{}", "é".repeat(DOC_QUERY_MAX_BYTES))).len(),
            DOC_QUERY_MAX_BYTES - 1
        );
    }

    #[test]
    fn without_an_embedder_docs_and_history_are_skipped_and_declared() {
        let a = assemble(0, 100, vec![], vec![], None, None).expect("ok");
        assert!(a.without_embedder);
    }

    #[test]
    fn homonym_definitions_are_marked_candidates() {
        use crate::review::test_support::MapSource;
        let files = vec![
            ("a.rs".to_string(), "fn parse() {}\n".to_string()),
            ("b.rs".to_string(), "fn parse() {}\n".to_string()),
            ("c.rs".to_string(), "fn only() {}\n".to_string()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(
            &[
                ("a.rs", "fn parse() {}\n"),
                ("b.rs", "fn parse() {}\n"),
                ("c.rs", "fn only() {}\n"),
            ],
            &[],
        );
        let defs = definitions_for("call `parse` and `only` here", &index, &src).expect("reads");
        let parse: Vec<_> = defs.iter().filter(|d| d.label == "parse").collect();
        assert_eq!(parse.len(), 2);
        assert!(parse
            .iter()
            .all(|d| d.candidate && d.kind == ContextKind::Definition));
        let only: Vec<_> = defs.iter().filter(|d| d.label == "only").collect();
        assert_eq!(only.len(), 1);
        assert!(!only[0].candidate);
    }

    #[test]
    fn definitions_are_found_from_raw_batch_code_without_backticks() {
        // [CP2 MS3 loop 2] The input is the batch's raw code: every identifier counts, quoted or
        // not.
        use crate::review::test_support::MapSource;
        let files = vec![("a.rs".to_string(), "fn parse() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", "fn parse() {}\n")], &[]);
        let defs = definitions_for(
            "fn caller() {\n    parse();\n    parse();\n}\n",
            &index,
            &src,
        )
        .expect("reads");
        assert_eq!(
            defs.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["parse"],
            "[CP2 MS3 loop 12] one per definition"
        );
    }

    #[test]
    fn a_referenced_type_reaches_the_definitions() {
        // [CP2 MS3 loop 10] RV07(2) names referenced types too: a `struct` is a top-level unit with
        // a name.
        use crate::review::test_support::MapSource;
        let files = vec![(
            "a.rs".to_string(),
            "struct Config {\n    n: u32,\n}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", "struct Config {\n    n: u32,\n}\n")], &[]);
        let defs =
            definitions_for("fn f(c: Config) -> u32 {\n    c.n\n}\n", &index, &src).expect("reads");
        assert_eq!(
            defs.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["Config"]
        );
    }

    #[test]
    fn index_drops_reach_the_assembled_declarations() {
        // [CP2 MS3 loop 2] Secret-dropped doc chunks and a MAX_DOC_CHUNKS cut are declared, never
        // silent.
        let mut a = assemble(0, 100, vec![], vec![], Some(vec![]), Some(vec![])).expect("ok");
        declare_index(
            &mut a,
            &IndexDeclarations {
                withheld: 2,
                truncated: true,
            },
        );
        assert_eq!(a.withheld, 2);
        assert!(
            a.trimmed
                .iter()
                .any(|(k, l)| *k == ContextKind::Documentation && l.contains("MAX_DOC_CHUNKS")),
            "{:?}",
            a.trimmed
        );
    }

    #[test]
    fn the_units_that_name_a_changed_unit_are_its_callers() {
        // [CP2 MS3 loop 1] RV07(2).
        use crate::review::test_support::MapSource;
        let a = "fn parse() {}\n";
        let b = "fn use_it() {\n    parse();\n}\n\nfn unrelated() {}\n";
        let files = vec![
            ("a.rs".to_string(), a.to_string()),
            ("b.rs".to_string(), b.to_string()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a), ("b.rs", b)], &[]);
        let (callers, capped) =
            callers_for(&["parse"], &index, &src, &LanguageRegistry::new()).expect("reads");
        assert_eq!(
            callers
                .iter()
                .map(|c| (c.kind.clone(), c.file.as_str(), c.label.as_str()))
                .collect::<Vec<_>>(),
            vec![(ContextKind::Caller, "b.rs", "use_it")],
            "the definition in a.rs is not its own caller"
        );
        assert!(capped.is_empty());
    }

    #[test]
    fn callers_beyond_the_cap_are_named_never_dropped_silently() {
        // [CP2 MS3 loop 8] §0.2: a name with more callers than the cap comes back so StoreRag
        // declares it.
        use crate::review::test_support::MapSource;
        let a = "fn parse() {}\n".to_string();
        let b: String = (0..=MAX_CALLERS_PER_NAME)
            .map(|i| format!("fn c{i}() {{\n    parse();\n}}\n\n"))
            .collect();
        let files = vec![
            ("a.rs".to_string(), a.clone()),
            ("b.rs".to_string(), b.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a.as_str()), ("b.rs", b.as_str())], &[]);
        let (callers, capped) =
            callers_for(&["parse"], &index, &src, &LanguageRegistry::new()).expect("reads");
        assert_eq!(callers.len(), MAX_CALLERS_PER_NAME);
        assert_eq!(capped, vec!["parse".to_string()]);
    }

    #[test]
    fn the_kept_context_is_a_prefix_and_everything_after_the_first_misfit_is_trimmed() {
        // [CP2 MS3 loop 8] RV07 «se recorta desde el final»: a smaller item after a misfit is never
        // kept.
        let docs = vec![
            item(ContextKind::Documentation, "big", 95),
            item(
                ContextKind::Documentation,
                "small",
                CONTEXT_ITEM_FRAME_TOKENS + 1,
            ),
        ];
        let a = assemble(10, 100, vec![], vec![], Some(docs), Some(vec![])).expect("fits");
        assert!(
            a.items.is_empty(),
            "{:?}",
            a.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.trimmed,
            vec![
                (ContextKind::Documentation, "big".to_string()),
                (ContextKind::Documentation, "small".to_string())
            ]
        );
    }
}
