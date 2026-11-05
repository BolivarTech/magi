// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Project documentation as chunks for the `code-review` context (Red stubs).

use crate::review::error::ReviewError;
use crate::review::languages::LanguageRegistry;
use crate::review::source::GitCommand;

/// One documentation chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocChunk {
    pub path: String,
    pub section: String,
    pub text: String,
    pub hash: String,
}

/// One documentation source of a revision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocSource {
    pub path: String,
    pub key: String,
}

pub const DOC_SOURCES: &[&str] = &["README.md", "CLAUDE.md", "docs/", "adr/"];
pub const MAX_DOC_CHUNK_BYTES: usize = 8_192;
pub const MAX_DOC_CHUNKS: usize = 20_000;
pub const DOC_INDEX_VERSION: u32 = 1;
pub const MAX_GC_REFS: usize = 64;

pub fn is_doc_source(_path: &str) -> bool {
    false
}

pub fn chunk_markdown(_path: &str, _text: &str) -> Vec<DocChunk> {
    Vec::new()
}

pub fn chunk_doc_comments(_path: &str, _text: &str, _registry: &LanguageRegistry) -> Vec<DocChunk> {
    Vec::new()
}

pub fn chunk_source(_path: &str, _text: &str, _registry: &LanguageRegistry) -> Vec<DocChunk> {
    Vec::new()
}

pub fn without_secrets(_chunks: Vec<DocChunk>) -> (Vec<DocChunk>, Vec<String>) {
    (Vec::new(), Vec::new())
}

pub fn bounded(
    _sources: Vec<(DocSource, Vec<DocChunk>)>,
) -> (Vec<(DocSource, Vec<DocChunk>)>, bool) {
    (Vec::new(), false)
}

pub fn worktree_key(_bytes: &[u8]) -> String {
    String::new()
}

pub fn doc_index_id(_model_id: &str, _dim: usize) -> String {
    String::new()
}

pub fn parse_ls_tree(
    _out: &[u8],
    _keep: &dyn Fn(&str) -> bool,
) -> Result<Vec<DocSource>, ReviewError> {
    Ok(Vec::new())
}

pub fn revision_doc_sources(
    _git: &dyn GitCommand,
    _rev: &str,
    _keep: &dyn Fn(&str) -> bool,
) -> Result<Vec<DocSource>, ReviewError> {
    Ok(Vec::new())
}

pub fn live_doc_sources(
    _git: &dyn GitCommand,
    _keep: &dyn Fn(&str) -> bool,
) -> Result<Option<std::collections::BTreeSet<DocSource>>, ReviewError> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_is_chunked_by_heading_with_its_section_path() {
        let c = chunk_markdown("README.md", "# A\nintro\n## B\nbody b\n## C\nbody c\n");
        assert_eq!(
            c.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["A", "A > B", "A > C"]
        );
        assert!(c[1].text.contains("body b"));
    }

    #[test]
    fn a_heading_inside_a_code_fence_does_not_start_a_section() {
        // [CP2 MS3 loop 2] A shell comment in a fenced block is not an ATX heading.
        let c = chunk_markdown("README.md", "# Usage\n```sh\n# install\nmake\n```\nmore\n");
        assert_eq!(c.len(), 1);
        assert!(c[0].text.contains("# install"));
        // [CP2 MS3 loop 5] Tilde fences too.
        let t = chunk_markdown("README.md", "# Usage\n~~~\n# install\n~~~\nmore\n");
        assert_eq!(t.len(), 1, "{t:?}");
        // [CP2 MS3 loop 7] An unclosed fence runs to the end of the file (declared).
        let u = chunk_markdown("README.md", "# A\n```\n# B\n");
        assert_eq!(u.len(), 1, "{u:?}");
    }

    #[test]
    fn a_markdown_file_without_headings_is_one_chunk() {
        let c = chunk_markdown("README.md", "just text\nmore\n");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].section, "");
        assert!(
            chunk_markdown("README.md", "").is_empty(),
            "[CP2 MS3 loop 13] an empty file has nothing to index"
        );
    }

    #[test]
    fn the_hash_changes_with_the_text_and_not_otherwise() {
        let a = chunk_markdown("d.md", "# A\nx\n");
        let b = chunk_markdown("d.md", "# A\nx\n");
        let c = chunk_markdown("d.md", "# A\ny\n");
        assert_eq!(a[0].hash, b[0].hash);
        assert_ne!(a[0].hash, c[0].hash);
        assert_eq!(a[0].hash.len(), 64);
        // [CP2 MS3 loop 14 (pasada final)] Each hash input on its own: the path and the section count too. The hash has no
        // revision input, so the same section in two revisions (a and b above, read twice) keeps its hash: RV-S37's reuse.
        assert_ne!(
            a[0].hash,
            chunk_markdown("e.md", "# A\nx\n")[0].hash,
            "path"
        );
        assert_ne!(
            a[0].hash,
            chunk_markdown("d.md", "# B\nx\n")[0].hash,
            "section"
        );
    }

    #[test]
    fn a_long_section_is_split_into_bounded_parts() {
        // [CP2 MS3 loop 14 (pasada final)] One request never exceeds an embedder's input: parts at line boundaries.
        let body = "x\n".repeat(MAX_DOC_CHUNK_BYTES); // twice the bound
        let c = chunk_markdown("README.md", &format!("# A\n{body}"));
        assert!(c.len() >= 2, "{}", c.len());
        assert!(
            c.iter().all(|x| x.text.len() <= MAX_DOC_CHUNK_BYTES),
            "every part within the bound"
        );
        assert_eq!(c[0].section, "A (1)");
        assert_eq!(c[1].section, "A (2)");
    }

    #[test]
    fn rust_doc_comments_become_chunks_named_after_their_item() {
        let src =
            "/// Adds two numbers.\n/// Panics never.\nfn add(a: u32, b: u32) -> u32 { a + b }\n";
        let c = chunk_doc_comments("src/m.rs", src, &LanguageRegistry::new());
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].section, "add");
        assert!(c[0].text.contains("Adds two numbers."));
        // [CP2 MS3 loop 13] Every item keyword names its section, not only `fn`.
        let s = chunk_doc_comments(
            "src/m.rs",
            "/// A config.\npub struct Config { x: u8 }\n",
            &LanguageRegistry::new(),
        );
        assert_eq!(
            s.iter().map(|c| c.section.as_str()).collect::<Vec<_>>(),
            vec!["Config"]
        );
    }

    #[test]
    fn every_item_shape_names_its_section() {
        // [CP2 MS3 loop 14 (pasada final)] The precise naming rule, Rust and Python.
        let reg = LanguageRegistry::new();
        let name = |p: &str, src: &str| {
            chunk_doc_comments(p, src, &reg)
                .iter()
                .map(|c| c.section.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            name("src/m.rs", "/// D.\n#[derive(Debug)]\npub struct Config;\n"),
            vec!["Config"],
            "attribute skipped"
        );
        assert_eq!(
            name("src/m.rs", "/// S.\npub const fn size() -> usize { 1 }\n"),
            vec!["size"],
            "const fn"
        );
        assert_eq!(
            name("src/m.rs", "/// I.\nimpl Display for Config {}\n"),
            vec!["Config"],
            "impl"
        );
        assert_eq!(
            name("src/m.rs", "/// M.\nmacro_rules! twice { () => {} }\n"),
            vec!["twice"],
            "macro_rules!"
        );
        assert_eq!(
            name(
                "m.py",
                "def f():\n    \"\"\"Explains f.\"\"\"\n    return 1\n"
            ),
            vec!["f"],
            "def"
        );
        assert_eq!(
            name("m.py", "@dataclass\nclass C:\n    \"\"\"A C.\"\"\"\n"),
            vec!["C"],
            "decorated class"
        );
        assert_eq!(
            name("m.py", "\"\"\"Module docs.\"\"\"\nx = 1\n"),
            vec![""],
            "module docstring"
        );
        assert!(
            name("m.py", "def f():\n    x = 1\n    \"\"\"not docs\"\"\"\n").is_empty(),
            "a later string is data"
        );
    }

    #[test]
    fn plain_comments_and_string_data_are_not_documentation() {
        // [CP2 MS3 loop 13] The false-positive side of the doc-comment rules.
        let reg = LanguageRegistry::new();
        assert!(chunk_doc_comments("src/m.rs", "// just a note\nfn a() {}\n", &reg).is_empty());
        assert!(chunk_doc_comments("m.py", "PATTERN = \"\"\"not docs\"\"\"\n", &reg).is_empty());
        assert!(
            chunk_doc_comments("m.py", "# a comment\ndef f():\n    return 1\n", &reg).is_empty()
        );
    }

    #[test]
    fn inner_rust_and_python_doc_comments_become_chunks_too() {
        // [CP2 MS3 loop 4] The three documented forms, not only `///`.
        let r = chunk_doc_comments(
            "src/lib.rs",
            "//! Crate overview.\nfn a() {}\n",
            &LanguageRegistry::new(),
        );
        assert!(
            r.iter().any(|c| c.text.contains("Crate overview.")),
            "{r:?}"
        );
        let p = chunk_doc_comments(
            "m.py",
            "def f():\n    \"\"\"Explains f.\"\"\"\n    return 1\n",
            &LanguageRegistry::new(),
        );
        assert!(p.iter().any(|c| c.text.contains("Explains f.")), "{p:?}");
        // [CP2 MS3 loop 5] Block doc comments are rustdoc too.
        let b = chunk_doc_comments(
            "src/lib.rs",
            "/** Block docs. */\nfn b() {}\n",
            &LanguageRegistry::new(),
        );
        assert!(b.iter().any(|c| c.text.contains("Block docs.")), "{b:?}");
        // [CP2 MS3 loop 7] Python's other docstring quote.
        let q = chunk_doc_comments(
            "m.py",
            "def g():\n    '''Explains g.'''\n    return 1\n",
            &LanguageRegistry::new(),
        );
        assert!(q.iter().any(|c| c.text.contains("Explains g.")), "{q:?}");
    }

    #[test]
    fn each_source_file_is_chunked_by_its_kind() {
        // [CP2 MS3 loop 7] RV07(3): a source file gives its doc comments, never its code.
        let reg = LanguageRegistry::new();
        let rs = chunk_source(
            "src/m.rs",
            "/// Adds.\nfn add() { let body_marker = 1; }\n",
            &reg,
        );
        assert_eq!(rs.len(), 1, "{rs:?}");
        assert!(
            rs[0].text.contains("Adds.") && !rs[0].text.contains("body_marker"),
            "{rs:?}"
        );
        assert_eq!(
            chunk_source("README.md", "# A\nx\n## B\ny\n", &reg).len(),
            2
        );
        assert!(chunk_source("notes.txt", "# A\nx\n", &reg).is_empty());
        // [CP2 MS3 loop 14 (pasada final)] Under docs/ the extension still decides: code is never sent as documentation.
        let ex = chunk_source(
            "docs/example.rs",
            "/// Shows use.\nfn demo() { let body_marker = 1; }\n",
            &reg,
        );
        assert!(
            ex.iter().all(|c| !c.text.contains("body_marker"))
                && ex.iter().any(|c| c.text.contains("Shows use.")),
            "{ex:?}"
        );
    }

    #[test]
    fn a_chunk_with_a_secret_is_left_out() {
        // [RED-DRYRUN 2026-10-17, E3] One-line let: rustfmt splits the call and moves the allow marker off the
        // literal's line, and tests/no_hardcoded_secrets.rs goes red (measured).
        let text = "# Setup\nexport KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\n# Other\nfine\n"; // allow-secret-scan
        let c = chunk_markdown("README.md", text);
        let (kept, dropped) = without_secrets(c);
        assert_eq!(
            dropped,
            vec!["README.md".to_string()],
            "[CP2 MS3 loop 12] named, for RV22's skipped list"
        );
        assert_eq!(
            kept.iter().map(|x| x.section.as_str()).collect::<Vec<_>>(),
            vec!["Other"]
        );
        // [CP2 MS3 loop 13] A secret in a heading is a secret too (the section is sent with the chunk).
        let heading = "# sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\nfine\n"; // allow-secret-scan
        assert!(without_secrets(chunk_markdown("README.md", heading))
            .0
            .is_empty());
    }

    #[test]
    fn indexing_is_bounded_and_the_truncation_is_reported() {
        // [CP2 MS3 loop 6] Reverse order, so keeping the first by path needs the sort, not only the cut.
        // [RV21 D+B, 2026-10-20] One chunk per source here; whole sources are kept.
        let many: Vec<(DocSource, Vec<DocChunk>)> = (0..MAX_DOC_CHUNKS + 1)
            .rev()
            .map(|i| {
                let path = format!("docs/{i:05}.md");
                (
                    DocSource {
                        path: path.clone(),
                        key: format!("k{i}"),
                    },
                    vec![DocChunk {
                        path,
                        section: String::new(),
                        text: "x".into(),
                        hash: format!("{i}"),
                    }],
                )
            })
            .collect();
        let (kept, truncated) = bounded(many);
        assert_eq!(kept.len(), MAX_DOC_CHUNKS);
        assert!(truncated);
        assert_eq!(
            kept.first().map(|(s, _)| s.path.as_str()),
            Some("docs/00000.md")
        );
    }

    #[test]
    fn a_source_is_indexed_whole_or_left_for_a_later_review() {
        // [RV21 D+B, 2026-10-20] A half-indexed source would be remembered as indexed and never completed.
        let c = |p: &str, n: usize| -> (DocSource, Vec<DocChunk>) {
            (
                DocSource {
                    path: p.into(),
                    key: format!("k-{p}"),
                },
                (0..n)
                    .map(|i| DocChunk {
                        path: p.into(),
                        section: format!("{i}"),
                        text: "x".into(),
                        hash: format!("{p}{i}"),
                    })
                    .collect(),
            )
        };
        let (kept, truncated) = bounded(vec![
            c("a.md", MAX_DOC_CHUNKS - 1),
            c("b.md", 2),
            c("c.md", 1),
        ]);
        assert!(truncated);
        // [CP2 MS3 loop 13] b.md does not fit and is skipped; c.md still does.
        assert_eq!(
            kept.iter()
                .map(|(s, v)| (s.path.as_str(), v.len()))
                .collect::<Vec<_>>(),
            vec![("a.md", MAX_DOC_CHUNKS - 1), ("c.md", 1)]
        );
        // [CP2 MS3 loop 13] An oversized first source never starves the rest.
        let (kept, truncated) = bounded(vec![c("a.md", MAX_DOC_CHUNKS + 1), c("b.md", 1)]);
        assert!(truncated);
        assert_eq!(
            kept.iter()
                .map(|(s, _)| s.path.as_str())
                .collect::<Vec<_>>(),
            vec!["b.md"]
        );
    }

    #[test]
    fn worktree_keys_are_content_addressed_and_never_look_like_a_git_id() {
        // [RV21 D+B, 2026-10-20]
        assert_eq!(worktree_key(b"x"), worktree_key(b"x"));
        assert_ne!(worktree_key(b"x"), worktree_key(b"y"));
        assert!(worktree_key(b"x").starts_with("wt:"));
        assert_eq!(worktree_key(b"x").len(), 3 + 64);
    }

    #[test]
    fn ls_tree_blobs_become_the_doc_sources_of_that_revision() {
        // [RV21 D+B, 2026-10-20] -z keeps spaces in paths; a submodule is not a blob; `keep` filters.
        // [RED-DRYRUN 2026-10-20, RV21] `\x00`, not `\0`: before a digit `\0` is clippy::octal_escapes.
        let out = b"100644 blob aaaa\tREADME.md\x00100644 blob bbbb\tsrc/a b.rs\x00160000 commit cccc\tvendor/sub\x00100644 blob dddd\tnotes.txt\x00";
        let s = parse_ls_tree(out, &|p: &str| p != "notes.txt").expect("parses");
        assert_eq!(
            s,
            vec![
                DocSource {
                    path: "README.md".into(),
                    key: "aaaa".into()
                },
                DocSource {
                    path: "src/a b.rs".into(),
                    key: "bbbb".into()
                }
            ]
        );
    }

    #[test]
    fn an_ls_tree_entry_that_does_not_parse_is_an_error_never_skipped() {
        // [RV21 D+B, 2026-10-20] A silent skip would leave a source out of the reviewed revision's set.
        assert!(matches!(
            parse_ls_tree(b"garbage\0", &|_: &str| true),
            Err(ReviewError::Git(_))
        ));
    }

    /// Answers each argv (joined by spaces) from a fixed table; anything else is a git error. [RV21 D+B, 2026-10-20]
    struct TableGit(Vec<(&'static str, Vec<u8>)>);
    impl GitCommand for TableGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            let k = args.join(" ");
            self.0
                .iter()
                .find(|(a, _)| *a == k)
                .map(|(_, out)| out.clone())
                .ok_or(ReviewError::Git(k))
        }
    }

    #[test]
    fn a_revision_lists_its_sources_with_ls_tree() {
        let git = TableGit(vec![(
            "ls-tree -r -z abc",
            b"100644 blob k1\tREADME.md\0".to_vec(),
        )]);
        assert_eq!(
            revision_doc_sources(&git, "abc", &|_: &str| true).expect("lists"),
            vec![DocSource {
                path: "README.md".into(),
                key: "k1".into()
            }]
        );
    }

    #[test]
    fn live_doc_sources_are_the_union_of_every_local_ref() {
        // [RV21 D+B, 2026-10-20] RV-S39: what a branch, a tag or HEAD still reaches stays indexed.
        let git = TableGit(vec![
            ("rev-parse --verify HEAD^{commit}", b"h1\n".to_vec()),
            (
                "for-each-ref --format=%(objectname) refs/heads refs/tags",
                b"h1\nb2\n".to_vec(),
            ),
            ("ls-tree -r -z h1", b"100644 blob k1\tREADME.md\0".to_vec()),
            ("ls-tree -r -z b2", b"100644 blob k2\tREADME.md\0".to_vec()),
        ]);
        let live = live_doc_sources(&git, &|_: &str| true)
            .expect("ok")
            .expect("under the cap");
        assert_eq!(
            live.iter().map(|s| s.key.as_str()).collect::<Vec<_>>(),
            vec!["k1", "k2"]
        );
    }

    #[test]
    fn past_the_ref_cap_the_clean_up_is_skipped_never_partial() {
        // [RV21 D+B, 2026-10-20] A partial live set would delete what an unwalked ref still reaches.
        let refs: String = (0..=MAX_GC_REFS).map(|i| format!("r{i}\n")).collect();
        let git = TableGit(vec![
            ("rev-parse --verify HEAD^{commit}", b"r0\n".to_vec()),
            (
                "for-each-ref --format=%(objectname) refs/heads refs/tags",
                refs.into_bytes(),
            ),
        ]);
        assert_eq!(live_doc_sources(&git, &|_: &str| true).expect("ok"), None);
    }

    #[test]
    fn an_unborn_head_contributes_nothing_and_is_not_an_error() {
        // [CP2 MS3 loop 13] A repository with no commit yet: rev-parse fails, no refs, an empty live set.
        let git = TableGit(vec![(
            "for-each-ref --format=%(objectname) refs/heads refs/tags",
            Vec::new(),
        )]);
        assert_eq!(
            live_doc_sources(&git, &|_: &str| true).expect("ok"),
            Some(std::collections::BTreeSet::new())
        );
    }

    #[test]
    fn a_git_that_does_not_answer_skips_the_clean_up() {
        // [CP2 MS3 loop 14 (pasada final)] HEAD unreadable while refs exist is not «unborn»: never delete on a broken git.
        let refs_but_no_head = TableGit(vec![
            (
                "for-each-ref --format=%(objectname) refs/heads refs/tags",
                b"b2\n".to_vec(),
            ),
            ("ls-tree -r -z b2", b"100644 blob k2\tREADME.md\0".to_vec()),
        ]);
        assert_eq!(
            live_doc_sources(&refs_but_no_head, &|_: &str| true).expect("ok"),
            None
        );
        let nothing_answers = TableGit(vec![]);
        assert_eq!(
            live_doc_sources(&nothing_answers, &|_: &str| true).expect("ok"),
            None
        );
    }

    #[test]
    fn the_index_id_names_the_model_and_the_chunker_version() {
        // [CP2 MS3 loop 10] Same dimension, other model or other chunker: never the same index.
        // [CP2 MS3 loop 11] The dimension too: every change of the vectors' identity is another index.
        assert_eq!(
            doc_index_id("nomic-embed-text", 768),
            format!("nomic-embed-text#768#{DOC_INDEX_VERSION}")
        );
        assert_ne!(doc_index_id("a", 3), doc_index_id("b", 3));
        assert_ne!(doc_index_id("a", 3), doc_index_id("a", 5));
    }

    #[test]
    fn doc_sources_are_recognized_by_path() {
        for p in [
            "README.md",
            "CLAUDE.md",
            "docs/guide.md",
            "docs/adr/0001.md",
            "adr/0001.md",
        ] {
            assert!(is_doc_source(p), "{p}");
        }
        assert!(!is_doc_source("src/main.rs"));
        assert!(
            !is_doc_source("sub/CLAUDE.md"),
            "[CP2 MS3 loop 13] root files by exact path"
        );
    }
}
