// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! Hunk parser — splits unified diffs into atomic reviewable units.

/// An atomic reviewable unit from a unified diff.
///
/// Each hunk contains the file header (`diff --git`, `---`, `+++`)
/// prepended to the hunk body starting at a `@@` marker.
#[derive(Debug, Clone)]
pub struct Hunk {
    /// Source file path.
    pub file: String,
    /// Full content: file header + hunk body.
    pub content: String,
    /// Total line count of `content`.
    pub lines: usize,
}

/// Parse a unified diff into atomic hunks.
///
/// Splits on `@@` markers and prepends the file header to each hunk; a diff that opens with a
/// `@@` marker has no header, and each of its hunks starts at its marker.
/// If no `@@` markers are found, returns the entire diff as a single hunk.
///
/// # Arguments
///
/// * `file` - Source file path for the diff.
/// * `diff` - Unified diff content for a single file.
///
/// # Returns
///
/// A vector of `Hunk` values, one per `@@` marker found.
///
/// # Examples
///
/// ```
/// use magi_rs::review::hunk::parse_hunks;
///
/// let diff = "diff --git a/f.c b/f.c\n--- a/f.c\n+++ b/f.c\n@@ -1,3 +1,4 @@\n line1\n+added\n line2\n";
/// let hunks = parse_hunks("f.c", diff);
/// assert_eq!(hunks.len(), 1);
/// ```
pub fn parse_hunks(file: &str, diff: &str) -> Vec<Hunk> {
    if diff.is_empty() {
        return vec![];
    }
    let header = extract_header(diff);
    let bodies = split_on_hunk_markers(diff);
    if bodies.is_empty() {
        return vec![Hunk {
            file: file.to_string(),
            content: diff.to_string(),
            lines: diff.lines().count(),
        }];
    }
    bodies
        .into_iter()
        .map(|body| {
            // MAGI FIX (S6 loop 18, B1-6): an empty header is not joined, so a headerless hunk
            // starts at its `@@` marker instead of a blank line.
            let content = if header.is_empty() {
                body
            } else {
                format!("{}\n{}", header, body)
            };
            let lines = content.lines().count();
            Hunk {
                file: file.to_string(),
                content,
                lines,
            }
        })
        .collect()
}

/// Extract the file header lines (before the first `@@` marker).
fn extract_header(diff: &str) -> String {
    diff.lines()
        .take_while(|line| !line.starts_with("@@"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split diff into hunk bodies starting at each `@@` marker.
fn split_on_hunk_markers(diff: &str) -> Vec<String> {
    let mut bodies = Vec::new();
    let mut current: Option<Vec<&str>> = None;

    for line in diff.lines() {
        if line.starts_with("@@") {
            if let Some(chunk) = current.take() {
                bodies.push(chunk.join("\n"));
            }
            current = Some(vec![line]);
        } else if let Some(ref mut chunk) = current {
            chunk.push(line);
        }
    }
    if let Some(chunk) = current {
        bodies.push(chunk.join("\n"));
    }
    bodies
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_fixture(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("review")
            .join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to load fixture {}: {}", name, e))
    }

    // --- extract_header ---

    #[test]
    fn extract_header_returns_lines_before_first_hunk_marker() {
        let diff = load_fixture("simple.diff");
        let header = extract_header(&diff);
        assert!(
            header.contains("diff --git"),
            "Header should contain 'diff --git'"
        );
        assert!(
            header.contains("--- a/src/main.c"),
            "Header should contain '---' line"
        );
        assert!(
            header.contains("+++ b/src/main.c"),
            "Header should contain '+++' line"
        );
        assert!(
            !header.contains("@@"),
            "Header should not contain @@ markers"
        );
    }

    #[test]
    fn extract_header_from_mode_change_returns_full_content() {
        let diff = load_fixture("mode_change.diff");
        let header = extract_header(&diff);
        assert!(
            header.contains("diff --git"),
            "Mode-change header should contain 'diff --git'"
        );
        assert!(
            header.contains("old mode"),
            "Mode-change header should contain 'old mode'"
        );
    }

    // --- split_on_hunk_markers ---

    #[test]
    fn split_single_hunk_returns_one_body() {
        let diff = load_fixture("simple.diff");
        let bodies = split_on_hunk_markers(&diff);
        assert_eq!(
            bodies.len(),
            1,
            "simple.diff should produce exactly 1 hunk body"
        );
        assert!(
            bodies[0].starts_with("@@"),
            "Hunk body should start with @@ marker"
        );
    }

    #[test]
    fn split_no_hunk_markers_returns_empty() {
        let diff = load_fixture("mode_change.diff");
        let bodies = split_on_hunk_markers(&diff);
        assert!(
            bodies.is_empty(),
            "Mode-change diff has no @@ markers, should return empty"
        );
    }

    // --- parse_hunks ---

    #[test]
    fn parse_simple_diff_returns_one_hunk() {
        let diff = load_fixture("simple.diff");
        let hunks = parse_hunks("src/main.c", &diff);
        assert_eq!(hunks.len(), 1, "simple.diff should produce 1 hunk");
    }

    #[test]
    fn parse_hunk_contains_file_header() {
        let diff = load_fixture("simple.diff");
        let hunks = parse_hunks("src/main.c", &diff);
        let content = &hunks[0].content;
        assert!(
            content.contains("diff --git"),
            "Hunk content should contain file header"
        );
        assert!(
            content.contains("@@"),
            "Hunk content should contain @@ marker"
        );
    }

    #[test]
    fn parse_hunk_has_correct_file_path() {
        let diff = load_fixture("simple.diff");
        let hunks = parse_hunks("src/main.c", &diff);
        assert_eq!(
            hunks[0].file, "src/main.c",
            "Hunk file should match the input file path"
        );
    }

    #[test]
    fn parse_hunk_line_count_matches_content() {
        let diff = load_fixture("simple.diff");
        let hunks = parse_hunks("src/main.c", &diff);
        let expected_lines = hunks[0].content.lines().count();
        assert_eq!(
            hunks[0].lines, expected_lines,
            "Hunk lines field should match actual line count"
        );
    }

    #[test]
    fn parse_multi_hunk_file_returns_multiple_hunks() {
        // multi_file.diff has 2 hunks for src/main.c
        let diff = load_fixture("multi_file.diff");

        // Extract only src/main.c portion from the multi-file diff
        let main_c_diff = extract_single_file_diff(&diff, "src/main.c");
        let hunks = parse_hunks("src/main.c", &main_c_diff);
        assert_eq!(
            hunks.len(),
            2,
            "src/main.c in multi_file.diff should produce 2 hunks"
        );
    }

    #[test]
    fn parse_each_hunk_has_header_prepended() {
        let diff = load_fixture("multi_file.diff");
        let main_c_diff = extract_single_file_diff(&diff, "src/main.c");
        let hunks = parse_hunks("src/main.c", &main_c_diff);
        assert!(!hunks.is_empty(), "main.c should produce hunks to check");
        for (i, hunk) in hunks.iter().enumerate() {
            assert!(
                hunk.content.contains("diff --git"),
                "Hunk {} should have file header prepended",
                i
            );
        }
    }

    #[test]
    fn parse_mode_change_returns_single_hunk_as_fallback() {
        let diff = load_fixture("mode_change.diff");
        let hunks = parse_hunks("scripts/build.sh", &diff);
        assert_eq!(
            hunks.len(),
            1,
            "Mode-change diff (no @@) should return 1 fallback hunk"
        );
        assert!(
            hunks[0].content.contains("old mode"),
            "Fallback hunk should contain the original diff content"
        );
    }

    #[test]
    fn parse_empty_diff_returns_empty() {
        let hunks = parse_hunks("empty.c", "");
        assert!(hunks.is_empty(), "Empty diff should produce no hunks");
    }

    /// Helper: extract the diff section for a single file from a multi-file diff.
    fn extract_single_file_diff(full_diff: &str, target_file: &str) -> String {
        let marker = format!("diff --git a/{}", target_file);
        let lines: Vec<&str> = full_diff.lines().collect();
        let mut start = None;
        let mut end = lines.len();

        for (i, line) in lines.iter().enumerate() {
            if line.starts_with(&marker) {
                start = Some(i);
            } else if line.starts_with("diff --git") && start.is_some() {
                end = i;
                break;
            }
        }

        match start {
            Some(s) => lines[s..end].join("\n"),
            None => String::new(),
        }
    }

    #[test]
    fn crlf_files_keep_their_line_numbers() {
        let diff = "@@ -1,2 +1,3 @@\r\n a\r\n+b\r\n c\r\n";
        let hunks = parse_hunks("a.rs", diff);
        assert_eq!(hunks.len(), 1);
        // `Hunk::lines` counts the hunk's content lines (header + body: 5 here),
        // not new-side lines; the property is that CRLF parses exactly like LF.
        let lf = parse_hunks("a.rs", &diff.replace("\r\n", "\n"));
        assert_eq!(hunks[0].lines, lf[0].lines);
        assert_eq!(hunks[0].content, lf[0].content);
        assert!(hunks[0].content.contains("+b"));
    }

    #[test]
    fn no_prefix_of_a_hostile_diff_panics_the_hunk_parser() {
        // Release is panic=abort: malformed headers, a truncated binary patch and CRLF
        // must not abort.
        let hostile = concat!(
            "@@ -x,y +",
            "\n@@ -1,99999999999 +1 @@\n+a\n",
            "@@@ -0,0 +0,0 @@@\r\n",
            "diff --git a/b.bin b/b.bin\nGIT binary patch\nliteral 12\nzcm\n",
            "@@ -3 +3 @@\r\n-x\r\n\\ No newline at end of file\r\n+"
        );
        for end in hostile
            .char_indices()
            .map(|(i, _)| i)
            .chain([hostile.len()])
        {
            let _ = parse_hunks("a.rs", &hostile[..end]);
        }
    }

    #[test]
    fn a_headerless_diff_gives_hunks_that_start_at_their_marker() {
        // [MAGI S6 loop 18, B1-6] an empty header was still joined with a newline, so the
        // hunk opened with a blank line and counted it.
        let hunks = parse_hunks("a.rs", "@@ -1 +1 @@\n-a\n+b\n@@ -9 +9 @@\n-c\n+d\n");
        assert_eq!(hunks.len(), 2);
        assert_eq!(hunks[0].content, "@@ -1 +1 @@\n-a\n+b");
        assert_eq!(hunks[0].lines, 3);
        assert_eq!(hunks[1].content, "@@ -9 +9 @@\n-c\n+d");
    }
}
