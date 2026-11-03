// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Construct detection for [`BraceFamilyExtractor`]: the line-by-line passes that find
//! functions, types and declaration groups, their member passes, and the attribute and
//! doc-comment recovery around each construct.

use super::*;

/// What a C or C++ declaration line never opens with: a closing brace, a `)` (it closes a list
/// wrapped from a line above: `);`, `) const;`, `),`), a preprocessor `#` or a comment.
const NOT_DECLARATION_OPENERS: [&str; 6] = ["}", ")", "#", "//", "/*", "*"];

/// What a line opening a declaration list never opens with besides
/// [`NOT_DECLARATION_OPENERS`]: a constructor's initializer list (`: a_(0),`) or a list
/// continuation (`, b`).
const NOT_DECLARATION_LIST_OPENERS: [&str; 2] = [":", ","];

impl BraceFamilyExtractor {
    /// Detect all constructs in the file or, with `members`, in a block's body, where a
    /// signature that ends in `;` is a member too ([`Self::member_prototypes`]).
    ///
    /// No constructs when a line number does not fit `u32` ([`Self::try_detect_constructs`]):
    /// the file's changed lines then fall into gap units and a block is reviewed whole, as
    /// [`shift_members`] leaves it, never split at fabricated coordinates.
    ///
    /// Invariant: the constructs come sorted by their first line and their ranges are disjoint,
    /// which `ChangedLines::assign` and `split_into_members` rely on; lines no construct holds
    /// are the gaps. O(n) in the lines ([`Self::try_detect_constructs`]).
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines: a whole file, or the lines of a block's body.
    /// * `scanned` - The scanner's record of each of those lines, in the same order.
    /// * `members` - Whether the lines are a block's body (a class, an `impl`, a module).
    ///
    /// # Returns
    ///
    /// The constructs, 1-based line numbers counted from the first of `lines`.
    // MAGI FIX (S6 loop 14): the arguments, the result and the invariant are documented.
    pub(super) fn detect_constructs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        members: bool,
    ) -> Vec<Construct> {
        let budget = SearchBudget::for_lines(lines.len());
        self.try_detect_constructs(lines, scanned, members, &budget)
            .unwrap_or_default()
    }

    /// The constructs of [`Self::detect_constructs`], or `None` when a line number does not
    /// fit `u32`. The 8 MiB file cap keeps that out of reach (8 388 608 lines at most); the
    /// check makes the failure mode explicit instead of a `u32::MAX` fallback.
    ///
    /// A heuristic of several rules, tried in this order on each line: a Go type group, a block
    /// or function keyword, a C-family signature, then a declaration or a declaration group.
    /// Each rule lives in its own helper and the order is the contract, since a later rule would
    /// misread what an earlier one claims (Java's `public class` as a function): a new rule or
    /// language goes into a helper placed in that order, and the test
    /// `construct_ranges_are_sorted_and_disjoint` with the per-language tests pins the result.
    /// A known maintenance hotspot, declared. Splitting it into a table of rule strategies is
    /// deferred until the heuristic set settles after MS3: the rules' order is the contract,
    /// and a large refactor at the end of the review gate risks moving it.
    /// O(n) in the lines, plus the helpers' forward searches, which share `budget`
    /// ([`SearchBudget`]) and so add O(n) line visits in all.
    ///
    /// Each construct found moves the scan past its last line, so no two ranges overlap; the
    /// scan back for its doc comments never goes above the line after the previous construct
    /// ([`Self::scan_back_for_docs`]).
    ///
    /// Units are line ranges, so two constructs that share a line (`struct S {} impl S {`,
    /// `fn a() { 1 } fn b() {`) are one unit, named after the first, declared: the second one's
    /// header is on that line, and keeping its body with it is the only cut that leaves no
    /// header apart from its body. Its range ends where the last block the line leaves open
    /// closes, and the constructs after it keep their own units.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `members` - Whether the lines are a block's body: a signature ending in `;` is then a
    ///   member ([`Self::member_prototypes`]), and a JavaScript or TypeScript method directly in
    ///   it may be named by a reserved word (`reserved_word_methods`). A signature deeper than
    ///   the body's first line is then no member: it sits inside a member no rule read (a
    ///   method of an anonymous class in a field's initializer), which stays one piece.
    /// * `budget` - The line visits the forward searches of this pass may spend.
    ///
    /// # Returns
    ///
    /// The constructs, sorted and disjoint, or `None` when a line number does not fit `u32`.
    pub(super) fn try_detect_constructs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        members: bool,
        budget: &SearchBudget,
    ) -> Option<Vec<Construct>> {
        // MAGI FIX (S6 loop 10): the conversions fell back to `u32::MAX`, `1` or the line
        // number, unlike `shift_members`; an unrepresentable line now aborts detection.
        let mut constructs = Vec::new();
        let mut i = 0;
        let depth_before = |k: usize| {
            scanned
                .get(k)
                .map(|s| i64::from(s.depth_after) - i64::from(s.brace_delta))
        };
        let stops = |j: usize| self.starts_construct(lines, scanned, j, members);
        let search = BraceSearch {
            stops: &stops,
            budget,
        };

        while let Some(line) = lines.get(i) {
            if scanned.get(i).is_some_and(|s| s.is_non_code) {
                i += 1;
                continue;
            }

            let trimmed = line.trim();
            let line_num = line_number(i)?;
            // MAGI FIX (S6 loop 12): the scan back for docs crossed into a one-line construct
            // that opens with an attribute prefix (`@Override public String toString() { .. }`),
            // so two ranges overlapped; it now stops at the line after the previous construct.
            let floor = constructs
                .last()
                .map_or(0, |c: &Construct| c.end_line as usize);

            // MAGI FIX (loop 26, 3): a `template <` header wrapped over lines was read as a
            // declaration list and cut from the construct it declares; its lines are passed
            // over to the one that closes it, and the scan back for docs recovers them.
            if let Some(close) = self.wrapped_template_header_end(lines, scanned, i, &search) {
                i = close;
                continue;
            }
            // MAGI FIX (S6 loop 7): Go's `type (` opened one nameless struct and its later
            // members fell into gaps; each member is a construct of its own.
            if let Some((last, group)) = self.grouped_types(lines, scanned, i, &search) {
                constructs.extend(group?);
                i = last + 1;
                continue;
            }
            // MAGI FIX (loop 28, B2b1a): a construct on the line closing a wrapped attribute or
            // template header (`typename U> class Box {`) matched no keyword; the line is read
            // after the attribute's tail, and the attribute's lines belong to the construct.
            let keyword = self
                .construct_keyword(trimmed)
                .map(|found| (i, found))
                .or_else(|| {
                    self.after_wrapped_attribute_tail(lines, scanned, i, floor, &search, |rest| {
                        self.construct_keyword(rest)
                    })
                });
            if let Some((head, (kind, name))) = keyword {
                let start = self.scan_back_for_docs(lines, scanned, head, floor);
                // A line closing an attribute's brace before opening the construct's
                // (`}) export class C {`) nets no brace; it opens one after its lowest point.
                let reopens =
                    head != i && scanned.get(i).is_some_and(|s| s.brace_delta > s.min_delta);
                let mut end = if reopens {
                    block_end_after(lines, scanned, i, line_num, Some(i), &search)?
                } else {
                    brace_block_end(lines, scanned, i, line_num, &search)?
                };
                // MAGI FIX (S6 loop 17, B2-2): a `macro_rules! m (` ... `);` ended at the first
                // brace of its arms; it runs to its own closing delimiter.
                if self.opens_macro_rules(trimmed) {
                    end = end.max(line_number(group_end(lines, scanned, i, &search))?);
                }
                constructs.push(Construct {
                    kind,
                    name,
                    start_line: line_number(start)?,
                    end_line: end,
                });
                // Skip past this construct
                i = (end as usize).max(i + 1);
                continue;
            }
            let prototypes = members && self.member_prototypes();
            // A signature deeper than the body's own lines sits inside a member no rule read
            // (an anonymous class in a field's initializer, `r = new Runnable() {`): taking it
            // for a member cut the field's header from its body.
            let nested = members && depth_before(i) != depth_before(0);
            // MAGI FIX (loop 28 sweep): a signature on the line closing a wrapped annotation
            // (`x = 1) void m() {` under `@Ann(`) was no signature, the tail's `=` or `)` read as
            // its own; the line is read after the annotation's tail too.
            let signature = (!nested)
                .then(|| {
                    self.line_signature(line, scanned, i, members)
                        .map(|name| (i, name))
                        .or_else(|| {
                            self.after_wrapped_attribute_tail(
                                lines,
                                scanned,
                                i,
                                floor,
                                &search,
                                |rest| self.line_signature(rest, scanned, i, members),
                            )
                        })
                })
                .flatten();
            if let Some((attribute_head, name)) = signature {
                // MAGI FIX (S6 loop 15): the brace search ran twice, once here and once in
                // `brace_block_end`, spending a wrapped header's visits of the budget twice.
                // A line closing an annotation's brace before opening the body's (`}) void m() {`)
                // nets no brace; it opens one after its lowest point.
                let open = if attribute_head != i
                    && scanned.get(i).is_some_and(|s| s.brace_delta > s.min_delta)
                {
                    Some(i)
                } else {
                    open_brace_line(lines, scanned, i, &search)
                };
                let opens = open.is_some();
                // MAGI FIX (S6 loop 18, B2b-8): a one-line function was told by its text ending
                // in `}`, so a comment after the brace (`void f() {} // note`) left it in a gap;
                // the scanner says whether its code ends in `}`.
                let one_line = scanned.get(i).is_some_and(|s| s.ends_block);
                if opens || one_line || prototypes {
                    let head = self
                        .return_type_start(lines, scanned, i, floor, members)
                        .min(attribute_head);
                    let start = self.scan_back_for_docs(lines, scanned, head, floor);
                    // MAGI FIX (S6 loop 5): an abstract member (`User find(long id);`) ends at its
                    // `;`, which may sit on a later line of a wrapped signature.
                    let end = if opens || one_line {
                        block_end_after(lines, scanned, i, line_num, open, &search)?
                    } else {
                        line_number(statement_end(lines, scanned, i, &search))?
                    };
                    let end =
                        self.past_member_initializers(lines, scanned, i, open, end, &search)?;
                    constructs.push(Construct {
                        kind: UnitKind::Function,
                        name,
                        start_line: line_number(start)?,
                        end_line: end,
                    });
                    i = (end as usize).max(i + 1);
                    continue;
                }
            }
            // Check declaration keywords
            // MAGI FIX (S6 loop 8): `int x,` opened no declaration, so a declaration list
            // wrapped after a comma lost its first line to a gap unit.
            if self.is_declaration(trimmed, scanned.get(i))
                || self.opens_declaration_list(trimmed)
                || self.opens_wrapped_prototype(lines, scanned, i, &search)
            {
                // MAGI FIX (S6 loop 5): `const f = () => {` ended on its first line and its body
                // fell into a gap; a function expression is a function, with its whole body.
                if let Some((name, end)) = self.function_expression(lines, scanned, i, &search) {
                    let start = self.scan_back_for_docs(lines, scanned, i, floor);
                    constructs.push(Construct {
                        kind: UnitKind::Function,
                        name,
                        start_line: line_number(start)?,
                        end_line: line_number(end)?,
                    });
                    i = end + 1;
                    continue;
                }
                let group_start = i;
                // MAGI FIX (S6 loop 7): a `#define` continued by `\` ends on its last line.
                let end_of = |j: usize| self.declaration_end(lines, scanned, j, &search);
                // A declaration joins the group unless it is a function expression, its own unit,
                // or opens a construct: the main scan tries constructs first, and so does this.
                // MAGI FIX (S6 loop 13): a comment line ending in `;` joined the group; only code
                // does, as in the main scan.
                // MAGI FIX (loop 27, B2b1a): `export class C {` after an import joined its group
                // (`export` is a declaration keyword) and the group ran through the class; and a
                // template header wrapped over lines (`template <typename T,`), which the main
                // scan passes over first, joined it as a declaration list.
                let joins = |j: usize| {
                    scanned.get(j).is_some_and(|s| !s.is_non_code)
                        && self
                            .wrapped_template_header_end(lines, scanned, j, &search)
                            .is_none()
                        && (lines.get(j).is_some_and(|l| {
                            self.construct_keyword(l.trim()).is_none()
                                && (self.is_declaration(l.trim(), scanned.get(j))
                                    || self.opens_declaration_list(l.trim()))
                        }) || self.opens_wrapped_prototype(lines, scanned, j, &search))
                        && self
                            .function_expression(lines, scanned, j, &search)
                            .is_none()
                };
                let mut group_end = end_of(i);
                // Scan forward for contiguous declarations, one blank line allowed between two.
                loop {
                    let next = group_end + 1;
                    if joins(next) {
                        group_end = end_of(next);
                    } else if lines.get(next).is_some_and(|l| l.trim().is_empty())
                        && joins(next + 1)
                    {
                        group_end = end_of(next + 1);
                    } else {
                        break;
                    }
                }
                let head = self.return_type_start(lines, scanned, group_start, floor, members);
                let doc_start = self.scan_back_for_docs(lines, scanned, head, floor);
                let name = self.declaration_group_name(self.strip_modifiers(trimmed, false));
                constructs.push(Construct {
                    kind: UnitKind::DeclarationGroup,
                    name,
                    start_line: line_number(doc_start)?,
                    end_line: line_number(group_end)?,
                });
                i = group_end + 1;
                continue;
            }

            i += 1;
        }

        Some(constructs)
    }

    /// The 0-based last line of the declaration at line `j`: the end of the block its value
    /// opens ([`declaration_block`]), of a `#define` continued by `\` ([`preprocessor_end`]), of
    /// a declaration list wrapped after a comma ([`Self::declaration_list_end`]), of a call
    /// its initializer opens ([`call_initializer_end`]), of the parentheses or brackets its line
    /// leaves open ([`group_end`]) or of the value after a bare `=` ([`bare_initializer_end`]),
    /// whichever is last; `j` itself otherwise. O(k) for the k lines
    /// of the declaration.
    ///
    /// A declaration that runs to its closing delimiter is one declaration however many members
    /// it holds (a Go `var (` group, a long array literal), so a declaration group too large for
    /// one batch does not split inside it ([`Self::split_declaration_group`]) and ends the review
    /// with the typed `ReviewError::TooLargeForBatch` (RV03), declared: never a silent cut.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `j` - The 0-based line the declaration starts on.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// The declaration's last line, 0-based, never before `j`.
    fn declaration_end(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        search: &BraceSearch<'_>,
    ) -> usize {
        // MAGI FIX (S6 loop 17, B2-1): a declaration that leaves a `(` or `[` open (Go's
        // `var (`, an array literal over lines, a wrapped call) ended at its first line or at a
        // brace inside it; it now runs to its matching closing delimiter.
        declaration_block(lines, scanned, j, search)
            .map_or(j, |b| b.1)
            .max(preprocessor_end(lines, j))
            .max(self.declaration_list_end(lines, scanned, j))
            .max(call_initializer_end(lines, scanned, j, search))
            .max(group_end(lines, scanned, j, search))
            // MAGI FIX (S6 loop 19, B2b-5): a declaration wrapped after a bare `=` ended on its
            // first line, and its value fell into a gap unit.
            .max(bare_initializer_end(lines, scanned, j, search))
    }

    /// The pieces of a declaration group too large for one batch: each declaration with its doc
    /// comments and attributes, and each line between two (a blank line) as a piece with the
    /// group's kind and name ([`split_into_members`]). Splitting at declarations is no silent
    /// cut (RV03): every line stays in exactly one piece. Empty when the group holds no
    /// declaration or a line number does not fit `u32`, so the caller batches it whole.
    /// O(n) in the group's lines, plus each declaration's forward search.
    pub(super) fn split_declaration_group(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        // MAGI FIX (S6 loop 12): a group was never split, so a generated header or a constant
        // table over the batch limit refused the whole review although each declaration fits.
        let lines: Vec<&str> = unit.content.lines().collect();
        let scanned = CodeScanner::new(self.lexical_rules.clone()).scan_all(&unit.content);
        let stops = |j: usize| self.starts_construct(&lines, &scanned, j, false);
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        let mut members: Vec<Member> = Vec::new();
        let mut floor = 0;
        let mut j = 0;
        while let Some(line) = lines.get(j) {
            let trimmed = line.trim();
            // MAGI FIX (S6 loop 25, B2b1a): the group joins a wrapped prototype
            // (`int foo(` ... `);`), so its split must see one too, or the prototype was cut at
            // its first parameter line.
            let declares = scanned.get(j).is_some_and(|s| !s.is_non_code)
                && (self.is_declaration(trimmed, scanned.get(j))
                    || self.opens_declaration_list(trimmed)
                    || self.opens_wrapped_prototype(&lines, &scanned, j, &search));
            if !declares {
                j += 1;
                continue;
            }
            let end = self.declaration_end(&lines, &scanned, j, &search).max(j);
            let start = self.scan_back_for_docs(&lines, &scanned, j, floor);
            let Some((first, last)) = line_number(start).zip(line_number(end)) else {
                return Vec::new();
            };
            members.push((UnitKind::DeclarationGroup, unit.name.clone(), first, last));
            floor = end + 1;
            j = end + 1;
        }
        split_into_members(unit, members)
    }

    /// The members of a grouped type declaration at line `i` (Go's `type (` ... `)`) and the
    /// group's last line (0-based, its `)`), or `None` when line `i` opens none. Each member is a
    /// construct with its doc comments: a struct, an interface (as a trait) or a defined type or
    /// alias (as a struct, as a lone `type ID int` is), named by its first word and ending on the
    /// line that closes the parentheses and brackets it opens (a wrapped `func(` type), then at
    /// the closing brace of a block that line opens, or on its own line when it opens neither,
    /// never past the group's last member line. The `type (` and `)` lines
    /// belong to no member, so they fall into gap units. O(k) for the k lines of the group.
    ///
    /// The group's end is found by counting parentheses in code context ([`code_group_delta`]),
    /// so one inside a raw string, a string or a comment moves nothing; a
    /// group that never closes runs to the end of the file, its members still found one by one,
    /// so every line stays in exactly one unit either way. A member's closing-brace search stays
    /// inside the group and spends the pass's `search` budget ([`SearchBudget`]); a member it
    /// does not close runs to the group's end. The groups of a file are disjoint, so their walks
    /// add O(n) in all. The members are `None` when a line
    /// number does not fit `u32`, which aborts detection ([`Self::try_detect_constructs`]).
    fn grouped_types(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        search: &BraceSearch<'_>,
    ) -> Option<(usize, Option<Vec<Construct>>)> {
        if !self.block_keywords.iter().any(|k| k == "type ") {
            return None;
        }
        let rest = keyword_rest(lines.get(i)?.trim(), "type ")?;
        if !rest.trim_start().starts_with('(') {
            return None;
        }
        let mut depth = 0i64;
        let close = (i..lines.len()).find(|&j| {
            // MAGI FIX (B2b1 B1): the parentheses were counted on the line text, so a `)` in a
            // raw string cut the group short and a `(` in a comment absorbed the next function.
            depth += code_group_delta(scanned, j);
            depth <= 0
        });
        let last = close.unwrap_or(lines.len().saturating_sub(1));
        let body_end = close.unwrap_or(lines.len());
        let mut members = Some(Vec::new());
        let mut floor = i + 1;
        let mut j = i + 1;
        while j < body_end {
            let trimmed = lines.get(j).map_or("", |l| l.trim());
            let name: String = trimmed
                .chars()
                .take_while(|&c| is_identifier_char(c))
                .collect();
            if scanned.get(j).is_none_or(|s| s.is_non_code) || name.is_empty() {
                j += 1;
                continue;
            }
            let kind = go_type_kind(trimmed);
            // MAGI FIX (S6 loop 19, B2b-2): the search ran past the group's `)` to the end of
            // the file, outside the pass budget, so a file of groups whose members never close
            // inside them took O(n²) visits; it stays inside the group and spends the budget. A
            // member not closed there, or whose search the budget cut, runs to the group's end,
            // so no line of the group is lost.
            // MAGI FIX (B2b1 7): a member ended on its first line unless that line opened a
            // brace, so a wrapped `func(` type or type parameter list (`List[`) gave each later
            // line a member of its own. The member first runs to the line closing the
            // parentheses and brackets it opens ([`group_end`]), kept inside the group, then to
            // the closing brace of a block that line opens.
            let open = group_end(lines, scanned, j, search)
                .min(body_end.saturating_sub(1))
                .max(j);
            let end = if scanned.get(open).is_some_and(|s| s.brace_delta > 0) {
                find_closing_brace_within(scanned, open, body_end, || search.spend())
                    .unwrap_or(body_end.saturating_sub(1))
                    .max(open)
            } else {
                open
            };
            let start = self.scan_back_for_docs(lines, scanned, j, floor);
            let member = line_number(start)
                .zip(line_number(end))
                .map(|(start_line, end_line)| Construct {
                    kind,
                    name,
                    start_line,
                    end_line,
                });
            members = members.zip(member).map(|(mut all, m)| {
                all.push(m);
                all
            });
            floor = end + 1;
            j = end + 1;
        }
        Some((last, members))
    }

    /// The kind and name of the block or function construct a trimmed line opens, matched on
    /// the line with its visibility and qualifiers stripped ([`Self::strip_modifiers`]) or,
    /// failing that, on the line as written. Block keywords are tried before function keywords:
    /// Java's `public class` would otherwise read as a function named `class`. O(k + n) for k
    /// keywords and a line of n characters.
    pub(super) fn construct_keyword(&self, trimmed: &str) -> Option<(UnitKind, String)> {
        // MAGI FIX (S6 loop 4): qualifiers combine freely (`pub(super) async unsafe fn`); the
        // tables of fixed prefixes missed most combinations.
        // MAGI FIX (S6 loop 8): `template<typename T> class Box {` was no construct at all.
        // MAGI FIX (S6 loop 10): `@Deprecated public class A {` was no construct at all.
        // MAGI FIX (S6 loop 13): `struct __attribute__((packed)) Hdr {` was named after the
        // attribute; the C attributes are read past first.
        // MAGI FIX (S6 loop 14): `#[inline] fn f() {` was no construct and its body fell into a
        // gap unit; the Rust attributes on the line are read past first.
        let trimmed = self.without_rust_attributes(trimmed);
        let attributes = self.without_c_attributes(trimmed);
        let annotations = self.without_annotations_among_modifiers(&attributes);
        let trimmed = self.without_template_header(&annotations);
        let candidates = [self.strip_modifiers(trimmed, true), trimmed];
        candidates
            .iter()
            .find_map(|&text| {
                // MAGI FIX (S6 loop 6): `struct Foo x = {1};` is a variable of a struct type,
                // a declaration, not a struct.
                // MAGI FIX (S6 loop 7): `struct Point make_point(void) {` is a function returning
                // a struct; the signature path names it.
                self.match_block_keyword(text)
                    .filter(|(_, rest)| {
                        !(self.semicolon_declarations
                            && (declares_variable(rest) || returns_from_function(rest)))
                    })
                    .map(|(kind, rest)| {
                        let name = construct_name(rest, text);
                        // MAGI FIX (B2b1 10): `namespace a::b {` was named `a`.
                        let name = if kind == UnitKind::Module {
                            nested_namespace_name(rest, name)
                        } else {
                            name
                        };
                        (kind, name)
                    })
            })
            .or_else(|| {
                candidates.iter().find_map(|&text| {
                    self.match_function_keyword(text)
                        .map(|rest| (UnitKind::Function, construct_name(rest, text)))
                })
            })
    }

    /// Whether a trimmed line opens a declaration: a declaration keyword starts the line as
    /// written or with its visibility stripped (`pub(crate) use`), or, with
    /// `semicolon_declarations`, it is code ending in `;` that does not open with `}`, `)`, `#` or
    /// a comment: a line opening with `)` closes a list wrapped from a line above (`);`,
    /// `) const;`), so it is never a declaration of its own. With `scanned`, "ending in `;`" is
    /// the scanner's end of statement, so a comment after the code (`static int n = 0; // note`)
    /// does not hide it, as in [`Self::signature_candidate`]; without it, the trimmed text must
    /// end in `;`. Only lines outside every construct are asked, so a statement inside a function
    /// body is never taken for one. O(k + n).
    ///
    /// # Arguments
    ///
    /// * `trimmed` - The line, trimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    ///
    /// # Returns
    ///
    /// Whether the line opens a declaration.
    fn is_declaration(&self, trimmed: &str, scanned: Option<&ScannedLine>) -> bool {
        // MAGI FIX (S6 loop 14): `#[allow(x)] const X: u8 = 1;` fell into a gap unit.
        let trimmed = self.without_rust_attributes(trimmed);
        // MAGI FIX (S6 loop 4): a C global with no declaration keyword fell into a gap.
        // MAGI FIX (S6 loop 6): so did one whose initializer opens on a later line
        // (`int a[] = {`, `int n =`); its value is the declaration's block or next statement.
        let semicolon = self.semicolon_declarations
            // MAGI FIX (S6 loop 16): `static int n = 0; // note` fell into a gap unit.
            && (scanned.map_or(trimmed.ends_with(';'), |s| s.ends_statement)
                || opens_initializer(trimmed))
            && !NOT_DECLARATION_OPENERS
                .iter()
                .any(|p| trimmed.starts_with(p));
        semicolon
            || self.match_declaration_keyword(trimmed)
            || self.match_declaration_keyword(self.strip_modifiers(trimmed, false))
    }

    /// Whether a trimmed line opens a C or C++ declaration list wrapped after a comma (`int x,`,
    /// `static const char *a = "x",`), for extractors with `semicolon_declarations`: code ending
    /// in `,` that does not open with what no declaration opens with ([`NOT_DECLARATION_OPENERS`],
    /// a `)` included: it closes a list wrapped from a line above) nor with `:` or `,`
    /// ([`NOT_DECLARATION_LIST_OPENERS`]). Only the main scan and its
    /// declaration groups ask it, never [`Self::starts_construct`]: a constructor's initializer
    /// line (`b_{0},`) ends in `,` too, and stopping a signature's brace search there cut the
    /// function. A comment after the comma (`int x, // note`) hides it, declared: that line falls
    /// into a gap unit as before. O(n) in the line.
    fn opens_declaration_list(&self, trimmed: &str) -> bool {
        self.semicolon_declarations
            && trimmed.ends_with(',')
            // MAGI FIX (loop 28, B2b1a): a stray `),` opened a list that ran over a function.
            && !NOT_DECLARATION_OPENERS
                .iter()
                .chain(&NOT_DECLARATION_LIST_OPENERS)
                .any(|p| trimmed.starts_with(p))
    }

    /// The 0-based last line of the declaration list that line `i` opens
    /// ([`Self::opens_declaration_list`]): the first later code line that ends in `;`. The walk
    /// ends earlier, on the line before, at a blank line, a line whose braces do not balance (an
    /// initializer block, which joins the group as a declaration of its own) or a line that opens
    /// a block or function construct, so the list never takes a later construct; a line ending
    /// in `,` is a declarator of the list (`g(void),`, `(*fp)(int),`), never a function header.
    /// `i` when line `i`
    /// opens no list. O(k) for the k lines walked.
    fn declaration_list_end(&self, lines: &[&str], scanned: &[ScannedLine], i: usize) -> usize {
        if !lines
            .get(i)
            .is_some_and(|l| self.opens_declaration_list(l.trim()))
        {
            return i;
        }
        let mut last = i;
        for (k, line) in lines.iter().enumerate().skip(i + 1) {
            let trimmed = line.trim();
            let Some(record) = scanned.get(k) else {
                break;
            };
            if trimmed.is_empty() {
                break;
            }
            if record.is_non_code {
                continue;
            }
            if record.ends_statement {
                return k;
            }
            // MAGI FIX (B2b1 8): a list line holding a function declarator (`g(void),`) read as a
            // signature and ended the list before it; a line that ends in `,` continues the list,
            // as no function header does (its parameters would close first).
            if record.brace_delta != 0
                || self.construct_keyword(trimmed).is_some()
                || (!trimmed.ends_with(',')
                    && self.signature_name(line, Some(record), false).is_some())
            {
                break;
            }
            last = k;
        }
        last
    }

    /// `trimmed` without the visibility words it opens with and, when `qualifiers`, the
    /// qualifier words after them, in any order and number. A word counts only when a blank or
    /// a `(` follows it (`public`, not `publicity`); a `(...)` right after a word is part of it
    /// (`pub(in crate::a)`), and a quoted string after a word is too (`extern "C"`). O(n) in
    /// the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    /// * `qualifiers` - Whether the qualifier words are stripped too.
    ///
    /// # Returns
    ///
    /// The rest of the line, trimmed at its start; `trimmed` when it opens with no such word.
    fn strip_modifiers<'t>(&self, trimmed: &'t str, qualifiers: bool) -> &'t str {
        let words = self
            .modifiers
            .visibility
            .iter()
            .chain(self.modifiers.qualifiers.iter().filter(|_| qualifiers));
        let mut rest = trimmed;
        loop {
            let Some(after) = words.clone().find_map(|w| strip_modifier(rest, w)) else {
                return rest;
            };
            rest = after;
        }
    }

    /// `trimmed` without the C++ template header it opens with ([`strip_template_header`]), for
    /// extractors with `template_headers`; `trimmed` otherwise. O(n) in the line.
    fn without_template_header<'t>(&self, trimmed: &'t str) -> &'t str {
        if self.template_headers {
            strip_template_header(trimmed)
        } else {
            trimmed
        }
    }

    /// `trimmed` without the complete annotations it opens with (`@Override`,
    /// `@SuppressWarnings("x")`, `@Component({ a: 1 })`), for extractors whose
    /// `attribute_prefixes` hold `@` (Java, JavaScript, TypeScript); `trimmed` otherwise.
    ///
    /// An annotation is `@`, a name of identifier characters and dots, and optionally its
    /// arguments in balanced parentheses on the same line. `@interface` declares an annotation
    /// type and is kept. An annotation whose arguments do not close on the line is kept with
    /// everything after it, so the line still opens with `@` and is no signature, as before.
    /// Parentheses inside the arguments' strings do not count ([`balanced_end_outside_quotes`]
    /// with [`ANNOTATION_QUOTES`]), so `@Component(")")` closes at its last `)`; a string that
    /// never closes on the line keeps the annotation, as above. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    ///
    /// # Returns
    ///
    /// The rest of the line after its leading annotations, trimmed at its start.
    fn without_annotations<'t>(&self, trimmed: &'t str) -> &'t str {
        if !self.attribute_prefixes.iter().any(|p| p == "@") {
            return trimmed;
        }
        let mut rest = trimmed;
        while let Some(after_at) = rest.strip_prefix('@') {
            let name_len: usize = after_at
                .chars()
                .take_while(|&c| is_identifier_char(c) || c == '.')
                .map(char::len_utf8)
                .sum();
            let (name, mut tail) = after_at.split_at(name_len);
            if name.is_empty() || name == "interface" {
                return rest;
            }
            if tail.starts_with('(') {
                // MAGI FIX (S6 loop 14): `@Component(")")` closed at the quoted `)` and the
                // class after it was no unit; the quoted parentheses no longer count.
                let Some(close) = balanced_end_outside_quotes(tail, '(', ')', &ANNOTATION_QUOTES)
                else {
                    return rest;
                };
                tail = tail.get(close..).unwrap_or("");
            }
            rest = tail.trim_start();
        }
        rest
    }

    /// `trimmed` without the complete annotations it opens with and those mixed among the
    /// modifier words that follow ([`Self::without_annotations`], [`Self::strip_modifiers`]):
    /// Java and TypeScript let annotations and modifiers come in any order
    /// (`public @Deprecated final class C {` reads `public final class C {`,
    /// `export @dec class C {` reads `export class C {`). The modifiers are kept, so a word that
    /// is a modifier only before a blank (`get() {`) still names what it names. `trimmed`'s
    /// leading annotations only, borrowed, when no annotation follows a modifier. O(n) in the
    /// line: each round consumes what it strips.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    ///
    /// # Returns
    ///
    /// The line without those annotations.
    fn without_annotations_among_modifiers<'t>(&self, trimmed: &'t str) -> Cow<'t, str> {
        // MAGI FIX (loop 27, B2b1a): annotations were read past only before the modifiers, so
        // `public @Deprecated class C {` was no class and `public @A final @B("x") class C {` a
        // function named `B`.
        let mut rest = self.without_annotations(trimmed);
        let mut kept = String::new();
        loop {
            let after_modifiers = self.strip_modifiers(rest, true);
            let after_annotations = self.without_annotations(after_modifiers);
            if after_annotations.len() == after_modifiers.len() {
                break;
            }
            kept.push_str(rest.get(..rest.len() - after_modifiers.len()).unwrap_or(""));
            rest = after_annotations;
        }
        if kept.is_empty() {
            Cow::Borrowed(rest)
        } else {
            kept.push_str(rest);
            Cow::Owned(kept)
        }
    }

    /// `trimmed` without the complete Rust outer attributes it opens with (`#[inline]`,
    /// `#[derive(Debug)] #[cfg(test)]`), for extractors whose `attribute_prefixes` hold
    /// [`RUST_ATTRIBUTE_PREFIX`] (Rust); `trimmed` otherwise.
    ///
    /// An attribute is `#[` up to its matching `]`, read past its strings and raw strings
    /// ([`rust_attribute_end`]), so `#[doc = "]"]` and `#[doc = r#"a"b"#]` close at their last
    /// `]`. An attribute that does not close on the line is kept with everything after it, so
    /// the line still opens with `#[` and reads as before: an attribute line, joined to the item
    /// below it ([`Self::scan_back_for_docs`]). O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    ///
    /// # Returns
    ///
    /// The rest of the line after its leading attributes, trimmed at its start.
    fn without_rust_attributes<'t>(&self, trimmed: &'t str) -> &'t str {
        if !self
            .attribute_prefixes
            .iter()
            .any(|p| p == RUST_ATTRIBUTE_PREFIX)
        {
            return trimmed;
        }
        let mut rest = trimmed;
        while rest.starts_with(RUST_ATTRIBUTE_PREFIX) {
            let attribute = rest.get(1..).unwrap_or("");
            let Some(end) = rust_attribute_end(attribute) else {
                return rest;
            };
            rest = attribute.get(end..).unwrap_or("").trim_start();
        }
        rest
    }

    /// `trimmed` without its C and C++ attributes ([`strip_c_attributes`]), for extractors with
    /// `semicolon_declarations` (C and C++); `trimmed` otherwise. O(n) in the line.
    fn without_c_attributes<'t>(&self, trimmed: &'t str) -> Cow<'t, str> {
        if self.semicolon_declarations {
            strip_c_attributes(trimmed)
        } else {
            Cow::Borrowed(trimmed)
        }
    }

    /// Whether a trimmed line opens a Rust `macro_rules!` definition ([`MACRO_RULES_KEYWORD`]),
    /// its outer attributes read past (`#[macro_export] macro_rules! m {`). Only the Rust
    /// extractor holds the keyword in its function table, so no other language matches. O(n).
    fn opens_macro_rules(&self, trimmed: &str) -> bool {
        self.function_keywords
            .iter()
            .any(|k| k == MACRO_RULES_KEYWORD)
            && macro_rules_rest(self.without_rust_attributes(trimmed)).is_some()
    }

    /// The text after the function keyword `trimmed` opens with ([`keyword_rest`]), if any, or,
    /// with `anonymous_functions`, after the keyword's bare word glued to a `(`
    /// (`function() {` gives `() {`). The `macro_rules!` keyword is matched by
    /// [`macro_rules_rest`], its `!` glued or spaced.
    fn match_function_keyword<'t>(&self, trimmed: &'t str) -> Option<&'t str> {
        self.function_keywords.iter().find_map(|kw| {
            if kw == MACRO_RULES_KEYWORD {
                return macro_rules_rest(trimmed);
            }
            keyword_rest(trimmed, kw).or_else(|| {
                let word = kw.strip_suffix(' ')?;
                trimmed
                    .strip_prefix(word)
                    .filter(|rest| self.anonymous_functions && rest.starts_with('('))
            })
        })
    }

    /// Whether line `j` (0-based) is code that opens a construct: a function, block or
    /// declaration keyword, or a signature as the main scan reads it
    /// ([`Self::line_signature`]). O(k) in the keywords.
    pub(super) fn starts_construct(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        members: bool,
    ) -> bool {
        if scanned.get(j).is_none_or(|s| s.is_non_code) {
            return false;
        }
        let Some(line) = lines.get(j) else {
            return false;
        };
        let trimmed = line.trim();
        // MAGI FIX (S6 loop 18, B2b-9): the stop rule ruled out a reserved word unlike the main
        // scan, so a member with no `;` (`abstract foo(): void`) took the body of the
        // `delete() {` after it; both read a signature through `line_signature`.
        self.construct_keyword(trimmed).is_some()
            || self.is_declaration(trimmed, scanned.get(j))
            || self.line_signature(line, scanned, j, members).is_some()
    }

    /// The function name of the signature on line `j` (0-based), as the main scan reads it:
    /// [`Self::signature_name`], or, for a line directly in a class body of a language with
    /// `reserved_word_methods`, [`Self::signature_candidate`], where a reserved word names a
    /// method (`delete() {`). Shared by the main scan and the forward searches' stop rule
    /// ([`Self::starts_construct`]), so the two never disagree on where a construct starts. O(n)
    /// in the line.
    ///
    /// The depth of a block body's own lines is the depth before its first line: a line at that
    /// depth is a direct member, one deeper sits inside a member (a static block).
    ///
    /// # Arguments
    ///
    /// * `line` - Line `j`, untrimmed.
    /// * `scanned` - The scanner's record of each line of the text.
    /// * `j` - The 0-based line.
    /// * `members` - Whether the lines are a block's body.
    ///
    /// # Returns
    ///
    /// The function name, or `None` when the line is no signature.
    // MAGI FIX (S6 loop 14): `delete() {` in a JavaScript class was no member; directly in a
    // class body a reserved word names a method.
    fn line_signature(
        &self,
        line: &str,
        scanned: &[ScannedLine],
        j: usize,
        members: bool,
    ) -> Option<String> {
        let depth_before = |k: usize| {
            scanned
                .get(k)
                .map(|s| i64::from(s.depth_after) - i64::from(s.brace_delta))
        };
        let prototypes = members && self.member_prototypes();
        if members && self.reserved_word_methods && depth_before(j) == depth_before(0) {
            self.signature_candidate(line, scanned.get(j), prototypes)
        } else {
            self.signature_name(line, scanned.get(j), prototypes)
        }
    }

    /// The function name of a C-family signature line, for extractors with
    /// `signature_functions`: a line that holds `(`, whose code does not end in `;` (a prototype;
    /// with `scanned`, a comment after the code does not count), does not
    /// open with `#` (but for `computed_member_names`, where it opens a private name), a comment, `}`, `@`, `,` or a single `:` (a C++ constructor's initializer line,
    /// `: value_(v) {`, continues the signature above it), has no `=` before its `(`, and names an identifier,
    /// not a word of its `not_function_names`, just before the `(` (`int *parse(` gives `parse`, `void Foo::bar(` gives
    /// `bar`). The line may be indented: a function inside a namespace, a class or an
    /// `extern "C"` block is a function too. Only lines outside every construct are asked, so a
    /// call inside a recognised function body is never taken for one. O(n) in the line.
    ///
    /// A C++ template header is read past first ([`Self::without_template_header`]): alone on its
    /// line (`template <typename T>`) it is no signature, since nothing follows it, so it never
    /// names a function after its last parameter; it joins the construct below as one of the
    /// extractor's `attribute_prefixes`.
    ///
    /// # Arguments
    ///
    /// * `line` - The line, untrimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    /// * `prototypes` - Whether a line whose code ends in `;` is a signature too: an abstract
    ///   member of an interface or class body ([`Self::member_prototypes`]).
    ///
    /// # Returns
    ///
    /// The function name, or `None` when the line is no signature.
    pub(super) fn signature_name(
        &self,
        line: &str,
        scanned: Option<&ScannedLine>,
        prototypes: bool,
    ) -> Option<String> {
        // MAGI FIX (S6 loop 4): `catch (e) {` read as a function named `catch`.
        self.signature_candidate(line, scanned, prototypes)
            .filter(|name| !self.not_function_names.contains(&name.as_str()))
    }

    /// The name [`Self::signature_name`] reads on a signature line before it rules out the words
    /// of `not_function_names`: the same rules and arguments, with a reserved word kept as a
    /// name. Only a direct member of a JavaScript or TypeScript class body is read with it
    /// (`reserved_word_methods`), where `delete() {` declares a method. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `line` - The line, untrimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    /// * `prototypes` - Whether a line whose code ends in `;` is a signature too.
    ///
    /// # Returns
    ///
    /// The identifier before the line's `(`, or `None` when the line is no signature.
    fn signature_candidate(
        &self,
        line: &str,
        scanned: Option<&ScannedLine>,
        prototypes: bool,
    ) -> Option<String> {
        // MAGI FIX (S6 loop 3): indentation no longer disqualifies a signature.
        if !self.signature_functions {
            return None;
        }
        // MAGI FIX (S6 loop 10): `@Override public void run() {` opened with `@` and was no
        // signature; the complete annotations on the line are read past first.
        let annotations = self.without_annotations_among_modifiers(line.trim());
        let trimmed: &str = &annotations;
        // MAGI FIX (S6 loop 4): a trailing comment ending in `;` is not a prototype's `;`.
        let prototype = scanned.map_or(trimmed.ends_with(';'), |s| s.ends_statement);
        // An annotation or decorator (`@SuppressWarnings("x")`) is no signature either.
        // MAGI FIX (S6 loop 7): a `*` opens a comment's continuation line only when the line
        // starts inside the comment; in code it is a generator method (`*items() {`).
        let comment_star = trimmed.starts_with('*') && scanned.is_none_or(|s| s.starts_in_literal);
        // MAGI FIX (S6 loop 9): a C++ constructor's initializer line (`: value_(v) {`,
        // `, count_(0)`) read as a signature named after the member, so the constructor's brace
        // search stopped there; it continues the signature above it.
        let initializer =
            trimmed.starts_with(',') || (trimmed.starts_with(':') && !trimmed.starts_with("::"));
        // MAGI FIX (S6 loop 17, B2-3): a private method (`#m() {`) opened with `#` and was read
        // as a preprocessor line; in JavaScript and TypeScript `#` opens a private name.
        let directive = trimmed.starts_with('#') && !self.computed_member_names;
        if (prototype && !prototypes)
            || comment_star
            || initializer
            || directive
            || ["//", "/*", "}", "@"]
                .iter()
                .any(|p| trimmed.starts_with(p))
        {
            return None;
        }
        // MAGI FIX (S6 loop 8): `template <void (*F)(int)> void call() {` was named `void`; a
        // header alone on its line (`template <typename T>`) is no signature: nothing follows it.
        // MAGI FIX (S6 loop 13): `__attribute__((noreturn)) void die(void) {` was a function
        // named `__attribute__`; the C attributes are read past first.
        let attributes = self.without_c_attributes(trimmed);
        let text = self.without_template_header(&attributes);
        self.operator_overloads
            .then(|| operator_overload_name(text))
            .flatten()
            .or_else(|| {
                self.computed_member_names
                    .then(|| private_or_computed_name(text))
                    .flatten()
            })
            .or_else(|| identifier_before_paren(text))
            // MAGI FIX (B2b1 10): `foo?(x: number): void;` named no member.
            .or_else(|| {
                self.optional_members
                    .then(|| optional_member_name(text))
                    .flatten()
            })
    }

    /// The kind of the block `trimmed` opens and the text after its keyword
    /// ([`keyword_rest`]), if it opens one.
    fn match_block_keyword<'t>(&self, trimmed: &'t str) -> Option<(UnitKind, &'t str)> {
        for kw in &self.block_keywords {
            if let Some(rest) = keyword_rest(trimmed, kw) {
                // MAGI FIX (B2b1 9): `namespace` is no reserved word in TypeScript or C, so
                // `namespace = 2;` assigns a variable; a namespace is named, never assigned.
                // MAGI FIX (B2b1a): a C++ alias (`namespace fs = std::filesystem;`) is a
                // declaration too: a `=` before the block's `{` or a `;` makes no namespace.
                if kw.trim() == "namespace"
                    && rest
                        .split(['{', ';'])
                        .next()
                        .is_some_and(|head| head.contains('='))
                {
                    continue;
                }
                // MAGI FIX (S6 loop 5): the tables hold bare keywords and the modifiers are
                // stripped before they are matched, so the `pub struct` and `export class` arms
                // could never match; a bare keyword names the kind.
                let kind = match kw.trim() {
                    // MAGI FIX (S6 loop 13): `type Reader interface {` was a struct.
                    "type" => go_type_kind(rest),
                    "struct" | "union" => UnitKind::Struct,
                    "enum" | "const enum" => UnitKind::Enum,
                    "trait" | "interface" => UnitKind::Trait,
                    "impl" => UnitKind::Impl,
                    // MAGI FIX (S6 loop 6): a module was function-like, so it never split.
                    // MAGI FIX (B2a 1): a C++ namespace holds free functions, as a module does.
                    "mod" | "namespace" => UnitKind::Module,
                    _ => UnitKind::Class, // class, record, @interface
                };
                // MAGI FIX (S6 loop 7): C++'s `enum class Color` was named `class`.
                let rest = if kind == UnitKind::Enum {
                    ["class ", "struct "]
                        .iter()
                        .find_map(|w| rest.strip_prefix(w))
                        .unwrap_or(rest)
                } else {
                    rest
                };
                return Some((kind, rest));
            }
        }
        None
    }

    /// Whether the line opens with a declaration keyword followed by any blank or by the list,
    /// path or string it declares ([`declaration_keyword_rest`]).
    fn match_declaration_keyword(&self, trimmed: &str) -> bool {
        // MAGI FIX (S6 loop 14): `use::{a, b};`, `import{foo} from "x";` and
        // `#include<stdio.h>` fell into gap units.
        self.declaration_keywords
            .iter()
            .any(|kw| declaration_keyword_rest(trimmed, kw).is_some())
    }

    /// The 0-based first line of the construct whose declaration is at `idx`: `idx` moved up
    /// over the doc comments and attributes right above it. A doc line opens with a doc prefix
    /// (`///`, `/**`); a block comment that spans lines (`/**` ... `*/`) counts whole when its
    /// opening line does ([`Self::block_doc_start`]). O(k) for the k lines passed over. Up to
    /// [`MAX_BLANK_LINES_AFTER_ATTRIBUTE`] blank lines are passed over when an attribute stands
    /// above them ([`Self::attribute_above_blanks`]); above a doc comment or anything else, a
    /// blank line ends the walk.
    ///
    /// The walk never goes above `floor`, the 0-based line after the previous construct: a
    /// one-line construct may itself open with an attribute prefix (Java's
    /// `@Override public String toString() { .. }`, C++'s `template<typename T> void f() { .. }`),
    /// and taking it would make two construct ranges overlap, which `ChangedLines::assign` and
    /// `split_into_members` rule out. `floor` is also returned when a multi-line comment or
    /// attribute recovered below it. O(k) for the k lines passed over, at most `idx - floor`
    /// plus the lines of one such comment or attribute.
    fn scan_back_for_docs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        idx: usize,
        floor: usize,
    ) -> usize {
        let mut start = idx;
        while start > floor {
            let Some(prev) = lines.get(start - 1).map(|l| l.trim()) else {
                break;
            };
            let is_doc = self
                .doc_comment_prefixes
                .iter()
                .any(|p| prev.starts_with(p.as_str()));
            let is_attr = self
                .attribute_prefixes
                .iter()
                .any(|p| prev.starts_with(p.as_str()));
            if is_doc || is_attr {
                start -= 1;
            } else if let Some(open) = self
                .block_doc_start(lines, scanned, start - 1)
                .or_else(|| self.multi_line_attribute_start(lines, scanned, start - 1, floor))
            {
                start = open;
            } else if let Some(above) = self.attribute_above_blanks(lines, scanned, start, floor) {
                // MAGI FIX (B2b1 12): a blank line between an attribute and its item
                // (`#[derive(Debug)]`, blank, `struct S {`) cut the attribute from it.
                start = above + 1;
            } else {
                break;
            }
        }
        start.max(floor)
    }

    /// The 0-based line of the attribute right above the blank lines that end at line `start`
    /// (exclusive), for [`Self::scan_back_for_docs`]: at most [`MAX_BLANK_LINES_AFTER_ATTRIBUTE`]
    /// blank lines, then a line that opens with an attribute prefix or ends a multi-line
    /// attribute ([`Self::multi_line_attribute_start`]), never above `floor`. `None` when the
    /// line above `start` is not blank, more blank lines stand there, or what precedes them is
    /// no attribute (a plain comment, code): such a line stays apart. O(b + k) for the b blank
    /// lines and the k lines of a multi-line attribute.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `start` - The 0-based first line reached so far by the walk up.
    /// * `floor` - The 0-based line the walk never goes above.
    ///
    /// # Returns
    ///
    /// The attribute's 0-based last line, or `None`.
    fn attribute_above_blanks(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        start: usize,
        floor: usize,
    ) -> Option<usize> {
        let is_blank = |k: usize| lines.get(k).is_some_and(|l| l.trim().is_empty());
        let blanks = (floor..start)
            .rev()
            .take(MAX_BLANK_LINES_AFTER_ATTRIBUTE + 1)
            .take_while(|&k| is_blank(k))
            .count();
        if blanks == 0 || blanks > MAX_BLANK_LINES_AFTER_ATTRIBUTE {
            return None;
        }
        let above = start.checked_sub(blanks + 1).filter(|&k| k >= floor)?;
        let text = lines.get(above)?.trim();
        let opens_attribute = self
            .attribute_prefixes
            .iter()
            .any(|p| text.starts_with(p.as_str()));
        (opens_attribute
            || self
                .multi_line_attribute_start(lines, scanned, above, floor)
                .is_some())
        .then_some(above)
    }

    /// The 0-based opening line of the doc block comment that ends on line `last` (its `*/`):
    /// the lines above it that start inside the comment are passed over up to the line that
    /// opens it, which must open with a doc prefix (`/**`). `None` when `last` ends no block
    /// comment or the comment is no doc comment (`/* plain */`). O(k) for the k lines of the
    /// comment.
    fn block_doc_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        last: usize,
    ) -> Option<usize> {
        // MAGI FIX (S6 loop 5): the line above a declaration is `*/` for a multi-line JavaDoc,
        // JSDoc or C doc block, so the whole comment was left out of the unit.
        if !lines.get(last)?.trim_end().ends_with("*/") {
            return None;
        }
        let mut open = last;
        while scanned.get(open)?.starts_in_literal {
            open = open.checked_sub(1)?;
        }
        let opener = lines.get(open)?.trim();
        (open < last
            && self
                .doc_comment_prefixes
                .iter()
                .any(|p| p.starts_with("/*") && opener.starts_with(p.as_str())))
        .then_some(open)
    }

    /// The 0-based line that closes the C++ template header opened, and left open, on line `i`
    /// (`template <typename T,` ... `typename U>`), for extractors with `template_headers`.
    /// `None` when line `i` opens no template header, closes it on the same line (a header alone
    /// on its line, or before the construct it declares), the header never closes, or the
    /// pass budget runs out ([`SearchBudget`]); the lines then read as before, each in exactly
    /// one unit.
    ///
    /// Angle brackets are counted on the line text of code lines, comment lines passed over:
    /// a `>` of an operator in a default argument (`->`, `>=`) may miscount, and the header
    /// then reads as before, declared. O(k) for the k lines walked, each charged to `search`.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `i` - The 0-based line the header would open on.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// The 0-based line holding the header's closing `>`, always after `i`, or `None`.
    fn wrapped_template_header_end(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        search: &BraceSearch<'_>,
    ) -> Option<usize> {
        let first = lines.get(i)?.trim();
        if !self.template_headers
            || !opens_template_header(first)
            || !strip_template_header(first).is_empty()
        {
            return None;
        }
        let mut depth = 0i64;
        for (j, line) in lines.iter().enumerate().skip(i) {
            if j > i && !search.spend() {
                return None;
            }
            if scanned.get(j).is_none_or(|s| s.is_non_code) {
                continue;
            }
            depth += template_angle_delta(line);
            if depth <= 0 {
                return (j > i).then_some(j);
            }
        }
        None
    }

    /// The 0-based opening line of the attribute that spans lines and ends on line `last`
    /// (Rust's `#[serde(` ... `)]`): the lines above `last` are walked up while its square
    /// brackets stay unbalanced, up to the line that balances them, which must open with an
    /// attribute prefix ending in `[` (`#[`). `None` when `last` does not end in `]`, the
    /// extractor has no such prefix, `last` is not code, or the line that balances
    /// the brackets opens no attribute (a macro call `list! [` ... `]`).
    ///
    /// A line on the way that lies wholly inside a string or a comment (a text block, a template
    /// literal in the arguments) is passed over: it holds no bracket of the attribute.
    ///
    /// Brackets are counted on the line text: a bracket
    /// inside a string may miscount, and the attribute then falls into a gap unit as before, so
    /// no line is lost. O(k) for the k lines walked.
    ///
    /// An annotation of Java, JavaScript or TypeScript whose arguments wrap over lines
    /// (`@SuppressWarnings(` ... `)`, `@Component({` ... `})`) is recovered the same way on its
    /// parentheses: `last` ends in `)` and the line that balances them opens with an attribute
    /// prefix that does not end in `[` (`@`).
    ///
    /// For an extractor with `template_headers` (C++), a template header wrapped over lines
    /// (`template <typename T,` then `typename U>`) is recovered the same way on its angle
    /// brackets: `last` ends in `>` and the line that balances them opens a template header
    /// ([`opens_template_header`]).
    ///
    /// The walk never goes above `floor`, the line after the previous construct: an attribute
    /// opening above it would overlap that construct, and an unbounded walk made a file of
    /// closing lines between constructs (`)` then `fn f() {}`, repeated) cost O(n²) line visits.
    /// Each walk stays inside the gap above its own construct, and the gaps of a pass are
    /// disjoint, so the walks add O(n) in all.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `last` - The 0-based line the attribute would end on.
    /// * `floor` - The 0-based line the walk never goes above.
    ///
    /// # Returns
    ///
    /// The attribute's 0-based opening line, or `None`.
    fn multi_line_attribute_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        last: usize,
        floor: usize,
    ) -> Option<usize> {
        // MAGI FIX (S6 loop 10): the line above an item under a multi-line attribute is `)]`,
        // so the attribute was cut from the item it controls and reviewed as a gap unit.
        // MAGI FIX (S6 loop 12): a wrapped `@` annotation ends in `)` or `})`, which the `[`
        // rule never matched, so it was cut from its member too.
        // MAGI FIX (loop 26, 3): a C++ `template <` header wrapped over lines ends in `>`; it is
        // recovered on its angle brackets, up to the line that opens it.
        let brackets = self.attribute_brackets(lines.get(last)?.trim_end().chars().last()?)?;
        // MAGI FIX (S6 loop 19, B2b-4): the line the walk starts on must be code; a line wholly
        // inside a string or a comment further up holds no bracket and is passed over.
        if scanned.get(last)?.is_non_code {
            return None;
        }
        let unclosed = brackets.net_closers(lines.get(last)?);
        let above = last.checked_sub(1).filter(|&k| k >= floor)?;
        if unclosed <= 0 {
            return None;
        }
        self.attribute_opening_above(lines, scanned, floor..=above, unclosed, brackets, || true)
    }

    /// The bracket pair a multi-line attribute ending in `close` is recovered on, or `None` when
    /// none ends so in this language: `]` (Rust's `#[serde(` ... `)]`, C's `[[deprecated(` ...
    /// `)]]`), `)` (an `@` annotation's arguments) and, with `template_headers`, `>` (a C++
    /// template header). O(1).
    fn attribute_brackets(&self, close: char) -> Option<AttributeBrackets> {
        let (open, bracket_prefix) = match close {
            ']' => ('[', true),
            ')' => ('(', false),
            '>' if self.template_headers => ('<', false),
            _ => return None,
        };
        Some(AttributeBrackets {
            open,
            close,
            bracket_prefix,
        })
    }

    /// The 0-based line that opens the attribute whose `unclosed` closers the lines below `span`
    /// leave unmatched: the walk goes up from the last line of `span`, adding each code line's
    /// closers less its openers ([`AttributeBrackets::net_closers`]) and passing over lines
    /// wholly inside a string or a comment, until the count balances, on a line that must open
    /// with an attribute prefix of the pair's kind ([`AttributeBrackets`]) or, for `>`, a
    /// template header. `None` when it balances on another line, goes below zero, would leave
    /// `span`, or `visit` stops it. O(k) for the k lines walked.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `span` - The 0-based lines the walk may look at: it starts on the last and never goes
    ///   above the first.
    /// * `unclosed` - The closers the lines below `span` leave unmatched, at least 1.
    /// * `brackets` - The bracket pair counted.
    /// * `visit` - Called once before each line looked at; `false` stops the walk.
    ///
    /// # Returns
    ///
    /// The attribute's 0-based opening line, or `None`.
    fn attribute_opening_above(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        span: std::ops::RangeInclusive<usize>,
        mut unclosed: i64,
        brackets: AttributeBrackets,
        mut visit: impl FnMut() -> bool,
    ) -> Option<usize> {
        let opens_attribute = |text: &str| {
            if brackets.close == '>' {
                return opens_template_header(text);
            }
            self.attribute_prefixes.iter().any(|p| {
                p.ends_with('[') == brackets.bracket_prefix && text.starts_with(p.as_str())
            })
        };
        let floor = *span.start();
        let mut j = *span.end();
        loop {
            if !visit() {
                return None;
            }
            if !scanned.get(j)?.is_non_code {
                let line = lines.get(j)?;
                unclosed += brackets.net_closers(line);
                if unclosed <= 0 {
                    return (unclosed == 0 && opens_attribute(line.trim())).then_some(j);
                }
            }
            // MAGI FIX (loop 26, 1): the walk had no floor and climbed to the top of the file
            // while the brackets stayed unbalanced, so `)` lines between functions cost O(n²).
            j = j.checked_sub(1).filter(|&k| k >= floor)?;
        }
    }

    /// What line `i` reads as after the tail of an attribute or a C++ template header wrapped from
    /// the lines above it (`typename U> class Box {` under `template <typename T,`,
    /// `}) export class C {` under `@Component({`, `Debug)] struct S {` under `#[derive(`), read
    /// by `read`, with the 0-based line the attribute opens on.
    ///
    /// The tail is the line's text up to the first point where its closers less its openers of
    /// one bracket pair ([`Self::attribute_brackets`]) are highest
    /// ([`AttributeBrackets::leading_tail_end`]); the rest of the line must be non-empty and read
    /// as something by `read` before the lines above are walked, up to the attribute's opening
    /// line ([`Self::attribute_opening_above`]), each line charged to the pass's budget
    /// ([`SearchBudget`]) and never above `floor`. `None` when the line is no code, opens with no
    /// such tail, `read` reads nothing, or the walk finds no attribute. O(n) in the line plus the
    /// k lines walked.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `i` - The 0-based line.
    /// * `floor` - The 0-based line the walk never goes above.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]); only its budget is spent.
    /// * `read` - What the rest of the line is read as (a construct's kind and name).
    ///
    /// # Returns
    ///
    /// The attribute's 0-based opening line and what `read` read, or `None`.
    pub(super) fn after_wrapped_attribute_tail<T>(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        floor: usize,
        search: &BraceSearch<'_>,
        read: impl Fn(&str) -> Option<T>,
    ) -> Option<(usize, T)> {
        let line = lines.get(i)?;
        if scanned.get(i).is_none_or(|s| s.is_non_code) {
            return None;
        }
        let above = i.checked_sub(1).filter(|&k| k >= floor)?;
        [']', ')', '>']
            .into_iter()
            .filter_map(|close| self.attribute_brackets(close))
            .find_map(|brackets| {
                let (end, unclosed) = brackets.leading_tail_end(line)?;
                let rest = line.get(end..)?.trim();
                let value = (!rest.is_empty()).then(|| read(rest)).flatten()?;
                let start = self.attribute_opening_above(
                    lines,
                    scanned,
                    floor..=above,
                    unclosed,
                    brackets,
                    || search.spend(),
                )?;
                Some((start, value))
            })
    }

    /// The name and last line (0-based) of the function expression the declaration at line `i`
    /// assigns (`const f = () => {`, `let g = async function (x) {`, an arrow whose parameters
    /// wrap over several lines): a `const`, `let` or `var` whose value is a `function` or an
    /// arrow function, its `=>` at the value's top level up to the block's opening brace
    /// ([`arrow_at_top_level`]), and whose value opens a block
    /// ([`declaration_block`]). O(k) for the k lines of its header and body.
    fn function_expression(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        search: &BraceSearch<'_>,
    ) -> Option<(String, usize)> {
        let text = self.strip_modifiers(lines.get(i)?.trim(), false);
        let rest = FUNCTION_EXPRESSION_KEYWORDS
            .iter()
            .find_map(|kw| blank_keyword_rest(text, kw))?;
        let name: String = rest
            .chars()
            .take_while(|&c| is_identifier_char(c))
            .collect();
        let value = rest.split_once('=')?.1.trim_start();
        let (open, close) = declaration_block(lines, scanned, i, search)?;
        let arrow = arrow_at_top_level(
            std::iter::once(value).chain(lines.get(i + 1..=open)?.iter().copied()),
        );
        let function = opens_function_value(value);
        (!name.is_empty() && (arrow || function)).then_some((name, close))
    }

    /// The last line, 1-based, of the function whose signature is at line `i`, read past a C++
    /// constructor's member initializers (`member_initializers`): when its header opens a member
    /// initializer list ([`opens_member_initializer_list`]), the block found at `open` may be an
    /// initializer's (`Foo::Foo() : arr{` ... `1, 2} {`), so while the line closing it opens
    /// another brace after its lowest point (`} {`, `}, b{`), or the next code line starts the
    /// next initializer or the body (`, b{`, `{`, Allman), the block opened there is searched
    /// too. Without a list, or in another language, `end` is kept. Each search starts past the
    /// last one's closing line, so the walk is linear in the constructor's lines; the lines
    /// between two blocks are charged to the pass's [`SearchBudget`].
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `i` - The 0-based signature line.
    /// * `open` - The 0-based line of the brace the search found, if any.
    /// * `end` - The function's last line as found from `open`, 1-based.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// The last line, 1-based, or `None` when it does not fit `u32`.
    // MAGI FIX (loop 29, B2b1a): a braced member initializer wrapped over lines ended the
    // constructor, its body falling into later units.
    fn past_member_initializers(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        open: Option<usize>,
        end: u32,
        search: &BraceSearch<'_>,
    ) -> Option<u32> {
        let Some(open) = open.filter(|_| self.member_initializers) else {
            return Some(end);
        };
        if !opens_member_initializer_list(lines.get(i..=open).unwrap_or(&[])) {
            return Some(end);
        }
        let Some(mut close) = (end as usize).checked_sub(1) else {
            return Some(end);
        };
        loop {
            let reopens = scanned
                .get(close)
                .is_some_and(|s| s.brace_delta > s.min_delta);
            let next_open = if reopens {
                Some(close)
            } else {
                self.next_initializer_or_body(lines, scanned, close + 1, search)
                    .and_then(|j| open_brace_line(lines, scanned, j, search))
            };
            let Some(next_open) = next_open else {
                break;
            };
            let next_close = find_closing_brace(scanned, next_open);
            if next_close <= close {
                break;
            }
            close = next_close;
        }
        line_number(close)
    }

    /// The first code line from `from` on (blank lines and comments passed over, each charged to
    /// the pass's budget) when it starts a constructor's next member initializer or its body
    /// (`, b{`, `{`); `None` for any other line, past the text's end or once the budget is spent.
    fn next_initializer_or_body(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        from: usize,
        search: &BraceSearch<'_>,
    ) -> Option<usize> {
        (from..lines.len())
            .take_while(|_| search.spend())
            .find(|&j| {
                scanned.get(j).is_some_and(|s| !s.is_non_code)
                    && lines.get(j).is_some_and(|l| !l.trim().is_empty())
            })
            .filter(|&j| {
                lines
                    .get(j)
                    .is_some_and(|l| l.trim_start().starts_with(['{', ',']))
            })
    }

    /// Whether a signature ending in `;` is a member of a block body: in Java, JavaScript and
    /// TypeScript an interface or abstract class declares its methods so (`User find(long id);`).
    /// C and C++ read such a line as a declaration instead (`semicolon_declarations`), which
    /// groups the prototypes of a class body. O(1).
    fn member_prototypes(&self) -> bool {
        self.signature_functions && !self.semicolon_declarations
    }

    /// Whether line `j` opens a C or C++ prototype wrapped over lines (`int foo(` ...
    /// `int b);`), for extractors with `semicolon_declarations`: a code line that names a
    /// function ([`Self::signature_name`]), leaves a parenthesis open in code context
    /// ([`ScannedLine::group_delta`]) and whose list closes on a line whose code ends in `;`
    /// ([`group_end`]). Such a line opens no declaration by [`Self::is_declaration`] (its code
    /// ends in `(` or in a parameter), so without this rule its first lines fell into a gap and
    /// the closing `);` read as a declaration of its own. O(k) for the k lines of the list, from
    /// the pass's [`SearchBudget`].
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `j` - The 0-based line asked about.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// `true` when line `j` opens a wrapped prototype.
    fn opens_wrapped_prototype(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        search: &BraceSearch<'_>,
    ) -> bool {
        let Some(record) = scanned.get(j).filter(|s| !s.is_non_code) else {
            return false;
        };
        if !self.semicolon_declarations
            || record.group_delta <= 0
            || !lines
                .get(j)
                .is_some_and(|l| self.signature_name(l, Some(record), false).is_some())
        {
            return false;
        }
        let close = group_end(lines, scanned, j, search);
        close > j && scanned.get(close).is_some_and(|s| s.ends_statement)
    }

    /// The 0-based first line of the construct whose signature is on line `i`: the line above
    /// it when that line holds only its return type, written GNU style on a line of its own
    /// (`static int` over `foo (int a)`), else `i`. Only for extractors with
    /// `semicolon_declarations` and outside a block body (`members`), where a line of type words
    /// alone (identifier characters, blanks, `*`, `&`, `<`, `>` and `::`, not ending in `:`, its
    /// angle brackets balanced) is no construct of its own: it fell into a gap unit, split from
    /// its function or prototype.
    /// Never above `floor`, the line after the previous construct. O(n) in the two lines.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `i` - The 0-based line of the signature.
    /// * `floor` - The lowest 0-based line the construct may start on.
    /// * `members` - Whether the lines are a block's body.
    ///
    /// # Returns
    ///
    /// `i - 1` when that line is the signature's return type, `i` otherwise.
    fn return_type_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        floor: usize,
        members: bool,
    ) -> usize {
        let Some(above) = i.checked_sub(1).filter(|&a| a >= floor) else {
            return i;
        };
        let code = |k: usize| {
            scanned
                .get(k)
                .is_some_and(|s| !s.is_non_code && !s.starts_in_literal)
        };
        // MAGI FIX (loop 26, 3): the last line of a wrapped template header (`typename U>`) read
        // as a return type, so the header's earlier lines were cut from the function; a return
        // type's angle brackets balance on its line.
        let type_words = lines.get(above).map(|l| l.trim()).is_some_and(|t| {
            !t.is_empty()
                && !t.ends_with(':')
                && t.chars()
                    .all(|c| is_identifier_char(c) || RETURN_TYPE_CHARS.contains(c))
                && t.matches('<').count() == t.matches('>').count()
        });
        let signature = lines
            .get(i)
            .is_some_and(|l| self.signature_name(l, scanned.get(i), true).is_some());
        if self.semicolon_declarations && !members && code(above) && type_words && signature {
            above
        } else {
            i
        }
    }

    /// Generate a descriptive name for a declaration group from its first line, with the
    /// visibility words already stripped ([`Self::strip_modifiers`]): only bare keywords can
    /// match here, so `pub use` reads as `use` and a JavaScript `export { a };` has no name of
    /// its own (`declarations`).
    fn declaration_group_name(&self, first_line: &str) -> String {
        // MAGI FIX (S6 loop 12): the `pub ...` and `export ` arms could never match, since the
        // caller strips the visibility words first.
        let trimmed = first_line.trim();
        let opens = |kw: &str| declaration_keyword_rest(trimmed, kw).is_some();
        if opens("use ") || opens("import ") || opens("from ") {
            "imports".to_string()
        } else if opens("const ") {
            "constants".to_string()
        } else if opens("static ") {
            "statics".to_string()
        } else if opens("#define ") {
            "defines".to_string()
        } else if opens("typedef ") {
            "typedefs".to_string()
        } else if opens("extern ") {
            "externs".to_string()
        } else if opens("type ") {
            "type aliases".to_string()
        } else if opens("let ") || opens("var ") {
            "variables".to_string()
        } else if opens("package ") {
            "package".to_string()
        } else {
            "declarations".to_string()
        }
    }
}

/// The most blank lines that may stand between an attribute and the construct it governs
/// (`@Override`, blank, `public void f() {`) for the attribute to stay in the construct's unit
/// ([`BraceFamilyExtractor::scan_back_for_docs`]). A starting point, not a measurement: one or
/// two blank lines are a style slip; more read as a separation, and the attribute then stays a
/// gap unit, as before, with no line lost.
const MAX_BLANK_LINES_AFTER_ATTRIBUTE: usize = 2;

/// The characters besides identifier ones that a GNU-style return type line may hold
/// (`static const char *`, `std::vector<int> &`), read by
/// [`BraceFamilyExtractor::return_type_start`].
const RETURN_TYPE_CHARS: &str = " 	*&<>:";
