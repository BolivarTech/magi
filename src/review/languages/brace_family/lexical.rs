// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Free text helpers over one line or signature: identifiers, operator and declarator
//! names, keyword and modifier stripping, C and Rust attribute skipping, and construct naming.

use super::*;

/// The 1-based line number of the 0-based line `idx`.
///
/// # Arguments
///
/// * `idx` - A 0-based line index.
///
/// # Returns
///
/// `idx + 1`, or `None` when it does not fit `u32`. O(1).
pub(super) fn line_number(idx: usize) -> Option<u32> {
    u32::try_from(idx.checked_add(1)?).ok()
}

/// Whether `c` can be part of an identifier in a brace-family language: a letter or digit
/// (Unicode included), `_`, or `$` (JavaScript and TypeScript; GCC accepts it in C too).
pub(super) fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

/// The identifier that names the function `line` declares, read just before the `(` that opens
/// its parameter list ([`parameter_list_head`]; `int *parse(` gives `parse`,
/// `public static <T> List<T> wrap(` gives `wrap`, `decltype(auto) f(` gives `f`,
/// `std::function<void(int)> make(` gives `make`), or inside the declarator in parentheses of a
/// function returning a function pointer (`void (*get_handler(int sig))(int)` gives
/// `get_handler`).
///
/// A C++ operator overload is named before this rule applies ([`operator_overload_name`]).
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// `None` when the line has no `(`, holds `=` before it (an assignment or initializer; the `=` of
/// an arrow in a type bound does not count, `ListHead::assigns`), or no identifier ends right
/// before it. O(n) in the line.
pub(super) fn identifier_before_paren(line: &str) -> Option<String> {
    // MAGI FIX (S6 loop 15): `decltype(auto) f() {` was named `decltype`, its operand's `(`
    // being taken for the parameter list; the list's own `(` is looked for instead.
    let head = parameter_list_head(line)?;
    if head.assigns() {
        return None;
    }
    // MAGI FIX (S6 loop 6): `$get() {` named no member, so a class could not split there.
    // MAGI FIX (S6 loop 18, B2b-4): `foo<T>(x: T) {` ended in `>`, so no identifier named it
    // and a generic TypeScript method was no member; its type parameters are read past.
    Some(trailing_identifier(without_type_parameters(head.name_text)))
        .filter(|n| !n.is_empty())
        .map(str::to_string)
}

/// The name of a TypeScript optional member, its `?` glued between the name and the parameter
/// list ([`parameter_list_head`]; `foo?(x: number): void;` and `qux?<T>(x: T): T;` give `foo`
/// and `qux`), without the `?`. A blank before the `?` (`a ?(b)`, a conditional) names nothing,
/// nor does an `=` before the `(` (an assignment). O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line, trimmed.
///
/// # Returns
///
/// The member's name, or `None`.
pub(super) fn optional_member_name(line: &str) -> Option<String> {
    let head = parameter_list_head(line)?;
    if head.assigns() {
        return None;
    }
    let name = without_type_parameters(head.name_text).strip_suffix('?')?;
    let ident = trailing_identifier(name);
    (!ident.is_empty() && name.ends_with(is_identifier_char)).then(|| ident.to_string())
}

/// The name of a JavaScript or TypeScript member whose key is computed or private, as written
/// before the `(` of its parameter list ([`parameter_list_head`]): a computed key in balanced
/// square brackets (`*[Symbol.iterator]() {` gives `[Symbol.iterator]`) or an identifier right
/// after a `#` (`static #s() {` gives `#s`). `None` for any other line, which
/// [`identifier_before_paren`] then names, and when an `=` comes before the `(` (an assignment).
/// O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line, trimmed.
///
/// # Returns
///
/// The member's name, or `None`.
pub(super) fn private_or_computed_name(line: &str) -> Option<String> {
    let head = parameter_list_head(line)?;
    if head.assigns() {
        return None;
    }
    let text = without_type_parameters(head.name_text);
    if text.ends_with(']') {
        let open = matching_open(text, '[', ']')?;
        return text.get(open..).map(str::to_string);
    }
    let ident = trailing_identifier(text);
    let before = text.get(..text.len().checked_sub(ident.len())?)?;
    (!ident.is_empty() && before.ends_with('#')).then(|| format!("#{ident}"))
}

/// `text` without the type parameter list it ends with (`foo<T>` gives `foo`,
/// `[key]<K, V>` gives `[key]`), blanks at its end passed over; `text` trimmed at its end when it
/// ends in no `>` or the list never opens. A generic method (`foo<T>(x: T) {`) and a C++
/// explicit specialization (`void f<int>(int x) {`) are named so before their `(`. O(n) in the
/// text.
///
/// # Arguments
///
/// * `text` - The text of a signature line before its parameter list.
///
/// # Returns
///
/// The text before the type parameter list.
fn without_type_parameters(text: &str) -> &str {
    let text = text.trim_end();
    if !text.ends_with('>') {
        return text;
    }
    matching_open(text, '<', '>')
        .and_then(|at| text.get(..at))
        .map_or(text, str::trim_end)
}

/// The characters whose `>` after them is an arrow, never an angle bracket: `->` (a Rust or C++
/// return type, `Fn() -> U`) and `=>` (a TypeScript function type, `() => void`).
const ARROW_HEADS: [char; 2] = ['-', '='];

/// Whether the `>` at byte offset `at` of `text` closes an angle bracket, not an arrow
/// ([`ARROW_HEADS`]). O(1).
fn closes_angle(text: &str, at: usize) -> bool {
    !text
        .get(..at)
        .and_then(|before| before.chars().next_back())
        .is_some_and(|p| ARROW_HEADS.contains(&p))
}

/// The byte offset of the `open` that matches the `close` `text` ends with (`[` for `]`, `<` for
/// `>`), counting the pairs between, or `None` when it never opens. For `>`, the head of an arrow
/// (`() => void`, `Fn() -> U`) is no close ([`closes_angle`]). O(n) in the text.
fn matching_open(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0u32;
    for (at, c) in text.char_indices().rev() {
        match c {
            // MAGI FIX (S6 loop 25, B2b1b): the `>` of `=>` in `foo<T extends () => void>`
            // closed a bracket that never opened, so the generic method named no member.
            '>' if c == close && !closes_angle(text, at) => {}
            c if c == close => depth = depth.saturating_add(1),
            c if c == open => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(at);
                }
            }
            _ => {}
        }
    }
    None
}

/// The words of a C++ return type whose operand is parenthesized (`decltype(auto)`,
/// `decltype(a + b)`): the `(` right after one opens no parameter list.
const TYPE_OPERATORS: [&str; 1] = ["decltype"];

/// The characters a declarator in parentheses opens with (`(*f)`, `(&r)`, a block's `(^b)`).
const DECLARATOR_OPENERS: [char; 3] = ['*', '&', '^'];

/// Where a signature line's parameter list starts ([`parameter_list_head`]).
struct ListHead<'t> {
    /// The line's text before the `(` the head was found at, for the `=` rule.
    before: &'t str,
    /// The text whose last identifier names the function: `before` itself, or the inside of a
    /// declarator in parentheses up to its own `(`.
    name_text: &'t str,
}

impl<'t> ListHead<'t> {
    /// A head named by the identifier just before its `(`. O(1).
    fn at(before: &'t str) -> Self {
        Self {
            before,
            name_text: before,
        }
    }

    /// Whether the text before the `(` holds an `=`, an assignment or initializer (`x = f(`),
    /// other than the `=` of an arrow (`foo<T extends () => void>(`), which a type bound holds,
    /// and an `=` inside angle brackets, a type parameter's default (`get<T = unknown>(`). The
    /// `>` of an arrow closes no bracket ([`closes_angle`]). O(n) in the text.
    // MAGI FIX (S6 loop 25, B2b1b): the arrow of a bound read as an assignment, so a generic
    // method bounded by a function type named no member.
    // MAGI FIX (loop 26, 5): the `=` of a generic default read as an assignment, so the method
    // was no member of its class.
    fn assigns(&self) -> bool {
        let mut angles = 0usize;
        self.before.char_indices().any(|(at, c)| match c {
            '<' => {
                angles += 1;
                false
            }
            '>' if closes_angle(self.before, at) => {
                angles = angles.saturating_sub(1);
                false
            }
            '=' => {
                angles == 0
                    && !self
                        .before
                        .get(at + 1..)
                        .is_some_and(|r| r.starts_with('>'))
            }
            _ => false,
        })
    }
}

/// The head of the parameter list `line` opens, read past what a return type may put in
/// parentheses before it:
///
/// * a `(` right after a word of [`TYPE_OPERATORS`] is that word's operand, read past up to its
///   matching `)` (`decltype(auto) f(`);
/// * a `(` inside angle brackets is a template argument, read past the same way
///   (`std::function<void(int)> make(`); the `>` of an arrow (`->`, `=>`) closes none
///   ([`closes_angle`]), so `foo<T extends () => void>(` is read past its bound;
/// * a `(` after a return type word, opening with a [`DECLARATOR_OPENERS`] character and followed
///   by another `(` once it closes, is a declarator in parentheses: the function is named
///   inside it (`void (*f)()`, `void (*get_handler(int sig))(int)`). A keyword of
///   [`NOT_FUNCTION_NAMES`] before it (`while (*s)(x)`) is no return type.
///
/// When a `<` is left open (a comparison, `a < f(x)`, or an operator, `operator<(`), the line's
/// first `(` is taken, as before this rule, so nothing it named before is lost; an operand that
/// never closes on the line leaves its `(` as the list's, so the line is named by the type word,
/// which [`NOT_FUNCTION_NAMES`] rules out. A heuristic on one line's text: a parenthesis inside a
/// string of the return type miscounts. O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// The head, or `None` when the line holds no `(` that opens a parameter list.
fn parameter_list_head(line: &str) -> Option<ListHead<'_>> {
    // MAGI FIX (S6 loop 15): `std::function<void(int)> make() {` and `void (*f)() {` were
    // named `void`, the first `(` being taken for the parameter list.
    let first = || line.split_once('(').map(|(before, _)| ListHead::at(before));
    let mut from = 0;
    let mut angles = 0usize;
    loop {
        let Some((offset, c)) = line
            .get(from..)?
            .char_indices()
            .find(|&(_, c)| matches!(c, '(' | '<' | '>'))
        else {
            return if angles > 0 { first() } else { None };
        };
        let at = from + offset;
        if c != '(' {
            angles = if c == '<' {
                angles + 1
            } else if closes_angle(line, at) {
                angles.saturating_sub(1)
            } else {
                angles
            };
            from = at + c.len_utf8();
            continue;
        }
        let before = line.get(..at)?;
        let group_end = line
            .get(at..)
            .and_then(|t| balanced_end(t, '(', ')'))
            .map(|end| at + end);
        let word = trailing_identifier(before);
        let read_past = angles > 0 || TYPE_OPERATORS.contains(&word);
        match group_end {
            Some(end) if read_past => from = end,
            None if angles > 0 => return first(),
            None => return Some(ListHead::at(before)),
            Some(end) => {
                let name_text = declarator_name_text(line, at, end, word).unwrap_or(before);
                return Some(ListHead { before, name_text });
            }
        }
    }
}

/// The inside of the declarator in parentheses that spans `line[open..end]`, up to its own `(`
/// (`*get_handler` in `void (*get_handler(int sig))(int)`, `*f` in `void (*f)()`), when it is
/// one: `word`, the identifier before it, is a return type (not empty, not a keyword of
/// [`NOT_FUNCTION_NAMES`]), its inside opens with a [`DECLARATOR_OPENERS`] character, and a `(`
/// follows its `)`. O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line.
/// * `open` - The byte offset of the group's `(`.
/// * `end` - The byte offset just past its matching `)`.
/// * `word` - The identifier just before the `(`.
///
/// # Returns
///
/// The declarator's text, or `None` when the group is no declarator.
fn declarator_name_text<'t>(line: &'t str, open: usize, end: usize, word: &str) -> Option<&'t str> {
    if word.is_empty() || NOT_FUNCTION_NAMES.contains(&word) {
        return None;
    }
    let inner = line.get(open + 1..end.checked_sub(1)?)?.trim_start();
    let followed = line.get(end..)?.trim_start().starts_with('(');
    if !(followed && inner.starts_with(DECLARATOR_OPENERS)) {
        return None;
    }
    Some(inner.split_once('(').map_or(inner, |(name, _)| name))
}

/// The identifier `text` ends with, blanks at its end passed over (`int *parse ` gives
/// `parse`); empty when it ends in no identifier character ([`is_identifier_char`]). O(n) in the
/// text.
fn trailing_identifier(text: &str) -> &str {
    let text = text.trim_end();
    let start = text
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_identifier_char(c))
        .last()
        .map_or(text.len(), |(at, _)| at);
    text.get(start..).unwrap_or("")
}

/// The characters a C++ overloaded operator's symbol is made of (`=`, `==`, `<<=`, `[]`).
const OPERATOR_SYMBOL_CHARS: &str = "=!<>+-*/%&|^~[],";

/// The name of the C++ operator overload `before` declares, when it ends in the word `operator`
/// and a symbol, blanks between them or inside it passed over (`Foo& operator=` gives
/// `operator=`, `bool operator ==` gives `operator==`, `int operator [ ]` gives `operator[]`), or
/// in the word `operator`, a blank and an allocation operator of [`ALLOCATION_OPERATORS`]
/// (`void* operator new` gives `operator new`, `void operator delete [ ]` gives
/// `operator delete[]`).
///
/// # Arguments
///
/// * `before` - The text of a line before its first `(`.
///
/// # Returns
///
/// The operator's name, or `None` when `before` names no such operator (a call operator,
/// `operator()`, has its symbol after the first `(` and is named before this rule
/// ([`call_operator`]); a conversion operator, `operator bool`, is named by
/// [`identifier_before_paren`]'s general rule, after its type). O(n) in the text.
pub(super) fn operator_name(before: &str) -> Option<String> {
    let (head, symbol) = before.trim_end().rsplit_once("operator")?;
    let word_starts = head
        .chars()
        .next_back()
        .is_none_or(|c| !is_identifier_char(c));
    if !word_starts {
        return None;
    }
    // MAGI FIX (S6 loop 18, B2b-7): the symbol had to touch the word, so `bool operator <(` was
    // no function; only C++ asks ([`operator_overload_name`]), where the blanks mean nothing.
    let word: String = symbol.chars().filter(|c| !c.is_whitespace()).collect();
    if !word.is_empty() && word.chars().all(|c| OPERATOR_SYMBOL_CHARS.contains(c)) {
        return Some(format!("operator{word}"));
    }
    // MAGI FIX (S6 loop 14): `void* operator new(size_t n) {` reached `NOT_FUNCTION_NAMES` with
    // the word `new`, so the allocator was no function and its body fell into a gap unit.
    let allocation =
        symbol.starts_with(char::is_whitespace) && ALLOCATION_OPERATORS.contains(&word.as_str());
    allocation.then(|| format!("operator {word}"))
}

/// The name of the C++ operator overload `line` declares: a call operator ([`call_operator`],
/// `operator()`) or an operator named before the `(` of its parameter list ([`operator_name`],
/// `operator=`, `operator <`, `operator new`). Asked only for an extractor with
/// `operator_overloads` (C++), where `operator` is a keyword: elsewhere it is an identifier
/// like any other, so a JavaScript `const operator = (x) => {` names no overload. O(n × k) in
/// the line, for k occurrences of the word.
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// The operator's name, or `None` when the line declares no operator overload.
// MAGI FIX (S6 loop 18, B2b-7): moved out of `identifier_before_paren`, so the spaced forms
// are read in C++ only.
pub(super) fn operator_overload_name(line: &str) -> Option<String> {
    // MAGI FIX (S6 loop 17, B2-5): `bool operator()(int x) {` was named `operator`: its symbol is
    // the line's first `()`, taken for the parameter list.
    if call_operator(line) {
        return Some(CALL_OPERATOR_NAME.to_string());
    }
    // MAGI FIX (S6 loop 9): `Foo& operator=(` held an `=` before its `(` and named nothing.
    operator_name(parameter_list_head(line)?.name_text)
}

/// The name of a C++ call operator overload ([`call_operator`]).
const CALL_OPERATOR_NAME: &str = "operator()";

/// Whether `line` declares a C++ call operator overload (`bool operator()(int x) const {`,
/// `auto operator ( ) (int x) {`): the word `operator` (no identifier character before it), then
/// after optional blanks an empty `()`, then after optional blanks the `(` of the parameter list.
/// Its symbol is the line's first `()`, so the general rule ([`parameter_list_head`]) takes it
/// for the parameter list and names the function `operator`. O(n × k) in the line, for k
/// occurrences of the word.
fn call_operator(line: &str) -> bool {
    line.match_indices("operator").any(|(at, word)| {
        let word_starts = line
            .get(..at)
            .and_then(|before| before.chars().next_back())
            .is_none_or(|c| !is_identifier_char(c));
        let after = line.get(at + word.len()..).unwrap_or("").trim_start();
        let parameters = after
            .strip_prefix('(')
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix(')'))
            .map(str::trim_start);
        word_starts && parameters.is_some_and(|rest| rest.starts_with('('))
    })
}

/// The C++ allocation operators, whose symbol is a word with an optional `[]`
/// (`operator new`, `operator delete[]`), named with one space after `operator` and the
/// brackets glued to the word, however they were spaced on the line.
const ALLOCATION_OPERATORS: [&str; 4] = ["new", "new[]", "delete", "delete[]"];

/// `text` without the modifier `word` it opens with, and what belongs to the word: a `(...)`
/// right after it (`pub(crate)`) or, after blanks, a quoted string (`extern "C"`); the rest is
/// trimmed at its start. `None` when `text` does not open with `word` followed by a blank or a
/// `(`. O(n) in the text.
///
/// Blanks may stand before the `(` only when it holds a Rust visibility path
/// ([`is_visibility_path`]: `pub (crate)`, `pub (in a::b)`), which rustfmt never writes but the
/// compiler accepts; any other spaced `(` is no part of the word (`async (x) => {`).
pub(super) fn strip_modifier<'t>(text: &'t str, word: &str) -> Option<&'t str> {
    let rest = text.strip_prefix(word)?;
    if let Some(inner) = rest.strip_prefix('(') {
        let (_, after) = inner.split_once(')')?;
        return Some(after.trim_start());
    }
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    // MAGI FIX (loop 26, 6): `pub (crate) fn f() {` kept its `(crate)`, so the item was no
    // construct and fell into a gap unit.
    if let Some((path, after)) = rest.strip_prefix('(').and_then(|r| r.split_once(')')) {
        if is_visibility_path(path) {
            return Some(after.trim_start());
        }
    }
    match rest.strip_prefix('"').and_then(|r| r.split_once('"')) {
        Some((_, after)) => Some(after.trim_start()),
        None => Some(rest),
    }
}

/// The paths a Rust visibility restricts an item to, inside its parentheses (`pub(crate)`);
/// `in` opens an explicit path (`pub(in a::b)`).
const VISIBILITY_PATHS: [&str; 3] = ["crate", "self", "super"];

/// The word that opens an explicit Rust visibility path (`pub(in a::b)`).
const VISIBILITY_IN_KEYWORD: &str = "in";

/// Whether `path`, the inside of a parenthesized group after a modifier, is a Rust visibility
/// path: a word of [`VISIBILITY_PATHS`] or [`VISIBILITY_IN_KEYWORD`] followed by a blank and a
/// path. O(n) in the text.
///
/// # Arguments
///
/// * `path` - The text between the parentheses.
///
/// # Returns
///
/// `true` when the group restricts a visibility.
fn is_visibility_path(path: &str) -> bool {
    let path = path.trim();
    VISIBILITY_PATHS.contains(&path)
        || path
            .strip_prefix(VISIBILITY_IN_KEYWORD)
            .is_some_and(|rest| rest.starts_with(char::is_whitespace) && !rest.trim().is_empty())
}

/// The text after keyword `kw` at the start of `text`. A table keyword ends in a blank
/// (`impl `), and the word may also be followed directly by a generic parameter list
/// (`impl<T: Clone> Foo<T>`), which is passed over, or by the brace of an anonymous construct
/// (`class{`), which is kept. The keyword may also be the whole line (`union`, its brace on the
/// next line, Allman style); the rest is then empty and the construct is named `unnamed`
/// ([`construct_name`]). `None` when `text` does not open with the keyword. O(n) in the text.
pub(super) fn keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    if let Some(rest) = blank_keyword_rest(text, kw) {
        return Some(rest);
    }
    // MAGI FIX (S6 loop 5): `impl<T> Foo<T>` matched no `impl ` and its methods lost the header.
    let word = kw.strip_suffix(' ')?;
    let rest = text.strip_prefix(word)?;
    // MAGI FIX (S6 loop 6): `export default class{` matched no `class ` and was no class.
    // MAGI FIX (S6 loop 14): a keyword alone on its line (`union`, then `{` on the next) was no
    // construct, so an anonymous type's variable fell into gap units around its declarator.
    if rest.is_empty() || rest.starts_with('{') {
        return Some(rest);
    }
    rest.starts_with('<').then(|| skip_generics(rest))
}

/// The text after keyword `kw` at the start of `text`: `text` past `kw` as written or, for a
/// table keyword that ends in a blank (`fn `, `const `), past the bare word and any run of
/// blanks after it (`fn\tfoo`, `const  X`), trimmed at its start. `None` when `text` does not
/// open with the keyword followed by a blank (`constant`). O(n) in the text.
///
/// # Arguments
///
/// * `text` - A trimmed line, or its rest after the modifiers.
/// * `kw` - A keyword of a table.
///
/// # Returns
///
/// The text after the keyword, or `None`.
pub(super) fn blank_keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    // MAGI FIX (S6 loop 13): a tab after the keyword (`fn\tfoo`) matched no `fn `.
    // MAGI FIX (S6 loop 14): a keyword matched as written kept the blanks after it, so
    // `const  f = () => {` named an empty function and `enum  class Color` read as `class`.
    if let Some(rest) = text.strip_prefix(kw) {
        return Some(rest.trim_start());
    }
    let rest = text.strip_prefix(kw.strip_suffix(' ')?)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

/// What may follow a declaration keyword's bare word with no blank between and still open the
/// declaration: a list (`use::{a}`, `import{foo}`, `export{foo}`), a path (`use::a`), an included
/// file (`#include<stdio.h>`, `#include"a.h"`), a quoted module or linkage (`import'./m'`,
/// `extern"C"`) or a namespace import (`import*as m`). An identifier character never does, so a
/// longer word (`usefoo`, `importer`) is no keyword.
const GLUED_DECLARATION_OPENERS: [&str; 6] = ["{", "::", "<", "\"", "'", "*"];

/// The text after declaration keyword `kw` at the start of `text`: as [`blank_keyword_rest`],
/// or, when the keyword's bare word is followed directly by one of
/// [`GLUED_DECLARATION_OPENERS`], the text from that opener on. O(n) in the text.
///
/// # Arguments
///
/// * `text` - A trimmed line, or its rest after the modifiers.
/// * `kw` - A declaration keyword of a table, ending in a blank.
///
/// # Returns
///
/// The text after the keyword, or `None` when `text` does not open with it.
pub(super) fn declaration_keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    if let Some(rest) = blank_keyword_rest(text, kw) {
        return Some(rest);
    }
    let rest = text.strip_prefix(kw.strip_suffix(' ')?)?;
    GLUED_DECLARATION_OPENERS
        .iter()
        .any(|opener| rest.starts_with(opener))
        .then_some(rest)
}

/// `text` after the balanced `<...>` it opens with, trimmed at its start; empty when the list
/// never closes. The `>` of an arrow (`impl<T: Fn(u8) -> u8>`, `<T extends () => void>`) closes
/// nothing ([`closes_angle`]). O(n) in the text.
pub(super) fn skip_generics(text: &str) -> &str {
    let mut depth = 0u32;
    for (i, c) in text.char_indices() {
        match c {
            '<' => depth = depth.saturating_add(1),
            // MAGI FIX (S6 loop 6): the `>` of a `->` in a bound closed the list early.
            // MAGI FIX (S6 loop 25, B2b1b): so did the `>` of a TypeScript `=>`.
            '>' if !closes_angle(text, i) => {}
            '>' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return text.get(i + c.len_utf8()..).unwrap_or("").trim_start();
                }
            }
            _ => {}
        }
    }
    ""
}

/// `text` after the C++ template header it opens with: the word `template` followed, after any
/// blanks, by a balanced `<...>` ([`skip_generics`]), trimmed at its start; empty when the
/// header is the whole line or never closes on it. `text` when it opens with no such header
/// (`templated()`, `template class Foo<int>;`). O(n) in the text.
///
/// A header wrapped over several lines (`template <typename T,` then `typename U>`) leaves this
/// empty on its first line; detection passes over its lines and recovers them for the construct
/// they declare (`BraceFamilyExtractor::wrapped_template_header_end`).
pub(super) fn strip_template_header(text: &str) -> &str {
    match text.strip_prefix(TEMPLATE_KEYWORD).map(str::trim_start) {
        Some(rest) if rest.starts_with('<') => skip_generics(rest),
        _ => text,
    }
}

/// Whether `text` opens with a C++ template header: the word `template` followed, after any
/// blanks, by `<` (`template <typename T,`, `template<int N>`). O(n) in the text.
///
/// # Arguments
///
/// * `text` - A trimmed line.
///
/// # Returns
///
/// `true` when the line opens a template header, closed on it or not.
pub(super) fn opens_template_header(text: &str) -> bool {
    text.strip_prefix(TEMPLATE_KEYWORD)
        .map(str::trim_start)
        .is_some_and(|rest| rest.starts_with('<'))
}

/// The Go keyword that declares an interface type (`type Reader interface {`).
const GO_INTERFACE_KEYWORD: &str = "interface";

/// The kind of a Go type declaration from the text after `type` (or a member line of a
/// `type (` group): its name, an optional type parameter list in brackets (`Box[T any]`), then
/// the keyword [`GO_INTERFACE_KEYWORD`] makes a trait, anything else (a struct, a defined type,
/// an alias) a struct. A Go interface's methods carry no keyword, so it is reviewed whole, never
/// split at them. O(n) in the text.
pub(super) fn go_type_kind(rest: &str) -> UnitKind {
    let rest = rest.trim_start();
    let after_name = rest.trim_start_matches(is_identifier_char).trim_start();
    let after_params = if after_name.starts_with('[') {
        balanced_end(after_name, '[', ']')
            .and_then(|end| after_name.get(end..))
            .unwrap_or("")
            .trim_start()
    } else {
        after_name
    };
    let interface = after_params
        .strip_prefix(GO_INTERFACE_KEYWORD)
        .is_some_and(|after| !after.starts_with(is_identifier_char));
    if interface {
        UnitKind::Trait
    } else {
        UnitKind::Struct
    }
}

/// The C++ keyword that opens a template header (`template <typename T>`).
const TEMPLATE_KEYWORD: &str = "template";

/// The C and C++ attribute words whose argument list in parentheses is no call: GCC's
/// `__attribute__((packed))`, MSVC's `__declspec(dllexport)`, and the alignment specifiers
/// `alignas(16)` (C++11, C23) and `_Alignas(16)` (C11).
const C_ATTRIBUTE_WORDS: [&str; 4] = ["__attribute__", "__declspec", "alignas", "_Alignas"];

/// What opens a C or C++ standard attribute (`[[nodiscard]]`, C++11 and C23). No other C or
/// C++ token opens with it, so a line that does is an attribute line.
const STANDARD_ATTRIBUTE_OPEN: &str = "[[";

/// The line prefixes of the C and C++ extractors' `attribute_prefixes`: an attribute alone on
/// the line above a construct (`[[nodiscard]]`, `__attribute__((noreturn))`) or wrapped over
/// lines (`[[deprecated(` ... `)]]`) qualifies that construct and is reviewed with it, as one on
/// the construct's own line is read past ([`strip_c_attributes`]). O(1).
pub(super) fn c_attribute_prefixes() -> Vec<String> {
    // MAGI FIX (loop 27, B2b1a): an attribute line above a C or C++ function was a gap unit,
    // reviewed apart from the function it qualifies.
    std::iter::once(STANDARD_ATTRIBUTE_OPEN)
        .chain(C_ATTRIBUTE_WORDS)
        .map(str::to_string)
        .collect()
}

/// `text` without its C and C++ attributes, wherever they stand on the line: a standard
/// attribute (`[[nodiscard]]`, `[[gnu::packed]]`) and a word of [`C_ATTRIBUTE_WORDS`] with its
/// argument list in balanced parentheses. The blanks are collapsed to one space, so the
/// keyword and name read as without them (`struct __attribute__((packed)) Hdr {` reads
/// `struct Hdr {`). An attribute that does not close on the line is kept with the rest of the
/// line, so the line reads as before. The text is only matched, never reviewed: the unit keeps
/// the line as written. Strings are not read, declared: a `[[...]]` inside one
/// (`const char *s = "[[x]]";`) is stripped from the matched text as well, which leaves its
/// keyword, its name and its ending `;` or `{` as they were, and the scanner's records, which
/// read strings, still decide every brace. O(n × k) in the line, for k attribute words.
///
/// # Arguments
///
/// * `text` - A trimmed line.
///
/// # Returns
///
/// The line without its attributes, borrowed when it holds none.
pub(super) fn strip_c_attributes(text: &str) -> Cow<'_, str> {
    if !text.contains("[[") && !C_ATTRIBUTE_WORDS.iter().any(|w| text.contains(w)) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some((at, end)) = next_c_attribute(rest) {
        out.push_str(rest.get(..at).unwrap_or(""));
        out.push(' ');
        rest = rest.get(end..).unwrap_or("");
    }
    out.push_str(rest);
    Cow::Owned(out.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// The byte span (`start`, `end`) of the first complete C or C++ attribute in `text`
/// ([`strip_c_attributes`]), or `None` when there is none: an attribute word starts only where
/// no identifier character comes before it, and is followed, after any blanks, by a `(` whose
/// match closes on the line; a `[[` is matched by its closing `]]`. O(n × k) in the text.
fn next_c_attribute(text: &str) -> Option<(usize, usize)> {
    let mut previous: Option<char> = None;
    for (at, c) in text.char_indices() {
        let word_starts = previous.is_none_or(|p| !is_identifier_char(p));
        previous = Some(c);
        let rest = text.get(at..).unwrap_or("");
        if rest.starts_with("[[") {
            return balanced_end(rest, '[', ']').map(|end| (at, at + end));
        }
        if !word_starts {
            continue;
        }
        let Some(word) = C_ATTRIBUTE_WORDS.iter().find(|w| {
            rest.strip_prefix(**w)
                .is_some_and(|after| !after.starts_with(is_identifier_char))
        }) else {
            continue;
        };
        let after = rest.get(word.len()..).unwrap_or("");
        let args = after.trim_start();
        if !args.starts_with('(') {
            continue;
        }
        let offset = at + word.len() + (after.len() - args.len());
        return balanced_end(args, '(', ')').map(|end| (at, offset + end));
    }
    None
}

/// The byte offset just past the `close` that matches the `open` `text` starts with, or `None`
/// when it never closes. Every character counts, strings included. O(n) in the text.
fn balanced_end(text: &str, open: char, close: char) -> Option<usize> {
    balanced_end_outside_quotes(text, open, close, &[])
}

/// The quotes an annotation's arguments are read past ([`BraceFamilyExtractor::without_annotations`]):
/// Java's strings and character literals, and JavaScript's and TypeScript's three string forms.
/// A backtick is no Java token, so reading it as a quote there changes no valid line.
pub(super) const ANNOTATION_QUOTES: [char; 3] = ['"', '\'', '`'];

/// The keyword that opens a Rust macro-by-example definition (`macro_rules! name {`). It is in
/// the Rust function table: a macro is reviewed as one function-like unit, its token tree opaque,
/// so an item written inside it (`fn generated() {}`) is never a construct of its own, and it is
/// never split (`split_block` splits only blocks of items). Its end is the later of its brace
/// block and its own closing delimiter ([`group_end`]), so the `(` ... `);` and `[` ... `];`
/// forms end on their last line too. A macro invoked at the top level (`thread_local! {`) is no
/// definition and is read as before.
pub(super) const MACRO_RULES_KEYWORD: &str = "macro_rules! ";

/// The opening of a Rust outer attribute (`#[inline]`).
///
/// An inner attribute (`#![allow(dead_code)]`) does not open with it, on purpose: the Rust
/// reference applies an inner attribute to the item that contains it (the crate, a module, a
/// function), never to the item written after it, so joining it to that item
/// ([`BraceFamilyExtractor::scan_back_for_docs`]) would review it as part of code it does not
/// govern. It stays with its container instead: inside a module or `impl` body it falls into the
/// block's header piece when the block splits ([`BraceFamilyExtractor::split_block`]), and at the
/// top of a file it is a gap unit of its own. Either way its line lands in exactly one unit, so
/// none is lost. Pinned by `a_rust_inner_attribute_stays_with_its_container_and_no_line_is_lost`.
/// An inner doc comment (`//!`, `/*!`) is still joined to the item below it (it is one of the
/// Rust `doc_comment_prefixes`), a known difference, declared.
// MAGI FIX (S6 loop 17, B2-6): the inner-attribute policy, written where the prefix is defined.
pub(super) const RUST_ATTRIBUTE_PREFIX: &str = "#[";

/// The prefix letter of a Rust raw string (`r"..."`, `r#"..."#`), which a byte string's `b`
/// may precede (`br#"..."#`).
const RAW_STRING_PREFIX: char = 'r';

/// The letter that makes a Rust string a byte string (`b"..."`, `br"..."`).
const BYTE_STRING_PREFIX: char = 'b';

/// The byte offset just past the `]` that matches the `[` a Rust attribute's `text` starts with
/// (`[doc = "]"]` closes at its last `]`), or `None` when it never closes on the line.
///
/// A `"` string is read past, a `\` escaping the next character in it, and so is a raw string
/// (`r"..."`, `r#"a"b"#`, `br"..."`), which ends only at a `"` followed by as many `#` as it
/// opened with, so a quote or bracket inside one counts nothing. A `'` is no quote, since it may
/// open a lifetime (`T: 'a`), which has no closing quote. An `r` inside an identifier
/// (`for"x"`) opens no raw string. A string that never closes makes the result `None`, and the
/// caller keeps its text as it was. O(n) in the text: each character is read once.
///
/// # Arguments
///
/// * `text` - The attribute's text from its `[`.
///
/// # Returns
///
/// The offset just past the matching `]`, or `None`.
// MAGI FIX (S6 loop 25, B2b1a): a raw string holding a `"` left the attribute unclosed, so the
// item after it on the line was no construct.
pub(super) fn rust_attribute_end(text: &str) -> Option<usize> {
    let starts_word = |p: Option<char>| p.is_none_or(|p| !is_identifier_char(p));
    let mut depth = 0u32;
    let mut at = 0;
    // The two characters before `at`, nearest first: whether an `r` starts a word.
    let mut previous: [Option<char>; 2] = [None, None];
    while let Some(c) = text.get(at..).and_then(|rest| rest.chars().next()) {
        let after = at + c.len_utf8();
        let raw_prefix = c == RAW_STRING_PREFIX
            && (starts_word(previous[0])
                || (previous[0] == Some(BYTE_STRING_PREFIX) && starts_word(previous[1])));
        if let Some(end) = raw_prefix.then(|| raw_string_end(text, after)).flatten() {
            at = end?;
            previous = [Some('"'), None];
            continue;
        }
        match c {
            '"' => {
                at = quoted_end(text, after)?;
                previous = [Some('"'), None];
                continue;
            }
            '[' => depth = depth.saturating_add(1),
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(after);
                }
            }
            _ => {}
        }
        previous = [Some(c), previous[0]];
        at = after;
    }
    None
}

/// Where the Rust raw string whose `#` run or `"` starts at byte offset `from` of `text` ends:
/// `None` when no raw string opens there (no `"` after the `#` run: the `r` is a plain letter),
/// `Some(None)` when one opens but never closes on the line, and `Some(Some(end))` with the
/// offset just past its closing `"` and `#` run otherwise. O(n) in the text.
fn raw_string_end(text: &str, from: usize) -> Option<Option<usize>> {
    let rest = text.get(from..)?;
    let hashes = rest.len() - rest.trim_start_matches('#').len();
    if !rest.get(hashes..)?.starts_with('"') {
        return None;
    }
    let body = from + hashes + 1;
    let closer = format!("\"{}", "#".repeat(hashes));
    Some(
        text.get(body..)
            .and_then(|b| b.find(&closer))
            .map(|pos| body + pos + closer.len()),
    )
}

/// The offset just past the `"` that closes the string whose content starts at byte offset
/// `from` of `text`, a `\` escaping the next character, or `None` when it never closes. O(n) in
/// the text.
fn quoted_end(text: &str, from: usize) -> Option<usize> {
    let mut escaped = false;
    for (at, c) in text.get(from..)?.char_indices() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(from + at + 1);
        }
    }
    None
}

/// The byte offset just past the `close` that matches the `open` `text` starts with, counting
/// only the delimiters outside quoted text, or `None` when it never closes.
///
/// A character of `quotes` opens a quoted run that the same character ends; inside it a `\`
/// escapes the next character (`"a\"b"`). The scanner ([`CodeScanner`]) reports braces per
/// line, not where a string ends on it, so this reads one line's text on its own, with the
/// quotes of the language it reads. A quote that never closes leaves the delimiters after it
/// uncounted, so the result is `None` and the caller keeps its text as it was. O(n) in the
/// text.
///
/// # Arguments
///
/// * `text` - The text, starting with `open`.
/// * `open` - The opening delimiter.
/// * `close` - The closing delimiter.
/// * `quotes` - The characters that open and close quoted text; empty to count every character.
///
/// # Returns
///
/// The offset just past the matching `close`, or `None`.
pub(super) fn balanced_end_outside_quotes(
    text: &str,
    open: char,
    close: char,
    quotes: &[char],
) -> Option<usize> {
    let mut depth = 0u32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (at, c) in text.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        if quotes.contains(&c) {
            quote = Some(c);
        } else if c == open {
            depth = depth.saturating_add(1);
        } else if c == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(at + c.len_utf8());
            }
        }
    }
    None
}

/// The name of a construct that has none (`export default function() {`, `class {`).
const UNNAMED_CONSTRUCT: &str = "unnamed";

/// The prefix of a Rust raw identifier (`r#match`), which [`construct_name`] keeps in the name.
const RAW_IDENTIFIER_PREFIX: &str = "r#";

/// The marker of a JavaScript generator declaration (`function *gen()`), which
/// [`construct_name`] passes over before the name.
const GENERATOR_MARKER: char = '*';

/// The name of a construct: the identifier that opens `rest`, the text after its keyword (after
/// a parenthesized receiver it opens with, as in Go's `func (r *T) Name(`), or, when none
/// does, the identifier of the second word of `text`, the matched line, or [`UNNAMED_CONSTRUCT`]
/// when that is no identifier either (`function() {`). A generator's `*` before
/// the name is passed over (`function *gen()` gives `gen`), and a Rust raw identifier keeps its
/// prefix (`fn r#match()` gives `r#match`). O(n) in the line.
pub(super) fn construct_name(rest: &str, text: &str) -> String {
    // MAGI FIX (S6 loop 18, B2b-6): `fn r#match()` was named `r` and `function *gen()` was
    // unnamed; a raw identifier keeps its prefix, as written, and a generator's `*` is passed
    // over.
    let ident = |t: &str| -> String {
        let (prefix, body) = t
            .strip_prefix(RAW_IDENTIFIER_PREFIX)
            .filter(|body| body.starts_with(is_identifier_char))
            .map_or(("", t), |body| (RAW_IDENTIFIER_PREFIX, body));
        let name: String = body
            .chars()
            .take_while(|&c| is_identifier_char(c))
            .collect();
        if name.is_empty() {
            name
        } else {
            format!("{prefix}{name}")
        }
    };
    let rest = rest
        .trim_start()
        .strip_prefix(GENERATOR_MARKER)
        .map_or(rest, str::trim_start);
    // MAGI FIX (S6 loop 6): a Go method's receiver left it unnamed.
    let rest = match rest.trim_start().strip_prefix('(') {
        Some(inner) => inner
            .split_once(')')
            .map_or(rest, |(_, after)| after.trim_start()),
        None => rest,
    };
    let name = ident(rest);
    if !name.is_empty() {
        return name;
    }
    let second = ident(text.split_whitespace().nth(1).unwrap_or(""));
    if second.is_empty() {
        UNNAMED_CONSTRUCT.to_string()
    } else {
        second
    }
}

/// The name of a C++17 nested namespace (`namespace a::b {` gives `a::b`): `name`, the first
/// identifier of `rest`, followed by each `::identifier` that comes right after it. `name` alone
/// when `rest` does not open with it (an unnamed namespace) or no `::` follows. O(n) in `rest`.
///
/// # Arguments
///
/// * `rest` - The text after the `namespace` keyword.
/// * `name` - The name [`construct_name`] read on it.
///
/// # Returns
///
/// The qualified name.
pub(super) fn nested_namespace_name(rest: &str, name: String) -> String {
    let Some(mut tail) = rest.trim_start().strip_prefix(name.as_str()) else {
        return name;
    };
    let mut qualified = name;
    while let Some(after) = tail.strip_prefix("::") {
        let len: usize = after
            .chars()
            .take_while(|&c| is_identifier_char(c))
            .map(char::len_utf8)
            .sum();
        let (part, next) = after.split_at(len);
        if part.is_empty() {
            break;
        }
        qualified.push_str("::");
        qualified.push_str(part);
        tail = next;
    }
    qualified
}
