// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Shared test fixture for the logging subsystem.
//!
//! These helpers are **not production API**. They live in one place because the
//! alternative is each task inventing its own and the tests quietly ceasing to
//! be comparable — two modules asserting on "the payload" while meaning two
//! different substrings.
//!
//! The module is `#[cfg(test)]`, so it adds no public surface and does not
//! count against the "no API without a consumer" rule.

/// The payload of a produced line, with its header and marker stripped.
///
/// A chunked line is `<header>id=<pid>-<hex16> n/N <payload>`, so the payload is
/// what follows the marker's two tokens. An unchunked line carries no marker at
/// all (REQ-L11), and there the header ends at the first space.
///
/// # Complexity
///
/// `O(n)` over the line.
pub(crate) fn payload_of(line: &str) -> &str {
    match line.find("id=") {
        Some(i) => line[i..].splitn(3, ' ').nth(2).unwrap_or(""),
        None => line.split_once(' ').map(|(_, p)| p).unwrap_or(line),
    }
}
