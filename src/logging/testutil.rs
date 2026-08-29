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

/// Builds one [`FileEntry`](crate::logging::retention::FileEntry) per age, in
/// the order given.
///
/// A **positive** age is that many days *before* `today`; a **negative** age is
/// a file dated in the future, which is the case R-L13e exists for.
///
/// # Why `mtime` is set *before* the epoch
///
/// The tests hand `plan` a `now` of `SystemTime::UNIX_EPOCH`. An `mtime` of the
/// epoch would then read as "written zero seconds ago", the skew guard would
/// protect **every** fixture file, and every scenario would come back all
/// `Keep` — which is what happened, and it looked exactly like an
/// implementation bug rather than a fixture one. Placing `mtime` two grace
/// periods before the epoch makes the fixture unambiguously old; a test that
/// wants the guard to fire sets `mtime` itself.
///
/// # Complexity
///
/// `O(n)` over the ages.
pub(crate) fn build_entries(
    ages_in_days: &[i64],
    today: time::Date,
) -> Vec<crate::logging::retention::FileEntry> {
    use crate::logging::retention::FileEntry;
    ages_in_days
        .iter()
        .map(|age| {
            let date = today - time::Duration::days(*age);
            FileEntry {
                name: crate::logging::rotation::file_name(date),
                date: Some(date),
                mtime: std::time::SystemTime::UNIX_EPOCH
                    - std::time::Duration::from_secs(
                        crate::logging::retention::MTIME_SKEW_GRACE_SECS * 2,
                    ),
                size: 0,
            }
        })
        .collect()
}
