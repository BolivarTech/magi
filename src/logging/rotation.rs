// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! Rotation decisions, as pure functions over dates.
//!
//! Nothing here touches the filesystem or reads a clock: the caller supplies
//! both dates and receives a decision. That split is what makes the behaviour
//! testable — proving "a process idle for three days lands in today's file"
//! needs two `Date` values, not three days of elapsed time.
//!
//! # The rule these functions exist to enforce
//!
//! Dates are **compared**, never advanced by 24 hours. The two are identical
//! for a process that runs continuously and differ the moment one goes idle
//! across a day boundary, which is exactly when a log is worth having.

use time::Date;

/// Prefix of every log file name.
const FILE_PREFIX: &str = "magi";
/// Extension of an uncompressed log file.
const FILE_EXTENSION: &str = "log";

/// Whether the currently open file should be rolled.
///
/// # Parameters
///
/// * `open_date` — the UTC date of the file currently open for writing.
/// * `now_utc` — the current UTC date.
///
/// # Returns
///
/// `true` when `now_utc` is strictly later than `open_date`.
///
/// **Rotation is monotonic.** A `now_utc` *earlier* than `open_date` — NTP
/// correcting backwards, a VM resuming from a snapshot — returns `false` and
/// writing continues in the current file. Rolling backwards would reopen in
/// append mode a file that was already closed, and possibly already compressed
/// and deleted, leaving a fresh `.log` living beside its own `.xz`.
///
/// # Examples
///
/// ```
/// use magi_rs::logging::rotation::should_roll;
/// use time::{Date, Month};
///
/// let open = Date::from_calendar_date(2026, Month::August, 14).unwrap();
/// let later = Date::from_calendar_date(2026, Month::August, 15).unwrap();
/// assert!(should_roll(open, later));
/// assert!(!should_roll(open, open));
/// assert!(!should_roll(later, open), "never roll backwards");
/// ```
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn should_roll(open_date: Date, now_utc: Date) -> bool {
    now_utc > open_date
}

/// The date of the file that writing should continue into.
///
/// # Parameters
///
/// * `open_date` — the UTC date of the file currently open for writing.
/// * `now_utc` — the current UTC date.
///
/// # Returns
///
/// The later of the two dates.
///
/// # Why this is a function and not an expression at the call site
///
/// The rule "compare dates, never add 24 hours" does **not** live in
/// [`should_roll`]: with an open date of the 14th and a now of the 17th, both
/// the comparison and the forbidden sum answer `true`, so no test of the
/// predicate can tell them apart. The difference shows up only in *which file*
/// is opened — `open + 1 day` names the 15th, comparison names the 17th. Left
/// inside the writer, that decision sits outside the reach of every test in
/// this module and the rule ends up with no guardian anywhere.
///
/// # Examples
///
/// ```
/// use magi_rs::logging::rotation::roll_target;
/// use time::{Date, Month};
///
/// let open = Date::from_calendar_date(2026, Month::August, 14).unwrap();
/// let now = Date::from_calendar_date(2026, Month::August, 17).unwrap();
/// // Three days idle lands in today's file, not in the day after the open one.
/// assert_eq!(roll_target(open, now), now);
/// ```
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn roll_target(open_date: Date, now_utc: Date) -> Date {
    if now_utc > open_date {
        now_utc
    } else {
        open_date
    }
}

/// The log file name for a date.
///
/// # Parameters
///
/// * `date` — the UTC date the file belongs to.
///
/// # Returns
///
/// `magi-YYYY-MM-DD.log`, zero-padded, so the names sort chronologically as
/// plain strings. Retention reads these names back, so the format is a
/// contract rather than a presentation choice.
///
/// # Examples
///
/// ```
/// use magi_rs::logging::rotation::file_name;
/// use time::{Date, Month};
///
/// let d = Date::from_calendar_date(2026, Month::January, 5).unwrap();
/// assert_eq!(file_name(d), "magi-2026-01-05.log");
/// ```
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn file_name(date: Date) -> String {
    format!(
        "{FILE_PREFIX}-{:04}-{:02}-{:02}.{FILE_EXTENSION}",
        date.year(),
        u8::from(date.month()),
        date.day()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Month;

    #[test]
    fn rolls_when_the_utc_date_advances() {
        let open = Date::from_calendar_date(2026, Month::August, 14).unwrap();
        let now = Date::from_calendar_date(2026, Month::August, 15).unwrap();
        assert!(should_roll(open, now));
    }

    #[test]
    fn rolls_to_the_current_day_after_a_three_day_idle_not_to_open_plus_one() {
        let open = Date::from_calendar_date(2026, Month::August, 14).unwrap();
        let now = Date::from_calendar_date(2026, Month::August, 17).unwrap();
        assert!(should_roll(open, now));
        // The destination, NOT the predicate, is where the forbidden 24h sum
        // shows up: `open + 1 day` would name the 15th, date comparison names
        // the 17th.
        assert_eq!(roll_target(open, now), now);
        assert_eq!(file_name(roll_target(open, now)), "magi-2026-08-17.log");
    }

    #[test]
    fn the_roll_target_of_a_backwards_clock_stays_on_the_open_file() {
        let open = Date::from_calendar_date(2026, Month::August, 15).unwrap();
        let now = Date::from_calendar_date(2026, Month::August, 14).unwrap();
        assert!(!should_roll(open, now));
        assert_eq!(
            roll_target(open, now),
            open,
            "monotonic: never reopen a closed day"
        );
    }

    #[test]
    fn does_not_roll_backwards_when_the_clock_moves_into_the_past() {
        let open = Date::from_calendar_date(2026, Month::August, 15).unwrap();
        let now = Date::from_calendar_date(2026, Month::August, 14).unwrap();
        assert!(!should_roll(open, now));
    }

    #[test]
    fn does_not_roll_within_the_same_utc_day() {
        let d = Date::from_calendar_date(2026, Month::August, 14).unwrap();
        assert!(!should_roll(d, d));
    }

    #[test]
    fn file_name_uses_the_iso_date_with_the_log_extension() {
        let d = Date::from_calendar_date(2026, Month::January, 5).unwrap();
        assert_eq!(file_name(d), "magi-2026-01-05.log");
    }
}
