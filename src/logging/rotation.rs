// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Rotation decisions, as pure functions over dates.
//!
//! Nothing here touches the filesystem or reads a clock: the caller supplies
//! both dates and receives a decision.

use time::Date;

/// Whether the open file should be rolled.
pub fn should_roll(open_date: Date, now_utc: Date) -> bool {
    now_utc > open_date
}

/// The date of the file to roll into.
pub fn roll_target(open_date: Date, now_utc: Date) -> Date {
    if now_utc > open_date {
        now_utc
    } else {
        open_date
    }
}

/// The log file name for a date.
pub fn file_name(date: Date) -> String {
    format!(
        "magi-{:04}-{:02}-{:02}.log",
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
        // The destination, NOT the predicate, is where R-L03's forbidden 24h sum
        // shows up: `open + 1 day` would name the 15th, date comparison names the 17th.
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
