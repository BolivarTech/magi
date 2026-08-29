// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Retention decisions, as a pure function over a directory listing.

use std::time::SystemTime;

use time::Date;

/// A file whose `mtime` is within this of *now* is skipped: someone is writing
/// it.
///
/// Chosen, not measured — one day covers the normal writing life of a daily
/// file plus timezone slack.
pub const MTIME_SKEW_GRACE_SECS: u64 = 86_400;

/// What retention decides for one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Leave it alone.
    Keep,
    /// Compress it to `.xz`.
    Compress,
    /// Remove it.
    Delete,
}

/// One entry of the log directory, as retention needs to see it.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// File name as it appears on disk.
    pub name: String,
    /// Date parsed out of the name, or `None` when the name does not carry one.
    pub date: Option<Date>,
    /// Last modification time.
    pub mtime: SystemTime,
    /// Size in bytes.
    pub size: u64,
}

/// Operator-facing retention settings.
#[derive(Debug, Clone, Copy)]
pub struct RetentionConfig {
    /// Whether compression runs at all.
    pub compress: bool,
    /// Age in days past which a file is compressed.
    pub compress_after_days: i64,
    /// Age in days past which a file is deleted.
    pub retain_days: i64,
    /// Ceiling on the total bytes retention may leave on disk.
    pub max_total_bytes: u64,
}

/// Decides an [`Action`] for every entry, in the order given.
#[must_use]
pub fn plan(
    _files: &[FileEntry],
    _today: Date,
    _now: SystemTime,
    _cfg: &RetentionConfig,
) -> Vec<Action> {
    Vec::new() // Red-phase stub
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::testutil::build_entries;
    use time::Month;

    #[test]
    fn plans_compress_delete_and_keep_by_age_without_touching_disk() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let cfg = RetentionConfig {
            compress: true,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: u64::MAX,
        };
        let files = build_entries(&[1, 8, 30, 31, 40], today); // ages in days
        let got = plan(&files, today, SystemTime::UNIX_EPOCH, &cfg);
        assert_eq!(
            got,
            vec![
                Action::Keep,
                Action::Compress,
                Action::Compress,
                Action::Delete,
                Action::Delete
            ]
        );
    }

    #[test]
    fn todays_file_is_never_compressed_or_deleted_even_with_zero_retention() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let cfg = RetentionConfig {
            compress: true,
            compress_after_days: 0,
            retain_days: 0,
            max_total_bytes: u64::MAX,
        };
        let files = build_entries(&[0], today);
        assert_eq!(
            plan(&files, today, SystemTime::UNIX_EPOCH, &cfg),
            vec![Action::Keep]
        );
    }

    #[test]
    fn a_future_dated_file_is_purged_first_not_kept_forever() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let cfg = RetentionConfig {
            compress: true,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: u64::MAX,
        };
        let files = build_entries(&[-60], today); // dated in the future
        assert_eq!(
            plan(&files, today, SystemTime::UNIX_EPOCH, &cfg),
            vec![Action::Delete]
        );
    }

    #[test]
    fn a_file_being_written_right_now_is_skipped_whatever_its_name_says() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let now = SystemTime::now();
        let cfg = RetentionConfig {
            compress: true,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: u64::MAX,
        };
        let mut files = build_entries(&[-60], today);
        files[0].mtime = now; // someone is appending to it
        assert_eq!(plan(&files, today, now, &cfg), vec![Action::Keep]);
    }

    #[test]
    fn compress_false_emits_no_compress_action_but_still_deletes() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let cfg = RetentionConfig {
            compress: false,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: u64::MAX,
        };
        let files = build_entries(&[8, 40], today);
        assert_eq!(
            plan(&files, today, SystemTime::UNIX_EPOCH, &cfg),
            vec![Action::Keep, Action::Delete]
        );
    }

    #[test]
    fn the_total_byte_cap_deletes_the_oldest_first_and_never_todays_file() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        // Ages 0, 1 and 2 days; nothing is old enough for `retain_days`.
        let mut files = build_entries(&[0, 1, 2], today);
        for f in &mut files {
            f.size = 100;
        }
        let cfg = RetentionConfig {
            compress: false,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: 250,
        };
        // 300 bytes against a 250 cap: deleting the single oldest brings it under.
        assert_eq!(
            plan(&files, today, SystemTime::UNIX_EPOCH, &cfg),
            vec![Action::Keep, Action::Keep, Action::Delete]
        );
    }

    #[test]
    fn a_cap_smaller_than_todays_file_alone_keeps_it_and_does_not_pretend() {
        let today = Date::from_calendar_date(2026, Month::August, 31).unwrap();
        let mut files = build_entries(&[0], today);
        files[0].size = 5_000;
        let cfg = RetentionConfig {
            compress: false,
            compress_after_days: 7,
            retain_days: 30,
            max_total_bytes: 100,
        };
        // REQ-L15 wins over REQ-L18: today's file survives a cap it cannot
        // satisfy. The pathological case is REPORTED by the caller, never faked
        // by deleting the one file the process is writing to.
        assert_eq!(
            plan(&files, today, SystemTime::UNIX_EPOCH, &cfg),
            vec![Action::Keep]
        );
    }
}
