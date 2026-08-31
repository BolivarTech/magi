// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

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

/// A deterministic timestamp, so a test never depends on the wall clock.
///
/// # Complexity
///
/// `O(1)`.
pub(crate) fn fixed_ts() -> time::OffsetDateTime {
    time::Date::from_calendar_date(2026, time::Month::August, 14)
        .expect("a literal calendar date")
        .midnight()
        .assume_utc()
}

/// Runs `emit` with a subscriber that renders every event through
/// [`render_event`](crate::logging::render::render_event), and returns the last
/// rendered line.
///
/// # Why a subscriber and not a constructed event
///
/// `tracing::Event` has no usable public constructor: it exists only inside a
/// dispatcher while a macro emits it. Driving `render_event` therefore means
/// installing a subscriber and emitting for real — which is the better test
/// anyway, because it walks the production path instead of a fabricated one.
///
/// # Complexity
///
/// `O(1)` plus whatever `emit` does.
pub(crate) fn capture(emit: impl FnOnce()) -> String {
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;

    let buf: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    struct CaptureLayer(Arc<Mutex<Vec<String>>>);
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if let Ok(mut g) = self.0.lock() {
                g.push(crate::logging::render::render_event(event));
            }
        }
    }

    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&buf)));
    tracing::subscriber::with_default(subscriber, emit);

    let guard = buf
        .lock()
        .expect("the capture buffer is never poisoned here");
    guard.last().cloned().unwrap_or_default()
}

/// Renders one event through the real dispatcher and returns the line.
///
/// **A macro, not a function, and that is forced rather than chosen.** `tracing`
/// builds each callsite's `Metadata` as a `static`, so the target and the level
/// must be **constant at the emit site**: a function taking them as parameters
/// fails to compile with `E0435, attempt to use a non-constant value in a
/// constant`. The plan declared
/// `render_fixture(level, target, msg, fields) -> String`, which cannot exist.
/// Fields are named at the callsite for the same reason.
macro_rules! render_fixture {
    ($lvl:expr, $target:literal, $msg:expr $(, $k:ident = $v:expr)* $(,)?) => {{
        $crate::logging::testutil::capture(|| {
            tracing::event!(target: $target, $lvl, message = $msg $(, $k = $v)*);
        })
    }};
}
pub(crate) use render_fixture;
