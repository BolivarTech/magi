// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

//! A bounded queue and exactly one writer.
//!
//! # Why a queue and not a shared lock
//!
//! A single mutex around the file was **measured** and rejected: throughput at
//! 16 emitters fell to 0.40 of one emitter's, against a threshold of 0.50 fixed
//! before the run. The queue scores 1.5–2.0 on the same metric. Do not
//! reintroduce the lock.
//!
//! # The emitter never blocks, and that is the trade
//!
//! `try_send` and move on. A full channel **drops** the event and counts it; when
//! the pressure clears, one `warn!` says how many were lost. Losing events and
//! saying so is acceptable; stalling the agent is not — that is the whole reason
//! this subsystem exists on the far side of a queue.
//!
//! Measured cost of that trade, at 16 emitters in a closed loop: the submit path
//! runs 25–40x faster than under the lock, and 97.7 % of events are dropped. No
//! real workload here looks like that — a single-user terminal agent has the
//! main agent, three trio seats and a few tools, none spinning — but it is the
//! edge, and an operator should be able to read it beside the drop counter
//! rather than deduce it.
//!
//! # A whole event travels the channel, never a chunk
//!
//! Chunking happens **in the writer**, under its exclusive access. If chunks
//! were queued individually, two large events from different threads would
//! interleave their lines and both would be unreadable — and the `id=n/N`
//! markers would be the only way to reassemble them, which is a repair job for
//! something that should never break.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use crate::logging::auditor::Queued;
use crate::logging::LoggingError;

/// Total slots across both channels.
pub const LOG_CHANNEL_CAPACITY: usize = 8192;
/// Slots reserved for the priority channel.
pub const LOG_CHANNEL_HIGH_SLOTS: usize = 2048;
/// Slots for the ordinary channel.
pub const LOG_CHANNEL_LOW_SLOTS: usize = 6144;

/// **The total is a CHECKED relation, not a comment.**
///
/// `LOG_CHANNEL_CAPACITY` had no consumer: the code opens the two channels from
/// their own halves and never reads the sum, so the constant was a claim that
/// could drift from the numbers below it with nothing to notice. Asserting it at
/// compile time gives it the consumer G2 requires and turns the claim into a
/// guarantee, in the same move.
const _: () = assert!(LOG_CHANNEL_HIGH_SLOTS + LOG_CHANNEL_LOW_SLOTS == LOG_CHANNEL_CAPACITY);
/// Byte budget of the priority channel.
///
/// **The priority channel gets the LARGER budget while holding FEWER slots, and
/// the asymmetry is deliberate.** Its risk is size — the big events in this
/// system are HTTP error bodies, which are `ERROR` — while the ordinary
/// channel's risk is count. Giving the priority channel a quarter of the byte
/// budget made a 20 MiB event pass as `INFO` and be discarded while being an
/// `ERROR`: the guarantee failing in the direction the spec calls unacceptable.
pub const LOG_CHANNEL_HIGH_BYTES: usize = 48 * 1024 * 1024;
/// Byte budget of the ordinary channel.
pub const LOG_CHANNEL_LOW_BYTES: usize = 16 * 1024 * 1024;
/// The line terminator, named so the two write sites cannot drift.
const NEWLINE: &[u8] = b"
";
/// How long the parked writer waits before re-checking, as a backstop.
///
/// The condvar is what actually wakes it; this bounds the damage if a signal is
/// ever missed, rather than being the mechanism.
const PARK_POLL: std::time::Duration = std::time::Duration::from_millis(50);
/// Events the writer drains per batch before yielding.
pub const HIGH_BATCH: usize = 64;
/// How stale the writer's heartbeat may get before an emitter calls it hung.
pub const WRITER_STALL_SECS: u64 = 60;
/// Wait before the writer's ONE retry after a write failure.
///
/// **Chosen, not measured**, and the trade-off is not obvious: everything
/// emitted during the wait is lost. Shorter retries before a transient cause
/// clears — a disk that filled does not empty in five seconds — and spends the
/// single retry for nothing; longer widens the window in which every event goes
/// missing. Thirty seconds is a compromise between a retry that is worth making
/// and a gap an operator can still explain.
pub const WRITER_RETRY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30);

/// Which channel an event travels on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// `WARN` and above, plus every alarm.
    High,
    /// Everything else.
    Low,
}

/// What happened to a submitted event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submitted {
    /// It is on the queue.
    Queued,
    /// The channel was full; it was dropped and counted.
    DroppedFull,
    /// It alone exceeds its channel's byte budget. **Not congestion** — this is
    /// a problem with whoever emitted it, and the notice says so.
    DroppedOversized,
    /// The writer is gone. The file branch is off; nothing more is counted and
    /// nothing more is announced.
    WriterGone,
    /// The writer's heartbeat went stale with work pending. Permanent.
    WriterHung,
}

/// One channel: its sender, its byte budget and its reservation counter.
struct Channel {
    tx: SyncSender<Queued>,
    budget: usize,
    /// **Shared with the writer**, which is the half that gives bytes back.
    /// A reservation is a measure of what is IN the queue, so it has to fall as
    /// the queue drains; owned solely by the emitter side it only ever rises,
    /// and the budget silently becomes a lifetime quota.
    reserved: Arc<AtomicUsize>,
}

impl Channel {
    /// Reserves `len` bytes, or refuses.
    ///
    /// **`fetch_add` and then check the returned value**, never "read, then
    /// add": with a prior read, `k` concurrent emitters all see the same gap and
    /// all take it.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn reserve(&self, len: usize) -> Option<Reservation<'_>> {
        let before = self.reserved.fetch_add(len, Ordering::AcqRel);
        if before.saturating_add(len) > self.budget {
            self.reserved.fetch_sub(len, Ordering::AcqRel);
            return None;
        }
        Some(Reservation {
            counter: &self.reserved,
            len,
            committed: false,
        })
    }
}

/// Bytes taken from a channel's budget, returned on drop unless committed.
///
/// **A guard and not a manual release**, because between the `fetch_add` and the
/// `try_send` there is code that can unwind, and a hand-written release is lost
/// on that path forever: the counter stays high and the channel ends up refusing
/// events with capacity to spare.
struct Reservation<'a> {
    counter: &'a AtomicUsize,
    len: usize,
    committed: bool,
}

impl Reservation<'_> {
    /// Hands the bytes to the writer, which will release them once written.
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.counter.fetch_sub(self.len, Ordering::AcqRel);
        }
    }
}

/// What the retry writes through.
///
/// A trait for one reason: REQ-L37's retry is a **timing** behaviour over a
/// failing target, and a real `FileSink` cannot be made to fail on demand -- it
/// holds an open handle to a file the test would have to break underneath it.
/// The same argument `submit_at` already makes for injecting the clock.
pub(crate) trait ItemWriter {
    /// Writes one queued item.
    ///
    /// # Errors
    ///
    /// Whatever the underlying target reports.
    fn write_item(&mut self, item: &Queued) -> std::io::Result<()>;
}

impl ItemWriter for FileSink {
    fn write_item(&mut self, item: &Queued) -> std::io::Result<()> {
        self.write(item)
    }
}

/// Writes each day's events to its own file, from one thread.
pub struct DailyAppender {
    high: Channel,
    low: Channel,
    dropped: Arc<AtomicU64>,
    /// Set once the writer is known to be gone, so the failure is announced
    /// exactly once and never counted as congestion.
    writer_gone: Arc<AtomicU64>,
    dir: PathBuf,
    writer: Option<std::thread::JoinHandle<()>>,
    /// Last moment the writer was known to be alive and working.
    ///
    /// Stamped when it finishes a consume cycle, **when it parks on an empty
    /// queue**, and **when it enters the retry cooldown**. All three, because
    /// the mark has to refresh in every state where the writer is alive and
    /// deliberately not consuming — miss one and an emitter reads a stale mark
    /// with a non-empty queue and declares a hang that does not exist, which is
    /// a permanent shutdown of the file layer.
    heartbeat: Arc<std::sync::Mutex<std::time::Instant>>,
    /// Set once a hang has been declared. The file branch never comes back.
    hung: Arc<AtomicU64>,
    /// Wakes the writer when either channel gains work.
    ///
    /// **Without this the writer blocked on the PRIORITY channel and never woke
    /// for an ordinary one.** A session that emits only `INFO` — which is most
    /// of them — was never written at all until some `WARN` happened to arrive
    /// and drain the backlog behind it. It looked like a test artefact and was
    /// not: the canary found it because it emitted a single `INFO` and read an
    /// empty directory.
    wake: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl DailyAppender {
    /// Opens an appender over `dir`.
    ///
    /// # Errors
    ///
    /// [`LoggingError::DirCreate`] if the directory cannot be created.
    pub fn new(dir: &Path) -> Result<Self, LoggingError> {
        std::fs::create_dir_all(dir).map_err(|e| LoggingError::DirCreate {
            path: dir.to_path_buf(),
            source: e,
        })?;
        // REQ-L65. `create_dir_all` leaves the process umask's bits, commonly
        // 0755, so the directory holding a transcript of everything the agent
        // did would be world-readable on a shared machine.
        crate::logging::restrict(dir, crate::logging::OWNER_ONLY_DIR_MODE)?;
        let heartbeat = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
        let wake = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (high_tx, high_rx) = sync_channel(LOG_CHANNEL_HIGH_SLOTS);
        let (low_tx, low_rx) = sync_channel(LOG_CHANNEL_LOW_SLOTS);
        let high_reserved = Arc::new(AtomicUsize::new(0));
        let low_reserved = Arc::new(AtomicUsize::new(0));
        Ok(Self {
            high: Channel {
                tx: high_tx,
                budget: LOG_CHANNEL_HIGH_BYTES,
                reserved: Arc::clone(&high_reserved),
            },
            low: Channel {
                tx: low_tx,
                budget: LOG_CHANNEL_LOW_BYTES,
                reserved: Arc::clone(&low_reserved),
            },
            dropped: Arc::new(AtomicU64::new(0)),
            writer_gone: Arc::new(AtomicU64::new(0)),
            dir: dir.to_path_buf(),
            writer: Some(spawn_writer(
                dir.to_path_buf(),
                high_rx,
                low_rx,
                high_reserved,
                low_reserved,
                Arc::clone(&heartbeat),
                Arc::clone(&wake),
            )),
            heartbeat,
            wake,
            hung: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Offers one event. **Never blocks.**
    ///
    /// # Parameters
    ///
    /// * `item` — the whole audited event, never one of its chunks.
    /// * `priority` — which channel it travels on.
    /// * `len` — the length reserved for it, which is [`Audited::reserved_len`](crate::logging::auditor::Audited::reserved_len)
    ///   and therefore the length of the text as it will be QUEUED -- after
    ///   the audit and after the escape, which is what the writer releases.
    ///
    /// # Returns
    ///
    /// What happened, so the caller can count and warn.
    ///
    /// # Complexity
    ///
    /// `O(1)` plus the channel's own send.
    pub fn submit(&self, item: Queued, priority: Priority, len: usize) -> Submitted {
        self.submit_at(item, priority, len, std::time::Instant::now())
    }

    /// [`submit`](Self::submit) with the clock supplied, so the stall check is
    /// testable without waiting a minute.
    pub fn submit_at(
        &self,
        item: Queued,
        priority: Priority,
        len: usize,
        now: std::time::Instant,
    ) -> Submitted {
        // **The stall check runs BEFORE the send, and the order is observable.**
        // After it, a full channel sends the emitter down the discard path and
        // it never looks at the mark — so exactly when the writer is hung, which
        // is when the queue fills, the detector stops running. The hang would be
        // undetectable in its own symptom.
        if self.check_stalled(now) {
            return Submitted::WriterHung;
        }
        if self.writer_gone.load(Ordering::Acquire) != 0 {
            // Closed is the writer's death, not congestion. Counting these would
            // produce the per-turn echo this feature exists to remove.
            return Submitted::WriterGone;
        }
        let channel = match priority {
            Priority::High => &self.high,
            Priority::Low => &self.low,
        };
        if len > channel.budget {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Submitted::DroppedOversized;
        }
        let Some(reservation) = channel.reserve(len) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Submitted::DroppedFull;
        };
        match channel.tx.try_send(item) {
            Ok(()) => {
                reservation.commit();
                self.signal();
                Submitted::Queued
            }
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Submitted::DroppedFull
            }
            Err(TrySendError::Disconnected(_)) => {
                self.writer_gone.store(1, Ordering::Release);
                Submitted::WriterGone
            }
        }
    }

    /// Wakes the writer, whichever channel the work landed on.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn signal(&self) {
        let (lock, cv) = &*self.wake;
        if let Ok(mut ready) = lock.lock() {
            *ready = true;
            cv.notify_one();
        }
    }

    /// Declares a hang if the writer's mark has gone stale with work pending.
    ///
    /// **A hang does NOT inherit the retry.** A finished thread no longer
    /// exists; a hung one is still inside a `write` with its file open, and safe
    /// Rust can neither kill nor join it. Recreating it would leave two threads
    /// writing the same file.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn check_stalled(&self, now: std::time::Instant) -> bool {
        if self.hung.load(Ordering::Acquire) != 0 {
            return true;
        }
        let Ok(mark) = self.heartbeat.lock() else {
            return false;
        };
        if now.duration_since(*mark) > std::time::Duration::from_secs(WRITER_STALL_SECS) {
            self.hung.store(1, Ordering::Release);
            return true;
        }
        false
    }

    /// The writer's current heartbeat, for the tests that assert it refreshes.
    #[cfg(test)]
    fn heartbeat_at(&self) -> Option<std::time::Instant> {
        self.heartbeat.lock().ok().map(|m| *m)
    }

    /// How many events have been dropped so far.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Bytes queued on both channels together.
    ///
    /// Meaningful only because the writer now RELEASES what it wrote: before
    /// that, this number only ever rose and a drain built on it would never
    /// finish.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn queued_bytes(&self) -> usize {
        self.reserved(Priority::High) + self.reserved(Priority::Low)
    }

    /// Bytes currently reserved on a channel.
    #[must_use]
    pub fn reserved(&self, priority: Priority) -> usize {
        match priority {
            Priority::High => self.high.reserved.load(Ordering::Acquire),
            Priority::Low => self.low.reserved.load(Ordering::Acquire),
        }
    }

    /// The directory being written to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Closes the channels and waits for the writer to drain.
    ///
    /// # Complexity
    ///
    /// `O(queued)`.
    pub fn shutdown(mut self) {
        drop(std::mem::replace(&mut self.high.tx, sync_channel(1).0));
        drop(std::mem::replace(&mut self.low.tx, sync_channel(1).0));
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

/// Starts the single writer.
///
/// **The priority channel is drained first, in batches of [`HIGH_BATCH`]**, so a
/// saturated ordinary channel cannot starve an `ERROR`. The batch is bounded by
/// count so the ordinary channel is not starved in return.
///
/// # Complexity
///
/// `O(events)` over the run.
/// Refreshes the mark an emitter reads to decide the writer is alive.
///
/// Lifted out of `spawn_writer` with the retry it serves: the retry is REQ-L37
/// and needed a test, and a helper nested inside a thread closure cannot be
/// reached from one.
fn stamp(hb: &std::sync::Mutex<std::time::Instant>) {
    if let Ok(mut m) = hb.lock() {
        *m = std::time::Instant::now();
    }
}

/// Writes one item, taking the ONE retry a write failure is allowed.
///
/// # The retry hangs off the `io::Error`, not off the thread dying
///
/// `clippy::panic` is denied in this module, so a write failure
/// **structurally cannot** panic the writer: hanging the retry off thread
/// termination would have fired only on a bug — whose cause persists — while
/// the transient cases got permanent shutdown. And because the writer does
/// not die, the receiver does not drop and there is no sender to swap.
///
/// # Why the mark is stamped on the way INTO the cooldown
///
/// The writer sits here for thirty seconds consuming nothing. Events arriving
/// meanwhile leave the queue non-empty, so the first emitter to look would
/// read a stale mark against a full queue and declare a hang that does not
/// exist — which is PERMANENT. A transient disk error, the very case the
/// retry exists for, would end in permanent shutdown before the retry ever
/// happened. The mark has to refresh in **every** state where the writer is
/// alive and deliberately not consuming.
///
/// # Returns
///
/// `false` once the file branch must be shut down for good.
fn write_with_one_retry(
    sink: &mut impl ItemWriter,
    item: &Queued,
    hb: &std::sync::Mutex<std::time::Instant>,
    cooldown: std::time::Duration,
) -> bool {
    if sink.write_item(item).is_ok() {
        return true;
    }
    // **Nothing to stderr from here** (REQ-L39). This runs on a detached
    // thread for the whole session, so a write here lands on top of the
    // ratatui frame whenever the TUI holds the alternate screen — the exact
    // corruption `TuiNoticeSink` exists to prevent.
    //
    // Nor is anything lost by saying nothing: REQ-L37 asks for ONE notice
    // when the file branch shuts down, and the layer already emits it
    // through the notice sink when the writer's death turns the next submit
    // into `WriterGone`. A second announcement from a thread that cannot
    // reach the sink would be a duplicate on the surface where it is safe
    // and damage on the surface where it is not.
    //
    // Stamp, sleep, stamp: an emitter looking at any moment of this window
    // sees a live writer.
    stamp(hb);
    std::thread::sleep(cooldown);
    stamp(hb);
    sink.write_item(item).is_ok()
}

fn spawn_writer(
    dir: PathBuf,
    high: Receiver<Queued>,
    low: Receiver<Queued>,
    high_reserved: Arc<AtomicUsize>,
    low_reserved: Arc<AtomicUsize>,
    heartbeat: Arc<std::sync::Mutex<std::time::Instant>>,
    wake: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
) -> std::thread::JoinHandle<()> {
    /// What an item reserved, read off the item itself.
    ///
    /// **One source of truth.** The emitter reserves `Audited::reserved_len`
    /// and the writer releases the same field, so the two halves cannot drift
    /// into subtracting a different number than was added. An alarm reserves
    /// nothing (`submit` is called with `0`) and therefore releases nothing.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn reserved_of(item: &Queued) -> usize {
        match item {
            Queued::Line(a) => a.reserved_len(),
            Queued::Alarm(_) => 0,
        }
    }

    std::thread::spawn(move || {
        let mut sink = FileSink::new(dir);
        let mut high_closed = false;
        let mut low_closed = false;
        loop {
            let mut did_work = false;
            for _ in 0..HIGH_BATCH {
                match high.try_recv() {
                    Ok(item) => {
                        let held = reserved_of(&item);
                        let written = write_with_one_retry(
                            &mut sink,
                            &item,
                            &heartbeat,
                            WRITER_RETRY_COOLDOWN,
                        );
                        // Released whether or not the write landed: the bytes
                        // have left the queue either way, and holding them for a
                        // line that will never be written charges the budget for
                        // an event nobody can read.
                        high_reserved.fetch_sub(held, Ordering::AcqRel);
                        if !written {
                            return;
                        }
                        did_work = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        high_closed = true;
                        break;
                    }
                }
            }
            match low.try_recv() {
                Ok(item) => {
                    let held = reserved_of(&item);
                    let written =
                        write_with_one_retry(&mut sink, &item, &heartbeat, WRITER_RETRY_COOLDOWN);
                    low_reserved.fetch_sub(held, Ordering::AcqRel);
                    if !written {
                        return;
                    }
                    did_work = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => low_closed = true,
            }
            if did_work {
                stamp(&heartbeat);
            } else {
                // Nothing on either channel. **Stamp BEFORE parking**: an idle
                // writer otherwise accumulates an unboundedly stale mark, and
                // the first event to arrive lets another emitter — microseconds
                // later, with the queue momentarily non-empty — read a
                // three-hour-old mark and declare a hang that does not exist.
                stamp(&heartbeat);
                if high_closed && low_closed {
                    break;
                }
                // Wait on the SIGNAL, not on one channel. Blocking on the
                // priority receiver left an ordinary event unwritten until some
                // priority event happened to arrive — and most sessions emit
                // only ordinary ones.
                let (lock, cv) = &*wake;
                if let Ok(ready) = lock.lock() {
                    let (mut ready, _) = cv
                        .wait_timeout(ready, PARK_POLL)
                        .unwrap_or_else(|p| p.into_inner());
                    *ready = false;
                }
            }
        }
    })
}

/// The writer's exclusive view of the day's file.
///
/// Owns the chunking, because chunking under exclusive access is what keeps two
/// large events from different threads out of each other's lines.
struct FileSink {
    dir: PathBuf,
    open: Option<(time::Date, std::io::BufWriter<std::fs::File>)>,
}

impl FileSink {
    /// Builds a sink over a directory. Nothing is opened until the first write.
    fn new(dir: PathBuf) -> Self {
        Self { dir, open: None }
    }

    /// Writes one whole event, chunked, to today's file.
    ///
    /// # Complexity
    ///
    /// `O(n)` over the event.
    fn write(&mut self, item: &Queued) -> std::io::Result<()> {
        use std::io::Write as _;

        // **Escaped exactly once, and the two arms differ because their inputs
        // do.** A `Line` arrives already escaped: the layer runs stage 3 of
        // REQ-L64 before submitting, because the same escaped text also goes to
        // the screen branch. Escaping it again here doubled every backslash a
        // second time, so a workspace path reached the file as `C:\\Users`.
        // An `Alarm` is rendered right here and has had no other chance, so it
        // is escaped here — which is its only escape, not a second one.
        let text = match item {
            Queued::Line(a) => a.as_str().to_string(),
            Queued::Alarm(x) => {
                // **An alarm gets a header like any other line**, and it has to:
                // without one it carries no timestamp, no level and no `run=`,
                // so filtering the daily file by a run silently drops exactly
                // the lines that say a credential was masked. It is rendered at
                // ERROR because that is what it is.
                format!(
                    "{}{}",
                    crate::logging::render::header_of(
                        tracing::Level::ERROR,
                        "magi_rs::logging",
                        time::OffsetDateTime::now_utc(),
                        crate::logging::run_id(),
                    ),
                    crate::logging::render::escape_for_line(
                        &crate::logging::auditor::render_alarm(x)
                    )
                )
            }
        };
        let now = time::OffsetDateTime::now_utc().date();
        self.rotate_to(now)?;
        let Some((_, file)) = self.open.as_mut() else {
            return Ok(());
        };
        let id = crate::logging::chunk::EventId::new();
        // **The header is handed to the chunker AS the header**, not buried in
        // the payload. With an empty one, chunk 1 came out as
        // `id=<id> 1/N <timestamp> ...` -- the marker ahead of the header, the
        // reverse of what REQ-L08 declares -- and the budget was measured
        // against a header the chunker could not see.
        let cut = crate::logging::render::header_end(&text);
        let (header, body) = text.split_at(cut.min(text.len()));
        for line in crate::logging::chunk::split(
            body,
            header,
            &crate::logging::chunk::cont_header_for(&id, crate::logging::run_id()),
            id,
        ) {
            file.write_all(line.as_bytes())?;
            file.write_all(NEWLINE)?;
        }
        file.flush()
    }

    /// Opens the file for `today` if the open one belongs to another day.
    ///
    /// **The destination comes from `rotation::roll_target`**, never from a sum:
    /// the rule that forbids adding 24 hours shows up in WHICH file is opened,
    /// not in whether to roll.
    fn rotate_to(&mut self, today: time::Date) -> std::io::Result<()> {
        let target = match self.open.as_ref() {
            Some((open_date, _)) => crate::logging::rotation::roll_target(*open_date, today),
            None => today,
        };
        if matches!(self.open.as_ref(), Some((d, _)) if *d == target) {
            return Ok(());
        }
        let path = self.dir.join(crate::logging::rotation::file_name(target));
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        // **The mode at CREATION, not a chmod afterwards.** A chmod leaves the
        // file world-readable for the instant between the two calls, and that
        // instant is when a reader on a shared machine wins.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(crate::logging::OWNER_ONLY_FILE_MODE);
        }
        let file = options.open(&path)?;
        // And again for the file that ALREADY existed: `mode()` applies only to
        // a file this call creates, so a day's file first opened by a run with a
        // laxer umask would keep those bits for the rest of the day.
        //
        // Best effort and SILENT, deliberately: REQ-L35 says logging never
        // aborts a session, and REQ-L39 forbids this thread reaching stderr at
        // all. On the platform where the mode matters the call fails only if the
        // file is not ours, which the surrounding open would already have
        // failed on.
        let _ = crate::logging::restrict(&path, crate::logging::OWNER_ONLY_FILE_MODE);
        self.open = Some((target, std::io::BufWriter::new(file)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::auditor::{Auditor, SecretName};
    use tempfile::tempdir;

    /// Wraps text as a queued line with a reservation equal to its length.
    fn line(text: &str) -> (Queued, usize) {
        let (a, _) = Auditor::new().audit(text, "magi_rs::tests", None, text.len());
        (Queued::Line(a), text.len())
    }

    #[test]
    fn a_split_event_carries_its_header_before_the_chunk_marker() {
        // REQ-L08: chunk 1 is the FULL HEADER followed by `id=... 1/N`. The sink
        // used to hand the chunker an empty header and the whole rendered line
        // as payload, so chunk 1 came out `id=... 1/N <timestamp> ...` -- the
        // marker ahead of the header it is supposed to follow, and a budget
        // measured against a header the chunker could not see.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        // Prose, not padding: a long run of one character is what the auditor
        // calls a secret, and `***` never reaches the chunker at all.
        let long = "the quick brown fox jumps over the lazy dog ".repeat(200);
        let rendered = format!(
            "{}{long}",
            crate::logging::render::header_of(
                tracing::Level::INFO,
                "magi_rs::agent",
                time::OffsetDateTime::now_utc(),
                crate::logging::run_id(),
            )
        );
        let (audited, _) = Auditor::new().audit(&rendered, "magi_rs::tests", None, rendered.len());
        assert_eq!(
            appender.submit(Queued::Line(audited), Priority::Low, rendered.len()),
            Submitted::Queued
        );
        std::thread::sleep(std::time::Duration::from_millis(300));

        let path = dir.path().join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let written = std::fs::read_to_string(&path).unwrap_or_default();
        let lines: Vec<&str> = written.lines().filter(|l| !l.trim().is_empty()).collect();
        // The fixture must have SPLIT, or the ordering under test never arises.
        assert!(lines.len() > 1, "the fixture produced one line: {written}");

        let first = lines.first().copied().unwrap_or("");
        assert!(
            first.starts_with("20"),
            "chunk 1 does not begin with its timestamp: {first}"
        );
        let marker = first.find("id=").unwrap_or(usize::MAX);
        let separator = first.find(": ").unwrap_or(0);
        assert!(
            marker > separator,
            "the chunk marker precedes the header instead of following it: {first}"
        );
        appender.shutdown();
    }

    #[test]
    fn an_alarm_carries_a_header_so_the_run_filter_finds_it() {
        // Without one an alarm has no timestamp, no level and no `run=`, so
        // filtering the daily file by a run drops exactly the lines that say a
        // credential was masked -- the ones a reader is looking for.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("BASE_URL_PASSWORD"), &["hunter2-longer"]);
        let (_, alarm) = auditor.audit(
            "GET https://bob:hunter2-longer@example.com/v1",
            "magi_rs::agent",
            None,
            0,
        );
        let alarm = alarm.expect("a live secret must alarm");
        assert_eq!(
            appender.submit(Queued::Alarm(alarm), Priority::High, 0),
            Submitted::Queued
        );
        std::thread::sleep(std::time::Duration::from_millis(300));

        let path = dir.path().join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let written = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            written.contains("SECURITY"),
            "the fixture wrote no alarm: {written:?}"
        );
        assert!(
            written.contains(&format!("run={}", crate::logging::run_id())),
            "the alarm is invisible to the run filter: {written:?}"
        );
        assert!(
            written.contains("ERROR"),
            "and it must carry the level it is: {written:?}"
        );
        // **At the FRONT, not merely present.** Order is not cosmetic here:
        // `header_end` finds the first target separator to tell the header from
        // the body, so a header appended after the text makes that cut land
        // inside the message and hands the chunker the wrong prefix. Swapping
        // the two left every assertion above green, which is what sent this
        // one back for another line.
        assert!(
            written.starts_with("20"),
            "the alarm's header is not at the front: {written:?}"
        );
        appender.shutdown();
    }

    #[test]
    fn an_alarm_cannot_forge_a_second_line() {
        // REQ-L64 names this case in its own rustdoc: a foreign string that
        // survives unescaped produces what LOOKS like an independent log line,
        // "including one imitating an auditor alarm, with nothing to tell the
        // false from the real".
        //
        // The alarm's text is built from the SECRET NAME, which is ours, and the
        // TARGET, which is not: a target is a literal in whichever crate emitted
        // the event, and this tree logs foreign events from magi-core. Passing a
        // hostile one is exercising the declared contract, not inventing a
        // caller for it.
        //
        // This exists because the mutation found it missing. Dropping the
        // escaping from the sink's alarm arm left every logging test green.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        let auditor = Auditor::new();
        let name = SecretName::new("BASE_URL_PASSWORD");
        auditor.register_secret(name, &["hunter2-longer"]);
        let (_, alarm) = auditor.audit(
            "GET https://bob:hunter2-longer@example.com/v1",
            "magi_core::http\nSECURITY: a line that was never emitted",
            None,
            0,
        );
        let alarm = alarm.expect("a live secret must alarm");
        assert_eq!(
            appender.submit(Queued::Alarm(alarm), Priority::High, 0),
            Submitted::Queued
        );
        std::thread::sleep(std::time::Duration::from_millis(300));

        let path = dir.path().join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let written = std::fs::read_to_string(&path).unwrap_or_default();

        // The fixture must have written, or counting lines proves nothing.
        assert!(
            written.contains("SECURITY"),
            "no alarm reached the file: {written:?}"
        );
        assert_eq!(
            written.lines().filter(|l| !l.trim().is_empty()).count(),
            1,
            "the target's newline forged a second line: {written:?}"
        );
        appender.shutdown();
    }

    #[test]
    fn concurrent_threads_do_not_interleave_the_chunks_of_one_event() {
        let dir = tempdir().unwrap();
        let appender = Arc::new(DailyAppender::new(dir.path()).unwrap());

        // Eight emitters, each sending an event large enough to be chunked into
        // several lines. If a CHUNK travelled the channel instead of the whole
        // event, two of these would interleave and no event's lines would be
        // contiguous.
        let mut handles = Vec::new();
        for t in 0..8u8 {
            let appender = Arc::clone(&appender);
            handles.push(std::thread::spawn(move || {
                // Words with spaces, not one long run: an unbroken
                // alphanumeric run is exactly what `match_generic_secret_run`
                // treats as a secret, and a fixture like that comes back as a
                // single `***` — the redactor working correctly on a bad fixture.
                let body = format!("thread {t} says something here ").repeat(300);
                let (item, len) = line(&body);
                appender.submit(item, Priority::Low, len);
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        Arc::try_unwrap(appender).ok().unwrap().shutdown();

        let path = dir.path().join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let written = std::fs::read_to_string(&path).unwrap();

        // **Grouped by CONTENT, not by the `id=` marker.** The writer stamps
        // the id, so grouping by it is a guardian that cannot fail: each
        // `write()` call produces its own id by construction and its lines are
        // contiguous whatever the channel carried. The marker each thread puts
        // in its own body is the only thing the EMITTER controls, and therefore
        // the only thing that can tell "one whole event was queued" apart from
        // "its chunks were queued separately". Verified by mutation: with the
        // emitter chunking, the id version stayed green and this one goes red.
        let owner = |l: &str| (0..8u8).find(|t| l.contains(&format!("thread {t} says")));
        let mut seen: Vec<u8> = Vec::new();
        let mut last: Option<u8> = None;
        for l in written.lines() {
            let Some(t) = owner(l) else { continue };
            if last == Some(t) {
                continue;
            }
            assert!(
                !seen.contains(&t),
                "thread {t}'s lines resumed after another thread's: they interleaved"
            );
            seen.push(t);
            last = Some(t);
        }
        assert!(
            seen.len() >= 2,
            "the fixture must actually produce several events, got {}",
            seen.len()
        );
        assert!(
            written.lines().count() > seen.len(),
            "and they must actually be CHUNKED, or contiguity is trivially true"
        );
    }

    #[test]
    fn an_ordinary_event_is_written_even_when_the_writer_parked_first() {
        // **The defect this guards was real and shipped-shaped.** The writer
        // blocked on the PRIORITY channel, so an ordinary event that arrived
        // after it parked was never written — and most sessions emit nothing but
        // ordinary events. It looked like a test artefact when the canary found
        // it: an empty log directory after a single `info!`.
        //
        // The ordering is the fixture: the writer must be PARKED before the
        // event is offered. Offer first and the writer picks it up on its last
        // lap, and the bug is invisible.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(150));

        let (item, len) = line("an ordinary line, no alarm, no priority");
        assert_eq!(appender.submit(item, Priority::Low, len), Submitted::Queued);
        std::thread::sleep(std::time::Duration::from_millis(300));

        let path = dir.path().join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        let written = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            written.contains("an ordinary line"),
            "a low-priority event after the park was never written: {written:?}"
        );
        appender.shutdown();
    }

    /// A target that fails its first `n` writes and then succeeds.
    struct FlakyWriter {
        remaining_failures: usize,
        writes: usize,
    }

    impl ItemWriter for FlakyWriter {
        fn write_item(&mut self, _: &Queued) -> std::io::Result<()> {
            self.writes += 1;
            if self.remaining_failures > 0 {
                self.remaining_failures -= 1;
                return Err(std::io::Error::other("the disk said no"));
            }
            Ok(())
        }
    }

    #[test]
    fn one_transient_write_failure_is_retried_and_the_event_survives() {
        // REQ-L37's actual behaviour, which had no test: the retry, the second
        // attempt, and the event reaching the target. What stood here asserted
        // only that WRITER_RETRY_COOLDOWN < WRITER_STALL_SECS -- a comparison of
        // two constants that can fail only if somebody edits a constant, under a
        // name promising the behaviour it never touched.
        //
        // The cooldown is injected so this costs milliseconds rather than the
        // thirty seconds production waits, the same argument `submit_at` already
        // makes for the clock. Thirty seconds is why this went untested.
        let hb = std::sync::Mutex::new(std::time::Instant::now());
        let mut sink = FlakyWriter {
            remaining_failures: 1,
            writes: 0,
        };
        let (item, _) = line("an event that must survive one bad write");

        let kept =
            write_with_one_retry(&mut sink, &item, &hb, std::time::Duration::from_millis(10));

        assert!(kept, "one transient failure must not shut the branch down");
        assert_eq!(
            sink.writes, 2,
            "the retry is ONE retry, not zero and not two"
        );
    }

    #[test]
    fn a_second_failure_shuts_the_file_branch_down_for_good() {
        // The other half of REQ-L37: the retry is ONE. A target that is still
        // broken after the cooldown is not transient, and continuing to try it
        // per event turns a dead disk into a busy loop.
        let hb = std::sync::Mutex::new(std::time::Instant::now());
        let mut sink = FlakyWriter {
            remaining_failures: 99,
            writes: 0,
        };
        let (item, _) = line("an event nobody will read");

        let kept =
            write_with_one_retry(&mut sink, &item, &hb, std::time::Duration::from_millis(10));

        assert!(!kept, "a persistent failure must shut the branch down");
        assert_eq!(sink.writes, 2, "and it must stop after the one retry");
    }

    #[test]
    fn the_mark_stays_fresh_across_the_cooldown() {
        // The subtle half. During the cooldown the writer consumes nothing, so
        // events pile up and the queue is non-empty; an emitter reading a stale
        // mark against a full queue declares a hang, which is PERMANENT. The
        // transient disk error the retry exists for would end in permanent
        // shutdown before the retry ever happened.
        let hb = std::sync::Mutex::new(
            std::time::Instant::now() - std::time::Duration::from_secs(WRITER_STALL_SECS * 2),
        );
        let stale = *hb.lock().unwrap();
        let mut sink = FlakyWriter {
            remaining_failures: 1,
            writes: 0,
        };
        let (item, _) = line("an event during the cooldown");

        write_with_one_retry(&mut sink, &item, &hb, std::time::Duration::from_millis(10));

        assert!(
            *hb.lock().unwrap() > stale,
            "the mark went into the cooldown stale and came out stale"
        );
    }

    #[test]
    fn the_byte_counter_does_not_drift_on_discards() {
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        // An event that alone exceeds the channel budget is refused, and the
        // refusal must leave the counter exactly where it was.
        let before = appender.reserved(Priority::Low);
        let (item, _) = line("x");
        assert_eq!(
            appender.submit(item, Priority::Low, LOG_CHANNEL_LOW_BYTES + 1),
            Submitted::DroppedOversized
        );
        assert_eq!(
            appender.reserved(Priority::Low),
            before,
            "a refused reservation must be given back, or the channel ends up \
             rejecting with capacity to spare"
        );
        assert_eq!(appender.dropped(), 1);
        appender.shutdown();
    }

    #[test]
    fn an_oversized_event_is_discarded_before_being_audited() {
        // The size check runs BEFORE the audit: auditing an event that is about
        // to be thrown away pays for the whole scan to discard the result.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();
        let auditor = Auditor::new();
        auditor.register_secret(SecretName::new("K"), &["never-scanned-value"]);

        let huge = LOG_CHANNEL_HIGH_BYTES + 1;
        let (item, _) = line("placeholder");
        assert_eq!(
            appender.submit(item, Priority::High, huge),
            Submitted::DroppedOversized,
            "it is refused on size, not on content"
        );
        assert_eq!(appender.reserved(Priority::High), 0);
        appender.shutdown();
    }

    #[test]
    fn the_stall_check_runs_even_when_the_send_would_have_failed() {
        // **The order is observable and it is the point.** Run after the send, a
        // full channel takes the emitter down the discard path and it never
        // looks at the mark — so exactly when the writer is hung, which is when
        // the queue fills, the detector stops running. The hang becomes
        // undetectable in its own symptom.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        // A reservation larger than the budget: the send WOULD have been
        // refused. The stall check must still have run and won.
        let stale =
            std::time::Instant::now() + std::time::Duration::from_secs(WRITER_STALL_SECS + 1);
        let (item, _) = line("x");
        assert_eq!(
            appender.submit_at(item, Priority::Low, LOG_CHANNEL_LOW_BYTES + 1, stale),
            Submitted::WriterHung,
            "the hang must be reported, not the oversize: the check runs first"
        );
        appender.shutdown();
    }

    #[test]
    fn a_declared_hang_is_permanent_and_does_not_retry() {
        // A finished thread no longer exists; a hung one is still inside a
        // `write` with its file open, and safe Rust can neither kill nor join
        // it. Recreating it would leave two threads on the same file.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();
        let stale =
            std::time::Instant::now() + std::time::Duration::from_secs(WRITER_STALL_SECS + 1);
        let (a, len) = line("first");
        assert_eq!(
            appender.submit_at(a, Priority::Low, len, stale),
            Submitted::WriterHung
        );
        // Now with a perfectly fresh clock: still hung.
        let (b, len) = line("second");
        assert_eq!(
            appender.submit_at(b, Priority::Low, len, std::time::Instant::now()),
            Submitted::WriterHung,
            "the file layer never comes back"
        );
        appender.shutdown();
    }

    #[test]
    fn an_idle_writer_refreshes_its_mark_instead_of_letting_it_go_stale() {
        // **Asserts the mark ADVANCES, not that no hang was declared.** The
        // second is vacuously true: the mark is stamped at construction and the
        // stall window is a minute, so a test that only submits and checks
        // passes just as well with the park stamp deleted — which was run as a
        // mutation and stayed green. What the stamp buys is that an idle writer
        // does not accumulate staleness without bound, and only a mark that
        // moves shows that.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();
        let at_construction = appender.heartbeat_at().expect("a mark exists");

        // The writer reaches its park and stamps there. Poll for the condition
        // rather than sleeping a guessed interval.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut parked = at_construction;
        while std::time::Instant::now() < deadline {
            parked = appender.heartbeat_at().expect("a mark exists");
            if parked > at_construction {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            parked > at_construction,
            "the writer must stamp when it parks on an empty queue, or an idle process accumulates staleness and the next event kills the layer"
        );
        appender.shutdown();
    }

    /// REQ-L65: the directory and the active file are the owner's alone.
    ///
    /// The spec calls this non-negotiable for MS1 and the reason is plain: the
    /// log is a transcript of everything the agent did, and `create_dir_all`
    /// plus a default `OpenOptions` leave whatever the process umask says --
    /// commonly `0755` and `0644`. On a shared machine that is world-readable.
    ///
    /// **Unix only, and it is not run on the box this was written on.** There is
    /// no umask on Windows for it to correct; the file lives under `.magi/`,
    /// which `magi init` already restricts by ACL to the current user (REQ-H38)
    /// and a file created underneath inherits. Stated rather than left implied,
    /// because a reader on Windows will see this test never execute.
    #[cfg(unix)]
    #[test]
    fn the_directory_and_the_active_file_are_not_left_at_the_umask() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let root = dir.path().join("logs");
        let appender = DailyAppender::new(&root).unwrap();

        let (item, len) = line("a line that forces the day's file open");
        assert_eq!(appender.submit(item, Priority::Low, len), Submitted::Queued);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let path = root.join(crate::logging::rotation::file_name(
            time::OffsetDateTime::now_utc().date(),
        ));
        while !path.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        // The fixture must have produced the file, or the mode of a path that
        // does not exist is not what this asserts.
        assert!(path.exists(), "the day's file was never opened: {path:?}");

        let dir_mode = std::fs::metadata(&root).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            dir_mode,
            crate::logging::OWNER_ONLY_DIR_MODE,
            "the log directory is readable by more than its owner"
        );
        let file_mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            file_mode,
            crate::logging::OWNER_ONLY_FILE_MODE,
            "the active log file is readable by more than its owner"
        );
        appender.shutdown();
    }

    #[test]
    fn the_writer_thread_never_reaches_stderr() {
        // REQ-L39, and a SOURCE check because no behavioural test can see it.
        // The writer is a detached thread that lives for the whole session, so a
        // stderr write from it lands on top of the ratatui frame whenever the
        // TUI holds the alternate screen. A test would have to install a real
        // terminal, take the alternate screen and observe corruption -- and the
        // failure is intermittent even then, because it depends on the writer
        // choosing that moment.
        //
        // `CLAUDE.md` records this as a durable invariant precisely because it
        // looks harmless in review: three of these shipped in this module, one
        // of them added while fixing something else.
        // **Only the production half**, split at the test module. This very
        // test names the macros it forbids, so scanning the whole file makes it
        // its own first offender -- which it did, on the first run.
        let whole = include_str!("appender.rs");
        // Split on the MODULE marker at column zero, not on the attribute: an
        // attribute of the same name sits on a method above `spawn_writer`, so
        // splitting on it truncated the production half to nothing and the
        // scan came back empty — a guardian that passes by having nothing to
        // look at.
        let (source, _) = whole
            .split_once(
                "
#[cfg(test)]
mod tests {",
            )
            .expect("this module has a test section");
        // `println!` also matches `eprintln!`, which is the one that mattered
        // here, but `dbg!` writes to stderr and shares neither spelling. A
        // guardian that names some of the ways to reach a terminal is a
        // guardian for the ways somebody happened to think of.
        let offenders: Vec<&str> = source
            .lines()
            .filter(|l| l.contains("println!") || l.contains("dbg!"))
            .collect();
        assert!(
            offenders.is_empty(),
            "this module writes to a terminal the TUI may own: {offenders:?}"
        );
        // The fixture must have read the file, or an empty haystack proves
        // nothing about what is in it.
        assert!(
            source.contains("fn spawn_writer"),
            "the source check read the wrong half of the wrong file"
        );
    }

    #[test]
    fn the_writer_gives_the_bytes_back_so_the_budget_is_not_a_lifetime_quota() {
        // **The defect this guards makes the log die silently.** `commit()` marks
        // the reservation so `Drop` does not subtract, on the promise that the
        // writer releases it once written. If the writer never does, `reserved`
        // only ever grows: past the channel's budget every later event takes the
        // `DroppedFull` path forever, and nothing is written again for the life
        // of the process. At a typical ~120-byte INFO line that is a few hundred
        // thousand events -- a long session, not a hypothetical.
        //
        // The unit test below (`a_reservation_dropped_without_committing_...`)
        // is correct about the Channel in isolation: there, committing IS handing
        // the bytes on. What was missing is the other half of that hand-off, and
        // no test at that level could see it.
        let dir = tempdir().unwrap();
        let appender = DailyAppender::new(dir.path()).unwrap();

        // Far under the budget, so a refusal here can only mean a leak.
        let mut submitted = 0usize;
        for i in 0..200 {
            let (item, len) = line(&format!("event number {i}"));
            assert_eq!(
                appender.submit(item, Priority::Low, len),
                Submitted::Queued,
                "event {i} was refused with the budget nowhere near spent"
            );
            submitted += len;
        }
        // The fixture must have reserved something, or asserting it returns to
        // zero is asserting that nothing happened.
        assert!(submitted > 0, "the fixture reserved no bytes at all");

        // **Wait on the CONDITION with a generous failure deadline**, never on a
        // duration: under the `heavy` nextest group a fixed sleep is a guess.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while appender.reserved(Priority::Low) != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        assert_eq!(
            appender.reserved(Priority::Low),
            0,
            "the writer wrote {submitted} bytes and returned none of them, so the \
             channel's byte budget is a lifetime quota rather than a depth"
        );
        assert_eq!(appender.dropped(), 0, "nothing should have been refused");
        appender.shutdown();
    }

    #[test]
    fn a_reservation_dropped_without_committing_returns_its_bytes() {
        // The guard, not the manual release: between `fetch_add` and `try_send`
        // there is code that can unwind, and a hand-written release is lost on
        // that path forever.
        let channel = Channel {
            tx: sync_channel(1).0,
            budget: 100,
            reserved: Arc::new(AtomicUsize::new(0)),
        };
        {
            let _r = channel.reserve(40).expect("fits");
            assert_eq!(channel.reserved.load(Ordering::Acquire), 40);
        }
        assert_eq!(
            channel.reserved.load(Ordering::Acquire),
            0,
            "dropping without committing must return the bytes"
        );
        let kept = channel.reserve(40).expect("fits");
        kept.commit();
        assert_eq!(
            channel.reserved.load(Ordering::Acquire),
            40,
            "committing hands them to the writer instead"
        );
    }
}
