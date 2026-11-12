// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! Encrypted store of remembered review findings (RV07(4), RV08, RV19).
//!
//! The title and detail of a finding are encrypted with the data key, exactly like
//! [`crate::memory::review_docs::ReviewDocStore`]; the file, line, commit, status and the two
//! timestamps stay in the clear so rows can be selected and ordered without decrypting anything.
//!
//! # Lock discipline
//!
//! The connection lock is never held across an `.await` (nothing here is async) and never taken
//! while the data-key lock is held (R-V08): rows are encrypted before the lock is taken and
//! decrypted after it is released.
//!
//! # Examples
//!
//! ```ignore
//! let store = ReviewHistoryStore::new(conn, dek)?;
//! store.record(&entries)?;
//! let (due, total) = store.due(200)?;
//! let (validated, cut) = store.validated_on_line(&["src/a.rs".to_string()], &|_| true)?;
//! ```

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, TransactionBehavior};
use zeroize::Zeroizing;

use crate::memory::error::MemoryError;
use cryptovault::CryptoVault;
use magi_rs::review::history::{HistoryEntry, HistoryStatus};
use magi_rs::review::secrets::outward_text;
use magi_rs::vault::MaskedDek;

/// The tracing target of this module's WARNs (the review's).
const LOG_TARGET: &str = "magi_rs::review";

/// Files per `validated_on_line` query: one parameter each plus the fixed `LIMIT ?`, so 401 in
/// all, well under the 999-parameter limit of older SQLite builds (raise it to 998 at most).
const FILES_PER_QUERY: usize = 400;

/// Validated entries one batch reads at most (a starting point, not a measurement): the context
/// keeps only what fits after the code and the documentation, so more would be decrypted to be
/// trimmed. Past it the newest are kept and the cut is declared.
pub const MAX_HISTORY_PER_BATCH: usize = 32;

/// Raw rows (still encrypted, never decrypted here) one file batch of the context read scans
/// for entries on `HEAD`'s line, eight times [`MAX_HISTORY_PER_BATCH`]: a starting point. Past it
/// the read declares a cut rather than scan the whole history.
pub const MAX_HISTORY_SCAN: usize = 8 * MAX_HISTORY_PER_BATCH;

/// Rows the `due` count reads at most: its `total` (the review's backlog) saturates here, so the
/// count costs O(this) index entries, not O(history). Fifty times the per-review reclassification
/// cap, a starting point: a backlog beyond it only says "at least this many".
pub const MAX_COUNTED_HISTORY: usize = 10_000;

/// The columns every read selects, in the order [`RawRow`] reads them.
const COLUMNS: &str = "id, file, line, reviewed_commit, finding_blob, status, created_at";

/// The rows [`ReviewHistoryStore::due`] reads: every entry, least recently checked first, at
/// most `?1`.
fn due_sql() -> String {
    format!(
        "SELECT {COLUMNS} FROM code_review_history \
         ORDER BY checked_at, created_at, id LIMIT ?1"
    )
}

/// The rows [`ReviewHistoryStore::validated_on_files`] reads for one batch of `files` files:
/// the validated ones, newest first, at most the last parameter.
fn validated_sql(files: usize) -> String {
    let marks = vec!["?"; files].join(", ");
    format!(
        "SELECT {COLUMNS} FROM code_review_history \
         WHERE status IN ('{attended}', '{weak}') AND file IN ({marks}) \
         ORDER BY created_at DESC, id LIMIT ?",
        attended = HistoryStatus::Attended.as_str(),
        weak = HistoryStatus::WeakReject.as_str(),
    )
}

/// One row as stored: the finding text still encrypted.
struct RawRow {
    id: String,
    file: String,
    line: Option<i64>,
    reviewed_commit: String,
    finding_blob: Vec<u8>,
    status: String,
    /// Unix seconds of the recording, for the newest-first order of the context read.
    created_at: i64,
}

/// Encrypted store of remembered findings, over the shared connection.
pub struct ReviewHistoryStore {
    conn: Arc<Mutex<Connection>>,
    vault: CryptoVault,
    /// The data key, masked in RAM; the lock is never held across the connection lock (R-V08).
    dek: Mutex<MaskedDek>,
    /// Rows a read skipped because they did not decode (each one WARNed).
    corrupt: AtomicUsize,
}

fn db_err(e: rusqlite::Error) -> MemoryError {
    MemoryError::Storage(e.to_string())
}

/// Seconds since the Unix epoch; 0 if the clock is before it.
fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

impl ReviewHistoryStore {
    /// The shared connection, for tests that read raw rows.
    #[cfg(test)]
    pub(crate) fn connection_for_test(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    /// Opens the store over `conn`, making sure the schema exists.
    ///
    /// A database written before v0.25.0 is refused here too, read-only and with the message
    /// that names the missing table (RV19, never a migration), whatever open produced `conn`:
    /// the schema call is `CREATE ... IF NOT EXISTS` and would otherwise upgrade it in place.
    ///
    /// # Arguments
    /// * `conn` - the connection shared with `EncryptedSqliteMemory`.
    /// * `dek` - a masked duplicate of the data key.
    ///
    /// # Errors
    /// [`MemoryError::UnsupportedSchema`] for a database older than v0.25.0;
    /// [`MemoryError::Storage`] if the schema DDL fails.
    pub fn new(conn: Arc<Mutex<Connection>>, dek: MaskedDek) -> Result<Self, MemoryError> {
        {
            let c = conn.lock().unwrap_or_else(|p| p.into_inner());
            crate::system::database::init_store_schema(&c)?;
        }
        Ok(Self {
            conn,
            vault: CryptoVault::default(),
            dek: Mutex::new(dek),
            corrupt: AtomicUsize::new(0),
        })
    }

    fn locked_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Encrypts `plaintext` with the data key. Never call while holding the connection lock.
    fn seal(&self, plaintext: &str) -> Result<Vec<u8>, MemoryError> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.encrypt_with_key(k, plaintext))
            .map(String::into_bytes)
            .map_err(|e| MemoryError::Crypto(e.to_string()))
    }

    /// Decrypts a blob written by [`Self::seal`]. Same lock contract.
    fn unseal(&self, blob: &[u8]) -> Result<Zeroizing<String>, MemoryError> {
        let text = std::str::from_utf8(blob)
            .map_err(|e| MemoryError::Crypto(format!("stored blob is not text: {e}")))?;
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.decrypt_with_key(k, text))
            .map_err(|e| MemoryError::Crypto(e.to_string()))
    }

    /// The encrypted `(title, detail)` pair, as a self-describing JSON array: merged titles
    /// carry newlines, which any separator-based encoding would split.
    fn seal_text(&self, entry: &HistoryEntry) -> Result<Vec<u8>, MemoryError> {
        let json = serde_json::to_string(&(&entry.title, &entry.detail))
            .map_err(|e| MemoryError::Storage(format!("history text serialization: {e}")))?;
        self.seal(&json)
    }

    /// Rebuilds an entry from a raw row, decrypting its text (off the connection lock).
    fn decode(&self, row: RawRow) -> Result<HistoryEntry, MemoryError> {
        let json = self.unseal(&row.finding_blob)?;
        let (title, detail): (String, String) = serde_json::from_str(json.as_str())
            .map_err(|e| MemoryError::Storage(format!("history text deserialization: {e}")))?;
        let status = HistoryStatus::parse(&row.status).ok_or_else(|| {
            MemoryError::Storage(format!(
                "history entry `{}` has the unknown status `{}`",
                row.id, row.status
            ))
        })?;
        // MAGI FIX (MS3 §6 loop4): a line out of range read silently as no line; now an error.
        let line = row
            .line
            .map(|l| {
                u32::try_from(l).map_err(|_| {
                    MemoryError::Storage(format!(
                        "history entry `{}` has the line `{l}`, out of range",
                        row.id
                    ))
                })
            })
            .transpose()?;
        Ok(HistoryEntry {
            id: row.id,
            file: row.file,
            line,
            reviewed_commit: row.reviewed_commit,
            title,
            detail,
            status,
        })
    }

    /// How many rows this store's reads skipped because they did not decode.
    pub fn corrupt_rows(&self) -> usize {
        self.corrupt.load(Ordering::SeqCst)
    }

    /// The rows that decode; each one that does not (an unknown status, a line out of range, a
    /// blob that does not decrypt) is skipped with one WARN and counted in
    /// [`Self::corrupt_rows`], so one bad row never takes the whole read
    /// (`corrupt_row_is_skipped`). O(rows) decryptions.
    ///
    /// Declared (MS3 §6 loop12): the skipped row is never in the caller's `touched`, so it stays
    /// first in the due queue, is read and WARNed again each review and takes one of the 200 slots
    /// of that review. Rows come only from this store's own atomic writes under the never-delete
    /// open, so a row that fails to decode means a damaged file or a foreign key, and the WARN
    /// per review is the signal; rotating it away would hide it.
    // MAGI FIX (MS3 §6 loop11): one corrupt row failed the whole read.
    fn decode_all(&self, rows: Vec<RawRow>) -> Result<Vec<HistoryEntry>, MemoryError> {
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            match self.decode(row) {
                Ok(entry) => out.push(entry),
                Err(e) => {
                    self.corrupt.fetch_add(1, Ordering::SeqCst);
                    let error = outward_text(&e.to_string());
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        error = error.as_str(),
                        "a code-review history row could not be decoded and was skipped"
                    );
                }
            }
        }
        Ok(out)
    }

    /// [`Self::apply_review`] of `entries` alone (tests).
    ///
    /// # Errors
    /// As [`Self::apply_review`].
    #[cfg(test)]
    pub(crate) fn record(&self, entries: &[HistoryEntry]) -> Result<usize, MemoryError> {
        self.apply_review(&[], &[], 0, entries)
    }
}

/// Records new entries over `tx` (`sealed`: their encrypted text, in order; the caller commits);
/// an `id` already stored is kept as it is (first sighting wins), so its `reviewed_commit` stays
/// the one `classify` compares from. Replacing it would restart the clock on every review, and a
/// finding nobody attends would never become a weak reject.
///
/// The one exception is an `attended` entry: the code it pointed at has changed, so a new
/// finding on the same `file:line` (the same id) is a new finding. It replaces the old row whole
/// (text, commit, status) and is due first; otherwise it would be dropped while the old text kept
/// being served as validated context. O(entries). Returns the rows inserted or replaced, which
/// can be fewer than `entries.len()`: an id already stored and not `attended` is kept as it is
/// (first sighting wins), so callers must not read the count as the input's length.
fn record_in(
    tx: &Connection,
    entries: &[HistoryEntry],
    sealed: &[Vec<u8>],
) -> Result<usize, MemoryError> {
    let created_at = now_secs();
    let mut inserted = 0;
    {
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO code_review_history \
                     (id, file, line, reviewed_commit, finding_blob, status, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                     ON CONFLICT(id) DO UPDATE SET file = excluded.file, \
                     line = excluded.line, reviewed_commit = excluded.reviewed_commit, \
                     finding_blob = excluded.finding_blob, status = excluded.status, \
                     created_at = excluded.created_at, checked_at = 0 \
                     WHERE code_review_history.status = ?8",
                )
                .map_err(db_err)?;
            for (e, blob) in entries.iter().zip(sealed) {
                inserted += stmt
                    .execute(params![
                        e.id,
                        e.file,
                        e.line.map(i64::from),
                        e.reviewed_commit,
                        blob,
                        e.status.as_str(),
                        created_at,
                        HistoryStatus::Attended.as_str()
                    ])
                    .map_err(db_err)?;
            }
        }
    }
    Ok(inserted)
}

impl ReviewHistoryStore {
    /// Every entry, decrypted. O(n). No order is promised: reclassification reads
    /// [`Self::due`]. Test-only: no review path reads the whole history.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] or [`MemoryError::Crypto`].
    #[cfg(test)]
    pub(crate) fn all(&self) -> Result<Vec<HistoryEntry>, MemoryError> {
        let rows = self.raw_rows(&format!("SELECT {COLUMNS} FROM code_review_history"), [])?;
        self.decode_all(rows)
    }

    /// Reads raw rows under the lock; the caller decrypts them after it is released.
    fn raw_rows(&self, sql: &str, args: impl rusqlite::Params) -> Result<Vec<RawRow>, MemoryError> {
        raw_rows_in(&self.locked_conn(), sql, args)
    }

    /// At most `limit` entries, of any status, least recently checked first (`checked_at`,
    /// then `created_at`, then `id`), and how many entries exist in total, counted up to
    /// [`MAX_COUNTED_HISTORY`] (past it, `total` is that bound).
    ///
    /// Every status is due: a weak reject can become attended, and an attended entry whose
    /// change was reverted goes back to a weak reject. An entry that never resolves is checked
    /// again only after all the others, so it never starves them: that rotation needs the caller
    /// to touch ([`Self::apply_review`]) every entry it checked, changed or not, and `record_review` does
    /// (`record_review_touches_an_entry_whose_status_did_not_change`, and here, through that
    /// caller, `the_review_caller_touches_every_checked_entry_so_the_tail_is_never_starved`). Decrypts at most `limit`
    /// rows, so the cost of a review does not depend on the size of the history. The count and
    /// the rows are read under one lock and in one read transaction, one SQLite snapshot, so
    /// `total` matches the rows returned even while another process writes.
    ///
    /// Complexity: the rows come off the `code_review_history_due` index in order, stopping at
    /// `limit` (no sort); the count reads at most [`MAX_COUNTED_HISTORY`] entries of the
    /// narrowest index, with no row read or decrypted; O(limit) to decrypt. Pinned by
    /// `the_bounded_history_reads_are_served_by_their_indexes_not_by_a_full_sort`.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] or [`MemoryError::Crypto`].
    pub fn due(&self, limit: usize) -> Result<(Vec<HistoryEntry>, usize), MemoryError> {
        self.due_bounded(limit, MAX_COUNTED_HISTORY)
    }

    /// [`Self::due`] with the count stopped at `count_cap`: `total` is at most `count_cap`
    /// (`the_due_count_stops_at_its_bound_instead_of_reading_the_whole_history`).
    // MAGI FIX (MS3 §6 loop7): the count read every row on every review, O(history).
    fn due_bounded(
        &self,
        limit: usize,
        count_cap: usize,
    ) -> Result<(Vec<HistoryEntry>, usize), MemoryError> {
        let cap = i64::try_from(count_cap).unwrap_or(i64::MAX);
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        // MAGI FIX (MS3 §6 loop4): the count and the rows took two locks, so a write between
        // them made `total` disagree with the rows.
        // MAGI FIX (MS3 §6 loop5): one lock kept other threads out, not another process: the two
        // reads now share one read transaction, so one snapshot.
        let (total, rows) = {
            let c = self.locked_conn();
            let snapshot = c.unchecked_transaction().map_err(db_err)?;
            let total: i64 = snapshot
                .query_row(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM code_review_history LIMIT ?1)",
                    [cap],
                    |r| r.get(0),
                )
                .map_err(db_err)?;
            let rows = raw_rows_in(&snapshot, &due_sql(), [limit])?;
            snapshot.commit().map_err(db_err)?;
            (total, rows)
        };
        Ok((
            self.decode_all(rows)?,
            usize::try_from(total).unwrap_or(usize::MAX),
        ))
    }

    /// The validated (`attended` / `weak_reject`) entries on `files`, by SQL over the clear
    /// `file` and `status` columns, newest first and then by id (a stable order, so the tail the
    /// context trims is the same on every run), at most [`MAX_HISTORY_PER_BATCH`], and whether
    /// more existed. Only the returned rows are decrypted. This is the context read, once per
    /// batch. The files go in batches of [`FILES_PER_QUERY`] so no query passes SQLite's
    /// parameter limit (a mass rename touches thousands); each batch reads at most one row past
    /// the bound, raw.
    ///
    /// The text comes back decrypted, as it was stored: already masked
    /// (`entries_from_review` masks it before it is recorded,
    /// `a_finding_is_stored_with_its_secrets_masked_and_one_quoting_a_key_block_is_not_stored`), and
    /// masked again by the context read before it can leave the process
    /// (`a_history_entry_quoting_a_secret_reaches_the_context_masked`, `memory::review_rag`).
    ///
    /// Complexity: each batch searches the `code_review_history_by_file` index for its files'
    /// validated rows (never a scan of the history) and sorts only those, O(m log m) for the `m`
    /// such rows; at most 2 x (bound + 1) raw rows held at once (the newest are kept after each
    /// file batch), O(bound) decryptions.
    ///
    /// Test-only wrapper (MS3 §6 loop12): it accepts every reviewed commit. The production
    /// context read is [`Self::validated_on_line`], which adds the `HEAD` ancestry filter
    /// (`on_heads_line`) before the bound; this one is NOT the production read.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] or [`MemoryError::Crypto`].
    #[cfg(test)]
    pub(crate) fn validated_on_files(
        &self,
        files: &[String],
    ) -> Result<(Vec<HistoryEntry>, bool), MemoryError> {
        self.validated_on_line(files, &|_: &str| true)
    }

    /// [`Self::validated_on_files`] restricted to the entries whose reviewed commit `on_line`
    /// accepts (the context read passes its `HEAD` ancestry check), applied to the raw rows
    /// **before** the bound, so entries of another branch never fill it
    /// (`off_line_never_starves`). Each file batch reads at most [`MAX_HISTORY_SCAN`] raw rows
    /// (no decryption); when one reaches it with fewer than the bound kept, more may lie past it
    /// and the cut is declared. O(scan) `on_line` calls (the caller memoizes them per commit).
    ///
    /// # Errors
    /// As [`Self::validated_on_files`].
    // MAGI FIX (MS3 §6 loop11): the cap came before the ancestry filter.
    pub fn validated_on_line(
        &self,
        files: &[String],
        on_line: &dyn Fn(&str) -> bool,
    ) -> Result<(Vec<HistoryEntry>, bool), MemoryError> {
        // MAGI FIX (MS3 §6 loop4): no order (the trimmed tail varied run to run) and no bound
        // (every validated row of the files was decrypted for every batch).
        let distinct: Vec<&String> = files.iter().collect::<BTreeSet<_>>().into_iter().collect();
        let scan = i64::try_from(MAX_HISTORY_SCAN).unwrap_or(i64::MAX);
        let newest_first = |a: &RawRow, b: &RawRow| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        };
        let keep = MAX_HISTORY_PER_BATCH.saturating_add(1);
        let (mut raw, mut saturated) = (Vec::new(), false);
        for batch in distinct.chunks(FILES_PER_QUERY) {
            let mut args: Vec<&dyn rusqlite::ToSql> =
                batch.iter().map(|f| *f as &dyn rusqlite::ToSql).collect();
            args.push(&scan);
            let rows = self.raw_rows(
                &validated_sql(batch.len()),
                rusqlite::params_from_iter(args),
            )?;
            saturated |= rows.len() >= MAX_HISTORY_SCAN;
            raw.extend(rows.into_iter().filter(|r| on_line(&r.reviewed_commit)));
            // MAGI FIX (MS3 §6 loop7): every batch's rows were kept until the end, linear in the
            // files; the newest bound + 1 are kept after each one, so the buffer is O(bound).
            raw.sort_by(newest_first);
            raw.truncate(keep);
        }
        let cut = raw.len() > MAX_HISTORY_PER_BATCH || saturated;
        raw.truncate(MAX_HISTORY_PER_BATCH);
        Ok((self.decode_all(raw)?, cut))
    }

    /// [`Self::apply_review`] touching `ids` alone (tests): `checked_at = now`, all or nothing.
    ///
    /// # Errors
    /// As [`Self::apply_review`].
    #[cfg(test)]
    pub(crate) fn touch(&self, ids: &[String], now: i64) -> Result<(), MemoryError> {
        self.apply_review(&[], ids, now, &[]).map(|_| ())
    }

    /// The writes of one review in **one** transaction, all or nothing: each status in `updates`,
    /// `checked_at = now` on `touched`, then `entries` recorded (`record_in`: first sighting wins,
    /// an `attended` id replaced) (`apply_review_is_atomic`). Encryption runs before the lock.
    /// O(writes). The caller touches every entry it checked, changed or not, so the tail rotates.
    ///
    /// # Returns
    /// The rows `entries` inserted or replaced.
    ///
    /// Declared (MS3 §6 loop12): between `due` and this call another process may delete or replace
    /// a row; the unknown id then fails typed and the whole review's history update rolls back
    /// (`apply_review_is_atomic`), so a lost write is a review that records nothing, never a
    /// half-applied one, and the next review reclassifies the same queue. It matters only when two
    /// reviews of one database run at once, which the shared TUI runner and `magi query` do not
    /// do in the same process.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] naming an unknown id or a failing statement (nothing written);
    /// [`MemoryError::Crypto`].
    // MAGI FIX (MS3 §6 loop10): one statement each, so a failure left a review half applied.
    pub fn apply_review(
        &self,
        updates: &[(String, HistoryStatus)],
        touched: &[String],
        now: i64,
        entries: &[HistoryEntry],
    ) -> Result<usize, MemoryError> {
        let sealed: Vec<Vec<u8>> = entries
            .iter()
            .map(|e| self.seal_text(e))
            .collect::<Result<_, _>>()?;
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        let set = |sql: &str, a: &dyn rusqlite::ToSql, id: &str| match tx
            .execute(sql, params![a, id])
            .map_err(db_err)?
        {
            0 => Err(unknown_id(id)),
            _ => Ok(()),
        };
        for (id, status) in updates {
            set(
                "UPDATE code_review_history SET status = ?1 WHERE id = ?2",
                &status.as_str(),
                id,
            )?;
        }
        for id in touched {
            set(
                "UPDATE code_review_history SET checked_at = ?1 WHERE id = ?2",
                &now,
                id,
            )?;
        }
        let inserted = record_in(&tx, entries, &sealed)?;
        tx.commit().map_err(db_err)?;
        Ok(inserted)
    }

    /// [`Self::apply_review`] setting one status (tests).
    ///
    /// # Errors
    /// As [`Self::apply_review`].
    #[cfg(test)]
    pub(crate) fn update_status(&self, id: &str, status: HistoryStatus) -> Result<(), MemoryError> {
        self.apply_review(&[(id.to_string(), status)], &[], 0, &[])
            .map(|_| ())
    }
}

/// Reads raw rows over `c`, a connection whose lock the caller holds.
fn raw_rows_in(
    c: &Connection,
    sql: &str,
    args: impl rusqlite::Params,
) -> Result<Vec<RawRow>, MemoryError> {
    let mut stmt = c.prepare(sql).map_err(db_err)?;
    let rows = stmt
        .query_map(args, |r| {
            Ok(RawRow {
                id: r.get(0)?,
                file: r.get(1)?,
                line: r.get(2)?,
                reviewed_commit: r.get(3)?,
                finding_blob: r.get(4)?,
                status: r.get(5)?,
                created_at: r.get(6)?,
            })
        })
        .map_err(db_err)?;
    rows.collect::<Result<_, _>>().map_err(db_err)
}

/// The error for an id no row has; never a silent success.
fn unknown_id(id: &str) -> MemoryError {
    MemoryError::Storage(format!("no history entry has the id `{id}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use magi_rs::review::history::{HistoryEntry, HistoryStatus};

    use crate::memory::error::MemoryError;

    fn store_for_test() -> (ReviewHistoryStore, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        (
            ReviewHistoryStore::new(mem.shared_conn(), mem.data_key().expect("dek"))
                .expect("store"),
            tmp,
        )
    }

    fn stored(id: &str, line: Option<u32>, commit: &str) -> HistoryEntry {
        HistoryEntry {
            id: id.into(),
            file: "src/a.rs".into(),
            line,
            reviewed_commit: commit.into(),
            title: format!("title {id}"),
            detail: format!("detail {id}"),
            status: HistoryStatus::Pending,
        }
    }

    fn sorted(mut v: Vec<HistoryEntry>) -> Vec<HistoryEntry> {
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    fn raw_status(store: &ReviewHistoryStore, id: &str) -> String {
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT status FROM code_review_history WHERE id = ?1",
                [id],
                |r| r.get(0),
            )
            .expect("row")
    }

    #[test]
    fn apply_review_is_atomic() {
        // [MAGI S6 MS3 loop10, R2 Balthasar; controller ruling] A review's history writes were
        // one statement each, so a failure part-way left them half applied.
        let (store, _t) = store_for_test();
        store
            .record(&[stored("h1", Some(1), "c1")])
            .expect("record");
        let r = store.apply_review(
            &[("h1".into(), HistoryStatus::Attended)],
            &["nope".into()],
            5,
            &[stored("h2", Some(2), "c1")],
        );
        assert!(r.is_err());
        assert_eq!(raw_status(&store, "h1"), HistoryStatus::Pending.as_str());
        assert_eq!(store.all().expect("all").len(), 1, "h2 not recorded");
    }

    #[test]
    fn a_store_handed_an_older_database_refuses_it_and_creates_nothing() {
        // [MAGI S6 MS3 loop3, R3 Melchior, Balthasar] The refusal of a 0.24.x database (RV19)
        // rested on the caller's checked open: handed a raw connection, `new` created the
        // v0.25.0 tables and upgraded the file in place, so every later open passed.
        let dek = fresh_dek();
        let (conn, _old) = crate::system::database::pre_0_25_connection_for_test();
        let r = ReviewHistoryStore::new(Arc::clone(&conn), dek);
        assert!(
            // [MAGI S6 MS3 loop4, R3 Balthasar] typed, so a caller can tell it from storage.
            matches!(&r, Err(e @ MemoryError::UnsupportedSchema { missing_table: "code_review_docs", .. }) if e.to_string().contains("0.24.x")),
            "{:?}",
            r.as_ref().err()
        );
        assert_eq!(crate::system::database::review_tables_for_test(&conn), 0);
    }

    /// A masked data key of a fresh database.
    fn fresh_dek() -> MaskedDek {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db")
        .data_key()
        .expect("dek")
    }

    /// The `EXPLAIN QUERY PLAN` details of `sql`, with `params` bound.
    fn plan_of(
        store: &ReviewHistoryStore,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Vec<String> {
        let c = store.connection_for_test().lock().expect("lock");
        let mut stmt = c
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .expect("plan");
        stmt.query_map(params, |r| r.get::<_, String>(3))
            .expect("rows")
            .collect::<Result<_, _>>()
            .expect("details")
    }

    #[test]
    fn the_bounded_history_reads_are_served_by_their_indexes_not_by_a_full_sort() {
        // [MAGI S6 MS3 loop5, R3 Caspar, Melchior] The output caps did not bound the SQL work:
        // `due` ordered by (checked_at, created_at, id) with an index led by `status`, so every
        // review sorted the whole history to return at most a cap of rows.
        let (store, _tmp) = store_for_test();
        let due = plan_of(&store, &due_sql(), [32_i64]);
        assert!(
            due.iter()
                .any(|d| d.contains("USING INDEX code_review_history_due")),
            "{due:?}"
        );
        assert!(!due.iter().any(|d| d.contains("TEMP B-TREE")), "{due:?}");
        let validated = plan_of(
            &store,
            &validated_sql(2),
            rusqlite::params!["a.rs", "b.rs", 33_i64],
        );
        assert!(
            validated
                .iter()
                .any(|d| d.contains("USING INDEX code_review_history_by_file")),
            "{validated:?}"
        );
        assert!(
            !validated
                .iter()
                .any(|d| d.starts_with("SCAN code_review_history")),
            "{validated:?}"
        );
    }

    #[test]
    fn recorded_entries_come_back_whole() {
        let (store, _tmp) = store_for_test();
        let entries = vec![stored("i1", Some(7), "c1"), stored("i2", None, "c1")];
        assert_eq!(store.record(&entries).expect("record"), 2);
        assert_eq!(sorted(store.all().expect("all")), sorted(entries));
    }

    #[test]
    fn a_merged_title_and_detail_with_newlines_come_back_whole() {
        // [CP2 MS3 loop 3] entries_from_review joins merged titles with newlines; a separator join would split them.
        let (store, _tmp) = store_for_test();
        let e = HistoryEntry {
            title: "t1\nt2".into(),
            detail: "d1\n\nd2\n".into(),
            ..stored("i1", Some(7), "c1")
        };
        store.record(std::slice::from_ref(&e)).expect("record");
        assert_eq!(store.all().expect("all"), vec![e]);
    }

    #[test]
    fn the_finding_text_is_stored_encrypted() {
        let (store, _tmp) = store_for_test();
        let mut e = stored("i1", Some(7), "c1");
        e.title = "TITLE-PLAINTEXT-MARKER".into();
        e.detail = "DETAIL-PLAINTEXT-MARKER".into();
        store.record(&[e]).expect("record");
        let raw: Vec<u8> = store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row("SELECT finding_blob FROM code_review_history", [], |r| {
                r.get(0)
            })
            .expect("row");
        for marker in [
            &b"TITLE-PLAINTEXT-MARKER"[..],
            &b"DETAIL-PLAINTEXT-MARKER"[..],
        ] {
            assert!(
                !raw.windows(marker.len()).any(|w| w == marker),
                "finding text stored in the clear"
            );
        }
    }

    #[test]
    fn update_status_changes_only_that_entry() {
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1"), stored("i2", Some(9), "c1")])
            .expect("record");
        store
            .update_status("i1", HistoryStatus::Attended)
            .expect("update");
        let all = sorted(store.all().expect("all"));
        assert_eq!(
            (all[0].id.as_str(), all[0].status),
            ("i1", HistoryStatus::Attended)
        );
        assert_eq!(
            (all[1].id.as_str(), all[1].status),
            ("i2", HistoryStatus::Pending)
        );
        // Persisted in the row, in the serde spelling, not only cached in the store.
        assert_eq!(raw_status(&store, "i1"), "attended");
        assert_eq!(raw_status(&store, "i2"), "pending");
    }

    #[test]
    fn updating_an_unknown_id_is_an_error() {
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1")])
            .expect("record");
        let err = store
            .update_status("nope", HistoryStatus::Attended)
            .expect_err("unknown id");
        assert!(
            matches!(&err, MemoryError::Storage(m) if m.contains("nope")),
            "{err}"
        );
        assert_eq!(
            store.all().expect("all")[0].status,
            HistoryStatus::Pending,
            "the known entry is untouched"
        );
    }

    #[test]
    fn due_entries_rotate_so_a_stuck_one_never_starves_the_rest() {
        // [CP2 MS3 loop 6] An entry that stays Pending forever is checked again only after every other due entry.
        let (store, _tmp) = store_for_test();
        store
            .record(&[
                stored("a", Some(1), "c1"),
                stored("b", Some(2), "c1"),
                stored("c", Some(3), "c1"),
                stored("d", Some(4), "c1"),
            ])
            .expect("record");
        store
            .update_status("d", HistoryStatus::Attended)
            .expect("validated");
        let ids = |v: Vec<HistoryEntry>| v.into_iter().map(|e| e.id).collect::<Vec<_>>();
        let (first, total) = store.due(2).expect("due");
        // [CP2 MS3 loop 14 (pasada final)] Attended is due too (a revert can undo it), so the total counts all four.
        assert_eq!(
            (ids(first), total),
            (vec!["a".to_string(), "b".to_string()], 4)
        );
        store
            .touch(&["a".to_string(), "b".to_string()], 100)
            .expect("touch");
        let (second, _) = store.due(2).expect("due");
        assert_eq!(
            ids(second),
            vec!["c".to_string(), "d".to_string()],
            "least recently checked first"
        );
    }

    #[test]
    fn touching_an_unknown_id_is_an_error() {
        // [CP2 MS3 loop 7] Same contract as update_status: never a silent success.
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1")])
            .expect("record");
        let err = store
            .touch(&["i1".to_string(), "nope".to_string()], 5)
            .expect_err("unknown id");
        assert!(
            matches!(&err, MemoryError::Storage(m) if m.contains("nope")),
            "{err}"
        );
    }

    #[test]
    fn only_validated_entries_on_the_given_files_are_read_back() {
        // [CP2 MS3 loop 7] The context read: validated entries of the batch's files; never Pending, never another file.
        let (store, _tmp) = store_for_test();
        let other = HistoryEntry {
            file: "src/b.rs".into(),
            ..stored("o", Some(1), "c1")
        };
        store
            .record(&[
                stored("a", Some(1), "c1"),
                stored("p", Some(2), "c1"),
                stored("w", Some(3), "c1"),
                other,
            ])
            .expect("record");
        store
            .update_status("a", HistoryStatus::Attended)
            .expect("a");
        store
            .update_status("w", HistoryStatus::WeakReject)
            .expect("w");
        store
            .update_status("o", HistoryStatus::Attended)
            .expect("o");
        let ids: Vec<String> = sorted(
            store
                .validated_on_files(&["src/a.rs".to_string()])
                .expect("read")
                .0,
        )
        .into_iter()
        .map(|e| e.id)
        .collect();
        assert_eq!(ids, vec!["a".to_string(), "w".to_string()]);
        assert!(store.validated_on_files(&[]).expect("read").0.is_empty());
    }

    #[test]
    fn out_of_range_line_is_skipped() {
        // [MAGI S6 MS3 loop4, R3 Melchior] The write path is u32-typed, so only a hand edit puts
        // a negative or too large line in the clear column. It read as "no line" and held the
        // entry pending forever; now it is reported, like an unknown status.
        // MAGI FIX (MS3 §6 loop11): reported as a skipped, counted row (WARN), not a failed read.
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("bad", Some(3), "c1")])
            .expect("record");
        for (n, line) in [(1, -5_i64), (2, i64::from(u32::MAX) + 1)] {
            store
                .connection_for_test()
                .lock()
                .expect("lock")
                .execute(
                    "UPDATE code_review_history SET line = ?1 WHERE id = 'bad'",
                    [line],
                )
                .expect("edit");
            assert!(store.all().expect("read").is_empty(), "{line}");
            assert_eq!(store.corrupt_rows(), n, "{line}");
        }
    }

    #[test]
    fn validated_history_comes_in_a_stable_order_and_is_cut_at_its_bound() {
        // [MAGI S6 MS3 loop4, R3 Balthasar] The read had no ORDER BY (the trimmed tail could
        // change between identical runs) and no bound (every validated row of a file was
        // decrypted for every batch). Newest first, then by id; at most
        // MAX_HISTORY_PER_BATCH, the cut said.
        let (store, _tmp) = store_for_test();
        let n = MAX_HISTORY_PER_BATCH + 2;
        let entries: Vec<HistoryEntry> = (0..n)
            .rev()
            .map(|i| stored(&format!("e{i:03}"), Some(1), "c1"))
            .collect();
        store.record(&entries).expect("record");
        for e in &entries {
            store
                .update_status(&e.id, HistoryStatus::Attended)
                .expect("attend");
        }
        let (got, cut) = store
            .validated_on_files(&["src/a.rs".to_string()])
            .expect("read");
        let ids: Vec<String> = got.into_iter().map(|e| e.id).collect();
        let expected: Vec<String> = (0..MAX_HISTORY_PER_BATCH)
            .map(|i| format!("e{i:03}"))
            .collect();
        assert_eq!(ids, expected, "same created_at: by id");
        assert!(cut);
        let (few, cut) = store
            .validated_on_files(&["src/b.rs".to_string()])
            .expect("read");
        assert!(few.is_empty() && !cut);
    }

    #[test]
    fn corrupt_row_is_skipped() {
        // MAGI FIX (MS3 §6 loop11): one undecryptable row failed the whole read.
        let (store, _tmp) = store_for_test();
        let entries = [stored("bad", Some(1), "c1"), stored("good", Some(2), "c1")];
        store.record(&entries).expect("record");
        for e in &entries {
            store
                .update_status(&e.id, HistoryStatus::Attended)
                .expect("attend");
        }
        store
            .locked_conn()
            .execute(
                "UPDATE code_review_history SET finding_blob = x'00' WHERE id = 'bad'",
                [],
            )
            .expect("corrupt");
        let (got, _) = store
            .validated_on_files(&["src/a.rs".to_string()])
            .expect("the rest is read");
        let ids: Vec<String> = got.into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["good".to_string()]);
        assert_eq!(store.corrupt_rows(), 1);
        assert_eq!(store.due(10).expect("due").0.len(), 1);
    }

    #[test]
    fn off_line_never_starves() {
        // MAGI FIX (MS3 §6 loop11): the cap came before the ancestry filter, so newer entries of
        // another branch filled it and an older one on HEAD's line was never read.
        let (store, _tmp) = store_for_test();
        let mut entries: Vec<HistoryEntry> = (0..=MAX_HISTORY_PER_BATCH)
            .map(|i| stored(&format!("a{i:03}"), Some(1), "c2"))
            .collect();
        entries.push(stored("z", Some(1), "c1"));
        store.record(&entries).expect("record");
        for e in &entries {
            store
                .update_status(&e.id, HistoryStatus::Attended)
                .expect("attend");
        }
        let (got, cut) = store
            .validated_on_line(&["src/a.rs".to_string()], &|c: &str| c == "c1")
            .expect("read");
        let ids: Vec<String> = got.into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["z".to_string()]);
        assert!(!cut, "nothing on the line was cut");
    }

    #[test]
    fn the_review_caller_touches_every_checked_entry_so_the_tail_is_never_starved() {
        // [MAGI S6 MS3 loop7, R3 Melchior] `due` is starvation-free only if its caller touches
        // every entry it checked, changed or not. Pinned here, through the production caller
        // (`record_review`), with a git that answers nothing: no status changes, yet the entry
        // past the per-review cap comes first in the next review.
        struct Silent;
        impl magi_rs::review::source::GitCommand for Silent {
            fn run_bytes(
                &self,
                a: &[&str],
            ) -> Result<Vec<u8>, magi_rs::review::error::ReviewError> {
                Err(magi_rs::review::error::ReviewError::Git(a.join(" ")))
            }
        }
        use magi_rs::review::history::MAX_RECLASSIFY_PER_REVIEW as CAP;
        let (store, _tmp) = store_for_test();
        let entries: Vec<HistoryEntry> = (0..=CAP)
            .map(|i| stored(&format!("e{i:04}"), Some(1), "c1"))
            .collect();
        store.record(&entries).expect("record");
        let budget = std::time::Duration::from_secs(60);
        let review = crate::memory::review_rag::record_review;
        review(&store, &[], "HEAD", &Silent, 10, budget).expect("first review");
        let (next, _) = store.due(1).expect("due");
        assert_eq!(next[0].id, format!("e{CAP:04}"), "the tail starved");
    }

    #[test]
    fn the_due_count_stops_at_its_bound_instead_of_reading_the_whole_history() {
        // [MAGI S6 MS3 loop7, R3 Caspar] `due` counted every row on every review, O(history)
        // whatever its limit; the count now reads at most its bound and saturates there.
        let (store, _tmp) = store_for_test();
        let entries: Vec<HistoryEntry> = (0..5)
            .map(|i| stored(&format!("e{i}"), Some(1), "c1"))
            .collect();
        store.record(&entries).expect("record");
        let (rows, total) = store.due_bounded(2, 3).expect("due");
        assert_eq!((rows.len(), total), (2, 3));
        assert_eq!(store.due_bounded(2, 10).expect("due").1, 5);
    }

    #[test]
    fn validated_on_files_accepts_more_files_than_sqlite_parameters() {
        // [CP2 MS3 loop 14 (pasada final)] A mass rename touches more than 999 files: never one `IN (...)`.
        let (store, _tmp) = store_for_test();
        let a = HistoryEntry {
            file: "src/f1199.rs".into(),
            ..stored("a", Some(1), "c1")
        };
        store.record(&[a]).expect("record");
        store
            .update_status("a", HistoryStatus::Attended)
            .expect("a");
        let files: Vec<String> = (0..1_200).map(|i| format!("src/f{i}.rs")).collect();
        let ids: Vec<String> = store
            .validated_on_files(&files)
            .expect("read")
            .0
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(
            ids,
            vec!["a".to_string()],
            "the last file of the list is reached"
        );
    }

    #[test]
    fn recording_the_same_finding_again_keeps_the_first_sighting() {
        let (store, _tmp) = store_for_test();
        assert_eq!(
            store.record(&[stored("i1", Some(7), "c1")]).expect("first"),
            1
        );
        let mut again = stored("i1", Some(8), "c9");
        again.title = "reworded".into();
        assert_eq!(
            store.record(&[again]).expect("second"),
            0,
            "an existing id is not inserted again"
        );
        let all = store.all().expect("all");
        assert_eq!(all.len(), 1);
        assert_eq!(
            (
                all[0].reviewed_commit.as_str(),
                all[0].line,
                all[0].title.as_str()
            ),
            ("c1", Some(7), "title i1"),
            "the first sighting wins, so classify keeps comparing from the first reviewed commit"
        );
    }

    #[test]
    fn a_new_finding_on_the_line_of_an_attended_one_replaces_it() {
        // [MS3 final review m1] The attended finding's code is gone: a later finding on the same
        // file:line is a new one, never dropped while the old text keeps being served.
        let (store, _tmp) = store_for_test();
        store.record(&[stored("i1", Some(7), "c1")]).expect("first");
        store
            .update_status("i1", HistoryStatus::Attended)
            .expect("attended");
        store.touch(&["i1".to_string()], 100).expect("touch");
        let mut later = stored("i1", Some(7), "c9");
        later.title = "a new defect".into();
        later.detail = "found later".into();
        assert_eq!(
            store.record(std::slice::from_ref(&later)).expect("second"),
            1
        );
        assert_eq!(store.all().expect("all"), vec![later]);
        assert!(
            store
                .validated_on_files(&["src/a.rs".to_string()])
                .expect("read")
                .0
                .is_empty(),
            "the old attended text is no longer served"
        );
        let checked: i64 = store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT checked_at FROM code_review_history WHERE id = 'i1'",
                [],
                |r| r.get(0),
            )
            .expect("row");
        assert_eq!(checked, 0, "a new entry is due first");
    }

    #[test]
    fn a_new_finding_on_the_line_of_a_weak_reject_keeps_the_first_sighting() {
        // [MS3 final review m1] A weak reject's code is still there: the same line is the same
        // finding, and its clock keeps running from the first reviewed commit.
        let (store, _tmp) = store_for_test();
        store.record(&[stored("i1", Some(7), "c1")]).expect("first");
        store
            .update_status("i1", HistoryStatus::WeakReject)
            .expect("weak");
        assert_eq!(
            store
                .record(&[stored("i1", Some(7), "c9")])
                .expect("second"),
            0
        );
        let all = store.all().expect("all");
        assert_eq!(
            (all[0].reviewed_commit.as_str(), all[0].status),
            ("c1", HistoryStatus::WeakReject)
        );
    }

    #[test]
    fn validated_on_files_accepts_more_files_than_the_sqlite_variable_limit() {
        // Guardian: the bundled SQLite allows 32 766 variables, so the brief's 1 200 files do not
        // exercise the batching; 40 000 do.
        let (store, _tmp) = store_for_test();
        let a = HistoryEntry {
            file: "src/f39999.rs".into(),
            ..stored("a", Some(1), "c1")
        };
        store.record(&[a]).expect("record");
        store
            .update_status("a", HistoryStatus::Attended)
            .expect("a");
        let files: Vec<String> = (0..40_000).map(|i| format!("src/f{i}.rs")).collect();
        let ids: Vec<String> = store
            .validated_on_files(&files)
            .expect("read")
            .0
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(ids, vec!["a".to_string()]);
    }
}
