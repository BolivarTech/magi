// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
//! Read-only structural diagnosis of the vault DB (`magi vault diagnose`,
//! REQ-H32).
//!
//! [`diagnose`] is a **fourth, independent** read path into the encrypted
//! store, deliberately separate from [`crate::vault::open_envelope`] /
//! `system::database::open_with_state_machine` (the binary crate's unlocking
//! open): it never derives a KEK, never attempts the AEAD unwrap, and never
//! requires a master passphrase. It exists so an operator can inspect *why*
//! a DB will not open (§2.1 of `spec-behavior.md`) without first needing the
//! one thing that may be missing or forgotten.
//!
//! Because it duplicates rather than shares code with the unlocking open (by
//! design — a bug in one must never be masked by reusing the same query as
//! the other), the row-count guard tables are re-declared here as
//! [`GUARD_TABLES`], deliberately mirroring (not importing)
//! `system::database::DATA_TABLES` in the binary crate, which this library
//! crate cannot depend on (dependency direction: bin → lib, never the
//! reverse).
//!
//! **Never a secret VALUE:** [`diagnose`] only ever reads `vault_meta`'s two
//! FEC-encoded blobs (checked structurally, never decrypted) and, opt-in via
//! `include_names`, the `vault` table's `name` column — never a `value_blob`
//! (REQ-V09).

use rusqlite::{Connection, OptionalExtension};

use super::envelope::check_meta_fec;
use super::error::VaultError;

/// Data tables that must all exist and be empty for a DB with no envelope to
/// be a legitimate "fresh, not yet bootstrapped" state (§2.1). Mirrors
/// `system::database::DATA_TABLES` (binary crate) by deliberate duplication —
/// see the module docs.
const GUARD_TABLES: [&str; 4] = ["sessions", "messages", "knowledge", "memories"];

/// Substring `rusqlite` uses to report a missing table, checked via
/// `Display` so this stays robust across `rusqlite`'s error-struct shapes
/// (mirrors `system::database::map_table_err`'s identical approach).
const MISSING_TABLE_MARKER: &str = "no such table";

/// Per-table row counts reported by [`diagnose`] (REQ-H32c). `None` means the
/// table itself does not exist (reported, never a crash) rather than a count
/// of zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TableCounts {
    /// Row count of the `vault` table (secret metadata rows — never values).
    /// Frequently `None` right after `magi init`: the `vault` table is
    /// created lazily by the first unlock, not by `init_schema`.
    pub vault: Option<i64>,
    /// Row count of `sessions`.
    pub sessions: Option<i64>,
    /// Row count of `messages`.
    pub messages: Option<i64>,
    /// Row count of `knowledge`.
    pub knowledge: Option<i64>,
    /// Row count of `memories`.
    pub memories: Option<i64>,
}

/// The §2.1 structural verdict [`diagnose`] can reach **without** a
/// passphrase.
///
/// The full state machine (`system::database::open_or_bootstrap`) has a
/// fourth reachable outcome, `ok` (envelope present, FEC decodes, **and** the
/// AEAD unwrap succeeds under the supplied passphrase) — that outcome is
/// structurally unreachable here on purpose: this probe never attempts the
/// unwrap, so it can never distinguish "the right passphrase" from "a wrong
/// one" when the envelope and its FEC are both intact. That ambiguous case is
/// reported as [`DiagnoseVerdict::WrongPassphrasePossible`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnoseVerdict {
    /// Envelope present, FEC decodes. Whether the DB actually opens depends
    /// on a passphrase this probe never sees — REQ-H32 names this outcome
    /// "wrong-passphrase-possible".
    WrongPassphrasePossible,
    /// `DbCorrupt`-shaped: `vault_meta` itself is missing, a guard table
    /// (`sessions`/`messages`/`knowledge`/`memories`) is missing, its FEC is
    /// uncorrectable, or data is present in a guard table with no envelope to
    /// decrypt it. **Never** triggers a delete or a bootstrap — REQ-H20 / D-H10.
    Corrupt,
    /// No envelope row and every guard table is empty: a legitimate
    /// bootstrap candidate (the state a fresh `magi init` without `-p`
    /// leaves the DB in).
    Fresh,
}

impl DiagnoseVerdict {
    /// The exact lowercase, hyphenated label REQ-H32 specifies for CLI/JSON
    /// output (`"wrong-passphrase-possible"` / `"corrupt"` / `"fresh"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WrongPassphrasePossible => "wrong-passphrase-possible",
            Self::Corrupt => "corrupt",
            Self::Fresh => "fresh",
        }
    }
}

/// The full read-only report produced by [`diagnose`] (REQ-H32).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnoseReport {
    /// Whether `vault_meta` holds a `wrapped_dek` row.
    pub envelope_present: bool,
    /// `Some(true)`/`Some(false)` iff an envelope is present and its FEC did
    /// or did not decode; `None` when there is no envelope to check.
    pub fec_ok: Option<bool>,
    /// Per-table row counts (REQ-H32c).
    pub counts: TableCounts,
    /// The §2.1 verdict computed from the above, without a passphrase.
    pub verdict: DiagnoseVerdict,
    /// The `vault` table's secret **names** — populated only when
    /// `include_names` was `true`; **never** a value (REQ-V09).
    pub names: Option<Vec<String>>,
}

/// The four possible shapes of the `vault_meta` table as seen by a
/// read-only, passphrase-less probe.
enum MetaState {
    /// The `vault_meta` table itself does not exist (schema corruption).
    TableMissing,
    /// The table exists and holds NEITHER a `salt` nor a `wrapped_dek` row —
    /// a legitimate "not yet bootstrapped" candidate.
    NoEnvelope,
    /// The table exists and holds a `salt` row but no `wrapped_dek` row: a
    /// half-written bootstrap (crashed between the two inserts of
    /// `bootstrap_envelope`'s transaction). Always corrupt, regardless of the
    /// guard tables — never a legitimate fresh state.
    PartialSaltOnly,
    /// The table exists and has a `wrapped_dek` row; `salt` may still be
    /// absent (itself a corruption signal, handled by the caller).
    Envelope {
        /// The FEC-encoded `wrapped_dek` blob.
        wrapped_dek_fec: Vec<u8>,
        /// The FEC-encoded `salt` blob, if the row exists.
        salt_fec: Option<Vec<u8>>,
    },
}

/// Returns `true` iff `err`'s `Display` indicates a missing table, mirroring
/// `system::database::map_table_err`'s identical string-match approach
/// (rusqlite does not expose a structured "no such table" variant).
fn is_missing_table(err: &rusqlite::Error) -> bool {
    err.to_string().contains(MISSING_TABLE_MARKER)
}

/// Counts the rows of `table`, reporting a missing table as `Ok(None)`
/// instead of an error (REQ-H32c: "wrapped so a missing table is reported,
/// not a crash"). `table` is always one of the fixed constants this module
/// controls, never caller input — no injection risk.
///
/// # Errors
/// [`VaultError::Storage`] on any SQLite failure other than a missing table.
fn count_or_missing(conn: &Connection, table: &str) -> Result<Option<i64>, VaultError> {
    match conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0)) {
        Ok(n) => Ok(Some(n)),
        Err(e) if is_missing_table(&e) => Ok(None),
        Err(e) => Err(VaultError::Storage(e.to_string())),
    }
}

/// Reads all five table row counts (REQ-H32c): the `vault` table plus every
/// entry of [`GUARD_TABLES`], in that constant's declared order.
///
/// # Errors
/// [`VaultError::Storage`] on any SQLite failure other than a missing table.
fn read_table_counts(conn: &Connection) -> Result<TableCounts, VaultError> {
    let vault = count_or_missing(conn, "vault")?;
    let mut guard_counts = [None; GUARD_TABLES.len()];
    for (slot, table) in guard_counts.iter_mut().zip(GUARD_TABLES) {
        *slot = count_or_missing(conn, table)?;
    }
    let [sessions, messages, knowledge, memories] = guard_counts;
    Ok(TableCounts {
        vault,
        sessions,
        messages,
        knowledge,
        memories,
    })
}

/// Reads the `vault_meta` table's shape without ever attempting an unwrap.
///
/// Always reads BOTH the `wrapped_dek` and `salt` rows (never short-circuits
/// on `wrapped_dek` alone) so a lingering `salt` row with no `wrapped_dek` —
/// a half-written bootstrap — is classified as [`MetaState::PartialSaltOnly`]
/// rather than falling through to [`MetaState::NoEnvelope`].
///
/// # Errors
/// [`VaultError::Storage`] on any SQLite failure other than a missing table.
fn read_meta_state(conn: &Connection) -> Result<MetaState, VaultError> {
    let wrapped_dek_fec: Option<Vec<u8>> = match conn
        .query_row(
            "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
            [],
            |r| r.get(0),
        )
        .optional()
    {
        Ok(v) => v,
        Err(e) if is_missing_table(&e) => return Ok(MetaState::TableMissing),
        Err(e) => return Err(VaultError::Storage(e.to_string())),
    };
    let salt_fec: Option<Vec<u8>> = conn
        .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
            r.get(0)
        })
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;

    match (wrapped_dek_fec, salt_fec) {
        (None, None) => Ok(MetaState::NoEnvelope),
        (None, Some(_)) => Ok(MetaState::PartialSaltOnly),
        (Some(wrapped_dek_fec), salt_fec) => Ok(MetaState::Envelope {
            wrapped_dek_fec,
            salt_fec,
        }),
    }
}

/// Reads every `vault` table `name` (never a `value_blob` — REQ-V09), sorted
/// for deterministic output. A missing `vault` table (e.g. immediately after
/// `magi init`, before the first unlock creates it) yields an empty list,
/// not an error.
///
/// # Errors
/// [`VaultError::Storage`] on any SQLite failure other than a missing table.
fn read_vault_names(conn: &Connection) -> Result<Vec<String>, VaultError> {
    let mut stmt = match conn.prepare("SELECT name FROM vault ORDER BY name ASC") {
        Ok(stmt) => stmt,
        Err(e) if is_missing_table(&e) => return Ok(Vec::new()),
        Err(e) => return Err(VaultError::Storage(e.to_string())),
    };
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let mut names = Vec::new();
    for row in rows {
        names.push(row.map_err(|e| VaultError::Storage(e.to_string()))?);
    }
    Ok(names)
}

/// Applies the §2.1 never-delete guard over [`GUARD_TABLES`]: `None` in any
/// entry means a missing table (corruption, regardless of the rest); all
/// present and zero means "fresh, not yet bootstrapped"; any present count
/// above zero means data with nothing to decrypt it (corruption).
///
/// Checks "is any count non-zero" directly (`any`) rather than summing the
/// four `i64` counts first: a `.sum()` can overflow — and panic under debug
/// overflow-checks — for a sufficiently large DB, when the verdict never
/// actually needs the total, only whether at least one table holds data.
fn guard_verdict(counts: &TableCounts) -> DiagnoseVerdict {
    let entries = [
        counts.sessions,
        counts.messages,
        counts.knowledge,
        counts.memories,
    ];
    if entries.iter().any(Option::is_none) {
        return DiagnoseVerdict::Corrupt;
    }
    if entries.iter().flatten().any(|&n| n > 0) {
        DiagnoseVerdict::Corrupt
    } else {
        DiagnoseVerdict::Fresh
    }
}

/// Produces the read-only, passphrase-less §2.1 report (REQ-H32) for an
/// already-open connection.
///
/// `conn` should be opened read-only by the caller (this module never opens
/// or resolves a path itself — R-V02's "never resolves either on its own"
/// applies here too); nothing here mutates a row, and no envelope is ever
/// unlocked.
///
/// # Errors
/// [`VaultError::Storage`] on a SQLite failure other than a missing table —
/// every OTHER anomaly this function might find (missing `vault_meta`,
/// missing guard table, uncorrectable FEC, data with no envelope) is
/// reported structurally via the returned [`DiagnoseReport::verdict`], never
/// as an `Err` (REQ-H32: "explains the state clearly rather than erroring
/// out").
pub fn diagnose(conn: &Connection, include_names: bool) -> Result<DiagnoseReport, VaultError> {
    let counts = read_table_counts(conn)?;
    let meta = read_meta_state(conn)?;

    let (envelope_present, fec_ok, verdict) = match meta {
        MetaState::TableMissing => (false, None, DiagnoseVerdict::Corrupt),
        MetaState::NoEnvelope => (false, None, guard_verdict(&counts)),
        // A salt row with no wrapped_dek is a half-written bootstrap: always
        // corrupt, regardless of how empty the guard tables are.
        MetaState::PartialSaltOnly => (false, None, DiagnoseVerdict::Corrupt),
        MetaState::Envelope {
            wrapped_dek_fec,
            salt_fec: None,
        } => {
            // A wrapped_dek row with no matching salt row is corrupt metadata,
            // mirroring `open_existing_envelope`'s identical mapping.
            let _ = wrapped_dek_fec;
            (true, Some(false), DiagnoseVerdict::Corrupt)
        }
        MetaState::Envelope {
            wrapped_dek_fec,
            salt_fec: Some(salt_fec),
        } => {
            let ok = check_meta_fec(&salt_fec, &wrapped_dek_fec).is_ok();
            let verdict = if ok {
                DiagnoseVerdict::WrongPassphrasePossible
            } else {
                DiagnoseVerdict::Corrupt
            };
            (true, Some(ok), verdict)
        }
    };

    let names = if include_names {
        Some(read_vault_names(conn)?)
    } else {
        None
    };

    Ok(DiagnoseReport {
        envelope_present,
        fec_ok,
        counts,
        verdict,
        names,
    })
}

/// One label/count pair for [`format_diagnose_report`], factored out so the
/// five-table loop is a single formatting rule instead of five near-identical
/// lines.
const COUNT_LABELS: [&str; 5] = ["vault", "sessions", "messages", "knowledge", "memories"];

/// Renders a count for display: the number, or `"missing"` for `None` — never
/// a bare `-1` or other magic sentinel.
fn format_count(count: Option<i64>) -> String {
    match count {
        Some(n) => n.to_string(),
        None => "missing".to_string(),
    }
}

/// Formats a [`DiagnoseReport`] into human-readable lines for stdout
/// (REQ-H32 / SC-H35): never a secret value, only structure.
#[must_use]
pub fn format_diagnose_report(report: &DiagnoseReport) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "envelope: {}",
        if report.envelope_present {
            "present"
        } else {
            "absent"
        }
    ));
    lines.push(format!(
        "fec: {}",
        match report.fec_ok {
            Some(true) => "ok",
            Some(false) => "corrupt",
            None => "n/a",
        }
    ));
    lines.push(format!("verdict: {}", report.verdict.as_str()));
    lines.push("counts:".to_string());
    let counts = [
        report.counts.vault,
        report.counts.sessions,
        report.counts.messages,
        report.counts.knowledge,
        report.counts.memories,
    ];
    for (label, count) in COUNT_LABELS.iter().zip(counts) {
        lines.push(format!("  {label}: {}", format_count(count)));
    }
    if let Some(names) = &report.names {
        if names.is_empty() {
            lines.push("names: (none)".to_string());
        } else {
            lines.push(format!("names: {}", names.join(", ")));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::{bootstrap_envelope, wire, MaskedDek, SecretStore};
    use std::sync::{Arc, Mutex};
    use zeroize::Zeroizing;

    /// The five-table schema `magi init` creates (no `vault` table — that one
    /// is created lazily by [`wire`]).
    const INIT_SCHEMA_SQL: &str = "
        CREATE TABLE sessions (id TEXT PRIMARY KEY);
        CREATE TABLE messages (id INTEGER PRIMARY KEY);
        CREATE TABLE knowledge (key TEXT PRIMARY KEY);
        CREATE TABLE memories (id TEXT PRIMARY KEY);
        CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);
    ";

    fn fresh_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("mem db");
        conn.execute_batch(INIT_SCHEMA_SQL).expect("schema");
        conn
    }

    #[test]
    fn test_fresh_db_with_no_envelope_and_empty_tables_is_fresh() {
        let conn = fresh_conn();
        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(!report.envelope_present);
        assert_eq!(report.fec_ok, None);
        assert_eq!(report.verdict, DiagnoseVerdict::Fresh);
        assert_eq!(report.counts.sessions, Some(0));
        assert_eq!(report.counts.vault, None, "vault table not yet created");
        assert!(report.names.is_none());
        // No secret bytes anywhere in the rendered report.
        let rendered = format_diagnose_report(&report).join("\n");
        assert!(!rendered.to_lowercase().contains("secret"));
    }

    #[test]
    fn test_bootstrapped_envelope_with_empty_tables_is_wrong_passphrase_possible() {
        // `magi init -p` shape: envelope present, FEC intact, no data yet.
        // This probe never attempts the unwrap, so it reports the ambiguous
        // "could be right, could be wrong" state rather than a false "ok".
        let conn = fresh_conn();
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "bootstrap-master-passphrase").expect("bootstrap");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("insert salt");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("insert wrapped_dek");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(report.envelope_present);
        assert_eq!(report.fec_ok, Some(true));
        assert_eq!(report.verdict, DiagnoseVerdict::WrongPassphrasePossible);
        assert_eq!(report.counts.sessions, Some(0));
    }

    #[test]
    fn test_data_present_without_envelope_is_corrupt_and_never_touches_the_db() {
        let conn = fresh_conn();
        conn.execute("INSERT INTO sessions (id) VALUES ('s1')", [])
            .expect("seed a session row");

        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .expect("count before");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(!report.envelope_present);
        assert_eq!(report.verdict, DiagnoseVerdict::Corrupt);

        let after: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .expect("count after");
        assert_eq!(before, after, "diagnose must never mutate the DB");
        assert_eq!(after, 1);
    }

    #[test]
    fn test_missing_guard_table_is_corrupt() {
        let conn = Connection::open_in_memory().expect("mem db");
        // Deliberately omit `memories` to simulate a partial/foreign schema.
        conn.execute_batch(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY);
             CREATE TABLE messages (id INTEGER PRIMARY KEY);
             CREATE TABLE knowledge (key TEXT PRIMARY KEY);
             CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);",
        )
        .expect("partial schema");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert_eq!(report.verdict, DiagnoseVerdict::Corrupt);
        assert_eq!(report.counts.memories, None);
    }

    #[test]
    fn test_fec_corrupt_vault_meta_is_corrupt() {
        let conn = fresh_conn();
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "bootstrap-master-passphrase").expect("bootstrap");
        for b in wrapped_fec.iter_mut() {
            *b ^= 0xFF; // damage beyond the FEC's correction capacity
        }
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("insert salt");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("insert corrupt wrapped_dek");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(report.envelope_present);
        assert_eq!(report.fec_ok, Some(false));
        assert_eq!(report.verdict, DiagnoseVerdict::Corrupt);
    }

    #[test]
    fn test_wrapped_dek_without_salt_row_is_corrupt() {
        let conn = fresh_conn();
        let vault = cryptovault::CryptoVault::default();
        let (_salt_fec, wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "bootstrap-master-passphrase").expect("bootstrap");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("insert wrapped_dek without salt");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(report.envelope_present);
        assert_eq!(report.fec_ok, Some(false));
        assert_eq!(report.verdict, DiagnoseVerdict::Corrupt);
    }

    #[test]
    fn test_salt_without_wrapped_dek_is_corrupt_not_fresh() {
        // MAGI re-gate INFO: a `vault_meta` row holding `salt` but NOT
        // `wrapped_dek` is a half-written/interrupted bootstrap (crashed
        // between the two inserts of `bootstrap_envelope`'s transaction) —
        // never a legitimate "fresh, not yet bootstrapped" state, regardless
        // of how empty the guard tables are. `read_meta_state` used to only
        // ever look at the `wrapped_dek` row, so a lingering `salt`-only row
        // fell all the way through to `NoEnvelope` and, with empty guard
        // tables, reported `Fresh` — masking the corruption.
        let conn = fresh_conn();
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, _wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "bootstrap-master-passphrase").expect("bootstrap");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("insert salt without wrapped_dek");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(
            !report.envelope_present,
            "no wrapped_dek row means no envelope, by definition"
        );
        assert_eq!(
            report.verdict,
            DiagnoseVerdict::Corrupt,
            "a salt-only row must never be reported as Fresh"
        );
    }

    #[test]
    fn test_envelope_present_but_data_table_missing_is_corrupt() {
        // MAGI re-gate WARNING: spec-behavior.md §2.1 says "envelope present
        // + a data table ABSENT ⇒ DbCorrupt" — a missing table means the DB
        // cannot open at all, regardless of which passphrase is supplied, so
        // it must never be reported as the ambiguous
        // `WrongPassphrasePossible` (which implies the right passphrase
        // would open it fine). Before the fix, `diagnose` only inspected the
        // FEC-decode result for the `Envelope` branch and never consulted
        // `counts`, so an intact envelope over a schema missing `memories`
        // was misreported as `WrongPassphrasePossible`.
        let conn = Connection::open_in_memory().expect("mem db");
        // Deliberately omit `memories` — everything else present, including
        // an intact envelope.
        conn.execute_batch(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY);
             CREATE TABLE messages (id INTEGER PRIMARY KEY);
             CREATE TABLE knowledge (key TEXT PRIMARY KEY);
             CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);",
        )
        .expect("partial schema");
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "bootstrap-master-passphrase").expect("bootstrap");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("insert salt");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("insert wrapped_dek");

        let report = diagnose(&conn, false).expect("diagnose ok");
        assert!(report.envelope_present);
        assert_eq!(report.fec_ok, Some(true));
        assert_eq!(
            report.verdict,
            DiagnoseVerdict::Corrupt,
            "a missing data table alongside an intact envelope must be Corrupt, \
             never WrongPassphrasePossible"
        );
        assert_eq!(report.counts.memories, None);
    }

    #[test]
    fn test_names_opt_in_lists_names_never_values() {
        let conn = fresh_conn();
        let dek = MaskedDek::new(Zeroizing::new(vec![7u8; 32])).expect("32B dek");
        let mut store = wire(Arc::new(Mutex::new(conn)), dek).expect("wire");
        store
            .set("OPENAI_API_KEY", "sk-super-secret-PROBE-value")
            .expect("seed a secret");
        let conn = store.debug_conn();
        let guard = conn.lock().expect("lock");

        // Without `--names`: only counts, no name/value anywhere.
        let report_no_names = diagnose(&guard, false).expect("diagnose ok");
        assert!(report_no_names.names.is_none());
        assert_eq!(report_no_names.counts.vault, Some(1));

        // With `--names`: the name appears, the value never does.
        let report_names = diagnose(&guard, true).expect("diagnose ok");
        let names = report_names.names.clone().expect("names populated");
        assert_eq!(names, vec!["OPENAI_API_KEY".to_string()]);
        let rendered = format_diagnose_report(&report_names).join("\n");
        assert!(rendered.contains("OPENAI_API_KEY"));
        assert!(!rendered.contains("sk-super-secret-PROBE-value"));
    }

    #[test]
    fn test_guard_verdict_does_not_overflow_or_panic_on_huge_counts() {
        // MAGI re-gate finding: summing four `i64` row counts with `.sum()`
        // can overflow (and panic under debug overflow-checks) for a
        // sufficiently large DB. `guard_verdict` only needs to know whether
        // ANY guard table holds data, never the total, so this must neither
        // panic nor misreport with counts near `i64::MAX`.
        let counts = TableCounts {
            vault: None,
            sessions: Some(i64::MAX),
            messages: Some(i64::MAX),
            knowledge: Some(0),
            memories: Some(0),
        };
        assert_eq!(guard_verdict(&counts), DiagnoseVerdict::Corrupt);
    }

    #[test]
    fn test_guard_verdict_is_fresh_when_huge_counts_are_all_zero() {
        // Non-overflow companion case: large but all-zero counts must still
        // report Fresh, proving the no-overflow fix didn't flip the verdict.
        let counts = TableCounts {
            vault: None,
            sessions: Some(0),
            messages: Some(0),
            knowledge: Some(0),
            memories: Some(0),
        };
        assert_eq!(guard_verdict(&counts), DiagnoseVerdict::Fresh);
    }

    #[test]
    fn test_diagnose_never_requires_or_accepts_a_passphrase() {
        // Structural: `diagnose`'s signature has no passphrase parameter at
        // all, so a caller without one can still call it. This test locks
        // that in by calling it on a DB whose envelope, if opened for real,
        // would need one — `diagnose` succeeds regardless.
        let conn = fresh_conn();
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) =
            bootstrap_envelope(&vault, "some-real-passphrase").expect("bootstrap");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("insert salt");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("insert wrapped_dek");

        assert!(diagnose(&conn, false).is_ok());
    }
}
