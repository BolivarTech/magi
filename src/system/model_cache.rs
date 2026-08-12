// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-11

//! Persistent cache of measured model capabilities (REQ-R25).
//!
//! # Why a cache is viable here, when caches usually are not
//!
//! The usual objection is invalidation. It does not bite here because of one observation: **a
//! configured model does not change its properties; what happens in practice is that a NEW one
//! appears and replaces a deprecated one.** So the model's identity is its tag, a new model is a
//! new key — a miss, therefore measured — and invalidation resolves itself. There is no TTL, no
//! freshness heuristic and no date that ages.
//!
//! # It cannot cache a failure, by construction
//!
//! [`CachedCapability`] holds a window and an optional digest and **nothing else**, so there is no
//! value in this module's vocabulary that means "not measured". That is deliberate and it is the
//! most dangerous point of the design turned into a type: a cold Ollama daemon does not answer the
//! probe within its ceiling, and the first run of a fresh install is exactly when that happens. If
//! that run wrote "not measured" rows, the cache would be **permanently poisoned** — a transient
//! condition frozen into a final one, the precise opposite of what the cache exists to achieve.
//! A runtime check would have been a rule someone could forget; an API that cannot express the
//! mistake is not.
//!
//! # The key is the PAIR, and the endpoint is the REDACTED one
//!
//! A tag is unique only *within* an endpoint: the same `qwen3.5:397b-cloud` against a local daemon
//! and against one on the LAN need not be the same model. And the endpoint stored is the redacted
//! form — never the one resolved with credentials. The database is encrypted, but that protects
//! the file; it does not make it right to write a secret into it.
//!
//! # It lives in the BIN, not in `magi_rs::magi`
//!
//! Not a placement preference: `system::database::init_schema` — the single source of truth for
//! every table in this database — is **bin-only**, so a lib module physically cannot reach it and
//! would have to carry a second `CREATE TABLE` free to drift from the first. `memory::store`, the
//! closest analogue, sits here for the same reason.
//!
//! # It shares the connection and the key, and owns neither
//!
//! Same shape as `SqliteVectorStore`: the `Arc<Mutex<Connection>>` and a masked duplicate of the
//! DEK come from the already-open encrypted database, and the schema comes from
//! `database::init_schema` rather than a second `CREATE TABLE` that could drift from it.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use cryptovault::CryptoVault;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use magi_rs::vault::MaskedDek;

/// A capability that was **successfully measured**.
///
/// There is deliberately no variant for "could not measure": see the module docs. Constructing one
/// is the same as asserting the measurement succeeded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedCapability {
    /// Context window in tokens, already validated against the probe's admissible range.
    pub window: usize,
    /// SHA-256 of the manifest — 64 lowercase hex — or `None` when it did not resolve.
    ///
    /// **Its meaning is "the one it had when we measured it", not "the current one"**, and it is
    /// never re-verified: the digest is read once, on the first trip, and a consumer that reads
    /// this field as current is wrong. `None` is inert — a digest that never resolved cannot
    /// collide with anything.
    pub digest: Option<String>,
}

/// Why a cache operation failed.
///
/// Every variant is **degradable**: the whole measurement subsystem fails open, so a caller that
/// cannot read or write the cache measures instead of aborting.
///
/// # Both variants carry text composed by ANOTHER crate, so both are redacted at construction
///
/// `rusqlite` and `cryptovault` compose these messages, and this error reaches a startup notice.
/// No path was found by which a credential could land in one — the endpoint is redacted long
/// before it reaches this module — so this is rule conformance rather than a demonstrated leak,
/// and it is stated that way instead of dressed up. The rule earns its keep anyway: it is exactly
/// the shape that produced five findings in the previous milestone, every one green against every
/// gate, and each of those also looked harmless until the string that carried the credential was
/// traced to where it ended up.
#[derive(Debug, Error)]
pub enum CacheError {
    /// The database rejected the statement.
    #[error("model capability cache: storage failed: {0}")]
    Storage(String),
    /// The blob could not be sealed or opened with the DEK.
    #[error("model capability cache: crypto failed: {0}")]
    Crypto(String),
}

/// Cache of measured model capabilities, over the already-open encrypted database.
pub struct ModelCapabilityCache {
    /// Shared with the encrypted memory store; this type never opens its own database.
    conn: Arc<Mutex<Connection>>,
    /// Its own `CryptoVault`, like `SqliteVectorStore`: it is stateless configuration, so a
    /// second one is cheaper than threading a reference and costs no key material.
    vault: CryptoVault,
    /// A masked DUPLICATE of the DEK. The plaintext key exists only inside `with_dek`.
    dek: Mutex<MaskedDek>,
}

impl ModelCapabilityCache {
    /// Builds the cache over an open connection and a masked duplicate of the DEK.
    ///
    /// The schema is created through `database::init_schema`, which is the single source of truth
    /// for every table in this database — a local `CREATE TABLE` here would be a second definition
    /// free to drift from it.
    ///
    /// # Errors
    /// [`CacheError::Storage`] if the schema could not be created.
    pub fn new(conn: Arc<Mutex<Connection>>, dek: MaskedDek) -> Result<Self, CacheError> {
        {
            let c = conn.lock().unwrap_or_else(|p| p.into_inner());
            crate::system::database::init_schema(&c).map_err(|e| {
                CacheError::Storage(
                    magi_rs::redact::redact_foreign_error(&e)
                        .as_str()
                        .to_owned(),
                )
            })?;
        }
        Ok(Self {
            conn,
            vault: CryptoVault::default(),
            dek: Mutex::new(dek),
        })
    }

    /// Reads a cached capability for `(endpoint_redacted, model)`.
    ///
    /// A miss is `Ok(None)` — the ordinary case for a model seen for the first time, not an error.
    ///
    /// # Errors
    /// [`CacheError::Storage`] on a database failure, [`CacheError::Crypto`] if the row exists but
    /// cannot be opened with this DEK.
    pub fn get(
        &self,
        endpoint_redacted: &str,
        model: &str,
    ) -> Result<Option<CachedCapability>, CacheError> {
        // Read the raw blob UNDER the lock and decrypt OFF it (R-V08): the crypto is orders of
        // magnitude slower than the query, and holding a database lock across it serialises every
        // other reader for no reason.
        let blob: Option<String> = {
            let c = self.conn.lock().unwrap_or_else(|p| p.into_inner());
            c.query_row(
                "SELECT capability_blob FROM model_capabilities WHERE endpoint = ?1 AND model = ?2",
                rusqlite::params![endpoint_redacted, model],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(CacheError::Storage(
                    magi_rs::redact::redact_foreign_error(&other)
                        .as_str()
                        .to_owned(),
                )),
            })?
        };

        let Some(blob) = blob else { return Ok(None) };
        let plain = self.unseal(&blob)?;
        serde_json::from_str(&plain).map(Some).map_err(|e| {
            CacheError::Crypto(
                magi_rs::redact::redact_foreign_error(&e)
                    .as_str()
                    .to_owned(),
            )
        })
    }

    /// Persists a measured capability for `(endpoint_redacted, model)`.
    ///
    /// Taking a [`CachedCapability`] rather than a measurement result is what makes "only
    /// successful measurements are persisted" unbreakable rather than remembered.
    ///
    /// # Errors
    /// [`CacheError::Crypto`] if the value cannot be sealed, [`CacheError::Storage`] on a database
    /// failure.
    pub fn put(
        &self,
        endpoint_redacted: &str,
        model: &str,
        capability: &CachedCapability,
    ) -> Result<(), CacheError> {
        let plain = serde_json::to_string(capability).map_err(|e| {
            CacheError::Crypto(
                magi_rs::redact::redact_foreign_error(&e)
                    .as_str()
                    .to_owned(),
            )
        })?;
        // Sealed BEFORE taking the connection lock, same reason as in `get`.
        let blob = self.seal(&plain)?;
        let c = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        c.execute(
            "INSERT INTO model_capabilities (endpoint, model, capability_blob) VALUES (?1, ?2, ?3)
             ON CONFLICT(endpoint, model) DO UPDATE SET capability_blob = excluded.capability_blob",
            rusqlite::params![endpoint_redacted, model, blob],
        )
        .map(|_| ())
        .map_err(|e| {
            CacheError::Storage(
                magi_rs::redact::redact_foreign_error(&e)
                    .as_str()
                    .to_owned(),
            )
        })
    }

    /// Drops the rows whose `(endpoint, model)` pair is no longer configured, and returns how many
    /// were removed.
    ///
    /// **Order is not identity.** `configured` is treated as a SET: reordering the pool changes
    /// which candidate is preferred, never which models exist, so a reorder must prune nothing.
    /// An implementation that folded the position into the key would re-measure everything on
    /// every start, and nothing would fail visibly — the only symptom is the cache never paying
    /// off, which is what it exists for.
    ///
    /// # Errors
    /// [`CacheError::Storage`] on a database failure.
    pub fn prune_absent(&self, configured: &[(String, String)]) -> Result<usize, CacheError> {
        let keep: BTreeSet<(&str, &str)> = configured
            .iter()
            .map(|(e, m)| (e.as_str(), m.as_str()))
            .collect();

        let c = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let existing: Vec<(String, String)> = {
            let mut stmt = c
                .prepare("SELECT endpoint, model FROM model_capabilities")
                .map_err(|e| {
                    CacheError::Storage(
                        magi_rs::redact::redact_foreign_error(&e)
                            .as_str()
                            .to_owned(),
                    )
                })?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    CacheError::Storage(
                        magi_rs::redact::redact_foreign_error(&e)
                            .as_str()
                            .to_owned(),
                    )
                })?;
            rows.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| {
                CacheError::Storage(
                    magi_rs::redact::redact_foreign_error(&e)
                        .as_str()
                        .to_owned(),
                )
            })?
        };

        let mut removed = 0usize;
        for (endpoint, model) in existing {
            if !keep.contains(&(endpoint.as_str(), model.as_str())) {
                c.execute(
                    "DELETE FROM model_capabilities WHERE endpoint = ?1 AND model = ?2",
                    rusqlite::params![endpoint, model],
                )
                .map_err(|e| {
                    CacheError::Storage(
                        magi_rs::redact::redact_foreign_error(&e)
                            .as_str()
                            .to_owned(),
                    )
                })?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// How many rows the table holds. Exists so a test can assert that a failed measurement wrote
    /// **nothing**, which is a property about absence and cannot be observed through `get`.
    ///
    /// `cfg(test)` because that is the whole truth about it: production never counts rows, and an
    /// accessor kept alive by an `#[allow]` would claim a caller it does not have.
    ///
    /// # Errors
    /// [`CacheError::Storage`] on a database failure.
    #[cfg(test)]
    pub fn row_count(&self) -> Result<usize, CacheError> {
        let c = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        c.query_row("SELECT COUNT(*) FROM model_capabilities", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| usize::try_from(n).unwrap_or(0))
        .map_err(|e| {
            CacheError::Storage(
                magi_rs::redact::redact_foreign_error(&e)
                    .as_str()
                    .to_owned(),
            )
        })
    }

    /// Encrypts with the masked DEK. **Never called while holding the connection lock** (R-V08).
    fn seal(&self, plaintext: &str) -> Result<String, CacheError> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.encrypt_with_key(k, plaintext))
            .map_err(|e| {
                CacheError::Crypto(
                    magi_rs::redact::redact_foreign_error(&e)
                        .as_str()
                        .to_owned(),
                )
            })
    }

    /// Decrypts with the masked DEK. Same lock contract as [`Self::seal`].
    fn unseal(&self, blob: &str) -> Result<zeroize::Zeroizing<String>, CacheError> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.decrypt_with_key(k, blob))
            .map_err(|e| {
                CacheError::Crypto(
                    magi_rs::redact::redact_foreign_error(&e)
                        .as_str()
                        .to_owned(),
                )
            })
    }
}

/// Unit tests for the model capability cache.
#[cfg(test)]
mod tests {
    use super::*;

    /// An in-memory database and a FIXED key.
    ///
    /// No passphrase and therefore **no Argon2**: `MaskedDek::new` takes the 32 raw bytes, so these
    /// tests cost nothing in key derivation and do not belong in nextest's `heavy` group. That
    /// matters beyond speed — widening that group's filter to reach a leaf module is how the cap
    /// silently stopped applying once before in this repo.
    fn cache() -> ModelCapabilityCache {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        let dek = MaskedDek::new(zeroize::Zeroizing::new(vec![7u8; 32])).expect("32 bytes");
        ModelCapabilityCache::new(Arc::new(Mutex::new(conn)), dek).expect("schema")
    }

    /// A successful measurement.
    fn measured(window: usize, digest: Option<&str>) -> CachedCapability {
        CachedCapability {
            window,
            digest: digest.map(str::to_owned),
        }
    }

    const ENDPOINT: &str = "http://localhost:11434/v1";

    /// SC-R25: what was measured is REMEMBERED. A second read of an unchanged entry is a HIT, so
    /// the caller has no reason to probe.
    #[test]
    fn a_measured_model_is_remembered_across_reads() {
        let cache = cache();
        cache
            .put(ENDPOINT, "a", &measured(128_000, Some(&"f".repeat(64))))
            .expect("put");
        assert_eq!(
            cache.get(ENDPOINT, "a").expect("get"),
            Some(measured(128_000, Some(&"f".repeat(64)))),
            "a warm entry must come back intact, digest included"
        );
    }

    /// SC-R26 — THE MOST DANGEROUS POINT OF THE DESIGN. A cold start must not poison the cache.
    ///
    /// Here it is asserted as the type property it is: there is no value of [`CachedCapability`]
    /// that means "not measured", so a run where every probe failed has nothing to write and the
    /// table stays empty. The next run retries. Persisting a failure would freeze a transient
    /// condition into a permanent one.
    #[test]
    fn a_cold_start_leaves_the_table_empty_rather_than_caching_the_failure() {
        let cache = cache();
        // A cold run measures nothing, so it calls `put` for nothing.
        assert_eq!(
            cache.row_count().expect("count"),
            0,
            "no 'not measured' row may exist — the API cannot even express one"
        );
        assert!(cache.get(ENDPOINT, "a").expect("get").is_none());
    }

    /// SC-R27: the key is the PAIR. The same tag against another endpoint need not be the same
    /// model, so it is a miss and gets measured on its own.
    #[test]
    fn the_same_tag_on_another_endpoint_is_a_miss() {
        let cache = cache();
        cache
            .put(
                "http://a:11434/v1",
                "qwen3.5:397b-cloud",
                &measured(128_000, None),
            )
            .expect("put");
        // The HIT is asserted first, and that is not ceremony: without it this test passes
        // against a `put` that stores nothing at all, since "absent from the other endpoint" is
        // then trivially true. An absence assertion means nothing until the presence it
        // contrasts with is established in the SAME test.
        assert!(
            cache
                .get("http://a:11434/v1", "qwen3.5:397b-cloud")
                .expect("get")
                .is_some(),
            "the entry must exist where it was written"
        );
        assert!(
            cache
                .get("http://b:11434/v1", "qwen3.5:397b-cloud")
                .expect("get")
                .is_none(),
            "the same tag on another endpoint need not be the same model: it is a MISS"
        );
    }

    /// SC-R41 — SILENT-FAILURE GUARDIAN. Reordering the pool must prune nothing: membership is the
    /// question, position is not. An implementation that folded the index into the key would
    /// re-measure everything on every start with nothing failing visibly.
    #[test]
    fn reordering_the_configured_models_prunes_nothing() {
        let cache = cache();
        for model in ["a", "b", "c"] {
            cache
                .put(ENDPOINT, model, &measured(1_000, None))
                .expect("put");
        }
        let reordered: Vec<(String, String)> = ["c", "a", "b"]
            .iter()
            .map(|m| (ENDPOINT.to_owned(), (*m).to_owned()))
            .collect();

        assert_eq!(
            cache.prune_absent(&reordered).expect("prune"),
            0,
            "reordering is a set-membership question, not an identity change"
        );
        assert_eq!(cache.row_count().expect("count"), 3);
    }

    /// REQ-R25: a model that LEFT the configuration has its row pruned, and only that one.
    #[test]
    fn a_model_removed_from_the_configuration_is_pruned_and_the_others_survive() {
        let cache = cache();
        for model in ["a", "b"] {
            cache
                .put(ENDPOINT, model, &measured(1_000, None))
                .expect("put");
        }
        let remaining = vec![(ENDPOINT.to_owned(), "a".to_owned())];

        assert_eq!(cache.prune_absent(&remaining).expect("prune"), 1);
        assert!(cache.get(ENDPOINT, "a").expect("get").is_some());
        assert!(cache.get(ENDPOINT, "b").expect("get").is_none());
    }

    /// SC-R30, the distinction the plan flags as the easiest to get wrong: an **absent or empty**
    /// table is NOT "no cache available".
    ///
    /// It is created by `init_schema` and filled by the first run. Confusing the two would degrade
    /// to a stateless measurement and persist **nothing** on exactly the clean start where the
    /// cache pays off most — and the symptom would be a cache that never warms, which looks like
    /// slowness rather than a defect.
    ///
    /// "No cache available" is exactly two other conditions: no encrypted database open in this
    /// run, and a table that exists but cannot be read or written. Both produce `None` at
    /// construction, never an empty table.
    #[test]
    fn an_empty_table_persists_normally_instead_of_degrading() {
        let cache = cache();
        assert_eq!(
            cache.row_count().expect("count"),
            0,
            "precondition: the table exists and is empty"
        );

        cache
            .put(ENDPOINT, "a", &measured(128_000, None))
            .expect("an empty table must accept a write, not refuse one");
        assert_eq!(
            cache.row_count().expect("count"),
            1,
            "an empty table PERSISTS; it does not degrade to stateless"
        );
        assert!(cache.get(ENDPOINT, "a").expect("get").is_some());
    }

    /// SC-R58: the credential never reaches the database — **not even an encrypted one** — and the
    /// redacted form still hits on the next read.
    ///
    /// Both halves are load-bearing and for different reasons. The first is the security contract:
    /// encryption protects the file, it does not make writing a secret into it acceptable. The
    /// second is what makes the guardian useful — an implementation that redacted on READ instead
    /// of on WRITE would pass the first half and then miss every lookup **in silence**, the only
    /// symptom being a cache that re-measures on every start.
    #[test]
    fn the_cache_key_holds_the_redacted_endpoint_and_still_hits() {
        const CANARY: &str = "c4n4ry-s3cr3t";
        let cache = cache();
        let resolved = format!("http://alice:{CANARY}@localhost:11434/v1");
        let redacted = magi_rs::redact::redact_url(&resolved);
        assert!(
            !redacted.contains(CANARY),
            "precondition: the redactor must actually redact"
        );

        cache
            .put(&redacted, "a", &measured(128_000, None))
            .expect("put");

        // The credential is nowhere in the database — key column included.
        let dump: String = {
            let c = cache.conn.lock().expect("lock");
            let mut stmt = c
                .prepare("SELECT endpoint, model, capability_blob FROM model_capabilities")
                .expect("prepare");
            stmt.query_map([], |row| {
                Ok(format!(
                    "{}|{}|{}",
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?
                ))
            })
            .expect("query")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("rows")
            .join("\n")
        };
        assert!(
            !dump.contains(CANARY),
            "the credential reached the DB: {dump}"
        );
        assert!(
            !dump.contains("alice"),
            "the username reached the DB: {dump}"
        );

        // And the same redaction of the same resolved URL still finds it.
        assert!(
            cache
                .get(&magi_rs::redact::redact_url(&resolved), "a")
                .expect("get")
                .is_some(),
            "redacting on write must not break the key's identity"
        );
    }
}
