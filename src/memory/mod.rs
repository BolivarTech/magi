// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Memory subsystem — tiered, consultable memory with forgetting.
//!
//! Organised as a set of cooperating sub-modules. This first task adds only the
//! configuration surface (`config`); subsequent tasks will add storage, retrieval,
//! decay/eviction, context assembly, and benchmarking.

pub mod clock;
pub mod config;
pub mod decay;
pub mod embedding;
pub mod error;
pub mod index;
pub mod retrieval;
pub mod salience;
pub mod store;
pub mod tokens;

// Narrow allow: re-exports consumed by later tasks; tests use them via `use super::*`.
#[allow(unused_imports)]
pub use error::{EmbeddingError, MemoryError};
// Narrow allow: `recall` and `RankedMemory` are the stable B1/D-13 seam consumed by the
// context assembler (Task 11) and future Agent Society consumers (AS-REQ-11).
#[allow(unused_imports)]
pub use retrieval::{recall, RankedMemory};

/// Kind of a stored memory: an episodic turn, or a durable preference.
///
/// The on-disk representation is a stable lowercase string (see [`as_str`]).
/// This follows the same discipline as the message `Role` serialization so
/// that a schema change is an explicit, detectable migration — not a silent
/// refactor.
///
/// [`as_str`]: MemoryKind::as_str
// Narrow allow: enum consumed by the vector store in Task 4.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// A single conversational turn or tool-result, ordered by session time.
    Episodic,
    /// A durable user preference that is always injected into the context
    /// profile and is never evicted by decay.
    Preference,
}

impl MemoryKind {
    /// Returns the stable on-disk string for this kind.
    ///
    /// Values: `"episodic"` | `"preference"`. Parsed back by exact match;
    /// changing these strings requires an explicit DB migration.
    // Narrow allow: called by the store serializer in Task 4.
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic",
            MemoryKind::Preference => "preference",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_kind_serializes_to_stable_strings() {
        assert_eq!(MemoryKind::Episodic.as_str(), "episodic");
        assert_eq!(MemoryKind::Preference.as_str(), "preference");
    }

    #[test]
    fn test_embedding_error_converts_into_memory_error_and_formats() {
        // #[from] lets `?` lift an EmbeddingError into a MemoryError.
        fn lift() -> Result<(), MemoryError> {
            Err(EmbeddingError::Auth)?;
            Ok(())
        }
        let e = lift().unwrap_err();
        assert!(matches!(e, MemoryError::Embedding(EmbeddingError::Auth)));
        assert_eq!(
            e.to_string(),
            "embedding error: embedding auth failed (401/403)"
        );
    }

    #[test]
    fn test_rate_limited_variant_formats() {
        assert_eq!(
            EmbeddingError::RateLimited.to_string(),
            "embedding rate-limited (429)"
        );
    }
}
